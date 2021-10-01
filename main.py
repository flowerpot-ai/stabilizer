import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

from stabilizer.model import PoolerClassifier
from stabilizer.dataset import TextLabelDataset
from stabilizer.reproducibility import seed_torch
from stabilizer.trainer import train_step, evaluate_step
from stabilizer.reinitialize import reinit_autoencoder_model
from stabilizer.llrd import get_optimizer_parameters_with_llrd

from transformers import get_scheduler, AdamW, AutoModel, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

"""
config = {'train_data_path': 'data/glue/cola/train.jsonl',
          'valid_data_path': 'data/glue/cola/valid.jsonl',
          'batch_size': 32,
          'pretrained_tokenizer_name_or_path': 'models/bert-base-uncased',
          'pretrained_model_name_or_path': 'models/bert-base-uncased',
          'device_name': 'cpu',
          'dropout_prob': 0.1,
          'num_classes': 1,
          'lr': 2e-5,
          'num_epochs': 3,
          'validate_every_n_iteration': 10,
          'dataloader_seed': 41,
          'layer_initialization_seed': 1000,
          'dropout_seed': 1234,
          'reinit_encoder': True,
          'reinit_num_layers': 4,
          'apply_llrd': True,
          'multiplicative_factor': 0.95}

python main.py --train_data_path data/glue/cola/train.jsonl --valid_data_path data/glue/cola/valid.jsonl --batch_size 32 \
    --pretrained_tokenizer_name_or_path models/bert-base-uncased --pretrained_model_name_or_path models/bert-base-uncased \
    --device_name cuda --dropout_prob 0.1 --num_classes 1 --lr 2e-5 --num_epochs 3 --validate_every_n_iteration 10 \
    --dataloader_seed 41 --layer_initialization_seed 1000 --dropout_seed 1234 --reinit_encoder True --reinit_num_layers 4 \
    --apply_llrd True --multiplicative_factor 0.95
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Pooler classifier on CoLA dataset")
    parser.add_argument("--train_data_path", type=str, help="Path of the training data file")
    parser.add_argument("--valid_data_path", type=str, help="Path of the validation data file")
    parser.add_argument(
        "--pretrained_tokenizer_name_or_path",
        type=str,
        help="Path of the directory that contains the tokenizer",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="Path of the directory that contains the pretrained model",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--device_name",
        type=str,
        choices=("cpu", "cuda"),
        help="Device to train the algorithm",
    )
    parser.add_argument("--dropout_prob", type=float, help="Value of dropout")
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Num of output classes for the classification task",
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--validate_every_n_iteration", type=int, help="How often to validate")
    parser.add_argument("--dropout_seed", type=int, default=random.randint(a=0, b=10000))
    parser.add_argument("--layer_initialization_seed", type=int, default=random.randint(a=0, b=10000))
    parser.add_argument("--dataloader_seed", type=int, default=random.randint(a=0, b=10000))
    parser.add_argument(
        "--reinit_encoder",
        type=bool,
        help="Should be the transformer encoder be reinitialized",
    )
    parser.add_argument(
        "--reinit_num_layers",
        type=int,
        help="Number of transformer encoder layers to be reinitialized",
    )
    parser.add_argument(
        "--apply_llrd",
        type=bool,
        help="Should apply Layerwise learning rate decay to the model parameters during optimization",
    )
    parser.add_argument(
        "--multiplicative_factor",
        type=float,
        help="Factor with which the learning rate should decrease for successive layers",
    )
    args = parser.parse_args()
    return args


def post_process_targets(targets):
    targets = targets.type(torch.int)
    targets = targets.cpu().detach().numpy().reshape(-1)
    return targets


def post_process_predictions(predictions):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions >= 0.5).type(torch.int)
    predictions = predictions.cpu().detach().numpy().reshape(-1)
    return predictions


def compute_matthews_corrcoef(targets, predictions):
    if len(np.unique(predictions)) > 1 and len(np.unique(targets)) > 1:
        score = matthews_corrcoef(y_true=targets, y_pred=predictions)
    else:
        score = 0.0
    return score


def run():
    # Read configuration
    config = parse_args().__dict__

    # Read training data
    train_data = pd.read_json(path_or_buf=config["train_data_path"], lines=True).set_index("idx")
    valid_data = pd.read_json(path_or_buf=config["valid_data_path"], lines=True).set_index("idx")

    # Prepate data to create dataset
    train_text_excerpts = train_data["text"].tolist()
    valid_text_excerpts = valid_data["text"].tolist()
    train_labels = torch.from_numpy(train_data["label"].to_numpy().reshape(-1, 1)).type(torch.float32)
    valid_labels = torch.from_numpy(valid_data["label"].to_numpy().reshape(-1, 1)).type(torch.float32)

    # Create Dataset
    train_dataset = TextLabelDataset(text_excerpts=train_text_excerpts, labels=train_labels)
    valid_dataset = TextLabelDataset(text_excerpts=valid_text_excerpts, labels=valid_labels)

    # Create DataLoader
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(config["dataloader_seed"])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=generator,
    )

    # Create tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_tokenizer_name_or_path"])
    transformer = AutoModel.from_pretrained(
        pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
        hidden_dropout_prob=config["dropout_prob"],
        attention_probs_dropout_prob=config["dropout_prob"],
    )

    # Reinitialize
    if config["reinit_encoder"]:
        seed_torch(config["layer_initialization_seed"])
        transformer.encoder = reinit_autoencoder_model(
            transformer.encoder, reinit_num_layers=config["reinit_num_layers"]
        )

    model = PoolerClassifier(
        transformer=transformer,
        transformer_output_size=transformer.config.hidden_size,
        transformer_output_dropout_prob=config["dropout_prob"],
        num_classes=config["num_classes"],
        task_specific_layer_seed=config["layer_initialization_seed"],
    )

    device = torch.device(config["device_name"])
    _ = model.to(device)

    # Define loss
    loss_fn = nn.BCEWithLogitsLoss()

    # Create optimizer
    if config["apply_llrd"]:
        model_parameters = get_optimizer_parameters_with_llrd(
            model=model,
            peak_lr=config["lr"],
            multiplicative_factor=config["multiplicative_factor"],
        )
    else:
        model_parameters = model.parameters()
    optimizer = AdamW(params=model_parameters, lr=config["lr"])

    # Create scheduler
    num_training_steps = config["num_epochs"] * len(train_dataloader)
    num_warmup_steps = num_training_steps // 10
    logger.info(f"Number of training steps: {num_training_steps}")
    logger.info(f"Number of warmup steps: {num_warmup_steps}")

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Add dropout seed
    seed_torch(config["dropout_seed"])

    # Start training
    iteration_num = 0
    for epoch in range(config["num_epochs"]):
        for batch in train_dataloader:
            batch_inputs = tokenizer(
                text=batch["text_excerpt"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            batch_targets = batch["label"].to(device)
            train_outputs = train_step(
                model=model,
                inputs=batch_inputs,
                targets=batch_targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if iteration_num % config["validate_every_n_iteration"] == 0:
                valid_targets, valid_predictions = [], []
                for batch in valid_dataloader:
                    batch_inputs = tokenizer(
                        text=batch["text_excerpt"],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    batch_targets = batch["label"].to(device)
                    valid_outputs = evaluate_step(
                        model=model,
                        inputs=batch_inputs,
                        targets=batch_targets,
                        loss_fn=loss_fn,
                    )
                    valid_targets.extend(valid_outputs["targets"])
                    valid_predictions.extend(valid_outputs["predictions"])
                valid_targets = torch.vstack(valid_targets)
                valid_predictions = torch.vstack(valid_predictions)
                valid_loss = loss_fn(valid_predictions, valid_targets)
                valid_targets = post_process_targets(valid_targets)
                valid_predictions = post_process_predictions(valid_predictions)
                valid_score = compute_matthews_corrcoef(targets=valid_targets, predictions=valid_predictions)
                logger.info(f"Iteration num: {iteration_num}, Train loss: {train_outputs['loss']}")
                logger.info(f"Iteration num: {iteration_num}, Valid loss: {valid_loss}, Valid score: {valid_score}")
            iteration_num += 1


if __name__ == "__main__":
    run()
