import torch
import unittest
from torch import nn as nn
from transformers import AutoModel
from stabilizer.model import PoolerClassifier
from stabilizer.reinitialize import reinit_autoencoder_model
from transformers import logging

logging.set_verbosity_error()


class TestReinitialize(unittest.TestCase):
    def test_reinit_autoencoder_model(self):
        transformer = AutoModel.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        transformer.encoder = reinit_autoencoder_model(transformer.encoder, reinit_num_layers=1)
        model = PoolerClassifier(
            transformer=transformer,
            transformer_output_size=transformer.config.hidden_size,
            transformer_output_dropout_prob=0.1,
            num_classes=1,
        )

        vals = []
        for module in model.transformer.encoder.layer[-1].modules():

            if isinstance(module, nn.Linear):
                k = torch.isclose(module.weight.data.mean(), torch.tensor(0.0), atol=1e-4)
                v = torch.isclose(
                    module.weight.data.std().detach().cpu(),
                    torch.tensor(0.02),
                    atol=1e-4,
                )
                vals.append(k.cpu().numpy())
                vals.append(v.cpu().numpy())
                vals.append(all(module.bias.data.detach().cpu() == torch.zeros(module.bias.data.shape)))

        self.assertEqual(all(vals), True)


if __name__ == "__main__":
    unittest.main()
