# Stabilizer
### Stabilize and achieve excellent performance with transformers.  
The stabilizer library offer solutions to tackle one of the biggest challenges that comes along with training State of the art Transformer models, **Unstable training**

## Unstable training
Unstable training is the phenomenon in which training large transformer models with trivial changes such as changing the random seed drastically changes the performance of the model. Here is a screenshot of finetuning the CoLA dataset from GLUE tasks with two different random seeds applied only to the dropout of the transformer model.

![dropout_random_seed](https://i.ibb.co/jyx3tLT/baseline-dropout-seed.png)

## Installation
`pip install stabilizer`


## Techniques currently implemented in this library
1. Reinitialization
2. Layerwise Learning Rate Decay


### Reinitialization
Reinitialize the last `n` layers of the transformer encoder. This technique works well because we reinitialize the task specific parameters that the pretrained models have learnt specific to the pretraining task.
```python
from stabilizer.reinitialize import reinit_autoencoder_model
from transformers import AutoModel

transformer = AutoModel.from_pretrained(
    pretrained_model_name_or_path="bert-base-uncased",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
transformer.encoder = reinit_autoencoder_model(
    transformer.encoder, reinit_num_layers=1
)
```
Here is the result of the same model but reinitialized last 4 layers applied on the CoLA dataset. You can see that the model has converged to almost the same performance with reinitialization.
![reinit_random_seed](https://i.ibb.co/1MyPbfG/reinit-dropout-seed.png)



### Layerwise Learning Rate Decay
Apply layerwise learning rate to the transformer layers. Starting from the task specific layer every layer before it gets an exponentially decreasing learning rate. 


```python

from stabilizer.llrd import get_optimizer_parameters_with_llrd
from stabilizer.model import PoolerClassifier

from transformers import AdamW, AutoModel


    transformer = AutoModel.from_pretrained(
        pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
        hidden_dropout_prob=config["dropout_prob"],
        attention_probs_dropout_prob=config["dropout_prob"],
    )
    
    model = PoolerClassifier(
        transformer=transformer,
        transformer_output_size=transformer.config.hidden_size,
        transformer_output_dropout_prob=config["dropout_prob"],
        num_classes=config["num_classes"],
        task_specific_layer_seed=config["layer_initialization_seed"],
    )

    model_parameters = get_optimizer_parameters_with_llrd(
        model=model,
        peak_lr=config["lr"],
        multiplicative_factor=config["multiplicative_factor"],
    )
    optimizer = AdamW(params=model_parameters, lr=config["lr"])


```

Here is the result of the same model but with LLRD applied on the CoLA dataset. Here you can see that the model has diverged quite a lot by applying LLRD. Therefore as we discussed earlier their is no universal remedy yet but some techniques work well on some datasets
![llrd_random_seed](https://i.ibb.co/jkLJSP0/llrd-dropout-seed.png)