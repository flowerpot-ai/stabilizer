from torch import nn
from stabilizer.reproducibility import seed_torch


class PoolerClassifier(nn.Module):
    def __init__(
        self,
        transformer,
        transformer_output_size,
        transformer_output_dropout_prob,
        num_classes,
        task_specific_layer_seed=None,
    ):
        super(PoolerClassifier, self).__init__()
        self.transformer = transformer
        self.transformer_output_size = transformer_output_size
        self.transformer_output_dropout = nn.Dropout(p=transformer_output_dropout_prob)
        if task_specific_layer_seed is not None:
            seed_torch(task_specific_layer_seed)
        self.task_specific_layer = nn.Linear(self.transformer_output_size, num_classes)

    def forward(self, inputs):
        transformer_outputs = self.transformer(**inputs)
        vectors = self.transformer_output_dropout(transformer_outputs.pooler_output)
        logits = self.task_specific_layer(vectors)
        return logits
