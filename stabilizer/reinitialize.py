import logging
import transformers
from torch import nn

AUTOENCODINGMODELS = [
    "Bert",
    "Roberta",
    "DistilBert",
    "Albert",
    "XLMRoberta",
    "BertModel",
]


def reinit_autoencoder_model(encoder, reinit_num_layers=0):
    """reinitialize autoencoder model layers"""

    if not any(k in encoder.config.architectures[0] for k in AUTOENCODINGMODELS):
        logging.ERROR("Model not Autoencoding based")
        return encoder

    if reinit_num_layers:
        for layer in encoder.layer[-reinit_num_layers:]:

            for module in layer.modules():

                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(
                        mean=0.0, std=encoder.config.initializer_range
                    )
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(
                        mean=0.0, std=encoder.config.initializer_range
                    )
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

        return encoder
