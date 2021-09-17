import logging
from torch import nn

AUTOENCODINGMODELS = ["Bert", "Roberta", "DistilBert", "Albert", "XLMRoberta"]


def ReinitAutoencoderModel(model, reinit_num_layers=0):

    """reinitialize autoencoder model layers"""

    if not any(k in model.model.config.architectures[0] for k in AUTOENCODINGMODELS):
        logging.ERROR("Model not Autoencoding based")
        return model

    if reinit_num_layers:
        for layer in model.model.encoder.layer[-reinit_num_layers:]:

            for module in layer.modules():

                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(
                        mean=0.0, std=model.model.config.initializer_range
                    )
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(
                        mean=0.0, std=model.model.config.initializer_range
                    )
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    return model
