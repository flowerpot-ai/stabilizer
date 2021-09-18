import unittest
from transformers import AutoModel
from stabilizer.model import PoolerClassifier
from stabilizer.reinitialize import reinit_autoencoder_model

class TestReinitialize(unittest.TestCase):
    def test_reinit_autoencoder_model(self):
        transformer = AutoModel.from_pretrained(pretrained_model_name_or_path='models/bert-base-uncased/',
                                        hidden_dropout_prob=0.1,
                                        attention_probs_dropout_prob=0.1)
        transformer.encoder = reinit_autoencoder_model(transformer.encoder, reinit_num_layers=2)
        model = PoolerClassifier(transformer=transformer,
                            transformer_output_size=transformer.config.hidden_size,
                            transformer_output_dropout_prob=0.1,
                            num_classes=1)
        self.assertEqual(model.transformer.encoder.config.attention_probs_dropout_prob, 0.1)        


if __name__ == '__main__':
    unittest.main()