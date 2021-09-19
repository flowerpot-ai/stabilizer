import unittest
from transformers import AutoModel
from stabilizer.model import PoolerClassifier
from stabilizer.llrd import get_optimizer_parameters_with_llrd
from transformers import logging

logging.set_verbosity_error()


class TestLlrd(unittest.TestCase):
    def test_get_optimizer_parameters_with_llrd(self):
        transformer = AutoModel.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        model = PoolerClassifier(
            transformer=transformer,
            transformer_output_size=transformer.config.hidden_size,
            transformer_output_dropout_prob=0.1,
            num_classes=1,
        )
        optimizer_parameters = get_optimizer_parameters_with_llrd(
            model=model, peak_lr=2e-5, multiplicative_factor=0.95
        )
        self.assertEqual(len(optimizer_parameters), 14)


if __name__ == "__main__":
    unittest.main()
