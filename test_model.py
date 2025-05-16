import unittest
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import load_dataset, prepare_input_data, generate_text

class TestGPT2Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.data_path = "data/sample_text.txt"
        cls.model_path = "models/test_model"
        cls.test_prompt = "Artificial intelligence is"
        
        # Ensure test data exists
        assert os.path.exists(cls.data_path), f"Test data not found at {cls.data_path}"
        
        # Initialize tokenizer and model
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Add padding token
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.resize_token_embeddings(len(cls.tokenizer))

    def test_data_loading(self):
        """Test if data can be loaded and preprocessed."""
        texts = load_dataset(self.data_path)
        self.assertIsInstance(texts, list)
        self.assertTrue(len(texts) > 0)
        
        # Test data preparation
        input_ids = prepare_input_data(texts, self.tokenizer)
        self.assertIsInstance(input_ids, torch.Tensor)

    def test_text_generation(self):
        """Test if model can generate text."""
        generated_text = generate_text(
            self.model,
            self.tokenizer,
            self.test_prompt,
            max_length=50
        )
        
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > len(self.test_prompt))
        self.assertTrue(generated_text.startswith(self.test_prompt))

    def test_model_saving_loading(self):
        """Test if model can be saved and loaded."""
        # Save model
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        
        # Load model
        loaded_model = GPT2LMHeadModel.from_pretrained(self.model_path)
        loaded_tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        
        # Test generation with loaded model
        generated_text = generate_text(
            loaded_model,
            loaded_tokenizer,
            self.test_prompt
        )
        self.assertIsInstance(generated_text, str)

if __name__ == '__main__':
    unittest.main() 