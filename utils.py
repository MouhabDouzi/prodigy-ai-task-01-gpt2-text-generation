import torch
from transformers import GPT2Tokenizer
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> List[str]:
    """Load and preprocess the training data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')
        return [line for line in data if line.strip()]
    except FileNotFoundError:
        logger.error(f"Could not find file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def prepare_input_data(texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 512) -> torch.Tensor:
    """Tokenize and prepare input data for training."""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings['input_ids']

def save_model_checkpoint(model, tokenizer, output_dir: str, step: int):
    """Save model and tokenizer checkpoints."""
    try:
        checkpoint_dir = f"{output_dir}/checkpoint-{step}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

def generate_text(model, tokenizer, prompt: str, max_length: int = 100) -> str:
    """Generate text based on a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise 