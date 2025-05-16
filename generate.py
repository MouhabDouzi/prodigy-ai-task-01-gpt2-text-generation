import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import generate_text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # Load model and tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Generate text
    try:
        generated_text = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length
        )
        print("\nGenerated Text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to generate from")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum length of generated text")
    
    args = parser.parse_args()
    main(args) 