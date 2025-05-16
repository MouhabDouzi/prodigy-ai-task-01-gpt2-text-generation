import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
from utils import load_dataset, prepare_input_data, save_model_checkpoint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings.shape[0]

    def __getitem__(self, idx):
        return self.encodings[idx]

def train(args):
    # Initialize wandb
    wandb.init(project="gpt2-finetuning", config=args)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Load and prepare dataset
    texts = load_dataset(args.data_path)
    input_ids = prepare_input_data(texts, tokenizer, args.max_length)
    dataset = TextDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            wandb.log({"batch_loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss})

        if (epoch + 1) % args.save_every == 0:
            save_model_checkpoint(model, tokenizer, args.output_dir, epoch + 1)

    # Save final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every n epochs")
    
    args = parser.parse_args()
    train(args) 