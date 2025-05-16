import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained VGG19 model
        vgg19 = models.vgg19(pretrained=True).features.eval()
        self.features = vgg19
        # Layers for style and content loss
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        style_features = []
        content_features = []
        for name, layer in self.features.named_children():
            x = layer(x)
            if name in self.style_layers:
                style_features.append(x)
            if name in self.content_layers:
                content_features.append(x)
        return style_features, content_features

def load_image(image_path, size=None):
    """Load and preprocess image."""
    image = Image.open(image_path)
    if size:
        image = image.resize((size, size))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, path):
    """Save the generated image."""
    image = tensor.clone().detach()
    image = image.squeeze(0)
    
    # Denormalize
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # Convert to PIL Image and save
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save(path)

def gram_matrix(x):
    """Calculate Gram matrix."""
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(c * h * w)

def style_transfer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images
    content_img = load_image(args.content_image, args.image_size).to(device)
    style_img = load_image(args.style_image, args.image_size).to(device)
    
    # Initialize model
    model = VGG19Features().to(device)
    
    # Initialize target image
    target = content_img.clone().requires_grad_(True)
    
    # Setup optimizer
    optimizer = optim.LBFGS([target])
    
    # Get original features
    style_features_orig, _ = model(style_img)
    _, content_features_orig = model(content_img)
    
    # Calculate target Gram matrices
    style_grams = [gram_matrix(feat) for feat in style_features_orig]
    
    def closure():
        optimizer.zero_grad()
        
        # Get current features
        style_features, content_features = model(target)
        
        # Content loss
        content_loss = sum(nn.MSELoss()(cf[0], cf_orig[0])
                          for cf, cf_orig in zip(content_features, content_features_orig))
        
        # Style loss
        style_loss = 0
        for sf, gram_orig in zip(style_features, style_grams):
            gram = gram_matrix(sf)
            style_loss += nn.MSELoss()(gram[0], gram_orig[0])
        
        # Total loss
        total_loss = args.content_weight * content_loss + args.style_weight * style_loss
        total_loss.backward()
        
        return total_loss
    
    logger.info("Starting style transfer...")
    for step in range(args.num_steps):
        optimizer.step(closure)
        if (step + 1) % 50 == 0:
            logger.info(f"Step {step+1}/{args.num_steps}")
    
    # Save result
    save_image(target, args.output_image)
    logger.info(f"Style transfer complete. Result saved to {args.output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content_image", type=str, required=True,
                        help="Path to content image")
    parser.add_argument("--style_image", type=str, required=True,
                        help="Path to style image")
    parser.add_argument("--output_image", type=str, required=True,
                        help="Path to save the output image")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size of the images")
    parser.add_argument("--content_weight", type=float, default=1,
                        help="Weight for content loss")
    parser.add_argument("--style_weight", type=float, default=1e6,
                        help="Weight for style loss")
    parser.add_argument("--num_steps", type=int, default=300,
                        help="Number of optimization steps")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_image).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    style_transfer(args) 