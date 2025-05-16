import unittest
import torch
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from style_transfer import VGG19Features, load_image, save_image, gram_matrix

class TestStyleTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create dummy test images
        cls.content_size = (256, 256)
        cls.style_size = (256, 256)
        
        # Create and save test images
        content_img = Image.new('RGB', cls.content_size, color='blue')
        style_img = Image.new('RGB', cls.style_size, color='red')
        
        cls.content_path = cls.test_dir / "content.jpg"
        cls.style_path = cls.test_dir / "style.jpg"
        cls.output_path = cls.test_dir / "output.jpg"
        
        content_img.save(cls.content_path)
        style_img.save(cls.style_path)

    def test_vgg19_features(self):
        """Test VGG19 feature extraction."""
        model = VGG19Features()
        x = torch.randn(1, 3, 256, 256)
        
        # Get features
        style_features, content_features = model(x)
        
        # Check number of feature maps
        self.assertEqual(len(style_features), len(model.style_layers))
        self.assertEqual(len(content_features), len(model.content_layers))
        
        # Check if features are not None
        self.assertTrue(all(f is not None for f in style_features))
        self.assertTrue(all(f is not None for f in content_features))

    def test_image_loading(self):
        """Test image loading functionality."""
        # Test with size parameter
        img = load_image(self.content_path, size=128)
        self.assertEqual(img.shape[0], 1)  # batch size
        self.assertEqual(img.shape[1], 3)  # channels
        self.assertEqual(img.shape[2], 128)  # height
        self.assertEqual(img.shape[3], 128)  # width
        
        # Test without size parameter
        img = load_image(self.content_path)
        self.assertEqual(img.shape[0], 1)
        self.assertEqual(img.shape[1], 3)

    def test_image_saving(self):
        """Test image saving functionality."""
        # Create test tensor
        tensor = torch.randn(1, 3, 64, 64)
        save_path = self.test_dir / "test_save.jpg"
        
        # Save and verify
        save_image(tensor, save_path)
        self.assertTrue(save_path.exists())
        
        # Check if saved image can be opened
        img = Image.open(save_path)
        self.assertEqual(img.size, (64, 64))

    def test_gram_matrix(self):
        """Test Gram matrix calculation."""
        # Create test input
        x = torch.randn(1, 3, 32, 32)
        gram = gram_matrix(x)
        
        # Check shape
        self.assertEqual(gram.shape[0], 1)  # batch size
        self.assertEqual(gram.shape[1], 3)  # channels
        self.assertEqual(gram.shape[2], 3)  # channels
        
        # Check if matrix is symmetric
        self.assertTrue(torch.allclose(gram[0], gram[0].t()))

    def test_style_transfer_pipeline(self):
        """Test the complete style transfer pipeline."""
        # Load images
        content_img = load_image(self.content_path, size=128)
        style_img = load_image(self.style_path, size=128)
        
        # Initialize model
        model = VGG19Features()
        
        # Get features
        style_features, content_features = model(content_img)
        style_features_style, _ = model(style_img)
        
        # Calculate Gram matrices
        style_grams = [gram_matrix(feat) for feat in style_features_style]
        
        # Check shapes
        self.assertEqual(len(style_grams), len(model.style_layers))
        self.assertEqual(content_features[0].shape[1], 512)  # VGG19 feature dimension

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 