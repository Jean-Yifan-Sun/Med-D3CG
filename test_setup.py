#!/usr/bin/env python
"""
Test script to verify PyTorch and data loading setup
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pytorch():
    """Test PyTorch installation"""
    logger.info("=" * 50)
    logger.info("Testing PyTorch Installation")
    logger.info("=" * 50)
    
    try:
        import torch
        logger.info(f"✓ PyTorch imported successfully")
        logger.info(f"  Version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU device: {torch.cuda.get_device_name(0)}")
        
        # Test tensor creation
        x = torch.randn(2, 3)
        logger.info(f"  Tensor creation: OK")
        logger.info(f"  Tensor shape: {x.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ PyTorch test failed: {e}")
        return False

def test_image_loading():
    """Test image loading from dataset"""
    logger.info("=" * 50)
    logger.info("Testing Image Loading")
    logger.info("=" * 50)
    
    from PIL import Image
    import torch
    from torchvision import transforms
    
    test_dir = "./data/acdc_wholeheart/25022_JPGs"
    
    try:
        # Check if directory exists
        if not os.path.isdir(test_dir):
            logger.error(f"✗ Directory not found: {test_dir}")
            logger.info(f"  Current working directory: {os.getcwd()}")
            logger.info(f"  Directory contents: {os.listdir('./data') if os.path.isdir('./data') else 'data/ not found'}")
            return False
        
        logger.info(f"✓ Directory found: {test_dir}")
        
        # Count images
        image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"  Image files found: {len(image_files)}")
        
        if len(image_files) == 0:
            logger.error(f"✗ No image files found in {test_dir}")
            return False
        
        # Try loading a sample image
        sample_image = image_files[0]
        sample_path = os.path.join(test_dir, sample_image)
        
        logger.info(f"  Loading sample image: {sample_image}")
        img = Image.open(sample_path)
        logger.info(f"  Image size: {img.size}")
        logger.info(f"  Image mode: {img.mode}")
        
        # Test transformation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img_tensor = transform(img)
        logger.info(f"  Transformed tensor shape: {img_tensor.shape}")
        logger.info(f"✓ Image loading test passed")
        
        return True
    except Exception as e:
        logger.error(f"✗ Image loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_class():
    """Test the UnconditionalDataset class"""
    logger.info("=" * 50)
    logger.info("Testing UnconditionalDataset Class")
    logger.info("=" * 50)
    
    try:
        # Import after PyTorch test
        from unconditional_train import UnconditionalDataset
        
        test_dir = "./data/acdc_wholeheart/25022_JPGs"
        
        if not os.path.isdir(test_dir):
            logger.error(f"✗ Directory not found: {test_dir}")
            return False
        
        logger.info(f"  Initializing UnconditionalDataset with {test_dir}")
        dataset = UnconditionalDataset(test_dir, image_size=256, is_rgb=False)
        
        logger.info(f"✓ Dataset initialized successfully")
        logger.info(f"  Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            logger.error(f"✗ Dataset is empty!")
            return False
        
        # Try loading a sample
        logger.info(f"  Loading sample from dataset...")
        sample = dataset[0]
        logger.info(f"  Sample image shape: {sample['image'].shape}")
        logger.info(f"✓ Dataset class test passed")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dataset class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("Starting validation tests...")
    logger.info("")
    
    results = []
    
    # Test PyTorch
    pytorch_ok = test_pytorch()
    results.append(("PyTorch", pytorch_ok))
    logger.info("")
    
    # Test image loading
    image_loading_ok = test_image_loading()
    results.append(("Image Loading", image_loading_ok))
    logger.info("")
    
    # Test dataset class (only if PyTorch is OK)
    if pytorch_ok:
        dataset_ok = test_dataset_class()
        results.append(("Dataset Class", dataset_ok))
    logger.info("")
    
    # Summary
    logger.info("=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    
    all_ok = all(result[1] for result in results)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info("")
    if all_ok:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
