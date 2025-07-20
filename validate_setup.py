#!/usr/bin/env python3
"""
Validation script to test TransReID training setup.
This script validates that all components are properly configured.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        import torch
        import torchvision
        import timm
        import yacs
        import cv2
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ TorchVision {torchvision.__version__}")
        print(f"✓ timm {timm.__version__}")
        print(f"✓ OpenCV available")
        print(f"✓ YACS available")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_transreid_modules():
    """Test if TransReID modules can be imported."""
    print("\nTesting TransReID modules...")
    
    try:
        from config import cfg
        from datasets import make_dataloader
        from model import make_model
        from solver import make_optimizer
        from loss import make_loss
        print("✓ All TransReID modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ TransReID import error: {e}")
        return False

def test_config_files():
    """Test if configuration files are accessible."""
    print("\nTesting configuration files...")
    
    config_files = [
        "configs/Market/vit_transreid_stride.yml",
        "configs/MSMT17/vit_transreid_stride.yml", 
        "configs/DukeMTMC/vit_transreid_stride.yml"
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file}")
            all_exist = False
    
    return all_exist

def test_pretrained_models():
    """Test if pretrained models are available."""
    print("\nTesting pretrained models...")
    
    checkpoint_dir = Path.home() / ".cache" / "torch" / "checkpoints"
    model_file = checkpoint_dir / "jx_vit_base_p16_224-80ecf9dd.pth"
    
    if model_file.exists():
        print(f"✓ ViT-Base pretrained model found: {model_file}")
        return True
    else:
        print(f"✗ ViT-Base pretrained model missing: {model_file}")
        print("  Run: python setup_pretrained_models.py")
        return False

def test_training_script():
    """Test if the training script can be invoked."""
    print("\nTesting training script...")
    
    # Test if train.py can show help without errors
    import subprocess
    try:
        result = subprocess.run([sys.executable, "train.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Training script is functional")
            return True
        else:
            print(f"✗ Training script error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to test training script: {e}")
        return False

def main():
    """Run all validation tests."""
    print("="*50)
    print("TransReID Setup Validation")
    print("="*50)
    
    tests = [
        test_imports,
        test_transreid_modules, 
        test_config_files,
        test_pretrained_models,
        test_training_script
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("Validation Summary")
    print("="*50)
    
    if all(results):
        print("🎉 All tests passed! Your TransReID setup is ready for training.")
        print("\nNext steps:")
        print("1. Organize your datasets in the 'data' directory")
        print("2. Run: ./quick_start.sh to verify dataset setup")
        print("3. Start training with the provided commands")
    else:
        print("❌ Some tests failed. Please address the issues above.")
        failed_count = len([r for r in results if not r])
        print(f"\nFailed tests: {failed_count}/{len(results)}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)