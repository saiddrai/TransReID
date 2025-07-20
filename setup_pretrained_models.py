#!/usr/bin/env python3
"""
Setup script to download and organize pretrained models for TransReID training.
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filename, directory):
    """Download a file if it doesn't exist."""
    filepath = directory / filename
    
    if filepath.exists():
        print(f"✓ {filename} already exists")
        return str(filepath)
    
    print(f"Downloading {filename}...")
    directory.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, str(filepath))
        print(f"✓ Downloaded {filename}")
        return str(filepath)
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return None

def setup_pretrained_models():
    """Download pretrained Vision Transformer models."""
    
    # Create checkpoints directory
    checkpoint_dir = Path.home() / ".cache" / "torch" / "checkpoints"
    
    # Model URLs and filenames
    models = {
        "jx_vit_base_p16_224-80ecf9dd.pth": 
            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        "vit_small_p16_224-15ec54c9.pth":
            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
        "deit_base_distilled_patch16_224-df68dfff.pth":
            "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
        "deit_small_distilled_patch16_224-649709d9.pth":
            "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    }
    
    print("Setting up pretrained models for TransReID...")
    print(f"Target directory: {checkpoint_dir}")
    
    downloaded_models = {}
    
    for filename, url in models.items():
        filepath = download_file(url, filename, checkpoint_dir)
        if filepath:
            downloaded_models[filename] = filepath
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    
    if downloaded_models:
        print("\nDownloaded models:")
        for filename, filepath in downloaded_models.items():
            print(f"  ✓ {filename}")
            print(f"    Path: {filepath}")
        
        print(f"\nAll models are stored in: {checkpoint_dir}")
        print("\nYou can now run training with the provided config files.")
        print("The config files will automatically use these pretrained models.")
    else:
        print("\nNo models were downloaded. Please check your internet connection.")
    
    return downloaded_models

if __name__ == "__main__":
    setup_pretrained_models()