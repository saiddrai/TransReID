#!/bin/bash

# Quick start script for TransReID training on person datasets
# This script helps setup and train on Market-1501, MSMT17, and DukeMTMC-reID

echo "========================================="
echo "TransReID Person Re-ID Training Setup"
echo "========================================="

# Function to check if directory exists
check_directory() {
    if [ -d "$1" ]; then
        echo "✓ Found: $1"
        return 0
    else
        echo "✗ Missing: $1"
        return 1
    fi
}

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo "✓ Found: $1"
        return 0
    else
        echo "✗ Missing: $1"
        return 1
    fi
}

echo ""
echo "1. Checking dataset directories..."
echo "-----------------------------------"

# Check for datasets
DATASETS_OK=true

if ! check_directory "data/market1501"; then
    echo "  Please download Market-1501 and place it in data/market1501/"
    DATASETS_OK=false
fi

if ! check_directory "data/MSMT17"; then
    echo "  Please download MSMT17 and place it in data/MSMT17/"
    DATASETS_OK=false
fi

if ! check_directory "data/dukemtmcreid"; then
    echo "  Please download DukeMTMC-reID and place it in data/dukemtmcreid/"
    DATASETS_OK=false
fi

echo ""
echo "2. Checking pretrained models..."
echo "--------------------------------"

# Check for pretrained model
CHECKPOINT_DIR="$HOME/.cache/torch/checkpoints"
PRETRAINED_MODEL="$CHECKPOINT_DIR/jx_vit_base_p16_224-80ecf9dd.pth"

if ! check_file "$PRETRAINED_MODEL"; then
    echo "Setting up pretrained models..."
    python setup_pretrained_models.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Pretrained models setup complete"
    else
        echo "✗ Failed to setup pretrained models"
        echo "Please run: python setup_pretrained_models.py"
    fi
fi

echo ""
echo "3. Training Commands"
echo "-------------------"

if [ "$DATASETS_OK" = true ]; then
    echo "Your datasets are ready! Use these commands to start training:"
    echo ""
    echo "Market-1501 (TransReID with stride optimization):"
    echo "  python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID \"('0')\""
    echo ""
    echo "MSMT17 (TransReID with stride optimization):"
    echo "  python train.py --config_file configs/MSMT17/vit_transreid_stride.yml MODEL.DEVICE_ID \"('0')\""
    echo ""
    echo "DukeMTMC-reID (TransReID with stride optimization):"
    echo "  python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID \"('0')\""
    echo ""
    echo "Note: Change MODEL.DEVICE_ID to your GPU ID (e.g., '1', '2', etc.)"
    echo ""
    echo "Training outputs will be saved in ../logs/ directory"
    echo ""
    echo "For more options, see TRAINING_GUIDE.md"
else
    echo "Please setup your datasets first. See TRAINING_GUIDE.md for details."
fi

echo ""
echo "========================================="
echo "Setup check complete!"
echo "========================================="