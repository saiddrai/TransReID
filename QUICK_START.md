# Quick Start Guide: Training TransReID on Person Datasets

## Overview
This guide shows you how to train TransReID (Transformer-based Person Re-Identification) on three popular person datasets: Market-1501, MSMT17, and DukeMTMC-reID.

## Step 1: Environment Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Pretrained Models
```bash
python setup_pretrained_models.py
```

This will download the required Vision Transformer pretrained models to `~/.cache/torch/checkpoints/`.

## Step 2: Dataset Preparation

### Download Datasets
Download the three person re-identification datasets:
- **Market-1501**: [Download Link](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
- **MSMT17**: [Download Link](https://arxiv.org/abs/1711.08565) 
- **DukeMTMC-reID**: [Download Link](https://arxiv.org/abs/1609.01775)

### Organize Datasets
Create a `data` directory and organize as follows:
```
data/
├── market1501/          # Market-1501 dataset (lowercase)
│   └── images/
├── MSMT17/              # MSMT17 dataset (uppercase)
│   └── images/
└── dukemtmcreid/        # DukeMTMC-reID dataset (lowercase)
    └── images/
```

**Important**: Use exact folder names as shown above.

## Step 3: Quick Setup Check
```bash
./quick_start.sh
```

This script will verify your setup and show training commands.

## Step 4: Start Training

### Market-1501
```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

### MSMT17
```bash
python train.py --config_file configs/MSMT17/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

### DukeMTMC-reID
```bash
python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

**Note**: Change `'0'` to your GPU ID if using a different GPU.

## Step 5: Monitor Training

- **Training logs**: Saved in `../logs/[dataset]_vit_transreid_stride/`
- **Model checkpoints**: Saved every 120 epochs
- **Training time**: ~8-12 hours per dataset on V100 GPU

## Alternative Training Options

### Basic Transformer (Lower Memory Usage)
```bash
# Market-1501 baseline
python train.py --config_file configs/Market/vit_base.yml MODEL.DEVICE_ID "('0')"

# DukeMTMC-reID baseline  
python train.py --config_file configs/DukeMTMC/vit_base.yml MODEL.DEVICE_ID "('0')"
```

### TransReID without Stride Optimization (Medium Memory)
```bash
# Market-1501 TransReID
python train.py --config_file configs/Market/vit_transreid.yml MODEL.DEVICE_ID "('0')"

# DukeMTMC-reID TransReID
python train.py --config_file configs/DukeMTMC/vit_transreid.yml MODEL.DEVICE_ID "('0')"
```

## Expected Performance

| Dataset | mAP | Rank-1 | Training Time | GPU Memory |
|---------|-----|--------|---------------|------------|
| Market-1501 | 89.0 | 95.1 | ~8 hours | 12GB |
| MSMT17 | 67.8 | 85.3 | ~12 hours | 12GB |
| DukeMTMC-reID | 82.2 | 90.7 | ~8 hours | 12GB |

## GPU Requirements

- **Minimum**: 8GB GPU memory (for baseline models)
- **Recommended**: 12GB+ GPU memory (for stride optimization)
- **Supported**: Single GPU training (distributed training available for VehicleID)

## Troubleshooting

### Common Issues

1. **"No module named 'torch._six'"**
   - Fixed in the repository. PyTorch compatibility issue resolved.

2. **"CUDA out of memory"**
   - Use configs without stride optimization: `vit_transreid.yml` instead of `vit_transreid_stride.yml`
   - Or use baseline configs: `vit_base.yml`

3. **"Dataset not found"**
   - Check dataset folder names exactly match: `market1501`, `MSMT17`, `dukemtmcreid`
   - Ensure `images/` subdirectory exists in each dataset folder

4. **"Pretrained model not found"**
   - Run: `python setup_pretrained_models.py`
   - Or manually download and place in `~/.cache/torch/checkpoints/`

### Manual Pretrained Model Setup
If automatic download fails:
1. Download [jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
2. Place in `~/.cache/torch/checkpoints/`

## Next Steps

1. **Evaluation**: Use `test.py` with trained models
2. **Custom configs**: Modify YAML files for custom training
3. **Multiple datasets**: Train sequentially on all three datasets

For detailed documentation, see `TRAINING_GUIDE.md`.