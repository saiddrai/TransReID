# TransReID Training Guide for Person Re-Identification Datasets

This guide explains how to train TransReID models on the three person re-identification datasets: Market-1501, MSMT17, and DukeMTMC-reID.

## Prerequisites

### 1. Environment Setup
```bash
# Install required dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models
Download one of the ImageNet pre-trained transformer models:
- **ViT-Base** (Recommended): [jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
- **ViT-Small**: [vit_small_p16_224-15ec54c9.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth)
- **DeiT-Base**: [deit_base_distilled_patch16_224-df68dfff.pth](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth)
- **DeiT-Small**: [deit_small_distilled_patch16_224-649709d9.pth](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth)

Store the downloaded model in a directory like `~/.cache/torch/checkpoints/` or update the `PRETRAIN_PATH` in config files.

### 3. Dataset Organization
Create a `data` directory and organize your datasets as follows:

```
data/
├── market1501/
│   └── images/
├── MSMT17/
│   └── images/
└── dukemtmcreid/
│   └── images/
```

**Important**: 
- For **Market-1501**: Use folder name `market1501` (lowercase)
- For **MSMT17**: Use folder name `MSMT17` (uppercase)
- For **DukeMTMC-reID**: Use folder name `dukemtmcreid` (lowercase)

## Training Commands

### Option 1: Using Pre-configured YAML Files (Recommended)

#### Market-1501
```bash
# Basic transformer baseline
python train.py --config_file configs/Market/vit_base.yml MODEL.DEVICE_ID "('0')"

# TransReID with all components (SIE + JPM)
python train.py --config_file configs/Market/vit_transreid.yml MODEL.DEVICE_ID "('0')"

# TransReID with stride optimization (Best performance)
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

#### MSMT17
```bash
# TransReID with stride optimization
python train.py --config_file configs/MSMT17/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

#### DukeMTMC-reID
```bash
# Basic transformer baseline
python train.py --config_file configs/DukeMTMC/vit_base.yml MODEL.DEVICE_ID "('0')"

# TransReID with all components
python train.py --config_file configs/DukeMTMC/vit_transreid.yml MODEL.DEVICE_ID "('0')"

# TransReID with stride optimization (Best performance)
python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

### Option 2: Using Generic Configuration with Parameters

```bash
python train.py --config_file configs/transformer_base.yml \
    MODEL.DEVICE_ID "('0')" \
    MODEL.STRIDE_SIZE [12,12] \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_VIEW False \
    MODEL.JPM True \
    MODEL.TRANSFORMER_TYPE 'vit_base_patch16_224_TransReID' \
    DATASETS.NAMES "('market1501')" \
    OUTPUT_DIR '../logs/market1501_custom'
```

Replace `DATASETS.NAMES` with:
- `"('market1501')"` for Market-1501
- `"('msmt17')"` for MSMT17  
- `"('dukemtmc')"` for DukeMTMC-reID

## Configuration Details

### Key Configuration Parameters

#### Model Components
- **SIE_CAMERA**: Side Information Embedding with camera info (True/False)
- **SIE_VIEW**: Side Information Embedding with view info (True/False)  
- **JPM**: Jigsaw Patch Module (True/False)
- **STRIDE_SIZE**: Patch stride size, e.g., [16,16], [14,14], [12,12]

#### Transformer Types
- `'vit_base_patch16_224_TransReID'`: ViT-Base (Recommended)
- `'vit_small_patch16_224_TransReID'`: ViT-Small
- `'deit_base_patch16_224_TransReID'`: DeiT-Base
- `'deit_small_patch16_224_TransReID'`: DeiT-Small

#### Performance vs Memory Trade-offs
- **TransReID with stride [12,12]**: 12GB GPU memory, highest accuracy
- **TransReID with stride [16,16]**: 7GB GPU memory, good accuracy
- **Baseline ViT**: Lower memory usage, baseline accuracy

### Training Configuration
- **Epochs**: 120
- **Batch Size**: 64
- **Learning Rate**: 0.008 (SGD)
- **Image Size**: 256x128
- **Optimizer**: SGD with momentum
- **Scheduler**: Cosine annealing with warmup

## Expected Results

### Performance Benchmarks (mAP | Rank-1)
- **Market-1501**: 89.0 | 95.1
- **MSMT17**: 67.8 | 85.3  
- **DukeMTMC-reID**: 82.2 | 90.7

## Training Process

1. **Training Duration**: ~8-12 hours on single V100 GPU
2. **Checkpoints**: Saved every 120 epochs in `OUTPUT_DIR`
3. **Logs**: Training logs saved in `OUTPUT_DIR`
4. **Evaluation**: Automatic evaluation at the end of training

## GPU Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory for stride optimization
- **Multi-GPU**: Use `dist_train.sh` for distributed training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use larger stride size
2. **Dataset Not Found**: Check dataset folder names and structure
3. **Pretrained Model Error**: Verify pretrained model path in config

### Custom Configuration

To modify configurations:
1. Copy an existing config file
2. Update parameters as needed
3. Use the custom config file for training

## Sequential Training

To train on all three datasets sequentially:

```bash
# Train on Market-1501
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# Train on MSMT17  
python train.py --config_file configs/MSMT17/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# Train on DukeMTMC-reID
python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

Each training will create separate output directories with trained models and logs.