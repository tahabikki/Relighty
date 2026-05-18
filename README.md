# Relighty — Face Shadow Removal

A deep learning project for automatic shadow removal from face images using U-Net with ResNet34 backbone.

## Prerequisites

- **Python 3.12+**
- **CUDA 12.4** (for GPU acceleration, optional)
- **Git** (to clone the repository)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Relighty
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

**Recommended: Use Setup Scripts (includes PyTorch)**

```bash
# Windows (CUDA 12.4)
setup.bat

# macOS / Linux
bash setup.sh
```

**Manual Installation:**
```bash
# Install PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
# or for CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

## Project Structure

```
Relighty/
├── configs/              # Configuration files
│   └── config.yaml       # Training hyperparameters
├── models/               # Model definitions
│   └── shadow_remover.py
├── train/                # Training script
│   └── train.py
├── evaluation/           # Evaluation & inference
│   ├── evaluate.py
│   └── inference.py
├── utils/                # Utility functions
│   ├── dataset.py
│   ├── losses.py
│   ├── color_utils.py
│   └── face_analyzer.py
├── dataset/              # Dataset folder
│   ├── input/            # Original images with shadows
│   ├── target/           # Shadow-free reference images
│   └── splits/           # Train/val split files
├── checkpoints/          # Trained model weights
│   └── 1-100/           # Models from epochs 1-100
├── Results/              # Output results
├── requirements.txt      # Python dependencies
├── setup.bat             # Windows setup
└── setup.sh              # Linux/macOS setup
```

## Quick Start

### 1. Prepare Dataset

Place your images in:
- `dataset/input/` — images with shadows
- `dataset/target/` — shadow-free images

The dataset splits (train/val) are configured in:
- `dataset/splits/train_input.txt`
- `dataset/splits/train_target.txt`
- `dataset/splits/val_input.txt`
- `dataset/splits/val_target.txt`

### 2. Configure Training

Edit `configs/config.yaml` to adjust:
- Model architecture
- Batch size & learning rate
- Training epochs
- Device (cuda/cpu/mps)

### 3. Train Model

```bash
python train/train.py
```

Models are saved to `checkpoints/1-100/` every 10 epochs (configurable).

### 4. Evaluate & Run Inference

```bash
# Evaluate on validation set
python evaluation/evaluate.py

# Run inference on test images
python evaluation/inference.py --input <image-path> --output <output-dir>
```

## Pretrained Models

Pre-trained checkpoints are included in `checkpoints/1-100/`:
- `shadow_removal_best.pth` — Best validation performance
- `shadow_removal_latest.pth` — Latest checkpoint
- `shadow_removal_epoch_*.pth` — Individual epoch checkpoints

## Configuration

Key settings in `configs/config.yaml`:

```yaml
model:
  name: unet_resnet34
  input_size: 256
  channels: 3

training:
  batch_size: 64
  epochs: 200
  lr: 0.0001
  device: auto  # auto | cuda | cpu | mps

early_stopping:
  enabled: true
  patience: 15
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| RAM | 4GB | 16GB+ |
| GPU VRAM | N/A | 4GB+ (CUDA) |
| Disk | 5GB | 20GB+ |

## Troubleshooting

**Issue: ModuleNotFoundError for torch**
- Run: `pip install torch torchvision`

**Issue: CUDA not available**
- Verify NVIDIA GPU drivers are installed
- Or switch to CPU mode in `config.yaml`

**Issue: DataLoader errors on Windows**
- Set `num_workers: 0` in `config.yaml`

## License

[Add your license here]

## Author

Relighty Development Team
