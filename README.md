# Relighty — Face and Neck Shadow Removal

A modular deep learning project for automatic shadow removal from face and neck regions using U-Net with ResNet34 backbone.

## Features

- **Transparent Background** - PNG output with alpha channel
- **Face + Neck + Forehead** - All connected as one piece
- **Texture Preservation** - Keeps original skin texture while removing shadows
- **Cross-Platform** - Works on Windows, macOS, and Linux (CUDA, MPS, CPU)
- **Config-Driven** - All paths from one config file

---

## Quick Start

### Step 1: Configure Dataset Path

Edit `configs/config.yaml`:

```yaml
data:
  dataset_root: dataset  # or absolute path: /path/to/your/dataset
```

### Step 2: Prepare Dataset (PNG with transparent background)

```bash
# Remove background from input images
python -m masking_bg.mask_bg_remove --input-folder dataset/input --output-folder dataset/input

# Remove background from target images
python -m masking_bg.mask_bg_remove --input-folder dataset/target --output-folder dataset/target
```

### Step 3: Create Train/Val Split

```bash
python -m utils.split
```

### Step 4: Train the Model

```bash
python -m training.train
```

### Step 5: Run Inference (PNG with transparent background)

```bash
# Single image
python -m evaluation.inference --input photo.jpg --output result.png

# Batch folder
python -m evaluation.inference --input Results/input --output Results/output

# Or use deployment pipeline
python -m deployment.pipeline --input photo.jpg --output result.png
```

---

## Complete Pipeline

### Training Flow
```
Input Image → mask_bg_remove.py → PNG with transparent BG
                                           ↓
                                    dataset/input/
                                    dataset/target/
                                           ↓
                              python -m utils.split
                                           ↓
                              dataset/splits/ (train/val files)
                                           ↓
                              python -m training.train
                                           ↓
                              checkpoints/ (model weights)
```

### Inference Flow
```
Input Image → mask_bg_remove.py → Get mask (face+neck+forehead)
                                              ↓
                           ShadowRemovalNet → Shadow-free image
                                              ↓
                           fix_light.py → Fix lighting + preserve texture
                                              ↓
                           Output: PNG with transparent BG (no shadows, good texture)
```

---

## Configuration

All settings in `configs/config.yaml`:

```yaml
# Dataset paths - EDIT THIS TO DEPLOY ANYWHERE
data:
  dataset_root: dataset         # or absolute path: /path/to/dataset
  input_subdir: input           # folder with input images
  target_subdir: target         # folder with target images
  splits_dir: dataset/splits

# Training
training:
  batch_size: 4
  epochs: 200
  device: auto                 # auto, cuda, cpu, mps

# Image size
model:
  input_size: 256
```

---

## Project Structure

```
Relighty/
├── configs/
│   └── config.yaml              # All configuration
├── data/
│   ├── input/                   # PNG with transparent background
│   ├── target/                  # PNG with transparent background
│   └── splits/                 # Train/val split files
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── models/
│   └── shadow_remover.py       # U-Net + ResNet34
├── masking_bg/
│   ├── mask_bg_remove.py      # Creates PNG with transparency
│   ├── face_neck_mask.py       # Face+Neck+Forehead mask
│   └── mediapipe_mask.py       # Face detection
├── training/
│   ├── train.py                # Training script
│   ├── dataset.py              # Uses PNG with alpha channel
│   └── losses.py               # L1 + SSIM loss
├── evaluation/
│   ├── evaluate.py             # Model evaluation
│   └── inference.py            # Inference (PNG output)
├── deployment/
│   └── pipeline.py            # End-to-end pipeline
└── postprocessing/
    └── fix_light.py           # Texture preservation + lighting fix
```

---

## Options for mask_bg_remove.py

```bash
# Face + Neck + Forehead (default)
python -m masking_bg.mask_bg_remove --input photo.jpg --output no_bg.png

# Face only (no neck, no forehead)
python -m masking_bg.mask_bg_remove --input photo.jpg --output no_bg.png --mask-mode face

# No forehead extension
python -m masking_bg.mask_bg_remove --input photo.jpg --output no_bg.png --no-forehead

# Sharper edges (no feathering)
python -m masking_bg.mask_bg_remove --input photo.jpg --output no_bg.png --no-feather
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Change dataset path | Edit `configs/config.yaml` - no code changes! |
| DataLoader issues on Windows | Set `num_workers: 0` in config |
| No GPU | Set `device: cpu` in config |

---

## License

[Your License]