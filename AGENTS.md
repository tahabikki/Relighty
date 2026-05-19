# Relighty - Face and Neck Shadow Removal System

A modular computer vision pipeline for removing shadows from face and neck regions using deep learning with MediaPipe facial landmark detection.

## Project Structure

```
Relighty/
├── configs/                 # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Dataset directory
│   ├── input/              # PNG with transparent background
│   ├── target/             # PNG with transparent background
│   └── splits/            # Train/validation split files
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── models/                 # Neural network architectures
│   └── shadow_remover.py  # U-Net with ResNet-34 encoder
├── masking_bg/            # Mask generation & background removal
│   ├── mask_bg_remove.py # PNG with transparent background
│   ├── face_neck_mask.py # Combined face+neck+forehead mask
│   └── mediapipe_mask.py # FaceMaskGenerator
├── preprocessing/         # Data augmentation
│   └── augment.py        # Augmentation utilities
├── training/              # Model training
│   ├── train.py          # Training script
│   ├── dataset.py        # Dataset class (uses PNG with alpha)
│   └── losses.py         # Loss functions
├── evaluation/            # Model evaluation & inference
│   ├── evaluate.py        # Evaluation metrics
│   └── inference.py       # Inference (PNG output)
├── postprocessing/       # Post-processing utilities
│   ├── fix_light.py      # Shadow removal with texture preservation
│   └── bg_remove.py      # Background removal
├── deployment/           # End-to-end pipeline
│   └── pipeline.py       # Complete pipeline
└── utils/                 # Utility scripts
    └── split.py          # Dataset splitting
```

## Usage

### Training

```bash
# Step 1: Prepare dataset with transparent background
python -m masking_bg.mask_bg_remove --input-folder dataset/input --output-folder dataset/input
python -m masking_bg.mask_bg_remove --input-folder dataset/target --output-folder dataset/target

# Step 2: Create train/val split
python -m utils.split

# Step 3: Train the model
python -m training.train

# Step 4: Resume training
python -m training.train --resume
```

### Inference

```bash
# Single image (outputs PNG with transparent background)
python -m evaluation.inference --input photo.jpg --output result.png

# Batch folder
python -m evaluation.inference --input Results/input --output Results/output

# Use deployment pipeline
python -m deployment.pipeline --input photo.jpg --output result.png
```

### Evaluation

```bash
# Evaluate on validation set
python -m evaluation.evaluate

# Use specific checkpoint
python -m evaluation.evaluate --checkpoint checkpoints/shadow_removal_best.pth
```

## Configuration

All paths are configured in `configs/config.yaml`. To deploy in any environment, simply edit this file.

### Dataset Path (Key for Deployment)

```yaml
data:
  # EDIT THIS: Point to your dataset location
  dataset_root: dataset  # relative path

  # OR use absolute path (works anywhere):
  # dataset_root: /path/to/your/dataset

  input_subdir: input      # folder with shadow images (PNG with transparency)
  target_subdir: target    # folder with clean images (PNG with transparency)
  splits_dir: dataset/splits
```

## Key Features

1. **Transparent Background**: PNG with alpha channel (using mask_bg_remove.py)
2. **Face+Neck+Forehead Mask**: All connected as one piece
3. **Texture Preservation**: Keeps original skin texture while removing shadows
4. **Adaptive Blending**: Dynamic strength based on shadow severity
5. **Cross-platform**: Works on CUDA, MPS, and CPU

## Dependencies

- torch >= 2.0
- mediapipe >= 0.10.0
- opencv-python
- numpy
- pyyaml
- tqdm
- rembg (optional)

Install with: `pip install -r requirements.txt`

## Pipeline Flow

### Training:
```
mask_bg_remove.py → PNG with transparency → train/val split → training → model weights
```

### Inference:
```
mask_bg_remove (get mask) → Model prediction → fix_light → Output PNG
```

All using the same mask_bg_remove approach for consistency!