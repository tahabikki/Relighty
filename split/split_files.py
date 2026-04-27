import os
import random
import argparse

parser = argparse.ArgumentParser(description="Split dataset into train/validation sets")
parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory path (should contain input/ and target/ subdirectories)")
args = parser.parse_args()

INPUT_DIR = os.path.join(args.dataset, "input")
TARGET_DIR = os.path.join(args.dataset, "target")
SPLIT_DIR = "split/splited_dataset_data"
TRAIN_RATIO = 0.8

random.seed(42)
os.makedirs(SPLIT_DIR, exist_ok=True)

input_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
target_files = sorted([f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

common_files = sorted(set(input_files) & set(target_files))
print(f"Total matching files: {len(common_files)}")

random.shuffle(common_files)
split_idx = int(len(common_files) * TRAIN_RATIO)
train_files = common_files[:split_idx]
val_files = common_files[split_idx:]

with open(os.path.join(SPLIT_DIR, "train_input.txt"), 'w') as f:
    for filename in train_files:
        f.write(os.path.join(INPUT_DIR, filename) + '\n')

with open(os.path.join(SPLIT_DIR, "train_target.txt"), 'w') as f:
    for filename in train_files:
        f.write(os.path.join(TARGET_DIR, filename) + '\n')

with open(os.path.join(SPLIT_DIR, "val_input.txt"), 'w') as f:
    for filename in val_files:
        f.write(os.path.join(INPUT_DIR, filename) + '\n')

with open(os.path.join(SPLIT_DIR, "val_target.txt"), 'w') as f:
    for filename in val_files:
        f.write(os.path.join(TARGET_DIR, filename) + '\n')

print(f"Train: {len(train_files)}, Val: {len(val_files)}")
print(f"Saved to {SPLIT_DIR}/")

# Update config
import yaml
config = {
    'model': {'name': 'ShadowRemovalUNet', 'input_size': 256, 'channels': 3},
    'training': {'batch_size': 8, 'epochs': 100, 'learning_rate': 0.0001, 'optimizer': 'AdamW', 'scheduler_patience': 10, 'scheduler_factor': 0.5, 'device': 'cuda'},
    'loss': {'l1_weight': 1.5, 'ssim_weight': 0.8, 'perceptual_weight': 0.3, 'edge_weight': 0.2},
    'data': {'input_dir': 'dataset_no_bg/input', 'target_dir': 'dataset_no_bg/target', 'split_dir': 'split/splited_dataset_data', 'output_size': 256},
    'checkpoint': {'dir': 'checkpoints', 'save_every': 5}
}
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print("config.yaml updated")