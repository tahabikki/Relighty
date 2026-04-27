import os

config = {
    'model': {
        'name': 'ShadowRemovalUNet',
        'input_size': 256,
        'channels': 3
    },
    'training': {
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 0.0001,
        'optimizer': 'AdamW',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'device': 'cuda'
    },
    'loss': {
        'l1_weight': 1.5,
        'ssim_weight': 0.8,
        'perceptual_weight': 0.3,
        'edge_weight': 0.2
    },
    'data': {
        'input_dir': 'dataset/input',
        'target_dir': 'dataset/target',
        'split_dir': 'split/splited_dataset_data',
        'output_size': 256
    },
    'checkpoint': {
        'dir': 'checkpoints',
        'save_every': 5
    }
}

import yaml
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("config.yaml updated for dataset folder")

# Also update split files
with open('split/splited_dataset_data/train_input.txt', 'w') as f:
    for i in range(1, 1021):
        f.write(f'dataset/input/photo{i}.jpg\n')

with open('split/splited_dataset_data/train_target.txt', 'w') as f:
    for i in range(1, 1021):
        f.write(f'dataset/target/photo{i}.jpg\n')

with open('split/splited_dataset_data/val_input.txt', 'w') as f:
    for i in range(1021, 1276):
        f.write(f'dataset/input/photo{i}.jpg\n')

with open('split/splited_dataset_data/val_target.txt', 'w') as f:
    for i in range(1021, 1276):
        f.write(f'dataset/target/photo{i}.jpg\n')

print("Split files updated: Train 1020, Val 255")