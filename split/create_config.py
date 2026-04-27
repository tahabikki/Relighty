import os
import yaml

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
        'optimizer': 'Adam',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'device': 'cuda'
    },
    'dataloader': {
        'num_workers': 0
    },
    'loss': {
        'l1_weight': 1.0,
        'ssim_weight': 0.5,
        'perceptual_weight': 0.1
    },
    'data': {
        'input_dir': 'dataset_no_bg/input',
        'target_dir': 'dataset_no_bg/target',
        'split_dir': 'split/splited_dataset_data',
        'output_size': 256,
        'augmentation': {
            'brightness_prob': 0.3,
            'gamma_prob': 0.3,
            'shadow_prob': 0.3
        }
    },
    'checkpoint': {
        'dir': 'checkpoints',
        'save_every': 5
    }
}

with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("config.yaml created")