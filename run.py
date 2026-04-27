#!/usr/bin/env python
"""
Simple training entry point - reads everything from config.yaml
Usage: python run.py
"""

import sys
import os
import traceback

try:
    print("Step 1: Adding project root to path...", flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("Step 2: Importing YAML...", flush=True)
    import yaml

    print("Step 3: Importing training module...", flush=True)
    from train.scripts.train_shadow_removal import train

    print("Step 4: All imports successful!", flush=True)

except Exception as e:
    print(f"\nERROR during imports: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


def main():
    try:
        print("\nStep 5: Loading config...", flush=True)
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("Step 6: Extracting model name...", flush=True)
        model_name = config.get('model', {}).get('name', 'unet').lower()

        print(f"Step 7: Validating model '{model_name}'...", flush=True)
        if model_name not in ['unet', 'resunet++']:
            print(f"ERROR: Invalid model '{model_name}'")
            print("Valid options: 'unet' or 'resunet++'")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Training Model: {model_name.upper()}")
        print(f"Config: configs/config.yaml")
        print(f"{'='*60}\n", flush=True)

        print("Step 8: Starting training...", flush=True)
        train(model_name=model_name)
        print("Step 9: Training complete!", flush=True)

    except Exception as e:
        print(f"\nERROR during training: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
