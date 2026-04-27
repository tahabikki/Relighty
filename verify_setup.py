#!/usr/bin/env python
"""Verify shadow removal training setup."""

import sys
import os

def check_files():
    """Verify all necessary files exist."""
    files_to_check = [
        'models_Relighty/shadow_remover.py',
        'utils/shadow_loss.py',
        'train/scripts/train_shadow_removal.py',
        'evaluation/remove_shadows_portrait.py',
        'configs/config.yaml',
        'split/splited_dataset_data/train_input.txt',
        'split/splited_dataset_data/val_input.txt',
        'SHADOW_REMOVAL_GUIDE.md',
    ]

    print("=" * 60)
    print("SHADOW REMOVAL - SETUP VERIFICATION")
    print("=" * 60)

    all_good = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print("[OK] " + file_path)
            if size_mb > 0.1:
                print("     Size: {:.1f} MB".format(size_mb))
        else:
            print("[MISSING] " + file_path)
            all_good = False

    print("\n" + "=" * 60)

    if all_good:
        print("[OK] Setup is ready!")
        print("\nNext steps:")
        print("1. Read the guide:")
        print("   type SHADOW_REMOVAL_GUIDE.md")
        print("\n2. Start training:")
        print("   python train/scripts/train_shadow_removal.py")
        print("\n3. After training, remove shadows from portraits:")
        print("   python evaluation/remove_shadows_portrait.py --input portrait.jpg")
    else:
        print("[ERROR] Some files are missing. Check above.")
        return False

    print("=" * 60)

    return all_good


def check_imports():
    """Verify Python imports work."""
    print("\nChecking Python imports...")

    imports = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('mediapipe', 'MediaPipe'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]

    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print("[OK] " + name)
        except ImportError:
            print("[MISSING] " + name)
            all_ok = False

    if not all_ok:
        print("\nInstall missing dependencies:")
        print("  pip install torch torchvision opencv-python numpy pillow mediapipe PyYAML tqdm")

    return all_ok


if __name__ == "__main__":
    files_ok = check_files()
    imports_ok = check_imports()

    if files_ok and imports_ok:
        print("\n[OK] Everything is ready! Run:")
        print("  python train/scripts/train_shadow_removal.py")
        sys.exit(0)
    else:
        print("\n[ERROR] Please fix the issues above.")
        sys.exit(1)
