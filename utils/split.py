#!/usr/bin/env python
"""
Split the dataset into train and validation sets.

Reads paired images from dataset/input/ and dataset/target/,
shuffles them with a fixed seed, then writes four .txt files
(one path per line) to dataset/splits/.

Run this ONCE before training:
    python utils/split.py

Options
───────
    --dataset   root of the dataset folder (default: dataset)
    --ratio     fraction used for training (default: 0.85)
    --seed      random seed for reproducibility (default: 42)
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    p = argparse.ArgumentParser(description="Dataset splitter")
    p.add_argument("--dataset", default="dataset",
                   help="Dataset folder with input/ and target/ subdirs")
    p.add_argument("--ratio",   type=float, default=0.85,
                   help="Training fraction (default 0.85)")
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()

    dataset_dir = ROOT / args.dataset
    input_dir   = dataset_dir / "input"
    target_dir  = dataset_dir / "target"
    out_dir     = ROOT / "dataset" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only keep files that exist in BOTH input/ and target/
    inp_names = {f.name for f in input_dir.iterdir()  if f.suffix.lower() in IMG_EXTS}
    tgt_names = {f.name for f in target_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    common = sorted(inp_names & tgt_names)

    if not common:
        print(
            f"ERROR: No matching files found.\n"
            f"  input/  : {len(inp_names)} images\n"
            f"  target/ : {len(tgt_names)} images\n"
            f"  Both dirs must contain files with identical names."
        )
        return

    random.seed(args.seed)
    random.shuffle(common)
    n_train  = int(len(common) * args.ratio)
    train    = common[:n_train]
    val      = common[n_train:]

    def _write(names, sub, filename):
        with open(out_dir / filename, "w") as f:
            for name in names:
                f.write(str(dataset_dir / sub / name) + "\n")

    _write(train, "input",  "train_input.txt")
    _write(train, "target", "train_target.txt")
    _write(val,   "input",  "val_input.txt")
    _write(val,   "target", "val_target.txt")

    print(f"Dataset : {dataset_dir}")
    print(f"Total   : {len(common)} matched pairs")
    print(f"Train   : {len(train)}  ({args.ratio*100:.0f}%)")
    print(f"Val     : {len(val)}    ({(1-args.ratio)*100:.0f}%)")
    print(f"Saved   : {out_dir}/")
    print()
    print("Next: python train/train.py")



if __name__ == "__main__":
    main()
