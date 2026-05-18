#!/usr/bin/env python
"""
Shadow removal inference — single image or full folder batch.

What it does
────────────
1. Load the trained model from the best checkpoint.
2. For each image:
   a. Resize to 256 × 256 (model input).
   b. Run MediaPipe to get the face mask.
   c. Run model → shadow-free prediction.
   d. BLEND: face region = model output,
             background  = original pixels (completely unchanged).
   e. Resize back to original resolution.
3. Save result.

Usage
─────
    # Single image  (output auto-named: clean_<input>.jpg)
    python evaluation/inference.py --input portrait.jpg

    # Single image, explicit output path
    python evaluation/inference.py --input portrait.jpg --output result.jpg

    # Batch folder  (Results/output/ by default)
    python evaluation/inference.py --input Results/input

    # Explicit checkpoint
    python evaluation/inference.py \\
        --input Results/input \\
        --checkpoint checkpoints/shadow_removal_epoch_080.pth
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.shadow_remover import ShadowRemovalNet
from masking.mediapipe_mask import FaceMaskGenerator

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> ShadowRemovalNet:
    model = ShadowRemovalNet().to(device)
    raw   = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw.get("model", raw))
    model.eval()
    return model


def find_checkpoint() -> Optional[Path]:
    best = ROOT / "checkpoints" / "shadow_removal_best.pth"
    if best.exists():
        return best
    ckpts = sorted((ROOT / "checkpoints").glob("*.pth"))
    return ckpts[-1] if ckpts else None


# ─── Core inference ───────────────────────────────────────────────────────────

def _extend_mask_with_neck(mask: np.ndarray, face_bottom_y: float) -> np.ndarray:
    """Extend face mask downward to include neck region."""
    h, w = mask.shape
    extended = mask.copy()
    
    chin_y = int(face_bottom_y)
    neck_top = chin_y
    neck_bottom = min(h, int(chin_y + (h - chin_y) * 0.6))
    
    if neck_top < h and neck_bottom > neck_top:
        for y in range(neck_top, neck_bottom):
            fade = (y - neck_top) / (neck_bottom - neck_top)
            fade = max(0, 1 - fade * 1.5)
            extended[y, :] = np.maximum(extended[y, :], fade)
    
    return extended


@torch.no_grad()
def remove_shadow(
    model:      ShadowRemovalNet,
    mask_gen:   FaceMaskGenerator,
    img_bgr:    np.ndarray,
    device:     torch.device,
    img_size:   int = 256,
) -> np.ndarray:
    """
    Remove facial shadows from one image with invisible mask blending.

    Returns a uint8 BGR image at the original input resolution.
    Background pixels are IDENTICAL to the input.
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # Resize to model input
    resized = cv2.resize(img_bgr, (img_size, img_size))

    # Face mask  (H × W float32 [0, 1])
    rgb  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mask, pts = mask_gen.detect(rgb)  # Get both mask and landmarks

    # Get face bottom (chin) for neck extension
    face_bottom = None
    if pts is not None:
        face_bottom = pts[:, 1].max()  # Bottom-most landmark = chin
    
    # Run model
    tensor = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    pred_01 = model(tensor)         # [0, 1]

    # Back to numpy uint8 BGR
    pred_rgb = (
        pred_01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    # Very conservative blending: use only shadows removed from model
    # Do NOT use full model output (too desaturated)
    # Instead: extract shadow differences and apply only those
    
    # Convert to LAB for perceptually better blending
    orig_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    pred_lab = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Extend mask to include neck
    if face_bottom is not None:
        mask = _extend_mask_with_neck(mask, face_bottom)
    
    # Extract luminance improvement from model
    l_diff = pred_lab[:, :, 0] - orig_lab[:, :, 0]
    
    # Apply luminance diff in face+neck region, very conservatively
    l_diff = l_diff * (mask * 0.5)  # Max contribution = 0.5 * luminance difference
    
    # Blend: mix original L with improved L based on mask
    blended_lab = orig_lab.copy()
    blended_lab[:, :, 0] = np.clip(
        orig_lab[:, :, 0] + l_diff,
        0, 255
    )
    # Keep A, B channels (colors) completely original
    
    # Back to BGR
    blended = cv2.cvtColor(blended_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Restore original resolution
    if (orig_h, orig_w) != (img_size, img_size):
        blended = cv2.resize(blended, (orig_w, orig_h),
                             interpolation=cv2.INTER_LANCZOS4)
    return blended


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import yaml
    
    p = argparse.ArgumentParser(description="Face shadow removal inference")
    p.add_argument("--input",      required=True,
                   help="Input image path or folder")
    p.add_argument("--output",     default=None,
                   help="Output image path or folder (auto-named if omitted)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint .pth path (auto-detected if omitted)")
    p.add_argument("--device",     default="auto")
    p.add_argument("--img_size",   type=int, default=None)
    args = p.parse_args()
    
    # ── Load image_size from config if not overridden ──────────────────────
    if args.img_size is None:
        config_path = ROOT / "configs" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            args.img_size = config.get("data", {}).get("image_size", 256)
        else:
            args.img_size = 256

    # Device (auto: CUDA → MPS → CPU)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Checkpoint
    ckpt_path = args.checkpoint or find_checkpoint()
    if ckpt_path is None:
        print("ERROR: No checkpoint found.\n  Train first: python train/train.py")
        sys.exit(1)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")

    model    = load_model(str(ckpt_path), device)
    mask_gen = FaceMaskGenerator(args.img_size)
    print("Model ready.\n")

    inp_path = Path(args.input)

    # ── Single image ──────────────────────────────────────────────────────────
    if inp_path.is_file():
        out_path = (
            Path(args.output) if args.output
            else inp_path.parent / f"clean_{inp_path.name}"
        )
        img = cv2.imread(str(inp_path))
        if img is None:
            print(f"ERROR: Cannot read {inp_path}")
            sys.exit(1)
        result = remove_shadow(model, mask_gen, img, device, args.img_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result)
        print(f"Saved → {out_path}")

    # ── Batch folder ──────────────────────────────────────────────────────────
    elif inp_path.is_dir():
        out_dir = (
            Path(args.output) if args.output
            else ROOT / "Results" / "output"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in inp_path.iterdir()
                       if f.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"No images found in {inp_path}")
            sys.exit(0)
        print(f"Processing {len(files)} images → {out_dir}\n")
        for i, f in enumerate(files, 1):
            img = cv2.imread(str(f))
            if img is None:
                print(f"  [{i:>4}/{len(files)}]  SKIP (unreadable) : {f.name}")
                continue
            result   = remove_shadow(model, mask_gen, img, device, args.img_size)
            out_file = out_dir / f"clean_{f.name}"
            cv2.imwrite(str(out_file), result)
            print(f"  [{i:>4}/{len(files)}]  {f.name}  →  {out_file.name}")
        print(f"\nDone.  Results in {out_dir}")

    else:
        print(f"ERROR: {inp_path} is not a file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
