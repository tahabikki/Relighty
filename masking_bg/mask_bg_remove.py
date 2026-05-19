"""
Mask-based Background Removal - Fixed version

This script uses the face+neck mask to define what is the object (foreground)
and removes everything else as background. Outputs PNG with transparent background.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from masking_bg.mediapipe_mask import FaceMaskGenerator
from masking_bg.face_neck_mask import create_face_neck_mask


def create_combined_face_neck_mask(
    image: np.ndarray,
    image_size: int = 256,
    extend_forehead: bool = True,
    forehead_extension: float = 0.08,
) -> np.ndarray:
    """
    Create a combined face+neck+forehead mask that's one piece (connected).
    """
    h, w = image.shape[:2]

    mask_gen = FaceMaskGenerator(image_size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mask, face_pts = mask_gen.detect(rgb)

    if face_pts is not None:
        full_mask = create_face_neck_mask(
            image, face_pts, None,
            extend_forehead=extend_forehead,
            forehead_extension=forehead_extension
        )
        return full_mask
    else:
        print("Warning: No face detected, using face-only mask")
        return face_mask


def apply_transparent_background(
    image: np.ndarray,
    mask: np.ndarray,
    feather_edges: bool = True,
    feather_radius: int = 21,
) -> np.ndarray:
    """
    Create PNG with transparent background.
    """
    h, w = image.shape[:2]

    # Ensure mask is the right size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    # Ensure mask is in [0, 1] range
    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    # Apply feathering
    if feather_edges:
        mask = cv2.GaussianBlur(mask, (feather_radius, feather_radius), 0)

    # Ensure valid range
    mask = np.clip(mask, 0.0, 1.0)

    # Create BGRA image
    b, g, r = cv2.split(image)
    
    # Alpha channel - where mask is 1, alpha is 255 (opaque), where mask is 0, alpha is 0 (transparent)
    alpha = (mask * 255).astype(np.uint8)
    
    # Ensure all channels are uint8
    b = b.astype(np.uint8)
    g = g.astype(np.uint8)
    r = r.astype(np.uint8)
    
    # Merge to create BGRA
    result = cv2.merge([b, g, r, alpha])

    return result


def process_single_image(
    input_path: Path,
    output_path: Path,
    mask_mode: str = "face_neck",
    feather: bool = True,
    extend_forehead: bool = True,
) -> bool:
    """
    Process a single image to remove background (transparent PNG).
    """
    print(f"Processing: {input_path}")
    
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Cannot read {input_path}")
        return False

    print(f"  Image shape: {image.shape}")

    if mask_mode == "face":
        mask_gen = FaceMaskGenerator(256)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask, _ = mask_gen.detect(rgb)
    else:
        mask = create_combined_face_neck_mask(
            image,
            image_size=256,
            extend_forehead=extend_forehead,
        )

    print(f"  Mask min/max: {mask.min():.3f} / {mask.max():.3f}")
    print(f"  Mask non-zero: {np.count_nonzero(mask)}")

    result = apply_transparent_background(
        image, mask,
        feather_edges=feather,
    )

    print(f"  Result shape: {result.shape}")
    print(f"  Alpha min/max: {result[:,:,3].min()} / {result[:,:,3].max()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != '.png':
        output_path = output_path.with_suffix('.png')

    # Save as PNG with alpha channel
    success = cv2.imwrite(str(output_path), result)

    if success:
        print(f"  Saved: {output_path}")
    else:
        print(f"  Error: Cannot write {output_path}")

    return success


def process_folder(
    input_folder: Path,
    output_folder: Path,
    mask_mode: str = "face_neck",
    feather: bool = True,
    extend_forehead: bool = True,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> int:
    """
    Process all images in a folder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])

    if not files:
        print(f"No images found in {input_folder}")
        return 0

    print(f"Processing {len(files)} images...")

    success_count = 0
    for i, f in enumerate(files, 1):
        output_path = output_folder / f"{f.stem}.png"
        if process_single_image(f, output_path, mask_mode, feather, extend_forehead):
            success_count += 1
        else:
            print(f"  Failed: {f.name}")

    print(f"\nDone: {success_count}/{len(files)} images processed")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Remove background using face+neck mask. "
                   "Outputs PNG with transparent background."
    )

    parser.add_argument("--input", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--input-folder", type=Path, help="Input folder path")
    parser.add_argument("--output-folder", type=Path, help="Output folder path")
    parser.add_argument("--mask-mode", type=str, default="face_neck",
                        choices=["face", "face_neck"])
    parser.add_argument("--no-feather", action="store_true", help="Disable edge feathering")
    parser.add_argument("--no-forehead", action="store_true", help="Disable forehead extension")

    args = parser.parse_args()

    mask_mode = args.mask_mode
    feather = not args.no_feather
    extend_forehead = not args.no_forehead

    if args.input and args.output:
        process_single_image(args.input, args.output, mask_mode, feather, extend_forehead)
    elif args.input_folder and args.output_folder:
        process_folder(args.input_folder, args.output_folder, mask_mode, feather, extend_forehead)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()