"""
Shadow Application Script - Realistic Shadows

Add natural-looking shadows to face and neck regions.
Creates shadows that look like real lighting conditions.

Usage:
    # Single image with realistic shadow
    python -m shadow_apply.apply_shadow --input photo.jpg --output shadow.jpg

    # Batch with natural variation
    python -m shadow_apply.apply_shadow --input-folder dataset/target --output-folder dataset_shaded

    # Custom settings
    python -m shadow_apply.apply_shadow --input photo.jpg --output shadow.jpg --intensity 0.6 --side left
"""
import argparse
import sys
from pathlib import Path
import random

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from masking_bg.mediapipe_mask import FaceMaskGenerator
from masking_bg.face_neck_mask import create_face_neck_mask


def create_realistic_face_shadow(
    image: np.ndarray,
    mask: np.ndarray,
    side: str = "auto",
    intensity: float = 0.5,
) -> np.ndarray:
    """
    Create realistic shadow on face that looks like natural lighting.

    Args:
        image: Input BGR image
        mask: Face+neck mask
        side: "left", "right", "auto" (random)
        intensity: Shadow strength (0.0-1.0)

    Returns:
        Image with realistic shadow
    """
    h, w = image.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    if side == "auto":
        side = random.choice(["left", "right"])

    mask_f = mask.copy()

    if side == "left":
        x_gradient = np.linspace(1.0, 1.0 - intensity, w)
    else:
        x_gradient = np.linspace(1.0 - intensity, 1.0, w)

    x_gradient = np.tile(x_gradient, (h, 1))

    y_gradient = np.linspace(1.0, 1.0 - (intensity * 0.3), h)
    y_gradient = y_gradient[:, np.newaxis]
    y_gradient = np.tile(y_gradient, (1, w))

    gradient = x_gradient * y_gradient

    shadow_mask = mask_f * gradient

    shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)

    shadow_mask = np.clip(shadow_mask, 0, 1)

    mask_3d = shadow_mask[:, :, np.newaxis]

    image_f = image.astype(np.float32)
    shadowed = image_f * mask_3d

    result = np.clip(shadowed, 0, 255).astype(np.uint8)

    return result


def create_natural_lighting_shadow(
    image: np.ndarray,
    mask: np.ndarray,
    light_angle: float = 0.0,
    intensity: float = 0.5,
    softness: float = 0.5,
) -> np.ndarray:
    """
    Create shadow based on lighting angle (in radians).

    0.0 = light from right
    0.5 = light from top
    1.0 = light from left

    Args:
        image: Input BGR image
        mask: Face+neck mask
        light_angle: Light direction (0.0-1.0)
        intensity: Shadow strength
        softness: Shadow edge softness

    Returns:
        Image with natural shadow
    """
    h, w = image.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    center_x = w // 2

    y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')

    shift = (light_angle - 0.5) * w
    x_shifted = x_coords + shift

    x_norm = (x_shifted - center_x) / center_x
    x_norm = np.clip(x_norm, -1, 1)

    shadow_strength = (-x_norm * intensity + 0.5) * 2
    shadow_strength = np.clip(shadow_strength, 1.0 - intensity, 1.0)

    final_mask = mask * shadow_strength

    blur_size = int(15 + softness * 25)
    if blur_size % 2 == 0:
        blur_size += 1
    final_mask = cv2.GaussianBlur(final_mask, (blur_size, blur_size), 0)

    final_mask = np.clip(final_mask, 0, 1)
    mask_3d = final_mask[:, :, np.newaxis]

    image_f = image.astype(np.float32)
    shadowed = image_f * mask_3d

    return np.clip(shadowed, 0, 255).astype(np.uint8)


def create_ambient_shadow(
    image: np.ndarray,
    mask: np.ndarray,
    shadow_pct: float = 0.4,
) -> np.ndarray:
    """
    Create ambient/diffuse shadow (like in overcast lighting).

    Args:
        image: Input BGR image
        mask: Face+neck mask
        shadow_pct: Percentage of shadow to apply

    Returns:
        Image with soft ambient shadow
    """
    h, w = image.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    shadow_value = 1.0 - shadow_pct

    noisy_mask = mask + np.random.normal(0, 0.05, (h, w))
    noisy_mask = np.clip(noisy_mask, 0, 1)

    noisy_mask = cv2.GaussianBlur(noisy_mask, (41, 41), 0)

    final_mask = noisy_mask * shadow_value + (1 - noisy_mask)
    final_mask = np.clip(final_mask, 0, 1)

    mask_3d = final_mask[:, :, np.newaxis]
    image_f = image.astype(np.float32)

    return (image_f * mask_3d).clip(0, 255).astype(np.uint8)


def process_single_image(
    input_path: Path,
    output_path: Path,
    shadow_type: str = "realistic",
    side: str = "auto",
    intensity: float = 0.5,
    light_angle: float = 0.25,
    extend_forehead: bool = True,
) -> bool:
    """Process a single image."""
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Cannot read {input_path}")
        return False

    mask_gen = FaceMaskGenerator(256)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mask, face_pts = mask_gen.detect(rgb)

    if face_pts is not None:
        full_mask = create_face_neck_mask(
            image, face_pts, None,
            extend_forehead=extend_forehead,
            forehead_extension=0.08
        )
    else:
        full_mask = face_mask

    if shadow_type == "realistic":
        result = create_realistic_face_shadow(image, full_mask, side, intensity)
    elif shadow_type == "lighting":
        result = create_natural_lighting_shadow(image, full_mask, light_angle, intensity)
    elif shadow_type == "ambient":
        result = create_ambient_shadow(image, full_mask, intensity)
    else:
        result = image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), result)

    if success:
        print(f"Saved: {output_path}")
    return success


def process_folder(
    input_folder: Path,
    output_folder: Path,
    shadow_type: str = "realistic",
    side: str = "auto",
    intensity: float = 0.5,
    light_angle: float = 0.25,
    extend_forehead: bool = True,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> int:
    """Process all images in a folder."""
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
        output_path = output_folder / f.name
        
        s = side if side != "auto" else random.choice(["left", "right"])
        
        if process_single_image(f, output_path, shadow_type, s, intensity, light_angle, extend_forehead):
            success_count += 1

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(files)}")

    print(f"\nDone: {success_count}/{len(files)} images processed")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Apply realistic shadows to face and neck regions. "
                   "Creates natural-looking shadows like real lighting."
    )

    parser.add_argument("--input", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument("--input-folder", type=Path, help="Input folder path")
    parser.add_argument("--output-folder", type=Path, help="Output folder path")

    parser.add_argument("--type", type=str, default="realistic",
                        choices=["realistic", "lighting", "ambient"],
                        help="Shadow type: realistic, lighting, or ambient")

    parser.add_argument("--side", type=str, default="auto",
                        choices=["auto", "left", "right"],
                        help="Shadow side (auto = random)")

    parser.add_argument("--intensity", type=float, default=0.5,
                        help="Shadow intensity (0.0-1.0), default: 0.5")

    parser.add_argument("--angle", type=float, default=0.25,
                        help="Light angle for 'lighting' type (0.0-1.0)")

    parser.add_argument("--no-forehead", action="store_true",
                        help="Disable forehead extension")

    args = parser.parse_args()

    extend_forehead = not args.no_forehead

    if args.input and args.output:
        process_single_image(
            args.input, args.output,
            args.type, args.side, args.intensity, args.angle, extend_forehead
        )
    elif args.input_folder and args.output_folder:
        process_folder(
            args.input_folder, args.output_folder,
            args.type, args.side, args.intensity, args.angle, extend_forehead
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Realistic shadow (default)")
        print("  python -m shadow_apply.apply_shadow --input photo.jpg --output shadow.jpg")
        print()
        print("  # From left side")
        print("  python -m shadow_apply.apply_shadow --input photo.jpg --output shadow.jpg --side left --intensity 0.6")
        print()
        print("  # Lighting angle")
        print("  python -m shadow_apply.apply_shadow --input photo.jpg --output shadow.jpg --type lighting --angle 0.1")
        print()
        print("  # Ambient/diffuse shadow")
        print("  python -m shadow_apply.apply_shadow --input photo.jpg --output shadow.jpg --type ambient")


if __name__ == "__main__":
    main()