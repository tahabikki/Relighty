"""
Masking_bg Module - Face+Neck Mask Generation & Background Removal

This module handles:
- FaceMaskGenerator: MediaPipe-based face mask generation
- create_face_neck_mask: Combined face+neck mask for training
- mask_bg_remove: Remove background using face+neck mask
- bg_remove: Background removal using rembg
- Combined mask for both face and neck regions (focus area)

The 'bg' in masking_bg means:
- MASK = face+neck region (the focus area for shadow removal)
- BG = background (to be removed)

Usage:
    from masking_bg import FaceMaskGenerator, create_face_neck_mask

    # For training - single mask containing both face and neck
    mask = create_face_neck_mask(image, landmarks, image_size=256)

    # For inference - face mask only
    mask_gen = FaceMaskGenerator(image_size=256)
    face_mask, landmarks = mask_gen.detect(image_rgb)

    # Remove background using mask
    from masking_bg.mask_bg_remove import apply_mask_background_removal
    result = apply_mask_background_removal(image, mask)
"""

from pathlib import Path

from .mediapipe_mask import FaceMaskGenerator
from .face_neck_mask import create_face_neck_mask

MASKING_BG_ROOT = Path(__file__).resolve().parent

__all__ = [
    "FaceMaskGenerator",
    "create_face_neck_mask",
    "FaceMaskGenerator",
    "create_face_neck_mask",
]