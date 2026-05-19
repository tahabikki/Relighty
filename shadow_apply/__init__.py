"""
Shadow Apply Module - Add Realistic Shadows to Face Images

This module adds realistic, natural-looking shadows to face and neck regions.

Usage:
    from shadow_apply import create_realistic_face_shadow

    shadowed = create_realistic_face_shadow(image, mask, side="left", intensity=0.5)
"""

from .apply_shadow import (
    create_realistic_face_shadow,
    create_natural_lighting_shadow,
    create_ambient_shadow,
    process_single_image,
    process_folder,
)

__all__ = [
    "create_realistic_face_shadow",
    "create_natural_lighting_shadow",
    "create_ambient_shadow",
    "process_single_image",
    "process_folder",
]