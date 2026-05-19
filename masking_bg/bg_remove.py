"""
Background removal utilities using rembg.

This module provides functions to remove background from images
using the rembg library. Used in conjunction with face+neck masks
to focus training on the relevant region.
"""
import numpy as np
from typing import Optional, Tuple
import cv2

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def remove_background(image: np.ndarray, alpha_matting: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background from image using rembg.

    Args:
        image: Input BGR image (H, W, 3)
        alpha_matting: Whether to use alpha matting for better edges

    Returns:
        Tuple of (image without background, alpha mask)
        - image: BGR image with transparent background (white fill)
        - mask: Binary mask where 1 = foreground (face+neck), 0 = background
    """
    if not REMBG_AVAILABLE:
        raise ImportError("rembg is not installed. Install with: pip install rembg")

    output = remove(image, alpha_matting=alpha_matting)

    if output.shape[2] == 4:
        alpha = output[:, :, 3]
        mask = (alpha > 128).astype(np.uint8)

        b, g, r = cv2.split(output[:, :, :3])
        bg_removed = cv2.merge([b, g, r])

        white_bg = np.ones_like(bg_removed) * 255
        fg_mask = mask[:, :, np.newaxis]
        result = bg_removed * fg_mask + white_bg * (1 - fg_mask)
        result = result.astype(np.uint8)

        return result, mask

    return image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)


def create_focused_image(
    image: np.ndarray,
    face_neck_mask: np.ndarray,
    fill_value: int = 255
) -> np.ndarray:
    """
    Create focused image with background filled.

    Args:
        image: Input BGR image
        face_neck_mask: Face+neck mask (0-1 range)
        fill_value: Value to fill background with (255 = white)

    Returns:
        Image with background filled
    """
    h, w = image.shape[:2]

    if face_neck_mask.shape != (h, w):
        face_neck_mask = cv2.resize(face_neck_mask, (w, h))

    mask_3d = face_neck_mask[:, :, np.newaxis]
    bg_fill = np.full_like(image, fill_value)

    focused = (image * mask_3d + bg_fill * (1 - mask_3d)).astype(np.uint8)

    return focused


def get_foreground_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Simple foreground detection using edge detection.

    Args:
        image: Input BGR image
        threshold: Edge detection threshold

    Returns:
        Binary mask of foreground
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, threshold, threshold * 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask


def prepare_training_image(
    image: np.ndarray,
    face_neck_mask: np.ndarray,
    use_rembg: bool = False,
) -> np.ndarray:
    """
    Prepare image for training by focusing on face+neck region.

    Args:
        image: Input BGR image
        face_neck_mask: Face+neck mask from create_face_neck_mask
        use_rembg: Whether to also use rembg for background removal

    Returns:
        Focused image with background handled
    """
    if use_rembg and REMBG_AVAILABLE:
        bg_removed, rembg_mask = remove_background(image)
        combined_mask = face_neck_mask * rembg_mask
        combined_mask = np.clip(combined_mask, 0, 1)
        return create_focused_image(bg_removed, combined_mask)

    return create_focused_image(image, face_neck_mask)