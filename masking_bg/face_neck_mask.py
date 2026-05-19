"""
Combined Face + Neck Mask Generation Module.

This module creates a unified mask that encompasses both the face and neck regions.
Used for training the shadow removal model to learn shadow removal on both areas.

The mask is generated in a single pass combining:
- Face mask from MediaPipe facial landmarks
- Neck mask extended from jawline
"""
import cv2
import numpy as np
from typing import Optional, Tuple

JAWLINE_INDICES = [
    132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361,
]

CHIN_INDICES = [152, 377, 400, 148, 176, 149, 150, 136, 172, 58]

FULL_JAW_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454
]


def create_face_neck_mask(
    image: np.ndarray,
    face_pts: Optional[np.ndarray],
    image_size: Optional[int] = None,
    extend_forehead: bool = True,
    forehead_extension: float = 0.05,
) -> np.ndarray:
    """
    Create a combined face+neck+forehead mask in a single pass.

    This is the main entry point that creates one unified mask
    containing face, neck, and forehead regions - all connected as one piece.

    Args:
        image: Input BGR image (H, W, 3)
        face_pts: MediaPipe 478-point facial landmarks (478, 2) or None
        image_size: If provided, resize mask to this size
        extend_forehead: Whether to extend mask to include forehead
        forehead_extension: How much to extend forehead (as fraction of face height)

    Returns:
        Combined face+neck+forehead mask as float32 array in [0, 1] range, shape (H, W)
    """
    h, w = image.shape[:2]

    if face_pts is None or len(face_pts) < 478:
        return _create_fallback_mask(h, w)

    face_mask = _create_face_region_mask(face_pts, h, w)
    neck_mask = _create_neck_region_mask(image, face_pts, h, w)

    combined = np.maximum(face_mask, neck_mask)

    if extend_forehead:
        forehead_mask = _create_forehead_mask(face_pts, h, w, forehead_extension)
        combined = np.maximum(combined, forehead_mask)

    if image_size is not None and image_size != w:
        combined = cv2.resize(combined, (image_size, image_size),
                               interpolation=cv2.INTER_LINEAR)

    combined = cv2.GaussianBlur(combined, (31, 31), 0)
    combined = np.clip(combined, 0.0, 1.0)

    return combined.astype(np.float32)


def _create_forehead_mask(
    face_pts: np.ndarray,
    h: int,
    w: int,
    extension: float = 0.05,
) -> np.ndarray:
    """
    Create forehead mask that connects to face seamlessly.

    Args:
        face_pts: MediaPipe facial landmarks
        h: Image height
        w: Image width
        extension: How much to extend above face (as fraction of face height)

    Returns:
        Forehead mask as float32 array
    """
    face_top = face_pts[:, 1].min()
    face_bottom = face_pts[:, 1].max()
    face_height = face_bottom - face_top
    face_width = face_pts[:, 0].max() - face_pts[:, 0].min()

    forehead_extend = int(face_height * extension)
    forehead_top = max(0, int(face_top) - forehead_extend)

    forehead_mask = np.zeros((h, w), dtype=np.float32)

    center_x = int(face_pts[:, 0].mean())
    half_width = int(face_width * 0.55)

    cv2.ellipse(
        forehead_mask,
        (center_x, forehead_top),
        (half_width, int(face_height * 0.25)),
        0, 0, 360, 1.0, -1
    )

    forehead_mask = cv2.GaussianBlur(forehead_mask, (21, 21), 0)

    return forehead_mask


def _create_face_region_mask(face_pts: np.ndarray, h: int, w: int) -> np.ndarray:
    """Create face region mask from landmarks."""
    mask = np.zeros((h, w), dtype=np.float32)

    def get_visibility(lm):
        v = getattr(lm, "visibility", None)
        return float(v) if v is not None else 1.0

    vis_indices = np.array([
        i for i in range(len(face_pts))
        if get_visibility(type('obj', (), {'visibility': 1.0})) or True
    ], dtype=np.int32)

    if len(face_pts) >= 4:
        pts = face_pts.astype(np.int32)

        face_top = pts[:, 1].min()
        face_bottom = pts[:, 1].max()
        face_height = face_bottom - face_top
        face_left = pts[:, 0].min()
        face_right = pts[:, 0].max()

        forehead_extend = int(face_height * 0.025)
        forehead_top = max(0, int(face_top) - forehead_extend)

        forehead_pts = np.array([
            [face_left, forehead_top],
            [face_right, forehead_top]
        ], dtype=np.int32)

        extended_pts = np.vstack([pts, forehead_pts]).astype(np.int32)
        hull = cv2.convexHull(extended_pts)
        cv2.fillConvexPoly(mask, hull, 1.0)

    close_k = max(5, int(h * 0.02) | 1)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ck)

    min_face_dim = min(w, h)
    dil_r = max(1, int(min_face_dim * 0.003))
    if dil_r > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_r * 2 + 1,) * 2)
        mask_u8 = cv2.dilate(mask_u8, dk, iterations=1)

    return mask_u8.astype(np.float32) / 255.0


def _create_neck_region_mask(
    image: np.ndarray,
    face_pts: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """Create neck region mask from facial landmarks."""
    jaw_pts = np.array([
        [int(face_pts[idx, 0]), int(face_pts[idx, 1])]
        for idx in FULL_JAW_INDICES
    ], dtype=np.int32)

    chin_pts = np.array([
        [int(face_pts[idx, 0]), int(face_pts[idx, 1])]
        for idx in CHIN_INDICES
    ], dtype=np.int32)

    face_width = np.linalg.norm(jaw_pts[0] - jaw_pts[-1])
    neck_center_x = int(np.mean(jaw_pts[:, 0]))
    neck_bottom_y = int(np.max(chin_pts[:, 1]))
    neck_top_y = int(np.min(chin_pts[:, 1]))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.uint8(np.absolute(sobelx))
    _, jaw_edges = cv2.threshold(abs_sobel, 25, 255, cv2.THRESH_BINARY)

    jaw_line_mask = np.zeros((h, w), dtype=np.uint8)
    jaw_top = neck_top_y - 20
    jaw_bottom = neck_top_y + 30
    jaw_polygon = np.array([
        [0, max(0, jaw_top)], [w, max(0, jaw_top)],
        [w, min(h, jaw_bottom)], [0, min(h, jaw_bottom)]
    ], dtype=np.int32)
    cv2.fillPoly(jaw_line_mask, [jaw_polygon], 255)
    jaw_edge_detected = cv2.bitwise_and(jaw_edges, jaw_line_mask)

    kernel_h = np.ones((1, 5), np.uint8)
    jaw_edge_dilated = cv2.dilate(jaw_edge_detected, kernel_h, iterations=2)

    shoulder_y = min(h - 1, neck_bottom_y + int(face_width * 2.5))
    shoulder_extend = int(face_width * 1.2)
    left_shoulder = [neck_center_x - shoulder_extend, shoulder_y]
    right_shoulder = [neck_center_x + shoulder_extend, shoulder_y]

    neck_polygon = np.vstack([jaw_pts, right_shoulder, left_shoulder])
    geo_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geo_mask, [neck_polygon.astype(np.int32)], 255)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_lower = np.array([0, 133, 77])
    ycrcb_upper = np.array([255, 173, 127])
    skin_mask = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)

    skin_with_jaw = cv2.bitwise_and(skin_mask, cv2.bitwise_not(jaw_edge_dilated))
    final_mask = cv2.bitwise_and(geo_mask, skin_with_jaw)

    eroded = cv2.erode(final_mask, np.ones((11, 11), np.uint8), iterations=1)
    if cv2.countNonZero(eroded) > 100:
        final_mask = eroded
    else:
        final_mask = cv2.erode(geo_mask, np.ones((15, 15), np.uint8), iterations=2)

    final_mask = _enforce_symmetry(final_mask, neck_center_x)
    final_mask = _suppress_clothing_boundary(final_mask, image, neck_bottom_y)

    kernel = np.ones((7, 7), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    if num_labels > 1:
        jaw_y = int(np.max(jaw_pts[:, 1]))
        largest_label = 1
        largest_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            y_pos = stats[i, cv2.CC_STAT_TOP]
            if area > largest_area and y_pos < jaw_y + 150:
                largest_area = area
                largest_label = i
        final_mask = (labels == largest_label).astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(final_mask, (101, 101), 0)
    return blurred.astype(np.float32) / 255.0


def _enforce_symmetry(mask: np.ndarray, center_x: int) -> np.ndarray:
    """Enforce horizontal symmetry on the mask."""
    h, w = mask.shape
    left_half = mask[:, :center_x]
    right_half = mask[:, center_x:]

    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w == 0:
        return mask

    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]

    right_half_flipped = cv2.flip(right_half, 1)
    left_half_flipped = cv2.flip(left_half, 1)
    left_pixels = cv2.countNonZero(left_half)
    right_pixels = cv2.countNonZero(right_half)

    if left_pixels > right_pixels * 1.5:
        merged = np.hstack([left_half, left_half_flipped])
    elif right_pixels > left_pixels * 1.5:
        merged = np.hstack([right_half_flipped, right_half])
    else:
        combined = np.bitwise_or(left_half, right_half_flipped)
        combined_right = np.bitwise_or(right_half, left_half_flipped)
        merged = np.hstack([combined, combined_right])

    if merged.shape[1] < w:
        pad = np.zeros((h, w - merged.shape[1]), dtype=merged.dtype)
        merged = np.hstack([merged, pad])
    return merged


def _suppress_clothing_boundary(mask: np.ndarray, image: np.ndarray, neck_bottom_y: int) -> np.ndarray:
    """Suppress the clothing/shoulder boundary in the mask."""
    h, w = mask.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if edges.shape != mask.shape:
        edges = cv2.resize(edges, (w, h))

    lower_region = np.zeros((h, w), dtype=np.uint8)
    boundary_y = neck_bottom_y + int(h * 0.15)
    lower_polygon = np.array([
        [0, boundary_y], [w, boundary_y], [w, h], [0, h]
    ], dtype=np.int32)
    cv2.fillPoly(lower_region, [lower_polygon], 255)
    edges_lower = cv2.bitwise_and(edges, lower_region)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges_lower, kernel, iterations=2)
    suppressed = cv2.bitwise_and(mask, cv2.bitwise_not(edges_dilated))
    return suppressed


def _create_fallback_mask(h: int, w: int) -> np.ndarray:
    """Create a centered ellipse as fallback when no face is detected."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, int(h * 0.42)
    cv2.ellipse(mask, (cx, cy),
                (int(w * 0.38), int(h * 0.48)), 0, 0, 360, 1.0, -1)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_u8 = cv2.morphologyEx(
        (mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, ck
    )
    ksize = max(45, int(min(h, w) * 0.20) | 1)
    mask = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0,
                             (ksize, ksize), 0)
    return np.clip(mask, 0.0, 1.0)