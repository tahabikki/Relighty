"""
MediaPipe Face Landmark masking utility (Tasks API).
mediapipe >= 0.10.0  |  FaceLandmarker (478-point model)

Smart multi-layer mask strategy
────────────────────────────────
For every detected face:

  Layer 1 — Expanded face-oval polygon
    • 36 ordered landmarks trace the real silhouette
    • Polygon is scaled outward (8 %) from its centroid so wide jaws,
      broad temples and high foreheads are never clipped.

  Layer 2 — Visible-landmark hull
    • Every landmark with visibility > 0.25 is included.
    • convexHull of that set is filled at 90 % opacity.
    • Catches cheek / chin / forehead area that the oval may under-cover
      on rotated or asymmetric faces.

  Layer 3 — Morphological closing
    • Fills tiny concave gaps left by the polygon on angular faces
      (strong jaw, hollow temples, pronounced cheekbones).

  Layer 4 — Adaptive dilation
    • Kernel ∝ face bounding-box → proportional expansion regardless of
      image resolution or how close the subject is.

  Layer 5 — Adaptive Gaussian blur
    • Soft alpha edge ∝ face size → clean blending without hard cuts.

Fallback  — centred ellipse when no face is detected.
"""
import os
import sys
import urllib.request
import warnings
import logging
from pathlib import Path

# ── Silence Python-level noise ───────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.setdefault("GLOG_minloglevel",     "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU","1")


def _silence_stderr():
    """Redirect OS fd-2 to /dev/null — kills C++ GLOG / TF-Lite noise."""
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        fd      = os.open(os.devnull, os.O_WRONLY)
        saved   = os.dup(2)
        os.dup2(fd, 2); os.close(fd)
        try:    yield
        finally: os.dup2(saved, 2); os.close(saved)
    return _ctx()


with _silence_stderr():
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

_FaceLandmarker        = vision.FaceLandmarker
_FaceLandmarkerOptions = vision.FaceLandmarkerOptions

# MediaPipe 478-point model — all landmarks used for dynamic face shape
# (FACE_OVAL constant kept for reference but not used in smart_mask)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109,
]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


# ─────────────────────────────────────────────────────────────────────────────
class FaceMaskGenerator:
    """
    Precise, adaptive soft face mask — works for any face shape / orientation.
    """

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self._lm        = None
        self._init()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self):
        path = self._get_model()
        try:
            opts = _FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            with _silence_stderr():
                self._lm = _FaceLandmarker.create_from_options(opts)
        except Exception as e:
            raise RuntimeError(
                f"FaceLandmarker init failed: {e}\n"
                f"  model : {path}\n"
                f"  fix   : pip install -U mediapipe"
            ) from e

    def _get_model(self) -> Path:
        for p in [
            Path(__file__).parent / "face_landmarker.task",
            Path(__file__).parent.parent / "checkpoints" / "face_landmarker.task",
            Path("face_landmarker.task"),
        ]:
            if p.exists():
                return p
        dest = Path(__file__).parent / "face_landmarker.task"
        sys.stderr.write("  Downloading face_landmarker.task ...\n"); sys.stderr.flush()
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(dest))
            sys.stderr.write("  Download complete.\n"); sys.stderr.flush()
            return dest
        except Exception as e:
            raise RuntimeError(
                f"Download failed: {e}\n"
                f"  Manual: {_MODEL_URL}\n  Save to: {dest}"
            ) from e

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return soft float32 face mask (backward-compatible)."""
        mask, _ = self.detect(image_rgb)
        return mask

    def detect(self, image_rgb: np.ndarray):
        """
        Run detection once and return both the face mask and raw landmark pts.

        Returns:
            mask   : H × W  float32  [0=background, 1=face]
            pts_xy : (478, 2) float32 pixel coords, or None if no face found
        """
        h, w   = image_rgb.shape[:2]
        result = self._lm.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        )
        if not result.face_landmarks:
            return self._fallback(h, w), None

        lms    = result.face_landmarks[0]
        pts_xy = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        mask   = self._smart_mask(lms, h, w)
        return mask, pts_xy

    # ── Core smart mask ───────────────────────────────────────────────────────

    def _smart_mask(self, landmarks, h: int, w: int) -> np.ndarray:
        """Adaptive mask that follows each unique face shape using all landmarks."""

        # Pixel coordinates for every landmark
        pts_f = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks],
            dtype=np.float32,
        )
        
        # Find face bounds
        face_top = pts_f[:, 1].min()
        face_bottom = pts_f[:, 1].max()
        face_left = pts_f[:, 0].min()
        face_right = pts_f[:, 0].max()
        face_height = face_bottom - face_top
        face_width = face_right - face_left

        mask = np.zeros((h, w), dtype=np.float32)

        # ── Layer 1: Convex hull of ALL visible landmarks ─────────────────────
        # Each face shape is unique - let landmarks define the boundary naturally
        def _get_visibility(lm):
            """Safely get visibility, default to 1.0 if None or missing."""
            v = getattr(lm, "visibility", None)
            return float(v) if v is not None else 1.0
        
        vis_indices = np.array(
            [i for i, lm in enumerate(landmarks) 
             if _get_visibility(lm) > 0.1],
            dtype=np.int32
        )
        
        if len(vis_indices) >= 4:
            vis_pts = pts_f[vis_indices].astype(np.int32)
            
            # ── Intelligent forehead extension ─────────────────────────────────
            # Use ALL visible landmarks for full face coverage (keeps all face regions)
            # Then extend forehead minimally (only 2-3% above top) to include forehead
            # without touching hair
            top_y = vis_pts[:, 1].min()
            left_x = vis_pts[:, 0].min()
            right_x = vis_pts[:, 0].max()
            
            # Only extend 2-3% above top landmarks for safe forehead inclusion
            forehead_extend = int(face_height * 0.025)
            forehead_top = max(0, int(top_y) - forehead_extend)
            
            # Add minimal forehead extension points
            forehead_pts = np.array([
                [left_x, forehead_top],
                [right_x, forehead_top]
            ], dtype=np.int32)
            
            # Combine all visible landmarks with minimal forehead extension
            extended_pts = np.vstack([vis_pts, forehead_pts]).astype(np.int32)
            
            hull = cv2.convexHull(extended_pts)
            cv2.fillConvexPoly(mask, hull, 1.0)
        else:
            # Fallback if not enough landmarks
            return self._fallback(h, w)

        # ── Layer 2 : morphological closing ───────────────────────────────────
        # Fill small gaps in the hull
        close_k = max(5, int(face_height * 0.02) | 1)
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ck)

        # ── Layer 3 : minimal dilation ─────────────────────────────────────────
        # Only 1-2 pixels for very slight smoothing, not expansion
        dil_r = max(1, int(min(face_width, face_height) * 0.003))
        if dil_r > 0:
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_r * 2 + 1,) * 2)
            mask_u8 = cv2.dilate(mask_u8, dk, iterations=1)
        
        mask = mask_u8.astype(np.float32) / 255.0

        # ── Layer 4 : adaptive Gaussian blur for smooth edges ──────────────────
        # Large blur for invisible boundaries
        blur_r = max(35, int(min(face_width, face_height) * 0.15) | 1)
        mask = cv2.GaussianBlur(mask, (blur_r, blur_r), 0)

        return np.clip(mask, 0.0, 1.0)

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(self, h: int, w: int) -> np.ndarray:
        """Centred ellipse when no face is detected."""
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, int(h * 0.42)
        cv2.ellipse(mask, (cx, cy),
                    (int(w * 0.38), int(h * 0.48)), 0, 0, 360, 1.0, -1)
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_u8 = cv2.morphologyEx(
            (mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, ck
        )
        # Very aggressive blur for smooth edge
        ksize = max(45, int(min(h, w) * 0.20) | 1)
        mask  = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0,
                                 (ksize, ksize), 0)
        return np.clip(mask, 0.0, 1.0)

    def __del__(self):
        try:
            if self._lm is not None:
                self._lm.close()
        except Exception:
            pass
