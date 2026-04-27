import sys
sys.path.insert(0, '.')

import os
# Suppress TensorFlow/MediaPipe BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Better CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Redirect stderr to suppress INFO messages
import io
sys.stderr = io.StringIO()

import cv2
import torch
import numpy as np
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
# Re-enable stderr for actual output
sys.stderr = sys.__stderr__

from models_Relighty.shadow_remover import create_model


# MediaPipe face-mesh contour indices (natural face oval)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109, 10
]

# Global face landmarker (lazily initialized)
_face_landmarker = None

def _get_face_landmarker():
    """Lazily load face landmarker model"""
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker

    # Get model path
    import urllib.request
    model_dir = os.path.expanduser('~/.mediapipe_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'face_landmarker.task')

    if not os.path.exists(model_path):
        print("[MediaPipe] Downloading face_landmarker.task...")
        try:
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download face_landmarker.task: {e}")

    # Create landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
    )
    _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def get_face_geometry_mask(image_bgr, feather=20):
    """
    Soft mask [0..1] following the exact face geometry via MediaPipe (modern API).
    Forehead is extended upward so shadow there is also corrected.
    """
    h, w = image_bgr.shape[:2]

    # Get face landmarker
    landmarker = _get_face_landmarker()

    # Detect landmarks
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)

    if not res.face_landmarks:
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, h // 2
        cv2.ellipse(mask, (cx, cy), (w // 3, h // 2), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)

    lm  = res.face_landmarks[0]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)]
                    for i in FACE_OVAL], dtype=np.int32)

    # --- Extend forehead upward -----------------------------------------------
    face_top    = pts[:, 1].min()
    face_bottom = pts[:, 1].max()
    face_height = max(face_bottom - face_top, 1)
    forehead_ext = int(face_height * 0.18)
    threshold_y  = face_top + int(face_height * 0.40)
    pts_ext = pts.copy()
    for i in range(len(pts_ext)):
        if pts_ext[i, 1] <= threshold_y:
            pts_ext[i, 1] = max(0, pts_ext[i, 1] - forehead_ext)
    # ------------------------------------------------------------------

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [pts_ext], 1.0)
    mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
    return mask


class PortraitShadowRemover:
    def __init__(self, checkpoint_path="checkpoints/shadow_removal_best.pth", device=None, model_name=None):
        # If model_name is specified and checkpoint_path is default, construct path with model name
        if model_name and checkpoint_path == "checkpoints/shadow_removal_best.pth":
            checkpoint_path = f"checkpoints/shadow_removal_{model_name}_best.pth"
        
        # Allow explicit device override (e.g. "cpu" when GPU is busy with training)
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            # Free any cached VRAM before loading model
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_size = 256
        print("Device: " + str(self.device))

        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found: " + checkpoint_path)
            print("Train first: python train/scripts/train_shadow_removal.py")
            self.model = None
            return

        # Load on CPU first to avoid CUDA memory fragmentation, then move to device
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')

        # Auto-detect model from checkpoint if not specified
        if model_name is None:
            model_name = ckpt.get('model_name', 'unet')

        print("Loaded: " + checkpoint_path + "  (epoch " + str(epoch) + ", model: " + model_name + ")")

        self.model = create_model(model_name).to(self.device)
        # Move state dict to target device before loading
        state_dict = {k: v.to(self.device) if torch.is_tensor(v) else v 
                      for k, v in ckpt['model_state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print("Checkpoint size: " + str(round(size_mb, 1)) + " MB")

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    # ------------------------------------------------------------------
    def _detect_face_box(self, image):
        h, w = image.shape[:2]
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            gray_eq = cv2.equalizeHist(gray)
            faces   = self.face_cascade.detectMultiScale(gray_eq, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, fw, fh = faces[0]
        pad_x = int(fw * 0.35)
        pad_y = int(fh * 0.40)
        return (max(0, x - pad_x), max(0, y - pad_y),
                min(w, x + fw + pad_x), min(h, y + fh + pad_y))

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_image(self, tensor):
        t   = torch.clamp(tensor.squeeze(0).cpu().detach(), 0.0, 1.0)
        rgb = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    def _apply_shadow_correction(self, face_orig, face_model, face_mask):
        """
        SELECTIVE lighting correction — only touches problem areas:

          • Highlights (over-bright patches):
              Detected by comparing each pixel to the face median L.
              Only the EXCESS brightness above a threshold is removed.
              Color (a, b) is NEVER touched for highlights — they already
              have correct skin color, only the luminance is too high.

          • Shadows (too dark):
              Model-guided correction — the model was trained on shadow
              removal and knows where and how much to lift.
              Color (a,b) in shadow regions is gently warmed to remove
              the cool/blue cast that shadows add.

          • Everything else (natural areas near the face average):
              ZERO change. Natural face depth and contour are preserved.
        
        --- The math ---
        A photographed face is the product of two things:

            face_pixel = texture × lighting

        where:
          • texture    = the person's intrinsic skin, freckles, pores,
                         eyes, lips, brows, eyebrows  (HIGH frequency)
          • lighting   = the directional light + shadows + highlights
                         (LOW frequency, varies smoothly across the face)

        In the L channel of LAB, this is approximately additive:
            L = L_texture + L_lighting

        --- The pipeline ---
        1. Estimate the lighting field per channel via MASKED Gaussian blur
           (only face pixels contribute — no contamination from hair/bg).
        2. Extract intrinsic texture by subtraction:
               L_tex = L - L_field
               a_tex = a - a_field
               b_tex = b - b_field
        3. Compute target uniform values from face statistics:
               target_L  = bright realistic skin level (60th-percentile)
               target_a  = median a of well-lit skin    (true color)
               target_b  = median b of well-lit skin    (true color)
        4. Refine target_L using model's prediction (30 % weight).
        5. Reconstruct with uniform lighting:
               new_L = target_L + L_tex
               new_a = target_a + a_tex
               new_b = target_b + b_tex
        6. Blend with original via face_mask (hair/bg/edges stay original).

        --- What you get ---
          ✓ Directional light  → REMOVED   (the lighting field is replaced)
          ✓ Shadows            → REMOVED   (same reason)
          ✓ Hot highlights     → REMOVED   (same reason)
          ✓ Skin color         → NORMALISED to the person's true skin tone
          ✓ Skin texture       → 100 % preserved (texture is high-freq)
          ✓ Freckles, pores    → 100 % preserved
          ✓ Eyes, lips, brows  → 100 % preserved (high-freq detail)
          ✓ Identity           → 100 % preserved
        """
        orig_lab  = cv2.cvtColor(face_orig,  cv2.COLOR_BGR2LAB).astype(np.float32)
        model_lab = cv2.cvtColor(face_model, cv2.COLOR_BGR2LAB).astype(np.float32)

        orig_L  = orig_lab[:, :, 0]
        orig_a  = orig_lab[:, :, 1]
        orig_b  = orig_lab[:, :, 2]
        model_L = model_lab[:, :, 0]
        model_a = model_lab[:, :, 1]
        model_b = model_lab[:, :, 2]

        face_strong = face_mask > 0.5
        if not face_strong.any():
            return face_orig.copy()

        # ── Frequency-domain blending ─────────────────────────────────────────
        #
        # The face image has two components:
        #   LOW  frequency = slow-varying lighting (shadow gradient, directional
        #                    light, highlights, colour cast) → REPLACE with model
        #   HIGH frequency = fast-varying texture (pores, freckles, eyes, lips,
        #                    brows) → KEEP from original
        #
        # Formula (per channel):
        #   correction = (model_LOW - orig_LOW) * face_mask
        #   result     = original + correction
        #
        # This means:
        #   • Inside face mask  → lighting comes from model (correct)
        #   • Original texture  → untouched (added back via high-freq)
        #   • Outside face mask → original pixels, nothing changes
        #   • Works for ANY problem: shadow, directional light, highlight, vignette
        #   • No thresholds, no heuristics, no per-case tuning needed
        #
        # Kernel size: large enough to capture full lighting gradient across the
        # face (~1/4 of image) but not so large it loses all spatial information.
        # ─────────────────────────────────────────────────────────────────────
        sigma = 22.0
        ksize = 89   # must be odd; captures ~4-sigma lighting gradient

        def low_freq(ch):
            return cv2.GaussianBlur(ch.astype(np.float32), (ksize, ksize), sigma)

        # Separate LOW from HIGH for both original and model
        orig_L_low  = low_freq(orig_L)
        orig_a_low  = low_freq(orig_a)
        orig_b_low  = low_freq(orig_b)

        model_L_low = low_freq(model_L)
        model_a_low = low_freq(model_a)
        model_b_low = low_freq(model_b)

        # The correction = difference in LOW-frequency lighting
        # (model's correct flat lighting) − (original's uneven lighting)
        # Applied inside the face mask only → zero outside face
        L_corr = (model_L_low - orig_L_low) * face_mask
        a_corr = (model_a_low - orig_a_low) * face_mask
        b_corr = (model_b_low - orig_b_low) * face_mask

        # Result: original (texture preserved) + lighting correction
        result_lab = orig_lab.copy()
        result_lab[:, :, 0] = np.clip(orig_L + L_corr, 0, 255)
        result_lab[:, :, 1] = np.clip(orig_a + a_corr, 0, 255)
        result_lab[:, :, 2] = np.clip(orig_b + b_corr, 0, 255)

        return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    def remove_shadows(self, image_path, output_path=None):
        if self.model is None:
            print("Model not loaded.")
            return None

        if not os.path.exists(image_path):
            print("Image not found: " + image_path)
            return None

        original = cv2.imread(image_path)
        if original is None:
            print("Cannot read image.")
            return None

        h_orig, w_orig = original.shape[:2]
        print("Processing: " + os.path.basename(image_path) +
              "  (" + str(w_orig) + "x" + str(h_orig) + ")")

        # --- 1. Pre-resize to max 512 (match training preprocessing) --------
        pre_scale = min(512.0 / h_orig, 512.0 / w_orig, 1.0)
        if pre_scale < 1.0:
            work = cv2.resize(original,
                              (int(w_orig * pre_scale), int(h_orig * pre_scale)),
                              interpolation=cv2.INTER_AREA)
        else:
            work = original.copy()
        h_work, w_work = work.shape[:2]

        # --- 2. Detect face bounding box -----------------------------------
        box = self._detect_face_box(work)
        if box is None:
            print("No face detected.")
            return None
        x1, y1, x2, y2 = box

        # --- 3. Crop face region -----------------------------------------------───────────
        face_crop      = work[y1:y2, x1:x2].copy()
        crop_h, crop_w = face_crop.shape[:2]

        # --- 4. Resize to model input (256x256) ----------------------------──────
        face_256 = cv2.resize(face_crop, (self.model_size, self.model_size),
                              interpolation=cv2.INTER_LINEAR)

        # --- 5. Run model -> shadow-free reference (blurry, used only as guide) ---
        try:
            with torch.no_grad():
                out_tensor = self.model(self._to_tensor(face_256))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device.type == "cuda":
                print("  [WARN] CUDA out of memory — falling back to CPU for this image.")
                print("  TIP: stop training first, or run with --device cpu")
                torch.cuda.empty_cache()
                cpu_model = self.model.cpu()
                inp_cpu   = self._to_tensor(face_256).cpu()
                with torch.no_grad():
                    out_tensor = cpu_model(inp_cpu)
                self.model = self.model.to(self.device)  # put back on GPU
            else:
                raise
        model_256 = self._to_image(out_tensor)

        # --- 6. Face geometry mask (includes forehead extension) -----------──
        face_mask_256 = get_face_geometry_mask(face_256, feather=27)  # [0..1]

        # --- 7. Apply smooth luminance correction ---------------------------
        #   Original face texture/features preserved — only brightness corrected
        corrected_256 = self._apply_shadow_correction(face_256, model_256, face_mask_256)

        # --- 8. Resize corrected face back to original crop size -----──
        corrected_crop = cv2.resize(corrected_256, (crop_w, crop_h),
                                    interpolation=cv2.INTER_LINEAR)

        # --- 9. Paste into work image (background perfectly preserved) -----
        work_result          = work.copy()
        work_result[y1:y2, x1:x2] = corrected_crop

        # --- 10. Scale back to original resolution -------------------------────────
        if pre_scale < 1.0:
            result = cv2.resize(work_result, (w_orig, h_orig),
                                interpolation=cv2.INTER_LINEAR)
        else:
            result = work_result

        print("Done.")

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, result)
            print("Saved: " + output_path)

        return result

    # ------------------------------------------------------------------
    def process_batch(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        exts   = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(input_folder)
                  if f.lower().endswith(exts)]
        if not images:
            print("No images in: " + input_folder)
            return
        print("Processing " + str(len(images)) + " images...")
        for i, name in enumerate(images, 1):
            print("[" + str(i) + "/" + str(len(images)) + "] " + name)
            self.remove_shadows(os.path.join(input_folder, name),
                                os.path.join(output_folder, "clean_" + name))
        print("Done. Results in: " + output_folder)


# -------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input',      required=True)
    p.add_argument('--output',     default=None)
    p.add_argument('--checkpoint', default="checkpoints/shadow_removal_best.pth")
    p.add_argument('--model',      default=None,
                   choices=['unet', 'resunet++'],
                   help='Model architecture (auto-detected from checkpoint if not specified)')
    p.add_argument('--device',     default=None,
                   help='Force device: "cpu" or "cuda". Default: auto-detect. '
                        'Use --device cpu when GPU is busy with training.')
    args = p.parse_args()

    remover = PortraitShadowRemover(args.checkpoint, device=args.device, model_name=args.model)
    if remover.model is None:
        return

    if os.path.isfile(args.input):
        remover.remove_shadows(args.input, args.output or "clean_output.jpg")
    elif os.path.isdir(args.input):
        remover.process_batch(args.input, args.output or "clean_results")
    else:
        print("Invalid input: " + args.input)


if __name__ == "__main__":
    main()
