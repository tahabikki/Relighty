import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import io
# Suppress MediaPipe init messages in worker threads
old_stderr = sys.stderr
sys.stderr = io.StringIO()

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import mediapipe as mp
sys.stderr = old_stderr  # Restore stderr

from utils.face_analyzer import FaceAnalyzer

class FaceDataset(Dataset):
    def __init__(self, file_list, output_size=256, augment=True):
        with open(file_list, 'r') as f:
            self.input_paths = [line.strip() for line in f]

        self.valid_indices = []
        for idx, path in enumerate(self.input_paths):
            if not path or not os.path.exists(path):
                continue
            target_path = path.replace('/dataset/input/', '/dataset/target/')
            if not os.path.exists(target_path):
                continue
            img = cv2.imread(path)
            if img is None or img.size == 0:
                continue
            target = cv2.imread(target_path)
            if target is None or target.size == 0:
                continue
            self.valid_indices.append(idx)

        self.output_size = output_size
        self.augment     = augment
        # NOTE: CascadeClassifier and FaceAnalyzer are NOT stored here
        # because cv2 objects cannot be pickled for multiprocessing.
        # They are created lazily inside each worker (see _get_detectors).
        self._face_cascade  = None
        self._eye_cascade   = None
        self._face_analyzer = None

    def _get_detectors(self):
        """Lazy init — called inside __getitem__ so each worker builds its own."""
        if self._face_cascade is None:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self._face_analyzer = FaceAnalyzer()
        return self._face_cascade, self._eye_cascade, self._face_analyzer

    def __len__(self):
        return len(self.valid_indices)
    
    def create_face_neck_mask(self, image):
        _, _, face_analyzer = self._get_detectors()
        return face_analyzer.compute_face_confidence_map(image)

    def detect_and_align(self, image):
        face_cascade, eye_cascade, _ = self._get_detectors()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None, None, None
        x, y, w, h = faces[0]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)
        if len(eyes) >= 2:
            ex1, ey1, _, _ = eyes[0]
            ex2, ey2, _, _ = eyes[1]
            left_eye = np.array([x + ex1, y + ey1], dtype=np.float64)
            right_eye = np.array([x + ex2, y + ey2], dtype=np.float64)
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye
        else:
            cx, cy = x + w/2, y + h/2
            left_eye = np.array([cx - w*0.3, cy - h*0.15], dtype=np.float64)
            right_eye = np.array([cx + w*0.3, cy - h*0.15], dtype=np.float64)
        src_center = (left_eye + right_eye) / 2
        dst_center = np.array([112.0, 115.0], dtype=np.float64)
        eye_dist = np.linalg.norm(right_eye - left_eye)
        dst_dist = np.linalg.norm(np.array([152.0, 115.0]) - np.array([112.0, 115.0]))
        scale = dst_dist / eye_dist if eye_dist > 0 else 1.0
        tx = dst_center[0] - scale * src_center[0]
        ty = dst_center[1] - scale * src_center[1]
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float64)
        aligned = cv2.warpAffine(image, M, (self.output_size, self.output_size),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Mask created AFTER alignment — already in 256x256 space, no second warp needed
        mask = self.create_face_neck_mask(aligned)
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return aligned, M, mask
    
    def augment_pair(self, inp, tgt, mask):
        """
        Augment the (input, target, mask) triplet for PASSPORT PHOTO lighting normalisation.

        Goal: model learns to convert any unevenly lit face → uniformly lit face,
        while preserving the person's real skin colour exactly.

        Rules:
          Geometric augmentations  → apply to ALL THREE (inp, tgt, mask stay aligned)
          Global photometrics      → apply to BOTH inp and tgt  (consistent exposure)
          Lighting/shadow problems → apply to inp ONLY (target is the "clean" reference)
        """
        h, w = inp.shape[:2]
        rng  = np.random.rand

        # ══════════════════════════════════════════════════════════════════
        # A. GEOMETRIC  — inp, tgt AND mask all move together
        # ══════════════════════════════════════════════════════════════════

        # Horizontal flip (50 %) — passport photos can have shadow from either side.
        # This alone doubles the effective training set.
        if rng() < 0.5:
            inp  = cv2.flip(inp,  1)
            tgt  = cv2.flip(tgt,  1)
            mask = cv2.flip(mask, 1)

        # Slight rotation ±8° — people never hold perfectly upright for selfies
        if rng() < 0.25:
            angle = np.random.uniform(-8, 8)
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            kw    = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            inp   = cv2.warpAffine(inp,  M_rot, (w, h), **kw)
            tgt   = cv2.warpAffine(tgt,  M_rot, (w, h), **kw)
            mask  = cv2.warpAffine(mask, M_rot, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)

        # ══════════════════════════════════════════════════════════════════
        # B. GLOBAL PHOTOMETRICS — apply to BOTH (same overall exposure)
        # ══════════════════════════════════════════════════════════════════

        # Overall brightness — some cameras overexpose / underexpose
        if rng() < 0.40:
            factor = np.random.uniform(0.70, 1.30)
            inp = np.clip(inp.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            tgt = np.clip(tgt.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Gamma — camera sensor curves differ (phone vs DSLR vs webcam)
        if rng() < 0.30:
            gamma = np.random.uniform(0.75, 1.30)
            lut   = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
            inp   = lut[inp]
            tgt   = lut[tgt]

        # Warm / cool colour cast — indoor tungsten vs daylight vs fluorescent
        # Applied to BOTH so the model learns "normalise skin within the cast",
        # not "remove the cast" (that's a different problem).
        if rng() < 0.25:
            b_shift = np.random.uniform(0.85, 1.15)
            g_shift = np.random.uniform(0.92, 1.08)
            r_shift = np.random.uniform(0.85, 1.15)
            shifts  = np.array([b_shift, g_shift, r_shift], dtype=np.float32)
            inp = np.clip(inp.astype(np.float32) * shifts, 0, 255).astype(np.uint8)
            tgt = np.clip(tgt.astype(np.float32) * shifts, 0, 255).astype(np.uint8)

        # ══════════════════════════════════════════════════════════════════
        # C. PASSPORT-SPECIFIC LIGHTING PROBLEMS — inp ONLY
        #    These simulate exactly what people encounter when taking
        #    passport / ID photos themselves: side shadows, top shadows,
        #    phone-angle chin shadows, window light from one side, etc.
        # ══════════════════════════════════════════════════════════════════

        # Keep everything in float32 — np.linspace defaults to float64 which
        # doubles memory usage and can cause MemoryError in DataLoader workers.
        inp_f = inp.astype(np.float32)

        # C1. DIRECTIONAL LIGHT — one side brighter, other darker (most common).
        #     Simulates window light, flash from one side, outdoor directional sun.
        #     Two modes:
        #       • Shadow-only  (dark→lit gradient,  bright side stays at 1.0)
        #       • Full directional (bright side boosted ABOVE 1.0, dark side drops)
        #     The full-directional mode is critical: in real photos the lit side
        #     is over-bright, not just the shadow side being dark.
        if rng() < 0.60:
            shadow_str = float(np.random.uniform(0.20, 0.70))   # dark side multiplier
            bright_str = float(np.random.uniform(1.05, 1.35))   # lit side boost (>1)

            # 40% chance: simulate full directional light (dark side + bright side)
            # 60% chance: shadow only (bright side stays at 1.0)
            if rng() < 0.40:
                lo, hi = shadow_str, bright_str   # full directional
            else:
                lo, hi = shadow_str, 1.0          # shadow only

            if rng() < 0.5:   # direction: lo on RIGHT, hi on LEFT
                grad = np.linspace(hi, lo, w, dtype=np.float32).reshape(1, w, 1)
            else:              # direction: lo on LEFT, hi on RIGHT
                grad = np.linspace(lo, hi, w, dtype=np.float32).reshape(1, w, 1)
            inp_f = np.clip(inp_f * grad, 0, 255)

        # C2. Top-down directional — overhead (forehead bright/dark, chin opposite)
        if rng() < 0.25:
            shadow_str = float(np.random.uniform(0.40, 0.80))
            bright_str = float(np.random.uniform(1.05, 1.25))
            use_boost  = rng() < 0.35   # sometimes add over-bright side too
            if rng() < 0.6:   # brighter at top (forehead), dark at chin
                hi = bright_str if use_boost else 1.0
                grad = np.linspace(hi, shadow_str, h, dtype=np.float32).reshape(h, 1, 1)
            else:              # brighter at bottom (chin), dark at forehead
                hi = bright_str if use_boost else 1.0
                grad = np.linspace(shadow_str, hi, h, dtype=np.float32).reshape(h, 1, 1)
            inp_f = np.clip(inp_f * grad, 0, 255)

        # C3. Diagonal directional — window from upper-left/right corner
        if rng() < 0.30:
            shadow_str = float(np.random.uniform(0.30, 0.70))
            bright_str = float(np.random.uniform(1.05, 1.30))
            use_boost  = rng() < 0.40
            lo, hi = shadow_str, (bright_str if use_boost else 1.0)
            gx = np.linspace(hi, lo, w, dtype=np.float32) if rng() < 0.5 \
                 else np.linspace(lo, hi, w, dtype=np.float32)
            gy = np.linspace(lo + 0.10, hi, h, dtype=np.float32)
            grad = np.clip(np.outer(gy, gx).astype(np.float32), 0, 2).reshape(h, w, 1)
            inp_f = np.clip(inp_f * grad, 0, 255)

        # C4. Centre spotlight / vignette — strong frontal flash or ring light
        if rng() < 0.15:
            cx, cy   = w / 2, h / 2
            Y, X     = np.mgrid[0:h, 0:w]
            dist     = np.sqrt(((X - cx) / cx) ** 2 +
                               ((Y - cy) / cy) ** 2).astype(np.float32)
            strength = float(np.random.uniform(0.50, 0.85))
            vignette = np.clip(1.0 - (1.0 - strength) * dist,
                               strength, 1.0).astype(np.float32)
            inp_f = np.clip(inp_f * vignette.reshape(h, w, 1), 0, 255)

        # C5. Hot-spot — single bright patch (phone torch, bare bulb close-up)
        #     The model must learn to detect and normalise a localised overexposed zone.
        if rng() < 0.20:
            spot_x  = float(np.random.uniform(0.2, 0.8)) * w
            spot_y  = float(np.random.uniform(0.15, 0.7)) * h
            radius  = float(np.random.uniform(0.15, 0.35)) * min(h, w)
            boost   = float(np.random.uniform(1.15, 1.50))
            Y, X    = np.mgrid[0:h, 0:w]
            dist    = np.sqrt((X - spot_x) ** 2 + (Y - spot_y) ** 2).astype(np.float32)
            spot_w  = np.clip(1.0 - dist / radius, 0, 1).astype(np.float32)
            spot_w  = cv2.GaussianBlur(spot_w, (31, 31), 10)
            hotspot = 1.0 + (boost - 1.0) * spot_w   # 1.0 outside, boost at centre
            inp_f   = np.clip(inp_f * hotspot.reshape(h, w, 1), 0, 255)

        inp = inp_f.astype(np.uint8)

        return inp, tgt, mask
    
    def to_tensor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0   # [0, 1] — matches inference
        return torch.from_numpy(image).permute(2, 0, 1).float()
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        input_path = self.input_paths[real_idx]
        target_path = input_path.replace('/dataset/input/', '/dataset/target/')

        input_img = cv2.imread(input_path)
        if input_img is None:
            return self._empty()

        h, w = input_img.shape[:2]
        scale = min(512 / h, 512 / w)
        if scale < 1:
            input_img = cv2.resize(input_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        target_img = cv2.imread(target_path)
        if target_img is None:
            del input_img
            return self._empty()

        if scale < 1:
            target_img = cv2.resize(target_img, (int(target_img.shape[1] * scale), int(target_img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        inp, M_inp, mask = self.detect_and_align(input_img)
        tgt, _, _        = self.detect_and_align(target_img)

        del input_img, target_img

        if inp is None or tgt is None:
            return self._empty()

        if self.augment:
            inp, tgt, mask = self.augment_pair(inp, tgt, mask)

        inp_tensor  = self.to_tensor(inp)
        tgt_tensor  = self.to_tensor(tgt)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        # analysis_map not used by model — return zeros to keep tuple interface
        dummy = torch.zeros(3, self.output_size, self.output_size)

        return inp_tensor, tgt_tensor, mask_tensor, dummy
    
    def _empty(self):
        s = self.output_size
        return (torch.zeros(3, s, s), torch.zeros(3, s, s),
                torch.ones(1, s, s), torch.zeros(3, s, s))