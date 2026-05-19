"""
Shadow-removal dataset with transparent background.

Uses PNG images with alpha channel from mask_bg_remove.py.
The transparent background is already removed - all pixels are the face+neck region.

For each pair:
  - Loads PNG with alpha channel (transparent background)
  - Uses alpha channel as mask
  - Returns (input_tensor, target_tensor, mask_tensor)
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_data_path


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert uint8 BGR image to float32 CHW tensor in [0, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).permute(2, 0, 1)


def _load_image_with_alpha(path: str) -> tuple:
    """
    Load image with alpha channel.
    
    Returns:
        (bgr_image, alpha_mask)
        - bgr: BGR image (without alpha)
        - alpha: grayscale mask (0-255)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return None, None
    
    if img.shape[2] == 4:
        b, g, r, alpha = cv2.split(img)
        bgr = cv2.merge([b, g, r])
        return bgr, alpha
    else:
        h, w = img.shape[:2]
        return img, np.ones((h, w), dtype=np.uint8) * 255


def _augment_pair(inp: np.ndarray, tgt: np.ndarray, mask: np.ndarray):
    """
    Consistent augmentation applied to input, target, and mask.
    """
    if np.random.rand() < 0.5:
        inp = cv2.flip(inp, 1)
        tgt = cv2.flip(tgt, 1)
        mask = cv2.flip(mask, 1)

    if np.random.rand() < 0.3:
        factor = np.random.uniform(0.85, 1.15)
        inp = np.clip(inp.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        tgt = np.clip(tgt.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return inp, tgt, mask


class ShadowDataset(Dataset):
    """
    Paired shadow-removal dataset using PNG with transparent background.
    
    Uses images processed with mask_bg_remove.py - so background is already
    removed and all pixels are face+neck region.

    Args:
        input_list:  path to .txt file, one input  image path per line
        target_list: path to .txt file, one target image path per line
        augment:     True for training set, False for validation
        image_size:  model input resolution (square)
    """

    def __init__(
        self,
        input_list: str,
        target_list: str,
        augment: bool = False,
        image_size: int = 256,
    ):
        self.augment    = augment
        self.image_size = image_size

        with open(input_list)  as f:
            self.inputs  = [l.strip() for l in f if l.strip()]
        with open(target_list) as f:
            self.targets = [l.strip() for l in f if l.strip()]

        assert len(self.inputs) == len(self.targets), (
            f"Mismatch: {len(self.inputs)} inputs vs {len(self.targets)} targets"
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inp_bgr, inp_alpha = _load_image_with_alpha(self.inputs[idx])
        tgt_bgr, tgt_alpha = _load_image_with_alpha(self.targets[idx])

        if inp_bgr is None or tgt_bgr is None:
            z = torch.zeros(3, self.image_size, self.image_size)
            m = torch.zeros(1, self.image_size, self.image_size)
            return z, z, m

        inp_bgr = cv2.resize(inp_bgr, (self.image_size, self.image_size))
        tgt_bgr = cv2.resize(tgt_bgr, (self.image_size, self.image_size))
        
        inp_alpha = cv2.resize(inp_alpha, (self.image_size, self.image_size))
        tgt_alpha = cv2.resize(tgt_alpha, (self.image_size, self.image_size))

        if self.augment:
            inp_bgr, tgt_bgr, inp_alpha = _augment_pair(inp_bgr, tgt_bgr, inp_alpha)

        mask = (inp_alpha / 255.0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return _to_tensor(inp_bgr), _to_tensor(tgt_bgr), mask