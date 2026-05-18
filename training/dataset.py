"""
Shadow-removal dataset.

Reads paired (input, target) image lists from split txt files.
For each pair:
  - Resizes to model input size
  - Generates a MediaPipe face mask (lazy per-worker, multiprocessing safe)
  - Returns (input_tensor, target_tensor, face_mask_tensor)

Tensors are float32 in [0, 1] range, shape (C, H, W).
The face mask is shape (1, H, W) and is used to restrict the loss
to the face region, so the model never wastes gradients on background.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from masking.mediapipe_mask import FaceMaskGenerator


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert uint8 BGR image to float32 CHW tensor in [0, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).permute(2, 0, 1)   # C H W


def _augment_pair(inp: np.ndarray, tgt: np.ndarray):
    """
    Consistent augmentation applied to both input and target.
    All transforms preserve the pixel-wise correspondence.
    """
    # Horizontal flip
    if np.random.rand() < 0.5:
        inp = cv2.flip(inp, 1)
        tgt = cv2.flip(tgt, 1)

    # Brightness jitter (same factor on both so shadow offset is preserved)
    if np.random.rand() < 0.3:
        factor = np.random.uniform(0.85, 1.15)
        inp = np.clip(inp.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        tgt = np.clip(tgt.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return inp, tgt


class ShadowDataset(Dataset):
    """
    Paired shadow-removal dataset.

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

        # Mask generator is created lazily inside each DataLoader worker
        # to avoid pickling / fork issues with MediaPipe's TF backend.
        self._mask_gen = None

    @property
    def mask_gen(self) -> FaceMaskGenerator:
        if self._mask_gen is None:
            self._mask_gen = FaceMaskGenerator(self.image_size)
        return self._mask_gen

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inp_bgr = cv2.imread(self.inputs[idx])
        tgt_bgr = cv2.imread(self.targets[idx])

        if inp_bgr is None or tgt_bgr is None:
            # Graceful fallback for missing/corrupt files
            z = torch.zeros(3, self.image_size, self.image_size)
            m = torch.zeros(1, self.image_size, self.image_size)
            return z, z, m

        inp_bgr = cv2.resize(inp_bgr, (self.image_size, self.image_size))
        tgt_bgr = cv2.resize(tgt_bgr, (self.image_size, self.image_size))

        if self.augment:
            inp_bgr, tgt_bgr = _augment_pair(inp_bgr, tgt_bgr)

        # Face mask — computed from input image
        inp_rgb = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)
        mask_hw = self.mask_gen(inp_rgb)                         # H x W  float32
        mask    = torch.from_numpy(mask_hw).unsqueeze(0)        # 1 x H x W

        return _to_tensor(inp_bgr), _to_tensor(tgt_bgr), mask
