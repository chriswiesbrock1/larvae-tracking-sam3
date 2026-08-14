"""Small image helpers shared by the segmentation and analysis halves.

Deliberately free of any torch or transformers import, so that everything
downstream of SAM 3 stays usable on a machine without a GPU stack installed.
"""

from __future__ import annotations

import cv2
import numpy as np


def read_first_frame(video_path: str) -> np.ndarray:
    """Return the first frame of ``video_path`` as a BGR array.

    Raises
    ------
    RuntimeError
        If the video cannot be opened or contains no readable frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read the first frame of {video_path!r}.")

    return frame_bgr


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a green overlay of ``mask`` onto ``frame_bgr``."""
    idx = np.asarray(mask) > 0

    out = frame_bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255

    out[idx] = (out[idx] * (1.0 - alpha) + green[idx] * alpha).astype(np.uint8)
    return out


def load_mask(path: str) -> np.ndarray:
    """Read a binary mask PNG as a boolean array."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def mask_as_bgr(mask: np.ndarray, value: int = 180) -> np.ndarray:
    """Render a boolean mask as a grey BGR image.

    Used as a stand-in background when the schema has to be drawn without
    access to the original recording.
    """
    grey = (np.asarray(mask) > 0).astype(np.uint8) * value
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
