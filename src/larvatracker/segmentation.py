"""Droplet segmentation with SAM 3.

The recording geometry is static: droplets are pipetted onto a metallic
surface and do not move during the experiment. It is therefore sufficient to
segment the *first* frame and reuse that mask for the entire recording, which
turns an expensive video segmentation into a single forward pass.

The model is loaded in fp32 and executed under ``torch.autocast`` in fp16.
Loading the weights in fp16 directly causes dtype mismatches inside the SAM 3
video processor, so this split is intentional.
"""

from __future__ import annotations

import os

# Must be set before torch is imported, otherwise it has no effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np

from larvatracker.config import SegmentationConfig

# Re-exported for convenience; these live in a torch-free module so that the
# analysis half of the package can be imported without a GPU stack.
from larvatracker.imaging import overlay_mask, read_first_frame  # noqa: F401


def _resolve_device(require_cuda: bool = True):
    """Pick the inference device, failing loudly if CUDA was expected."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        return device

    if require_cuda:
        raise RuntimeError(
            "CUDA is not available. SAM 3 inference on CPU is impractically "
            "slow; check the PyTorch CUDA installation or pass --allow-cpu."
        )

    print("WARNING: running SAM 3 on CPU, this will be very slow.")
    return torch.device("cpu")


def load_sam3(
    config: SegmentationConfig | None = None,
    require_cuda: bool = True,
) -> tuple[object, object, object]:
    """Load the SAM 3 model and processor once.

    Loading the weights takes appreciably longer than segmenting a single
    frame, so a batch run loads them once and passes the result into every
    call of :func:`segment_first_frame`.

    Returns
    -------
    (device, model, processor)
    """
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    cfg = config or SegmentationConfig()
    device = _resolve_device(require_cuda=require_cuda)

    model = Sam3VideoModel.from_pretrained(cfg.model_id).to(device)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(cfg.model_id)

    return device, model, processor


def segment_first_frame(
    video_path: str,
    config: SegmentationConfig | None = None,
    require_cuda: bool = True,
    components: tuple[object, object, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment droplets in the first frame of a recording.

    Parameters
    ----------
    video_path:
        Path to the raw recording.
    config:
        Segmentation parameters; defaults are used when omitted.
    require_cuda:
        Abort instead of silently falling back to CPU inference.
    components:
        A ``(device, model, processor)`` tuple from :func:`load_sam3`. Pass
        this when processing several videos so the weights are loaded once.

    Returns
    -------
    (frame_bgr, mask)
        The first frame and a boolean mask that is the union of all instances
        SAM 3 returned for the prompt.
    """
    # Imported lazily so that the analysis half of the package stays usable
    # without a transformers/torch-CUDA installation.
    import torch

    cfg = config or SegmentationConfig()

    if components is None:
        components = load_sam3(cfg, require_cuda=require_cuda)

    device, model, processor = components

    frame_bgr = read_first_frame(video_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # A "video session" with a single frame. Keeping processing and storage on
    # the CPU keeps VRAM usage low and has proven more stable than running the
    # full pipeline on the GPU.
    session = processor.init_video_session(
        video=[frame_rgb],
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.float16,
    )
    processor.add_text_prompt(session, text=cfg.prompt)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.float16):
        outputs = model(inference_session=session, frame_idx=0)
        post = processor.postprocess_outputs(session, outputs)

    mask = _masks_to_union(post, cfg.threshold, frame_bgr.shape[:2])

    # Release the session before the next video; without this a long batch
    # accumulates VRAM until it fails partway through.
    del session, outputs, post
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return frame_bgr, mask


def _masks_to_union(post: dict, threshold: float, shape: tuple[int, int]) -> np.ndarray:
    """Reduce the per-instance masks returned by SAM 3 to a single boolean mask."""
    import torch

    masks = post.get("pred_masks", None)
    if masks is None:
        masks = post.get("masks", None)

    union = None

    if masks is not None:
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)

        if masks.ndim == 3:          # (n_instances, H, W)
            union = np.any(masks > threshold, axis=0)
        elif masks.ndim == 2:        # (H, W)
            union = masks > threshold

    if union is None or not union.any():
        print(
            f"WARNING: empty mask at threshold {threshold}. "
            "Try a lower threshold (e.g. 0.06) or a more specific prompt."
        )
        union = np.zeros(shape, dtype=bool)

    return union
