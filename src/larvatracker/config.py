"""Default parameters for the whole pipeline.

Every value here is a documented default, not a hard requirement. The command
line scripts in ``scripts/`` expose all of them as flags, so nothing in this
file needs to be edited to process a new dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The DeepLabCut model used in this project labels five points along the larva,
# from head (a) to tail (e).
DEFAULT_BODYPARTS: tuple[str, ...] = ("a", "b", "c", "d", "e")

# Acquisition frame rate of the Basler cameras used in the setup.
DEFAULT_FPS: float = 30.0


@dataclass
class SegmentationConfig:
    """Parameters for SAM 3 droplet segmentation and ROI extraction.

    Attributes
    ----------
    prompt:
        Text prompt handed to SAM 3. The model is prompted on the *first frame
        only*; droplets do not move, so a single-frame segmentation is enough
        and avoids running video inference over the whole recording.
    threshold:
        Soft threshold applied to the predicted mask logits. Lower values
        recover faint droplet borders at the cost of more background. Useful
        range is roughly 0.06-0.15.
    min_area_px:
        Connected components smaller than this are discarded as noise.
    padding_px:
        Extra margin added around each droplet bounding box before cutting the
        ROI video, so the larva is not clipped when it touches the border.
    overlay_alpha:
        Blending strength of the green mask overlay in the schema image.
    codec:
        FourCC code for the per-droplet ROI videos. ``mp4v`` is the most
        portable choice for OpenCV on Windows.
    mask_background:
        If True, pixels outside the droplet are blanked in the ROI videos. This
        keeps neighbouring droplets out of frame, which noticeably improves
        DeepLabCut tracking.
    """

    prompt: str = "clear water droplets on a metallic surface"
    threshold: float = 0.10
    min_area_px: int = 200
    padding_px: int = 8
    overlay_alpha: float = 0.45
    codec: str = "mp4v"
    mask_background: bool = True
    model_id: str = "facebook/sam3"


@dataclass
class AnalysisConfig:
    """Parameters for the analysis of DeepLabCut pose estimates.

    Attributes
    ----------
    bodyparts:
        Labels in the order they appear in the DeepLabCut CSV.
    fps:
        Frame rate used to convert frame counts into seconds and Hz.
    likelihood_threshold:
        Keypoints below this DeepLabCut likelihood are set to NaN before any
        further computation.
    smoothing_window:
        Width (in frames) of the centred rolling mean applied to the
        frame-to-frame displacement.
    onset_threshold:
        Smoothed displacement (px/frame) above which movement counts as a
        "heavy movement" onset.
    peak_prominence, peak_distance:
        ``scipy.signal.find_peaks`` parameters used for burst detection.
        Prominence suppresses tracking jitter, distance enforces a refractory
        period between bursts.
    bin_size_frames:
        Length of a time bin for the time-resolved analysis. The default of
        900 frames equals 30 s at 30 fps.
    """

    bodyparts: tuple[str, ...] = DEFAULT_BODYPARTS
    fps: float = DEFAULT_FPS
    likelihood_threshold: float = 0.6
    smoothing_window: int = 15
    onset_threshold: float = 4.0
    peak_prominence: float = 1.2
    peak_distance: int = 10
    bin_size_frames: int = 900

    def seconds(self, n_frames: float) -> float:
        """Convert a number of frames to seconds."""
        return float(n_frames) / self.fps
