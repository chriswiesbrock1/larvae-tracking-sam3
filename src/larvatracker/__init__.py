"""larvatracker — SAM 3 based droplet segmentation and larva posture analysis.

The package is organised along the two halves of the pipeline:

Acquisition / segmentation
    :mod:`larvatracker.segmentation`  SAM 3 inference on the first video frame
    :mod:`larvatracker.droplets`      connected components -> droplet schema
    :mod:`larvatracker.roi_videos`    per-droplet ROI video export
    :mod:`larvatracker.lcd_temperature`  chamber temperature off the LCD

Analysis of pose estimates
    :mod:`larvatracker.posture`   loading DeepLabCut output, body-axis sorting
    :mod:`larvatracker.metrics`   displacement, bursts, onset, time bins
    :mod:`larvatracker.scheme`    mapping droplet IDs to experimental groups
    :mod:`larvatracker.plotting`  per-droplet and population figures
    :mod:`larvatracker.stats`     baseline normalisation and group statistics
    :mod:`larvatracker.framewise`    per-frame movement joined with temperature
    :mod:`larvatracker.temperature`  movement versus temperature across groups
    :mod:`larvatracker.model`     mixed model for the temperature response
"""

__version__ = "0.1.0"

from larvatracker.config import (
    DEFAULT_BODYPARTS,
    DEFAULT_FPS,
    SegmentationConfig,
    AnalysisConfig,
)

__all__ = [
    "__version__",
    "DEFAULT_BODYPARTS",
    "DEFAULT_FPS",
    "SegmentationConfig",
    "AnalysisConfig",
]
