"""Shared command line plumbing for the analysis scripts.

Keeping the argument definitions here means every script that analyses
DeepLabCut output exposes exactly the same parameter names and defaults.
"""

from __future__ import annotations

import argparse

from larvatracker.config import DEFAULT_BODYPARTS, AnalysisConfig


def add_analysis_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every :class:`~larvatracker.config.AnalysisConfig` option."""
    defaults = AnalysisConfig()

    parser.add_argument(
        "--bodyparts",
        nargs="+",
        default=list(DEFAULT_BODYPARTS),
        help="keypoint labels in the order they appear in the DeepLabCut CSV",
    )
    parser.add_argument("--fps", type=float, default=defaults.fps, help="acquisition frame rate")
    parser.add_argument(
        "--likelihood",
        type=float,
        default=defaults.likelihood_threshold,
        help="DeepLabCut likelihood below which a keypoint is discarded",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=defaults.smoothing_window,
        help="width of the centred rolling mean, in frames",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=defaults.onset_threshold,
        help="smoothed displacement (px/frame) counting as heavy movement",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=defaults.peak_prominence,
        help="minimum peak prominence for burst detection",
    )
    parser.add_argument(
        "--peak-distance",
        type=int,
        default=defaults.peak_distance,
        help="minimum distance between bursts, in frames",
    )
    parser.add_argument(
        "--bin-size-frames",
        type=int,
        default=defaults.bin_size_frames,
        help="time bin length in frames (900 = 30 s at 30 fps)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from parsed arguments."""
    return AnalysisConfig(
        bodyparts=tuple(args.bodyparts),
        fps=args.fps,
        likelihood_threshold=args.likelihood,
        smoothing_window=args.smoothing_window,
        onset_threshold=args.onset_threshold,
        peak_prominence=args.peak_prominence,
        peak_distance=args.peak_distance,
        bin_size_frames=args.bin_size_frames,
    )
