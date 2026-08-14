#!/usr/bin/env python
"""Optional — detailed kinematics for a single droplet.

Where ``03_analyze_experiment.py`` reduces each larva to a handful of summary
numbers, this script keeps the full time course for one animal: body axis
angle, angular velocity, the bending angle at each interior keypoint, total
curvature, and per-keypoint displacement. Useful for inspecting an individual
trace or for sanity-checking the body-axis sorting.

Example
-------
::

    python scripts/06_droplet_kinematics.py \
        data/M4/droplet_videos/droplet_001DLC_Resnet50_...csv \
        --out-dir results/droplet_001
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from larvatracker.cli import add_analysis_arguments, config_from_args
from larvatracker.metrics import bodypart_metrics, step_displacement
from larvatracker.plotting import plot_droplet_traces, plot_kinematics
from larvatracker.posture import (
    angular_velocity,
    body_axis_angle,
    center_of_mass,
    curvature,
    joint_angles,
    load_dlc_csv,
    sort_series,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full kinematic time course for a single DeepLabCut CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", help="a single droplet_XXXDLC_*.csv file")
    parser.add_argument("--out-dir", default=".", help="where to write figures and tables")
    parser.add_argument(
        "--max-gap",
        type=int,
        default=10,
        help="longest NaN run (frames) that is bridged by interpolation",
    )
    return add_analysis_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv_path))[0]

    frames, points = load_dlc_csv(args.csv_path, config.bodyparts, config.likelihood_threshold)
    sorted_xy = sort_series(points)

    valid_fraction = np.isfinite(sorted_xy[:, :, 0]).mean()
    print(f"{len(frames)} frames, {valid_fraction:.1%} of keypoints usable")

    # --- kinematics ------------------------------------------------------
    theta = body_axis_angle(sorted_xy, max_gap=args.max_gap)
    omega = angular_velocity(theta)
    joints = joint_angles(sorted_xy)
    total_curvature = curvature(sorted_xy)

    joint_labels = [f"joint {config.bodyparts[i + 1]}" for i in range(joints.shape[1])]

    plot_kinematics(
        frames=frames,
        theta=theta,
        ang_vel=omega,
        joints=joints,
        total_curvature=total_curvature,
        out_path=os.path.join(args.out_dir, f"{stem}_kinematics.png"),
        joint_labels=joint_labels,
    )

    # --- displacement per keypoint --------------------------------------
    metrics_per_part = {
        part: bodypart_metrics(sorted_xy, i, config)
        for i, part in enumerate(config.bodyparts)
    }

    plot_droplet_traces(
        frames=frames,
        metrics_per_part=metrics_per_part,
        out_path=os.path.join(args.out_dir, f"{stem}_displacement.png"),
        title=stem,
    )

    # --- tabular export --------------------------------------------------
    com = center_of_mass(sorted_xy)

    table = pd.DataFrame({"frame": frames})
    table["body_axis_angle_rad"] = theta
    table["angular_velocity_rad_per_frame"] = omega
    table["curvature_rad"] = total_curvature
    table["com_x"] = com[:, 0]
    table["com_y"] = com[:, 1]

    for i, part in enumerate(config.bodyparts):
        table[f"{part}_x"] = sorted_xy[:, i, 0]
        table[f"{part}_y"] = sorted_xy[:, i, 1]
        table[f"{part}_step_px"] = step_displacement(sorted_xy[:, i, :])
        # Displacement of the keypoint relative to the animal's centre isolates
        # body bending from whole-animal translation.
        table[f"{part}_step_rel_com_px"] = step_displacement(sorted_xy[:, i, :] - com)

    for i, label in enumerate(joint_labels):
        table[label.replace(" ", "_") + "_rad"] = joints[:, i]

    table_path = os.path.join(args.out_dir, f"{stem}_kinematics.csv")
    table.to_csv(table_path, index=False)

    print(f"Figures and {table_path} written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
