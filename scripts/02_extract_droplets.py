#!/usr/bin/env python
"""Step 1b — rebuild the droplet schema from an existing mask.

Useful when the SAM 3 mask is fine but the droplet parameters are not: it
re-runs only the connected-component step, so a different ``--min-area-px`` or
``--padding-px`` can be tried without paying for GPU inference again. The mask
may also be a hand-corrected version of ``frame0_mask.png``.

Example
-------
::

    python scripts/02_extract_droplets.py data/M4/frame0_mask.png \
        --video data/M4.mp4 --min-area-px 150
"""

from __future__ import annotations

import argparse
import os
import sys

from larvatracker.config import SegmentationConfig
from larvatracker.droplets import find_droplets, save_schema_outputs
from larvatracker.imaging import load_mask, mask_as_bgr, read_first_frame
from larvatracker.roi_videos import write_droplet_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the droplet schema (and optionally ROI videos) from a mask PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = SegmentationConfig()

    parser.add_argument("mask", help="binary mask PNG, e.g. frame0_mask.png")
    parser.add_argument(
        "--video",
        default=None,
        help="raw recording; required for the overlay image and for ROI videos",
    )
    parser.add_argument("--out-dir", default=None, help="default: the mask's own folder")
    parser.add_argument("--min-area-px", type=int, default=defaults.min_area_px)
    parser.add_argument("--padding-px", type=int, default=defaults.padding_px)
    parser.add_argument("--codec", default=defaults.codec)
    parser.add_argument(
        "--videos", action="store_true", help="also write the ROI videos (needs --video)"
    )
    parser.add_argument("--pixel-table", action="store_true", help="also write droplet_pixels.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.videos and not args.video:
        print("error: --videos requires --video", file=sys.stderr)
        return 2

    config = SegmentationConfig(
        min_area_px=args.min_area_px, padding_px=args.padding_px, codec=args.codec
    )

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.mask))
    mask = load_mask(args.mask)

    droplets, id_mask = find_droplets(mask, config.min_area_px, config.padding_px)
    print(f"Droplets found: {len(droplets)}")

    if args.video:
        frame_bgr = read_first_frame(args.video)
    else:
        # No source frame available: render the schema on the mask itself.
        frame_bgr = mask_as_bgr(mask)

    save_schema_outputs(
        out_dir=out_dir,
        frame_bgr=frame_bgr,
        mask=mask,
        droplets=droplets,
        id_mask=id_mask,
        config=config,
        write_pixel_table=args.pixel_table,
    )
    print(f"Schema written to {out_dir}")

    if args.videos:
        write_droplet_videos(
            video_path=args.video,
            droplets=droplets,
            out_dir=os.path.join(out_dir, "droplet_videos"),
            codec=config.codec,
            mask_background=config.mask_background,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
