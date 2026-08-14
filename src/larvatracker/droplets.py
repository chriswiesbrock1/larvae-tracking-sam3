"""Turning a binary droplet mask into an addressable droplet schema.

The mask produced by :mod:`larvatracker.segmentation` is a single boolean
image. This module splits it into individual droplets via connected
components and assigns each one a stable numeric ID. That ID is what ties
together every downstream artefact: the ROI video ``droplet_007.mp4``, the row
``id=7`` in ``droplets.csv`` and the row ``Droplet=7`` in the experimental
scheme.

IDs are assigned in the order OpenCV returns the components, which is
row-major by the topmost pixel of each component. The numbering is therefore
reproducible for a given mask, but it is *not* stable across recordings — each
recording gets its own schema image for manual cross-checking.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np

from larvatracker.config import SegmentationConfig
from larvatracker.imaging import overlay_mask


@dataclass
class Droplet:
    """A single segmented droplet.

    Attributes
    ----------
    id:
        1-based droplet ID, used in all file names and tables.
    area_px:
        Number of mask pixels belonging to the droplet.
    cx, cy:
        Centroid in image coordinates.
    x0, y0, x1, y1:
        Padded bounding box, ``[x0, x1)`` by ``[y0, y1)``.
    roi_mask:
        Boolean mask of the droplet, cropped to the bounding box. Used to blank
        the background in the ROI video.
    """

    id: int
    area_px: int
    cx: float
    cy: float
    x0: int
    y0: int
    x1: int
    y1: int
    roi_mask: np.ndarray

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_row(self) -> dict:
        """Serialisable subset of the droplet, without the pixel mask."""
        return {
            "id": self.id,
            "area_px": self.area_px,
            "cx": round(float(self.cx), 3),
            "cy": round(float(self.cy), 3),
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


DROPLET_TABLE_COLUMNS = ["id", "area_px", "cx", "cy", "x0", "y0", "x1", "y1"]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _make_even(low: int, high: int, limit: int) -> tuple[int, int]:
    """Extend a ``[low, high)`` interval so its length is even.

    Video codecs operate on 2x2 macroblocks. Given an odd frame size, OpenCV's
    writer silently drops the last row or column instead of failing, which
    would make the ROI video disagree with the bounding box in ``droplets.csv``
    by one pixel. Growing the box by one keeps the two consistent.

    The interval is grown towards the end that still has room; if the image
    itself has an odd extent and the box already spans it, one pixel is given
    up instead.
    """
    if (high - low) % 2 == 0:
        return low, high
    if high < limit:
        return low, high + 1
    if low > 0:
        return low - 1, high
    return low, high - 1


def find_droplets(
    mask: np.ndarray,
    min_area_px: int = 200,
    padding_px: int = 8,
) -> tuple[list[Droplet], np.ndarray]:
    """Split a binary mask into individual droplets.

    Parameters
    ----------
    mask:
        Boolean or 0/255 mask of all droplets.
    min_area_px:
        Components below this area are treated as segmentation noise.
    padding_px:
        Margin added around each bounding box, clamped to the image.

    Returns
    -------
    (droplets, id_mask)
        The droplets in ID order, and a ``uint16`` label image where each pixel
        holds the ID of its droplet (0 = background). ``uint16`` is used so that
        experiments with more than 255 droplets remain representable.
    """
    binmask = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binmask.shape[:2]

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binmask, connectivity=8
    )

    droplets: list[Droplet] = []
    id_mask = np.zeros((height, width), dtype=np.uint16)
    droplet_id = 0

    for label in range(1, n_labels):  # label 0 is the background
        x, y, w, h, area = stats[label]

        if area < min_area_px:
            continue

        droplet_id += 1
        cx, cy = centroids[label]

        x0 = _clamp(x - padding_px, 0, width - 1)
        y0 = _clamp(y - padding_px, 0, height - 1)
        x1 = _clamp(x + w + padding_px, 0, width)
        y1 = _clamp(y + h + padding_px, 0, height)

        # Keep the ROI dimensions even so the exported video matches the box.
        x0, x1 = _make_even(x0, x1, width)
        y0, y1 = _make_even(y0, y1, height)

        id_mask[labels == label] = droplet_id
        roi_mask = (labels[y0:y1, x0:x1] == label)

        droplets.append(
            Droplet(
                id=droplet_id,
                area_px=int(area),
                cx=float(cx),
                cy=float(cy),
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                roi_mask=roi_mask,
            )
        )

    if not droplets:
        raise RuntimeError(
            "No droplets found. Lower --min-area-px, lower the segmentation "
            "threshold, or inspect frame0_mask.png."
        )

    return droplets, id_mask


def draw_schema(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    droplets: list[Droplet],
    alpha: float = 0.45,
) -> np.ndarray:
    """Render the annotated overview image used to read off droplet IDs."""
    overlay = overlay_mask(frame_bgr, mask, alpha=alpha)

    for drop in droplets:
        cv2.rectangle(
            overlay, (drop.x0, drop.y0), (drop.x1 - 1, drop.y1 - 1), (0, 255, 0), 1
        )
        cv2.putText(
            overlay,
            str(drop.id),
            (int(drop.cx), int(drop.cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def save_droplet_table(droplets: list[Droplet], path: str) -> None:
    """Write one row per droplet (ID, area, centroid, bounding box)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DROPLET_TABLE_COLUMNS)
        writer.writeheader()
        for drop in droplets:
            writer.writerow(drop.as_row())


def save_droplet_pixels(id_mask: np.ndarray, path: str) -> None:
    """Write the exact pixel coordinates of every droplet.

    The output has one row per mask pixel (``id, x, y``) and can become large
    for high-resolution recordings. It is only needed when droplet areas have
    to be re-measured or intersected with other image data; the bounding boxes
    in ``droplets.csv`` are sufficient for the standard pipeline.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y"])

        for droplet_id in range(1, int(id_mask.max()) + 1):
            ys, xs = np.where(id_mask == droplet_id)
            for x, y in zip(xs, ys):
                writer.writerow([droplet_id, int(x), int(y)])


def save_schema_outputs(
    out_dir: str,
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    droplets: list[Droplet],
    id_mask: np.ndarray,
    config: SegmentationConfig | None = None,
    write_pixel_table: bool = False,
) -> None:
    """Write the full set of schema artefacts for one recording.

    Produces ``frame0_mask.png``, ``frame0_overlay.png``, ``droplet_id_mask.png``,
    ``droplet_schema.png``, ``droplets.csv`` and optionally ``droplet_pixels.csv``.
    """
    cfg = config or SegmentationConfig()
    os.makedirs(out_dir, exist_ok=True)

    binmask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    schema = draw_schema(frame_bgr, mask, droplets, alpha=cfg.overlay_alpha)

    cv2.imwrite(os.path.join(out_dir, "frame0_mask.png"), binmask)
    cv2.imwrite(os.path.join(out_dir, "frame0_overlay.png"), overlay_mask(frame_bgr, mask, cfg.overlay_alpha))
    cv2.imwrite(os.path.join(out_dir, "droplet_id_mask.png"), id_mask)
    cv2.imwrite(os.path.join(out_dir, "droplet_schema.png"), schema)

    save_droplet_table(droplets, os.path.join(out_dir, "droplets.csv"))

    if write_pixel_table:
        save_droplet_pixels(id_mask, os.path.join(out_dir, "droplet_pixels.csv"))
