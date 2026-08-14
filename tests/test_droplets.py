"""Tests for the mask-to-droplet-schema step."""

import numpy as np
import pytest

from larvatracker.droplets import find_droplets


def circles_mask(height=240, width=360, centers=((60, 60), (180, 60), (300, 170)), radius=34):
    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[:height, :width]

    for cx, cy in centers:
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

    return mask


def test_finds_one_droplet_per_blob():
    droplets, id_mask = find_droplets(circles_mask(), min_area_px=100, padding_px=8)

    assert [d.id for d in droplets] == [1, 2, 3]
    assert sorted(np.unique(id_mask).tolist()) == [0, 1, 2, 3]


def test_small_blobs_are_discarded():
    mask = circles_mask()
    mask[5:8, 5:8] = True  # 9-pixel speck

    droplets, _ = find_droplets(mask, min_area_px=100, padding_px=0)

    assert len(droplets) == 3


def test_roi_dimensions_are_even_and_match_the_mask():
    """ROI videos are written at these dimensions; odd sizes get truncated."""
    droplets, _ = find_droplets(circles_mask(), min_area_px=100, padding_px=8)

    for drop in droplets:
        assert drop.width % 2 == 0
        assert drop.height % 2 == 0
        assert drop.roi_mask.shape == (drop.height, drop.width)


def test_boxes_stay_inside_an_odd_sized_image():
    mask = np.zeros((101, 99), dtype=bool)
    mask[:, :] = True

    droplets, _ = find_droplets(mask, min_area_px=10, padding_px=8)
    drop = droplets[0]

    assert 0 <= drop.x0 < drop.x1 <= 99
    assert 0 <= drop.y0 < drop.y1 <= 101
    assert drop.width % 2 == 0 and drop.height % 2 == 0


def test_empty_mask_raises():
    with pytest.raises(RuntimeError):
        find_droplets(np.zeros((50, 50), dtype=bool), min_area_px=100)
