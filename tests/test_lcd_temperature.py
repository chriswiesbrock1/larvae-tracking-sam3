"""Tests for the seven-segment LCD temperature reader.

The reader is verified against a synthetic display rendered with the same
geometry profile it searches for. That checks the locator, the digit decoding
and the temporal filtering without needing a real recording — and it fails
loudly if the geometry definitions and the sampling ever drift apart.
"""

import numpy as np
import pytest

from larvatracker.lcd_temperature import (
    DIGIT_PATTERNS,
    GEOMETRY_PROFILES,
    REFERENCE_FRAME_SIZE,
    LcdTemperatureReader,
    TemperatureConfig,
    fill_short_gaps,
    filter_temperature_records,
    geometry_offsets,
    write_temperature_csv,
)

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# Synthetic display
# ---------------------------------------------------------------------------

# Whether each segment runs horizontally (a, d, g) or vertically (b, c, e, f),
# in the a..g order used throughout. Only the orientation is defined here; the
# positions come from the geometry profile itself, so the rendered display and
# the reader's sample grid cannot drift apart.
SEGMENT_HORIZONTAL = (True, False, False, True, False, False, True)


def render_display(
    value: float,
    frame_size=(1080, 1920),
    origin=(1400, 300),
    background=(40, 40, 40),
    profile_name: str = "compact",
) -> np.ndarray:
    """Draw a three-digit cyan LCD showing ``value`` (e.g. 23.4) on a frame.

    Each lit segment is a dark bar centred on the sample point the reader will
    look at, taken straight from the geometry profile. The panel behind them is
    saturated cyan, which is what the locator scores for.
    """
    height, width = frame_size
    frame = np.full((height, width, 3), background, dtype=np.uint8)

    # Scene clutter, so the locator has to discriminate rather than simply
    # find the only structured region in an otherwise empty frame.
    rng = np.random.default_rng(3)
    for _ in range(60):
        cx, cy = rng.integers(0, width), rng.integers(0, height)
        cv2.circle(frame, (int(cx), int(cy)), int(rng.integers(10, 40)), (150, 150, 150), -1)

    profile = GEOMETRY_PROFILES[profile_name]
    ref_width, ref_height = REFERENCE_FRAME_SIZE
    scale = min(width / ref_width, height / ref_height)
    offsets = geometry_offsets(profile, scale, 0.0)

    tenths = int(round(value * 10))
    digits = [tenths // 100, (tenths // 10) % 10, tenths % 10]

    ox, oy = origin
    pad = int(round(16 * scale))
    cv2.rectangle(
        frame,
        (ox + int(offsets[:, :, 0].min()) - pad, oy + int(offsets[:, :, 1].min()) - pad),
        (ox + int(offsets[:, :, 0].max()) + pad, oy + int(offsets[:, :, 1].max()) + pad),
        (210, 190, 40),          # BGR: strong blue and green = cyan
        -1,
    )

    # Half-lengths taken from the cell the profile describes, so the bars meet
    # at the corners the way a real seven-segment glyph does. Drawing them
    # shorter would leave each segment as its own blob, and the digit-finding
    # in the calibration looks for whole digits.
    cell = offsets[0]
    half_horizontal = int(round(0.55 * (cell[:, 0].max() - cell[:, 0].min())))
    half_vertical = int(round(0.30 * (cell[:, 1].max() - cell[:, 1].min())))
    # Thick, high-contrast bars: these frames are written to mp4 and read back
    # in the calibration tests, and thin strokes do not survive the compression.
    thickness = max(5, int(round(9 * scale)))

    for digit_index, digit in enumerate(digits):
        pattern = DIGIT_PATTERNS[digit]

        for segment_index, lit in enumerate(pattern):
            if not lit:
                continue

            px, py = offsets[digit_index, segment_index]
            cx, cy = ox + int(px), oy + int(py)

            if SEGMENT_HORIZONTAL[segment_index]:
                start, end = (cx - half_horizontal, cy), (cx + half_horizontal, cy)
            else:
                start, end = (cx, cy - half_vertical), (cx, cy + half_vertical)

            cv2.line(frame, start, end, (15, 15, 15), thickness)

    return frame


def make_reader(frame: np.ndarray, origin=(1400, 300)) -> LcdTemperatureReader:
    """Build a reader anchored on a known display position.

    This isolates the decoding from the search: the anchor is handed in rather
    than found, so a failure points at the digit decoding and not at the
    locator.
    """
    profile = GEOMETRY_PROFILES["compact"]
    ref_width, ref_height = REFERENCE_FRAME_SIZE
    scale = min(frame.shape[1] / ref_width, frame.shape[0] / ref_height)

    candidate = {
        "geometry": "compact",
        "effective_scale": scale,
        "angle_deg": 0.0,
        "anchor_x": origin[0],
        "anchor_y": origin[1],
        "locator_score": 100.0,
        "kernel_size": 17,
        "sample_size": 5,
        "sample_offsets": geometry_offsets(profile, scale, 0.0),
    }

    reader = LcdTemperatureReader(frame.shape, candidate)
    reader.calibrate(frame)
    return reader


# ---------------------------------------------------------------------------
# Digit decoding
# ---------------------------------------------------------------------------

def test_digit_patterns_are_distinct_and_complete():
    """All ten digits must be present and no two may share a pattern."""
    assert sorted(DIGIT_PATTERNS) == list(range(10))
    assert len({tuple(p) for p in DIGIT_PATTERNS.values()}) == 10
    assert all(len(p) == 7 for p in DIGIT_PATTERNS.values())


@pytest.mark.parametrize("value", [21.1, 23.4, 29.9, 30.0, 37.5, 38.6, 40.2])
def test_reader_decodes_a_synthetic_display(value):
    frame = render_display(value)
    reader = make_reader(frame)

    result = reader.read(frame)

    assert result["raw_temperature_c"] == pytest.approx(value, abs=0.05)
    assert result["raw_valid"]
    assert result["segment_errors"] == 0


def test_reader_survives_a_small_camera_shift():
    """A few pixels of drift must be absorbed by the local search."""
    reader = make_reader(render_display(28.3))

    shifted = render_display(28.3, origin=(1402, 302))
    result = reader.read(shifted)

    assert result["raw_temperature_c"] == pytest.approx(28.3, abs=0.05)


def test_blank_frame_is_reported_as_invalid():
    """No display means no reading — not a confidently wrong one."""
    reader = make_reader(render_display(25.0))

    blank = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    result = reader.read(blank)

    assert not result["raw_valid"]


# ---------------------------------------------------------------------------
# Full-frame locator
# ---------------------------------------------------------------------------

def test_locator_finds_the_display_in_a_full_frame(tmp_path):
    """The one-off search must find the panel without being told where it is."""
    from larvatracker.lcd_temperature import locate_display

    value, origin = 24.6, (1300, 260)
    frame = render_display(value, origin=origin)

    path = str(tmp_path / "synthetic.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (frame.shape[1], frame.shape[0]), True
    )
    for _ in range(20):
        writer.write(frame)
    writer.release()

    reader, calibration = locate_display(path, TemperatureConfig(), verbose=False)

    assert calibration["raw_temperature_c"] == pytest.approx(value, abs=0.05)
    # The anchor should land on the drawn origin, up to the search stride.
    assert abs(calibration["anchor_x_px"] - origin[0]) <= 8
    assert abs(calibration["anchor_y_px"] - origin[1]) <= 8


def test_locator_refuses_a_frame_without_a_display(tmp_path):
    """Returning a best guess on an LCD-free video would be worse than failing."""
    from larvatracker.lcd_temperature import locate_display

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)

    path = str(tmp_path / "no_display.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480), True)
    for _ in range(20):
        writer.write(frame)
    writer.release()

    with pytest.raises(RuntimeError):
        locate_display(path, TemperatureConfig(), verbose=False)


# ---------------------------------------------------------------------------
# Temporal filtering
# ---------------------------------------------------------------------------

def records_from(values, valid=None):
    valid = valid if valid is not None else [True] * len(values)
    return [
        {
            "raw_temperature_c": v,
            "raw_valid": ok,
            "frame": i,
            "time_s": i / 30.0,
            "geometry": "compact",
            "confidence": 20.0,
            "shift_x": 0,
            "shift_y": 0,
            "segment_errors": 0,
            "anchor_x_px": 10,
            "anchor_y_px": 20,
            "scale": 1.0,
            "angle_deg": 0.0,
            "locator_score": 100.0,
        }
        for i, (v, ok) in enumerate(zip(values, valid))
    ]


def test_median_removes_a_single_frame_misread():
    """An LCD caught mid-refresh decodes to nonsense for one frame."""
    values = [25.0] * 20
    values[10] = 88.8

    filtered = filter_temperature_records(records_from(values))

    assert filtered[10] == pytest.approx(25.0)


def test_invalid_frames_are_recovered_from_neighbours():
    values = [25.0] * 20
    valid = [True] * 20
    valid[8:11] = [False] * 3

    filtered = filter_temperature_records(records_from(values, valid))

    assert np.isfinite(filtered[8:11]).all()


def test_long_dropouts_stay_missing():
    """A long unreadable stretch must not be invented."""
    config = TemperatureConfig(max_interpolation_gap_frames=5, median_window=1)

    values = [25.0] * 60
    valid = [True] * 60
    valid[20:50] = [False] * 30

    filtered = filter_temperature_records(records_from(values, valid), config)

    assert np.isnan(filtered[25:45]).all()


def test_fill_short_gaps_interpolates_linearly():
    values = np.array([10.0, np.nan, np.nan, np.nan, 14.0])

    filled = fill_short_gaps(values, max_gap=5)

    assert filled == pytest.approx([10.0, 11.0, 12.0, 13.0, 14.0])


def test_fill_short_gaps_leaves_edges_alone():
    """There is nothing to interpolate between at the start or the end."""
    values = np.array([np.nan, 10.0, 11.0, np.nan])

    filled = fill_short_gaps(values, max_gap=10)

    assert np.isnan(filled[0]) and np.isnan(filled[-1])


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_csv_labels_each_frame_with_how_its_value_arose(tmp_path):
    import pandas as pd

    values = [25.0] * 30
    values[5] = 88.8          # single misread -> corrected by the median
    valid = [True] * 30
    valid[15] = False         # rejected      -> recovered from neighbours

    path, summary = write_temperature_csv(
        records_from(values, valid), str(tmp_path), verbose=False
    )
    table = pd.read_csv(path)

    assert len(table) == 30
    assert table.loc[5, "status"] == "temporal_filter"
    assert table.loc[15, "status"] == "recovered_from_neighbors"
    assert table.loc[0, "status"] == "ok"
    assert summary["min_c"] == pytest.approx(25.0)
    assert summary["missing"] == 0


def test_csv_leaves_missing_values_empty(tmp_path):
    import pandas as pd

    config = TemperatureConfig(max_interpolation_gap_frames=2, median_window=1)
    valid = [False] * 20

    path, summary = write_temperature_csv(
        records_from([25.0] * 20, valid), str(tmp_path), config, verbose=False
    )
    table = pd.read_csv(path)

    assert table["temperature_c"].isna().all()
    assert summary["missing"] == 20


# ---------------------------------------------------------------------------
# Integration: one decode pass produces both ROI videos and the temperature
# ---------------------------------------------------------------------------

def test_single_pass_reads_a_temperature_ramp_and_cuts_roi_videos(tmp_path):
    """The whole non-SAM half of step 1, on a synthetic recording.

    A video is built with four droplets and an LCD showing a rising
    temperature. The ROI videos and the temperature trace must both come out
    of a single pass over the frames.
    """
    import pandas as pd

    from larvatracker.droplets import find_droplets
    from larvatracker.lcd_temperature import locate_display
    from larvatracker.roi_videos import write_droplet_videos

    height, width, n_frames = 1080, 1920, 60
    centres = [(300, 700), (500, 700), (700, 700), (900, 700)]
    radius = 70

    truth = np.round(np.linspace(22.0, 25.0, n_frames), 1)
    video_path = str(tmp_path / "rec.mp4")

    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height), True
    )
    for t in range(n_frames):
        frame = render_display(float(truth[t]), frame_size=(height, width), origin=(1400, 200))
        for i, (cx, cy) in enumerate(centres):
            cv2.circle(frame, (cx, cy), radius, (200, 200, 200), -1)
            angle = 2 * np.pi * t / 40 + i
            cv2.circle(
                frame,
                (int(cx + 25 * np.cos(angle)), int(cy + 25 * np.sin(angle))),
                10,
                (20, 20, 20),
                -1,
            )
        writer.write(frame)
    writer.release()

    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[:height, :width]
    for cx, cy in centres:
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

    droplets, _ = find_droplets(mask, min_area_px=500, padding_px=8)
    reader, _ = locate_display(video_path, verbose=False)

    records = []

    def on_frame(frame_idx, time_s, frame):
        record = reader.read(frame)
        record["frame"] = frame_idx
        record["time_s"] = time_s
        records.append(record)

    result = write_droplet_videos(
        video_path,
        droplets,
        str(tmp_path / "droplet_videos"),
        frame_callback=on_frame,
        progress_every=0,
    )

    assert len(droplets) == len(centres)
    assert result["frames"] == n_frames
    assert len(records) == n_frames

    # Every raw reading must match the rendered value exactly.
    raw = np.array([r["raw_temperature_c"] for r in records])
    assert np.array_equal(raw, truth)

    path, summary = write_temperature_csv(records, str(tmp_path), verbose=False)
    table = pd.read_csv(path)

    assert len(table) == n_frames
    assert summary["missing"] == 0

    # On a monotonic ramp the centred median can move a value by one display
    # step, which is 0.1 °C — but no further.
    assert np.abs(table["temperature_c"].to_numpy() - truth).max() <= 0.1001

    # Timestamps must be strictly increasing for the framewise export to be usable.
    assert (np.diff(table["time_s"].to_numpy()) > 0).all()


# ---------------------------------------------------------------------------
# Calibrating against a known reading
# ---------------------------------------------------------------------------

def write_synthetic_video(path, value, n_frames=20, origin=(1400, 200), frame_size=(1080, 1920)):
    height, width = frame_size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height), True
    )
    frame = render_display(value, frame_size=frame_size, origin=origin)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return str(path)


def test_calibration_recovers_the_known_reading():
    """The whole point: the calibration must reproduce what is on the screen.

    Calibrated on the rendered frame directly. Writing it to mp4 first would
    test the codec's treatment of thin dark strokes, which is not what this is
    about — real displays are far higher contrast than a drawn one.
    """
    from larvatracker.lcd_temperature import calibrate_display

    frame = render_display(26.4, origin=(1400, 200))

    calibration = calibrate_display("synthetic", known_temperature=26.4, frame=frame)

    assert calibration["decoded_temperature_c"] == pytest.approx(26.4, abs=0.05)
    assert calibration["segment_errors"] == 0
    assert calibration["geometry"] in GEOMETRY_PROFILES


def test_calibration_refuses_a_wrong_known_value():
    """Told the wrong number, it must fail rather than find something close."""
    from larvatracker.lcd_temperature import calibrate_display

    frame = render_display(26.4, origin=(1400, 200))

    with pytest.raises(RuntimeError):
        calibrate_display("synthetic", known_temperature=31.8, frame=frame)


def test_calibrated_reader_reads_every_frame():
    """A calibration that decodes one frame must keep working on the rest."""
    from larvatracker.lcd_temperature import calibrate_display, reader_at

    frame = render_display(24.8, origin=(1400, 200))
    calibration = calibrate_display("synthetic", known_temperature=24.8, frame=frame)

    reader = reader_at(
        frame.shape,
        GEOMETRY_PROFILES[calibration["geometry"]],
        calibration["scale"],
        tuple(calibration["anchor"]),
    )
    reader.calibrate(frame)

    for value in (24.8, 27.3, 31.5, 38.9):
        result = reader.read(render_display(value, origin=(1400, 200)))
        assert result["raw_temperature_c"] == pytest.approx(value, abs=0.05)
        assert result["raw_valid"]


def test_calibration_round_trips_through_json(tmp_path):
    from larvatracker.lcd_temperature import (
        calibrate_display,
        load_calibration,
        save_calibration,
    )

    frame = render_display(25.5, origin=(1400, 200))
    calibration = calibrate_display("synthetic", known_temperature=25.5, frame=frame)

    saved = save_calibration(calibration, str(tmp_path / "cal.json"))
    restored = load_calibration(saved)

    assert restored["geometry"] == calibration["geometry"]
    assert restored["anchor"] == calibration["anchor"]
    assert restored["scale"] == pytest.approx(calibration["scale"])


def test_digit_boxes_reject_the_display_frame():
    """A dark stripe spanning the ROI is the glass edge, not a digit.

    This is the component that produced a wrong calibration in practice: it is
    darker and larger than any digit, so anything ranking by size picks it.
    """
    from larvatracker.lcd_temperature import find_digit_boxes

    roi = (1350, 100, 1800, 400)
    roi_height = roi[3] - roi[1]

    clean = render_display(23.7, origin=(1400, 200))
    without_edge = find_digit_boxes(clean, roi)

    frame = clean.copy()
    # A bezel stripe running the full height of the ROI, as on the real display.
    cv2.rectangle(frame, (1700, 100), (1720, 400), (10, 10, 10), -1)
    with_edge = find_digit_boxes(frame, roi)

    # The stripe must not appear, and must not displace the digits either.
    assert all(box[3] < 0.9 * roi_height for box in with_edge)
    assert not any(box[0] >= 1700 for box in with_edge)
    assert len(with_edge) == len(without_edge)


def test_read_temperature_track_covers_every_frame(tmp_path):
    """Recovering a failed readout must not need the segmentation re-run."""
    from larvatracker.lcd_temperature import calibrate_display, read_temperature_track

    frame = render_display(27.3, origin=(1400, 200))
    calibration = calibrate_display("synthetic", known_temperature=27.3, frame=frame)

    path = write_synthetic_video(tmp_path / "cal.mp4", 27.3, n_frames=25)
    records = read_temperature_track(path, calibration, progress_every=0)

    assert len(records) == 25
    assert [r["frame"] for r in records] == list(range(25))
    # Timestamps must increase, or the framewise join in step 8 misaligns.
    assert all(
        records[i + 1]["time_s"] > records[i]["time_s"] for i in range(len(records) - 1)
    )
