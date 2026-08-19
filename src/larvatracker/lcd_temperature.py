"""Reading the chamber temperature off a seven-segment LCD in the video.

The heating stage has a small LCD thermometer in view of the camera. Rather
than logging the temperature separately, it is read back out of the recording,
which guarantees that every frame's temperature is the one that frame actually
saw — no clock drift between two devices.

The reader works in two stages, because a per-frame full-frame search would be
far too slow:

1. **Locate once.** A median of the first frames is scanned over the whole
   image at several scales and rotations. Every candidate position is scored
   by how well the sampled points decode as three seven-segment digits, plus
   how much the surrounding pixels look like a cyan LCD screen.
2. **Read cheaply.** Once located, only a small ROI around the display is
   touched per frame, with a few pixels of local search to absorb camera
   shake.

No absolute display coordinates are hard-coded: the geometry profiles describe
the *shape* of the digits relative to each other, so the same code handles
recordings where the LCD sits in a different corner or at a different angle.

Digit values are decoded from segment darkness rather than by OCR. A
morphological closing estimates the local background, and the difference to
the actual pixel value gives how dark each segment sample is. That is robust
to the LCD's own backlight and to overall exposure changes.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Segment order is a, b, c, d, e, f, g — the standard seven-segment naming,
# starting at the top bar and running clockwise, with g the middle bar.
DIGIT_PATTERNS: dict[int, tuple[int, ...]] = {
    0: (1, 1, 1, 1, 1, 1, 0),
    1: (0, 1, 1, 0, 0, 0, 0),
    2: (1, 1, 0, 1, 1, 0, 1),
    3: (1, 1, 1, 1, 0, 0, 1),
    4: (0, 1, 1, 0, 0, 1, 1),
    5: (1, 0, 1, 1, 0, 1, 1),
    6: (1, 0, 1, 1, 1, 1, 1),
    7: (1, 1, 1, 0, 0, 0, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1),
}

PATTERN_ARRAY = np.asarray([DIGIT_PATTERNS[d] for d in range(10)], dtype=np.float32)

# The geometries below are expressed for a 1920x1080 frame and rescaled to the
# actual resolution at run time.
REFERENCE_FRAME_SIZE = (1920, 1080)

# Two digit geometries cover the views present in the recordings: a compact,
# front-facing display and a larger one seen at an angle. Each profile gives
# the origin of the three digits and, within one digit, the sample point for
# each of the seven segments.
GEOMETRY_PROFILES: dict[str, dict[str, Any]] = {
    "compact": {
        "digit_origins_ref": ((0, 0), (51, 0), (102, 0)),
        "segment_points_ref": (
            (19, 8), (35, 29), (31, 70), (17, 91),
            (1, 70), (5, 29), (17, 49),
        ),
        "kernel_size_ref": 17,
    },
    "large_tilted": {
        "digit_origins_ref": ((0, 0), (62, 7), (124, 14)),
        "segment_points_ref": (
            (19, 13), (42, 35), (38, 79), (17, 107),
            (2, 79), (6, 35), (18, 60),
        ),
        "kernel_size_ref": 21,
    },
}


@dataclass
class TemperatureConfig:
    """Settings for locating and reading the LCD.

    Attributes
    ----------
    min_c, max_c:
        Plausible range for a decoded reading. Values outside are treated as
        misreads.
    locator_scales, locator_angles_deg:
        Grid the full-frame search covers. More entries make the search more
        robust and proportionally slower; it runs once per video.
    locator_stride_px:
        Step of the coarse search grid. The result is refined locally
        afterwards, so this does not need to be 1.
    locator_min_score:
        Below this the located candidate is rejected as not-an-LCD. Without the
        floor the search always returns its best guess, however poor.
    expected_start_min_c, expected_start_max_c:
        Recordings begin near room temperature. Penalising candidates that
        decode to something else at the start suppresses false positives from
        droplets and labels that happen to look digit-shaped.
    saturation_weight, cyan_weight, texture_weight:
        Score weights for the LCD's saturated cyan screen and its flat, low
        texture surface — the properties that distinguish it from printed text.
    alignment_search_px, local_refine_px:
        Radius of the one-off calibration search and of the per-frame search
        that absorbs camera shake.
    calibration_frames:
        Number of opening frames the median calibration image is built from.
    segment_threshold:
        Darkness above which a segment counts as lit.
    min_confidence:
        Minimum mean decoding margin for a frame's reading to count as valid.
    median_window:
        Width of the centred temporal median. The LCD refreshes asynchronously
        to the camera, so single frames catch it mid-transition; seven frames
        is about 0.23 s at 30 fps.
    max_interpolation_gap_frames:
        Longest run of unreadable frames that is bridged by interpolation.
    """

    min_c: float = 0.0
    max_c: float = 60.0

    locator_scales: tuple[float, ...] = (0.85, 1.00, 1.15)
    locator_angles_deg: tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0)
    locator_stride_px: int = 5
    locator_min_score: float = 40.0

    expected_start_min_c: float = 15.0
    expected_start_max_c: float = 45.0

    saturation_weight: float = 45.0
    cyan_weight: float = 0.45
    texture_weight: float = 0.35

    alignment_search_px: int = 10
    local_refine_px: int = 3
    calibration_frames: int = 15

    segment_threshold: float = 15.0
    min_confidence: float = 8.0

    median_window: int = 7
    max_interpolation_gap_frames: int = 15

    geometries: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(GEOMETRY_PROFILES)
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _odd_size(value: float, minimum: int = 3) -> int:
    """Round to an odd integer — OpenCV kernels need a defined centre."""
    size = max(minimum, int(round(value)))
    return size + 1 if size % 2 == 0 else size


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def geometry_offsets(profile: dict, scale: float, angle_deg: float) -> np.ndarray:
    """Sample points of all three digits, scaled and rotated.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(3, 7, 2)``: for each digit, the (x, y) offset
        of each segment sample relative to the display anchor.
    """
    origins = profile["digit_origins_ref"]
    points = profile["segment_points_ref"]

    base = np.asarray(
        [[(ox + px, oy + py) for px, py in points] for ox, oy in origins],
        dtype=np.float32,
    )

    angle = math.radians(angle_deg)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )

    return np.rint((base @ rotation.T) * scale).astype(np.int32)


def _darkness(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """How much darker each pixel is than its local background.

    A morphological closing fills in the thin dark segments, giving an estimate
    of the display background; subtracting the real image leaves the segments.
    This removes any dependence on absolute brightness.
    """
    gray_f = gray.astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    background = cv2.morphologyEx(gray_f, cv2.MORPH_CLOSE, kernel)
    return np.maximum(background - gray_f, 0.0)


# ---------------------------------------------------------------------------
# Full-frame search
# ---------------------------------------------------------------------------

def median_calibration_frame(video_path: str, n_frames: int = 15) -> np.ndarray:
    """Median of the opening frames.

    The median removes the larvae, which move, while keeping the static scene
    including the display. It also averages out sensor noise, which matters
    because the search decides on a few dozen pixel samples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path!r} for LCD search.")

    frames = []
    try:
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path!r} for LCD search.")

    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _scan_candidates(
    smooth_darkness: np.ndarray,
    saturation: np.ndarray,
    cyan: np.ndarray,
    texture: np.ndarray,
    offsets: np.ndarray,
    config: TemperatureConfig,
    geometry: str,
    scale: float,
    angle_deg: float,
    kernel_size: int,
    sample_size: int,
) -> dict | None:
    """Score every grid position for one geometry, scale and angle.

    All candidate positions are evaluated at once with array indexing rather
    than in a Python loop — a full frame at stride 5 is tens of thousands of
    positions per (scale, angle) combination.
    """
    height, width = smooth_darkness.shape[:2]
    radius = sample_size // 2

    x_start = max(0, -int(offsets[:, :, 0].min()) + radius)
    y_start = max(0, -int(offsets[:, :, 1].min()) + radius)
    x_stop = width - int(offsets[:, :, 0].max()) - radius
    y_stop = height - int(offsets[:, :, 1].max()) - radius

    if x_stop <= x_start or y_stop <= y_start:
        return None

    xs = np.arange(x_start, x_stop, config.locator_stride_px, dtype=np.int32)
    ys = np.arange(y_start, y_stop, config.locator_stride_px, dtype=np.int32)
    grid_x, grid_y = (a.ravel() for a in np.meshgrid(xs, ys))

    n = len(grid_x)
    if n == 0:
        return None

    total_errors = np.zeros(n, dtype=np.int16)
    confidence = np.zeros(n, dtype=np.float32)
    saturation_score = np.zeros(n, dtype=np.float32)
    cyan_score = np.zeros(n, dtype=np.float32)
    texture_score = np.zeros(n, dtype=np.float32)
    digits = []

    rows = np.arange(n)

    for digit_index in range(3):
        digit_offsets = offsets[digit_index]

        def sample(image):
            return np.stack(
                [image[grid_y + int(oy), grid_x + int(ox)] for ox, oy in digit_offsets],
                axis=1,
            )

        values = sample(smooth_darkness)

        # Signed margin to the threshold: positive when the segment is on the
        # correct side for that digit pattern.
        margins = np.where(
            PATTERN_ARRAY[None, :, :] > 0,
            values[:, None, :] - config.segment_threshold,
            config.segment_threshold - values[:, None, :],
        )
        errors = (margins < 0).sum(axis=2)
        scores = margins.mean(axis=2)

        best = np.argmax(scores - 100.0 * errors, axis=1)
        digits.append(best)

        total_errors += errors[rows, best].astype(np.int16)
        confidence += scores[rows, best]
        saturation_score += sample(saturation).mean(axis=1)
        cyan_score += sample(cyan).mean(axis=1)
        texture_score += sample(texture).mean(axis=1)

    confidence /= 3.0
    saturation_score /= 3.0
    cyan_score /= 3.0
    texture_score /= 3.0

    temperature = 10.0 * digits[0] + digits[1] + digits[2] / 10.0
    implausible = (temperature < config.expected_start_min_c) | (
        temperature > config.expected_start_max_c
    )

    score = (
        confidence
        - 100.0 * total_errors
        - 1000.0 * implausible.astype(np.float32)
        + config.saturation_weight * saturation_score
        + config.cyan_weight * cyan_score
        - config.texture_weight * texture_score
    )

    best_index = int(np.argmax(score))

    return {
        "geometry": geometry,
        "effective_scale": float(scale),
        "angle_deg": float(angle_deg),
        "anchor_x": int(grid_x[best_index]),
        "anchor_y": int(grid_y[best_index]),
        "raw_temperature_c": round(float(temperature[best_index]), 1),
        "confidence": float(confidence[best_index]),
        "segment_errors": int(total_errors[best_index]),
        "locator_score": float(score[best_index]),
        "kernel_size": int(kernel_size),
        "sample_size": int(sample_size),
        "sample_offsets": offsets,
    }


def locate_display(
    video_path: str,
    config: TemperatureConfig | None = None,
    verbose: bool = True,
) -> tuple["LcdTemperatureReader", dict]:
    """Find the LCD anywhere in the frame and return a reader for it.

    Raises
    ------
    RuntimeError
        If no candidate clears ``locator_min_score``. Failing loudly is
        deliberate: a silently mislocated display would produce a plausible
        looking but entirely wrong temperature trace.
    """
    cfg = config or TemperatureConfig()

    median_frame = median_calibration_frame(video_path, cfg.calibration_frames)
    height, width = median_frame.shape[:2]

    ref_width, ref_height = REFERENCE_FRAME_SIZE
    resolution_scale = min(width / ref_width, height / ref_height)

    gray = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(median_frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    # Blue minus red: high on the cyan LCD screen, near zero on neutral scene.
    cyan = median_frame[:, :, 0].astype(np.float32) - median_frame[:, :, 2].astype(np.float32)
    texture = np.abs(gray - cv2.boxFilter(gray, -1, (7, 7), normalize=True))

    best: dict | None = None

    for geometry, profile in cfg.geometries.items():
        for relative_scale in cfg.locator_scales:
            scale = resolution_scale * float(relative_scale)

            kernel_size = _odd_size(profile["kernel_size_ref"] * scale, minimum=5)
            sample_size = _odd_size(5.0 * scale, minimum=3)

            smooth = cv2.boxFilter(
                _darkness(gray, kernel_size), -1, (sample_size, sample_size), normalize=True
            )

            for angle_deg in cfg.locator_angles_deg:
                candidate = _scan_candidates(
                    smooth_darkness=smooth,
                    saturation=saturation,
                    cyan=cyan,
                    texture=texture,
                    offsets=geometry_offsets(profile, scale, float(angle_deg)),
                    config=cfg,
                    geometry=geometry,
                    scale=scale,
                    angle_deg=float(angle_deg),
                    kernel_size=kernel_size,
                    sample_size=sample_size,
                )

                if candidate is not None and (
                    best is None or candidate["locator_score"] > best["locator_score"]
                ):
                    best = candidate

    if best is None:
        raise RuntimeError("Full-frame LCD search found no candidate at all.")

    if best["locator_score"] < cfg.locator_min_score:
        raise RuntimeError(
            f"LCD candidate too uncertain: score {best['locator_score']:.1f} < "
            f"{cfg.locator_min_score:.1f}. Check that the display is in frame, "
            "or relax locator_min_score."
        )

    reader = LcdTemperatureReader(median_frame.shape, best, cfg)
    calibration = reader.calibrate(median_frame)

    if verbose:
        print(
            f"LCD found: x={calibration['anchor_x_px']} y={calibration['anchor_y_px']} "
            f"geometry={reader.geometry} scale={reader.scale:.2f} "
            f"angle={reader.angle_deg:+.0f}° score={reader.locator_score:.1f} "
            f"start={calibration['raw_temperature_c']:.1f} °C"
        )

    return reader, calibration


# ---------------------------------------------------------------------------
# Per-frame reader
# ---------------------------------------------------------------------------

class LcdTemperatureReader:
    """Read a three-digit temperature from an LCD at a known position.

    Only a small ROI around the display is processed per frame. Within that
    ROI the sample grid is shifted by a few pixels to find the best alignment,
    which absorbs slow camera drift without another full search.
    """

    def __init__(
        self,
        frame_shape,
        candidate: dict,
        config: TemperatureConfig | None = None,
    ):
        self.config = config or TemperatureConfig()

        height, width = int(frame_shape[0]), int(frame_shape[1])

        self.geometry = str(candidate["geometry"])
        self.scale = float(candidate["effective_scale"])
        self.angle_deg = float(candidate["angle_deg"])
        self.anchor_x = int(candidate["anchor_x"])
        self.anchor_y = int(candidate["anchor_y"])
        self.locator_score = float(candidate["locator_score"])

        offsets = np.asarray(candidate["sample_offsets"], dtype=np.int32)
        absolute = offsets.copy()
        absolute[:, :, 0] += self.anchor_x
        absolute[:, :, 1] += self.anchor_y

        kernel_size = int(candidate["kernel_size"])
        sample_size = int(candidate["sample_size"])

        # The ROI needs room for the closing kernel and for the search shift,
        # otherwise the background estimate is biased at the border.
        margin = max(
            20, kernel_size + self.config.alignment_search_px + self.config.local_refine_px
        )

        self.roi_x0 = _clamp(int(absolute[:, :, 0].min()) - margin, 0, width - 1)
        self.roi_y0 = _clamp(int(absolute[:, :, 1].min()) - margin, 0, height - 1)
        self.roi_x1 = _clamp(int(absolute[:, :, 0].max()) + margin + 1, 1, width)
        self.roi_y1 = _clamp(int(absolute[:, :, 1].max()) + margin + 1, 1, height)

        self.sample_points = absolute.copy()
        self.sample_points[:, :, 0] -= self.roi_x0
        self.sample_points[:, :, 1] -= self.roi_y0

        self.kernel_size = kernel_size
        self.sample_radius = max(1, sample_size // 2)
        self.search_radius = max(1, self.config.alignment_search_px)
        self.refine_radius = max(0, self.config.local_refine_px)

        self.shift_x = 0
        self.shift_y = 0
        self.calibrated = False

    # -- internals ---------------------------------------------------------

    def _roi_darkness(self, frame_bgr: np.ndarray) -> np.ndarray:
        roi = frame_bgr[self.roi_y0:self.roi_y1, self.roi_x0:self.roi_x1]
        if roi.size == 0:
            raise RuntimeError("The located LCD ROI is empty.")
        return _darkness(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), self.kernel_size)

    def _decode_digit(self, values: np.ndarray) -> tuple[int, float, int]:
        """Pick the digit whose segment pattern fits the sampled darkness best.

        Ranked first by how many segments are on the wrong side of the
        threshold, then by the mean margin, so a clean fit always beats a
        marginally better but inconsistent one.
        """
        threshold = self.config.segment_threshold

        margins = np.where(
            PATTERN_ARRAY > 0, values[None, :] - threshold, threshold - values[None, :]
        )
        errors = (margins < 0).sum(axis=1)
        scores = margins.mean(axis=1)

        best = min(range(10), key=lambda d: (int(errors[d]), -float(scores[d])))
        return int(best), float(scores[best]), int(errors[best])

    def _decode_at(self, darkness: np.ndarray, dx: int, dy: int) -> dict | None:
        height, width = darkness.shape[:2]
        radius = self.sample_radius

        digits, scores, errors = [], [], []

        for digit_points in self.sample_points:
            values = []
            for point_x, point_y in digit_points:
                cx, cy = int(point_x) + dx, int(point_y) + dy
                x0, x1 = cx - radius, cx + radius + 1
                y0, y1 = cy - radius, cy + radius + 1

                if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    return None

                values.append(float(darkness[y0:y1, x0:x1].mean()))

            digit, score, error = self._decode_digit(np.asarray(values, dtype=np.float32))
            digits.append(digit)
            scores.append(score)
            errors.append(error)

        # Three digits read as XX.X — the LCD has a fixed decimal position.
        value = 10.0 * digits[0] + digits[1] + digits[2] / 10.0

        return {
            "raw_temperature_c": round(float(value), 1),
            "digits": tuple(digits),
            "geometry": self.geometry,
            "confidence": float(np.mean(scores)),
            "segment_errors": int(sum(errors)),
            "shift_x": int(dx),
            "shift_y": int(dy),
            "scale": self.scale,
            "angle_deg": self.angle_deg,
            "locator_score": self.locator_score,
            "anchor_x_px": self.anchor_x + int(dx),
            "anchor_y_px": self.anchor_y + int(dy),
        }

    def _best_alignment(
        self,
        darkness: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
        position_penalty: float,
    ) -> dict:
        """Search a small shift window for the cleanest decoding.

        ``position_penalty`` biases the result towards the previous alignment,
        so a tie is broken by staying put rather than drifting frame to frame.
        """
        best, best_key = None, None

        for dy in range(center_y - radius, center_y + radius + 1):
            for dx in range(center_x - radius, center_x + radius + 1):
                result = self._decode_at(darkness, dx, dy)
                if result is None:
                    continue

                distance_sq = (dx - center_x) ** 2 + (dy - center_y) ** 2
                key = (
                    -result["segment_errors"],
                    result["confidence"] - position_penalty * distance_sq,
                )

                if best_key is None or key > best_key:
                    best_key, best = key, result

        if best is None:
            raise RuntimeError("Could not align the LCD sample grid locally.")

        return best

    # -- public ------------------------------------------------------------

    def calibrate(self, median_frame_bgr: np.ndarray) -> dict:
        """Fix the sample-grid shift once, using the median frame."""
        result = self._best_alignment(
            self._roi_darkness(median_frame_bgr),
            center_x=0,
            center_y=0,
            radius=self.search_radius,
            position_penalty=0.03,
        )

        self.shift_x = int(result["shift_x"])
        self.shift_y = int(result["shift_y"])
        self.calibrated = True
        return result

    def read(self, frame_bgr: np.ndarray) -> dict:
        """Read one frame.

        The returned dict always contains a value; ``raw_valid`` says whether
        it should be trusted. Invalid readings are kept rather than dropped so
        the temporal filter can see where the gaps are.
        """
        result = self._best_alignment(
            self._roi_darkness(frame_bgr),
            center_x=self.shift_x,
            center_y=self.shift_y,
            radius=self.refine_radius,
            position_penalty=0.10,
        )

        value = float(result["raw_temperature_c"])
        result["raw_valid"] = bool(
            result["segment_errors"] == 0
            and result["confidence"] >= self.config.min_confidence
            and self.config.min_c <= value <= self.config.max_c
        )
        return result

    def debug_image(self, frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        """Annotated crop of the display, for checking the decoding by eye.

        Green circles mark segments the decoder considered lit, red ones the
        segments it considered dark. If the reading is wrong, this image shows
        immediately whether the grid is misaligned or the threshold is off.
        """
        crop = frame_bgr[self.roi_y0:self.roi_y1, self.roi_x0:self.roi_x1].copy()

        digits = tuple(int(v) for v in result["digits"])
        dx, dy = int(result["shift_x"]), int(result["shift_y"])

        for digit_index, digit_points in enumerate(self.sample_points):
            pattern = DIGIT_PATTERNS[digits[digit_index]]
            for segment_index, (point_x, point_y) in enumerate(digit_points):
                colour = (0, 255, 0) if pattern[segment_index] else (0, 0, 255)
                cv2.circle(
                    crop,
                    (int(point_x) + dx, int(point_y) + dy),
                    max(2, self.sample_radius),
                    colour,
                    1,
                )

        label = (
            f"{self.geometry}: {result['raw_temperature_c']:.1f} C  "
            f"score={self.locator_score:.1f}  scale={self.scale:.2f}  "
            f"angle={self.angle_deg:+.0f}"
        )
        cv2.rectangle(crop, (0, crop.shape[0] - 24), (crop.shape[1], crop.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            crop, label, (5, crop.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Temporal filtering and export
# ---------------------------------------------------------------------------

def fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate short runs of NaN, leave long ones missing.

    Bridging a few unreadable frames is safe — the temperature ramp is smooth.
    Interpolating across a long dropout would invent a trajectory, so runs
    longer than ``max_gap`` stay NaN. Leading and trailing gaps are never
    filled, since there is nothing to interpolate between.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    n = len(out)
    i = 0

    while i < n:
        if np.isfinite(out[i]):
            i += 1
            continue

        start = i
        while i < n and not np.isfinite(out[i]):
            i += 1
        end = i

        gap = end - start
        if start > 0 and end < n and gap <= max_gap:
            out[start:end] = np.linspace(out[start - 1], out[end], gap + 2)[1:-1]

    return out


def filter_temperature_records(
    records: list[dict],
    config: TemperatureConfig | None = None,
) -> np.ndarray:
    """Centred temporal median over the valid readings, then gap filling.

    The median rather than a mean is deliberate: the failure mode of an LCD
    read is a single frame decoding to a completely different number, not a
    small perturbation. A mean would smear that across the window; the median
    ignores it.
    """
    cfg = config or TemperatureConfig()

    raw = np.asarray(
        [
            float(r["raw_temperature_c"]) if bool(r["raw_valid"]) else np.nan
            for r in records
        ],
        dtype=np.float64,
    )

    filtered = np.full(len(raw), np.nan, dtype=np.float64)
    radius = max(0, cfg.median_window // 2)

    for index in range(len(raw)):
        window = raw[max(0, index - radius): index + radius + 1]
        window = window[np.isfinite(window)]
        if window.size:
            filtered[index] = float(np.median(window))

    filtered = fill_short_gaps(filtered, cfg.max_interpolation_gap_frames)

    finite = np.isfinite(filtered)
    filtered[finite] = np.round(filtered[finite], 1)
    return filtered


TEMPERATURE_CSV_COLUMNS = [
    "frame",
    "time_s",
    "geometry",
    "temperature_c",
    "raw_temperature_c",
    "confidence",
    "raw_valid",
    "status",
    "shift_x_px",
    "shift_y_px",
    "segment_errors",
    "lcd_anchor_x_px",
    "lcd_anchor_y_px",
    "lcd_scale",
    "lcd_angle_deg",
    "lcd_locator_score",
]


def write_temperature_csv(
    records: list[dict],
    out_dir: str,
    config: TemperatureConfig | None = None,
    filename: str = "temperature.csv",
    verbose: bool = True,
) -> tuple[str, dict]:
    """Write the per-frame temperature table.

    Both the filtered and the raw reading are kept, plus a ``status`` column
    saying how the filtered value came about:

    ``ok``
        the raw reading was valid and the filter agreed with it
    ``temporal_filter``
        the raw reading was valid but the median moved it by 0.3 °C or more,
        typically an LCD caught mid-refresh
    ``recovered_from_neighbors``
        the raw reading was rejected and the value comes from surrounding
        frames
    ``missing``
        no value could be established

    Downstream code should use ``temperature_c`` and treat an empty cell as
    missing; ``raw_temperature_c`` is kept for auditing.
    """
    cfg = config or TemperatureConfig()
    filtered = filter_temperature_records(records, cfg)

    csv_path = os.path.join(out_dir, filename)
    counts = {"ok": 0, "temporal_filter": 0, "recovered_from_neighbors": 0, "missing": 0}

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPERATURE_CSV_COLUMNS)
        writer.writeheader()

        for record, value in zip(records, filtered):
            raw_value = float(record["raw_temperature_c"])
            raw_valid = bool(record["raw_valid"])
            has_value = bool(np.isfinite(value))

            if not has_value:
                status = "missing"
            elif not raw_valid:
                status = "recovered_from_neighbors"
            elif abs(raw_value - float(value)) >= 0.3:
                status = "temporal_filter"
            else:
                status = "ok"

            counts[status] += 1

            writer.writerow(
                {
                    "frame": int(record["frame"]),
                    "time_s": f"{float(record['time_s']):.6f}",
                    "geometry": str(record["geometry"]),
                    "temperature_c": f"{float(value):.1f}" if has_value else "",
                    "raw_temperature_c": f"{raw_value:.1f}",
                    "confidence": f"{float(record['confidence']):.3f}",
                    "raw_valid": int(raw_valid),
                    "status": status,
                    "shift_x_px": int(record["shift_x"]),
                    "shift_y_px": int(record["shift_y"]),
                    "segment_errors": int(record["segment_errors"]),
                    "lcd_anchor_x_px": int(record["anchor_x_px"]),
                    "lcd_anchor_y_px": int(record["anchor_y_px"]),
                    "lcd_scale": f"{float(record['scale']):.4f}",
                    "lcd_angle_deg": f"{float(record['angle_deg']):.2f}",
                    "lcd_locator_score": f"{float(record['locator_score']):.3f}",
                }
            )

    finite = filtered[np.isfinite(filtered)]
    summary = {
        "path": csv_path,
        "frames": len(records),
        "min_c": float(finite.min()) if finite.size else None,
        "max_c": float(finite.max()) if finite.size else None,
        **counts,
    }

    if verbose:
        print(
            f"Temperature: {csv_path} — {counts['recovered_from_neighbors']} frame(s) "
            f"recovered, {counts['temporal_filter']} smoothed, {counts['missing']} missing"
        )

    return csv_path, summary


# ---------------------------------------------------------------------------
# Deriving the digit geometry from a recording
# ---------------------------------------------------------------------------

# Where the seven segments sit inside a digit cell, as fractions of the cell.
# Order is a, b, c, d, e, f, g. The samples sit slightly inside the segment
# rather than on the cell border, so a small misalignment still lands on the
# segment rather than on the background.
def find_display_panel(frame_bgr: np.ndarray, quantile: float = 0.995) -> tuple[int, int, int, int]:
    """Rough bounding box of the LCD panel, found by its colour.

    The screen is the one place in the scene that is both strongly saturated
    and distinctly cyan. This only has to be approximately right: it bounds
    where the digits are looked for, nothing more.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    cyan = frame_bgr[:, :, 0].astype(np.float32) - frame_bgr[:, :, 2].astype(np.float32)

    score = saturation * 45.0 + cyan * 0.45
    height, width = frame_bgr.shape[:2]

    # The panel is a flat, uniform colour, so a large share of its pixels carry
    # exactly the same score. A strict `>` against the quantile then excludes
    # every one of them and the mask comes out empty — the failure looks like
    # "no display" when in fact the display is the biggest thing in the image.
    # Comparing with `>=`, and stepping the quantile down if that still finds
    # nothing, avoids it.
    mask = None
    for level in (quantile, 0.99, 0.98, 0.95):
        candidate = (score >= np.quantile(score, level)).astype(np.uint8)
        if candidate.any():
            mask = candidate
            break

    if mask is None:
        return 0, 0, width, height

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_labels <= 1:
        return 0, 0, width, height

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[largest, :4]
    margin = int(0.3 * max(w, h))

    return (
        _clamp(int(x) - margin, 0, width - 1),
        _clamp(int(y) - margin, 0, height - 1),
        _clamp(int(x + w) + margin, 1, width),
        _clamp(int(y + h) + margin, 1, height),
    )


def find_digit_boxes(
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    kernel_size: int = 21,
    threshold: float = 12.0,
) -> list[tuple[int, int, int, int]]:
    """Digit-like dark shapes inside ``roi``, left to right.

    Three kinds of component have to be rejected, and each one caused a wrong
    result before it was:

    * the **glass edge** of the display, a dark stripe spanning the full height
      of the ROI;
    * the **degree symbol and the decimal point**, far smaller than a digit;
    * anything much shorter than the tallest group, which is scene clutter.

    Adjacent digits may still touch and come out as one component. That is
    fine here — only the left edge and the height of the group are used.
    """
    x0, y0, x1, y1 = roi
    roi_height = y1 - y0

    grey = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    mask = (_darkness(grey, kernel_size) > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return []

    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n_labels)]
    boxes = [b for b in boxes if b[3] < 0.9 * roi_height]
    if not boxes:
        return []

    tallest = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= 0.7 * tallest]

    return sorted([(b[0] + x0, b[1] + y0, b[2], b[3]) for b in boxes], key=lambda b: b[0])


def reader_at(
    frame_shape,
    profile: dict,
    scale: float,
    anchor: tuple[int, int],
    config: TemperatureConfig | None = None,
    name: str = "calibrated",
    search_px: int | None = None,
) -> LcdTemperatureReader:
    """Build a reader at a known geometry, scale and position — no search.

    Once a recording has been calibrated, the display's position is known, and
    re-running the full-frame search only reintroduces the chance of settling
    somewhere else. The search exists to *find* an unknown display; with the
    answer in hand, using it is both faster and safer. The per-frame local
    search still runs, so camera drift is still absorbed.
    """
    cfg = config or TemperatureConfig()

    candidate = {
        "geometry": name,
        "effective_scale": float(scale),
        "angle_deg": 0.0,
        "anchor_x": int(anchor[0]),
        "anchor_y": int(anchor[1]),
        "locator_score": float("nan"),
        "kernel_size": _odd_size(profile["kernel_size_ref"] * scale, minimum=5),
        "sample_size": _odd_size(5.0 * scale, minimum=3),
        "sample_offsets": geometry_offsets(profile, scale, 0.0),
    }

    reader = LcdTemperatureReader(frame_shape, candidate, cfg)
    if search_px is not None:
        reader.search_radius = int(search_px)

    return reader


def calibrate_display(
    video_path: str,
    known_temperature: float,
    config: TemperatureConfig | None = None,
    roi: tuple[int, int, int, int] | None = None,
    scales=None,
    profiles: dict | None = None,
    frame: np.ndarray | None = None,
) -> dict:
    """Find where and at what scale the display reads the value you can see.

    The full-frame search in :func:`locate_display` scans the whole image and
    scores candidates on how digit-like and how LCD-coloured they are. That is
    the right tool when nothing is known, but it can settle on the wrong spot,
    and then it reports a confident temperature that is simply not what the
    screen says.

    This routine removes the guesswork in two steps:

    1. the display region is found by colour and the digits by shape, which
       narrows the anchor to a window of a few pixels;
    2. the anchor and scale are then chosen by the only criterion that cannot
       be faked — the decoded value has to equal ``known_temperature``, the
       number visible on screen at the start of the recording.

    Parameters
    ----------
    known_temperature:
        What the display shows at the start, read off by eye. Required: without
        it there is nothing to verify against, and a confidently wrong
        calibration is worse than none.
    frame:
        Calibrate on this image instead of reading the video. Only useful for
        testing; in normal use the median of the opening frames is the right
        thing to look at.

    Returns
    -------
    dict
        ``geometry``, ``scale``, ``anchor``, plus what was measured and how
        cleanly it decoded. Feed it to :func:`reader_at`.

    Raises
    ------
    RuntimeError
        If no combination reproduces ``known_temperature``. Better than
        returning the closest miss.
    """
    cfg = config or TemperatureConfig()
    profiles = profiles or GEOMETRY_PROFILES

    if frame is None:
        frame = median_calibration_frame(video_path, cfg.calibration_frames)

    roi = tuple(int(v) for v in roi) if roi is not None else find_display_panel(frame)
    boxes = find_digit_boxes(frame, roi)

    if not boxes:
        raise RuntimeError(
            f"no digit-like shapes inside {roi}. Pass an explicit ROI around the "
            "display, or check that it is in frame at all."
        )

    left = min(b[0] for b in boxes)
    top = int(np.median([b[1] for b in boxes]))

    height, width = frame.shape[:2]
    ref_width, ref_height = REFERENCE_FRAME_SIZE
    resolution_scale = min(width / ref_width, height / ref_height)

    # The digit height is already measured, so the scale does not have to be
    # searched blind. Each profile's sample points span a known fraction of the
    # glyph — the points sit inside the segments, so the drawn digit is roughly
    # 1.2x that span — which pins the scale to within a few percent. Sweeping a
    # narrow band around that estimate instead of the whole plausible range
    # turns thousands of trial decodes into a few hundred.
    digit_height = float(np.median([b[3] for b in boxes]))

    best = None

    for name, profile in profiles.items():
        points = np.asarray(profile["segment_points_ref"], dtype=float)
        span = float(points[:, 1].max() - points[:, 1].min())

        if scales is None:
            estimate = digit_height / (span * 1.18)
            # 2 % steps: the decoding is sensitive at that level, and a coarser
            # grid can straddle the correct scale without ever landing on it.
            profile_scales = np.arange(0.88, 1.13, 0.02) * estimate
        else:
            profile_scales = np.asarray(scales, dtype=float)

        for scale in profile_scales:
            for anchor_x in range(left - 14, left + 7, 2):
                for anchor_y in range(top - 16, top + 17, 2):
                    reader = reader_at(
                        frame.shape, profile, float(scale), (anchor_x, anchor_y),
                        cfg, name=name, search_px=2,
                    )

                    try:
                        result = reader.calibrate(frame)
                    except Exception:  # noqa: BLE001 - grid point does not fit
                        continue

                    if abs(result["raw_temperature_c"] - known_temperature) >= 0.05:
                        continue
                    if result["segment_errors"] != 0:
                        continue

                    quality = float(result["confidence"])
                    if best is None or quality > best[0]:
                        best = (quality, name, float(scale), (anchor_x, anchor_y), result)

    if best is None:
        raise RuntimeError(
            f"no geometry, scale or position reproduced {known_temperature} °C. "
            "Check the value you read off the screen, widen the ROI, or add a "
            "geometry profile for this display."
        )

    confidence, name, scale, anchor, result = best

    return {
        "video": video_path,
        "geometry": name,
        "scale": round(scale, 4),
        "anchor": [int(anchor[0]), int(anchor[1])],
        "known_temperature_c": float(known_temperature),
        "decoded_temperature_c": float(result["raw_temperature_c"]),
        "confidence": round(confidence, 2),
        "segment_errors": int(result["segment_errors"]),
        "roi": list(roi),
        "digit_boxes": [list(b) for b in boxes],
        "digit_height": int(np.median([b[3] for b in boxes])),
        "frame_size": [int(width), int(height)],
    }


def measure_coverage(
    video_path: str,
    calibration: dict,
    config: TemperatureConfig | None = None,
    every: int = 5,
    max_frames: int | None = None,
    profiles: dict | None = None,
) -> dict:
    """How many frames a calibration actually reads, sampled across the video.

    A calibration that decodes the opening frame is necessary but not
    sufficient: the display drifts, the ramp changes every digit, and a grid
    that sits marginally off will start failing halfway through. This samples
    the whole recording rather than trusting the first frame.
    """
    cfg = config or TemperatureConfig()
    profiles = profiles or GEOMETRY_PROFILES

    frame = median_calibration_frame(video_path, cfg.calibration_frames)
    reader = reader_at(
        frame.shape,
        profiles[calibration["geometry"]],
        calibration["scale"],
        tuple(calibration["anchor"]),
        cfg,
        name=calibration["geometry"],
    )
    reader.calibrate(frame)

    cap = cv2.VideoCapture(str(video_path))
    values, valid, sampled, index = [], 0, 0, 0

    try:
        while True:
            ok, image = cap.read()
            if not ok or (max_frames is not None and index >= max_frames):
                break

            if index % every == 0:
                result = reader.read(image)
                values.append(result["raw_temperature_c"])
                valid += int(result["raw_valid"])
                sampled += 1

            index += 1
    finally:
        cap.release()

    values = np.asarray(values, dtype=float)

    return {
        "frames_sampled": sampled,
        "frames_valid": valid,
        "coverage": round(valid / sampled, 4) if sampled else 0.0,
        "first_value": float(values[0]) if values.size else None,
        "min_value": float(values.min()) if values.size else None,
        "max_value": float(values.max()) if values.size else None,
    }


def read_temperature_track(
    video_path: str,
    calibration: dict,
    config: TemperatureConfig | None = None,
    profiles: dict | None = None,
    progress_every: int = 500,
) -> list[dict]:
    """Read the display on every frame, using an established calibration.

    This exists so that a recording whose temperature failed can be recovered
    without redoing the segmentation. Step 1 reads the display inside the pass
    that cuts the ROI videos, which is efficient when both are wanted — but
    when only the temperature is missing, running SAM 3 again to get it would
    be absurd.
    """
    cfg = config or TemperatureConfig()
    profiles = profiles or GEOMETRY_PROFILES

    frame = median_calibration_frame(video_path, cfg.calibration_frames)
    reader = reader_at(
        frame.shape,
        profiles[calibration["geometry"]],
        calibration["scale"],
        tuple(calibration["anchor"]),
        cfg,
        name=calibration["geometry"],
    )
    reader.calibrate(frame)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    records, index, last_time = [], 0, -1.0

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break

            # Same timestamp rule as the ROI export, so the two line up.
            position = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not math.isfinite(position) or position < 0 or (
                index > 0 and position <= last_time
            ):
                position = index / fps
                if index > 0:
                    position = max(position, last_time + 1.0 / fps)
            last_time = position

            record = reader.read(image)
            record["frame"] = index
            record["time_s"] = position
            records.append(record)

            index += 1
            if progress_every and index % progress_every == 0:
                print(f"  frames read: {index}")
    finally:
        cap.release()

    return records


def calibration_debug_image(
    video_path: str,
    calibration: dict,
    config: TemperatureConfig | None = None,
    profiles: dict | None = None,
) -> np.ndarray:
    """Annotated crop of the display under a calibration, for checking by eye."""
    cfg = config or TemperatureConfig()
    profiles = profiles or GEOMETRY_PROFILES

    frame = median_calibration_frame(video_path, cfg.calibration_frames)
    reader = reader_at(
        frame.shape,
        profiles[calibration["geometry"]],
        calibration["scale"],
        tuple(calibration["anchor"]),
        cfg,
        name=calibration["geometry"],
    )
    reader.calibrate(frame)

    return reader.debug_image(frame, reader.read(frame))


def save_calibration(calibration: dict, path: str) -> str:
    """Write a calibration to JSON so later recordings can reuse it."""
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)

    return path


def load_calibration(path: str) -> dict:
    """Read a calibration written by :func:`save_calibration`."""
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
