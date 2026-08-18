# Output formats

Every file the pipeline writes, and what is in it.

---

## Step 1 — segmentation outputs

Written to `<video_folder>/<video_name>/`.

### `frame0_mask.png`

8-bit binary image, 0 = background, 255 = droplet. The union of all instance
masks SAM 3 returned, after thresholding. This is the file to edit by hand when
two droplets merge into one component; re-run `02_extract_droplets.py`
afterwards.

### `frame0_overlay.png`

Frame 0 with the mask blended in green. For eyeballing segmentation quality.

### `droplet_schema.png`

The overlay plus a green bounding box and a red ID label per droplet. **The
reference image for droplet IDs** — the numbers used in `droplets.csv`, in the
ROI video file names and in the scheme file are the ones printed here.

### `droplet_id_mask.png`

16-bit single-channel PNG. Each pixel holds the ID of its droplet, 0 =
background. 16-bit rather than 8-bit so experiments with more than 255 droplets
stay representable — note that most image viewers will render this as a nearly
black image, which is expected.

Reading it back:

```python
import cv2
id_mask = cv2.imread("droplet_id_mask.png", cv2.IMREAD_UNCHANGED)  # uint16
droplet_7 = id_mask == 7
```

`cv2.IMREAD_UNCHANGED` is required; the default flag would truncate to 8-bit.

### `droplets.csv`

One row per droplet.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | int | 1-based droplet ID |
| `area_px` | int | mask pixels belonging to the droplet |
| `cx`, `cy` | float | centroid in image coordinates |
| `x0`, `y0` | int | top-left corner of the padded bounding box |
| `x1`, `y1` | int | bottom-right corner, exclusive |

The bounding box already includes `--padding-px` and is clamped to the image,
so `x1 - x0` is exactly the width of `droplet_XXX.mp4`.

### `droplet_pixels.csv` (optional, `--pixel-table`)

One row per mask pixel: `id, x, y`. Only needed when droplet areas have to be
re-measured or intersected with other image data — for the standard pipeline
the bounding boxes are sufficient. Expect millions of rows on a
high-resolution recording.

### `droplet_videos/droplet_XXX.mp4`

One ROI video per droplet, cropped to the bounding box, with pixels outside the
droplet blanked to black (unless `--keep-background`). Frame rate and length
match the source recording. `XXX` is the zero-padded droplet ID.

### `temperature.csv`

One row per frame, written when the LCD thermometer was located.

| Column | Type | Meaning |
| --- | --- | --- |
| `frame` | int | 0-based frame index |
| `time_s` | float | timestamp from the container, forced to increase |
| `geometry` | str | which digit geometry profile matched |
| `temperature_c` | float | **the value to use**; empty when unavailable |
| `raw_temperature_c` | float | the unfiltered per-frame decoding |
| `confidence` | float | mean decoding margin; higher is cleaner |
| `raw_valid` | 0/1 | whether the raw decoding passed all checks |
| `status` | str | how `temperature_c` came about, see below |
| `shift_x_px`, `shift_y_px` | int | sample-grid shift found for that frame |
| `segment_errors` | int | segments on the wrong side of the threshold |
| `lcd_anchor_x_px`, `lcd_anchor_y_px` | int | where the display was read |
| `lcd_scale`, `lcd_angle_deg` | float | geometry the locator settled on |
| `lcd_locator_score` | float | confidence of the one-off full-frame search |

`status` values:

| Value | Meaning |
| --- | --- |
| `ok` | the raw reading was valid and the filter agreed with it |
| `temporal_filter` | valid, but the median moved it by ≥ 0.3 °C — usually the LCD caught mid-refresh |
| `recovered_from_neighbors` | the raw reading was rejected; the value comes from surrounding frames |
| `missing` | no value could be established; `temperature_c` is empty |

Use `temperature_c` and treat an empty cell as missing — **not** as zero.
`raw_temperature_c` always holds a number, including for frames where that
number is nonsense, and exists for auditing rather than for analysis.

A useful quality check is the fraction of `missing` and
`recovered_from_neighbors` rows. A handful is normal; a large share means the
display was partly occluded or the locator settled on the wrong spot, which
`temperature_display_debug.png` will show at a glance.

### `temperature_display_debug.png`

The located display, magnified, with a circle on every segment sample point:
green where the decoder considered the segment lit, red where it considered it
dark. If a reading is wrong, this image separates a misaligned sample grid from
a badly chosen threshold.

### `_batch_summary.csv`

Written next to the input when a folder is processed. One row per recording:
droplet count, frame count, the temperature range found, the locator score, and
`status` (`ok`, `skipped_existing` or `error` with the message). The place to
look after an unattended batch run.

---

## Step 3/4 — analysis outputs

### `<Folder>_Summary_Results.csv` and `Combined_All_Folders_Summary.csv`

Long format: one row per droplet × keypoint × time bin, plus one `"full"` row
per droplet × keypoint covering the whole recording.

| Column | Type | Meaning |
| --- | --- | --- |
| `Folder` | str | experiment folder the row came from |
| `Droplet` | int | droplet ID, matches `droplets.csv` and the scheme file |
| `Group` | str | treatment group, `Unknown` without a scheme file |
| `BodyPart` | str | keypoint label **after** body-axis sorting |
| `Time_Bin` | str | `"full"`, or `"1"`, `"2"`, … for the time bins |
| `Time_Sec` | float | centre of the time bin in seconds; NaN for `"full"` |
| `Freq_Hz` | float | detected bursts per second |
| `Onset_Sec` | float | latency to first heavy movement; only on `"full"` rows |
| `Mean_Vel` | float | mean frame-to-frame displacement, px/frame |
| `Burst_Count` | int | number of detected bursts; only on `"full"` rows |

Two things worth remembering when reading this table:

- **`BodyPart` refers to the sorted order, not the DeepLabCut label.** Position
  `a` is whichever end of the animal the body axis put first, held consistent
  across the recording. It is not guaranteed to be the head.
- **`Time_Bin` is a string, including the numeric bins.** Mixing `"full"` and
  numbers in one column forces this. Filter with `df[df.Time_Bin != "full"]`
  and cast explicitly when a numeric axis is needed.
- **NaN in `Onset_Sec` means the threshold was never crossed**, i.e. the larva
  never moved heavily. Treating it as zero inverts the meaning.

### `Analysis_Results/droplet_XXX/*_traces.png`

One panel per keypoint: raw displacement (grey), smoothed displacement (blue),
detected bursts (red dots) and the movement onset (green dashed line). The
figure to check when burst counts look implausible.

### `Group_Comparison_Dashboard.png`

Burst frequency per group split by keypoint (top) and onset latency for the
reference keypoint (bottom). Whole-recording values only.

### `Frequency_Over_Time.png`

Burst frequency per time bin, one line per group, with a 95 % confidence band.

### `Population_Overview.png`

Frequency distribution per keypoint, plus a droplet × keypoint heatmap that
makes unusually active or unusually still droplets easy to spot.

---

## Step 8 — framewise export

### `Framewise_Movement_Temperature.csv` and `Combined_All_Folders_Framewise_Temperature.csv`

One row per larva, keypoint and frame — the input for step 7.

| Column | Type | Meaning |
| --- | --- | --- |
| `Folder` | str | experiment the row came from |
| `Droplet` | int | droplet ID |
| `Group` | str | treatment or genotype, `Unknown` without a scheme file |
| `BodyPart` | str | keypoint label **after** body-axis sorting |
| `Frame` | int | frame index, shared between movement and temperature |
| `Time_Sec` | float | timestamp taken from `temperature.csv` |
| `Temperature_C` | float | chamber temperature at that frame |
| `Movement_px_frame` | float | displacement since the previous frame; empty at frame 0 |
| `Movement_MA_px_frame` | float | the same, smoothed over `--smoothing-window` frames |

By default only frames carrying a temperature appear, so `Temperature_C` is
never empty. `--keep-missing-temperature` retains the rest.

These files get large — roughly *droplets x 5 x frames* rows per recording,
which is around half a million for a two-minute recording of 30 droplets.

### `_framewise_report.csv`

One row per experiment, and the first place to look after a batch run.

| Column | Meaning |
| --- | --- |
| `folder` | experiment name |
| `status` | `ok`, `skipped` or `error` |
| `reason` | why it was skipped or what failed |
| `temperature_coverage` | fraction of frames carrying a reading, 0–1 |
| `droplets`, `frames`, `rows` | size of what was processed |
| `rows_dropped_no_temperature` | rows removed for having no temperature |
| `frame_count_mismatch` | frame-count difference between movement and temperature |

**`temperature_coverage` is the column worth reading.** A recording at 0.42
produces a file that looks perfectly normal but rests on two fifths of the
frames. A non-zero `frame_count_mismatch` means the tracking and the
temperature came from different runs of step 1 — only the overlapping frames
were kept, and the recording should be reprocessed.

---

## Step 5 — statistics outputs

### `normalised_data.csv`

The filtered input plus two columns:

| Column | Meaning |
| --- | --- |
| `rep` | running index pairing repeated measurements of the same larva across bins |
| `Freq_Hz_norm` | `Freq_Hz` divided by the same larva's baseline-bin value |

Larvae with a zero baseline are absent from this file; the number dropped is
printed at run time.

### `posthoc_vs_control.csv`

One row per (time bin × treatment group) comparison against the control.

| Column | Meaning |
| --- | --- |
| `Time_Bin` | bin the comparison was run in |
| `Group` | treatment group |
| `n_treatment`, `n_control` | sample sizes entering the test |
| `median_treatment`, `median_control` | medians of the normalised frequency |
| `p_raw` | uncorrected two-sided Mann-Whitney p-value |
| `p_fdr` | Benjamini-Hochberg corrected p-value across all comparisons |
| `signif` | `p_fdr < 0.05` |

Comparisons with fewer than three samples on either side are skipped and do not
appear as rows. The mixed model summary and the omnibus tests are printed to
stdout rather than written to file — redirect if they need archiving.
