# Larva tracking in droplets with SAM 3

Behavioural tracking of individual *Drosophila* larvae in water droplets.
Droplets on a metallic surface are segmented once with
[SAM 3](https://huggingface.co/facebook/sam3), each droplet is cut out as its
own ROI video, pose estimation runs per droplet in DeepLabCut, and the
resulting keypoint trajectories are turned into locomotion metrics and group
statistics.

The setup was built to compare drug treatments (e.g. aspirin and ibuprofen at
several doses against an ethanol control) by their effect on larval locomotion
over the course of a recording.

---

## Pipeline

```
raw recording (.mp4)
        │
        │  1. scripts/01_segment_droplets.py          [SAM 3, GPU]
        ▼
droplet schema + one ROI video per droplet + temperature.csv
        │
        │  2. DeepLabCut                              [external]
        ▼
per-droplet keypoint CSVs (5 points, head → tail)
        │
        │  3. scripts/03_analyze_experiment.py
        │     scripts/04_batch_analysis.py
        ▼
long-format summary table + per-droplet figures
        │
        │  4. scripts/05_group_statistics.py
        ▼
baseline-normalised group comparison, mixed model, post-hoc tests
```

When the recording includes a temperature ramp there is a second branch:

```
droplet schema + ROI videos + temperature.csv
        │
        │  scripts/08_framewise_temperature.py
        ▼
one row per larva, keypoint and frame, with its temperature
        │
        │  scripts/07_temperature_response.py
        ▼
movement across temperature, compared across groups
        │
        │  scripts/09_temperature_model.py
        ▼
mixed model: one test per group instead of one per bin
```

### Why segment only the first frame

The droplets do not move during a recording — only the larvae inside them do.
Segmenting frame 0 and reusing that mask replaces a full video segmentation
with a single forward pass, and it guarantees that a droplet keeps the same ID
for the entire recording.

### Why the keypoints get re-sorted

DeepLabCut labels five points along the larva, but their identity is not stable
frame to frame: a symmetric, deforming animal makes head and tail labels swap
regularly. Per-keypoint metrics are meaningless while those swaps are present.

The correction is geometric. Per frame, the keypoints are projected onto their
first principal component (the body axis, via SVD) and re-sorted along it. The
axis direction is carried over between frames and flipped when it would
otherwise reverse, so "point 0" stays on the same end of the animal throughout.
See `sort_points_along_axis` in `src/larvatracker/posture.py`.

---

## Installation

Requires Python 3.10 and, for step 1, an NVIDIA GPU with CUDA. Steps 3–5 run on
CPU and need neither torch nor transformers.

```bash
conda env create -f environment.yml
conda activate larvae-tracking-sam3
pip install -e .
```

Or with pip only:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> **SAM 3 availability.** `Sam3VideoModel` ships with the transformers 5.0
> development line. If the import fails, install transformers from source:
> `pip install git+https://github.com/huggingface/transformers.git`

The `facebook/sam3` weights are downloaded from the Hugging Face Hub on first
use and cached locally.

---

## Usage

### 1. Segment droplets and cut ROI videos

```bash
python scripts/01_segment_droplets.py data/M4.mp4
```

Writes into `data/M4/`:

| File | Contents |
| --- | --- |
| `frame0_mask.png` | binary union mask of all droplets |
| `frame0_overlay.png` | frame 0 with the mask blended in green |
| `droplet_schema.png` | overlay plus bounding boxes and droplet IDs — **read the IDs off this image** |
| `droplet_id_mask.png` | 16-bit label image, pixel value = droplet ID |
| `droplets.csv` | one row per droplet: ID, area, centroid, bounding box |
| `droplet_videos/droplet_XXX.mp4` | one masked ROI video per droplet |
| `temperature.csv` | chamber temperature per frame, read off the LCD |
| `temperature_display_debug.png` | the decoded display, for checking by eye |

Folders work too, and the SAM 3 weights are then loaded once for the whole
batch:

```bash
python scripts/01_segment_droplets.py data/session/ --skip-completed
```

A failing recording is logged in `_batch_summary.csv` and the batch continues;
`--stop-on-error` changes that.

If the mask misses faint droplets, lower the threshold; if it picks up noise,
raise the minimum area:

```bash
python scripts/01_segment_droplets.py data/M4.mp4 --threshold 0.06 --min-area-px 400
```

Retuning the droplet parameters does not require another GPU pass — reuse the
mask (which may also be hand-corrected in any image editor):

```bash
python scripts/02_extract_droplets.py data/M4/frame0_mask.png --video data/M4.mp4 --videos
```

#### Temperature readout

The heating stage's LCD thermometer is read straight out of the recording, so
every frame's temperature is the one that frame actually saw — no clock drift
between camera and logger. The display is located once by scanning the whole
frame at several scales and rotations, then read from a small ROI in the same
pass that writes the ROI videos.

No display coordinates are hard-coded: the geometry profiles describe the shape
of the digits relative to each other, so a display in a different corner or at a
different angle is still found. If it cannot be located the recording is still
processed and only `temperature.csv` is missing; `--require-temperature` makes
that fatal instead, and `--no-temperature` skips the search altogether.

**If a recording ends up with no temperature, or with a low coverage in
`_framewise_report.csv`, calibrate it rather than lowering the threshold:**

```bash
python scripts/10_calibrate_lcd.py data/V1.mp4 --known-temperature 23.7 \
    --write-temperature
```

You supply the number visible on screen at the start; the calibration is then
chosen by having to decode to exactly that. `--write-temperature` produces
`temperature.csv` directly, without re-running SAM 3. Lowering
`--lcd-min-score` instead is how you get a confident, complete and wrong
temperature trace — the threshold exists precisely to prevent that.

### 2. Pose estimation (DeepLabCut, external)

Train or apply a DeepLabCut model that labels five points from head to tail on
`droplet_videos/*.mp4`. DeepLabCut writes its CSVs next to the videos and
appends the network name, e.g. `droplet_007DLC_Resnet50_....csv`. The analysis
scripts read the droplet ID back out of that filename, so keep the
`droplet_XXX` prefix intact.

### 3. Analyse one experiment

```bash
python scripts/03_analyze_experiment.py data/M4/droplet_videos --dashboard
```

With treatment groups, add a scheme file mapping droplet IDs to groups
(see `examples/scheme_template.csv`):

```bash
python scripts/03_analyze_experiment.py data/Q4/droplet_videos \
    --scheme data/Q4/Scheme.xlsx --dashboard
```

### 4. Batch over a whole project

```bash
python scripts/04_batch_analysis.py data/my_project --pattern "Q*_*"
```

Expected layout:

```
data/my_project/
  Q1_050526/
    droplet_videos/     # DeepLabCut CSVs
    Scheme.xlsx         # optional
  Q2_060526/
    ...
```

Produces `Combined_All_Folders_Summary.csv` at the project root.

### 5. Group statistics

```bash
python scripts/05_group_statistics.py data/my_project/Combined_All_Folders_Summary.csv \
    --control ETOH --bins 1 2 3 4 --group-map examples/group_map.json --out-dir results/
```

Each larva is divided by its own baseline bin, then
`Freq_Hz_norm ~ Group * Time` is fitted with a random intercept per larva, and
every treatment is compared against the control per time bin (Mann-Whitney,
Benjamini-Hochberg FDR).

### Optional: single-larva kinematics

```bash
python scripts/06_droplet_kinematics.py data/M4/droplet_videos/droplet_001DLC_...csv \
    --out-dir results/droplet_001
```

Full time course of body axis angle, angular velocity, joint angles, curvature
and per-keypoint displacement — useful for checking that the body-axis sorting
did its job.

### Optional: response to a temperature ramp

First join movement and temperature frame by frame:

```bash
python scripts/08_framewise_temperature.py data/Genotypes --pattern "V*_*"
```

Recordings whose LCD could not be read have no `temperature.csv`; they are
skipped and listed in `_framewise_report.csv` rather than aborting the batch.
That report also carries each recording's **temperature coverage** — the
fraction of frames that actually carry a reading. A recording at 40 % coverage
still produces output that looks fine, so check the column;
`--min-coverage 0.8` turns the check into a hard filter.

Then run the comparison:

```bash
python scripts/07_temperature_response.py \
    Analgetics/Combined_All_Folders_Framewise_Temperature.csv \
    Cisplatin/Combined_All_Folders_Framewise_Temperature.csv \
    --dataset-names Analgetics Cisplatin --out-dir results/temperature
```

Each larva is normalised to its own first seconds, frames are binned by chamber
temperature, and every treatment is tested against its own vehicle control
(Asp/Ibu vs ETOH, Dic vs DMSO, Cis vs PBS — configurable with `--control-map`).

`--control-map` takes a JSON file mapping a substring of the group name to the
group it should be compared against, so the same script handles designs that
are not drug/vehicle at all:

```json
{"germfree": "conventional", "painless": "conventional"}
```

Two guard rails run automatically and are worth reading in the output:

- **Coverage.** Recordings start and end at different temperatures, so the
  extreme bins come from a small subset of experiments. Bins reached by fewer
  than `--min-folders-per-bin` experiments are dropped from the figures and
  statistics, and listed in `bin_coverage.csv`. The default is a majority of
  the experiments present, capped at 5 — a fixed number would silently empty
  the analysis on a dataset of three recordings.
- **Vehicle agreement.** If the vehicle controls differ from each other, curves
  from different datasets are not comparable and the script says so. The
  per-drug figure and `stats_vs_control.csv` stay valid either way, because
  each treatment is only ever compared against its own vehicle.

### Optional: mixed model instead of per-bin tests

```bash
python scripts/09_temperature_model.py \
    results/temperature/per_larva_by_temperature.csv \
    --control conventional --out-dir results/model --compare-specifications
```

Step 7 runs a separate test in every temperature bin. That describes *where* a
difference sits, but it spends its power on thirty-odd tests: an effect that is
consistent yet modest at each single temperature will not survive the
correction. This step asks the question once per group instead:

```
log(Movement_norm) ~ Group * spline(Temperature) + Folder + (1 + Temperature | larva)
```

The `Group x spline` interaction is the biological question — does the *shape*
of the response depend on the group. Each group then gets one joint test
against the control across the whole range, Holm-corrected over groups.

Contrasts come back as **ratios**, because the response is modelled on the log
scale: 0.71 means that group moved 71 % as much as the control at that
temperature. The contrast curve is descriptive and its intervals are pointwise
— the omnibus test is what establishes whether there is a difference at all.

`--compare-specifications` fits competing spline degrees and random-effect
structures and ranks them by AIC, so the choice is visible rather than assumed.

Sample sizes are reported with the statistics, in `sample_sizes.csv` and in
`group_tests.csv` next to each p-value. **`N_Larvae` is the sample size** — one
random intercept per animal. `N_Observations` counts larva x temperature bin
and is tens of times larger; it is not an n. `sample_sizes_by_folder.csv`
breaks the animals down per recording, which is where an imbalance shows up.

---

## Output format

`03`/`04` write a long-format table with one row per droplet, keypoint and time
bin:

| Column | Meaning |
| --- | --- |
| `Folder` | experiment folder the row came from |
| `Droplet` | droplet ID, matching `droplets.csv` and the scheme file |
| `Group` | treatment group, or `Unknown` without a scheme file |
| `BodyPart` | keypoint label after body-axis sorting |
| `Time_Bin` | `"full"` for the whole recording, otherwise `"1"`, `"2"`, … |
| `Time_Sec` | centre of the time bin, in seconds |
| `Freq_Hz` | detected movement bursts per second |
| `Onset_Sec` | latency to the first heavy movement (`full` rows only) |
| `Mean_Vel` | mean frame-to-frame displacement, px/frame |
| `Burst_Count` | number of detected bursts (`full` rows only) |

More detail in [`docs/output_formats.md`](docs/output_formats.md); parameter
tuning in [`docs/pipeline.md`](docs/pipeline.md).

---

## Repository layout

```
src/larvatracker/       importable package
  config.py             all default parameters, documented
  segmentation.py       SAM 3 inference on the first frame
  imaging.py            torch-free OpenCV helpers
  lcd_temperature.py    chamber temperature read off the LCD in frame
  droplets.py           connected components → droplet schema
  roi_videos.py         per-droplet ROI video export
  posture.py            DeepLabCut loading, body-axis sorting, kinematics
  metrics.py            displacement, bursts, onset, time bins
  scheme.py             droplet ID → treatment group
  plotting.py           per-droplet and population figures
  stats.py              normalisation, mixed model, post-hoc tests
  framewise.py          per-frame movement joined with the temperature
  temperature.py        movement versus temperature across treatment groups
  model.py              mixed model for the temperature response
  pipeline.py           end-to-end drivers
  cli.py                shared command line arguments
scripts/                numbered command line entry points
tests/                  pytest suite for the geometry and metric code
docs/                   pipeline details and output formats
examples/               scheme template and group label mapping
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the parts where a silent error would be hard to notice later:
the body-axis sorting, the droplet geometry and the movement metrics. The
sorting test is the important one — it feeds a synthetic larva through the
pipeline with its keypoint labels randomly permuted in **every** frame and
asserts that a single consistent ordering comes back out, which drops the
spurious per-keypoint displacement from ~9.6 to ~0.4 px/frame.

## Data

No experimental data is tracked in this repository — `.gitignore` excludes
videos, masks, DeepLabCut output and result tables. Point the scripts at data
on disk or on a network share instead.

## License

MIT, see [LICENSE](LICENSE).
