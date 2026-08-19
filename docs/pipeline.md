# Pipeline details

Background on the design decisions and on which parameters to reach for when
something looks wrong.

---

## Step 1 — Segmentation (`01_segment_droplets.py`)

### What SAM 3 is asked to do

SAM 3 is prompted with text (`--prompt`, default
`"clear water droplets on a metallic surface"`) on frame 0 only. All returned
instance masks are thresholded and combined into a single boolean mask; the
separation into individual droplets happens afterwards with connected
components, not by trusting SAM 3's instance assignment.

That split is deliberate. Instance segmentation of many near-identical
droplets is the part a promptable model is least reliable at, while connected
components on a clean binary mask is exact and reproducible.

If larvae are clearly visible inside the droplets, a more specific prompt can
help:

```bash
--prompt "clear water droplets on a metallic surface, a small larva inside a droplet"
```

### Precision matters more than recall here

The model is loaded in fp32 and executed under `torch.autocast(fp16)`. Loading
the weights in fp16 directly triggers dtype mismatches inside the SAM 3 video
processor. Processing and video storage stay on the CPU, which keeps VRAM
usage low and has proven more stable than a fully-GPU pipeline.

### Tuning

| Symptom | Parameter | Direction |
| --- | --- | --- |
| Faint droplets missing from the mask | `--threshold` | lower, 0.06–0.10 |
| Background speckle in the mask | `--threshold` | raise, 0.12–0.15 |
| Noise fragments become their own droplets | `--min-area-px` | raise |
| Small droplets dropped | `--min-area-px` | lower |
| Larva clipped at the ROI border | `--padding-px` | raise |
| Two droplets merged into one component | — | see below |

Two touching droplets merge into a single connected component and get one ID.
There is no parameter for this; edit `frame0_mask.png` in any image editor,
draw a one-pixel background line between them, and re-run
`02_extract_droplets.py` on the corrected mask.

**Always check `droplet_schema.png` before running DeepLabCut.** It is the only
place where the ID assignment is visible, and every downstream table refers to
those IDs.

### Why ROI backgrounds are blanked

Bounding boxes of neighbouring droplets overlap. Without masking, a larva from
the adjacent droplet is visible in the corner of the ROI and DeepLabCut will
occasionally track it instead. `--keep-background` disables the masking if the
raw crop is needed for something else.

### Temperature readout

The chamber temperature is read off the LCD thermometer in view of the camera
rather than logged separately, which removes any clock drift between two
devices: every frame's temperature is the one that frame saw.

**Two stages.** A per-frame full-frame search would be far too slow, so the
display is located once on a median of the opening frames — the median removes
the moving larvae and averages out sensor noise — and afterwards only a small
ROI around it is touched, with a few pixels of local search to absorb camera
drift.

**How a candidate is scored.** Every grid position is decoded as three
seven-segment digits and scored by how cleanly the segments fall on either side
of the darkness threshold, plus how much the surrounding pixels look like an
LCD: saturated, cyan, and flat in texture. A reading that decodes to something
far from room temperature at the start of a recording is penalised heavily,
which is what stops droplets and printed labels from winning the search.

**Why darkness rather than OCR.** A morphological closing estimates the local
background and the difference to the actual pixel gives how dark each segment
sample is. That is independent of the LCD backlight and of overall exposure, so
the same thresholds work across recordings.

**Failure is loud.** If no candidate clears `--lcd-min-score`, the search
raises rather than returning its best guess — a mislocated display would
produce a plausible-looking but entirely wrong temperature trace. By default
the recording is still segmented and only the temperature is missing;
`--require-temperature` makes it fatal.

**Temporal filtering.** The LCD refreshes asynchronously to the camera, so
individual frames catch it mid-transition and decode to a completely different
number. A centred median over seven frames (about 0.23 s at 30 fps) removes
those; a median rather than a mean, because the failure mode is a single wild
value, not a small perturbation. Runs of unreadable frames shorter than
`max_interpolation_gap_frames` are then interpolated, longer ones stay missing.

On a monotonic ramp the median can shift a value by one display step, i.e.
0.1 °C. That is inherent to the filter and well below the display's own
resolution.

### When the search misses the display (`10_calibrate_lcd.py`)

**Do not simply lower `--lcd-min-score`.** On the recording this was written
for, the rejected candidate decoded to 98.8 °C while the screen showed 23.7 —
lowering the threshold would have produced a complete, confident and entirely
wrong temperature trace. The threshold was doing its job.

The failure is almost never a missing geometry. It is the anchor: the
full-frame search scores how digit-like and how LCD-coloured each position
looks, and on a busy scene it can settle a few pixels — or a few hundred —
away from the display. The same profile that failed on one recording reads
another perfectly.

`scripts/10_calibrate_lcd.py` removes the guesswork. The display region is
found by colour and the digits by shape, which narrows the anchor to a small
window; the anchor and scale are then chosen by the one criterion that cannot
be faked — the decoded value has to equal the number you can see on screen:

```bash
python scripts/10_calibrate_lcd.py data/V1.mp4 --known-temperature 23.7
```

If the display sits in a busy corner, `--roi X0 Y0 X1 Y1` narrows the search.
The result is a JSON file, and the tool reports what fraction of frames it
actually reads — a calibration that decodes the first frame but drifts later
is worse than useless.

Two ways to use it:

- `--write-temperature` reads the whole recording there and then and writes
  `temperature.csv`, **without re-running SAM 3**. This is the one to use when
  only the temperature failed and the segmentation is already done.
- `--lcd-calibration file.json` on step 1 uses it for a fresh run.

With a fixed camera the calibration from one recording usually fits the whole
session, but check the coverage per recording rather than assuming it.

**Tuning the automatic search.** If the recording starts outside 15–45 °C,
widen `--lcd-expected-start`, otherwise the correct candidate is penalised as
implausible. A genuinely different display shape needs a new entry in
`GEOMETRY_PROFILES` in `src/larvatracker/lcd_temperature.py` — but measure
before assuming that is the problem.

---

## Step 2 — Pose estimation (DeepLabCut, external)

Not part of this repository. What the analysis scripts assume:

- five keypoints per larva, labelled in head-to-tail order;
- DeepLabCut's standard CSV export (three header rows, then
  `frame, x, y, likelihood` per keypoint);
- the file name still contains `droplet_XXX`, which is how the droplet ID is
  recovered.

A different number of keypoints works — pass `--bodyparts` with the labels in
file order. Everything downstream is written against `len(bodyparts)`, with one
exception: `plot_group_dashboard` uses keypoint `a` as its reference for onset
latency, configurable via `reference_bodypart`.

---

## Step 3 — Analysis (`03_analyze_experiment.py`, `04_batch_analysis.py`)

### Order of operations

1. **Likelihood filter.** Keypoints below `--likelihood` (default 0.6) become
   NaN. Everything downstream is NaN-aware; no interpolation happens at this
   stage.
2. **Body-axis sorting.** Per frame, keypoints are projected onto their first
   principal component and re-sorted along it, with the axis direction carried
   over between frames. Frames with fewer than three valid keypoints are left
   as NaN rather than guessed.
3. **Displacement.** Euclidean shift between consecutive frames per keypoint,
   in px/frame.
4. **Smoothing.** Centred rolling mean over `--smoothing-window` frames
   (default 15, i.e. 0.5 s at 30 fps). Wide enough to suppress tracking
   jitter, narrow enough to keep individual bursts separable.
5. **Burst detection.** `scipy.signal.find_peaks` on the smoothed trace with
   `--peak-prominence` (default 1.2 px) and `--peak-distance` (default 10
   frames). Prominence is what separates real bouts from jitter; distance acts
   as a refractory period.
6. **Onset.** First frame whose smoothed displacement exceeds
   `--onset-threshold` (default 4 px/frame). NaN when the animal never reaches
   that level, which is itself informative — do not read NaN as zero.
7. **Time bins.** The recording is split into `--bin-size-frames` blocks
   (default 900 = 30 s at 30 fps) and burst frequency is scored per block. A
   trailing partial bin is discarded so every bin covers the same duration.

### Tuning burst detection

The default prominence of 1.2 px assumes the imaging scale of this setup. On a
different magnification it is the first thing to re-tune:

- **too many bursts** → raise `--peak-prominence`, or widen
  `--smoothing-window`;
- **bursts missed** → lower `--peak-prominence`; check first that the keypoint
  is actually being tracked (a mostly-NaN trace produces few peaks for a
  trivial reason);
- **one bout counted several times** → raise `--peak-distance`.

Run `06_droplet_kinematics.py` on a representative larva and look at the
detected peaks before committing to a parameter set for a whole project.

---

## Step 4 — Group statistics (`05_group_statistics.py`)

### Baseline normalisation

Absolute burst frequency varies strongly between individual larvae and between
recording days. Comparing raw frequencies across groups mostly measures that
variability. Each larva is therefore divided by its own first time bin, turning
the analysis into a within-subject comparison of the response.

Two consequences:

- Larvae that were completely immobile during the baseline bin have a baseline
  of zero and no defined ratio. They are **dropped**, not clipped, and the
  count is printed. If that count is large, the baseline bin may be too short
  or the animals may not have been acclimatised.
- The baseline bin equals 1.0 by construction and carries no information, so it
  is excluded from the model and from the post-hoc tests.

### Statistical model

`Freq_Hz_norm ~ C(Group) * C(Time)` with a random intercept per larva
(`statsmodels.mixedlm`). Repeated measurements of the same animal across bins
are not independent; the random intercept accounts for that.
`wald_test_terms()` gives one omnibus p-value per factor.

Post-hoc, every treatment is compared against `--control` separately per time
bin using a two-sided Mann-Whitney U test — normalised frequencies are ratios
and not normally distributed — with Benjamini-Hochberg FDR correction over all
comparisons at once.

### Group labels

Labels typed by hand drift over the course of a project. `--group-map` takes a
JSON file mapping lower-cased, whitespace-collapsed labels to canonical names
(see `examples/group_map.json`). Labels missing from the map are **reported and
excluded**, deliberately: a silent fallback would let a typo form its own group
of one and quietly shrink a real group.

---

## The temperature branch (`08_framewise_temperature.py`, `07_temperature_response.py`)

Steps 3 to 5 ask how much a larva moved *over time*. When the stage runs a
temperature ramp, the more interesting question is how much it moved *at a
given temperature* — which needs a different reduction, not a different
experiment.

### Step 8 — joining movement and temperature

Step 3 collapses each larva to summary numbers per time bin; the temperature
analysis needs the opposite, one row per larva, keypoint and frame. Step 8
produces that table by joining the per-frame displacement with
`temperature.csv`.

**The join is exact, not approximate.** Both sides are indexed by frame number
and both were produced in the same decode pass in step 1, so there is no clock
to align and nothing to interpolate between devices. The keypoints are
re-sorted along the body axis first, exactly as in step 3 — skipping that
would let a head/tail label swap appear as a large displacement, attributed to
whatever temperature that frame happened to be at.

**A missing display is not a failed experiment.** A recording whose LCD could
not be located has no `temperature.csv`; it is skipped and recorded in
`_framewise_report.csv` instead of aborting the batch. The rest of that
recording is still perfectly good for the time-based analysis in steps 3 to 5.

**Coverage is the number to check.** A recording read at 42 % coverage produces
a file that looks entirely normal but rests on two fifths of the frames — a
worse failure mode than a missing file, because nothing draws attention to it.
The fraction is reported per recording, a warning is printed below 80 %, and
`--min-coverage` turns it into a hard filter.

**A frame-count mismatch means trouble.** If the tracking and the temperature
disagree on how many frames the recording has, they came from different runs
of step 1. Only the overlap is kept and the difference is reported; the right
fix is to reprocess, not to trust the overlap.

Frames without a temperature are dropped by default, since they contribute
nothing to a temperature analysis and can be the majority of a poorly read
recording. `--keep-missing-temperature` keeps them.

Expect roughly *droplets x 5 x frames* rows per recording — around half a
million for two minutes of 30 droplets, so tens of MB per experiment.

### Step 7 — comparing groups across temperature

1. **Pool the keypoints.** The question is how *much* the animal moved, not
   which part, so the five keypoints are averaged per larva and frame.
2. **Normalise per larva** to its own opening seconds. Absolute movement
   depends on body size and on how well that droplet tracked; the ratio does
   not. Verify that the baseline window really is at ambient temperature — in
   the recordings this was built for, 25 °C is not reached before 40 s.
   Larvae that were immobile during the window have no defined ratio and are
   dropped rather than clipped.
3. **Bin by temperature** into `--temp-bin` wide bins, labelled by centre.
4. **Reduce to one value per larva and bin** *before* any statistics. A
   recording provides thousands of frames but only a handful of animals, and
   frames within one larva are heavily autocorrelated; testing on frames would
   treat that as independent evidence.
5. **Compare each group against its own control**, per bin, with a two-sided
   Mann-Whitney U test and Benjamini-Hochberg FDR across all comparisons.

### Two guard rails, and why they exist

**Bin coverage.** Recordings start and end at different temperatures, so the
extreme bins are populated by a small, non-random subset of experiments — a
group difference there is confounded with which recording got that far. Bins
below `--min-folders-per-bin` are dropped and listed in `bin_coverage.csv`.

The default is a *majority of the experiments present, capped at five*. A fixed
number does not survive datasets of different sizes: five is a sensible bar
among nineteen recordings and silently empties the analysis among three. If
everything is dropped the script now says so and exits, rather than reporting
zero larvae as though that were a result.

**Vehicle agreement.** If the control groups differ from each other, curves
from different datasets cannot be read against one another — the difference
may come from the vehicle or from the recording batch rather than the
treatment. The check runs automatically and warns. The per-drug figure and
`stats_vs_control.csv` stay valid regardless, because each group is only ever
compared against its own control.

### Controls are configurable

`--control-map` maps a substring of the group name to the group it should be
tested against. The default covers the drug/vehicle design this was built for
(`asp`/`ibu` → `ETOH`, `dic` → `DMSO`, `cis` → `PBS`), but the mechanism is
generic — a genotype comparison is just a different map:

```json
{"germfree": "conventional", "painless": "conventional"}
```

Groups with no entry are skipped rather than compared against something
arbitrary, so an unmapped group shows up as missing rows in
`stats_vs_control.csv` rather than as a wrong result.

---

## Reproducibility notes

- Droplet IDs are reproducible for a given mask but not comparable across
  recordings. Each recording gets its own `droplet_schema.png`.
- All parameters are CLI flags with defaults in `src/larvatracker/config.py`.
  Record the command line alongside the results; nothing needs to be edited in
  the source to process a new dataset.
- SAM 3 inference is not bit-reproducible across GPU models and driver
  versions. Commit `frame0_mask.png` or `droplets.csv` alongside published
  results if the exact segmentation matters.
