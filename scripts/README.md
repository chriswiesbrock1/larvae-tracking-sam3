# Scripts

Numbered by the order they are normally run. Every script supports `--help`.

| Script | Purpose |
| --- | --- |
| `10_calibrate_lcd.py` | Calibrate the temperature display against a value you read off the screen; recovers a failed readout |
| `01_segment_droplets.py` | SAM 3 segmentation, droplet schema, ROI videos and LCD temperature readout; handles folders in batch |
| `02_extract_droplets.py` | Rebuild the schema from an existing (or hand-corrected) mask |
| `03_analyze_experiment.py` | Analyse the DeepLabCut output of one experiment |
| `04_batch_analysis.py` | Run step 3 over every experiment folder of a project |
| `05_group_statistics.py` | Baseline normalisation, mixed model and post-hoc tests |
| `06_droplet_kinematics.py` | Full kinematic time course for a single larva |
| `08_framewise_temperature.py` | Join per-frame movement with the per-frame chamber temperature |
| `07_temperature_response.py` | Movement versus temperature, compared across treatment groups |
| `09_temperature_model.py` | Mixed model for the temperature response: one test per group instead of one per bin |

Pose estimation itself (step 2) happens in DeepLabCut and is not part of this
repository; see `docs/pipeline.md`.
