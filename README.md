# WEDMN — Vessel Destination Prediction

**IMPORTANT NOTE : The full raw input dataset and trained model
checkpoints can be made available to examiners on request.**

Clean, git-uploadable migration of `Main_forGitHub.ipynb`. Every notebook cell
is preserved **verbatim** (each file marks its source cells), and the runner
reproduces the notebook kernel: all stages execute in one shared namespace, so
cross-cell globals (`data`, `BEST_EPOCHS`, `fold_ids`, `E3`, ...) keep working
exactly as they did in Jupyter.

## Requirements

- **Python ≥ 3.10** (reference environment: Google Colab, **Python 3.13**,
  where all of the report's results were produced)
- **Keras ≥ 3** running on the **PyTorch backend** (torch ≥ 2.6; reference:
  torch 2.11 / keras 3.13). The code sets `KERAS_BACKEND=torch` automatically
  before importing keras — no manual configuration needed, but Keras 2.x will
  NOT work.
- `requirements.txt` gives a minimum-version install;
  `requirements-colab.txt` pins the exact reference versions for
  reproducibility.
- A GPU is optional: with the trained checkpoints in `Results/`, evaluation
  and analysis experiments run on CPU; retraining from scratch is GPU-advised.
- `cartopy` is only needed for `--experiment E34` (trajectory maps); all other
  experiments run without it.

## Quickstart

```bash
# 1. install dependencies (inside a fresh venv/conda env, Python >= 3.10)
pip install -r requirements.txt

# 2. place the input data (see "Input files" below) in the working directory 
#    either this folder itself, or any folder pointed to by WORK_DIR:
export WORK_DIR=/path/to/data        # optional; defaults to current directory
#    (Windows: set WORK_DIR=C:\path\to\data)

# 3. smoke-test the dispatcher, then run experiments
python main.py --help
python main.py --experiment E0A     # main training — reloads checkpoints if present
python main.py --experiment E5      # mixture vs shared feed-forward
python main.py --experiment H8      # ship-history contribution
```

Run `python main.py --help` for the full list. Each experiment resolves its own
prerequisites (Step-3b data load → model → training library → trained checkpoints).
Training uses the notebook's `skip_existing=True` guard, so once the 3-seed
model exists under `$WORK_DIR/Results/` it reloads instead of retraining.

## How this repository works (execution order)

Nothing in `core/` or `experiments/` is run directly — `main.py` is the only
entry point. Every `--experiment` automatically executes the required steps
in a fixed order (each at most once):

1. **core** — base imports → input-file check → dataset build (notebook
   section 3.b) → split, channels, model architecture, loader (4.1-4.6) →
   training library (5.1)
2. **train_config** — hyperparameters and sanity tripwires
3. **trained** — 3-seed main training; with checkpoints present in
   `Results/`, models are *reloaded, not retrained*
4. **experiment-specific prerequisites** where needed (E3-prereq, pooled
   predictions, CV folds)
5. **the requested experiment file itself**

The full file-by-file sequence is documented at the top of `main.py`, and the
console prints a banner for every stage and file as it executes.

## Repository layout

```
main.py                  argparse dispatcher  (EXPERIMENTS = {"E5": run_e5, ...})
runner.py                shared-namespace stage runner + prerequisite graph
core/                    notebook sections 0, 3.b, 4.1-4.6, 5.1 (data, model, training lib)
experiments/             one file per experiment (E0..E34, H8, stats, viz)
```

The repo deliberately starts at **Step 3b**: raw AIS data, the port-mapping
database, sanctions/IMO filters and all Step 1-3a preprocessing scripts are
NOT included. The code consumes only the preprocessed model-input files below.

| `--experiment` | What it runs |
|---|---|
| E0A / E0B / E0C | main training / k-fold CV / test-set evaluation |
| E1 / E1B / E1C | benchmark vs captain-declared destination + breakdowns |
| STATS / PARAMS | Step 5B statistics, Step 5C parameter estimates |
| VIZ / E34 | report visualisations / cartopy trajectory maps |
| E3 / E4 | channel ablations / gate-input ablations |
| H8 | ship-history contribution |
| E5 / E6 / E6CV / E7 / E7B | mixture vs shared-FF, expert count, gate behaviour |
| E8 / E9 / E10 / E11 | cold start, structural break, fleet effect, regularization |
| E15 / E18 | error structure, error slice & dice |

## Data availability — important note for reviewers

**This repository contains code only; no data or trained checkpoints are
included.** The underlying AIS-derived dataset contains vessel identifiers
(IMO numbers) and commercially sensitive movement information, and is
therefore not publicly distributed. As a consequence, the experiments cannot
be executed from a fresh clone of this repository alone: the input-file check
(`core/c02_input_checks.py`) will halt immediately with an explicit
"MISSING input" message. **The full input dataset and trained model
checkpoints can be made available to examiners on request.**

## Input files to place in `$WORK_DIR` (not in git — `.gitignore` excludes them)

**Required by every experiment** (the four Step-3 outputs; checked up-front by
`core/c02_input_checks.py`, which fails loudly if any is absent):

- [ ] `trajectories_gridded.parquet` — per-ping rows, gridded
- [ ] `segment_steps_index.parquet` — one row per (segment, grid visit)
- [ ] `trajectories_index_enriched.csv` — one row per voyage segment
- [ ] `step3_vocabularies.json` — port / size-class / subregion ids

**Optional, per experiment:**

- [ ] `lpg_port_reference_fixed.csv` — port names + coordinates; needed ONLY by
      the VIZ port-call maps (Section 6.4). Omit it if you skip VIZ.
- [ ] `Results/` — trained checkpoints (`*_final_main_lean2_seed*.pt` +
      `*_meta.json`); with these present, training stages reload instead of
      retraining (the notebook's `skip_existing=True` guard).

**Scripts still exec'd by three experiments — place at repo root if you run them:**

- [ ] `cold_start_tier2_pooled.py` (E8)
- [ ] `event_out_of_sample_rigorous.py` (E9)
- [ ] `exp1b_final_control_cell.py` (E10)

Note: the four input files still contain vessel identifiers (IMOs) and port
names — keep them in `$WORK_DIR`, out of the public repo (the `.gitignore`
already excludes `*.parquet`, `*.csv`, `*.json` data by extension where listed).

## Notes on the migration

- `WORK_DIR` was hard-coded to a Colab Drive path in the notebook; it now comes
  from the `WORK_DIR` env var (default: current directory). Colab-only cells
  (`drive.mount`, `%cd`, `!pip`) were removed/commented; installs live in
  `requirements.txt`.
- `KERAS_BACKEND=torch` is set in `core/c01_imports.py` **before** keras is
  imported — do not import keras earlier (the notebook's tripwire for this is
  preserved in `core/c31_train_config.py`).
- Cells were grouped by their notebook section headers; each generated file
  records the exact source cell indices, so nothing was rewritten.
