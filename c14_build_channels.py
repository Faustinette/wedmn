# Section 3.b.vi — Construct model input channels / dataset
# Executed by runner.py inside the shared namespace (notebook-kernel style).


# [2] EVALUATION BY PROGRESSION BAND
# Reports how prediction accuracy evolves over the course of a voyage:
# accuracy is computed separately for each progression band (how far along
# the trajectory a step is, per DEFAULT_PROGRESSION_BOUNDARIES), alongside
# two reference numbers — overall accuracy across all steps, and accuracy
# at the final step only. This is the reporting format used for the
# accuracy-vs-progression tables and plots.


# Voyage-progression bins used throughout the project: 20 uniform bands of
# 5% covering the full 0-100% range (<=5%, <=10%, ..., <=100%).
#
# Band assignment: each step's progression fraction is (step_index+1)/N,
# and the step falls into the FIRST band whose upper edge is >= that
# fraction — e.g. a step at 18% progression lands in the "<=20%" band.
# Bands are therefore mutually exclusive and exhaustive: summing band_total
# over all bands must equal the total step count (a useful sanity check if
# you ever modify these boundaries).
#
# Every evaluation function that takes a `boundaries` argument
# (evaluate_quartile_accuracy, evaluate_routed_progression_accuracy,
# evaluate_multi_regime_routing) defaults to this constant, and every plot
# built from their output inherits the same resolution — so accuracy-by-
# progression results are directly comparable across all experiments.

DEFAULT_PROGRESSION_BOUNDARIES = tuple(round(i * 0.05, 2) for i in range(1, 21))

# The 1° x 1° grid size is a design decision from Step 3a (the stage that
# produced trajectories_gridded.parquet), so this constant is set directly
# from that specification rather than estimated from the data.
# GRID_LAT_C / GRID_LON_C store the CENTER coordinate of each cell.
# Functions that draw grid-cell boundaries (e.g. plot_port_traffic) treat
# this constant as the source of truth; they may additionally verify it
# against the spacing of GRID_LAT_IDX vs GRID_LAT_C in the data, but only
# as a sanity check — a mismatch there indicates corrupted input, not a
# reason to change this value.

GRID_CELL_SIZE_DEG = 1.0

# Subfolder name (under work_dir) where train_residual_progression_variant,
# train_regime_model, and related functions save and load their results.
# Every function that follows this convention reads this one constant, so
# changing it here applies everywhere consistently. Default "Results"
# matches this project's folder layout; there is normally no reason to
# change it.

RESULTS_SUBFOLDER = "Results"


# Construct the dataset
assert DATA_SUBFOLDER == ""
data = Step3Data(WORK_DIR)
print(f"Step3Data ready: {len(data.traj_idx):,} segments  "
      f"n_ports={data.n_ports}  n_size_classes={data.n_size_classes}  "
      f"steps rows={len(data.steps_idx):,}")

# Enrich the data labels
enrich_arrival_labels(data)

# Build Ship History
data.history_index = VesselHistoryIndex(data.traj_idx)
print(f"ship-history index built: {len(data.history_index._by_imo):,} vessels, "
      f"{len(data.history_index.seg_to_imo):,} segments mapped")

# Get Subregion names
N_CLASSES = len(data.vocab["port_subregion_to_id"])
subregion_names = get_subregion_name_map(data)
print(f"\nN_CLASSES = {N_CLASSES} destination subregions:")
for cid in sorted(subregion_names):
    print(f"   c = {cid:2d}   {subregion_names[cid]}")
