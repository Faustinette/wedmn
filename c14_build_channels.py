# =============================================================================
# Section 3.b.vi — Construct model input channels / dataset
# Migrated verbatim from Main_forGitHub.ipynb cells [37, 38].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 37]
# ----------------------------------------------------------------------

# ═════════════════════════════════════════════════════════════════════════════
# [2] QUARTILE-BASED EVALUATION (matches the paper's Table II reporting
#     style: accuracy broken out by trajectory-progression quartile, plus
#     overall accuracy and final-step accuracy as reference points)
# ═════════════════════════════════════════════════════════════════════════════

# Default breakpoints: fine-grained in the early region (where the paper's
# own finding says the biggest gains/challenges are), coarser afterward.
# Each step is assigned to exactly ONE band based on its progression
# fraction (step_index+1)/N — e.g. a step at 18% progression falls in the
# "<=20%" band, not "<=15%". Bands are mutually exclusive, so summing
# band_total across all bands equals the total step count (a good sanity
# check to run if you ever modify these).
#
# Uniform 5% granularity throughout the FULL 0-100% range (20 bands) — a
# tick every 5%, not just in the early range. This is the project-wide
# standard going forward: every function that defaults to
# DEFAULT_PROGRESSION_BOUNDARIES (evaluate_quartile_accuracy,
# evaluate_routed_progression_accuracy, evaluate_multi_regime_routing,
# and every plot built from their output) now reports and plots at this
# resolution automatically, with no extra `boundaries=` argument needed.
DEFAULT_PROGRESSION_BOUNDARIES = tuple(round(i * 0.05, 2) for i in range(1, 21))


# This project's own gridding methodology (Step 3a's own
# trajectories_gridded.parquet output) uses a fixed 1° x 1° grid --
# GRID_LAT_C/GRID_LON_C are each grid cell's own CENTER coordinate.
# Documented directly by the person who built that stage, not derived
# -- functions that draw grid cell boundaries (e.g. plot_port_traffic)
# use this constant as the authoritative value, with an empirical
# check against the data's own GRID_LAT_IDX-to-GRID_LAT_C relationship
# as a cheap sanity check, not the primary source.
GRID_CELL_SIZE_DEG = 1.0

# Subfolder name (under work_dir) where train_residual_progression_variant,
# train_regime_model, and related functions save/load results. Change
# this ONE value (e.g. Step4c_train.RESULTS_SUBFOLDER = "SomeOtherName")
# if a given work_dir uses a different results-folder name -- every
# function that shares this convention reads from this constant, so a
# single override applies everywhere consistently. Default is "Results",
# matching this project's current folder convention.
RESULTS_SUBFOLDER = "Results"

# ----------------------------------------------------------------------
# [notebook cell 38]
# ----------------------------------------------------------------------
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
