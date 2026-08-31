# =============================================================================
# Step-3 input file inventory + shape checks (fails loudly if inputs missing)
# Migrated verbatim from Main_forGitHub.ipynb cells [27].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 27]
# ----------------------------------------------------------------------
# ==================== CELL 3a -- INPUT FILE INVENTORY ========================
# The four Step-3 outputs the model consumes, checked explicitly so a wrong
# path fails HERE with a clear message, not mid-training.
import os

INPUT_FILES = {
    "gridded":  "trajectories_gridded.parquet",        # per-ping rows, gridded (Step 2/3a)
    "steps":    "segment_steps_index.parquet",         # one row per (segment, grid visit)
    "traj":     "trajectories_index_enriched.csv",     # one row per voyage segment
    "vocab":    "step3_vocabularies.json",             # port / size-class / subregion ids
}
for key, fname in INPUT_FILES.items():
    path = os.path.join(WORK_DIR, fname)
    assert os.path.exists(path), f"MISSING input in main directory: {path}"
    print(f"  {key:8s} {fname:38s} {os.path.getsize(path)/1e6:8.1f} MB")
print("All four Step-3 inputs are present in the directory.")

# ==================== CELL 3b (light) -- INPUT SHAPES, NO HEAVY LOADS ========
import json
import pyarrow.parquet as pq

for key in ("gridded", "steps"):
    md = pq.ParquetFile(os.path.join(WORK_DIR, INPUT_FILES[key])).metadata
    print(f"{key:8s}: {md.num_rows:,} rows x {md.num_columns} cols")
with open(os.path.join(WORK_DIR, INPUT_FILES["vocab"])) as f:
    v = json.load(f)
print(f"vocab   : ports={len(v['port_to_id']):,}  "
      f"size classes={len(v['size_class_to_id'])}  "
      f"subregions={len(v['port_subregion_to_id'])}")
