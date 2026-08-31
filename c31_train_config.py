# =============================================================================
# Section 4.2-CONFIG — training config, guard pattern + tripwires
# Migrated verbatim from Main_forGitHub.ipynb cells [57].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 57]
# ----------------------------------------------------------------------
# ================= CELL 4.2-CONFIG (self-sufficient, guard pattern) ==========
# Defines anything not already set this session; never overrides what is.
if "SEEDS"      not in dir(): SEEDS = [42, 123, 7]
TEST_START = "2025-12-01"   # E0/E1 window -- PINNED (other sections set their own)
TEST_END   = "2026-03-01"   # matches methodology 4.8 / E1
if "EPOCHS"     not in dir(): EPOCHS = 25
if "PATIENCE"   not in dir(): PATIENCE = 3
if "BATCH_SIZE" not in dir(): BATCH_SIZE = 32
if "D_MODEL"    not in dir(): D_MODEL = 128
if "N_EXPERTS"  not in dir(): N_EXPERTS = 3
if "ALT_PROGRESSION_MODES" not in dir(): ALT_PROGRESSION_MODES = ["eta", "historical_avg_port"]
if "TARGET_COL" not in dir(): TARGET_COL = "ARR_SUBREGION_ID"
if "USE_DEPARTURE_PORT_CHANNEL" not in dir(): USE_DEPARTURE_PORT_CHANNEL = True

USE_SHIP_HISTORY      = True    # ship-history channel + GAT encoder (Sec 4.2.4)
GATE_SHIP_HISTORY     = True    # zero-init gate gamma on that channel
USE_SHIP_SIZE_CHANNEL = False   # OFF in Lean2
USE_DEPARTURE_GATE    = False   # OFF in Lean2

# hard prerequisites from earlier sections -- fail loudly, not mid-training:
for _req in ("data", "N_CLASSES", "train_residual_progression_variant",
             "evaluate_full_report_metrics", "BucketedWAYDataset"):
    assert _req in dir(), f"'{_req}' missing -- run the data cells / library cells L1-L5min first"


# --- tripwires: fail loudly, before any training ---
import keras
assert keras.backend.backend() == "torch", (
    f"Keras backend is {keras.backend.backend()!r} -- keras was imported before "
    "KERAS_BACKEND was set. Restart runtime, run cell 0.A first.")
for _req in ("data", "N_CLASSES", "train_residual_progression_variant",
             "evaluate_full_report_metrics", "BucketedWAYDataset", "_make_split"):
    assert _req in dir(), f"{_req!r} missing -- run Sections 3.b, 4.2-4.6 and 5.1 first"

print(f"training config: window [{TEST_START} -> {TEST_END}]  seeds {SEEDS}  "
      f"up to {EPOCHS} epochs, patience {PATIENCE}")
