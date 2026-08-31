# =============================================================================
# E8 — cold start (tier-2 pooled)
# Migrated verbatim from Main_forGitHub.ipynb cells [183].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 183]
# ----------------------------------------------------------------------
# WINDOW_BURN_IN_MONTHS = 6
# WINDOW_MONTHS = 3
# exec(open("cold_start_tier2_pooled.py").read())

# =============================================================================
# E8 -- cold-start tier-2 pooled (rolling windows, current base)
# =============================================================================
# Per-window models are TRAINED FRESH by design (each window's model may only
# see data before that window) -- nothing here reloads final_main checkpoints,
# and nothing overwrites them: condition names are cold_start_tier2_*.
assert "BEST_EPOCHS" in globals(), "run E3-PREREQ first (per-seed epochs)"

WINDOW_BURN_IN_MONTHS = 6
WINDOW_MONTHS = 3
SEEDS = [42, 123, 7]
PER_SEED_EPOCHS = dict(BEST_EPOCHS)          # {42:10, 123:10, 7:8} -- the final
                                             # model's val-selected optima
MODEL_VARIANT_TAG = "final_main_lean2"       # fresh names: no collision with any
                                             # old-vintage *_cleaned checkpoints
USE_SHIP_SIZE_CHANNEL = False                # explicit, matching the base
USE_DEPARTURE_PORT_CHANNEL = True
USE_DEPARTURE_GATE = False
ALT_PROGRESSION_MODES = ["eta", "historical_avg_port"]

exec(open("cold_start_tier2_pooled.py").read())
