# =============================================================================
# E9 — non-stationarity / out-of-sample event test
# Migrated verbatim from Main_forGitHub.ipynb cells [185].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 185]
# ----------------------------------------------------------------------
# ===== E9 -- rigorous out-of-sample event test (3 events, filtered v2) =====
REQUIRE_VALIDATION = True     # diagnosis pass: per-event best epochs first
EPOCHS = 25
PATIENCE = 3
MODEL_VARIANT_TAG = "lean2removed"     # lean2 architecture (script default)
exec(open("event_out_of_sample_rigorous.py").read())
