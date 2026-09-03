# E10 — fleet effect (exp1b final control)
# Migrated verbatim from Main_forGitHub.ipynb cells [187].
# Executed by runner.py inside the shared namespace (notebook-kernel style).


# [notebook cell 187]


# E10 — FLEET EFFECT (final control)
#
# Tests whether fleet-level context, the concurrent behavior of other
# vessels, contributes to prediction accuracy beyond the single-vessel
# signals used by the main model. Run as a controlled comparison against
# the main configuration: same data split, seeds and training budget, with
# only the fleet-context input varied, so any accuracy difference is
# attributable to the fleet signal itself.
#
# Implementation lives in exp1b_final_control_cell.py ; see README, "Scripts
# still exec'd by three experiments").
# Executed by runner.py inside the shared namespace after the trained-model
# stage, so the script can assume the dataset, training library and main
# checkpoints are available.

exec(open("exp1b_final_control_cell.py").read())
