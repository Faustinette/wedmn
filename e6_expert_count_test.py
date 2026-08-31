# =============================================================================
# E6 — expert count on TEST set (+ E5/E6 paired tests)
# Migrated verbatim from Main_forGitHub.ipynb cells [167, 168, 169, 170].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 167]
# ----------------------------------------------------------------------
# =============================================================================
# E6-0 -- K = 3 (main model, reloads final_main_lean2_*)
# =============================================================================
for seed in SEEDS:
    run_k(3, seed)          # stores TEST-side results into K_RESULTS internally

# =============================================================================
# E6-1 -- K = 2
# =============================================================================
for seed in SEEDS:
    run_k(2, seed)

# =============================================================================
# E6-2 -- K = 4
# =============================================================================
for seed in SEEDS:
    run_k(4, seed)

# ----------------------------------------------------------------------
# [notebook cell 168]
# ----------------------------------------------------------------------
# =============================================================================
# E6 RESULTS v2 -- rendered tables via display()
# =============================================================================
import numpy as np, pandas as pd
from IPython.display import display, HTML

_m3 = np.mean([100 * K_RESULTS[3][s]["overall_acc"] for s in SEEDS])
e6 = pd.DataFrame([
    {"Configuration": f"K = {k}" + ("  (main model)" if k == 3 else ""),
     "Mean Overall Accuracy (%)": np.mean([100 * K_RESULTS[k][s]["overall_acc"]
                                           for s in SEEDS]),
     "Std": np.std([100 * K_RESULTS[k][s]["overall_acc"] for s in SEEDS]),
     "Delta vs K=3 (pp)": np.mean([100 * K_RESULTS[k][s]["overall_acc"]
                                   for s in SEEDS]) - _m3}
    for k in sorted(K_RESULTS)]).round(2)

display(HTML("<b>E6 — expert-count sensitivity (TEST, fixed epochs, 3 seeds)</b>"))
display(e6)

e6_stage = summarize_accuracy_by_stage_multiseed(
    {f"K = {k}": K_RESULTS[k] for k in sorted(K_RESULTS)})
display(HTML("<b>E6 — accuracy by voyage stage (TEST)</b>"))
display(e6_stage)

plot_regime_comparison_with_variance(
    {f"K = {k}": [K_RESULTS[k][s]["progression_acc"] for s in SEEDS]
     for k in sorted(K_RESULTS)},
    TARGET_COL, WORK_DIR, save_name="e6_expert_count_sweep.png",
    ylabel="Test Accuracy (%)",
    title="E6 -- expert count K in {2, 3, 4} on TEST, fixed epochs, 3 seeds")

e6.to_csv(os.path.join(WORK_DIR, "e6_expert_count.csv"), index=False)
e6.merge(e6_stage, on="Configuration").to_csv(
    os.path.join(WORK_DIR, "e6_results_table.csv"), index=False)

# ----------------------------------------------------------------------
# [notebook cell 169]
# ----------------------------------------------------------------------
# =============================================================================
# E6 RESULTS -- expert count on TEST: K = 3 vs K = 2 and K = 4
# =============================================================================
import numpy as np, pandas as pd
_m3 = np.mean([K_RESULTS[3][s]["overall_acc"] for s in SEEDS])
e6 = pd.DataFrame([
    {"Configuration": f"K = {k}" + ("  (main model)" if k == 3 else ""),
     "Mean Overall Accuracy": np.mean([K_RESULTS[k][s]["overall_acc"] for s in SEEDS]),
     "Std": np.std([K_RESULTS[k][s]["overall_acc"] for s in SEEDS]),
     "Delta vs K=3": np.mean([K_RESULTS[k][s]["overall_acc"] for s in SEEDS]) - _m3}
    for k in sorted(K_RESULTS)])
print("--- E6: expert-count sensitivity (TEST, fixed epochs, 3 seeds) ---")
print(e6.to_string(index=False))
e6_stage = summarize_accuracy_by_stage_multiseed(
    {f"K = {k}": K_RESULTS[k] for k in sorted(K_RESULTS)})
print("\n--- E6: accuracy by voyage stage (TEST) ---")
print(e6_stage.to_string(index=False))
plot_regime_comparison_with_variance(
    {f"K = {k}": [K_RESULTS[k][s]["progression_acc"] for s in SEEDS]
     for k in sorted(K_RESULTS)},
    TARGET_COL, WORK_DIR, save_name="e6_expert_count_sweep.png",
    ylabel="Test Accuracy (%)",
    title="E6 -- expert count K in {2, 3, 4} on TEST, fixed epochs, 3 seeds")
e6.to_csv(os.path.join(WORK_DIR, "e6_expert_count.csv"), index=False)

# ----------------------------------------------------------------------
# [notebook cell 170]
# ----------------------------------------------------------------------
# =============================================================================
# E5/E6-TESTS -- formal paired tests: mixture vs shared FF, K=3 vs K=2
# =============================================================================
# Same-seed pairing (identical split + budget per pair), TEST side; paired t
# on per-seed deltas + sign consistency, overall AND early band -- the exact
# protocol of the channel-ablation tests (E14-B).
import numpy as np, pandas as pd
from scipy import stats as _st

_bounds = np.array(sorted(DEFAULT_PROGRESSION_BOUNDARIES))
_early = _bounds <= 0.20
def _early_acc(d):
    bc = np.array(d["band_correct"], dtype="float64")
    bt = np.array(d["band_total"], dtype="float64")
    return bc[_early].sum() / max(bt[_early].sum(), 1)

CONTRASTS = [
    ("Mixture vs shared FF", K_RESULTS[3], NO_MIXTURE),
    ("K=3 vs K=2",           K_RESULTS[3], K_RESULTS[2]),
]
rows = []
for name, A, B in CONTRASTS:
    for metric, f in [("overall", lambda d: d["overall_acc"]),
                      ("early band", _early_acc)]:
        d = np.array([f(A[s]) - f(B[s]) for s in SEEDS]) * 100
        t, p = _st.ttest_1samp(d, 0.0)
        rows.append(dict(contrast=name, metric=metric,
                         mean_delta_pp=round(d.mean(), 2),
                         per_seed=[round(x, 2) for x in d],
                         t=round(float(t), 2), p_paired=round(float(p), 4),
                         sign_consistent=bool((d > 0).all() or (d < 0).all())))
mt = pd.DataFrame(rows)
print(mt.to_string(index=False))
mt.to_csv(os.path.join(WORK_DIR, "e5e6_paired_tests.csv"), index=False)
