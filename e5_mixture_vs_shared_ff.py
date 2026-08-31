# =============================================================================
# E5 — mixture vs shared feed-forward
# Migrated verbatim from Main_forGitHub.ipynb cells [160, 161, 162, 163].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 160]
# ----------------------------------------------------------------------
# =============================================================================
# E5-E6 SHARED CONFIG  -- shared foundation for both experiments. (expert-count sweep: latest baseline, TEST-side, fixed epochs
# =============================================================================
assert all(n in globals() for n in ("E3", "_test_result", "BEST_EPOCHS")), \
    "run E3-PREREQ + E3-CONFIG v2 first (this reuses _test_result and BEST_EPOCHS)"

sweep_arch = dict(alt_progression_modes=ALT_PROGRESSION_MODES, gate_ship_history=True,
                  use_ship_history=True, use_departure_gate=USE_DEPARTURE_GATE,
                  stratify=True, stratify_by_pair=False, val_frac=0.15,
                  use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
                  use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
                  test_start=TEST_START, test_end=TEST_END,
                  batch_size=BATCH_SIZE, d_model=D_MODEL, work_dir=WORK_DIR)

K_RESULTS = {}   # n_experts -> {seed: test-result dict}
def run_k(n_experts, seed):
    if n_experts == 3:                       # = the final model: reload
        condition = f"final_main_lean2_seed{seed}"
        kwargs = dict(sweep_arch, epochs=EPOCHS, early_stopping_patience=PATIENCE)
    else:                                    # variants: fixed budget, fresh names
        condition = f"e5_k{n_experts}_final_main_seed{seed}"
        kwargs = dict(sweep_arch, epochs=int(BEST_EPOCHS[seed]),
                      early_stopping_patience=None)
    r = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=condition, seed=seed,
        n_experts=n_experts, skip_existing=True, **kwargs)
    tr = _test_result(r, seed)
    print(f"  {condition}: TEST acc={tr['overall_acc']:.3f}")
    K_RESULTS.setdefault(n_experts, {})[seed] = tr
    return r
print("E5 setup ready (K=3 reloads final_main_lean2_*; variants fixed-epoch)")

# ----------------------------------------------------------------------
# [notebook cell 161]
# ----------------------------------------------------------------------
# =============================================================================
# E5-1 -- MAIN MODEL (MoEFF, K = 3): reload + TEST pass
# =============================================================================
for seed in SEEDS:
    run_k(3, seed)          # stores TEST-side dicts into K_RESULTS internally

# ----------------------------------------------------------------------
# [notebook cell 162]
# ----------------------------------------------------------------------
# =============================================================================
# E5-2 -- NO MIXTURE: use_moe_ffn=False via session class-swap (not K=1)
# =============================================================================
# The trainer hardcodes use_moe_ffn=True; this wrapper forces the plain
# shared-FF path in the CASP layers for the duration of these runs only.
# Everything else identical to the main model (gate inputs become inert on
# the plain path -- the mechanism itself is what is removed).
_OrigWAY = WAYModel
class _NoMoEWAY(_OrigWAY):
    def __init__(self, *a, **k):
        k["use_moe_ffn"] = False
        super().__init__(*a, **k)

NO_MIXTURE = {}
WAYModel = _NoMoEWAY
try:
    for seed in SEEDS:
        cond = f"e5_nomoe_final_main_seed{seed}"
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=seed,
            n_experts=N_EXPERTS,                     # ignored on the plain path
            epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None,
            skip_existing=True, **{k_: v for k_, v in sweep_arch.items()
                                   if k_ not in ("epochs",)})
        NO_MIXTURE[seed] = _test_result(r, seed)
        print(f"  {cond}: TEST acc={NO_MIXTURE[seed]['overall_acc']:.3f}")
finally:
    WAYModel = _OrigWAY                              # restore, unconditionally
print("no-mixture runs done; WAYModel restored")

# ----------------------------------------------------------------------
# [notebook cell 163]
# ----------------------------------------------------------------------
# =============================================================================
# E5 RESULTS -- mixture vs shared feed-forward (TEST, fixed epochs)
# =============================================================================
import numpy as np, pandas as pd
_mix = [K_RESULTS[3][s]["overall_acc"] for s in SEEDS]
_nom = [NO_MIXTURE[s]["overall_acc"] for s in SEEDS]
e5 = pd.DataFrame([
    {"Configuration": "Main model (MoEFF, K = 3)",
     "Mean Overall Accuracy": np.mean(_mix), "Std": np.std(_mix), "Delta": 0.0},
    {"Configuration": "No mixture (shared FF, use_moe_ffn=False)",
     "Mean Overall Accuracy": np.mean(_nom), "Std": np.std(_nom),
     "Delta": np.mean(_nom) - np.mean(_mix)}])
print("--- E5: mixture vs shared feed-forward (TEST, fixed epochs, 3 seeds) ---")
print(e5.to_string(index=False))
e5_stage = summarize_accuracy_by_stage_multiseed({
    "Main model (MoEFF, K = 3)": K_RESULTS[3],
    "No mixture (shared FF)": NO_MIXTURE})
print("\n--- E5: accuracy by voyage stage (TEST) ---")
print(e5_stage.to_string(index=False))
plot_regime_comparison_with_variance(
    {"Main model (MoEFF, K = 3)": [K_RESULTS[3][s]["progression_acc"] for s in SEEDS],
     "No mixture (shared FF)": [NO_MIXTURE[s]["progression_acc"] for s in SEEDS]},
    TARGET_COL, WORK_DIR, save_name="e5_mixture_vs_shared_ff.png",
    ylabel="Test Accuracy (%)",
    title="E5 -- MoEFF (K=3) vs shared feed-forward on TEST, fixed epochs")
e5.to_csv(os.path.join(WORK_DIR, "e5_mixture_vs_shared_ff.csv"), index=False)
