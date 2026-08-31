# =============================================================================
# E6 — expert count, cross-validation
# Migrated verbatim from Main_forGitHub.ipynb cells [165].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 165]
# ----------------------------------------------------------------------
# =============================================================================
# E6-CV -- expert count under 5-fold cross-validation (train+val block only)
# =============================================================================
# Same folds as CELL 4.2-CV (asserted below). Per K: one model per fold,
# validated on the held-out fold -- folds are the replicates. K=3 RELOADS the
# CV checkpoints already trained by 4.2-CV; K=2/4 train fresh per fold.
# Regime matches the CV baseline (25 epochs, patience 3, fold self-selects);
# TEST is never touched anywhere in this cell.
import numpy as np, pandas as pd
from IPython.display import display, HTML
assert "fold_ids" in globals() and "CV_SEED" in globals(), \
    "run CELL 4.2-CV first (this reuses its stratified folds)"

K_CV = {}    # K -> {fold: val metrics}
for K in (3, 2, 4, 1 , 5):
    K_CV[K] = {}
    for k in range(N_FOLDS):
        val_k = fold_ids[k]; train_k = set(fold_of) - val_k
        cond = (f"final_main_lean2_cv{N_FOLDS}_fold{k}_seed{CV_SEED}" if K == 3
                else f"e6cv_k{K}_cv{N_FOLDS}_fold{k}_seed{CV_SEED}")
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=CV_SEED,
            alt_progression_modes=ALT_PROGRESSION_MODES,
            gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
            use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
            use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
            use_departure_gate=USE_DEPARTURE_GATE, n_experts=K,
            d_model=D_MODEL, batch_size=BATCH_SIZE,
            epochs=EPOCHS, early_stopping_patience=PATIENCE,
            train_ids_override=train_k, val_ids_override=val_k,
            work_dir=WORK_DIR, skip_existing=True)
        K_CV[K][k] = r["metrics"]
        print(f"  K={K} fold {k}: val acc {r['metrics']['overall_acc']:.4f}")

_m3 = np.mean([100 * K_CV[3][k]["overall_acc"] for k in range(N_FOLDS)])
e6cv = pd.DataFrame([
    {"Configuration": f"K = {K}" + ("  (main model)" if K == 3 else ""),
     "Mean Overall Accuracy (%)": np.mean([100 * K_CV[K][k]["overall_acc"]
                                           for k in range(N_FOLDS)]),
     "Std (folds)": np.std([100 * K_CV[K][k]["overall_acc"]
                            for k in range(N_FOLDS)]),
     "Delta vs K=3 (pp)": np.mean([100 * K_CV[K][k]["overall_acc"]
                                   for k in range(N_FOLDS)]) - _m3,
     "Per-fold deltas": [round(100 * (K_CV[K][k]["overall_acc"]
                          - K_CV[3][k]["overall_acc"]), 2)
                         for k in range(N_FOLDS)]}
    for K in sorted(K_CV)]).round(2)

display(HTML(f"<b>E6-CV — expert count, {N_FOLDS}-fold CV on the train+val "
             f"pool (val-side, fold-paired)</b>"))
display(e6cv)
e6cv_stage = summarize_accuracy_by_stage_multiseed(
    {f"K = {K}": K_CV[K] for K in sorted(K_CV)})
display(HTML("<b>E6-CV — accuracy by voyage stage</b>"))
display(e6cv_stage)
e6cv.to_csv(os.path.join(WORK_DIR, "e6cv_expert_count.csv"), index=False)
