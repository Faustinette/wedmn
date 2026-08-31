# =============================================================================
# E0-B — k-fold cross-validation on the train+val pool
# Migrated verbatim from Main_forGitHub.ipynb cells [62].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 62]
# ----------------------------------------------------------------------
# ================= CELL 4.2-CV -- k-fold cross-validation on the train+val pool
# Same pool as the headline split (departures strictly before TEST_START, the
# 1,125 post-TEST_END segments excluded); stratified by arrival subregion.
# Each fold: train on the other k-1 folds, validate on the held-out fold,
# same regime (25 epochs / patience 3). Complements the 3-seed protocol:
# seeds vary initialization on ONE split; CV varies the SPLIT itself.
import numpy as np, pandas as pd

N_FOLDS = 5
CV_SEED = 123                       # one seed across folds; folds are the replicates

tj = data.traj_idx.dropna(subset=[TARGET_COL]).copy()
tj["dep_ts"] = pd.to_datetime(tj["dep_ts"])
pool = tj[tj["dep_ts"] < pd.Timestamp(TEST_START)]
rng = np.random.default_rng(CV_SEED)
fold_of = {}
for _cls, g in pool.groupby(TARGET_COL):           # stratified assignment
    ids = g["seg_id"].astype(int).to_numpy()
    rng.shuffle(ids)
    for j, sid in enumerate(ids):
        fold_of[sid] = j % N_FOLDS
fold_ids = {k: {s for s, f in fold_of.items() if f == k} for k in range(N_FOLDS)}
print(f"pool {len(fold_of):,} segments -> {N_FOLDS} stratified folds "
      f"({[len(fold_ids[k]) for k in range(N_FOLDS)]})")

CV_RESULTS = {}
for k in range(N_FOLDS):
    val_k = fold_ids[k]
    train_k = set(fold_of) - val_k
    cond = f"final_main_lean2_cv{N_FOLDS}_fold{k}_seed{CV_SEED}"
    print("=" * 70); print(f"FOLD {k}: train {len(train_k):,}  val {len(val_k):,}  -> {cond}")
    r = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=cond, seed=CV_SEED,
        alt_progression_modes=ALT_PROGRESSION_MODES,
        gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
        use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
        use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
        use_departure_gate=USE_DEPARTURE_GATE, n_experts=N_EXPERTS,
        d_model=D_MODEL, batch_size=BATCH_SIZE,
        epochs=EPOCHS, early_stopping_patience=PATIENCE,
        train_ids_override=train_k, val_ids_override=val_k,
        work_dir=WORK_DIR, skip_existing=True)
    CV_RESULTS[k] = r["metrics"]
    print(f"  fold {k}: val overall_acc = {r['metrics']['overall_acc']:.4f}")

accs = np.array([CV_RESULTS[k]["overall_acc"] * 100 for k in range(N_FOLDS)])
print("\n" + "=" * 70)
print(f"{N_FOLDS}-fold CV: {accs.round(2).tolist()}")
print(f"mean {accs.mean():.2f}  std {accs.std():.2f}  "
      f"(3-seed single-split reference: 82.04 \u00b1 1.03)")
