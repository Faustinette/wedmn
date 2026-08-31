# =============================================================================
# E0-A — Main training (3 seeds, early-stopped; skip_existing reloads) + BEST_EPOCHS bridge + val accuracy by 5% bin
# Migrated verbatim from Main_forGitHub.ipynb cells [58, 59, 60].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 58]
# ----------------------------------------------------------------------
# NEED TO FILL
# =============================================================================
# SECTION 4.2 -- MAIN MODEL TRAINING (paste after LIB CELL L5-MIN)
# =============================================================================


# ================= CELL 4.2-CONFIG-LITE ======================================
# Only the names the notebook does NOT already define (verified against the
# live Main.ipynb). Everything else -- SEEDS, EPOCHS=25, PATIENCE=3,
# BATCH_SIZE, D_MODEL, N_EXPERTS, ALT_PROGRESSION_MODES, TARGET_COL,
# TEST_START/TEST_END, USE_DEPARTURE_PORT_CHANNEL -- is already in session.
USE_SHIP_HISTORY      = True    # ship-history channel + GAT encoder (Sec 4.2.4)
GATE_SHIP_HISTORY     = True    # zero-init gate gamma on that channel
USE_SHIP_SIZE_CHANNEL = False   # OFF in Lean2
USE_DEPARTURE_GATE    = False   # OFF in Lean2

print(f"training config: window [{TEST_START} -> {TEST_END}]  seeds {SEEDS}  "
      f"up to {EPOCHS} epochs, patience {PATIENCE}")


# ================= CELL 4.2a -- TRAIN, 3 SEEDS, EARLY-STOPPED ================
# Per seed: up to EPOCHS with live early stopping (PATIENCE) against the
# stratified 15% validation split; best-val-loss epoch checkpointed and
# reloaded. skip_existing=True -> a finished condition reloads from disk.
runs = {}
for seed in SEEDS:
    condition = f"final_main_lean2_seed{seed}"
    print("=" * 70); print(f"SEED {seed}  ->  {condition}"); print("=" * 70)
    runs[seed] = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=condition,
        alt_progression_modes=ALT_PROGRESSION_MODES,
        gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
        use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
        use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
        use_departure_gate=USE_DEPARTURE_GATE,
        n_experts=N_EXPERTS, d_model=D_MODEL,
        stratify=True, val_frac=0.15, seed=seed,
        test_start=TEST_START, test_end=TEST_END,
        epochs=EPOCHS, early_stopping_patience=PATIENCE,
        batch_size=BATCH_SIZE,
        work_dir=WORK_DIR, skip_existing=True,
    )


# ================= CELL 4.2b -- VALIDATION RESULTS (overall + by band) =======
# evaluate_full_report_metrics returns {band_label: metrics} with "Overall"
# as the last key (Early 0-20% / Mid 20-60% / Late 60-100% / Overall).
import pandas as pd

METRIC_COLS = ["standard_accuracy", "balanced_accuracy", "f1_macro"]
val_results = {}
overall_rows, band_rows = [], []
BEST_EPOCHS = {}
for seed in SEEDS:
    r = runs[seed]
    res = evaluate_full_report_metrics(
        r["model"], r["repr_layer"], r["val_loader"], r["core_and_alt_fn"], N_CLASSES,
        departure_ids_fn=r.get("departure_ids_fn"))
    val_results[seed] = res
    BEST_EPOCHS[seed] = r.get("best_epoch")
    ov = res["Overall"]
    overall_rows.append(dict(seed=seed,
                             epochs_run=r.get("epochs_run"), best_epoch=r.get("best_epoch"),
                             **{k: round(ov[k] * 100, 2) for k in METRIC_COLS}))
    for band, m in res.items():
        if m is not None:
            band_rows.append(dict(seed=seed, band=band,
                                  accuracy=round(m["standard_accuracy"] * 100, 2)))

overall_df = pd.DataFrame(overall_rows)
print("=" * 70); print("VALIDATION RESULTS -- OVERALL (per-step accuracy)"); print("=" * 70)
print(overall_df.to_string(index=False))
print("\nmean:", overall_df[METRIC_COLS].mean().round(2).to_dict())

band_order = [b for b, *_ in (("Early (0-20%)",), ("Mid (20-60%)",), ("Late (60-100%)",), ("Overall",))]
bands_df = (pd.DataFrame(band_rows)
            .pivot_table(index="band", columns="seed", values="accuracy")
            .reindex([b for b in band_order if b in {r["band"] for r in band_rows}]))
bands_df["mean"] = bands_df.mean(axis=1).round(2)
print("\n" + "=" * 70); print("VALIDATION ACCURACY ALONG THE TRAJECTORY (by progression band)"); print("=" * 70)
print(bands_df.to_string())

print(f"\nBEST_EPOCHS (feeds the final train+val retrain): {BEST_EPOCHS}")

# ----------------------------------------------------------------------
# [notebook cell 59]
# ----------------------------------------------------------------------
# ===== E0 A -> B BRIDGE: recover BEST_EPOCHS from the saved meta JSONs =====
import json, numpy as np
BEST_EPOCHS, EPOCHS_RUN = {}, {}
for seed in SEEDS:
    meta_path = os.path.join(WORK_DIR, "Results",
                             f"{TARGET_COL}_final_main_lean2_seed{seed}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    vh = meta["val_history"]
    BEST_EPOCHS[seed] = int(np.argmin(vh) + 1)
    EPOCHS_RUN[seed] = len(meta["history"])
    print(f"seed {seed}: epochs_run={EPOCHS_RUN[seed]}  "
          f"best_epoch={BEST_EPOCHS[seed]}  (best val_loss={min(vh):.4f})")
assert all(BEST_EPOCHS.get(s) for s in SEEDS)
print(f"\nBEST_EPOCHS recovered: {BEST_EPOCHS}")

# ----------------------------------------------------------------------
# [notebook cell 60]
# ----------------------------------------------------------------------
# ===== E0 A -- VALIDATION ACCURACY BY 5% PROGRESSION BIN =====
import pandas as pd
labels_5pct = _progression_labels(DEFAULT_PROGRESSION_BOUNDARIES)
rows = {}
for seed in SEEDS:
    pa = runs[seed]["metrics"]["progression_acc"]
    rows[seed] = [round(float(a) * 100, 2) for a in pa]
fine = pd.DataFrame(rows, index=labels_5pct[:len(next(iter(rows.values())))])
fine["mean"] = fine.mean(axis=1).round(2)
print("=" * 70); print("E0 A -- VALIDATION ACCURACY, 5% PROGRESSION BINS"); print("=" * 70)
print(fine.to_string())
fine.to_csv(os.path.join(WORK_DIR, "e0a_val_accuracy_5pct.csv"))
