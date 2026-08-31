# =============================================================================
# E0-C — Test-set evaluation (trained on train+val block) + test accuracy by 5% bin
# Migrated verbatim from Main_forGitHub.ipynb cells [64, 65].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 64]
# ----------------------------------------------------------------------
# =============================================================================
# E0 B -- MAIN TRAINING: train+val combined, evaluated ONCE on the TEST set
# Paste as two cells after E0 A. Requires from E0 A's session: BEST_EPOCHS,
# runs, and the pinned TEST_START/TEST_END (2025-12-01 -> 2026-03-01).
# =============================================================================


# ================= CELL E0B-1 -- FINAL RETRAIN (train+val, fixed epochs) =====
# The final-model protocol (methodology 4.5.3 / 4.6.1): per-seed epoch count
# fixed at the value early stopping selected in E0 A; training on the FULL
# pre-TEST_START pool (train+val combined) via train_ids_override; no early
# stopping -- there is no honest validation set left (the val ids passed
# below sit inside the training pool; with fixed epochs they decide nothing).
assert "BEST_EPOCHS" in dir() and all(BEST_EPOCHS.get(s) for s in SEEDS), \
    "BEST_EPOCHS missing/incomplete -- run E0 A (training + results cells) first"

final_runs = {}
for seed in SEEDS:
    train_ids, val_ids, test_ids = _make_split(
        data, TARGET_COL, val_frac=0.15, seed=seed, stratify=True,
        test_start=TEST_START, test_end=TEST_END)
    combined_ids = list(train_ids) + list(val_ids)
    condition = f"final_main_lean2_trainval_seed{seed}"
    print("=" * 70)
    print(f"SEED {seed}  ->  {condition}   fixed epochs = {BEST_EPOCHS[seed]}   "
          f"training on {len(combined_ids):,} segments (train+val combined)")
    print("=" * 70)
    final_runs[seed] = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=condition,
        alt_progression_modes=ALT_PROGRESSION_MODES,
        gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
        use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
        use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
        use_departure_gate=USE_DEPARTURE_GATE,
        n_experts=N_EXPERTS, d_model=D_MODEL,
        stratify=True, val_frac=0.15, seed=seed,
        test_start=TEST_START, test_end=TEST_END,
        epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None,
        train_ids_override=combined_ids, val_ids_override=list(val_ids),
        batch_size=BATCH_SIZE,
        work_dir=WORK_DIR, skip_existing=True,
    )
    final_runs[seed]["_test_ids"] = list(test_ids)


# ================= CELL E0B-2 -- THE SINGLE TEST EVALUATION (E1 numbers) =====
# The withheld window [TEST_START -> TEST_END] is scored HERE, ONCE.
# Do not iterate on anything after seeing these numbers (methodology 4.6.1).
import os
import pandas as pd

METRIC_COLS = ["standard_accuracy", "balanced_accuracy", "f1_macro"]
test_overall_rows, test_band_rows = [], []
for seed in SEEDS:
    r = final_runs[seed]
    test_loader = BucketedWAYDataset(
        data, target_col=TARGET_COL, batch_size=BATCH_SIZE,
        seg_id_subset=r["_test_ids"], shuffle=False, seed=seed,
        include_ship_history=USE_SHIP_HISTORY)
    res = evaluate_full_report_metrics(
        r["model"], r["repr_layer"], test_loader, r["core_and_alt_fn"], N_CLASSES,
        departure_ids_fn=r.get("departure_ids_fn"),
        eta_channel_lookup=r.get("eta_channel_lookup"))
    ov = res["Overall"]
    test_overall_rows.append(dict(seed=seed, epochs_trained=int(BEST_EPOCHS[seed]),
                                  **{k: round(ov[k] * 100, 2) for k in METRIC_COLS}))
    for band, m in res.items():
        if m is not None:
            test_band_rows.append(dict(seed=seed, band=band,
                                       accuracy=round(m["standard_accuracy"] * 100, 2)))

test_df = pd.DataFrame(test_overall_rows)
print("=" * 70)
print(f"E0 B -- FINAL MODEL, TEST SET [{TEST_START} -> {TEST_END}]")
print("=" * 70)
print(test_df.to_string(index=False))
print("\nmean:", test_df[METRIC_COLS].mean().round(2).to_dict())

_order = ["Early (0-20%)", "Mid (20-60%)", "Late (60-100%)", "Overall"]
bands_df = (pd.DataFrame(test_band_rows)
            .pivot_table(index="band", columns="seed", values="accuracy")
            .reindex([b for b in _order if b in {r["band"] for r in test_band_rows}]))
bands_df["mean"] = bands_df.mean(axis=1).round(2)
print("\n" + "=" * 70)
print("E0 B -- TEST ACCURACY ALONG THE TRAJECTORY (by progression band)")
print("=" * 70)
print(bands_df.to_string())

test_df.to_csv(os.path.join(WORK_DIR, "final_main_lean2_TEST_results.csv"), index=False)
print("\nSaved -> final_main_lean2_TEST_results.csv")

# ----------------------------------------------------------------------
# [notebook cell 65]
# ----------------------------------------------------------------------
# ===== E0 B -- TEST ACCURACY BY 5% PROGRESSION BIN (corrected) =====
import numpy as np, pandas as pd
bounds = list(DEFAULT_PROGRESSION_BOUNDARIES)
rows = {}
for seed in SEEDS:
    r = final_runs[seed]
    test_loader = BucketedWAYDataset(data, target_col=TARGET_COL, batch_size=BATCH_SIZE,
                                     seg_id_subset=r["_test_ids"], shuffle=False, seed=seed,
                                     include_ship_history=USE_SHIP_HISTORY)
    _seg, _true, _pred, _probs, _frac = _collect_full_predictions(
        r["model"], r["repr_layer"], test_loader, r["core_and_alt_fn"],
        departure_ids_fn=r.get("departure_ids_fn"),
        eta_channel_lookup=r.get("eta_channel_lookup"))
    band = np.clip(np.digitize(_frac, bounds), 0, len(bounds) - 1)
    correct = (_true == _pred)
    rows[seed] = [round(correct[band == b].mean() * 100, 2) if (band == b).any() else np.nan
                  for b in range(len(bounds))]
fine_t = pd.DataFrame(rows, index=_progression_labels(bounds))
fine_t["mean"] = fine_t.mean(axis=1).round(2)
print("=" * 70); print(f"E0 B -- TEST ACCURACY, 5% BINS  [{TEST_START} -> {TEST_END}]"); print("=" * 70)
print(fine_t.to_string())
fine_t.to_csv(os.path.join(WORK_DIR, "e0b_test_accuracy_5pct.csv"))
