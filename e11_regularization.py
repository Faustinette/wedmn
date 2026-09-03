# E11 — training regime and regularization
# Executed by runner.py inside the shared namespace (notebook-kernel style).


# E11-CONFIG: regularization and loss study (VAL-side, 25 epochs, patience 3)
#
# Regularization choice is a model-selection question, so it is evaluated
# on the validation set, like E0-A, and never on test. The baseline is the
# main model (dropout 0, weight decay 0), reloaded rather than retrained.
#
# Early stopping and the length-balanced loss weighting are always on and
# act as structural regularizers. This section therefore asks two things:
# (a) whether EXPLICIT regularization (dropout, weight decay) adds anything
# on top of them, and (b) what the length weighting itself contributes,
# via a removal ablation.

assert all(n in globals() for n in ("data", "train_residual_progression_variant")),     "run the data + L-cells first"
REG_GRID = [("baseline (none)",        dict(dropout_rate=0.0, weight_decay=0.0)),
            ("dropout 0.1",            dict(dropout_rate=0.1, weight_decay=0.0)),
            ("dropout 0.2",            dict(dropout_rate=0.2, weight_decay=0.0)),
            ("weight decay 1e-4",      dict(dropout_rate=0.0, weight_decay=1e-4)),
            ("dropout 0.1 + wd 1e-4",  dict(dropout_rate=0.1, weight_decay=1e-4))]
X_LABEL = "no length-weighting (loss ablation)"
REG = {}

def _cond_of(label, seed):
    if label.startswith("baseline"):
        return f"final_main_lean2_seed{seed}"
    if label == X_LABEL:
        return f"e16_no_length_weighting_final_main_seed{seed}"
    safe = label.replace(" ", "").replace("+", "_").replace(".", "p")
    return f"e16_{safe}_final_main_seed{seed}"

def run_reg(label, seed, **rk):
    cond = _cond_of(label, seed)
    r = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=cond, seed=seed,
        alt_progression_modes=ALT_PROGRESSION_MODES,
        gate_ship_history=True, use_ship_history=True,
        use_ship_size_channel=False, use_departure_port_channel=True,
        use_departure_gate=False, n_experts=N_EXPERTS, d_model=D_MODEL,
        stratify=True, val_frac=0.15, test_start=TEST_START, test_end=TEST_END,
        epochs=EPOCHS, early_stopping_patience=PATIENCE, batch_size=BATCH_SIZE,
        work_dir=WORK_DIR, skip_existing=True, **rk)
    REG.setdefault(label, {})[seed] = r["metrics"]
    print(f"  {cond}: val acc {r['metrics']['overall_acc']:.4f}")
    return r
print(f"E16 ready: {len(REG_GRID)} grid configs + 1 loss ablation, "
      f"{len(SEEDS)} seeds ({len(REG_GRID)*len(SEEDS)-len(SEEDS)+len(SEEDS)} runs; "
      f"baseline reloads)")


# [notebook cell 190]

# E16-RUN -- the regularizer grid (baseline reloads first)

for label, rk in REG_GRID:
    print("=" * 70); print(label)
    for seed in SEEDS:
        run_reg(label, seed, **rk)


# [notebook cell 191]


# E11-X: loss ablation, length-balanced weighting OFF (function swap, restored below)
#
# This ablates part of the LOSS DEFINITION, not an optional regularizer:
# the trainer hardcodes gradient_dropout_weights (the per-voyage length
# weighting) into both the train and the val loss. For these runs only,
# the global function is swapped for one returning uniform weights, and
# restored afterwards.
#
# WARNING: with uniform weights, val_history values live on a DIFFERENT
# loss scale; never compare those loss numbers against weighted runs.
# Accuracy metrics remain comparable.
#
# Expected effect: strongest on short-voyage and progression-band
# breakdowns, not necessarily on the overall accuracy number.
_orig_gd = gradient_dropout_weights

def _uniform_gd(lengths):
    import numpy as _np
    return _np.ones(len(_np.asarray(lengths)), dtype="float64")

gradient_dropout_weights = _uniform_gd
try:
    for seed in SEEDS:
        run_reg(X_LABEL, seed, dropout_rate=0.0, weight_decay=0.0)
finally:
    gradient_dropout_weights = _orig_gd
print("loss-ablation runs done; gradient_dropout_weights restored")


# [notebook cell 192]


# E16-RESULTS -- one table: grid + loss ablation, stages, epochs, CSV

import numpy as np, pandas as pd, json
from IPython.display import display, HTML
REG_GRID_FULL = REG_GRID + [(X_LABEL, {})]
_base = np.mean([100 * REG["baseline (none)"][s]["overall_acc"] for s in SEEDS])
rows = []
for label, _rk in REG_GRID_FULL:
    accs = [100 * REG[label][s]["overall_acc"] for s in SEEDS]
    eps = []
    for s in SEEDS:
        p = os.path.join(WORK_DIR, "Results",
                         f"{TARGET_COL}_{_cond_of(label, s)}.json")
        if os.path.exists(p):
            vh = json.load(open(p))["val_history"]
            eps.append(int(np.argmin(vh) + 1))
    rows.append({"Configuration": label,
                 "Mean Val Accuracy (%)": np.mean(accs), "Std": np.std(accs),
                 "Delta vs baseline (pp)": np.mean(accs) - _base,
                 "Best epochs (per seed)": eps})
e16 = pd.DataFrame(rows).round(2)
display(HTML("<b>E16 — regularisation grid + loss ablation "
             "(VAL, 25/patience-3, 3 seeds)</b>"))
display(e16)
e16_stage = summarize_accuracy_by_stage_multiseed(
    {label: REG[label] for label, _ in REG_GRID_FULL})
display(HTML("<b>E16 — accuracy by voyage stage (VAL)</b>"))
display(e16_stage)
e16.to_csv(os.path.join(WORK_DIR, "e16_regularization.csv"), index=False)
e16.merge(e16_stage, on="Configuration").to_csv(
    os.path.join(WORK_DIR, "e16_results_table.csv"), index=False)
print("\ncaption note: rows 2-5 audition OPTIONAL regularisers on top of "
      "early stopping; the final row ablates the length-weighted loss "
      "itself (a component of the training objective, not an option).")


# [notebook cell 193]

# Impact of Regularization (REMOUNT)
# exec(open("ablation_6_regularization.py").read())

# ----------------------------------------------------------------------
# [notebook cell 194]
# ----------------------------------------------------------------------
# Modify early stopping (and extend epochs ?)
# exec(open("ablation_6b_regularization_extended.py").read())
