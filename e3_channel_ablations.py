# E3: SINGLE CHANNEL ABLATIONS (TEST SIDE)
# E19: PAIRWISE CHANNEL ABLATIONS (INTERACTION STRUCTURE)
# H7: SHIP HISTORY ABLATION UNDER 5 FOLD CROSS VALIDATION
#
# CONTENTS, IN EXECUTION ORDER
#
#   Step  Label     What it does                              Output
#   1     PREREQ    Recover per seed best epochs from the     BEST_EPOCHS,
#                   saved main model meta files.              EPOCHS_RUN
#   2     E3 RUNS   Five single channel ablations, one        E3 dict entries,
#                   training (or checkpoint reload) each:     checkpoints
#                   spatial, local pattern, departure port,
#                   ship history, temporal encoding.
#   3     E3 TABLE  Ablation results table, stage plot.       e3_channel_
#                                                             ablation_test.csv
#                                                             (+ _stages.csv,
#                                                             .png)
#   4     E19 RUNS  Joint removal of each channel pair,       E19 dict,
#                   same protocol and budgets as E3.          checkpoints
#   5     E19 TABLE Interaction table: joint delta minus      e19_channel_
#                   the sum of single deltas, per pair,       interactions.csv
#                   with per seed t test.
#   6     H7        No history variant retrained per CV       console summary
#                   fold, fold paired against the CV
#                   baseline; sign test and combined
#                   8 contrast statement.
#
# PREREQUISITES, expected in the shared namespace before this file runs:
# the E3 configuration block (run_ablation, e3_table, _test_result, E3,
# plot_regime_comparison_with_variance and the loaded baseline) from
# experiments/e3_prereq.py, the trained main model conditions from E0,
# and, for step 6 only, fold_ids / fold_of / CV_SEED from the cross
# validation stage.
#
# All trainings use skip_existing=True: once a condition's checkpoint
# exists under Results/, it is reloaded, not retrained.



###############################################################################
# STEP 1. PREREQ: recover per seed best epochs from the saved metas
#
# Self sufficient: reads the final_main_lean2_* meta JSON files written by
# E0 A. best_epoch is the argmin of the saved validation loss history (the
# validation selected optimum); EPOCHS_RUN records how long each run
# actually trained before early stopping.

import json, numpy as np
BEST_EPOCHS, EPOCHS_RUN = {}, {}
for seed in SEEDS:
    meta_path = os.path.join(WORK_DIR, "Results",
                             f"{TARGET_COL}_final_main_lean2_seed{seed}.json")
    assert os.path.exists(meta_path), \
        f"missing {os.path.basename(meta_path)}; run E0 A (training) first"
    with open(meta_path) as f:
        meta = json.load(f)
    vh = meta["val_history"]
    BEST_EPOCHS[seed] = int(np.argmin(vh) + 1)
    EPOCHS_RUN[seed] = len(meta["history"])
    print(f"seed {seed}: epochs_run={EPOCHS_RUN[seed]}  best_epoch={BEST_EPOCHS[seed]}  "
          f"(best val_loss={min(vh):.4f})")
assert all(BEST_EPOCHS.get(s) for s in SEEDS)
print(f"\nBEST_EPOCHS for fixed-budget ablations: {BEST_EPOCHS}")


###############################################################################
# STEP 2. E3 RUNS: the five single channel ablations
#
# One run_ablation call per removed channel. Each trains 3 seeds at the
# per seed best epoch budget from step 1 (or reloads the finished
# checkpoints) and stores TEST side results into the E3 dict under the
# label given as first argument.

run_ablation("No spatial channel", condition_stem="base_channel_ablation_no_spatial_channel", use_spatial_channel=False)

run_ablation("No local pattern channel", condition_stem="base_channel_ablation_no_local_pattern_channel", use_local_pattern_channel=False)

run_ablation("No departure port channel", condition_stem="base_channel_ablation_no_departure_port_channel", use_departure_port_channel=False)

run_ablation("No ship history", condition_stem="channel_ablation_no_ship_history", gate_ship_history=False, use_ship_history=False)

run_ablation("No temporal encoding", condition_stem="base_channel_ablation_no_temporal_encoding",
             use_temporal_encoding=False)


###############################################################################
# STEP 3. E3 TABLE: results for the four channels plus temporal encoding
#
# Builds the ablation table and the stage resolved comparison plot, then
# saves both to CSV (overall and per stage) and the figure to PNG.

E3_ABLATIONS = ["No spatial channel", "No local pattern channel",
                "No departure port channel", "No ship history",
                "No temporal encoding"]

t3, s3 = e3_table(E3_ABLATIONS, "E3 channel + temporal-encoding ablation (TEST)")

plot_regime_comparison_with_variance(
    {lb: list(E3["prog"][lb].values()) for lb in E3_ABLATIONS},
    TARGET_COL, WORK_DIR, save_name="e3_channel_ablation_test.png",
    ylabel="Test Accuracy (%)",
    title="E3 channel ablations on TEST, fixed epochs, 3 seeds, mean \u00b1 std")

t3.to_csv(os.path.join(WORK_DIR, "e3_channel_ablation_test.csv"), index=False)
s3.to_csv(os.path.join(WORK_DIR, "e3_channel_ablation_test_stages.csv"), index=False)
print("\nSaved: e3_channel_ablation_test.csv + _stages.csv")


###############################################################################
# STEP 4. E19 RUNS: pairwise channel ablations
#
# Joint removal of every channel pair, TEST side, same fixed per seed
# budgets as E3. The interaction sign convention used in step 5:
#   interaction near 0: channels contribute independently (additive)
#   interaction above 0: redundant pair (each covers for the other, so
#     the joint loss is smaller than the additive prediction)
#   interaction below 0: synergistic pair (joint loss exceeds the parts)

import numpy as np, pandas as pd
from itertools import combinations
from IPython.display import display, HTML

CH = {  # label to trainer kwargs delta (aux channels; add spatial if wanted)
    "local":    dict(use_local_pattern_channel=False),
    "depport":  dict(use_departure_port_channel=False),
    "history":  dict(use_ship_history=False, gate_ship_history=False),
    "temporal": dict(use_temporal_encoding=False),
}
_base_kw = dict(alt_progression_modes=ALT_PROGRESSION_MODES,
                gate_ship_history=True, use_ship_history=True,
                use_ship_size_channel=False, use_departure_port_channel=True,
                use_departure_gate=False, n_experts=N_EXPERTS, d_model=D_MODEL,
                stratify=True, val_frac=0.15, test_start=TEST_START,
                test_end=TEST_END, batch_size=BATCH_SIZE,
                work_dir=WORK_DIR, skip_existing=True)

E19 = {}
for a, b in combinations(CH, 2):
    E19[(a, b)] = {}
    for s_ in SEEDS:
        cond = f"e19_no_{a}_no_{b}_final_main_seed{s_}"
        kw = dict(_base_kw, epochs=int(BEST_EPOCHS[s_]),
                  early_stopping_patience=None)
        kw.update(CH[a]); kw.update(CH[b])
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=s_, **kw)
        E19[(a, b)][s_] = _test_result(r, s_)
        print(f"  {cond}: TEST {E19[(a,b)][s_]['overall_acc']:.3f}")


###############################################################################
# STEP 5. E19 TABLE: interaction results
#
# For each pair: joint delta versus the full model, the additive
# prediction (sum of the two single ablation deltas from E3), and their
# difference (the interaction), with a per seed one sample t test and a
# sign consistency flag. E3 dictionary keys are matched by substring
# probe (_find) so the block tolerates label wording differences, and
# _acc tolerates both {seed: float} and {seed: {"overall_acc": ...}}
# entry shapes. Saves e19_channel_interactions.csv.

import numpy as np, pandas as pd
from scipy import stats as _st
from IPython.display import display, HTML

print("E3['overall'] keys:", list(E3["overall"].keys()))

def _find(sub):
    hits = [k for k in E3["overall"] if sub.lower() in k.lower()]
    assert len(hits) == 1, f"'{sub}' matched {hits}; adjust the probe strings"
    return hits[0]

FULL_KEY = _find("full")
SINGLE = {"local": _find("local"), "depport": _find("depart"),
          "history": _find("history"), "temporal": _find("temporal")}
print("matched:", FULL_KEY, "|", SINGLE)

def _acc(entry, s):          # tolerate {seed: float} or {seed: {"overall_acc": ...}}
    v = entry[s]
    return v["overall_acc"] if isinstance(v, dict) else v

_full = {s: _acc(E3["overall"][FULL_KEY], s) for s in SEEDS}
rows = []
for (a, b), res in E19.items():
    d_join = np.array([res[s]["overall_acc"] - _full[s] for s in SEEDS]) * 100
    d_a = np.array([_acc(E3["overall"][SINGLE[a]], s) - _full[s] for s in SEEDS]) * 100
    d_b = np.array([_acc(E3["overall"][SINGLE[b]], s) - _full[s] for s in SEEDS]) * 100
    inter = d_join - (d_a + d_b)
    t, p = _st.ttest_1samp(inter, 0.0)
    rows.append(dict(pair=f"{a}+{b}",
                     joint_delta=round(d_join.mean(), 2),
                     additive_pred=round((d_a + d_b).mean(), 2),
                     interaction=round(inter.mean(), 2),
                     per_seed_inter=[round(x, 2) for x in inter],
                     t=round(float(t), 2), p=round(float(p), 3),
                     sign_consistent=bool((inter > 0).all() or (inter < 0).all())))
e19 = pd.DataFrame(rows).sort_values("interaction")
print("E19 pairwise channel interactions (TEST, 3 seeds; "
      "interaction = joint minus additive; above 0 redundant, below 0 synergistic)")
print(e19.to_string(index=False))
e19.to_csv(os.path.join(WORK_DIR, "e19_channel_interactions.csv"), index=False)


###############################################################################
# STEP 6. H7: ship history ablation under 5 fold cross validation
#
# Per fold: train the no history variant on the same fold split as the
# cross validation baseline (which reloads), validation side, same
# training regime (fold self selects its stopping epoch under the shared
# epochs and patience settings). Reports fold paired deltas with a t test
# and a sign test, then the combined statement over all 8 contrasts
# (3 seeds test side from E3 plus 5 folds validation side here).

import numpy as np, pandas as pd
from scipy import stats as _st
assert "fold_ids" in globals() and "fold_of" in globals(), "run 4.2-CV first"
assert CV_SEED in (42, 123), "CV_SEED must match a trained baseline family"

H7 = {"base": {}, "nohist": {}}
for k in range(N_FOLDS):
    val_k = fold_ids[k]; train_k = set(fold_of) - val_k
    for arm, cond, delta in [
        ("base", f"final_main_lean2_cv{N_FOLDS}_fold{k}_seed{CV_SEED}", {}),
        ("nohist", f"h7_nohist_cv{N_FOLDS}_fold{k}_seed{CV_SEED}",
         dict(use_ship_history=False, gate_ship_history=False)),
    ]:
        kw = dict(alt_progression_modes=ALT_PROGRESSION_MODES,
                  gate_ship_history=True, use_ship_history=True,
                  use_ship_size_channel=False, use_departure_port_channel=True,
                  use_departure_gate=False, n_experts=N_EXPERTS,
                  d_model=D_MODEL, batch_size=BATCH_SIZE,
                  epochs=EPOCHS, early_stopping_patience=PATIENCE,
                  train_ids_override=train_k, val_ids_override=val_k,
                  work_dir=WORK_DIR, skip_existing=True)
        kw.update(delta)
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond,
            seed=CV_SEED, **kw)
        H7[arm][k] = r["metrics"]
        print(f"  fold {k} {arm}: val acc {r['metrics']['overall_acc']:.4f}")

d = np.array([H7["nohist"][k]["overall_acc"] - H7["base"][k]["overall_acc"]
              for k in range(N_FOLDS)]) * 100
t, p = _st.ttest_1samp(d, 0.0)
neg = int((d < 0).sum())
print(f"\nfold-paired delta (no-history \u2212 base): {d.mean():+.2f}pp "
      f"(per-fold {d.round(2).tolist()})")
print(f"paired t = {t:.2f}, p = {p:.4f}; negative in {neg}/{N_FOLDS} folds")
print(f"sign test p = {2*0.5**N_FOLDS:.4f} if {N_FOLDS}/{N_FOLDS} negative")
print(f"combined: with 3 test-side contrasts, {neg + 3}/8 negative "
      f"gives sign p = {2*0.5**(neg+3):.4f} if all agree")
