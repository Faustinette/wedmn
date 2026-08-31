# =============================================================================
# Step 5C — parameter estimates
# Migrated verbatim from Main_forGitHub.ipynb cells [105, 106, 107].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 105]
# ----------------------------------------------------------------------
# =============================================================================
# E14-A -- gate scalars across ALL saved fits: estimates-with-tests table
# =============================================================================
# Walks every available checkpoint family (E0 A val-selected, E0 B train+val,
# CV folds when present), reloads each via the trainer (skip_existing -> load,
# never retrain: guarded by meta-JSON existence), extracts the identifiable
# gate scalars by attribute, and prints the report table. Zero-init => H0:
# scale = 0 is the untrained state.
import os, json
import numpy as np, pandas as pd
from keras import ops as _ops

FAMILIES = [("A (val-selected)", "final_main_lean2_seed{s}", SEEDS),
            ("B (train+val)",    "final_main_lean2_trainval_seed{s}", SEEDS)]
if "N_FOLDS" in dir():
    FAMILIES.append(("CV fold", "final_main_lean2_cv{n}_fold{s}_seed42", range(N_FOLDS)))

def _extract(model, fit_label):
    out = []
    for li, layer in enumerate(model.casp_layers):
        sff = getattr(layer, "sff", None)
        if sff is None or getattr(sff, "n_experts", 1) <= 1: continue
        for i, sc in enumerate(getattr(sff, "alt_prog_scales", []) or []):
            nm = ALT_PROGRESSION_MODES[i] if i < len(ALT_PROGRESSION_MODES) else f"alt{i}"
            out.append(dict(fit=fit_label, layer=li, param=f"beta_{nm}",
                            value=float(_ops.convert_to_numpy(sc))))
        cg = getattr(sff, "content_gate_scale", None)
        if cg is not None:
            out.append(dict(fit=fit_label, layer=li, param="beta_content",
                            value=float(_ops.convert_to_numpy(cg))))
    return out

rows, loaded = [], []
for fam, pat, iterable in FAMILIES:
    for s_ in iterable:
        cond = pat.format(s=s_, n=globals().get("N_FOLDS", 5))
        meta = os.path.join(WORK_DIR, "Results", f"{TARGET_COL}_{cond}.json")
        if not os.path.exists(meta):
            print(f"  [skip] {cond} (no meta on disk)"); continue
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=(s_ if fam != "CV fold" else 42),
            alt_progression_modes=ALT_PROGRESSION_MODES,
            gate_ship_history=True, use_ship_history=True,
            use_ship_size_channel=False, use_departure_port_channel=True,
            use_departure_gate=False, n_experts=N_EXPERTS, d_model=D_MODEL,
            batch_size=BATCH_SIZE, epochs=1, early_stopping_patience=None,
            stratify=True, val_frac=0.15, test_start=TEST_START, test_end=TEST_END,
            work_dir=WORK_DIR, skip_existing=True)
        rows += _extract(r["model"], f"{fam}/s{s_}")
        loaded.append(cond)
gp = pd.DataFrame(rows)
print(f"\n{len(loaded)} fits loaded: {loaded}")
summ = gp.groupby(["layer", "param"])["value"].agg(
    mean="mean", std="std", n="count",
    pos=lambda v: int((v > 0).sum()), neg=lambda v: int((v < 0).sum()))
summ["sign_consistent"] = (summ["pos"] == summ["n"]) | (summ["neg"] == summ["n"])
summ["sign_test_p"] = [min(1.0, 2 * 0.5 ** n) if sc else np.nan
                       for n, sc in zip(summ["n"], summ["sign_consistent"])]
print("\n--- gate-parameter estimates across independent fits ---")
print(summ.round(4).to_string())
gp.to_csv(os.path.join(WORK_DIR, "e14_gate_parameters_allfits.csv"), index=False)
print("\nreport line template: 'beta_eta = {mean} +/- {std} across n={n} "
      "independent fits, sign-consistent (sign test p = {p})'")

# ----------------------------------------------------------------------
# [notebook cell 106]
# ----------------------------------------------------------------------
# =============================================================================
# E14-B -- paired ablation tests: are channel deltas larger than seed noise?
# =============================================================================
# Pairing: same seed, with vs without the channel (E3's dicts, TEST-side).
# Paired t across seeds (n=3) + the deltas themselves; small-n caveat printed.
from scipy import stats as _st
_full = E3["overall"]["Full (final model)"]
rows = []
for lb in [l for l in E3["overall"] if l != "Full (final model)"]:
    d = np.array([E3["overall"][lb][s] - _full[s] for s in SEEDS]) * 100
    t, p = _st.ttest_1samp(d, 0.0)
    rows.append(dict(ablation=lb, mean_delta_pp=d.mean().round(2),
                     per_seed=[round(x, 2) for x in d],
                     t=round(float(t), 2), p_paired_t=round(float(p), 4),
                     sign_consistent=bool((d < 0).all() or (d > 0).all())))
at = pd.DataFrame(rows).sort_values("mean_delta_pp")
print(at.to_string(index=False))
at.to_csv(os.path.join(WORK_DIR, "e14_ablation_tests.csv"), index=False)
print("\nn=3 pairs: p-values are indicative; sign consistency + magnitude vs "
      "the seed std carry the argument. State both in the report.")

# ----------------------------------------------------------------------
# [notebook cell 107]
# ----------------------------------------------------------------------
# =============================================================================
# E14-C (optional) -- classical anchor: multinomial logistic baseline
# =============================================================================
# A model where classical inference IS valid. Step-level features, segment-
# grouped split; sklearn for the accuracy row, statsmodels on a subsample for
# a proper coefficient/Wald table.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
tj = data.traj_idx.set_index("seg_id")
_dep_sub = {int(s): int(port_to_sub.get(tj.loc[s, "DEP_PORT_ID"], -1))
            for s in np.unique(seg)}
X = pd.DataFrame({
    "frac": frac,
    "dep_sub": [_dep_sub.get(int(s), -1) for s in seg],
    "declared": _cap if "_cap" in dir() else -1,     # from E13-B, else refused
    "month": [pd.Timestamp(tj.loc[int(s), "dep_ts"]).month for s in seg]})
y = true
Xe = pd.get_dummies(X, columns=["dep_sub", "declared", "month"], dtype=float)
rng2 = np.random.default_rng(0)
segs_u = np.unique(seg); rng2.shuffle(segs_u)
tr_segs = set(segs_u[: int(0.8 * len(segs_u))])
tr = np.isin(seg, list(tr_segs)); te = ~tr
lr = LogisticRegression(max_iter=200, multi_class="multinomial", C=1.0)
lr.fit(Xe[tr], y[tr])
acc = lr.score(Xe[te], y[te])
print(f"multinomial logistic baseline (interpretable features): "
      f"held-out step accuracy {100*acc:.2f}%  "
      f"(deep model val pooled: {100*(pred==true).mean():.2f}%)")
print("features:", list(X.columns), "-> one-hot dim", Xe.shape[1])
print("for the coefficient/Wald table: statsmodels MNLogit on a 20k-step "
      "segment-stratified subsample -- run separately (minutes).")
