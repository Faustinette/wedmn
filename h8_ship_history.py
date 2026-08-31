# =============================================================================
# H8 — ship-history contribution analysis (H1 damage-by-depth, H8 by slice)
# Migrated verbatim from Main_forGitHub.ipynb cells [150, 151, 152, 153, 154, 155, 156, 157, 158].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 150]
# ----------------------------------------------------------------------
# =============================================================================
# H0-PREREQ -- history depth & staleness lookups (test segments)
# =============================================================================
# Prior-voyage count, days since last voyage, per test segment (from traj_idx
# alone). Requires E17-PREREQ arrays (tseg/ttrue/tpred/tfrac/t_ok/tdur).
import numpy as np, pandas as pd
tj_ = data.traj_idx.dropna(subset=["dep_ts"]).copy()
tj_["dep_ts"] = pd.to_datetime(tj_["dep_ts"]); tj_["arr_ts"] = pd.to_datetime(tj_["arr_ts"])
tj_ = tj_.sort_values(["IMO", "dep_ts"])
tj_["n_prior"] = tj_.groupby("IMO").cumcount()
tj_["days_since_last"] = (tj_["dep_ts"]
    - tj_.groupby("IMO")["arr_ts"].shift(1)).dt.days
H = tj_.set_index("seg_id")[["n_prior", "days_since_last"]]
t_nprior = np.array([H.loc[int(s), "n_prior"] for s in tseg], dtype="float64")
t_stale = np.array([H.loc[int(s), "days_since_last"] for s in tseg], dtype="float64")
early_t = tfrac <= 0.20
print(f"prior-voyage count over test steps: median {np.median(t_nprior):.0f}, "
      f"max {t_nprior.max():.0f}; staleness median "
      f"{np.nanmedian(t_stale):.0f} days")

# ----------------------------------------------------------------------
# [notebook cell 151]
# ----------------------------------------------------------------------
# =============================================================================
# H1 (priority 1) -- Lens 3a: no-history ablation damage BY history depth
# =============================================================================
# Evaluation-only: reload the E3 no-ship-history checkpoints, collect per-step
# predictions on TEST, and stratify the damage by the vessel's history depth.
# Damage concentrating on high-history vessels = the causal signature.
NOHIST_COND = "e3_channel_ablation_no_ship_history_final_main_seed{s}"
import os
_S2,_P2 = [],[]
for s_ in SEEDS:
    cond = NOHIST_COND.format(s=s_)
    meta = os.path.join(WORK_DIR, "Results", f"{TARGET_COL}_{cond}.json")
    assert os.path.exists(meta), f"{cond}: not on disk -- fix NOHIST_COND"
    rv = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=cond, seed=s_,
        alt_progression_modes=ALT_PROGRESSION_MODES, gate_ship_history=False,
        use_ship_history=False, use_ship_size_channel=False,
        use_departure_port_channel=True, use_departure_gate=False,
        n_experts=N_EXPERTS, d_model=D_MODEL, stratify=True, val_frac=0.15,
        test_start=TEST_START, test_end=TEST_END, epochs=1,
        early_stopping_patience=None, batch_size=BATCH_SIZE,
        work_dir=WORK_DIR, skip_existing=True)
    t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
        batch_size=BATCH_SIZE, seg_id_subset=final_runs[s_]["_test_ids"],
        shuffle=False, seed=0, include_ship_history=False)
    a,b,c,d,e = _collect_full_predictions(rv["model"], rv["repr_layer"],
        t_loader, rv["core_and_alt_fn"],
        departure_ids_fn=rv.get("departure_ids_fn"),
        eta_channel_lookup=rv.get("eta_channel_lookup"))
    _S2.append(a); _P2.append(c)
nseg, npred = np.concatenate(_S2), np.concatenate(_P2)
assert len(nseg) == len(tseg) and (nseg == tseg).all(), "row alignment differs"
n_ok = npred == ttrue
BUCK = [("0", t_nprior == 0), ("1-2", (t_nprior >= 1) & (t_nprior <= 2)),
        ("3-5", (t_nprior >= 3) & (t_nprior <= 5)), ("6+", t_nprior >= 6)]
rows = []
from scipy import stats as _st
for lab, m in BUCK:
    dmg = 100 * (t_ok[m].mean() - n_ok[m].mean())
    b_ = int((t_ok[m] & ~n_ok[m]).sum()); c_ = int((~t_ok[m] & n_ok[m]).sum())
    z = (abs(b_ - c_) - 1) / np.sqrt(max(b_ + c_, 1))
    rows.append(dict(bucket=lab, n_steps=int(m.sum()),
                     with_hist=round(100*t_ok[m].mean(), 2),
                     without=round(100*n_ok[m].mean(), 2),
                     damage_pp=round(dmg, 2), mcnemar_z=round(float(z), 1),
                     p=float(2*_st.norm.sf(z))))
h1 = pd.DataFrame(rows); print(h1.to_string(index=False))
h1.to_csv(os.path.join(WORK_DIR, "h1_history_damage_by_depth.csv"), index=False)
print("\nreading: damage rising with depth = channel works through the "
      "vessel's own record (causal signature); flat damage = regularisation.")

# ----------------------------------------------------------------------
# [notebook cell 152]
# ----------------------------------------------------------------------
# =============================================================================
# H2 (priority 2) -- Lens 1: dose-response, clustered logistic
# =============================================================================
import statsmodels.api as sm
m = np.isfinite(t_nprior)
X = pd.DataFrame({"log1p_prior": np.log1p(t_nprior[m]),
                  "early": early_t[m].astype(float),
                  "frac": tfrac[m]})
X["early_x_prior"] = X["early"] * X["log1p_prior"]
X = pd.concat([X, pd.get_dummies(pd.Series(ttrue[m]).astype("int32"),
               prefix="cls", drop_first=True, dtype=float)], axis=1)
fit = sm.Logit(t_ok[m].astype(float), sm.add_constant(X)).fit(
    disp=0, method="bfgs", maxiter=300,
    cov_type="cluster", cov_kwds={"groups": tseg[m]})
keep = ["const", "log1p_prior", "early", "frac", "early_x_prior"]
print(fit.summary2().tables[1].loc[keep].round(4))
print("\nreading: log1p_prior > 0 = per-doubling history benefit; a positive "
      "early_x_prior = history worth MORE early (substitutes for track "
      "evidence). Segment-clustered SEs throughout.")

# ----------------------------------------------------------------------
# [notebook cell 153]
# ----------------------------------------------------------------------
# =============================================================================
# H3 (priority 3) -- Lens 4: the history gate scale across all fits
# =============================================================================
# Extends E14-A: find the zero-initialised ship-history gate scale by weight
# NAME (location differs from the sff scales), across the same 11 fits.
def _find_hist_scales(r):
    out = {}
    for obj, tag in [(r["model"], "model"), (r["repr_layer"], "repr")]:
        for w in getattr(obj, "weights", []):
            nm = w.name.lower() if hasattr(w, "name") else ""
            if ("hist" in nm or "ship" in nm) and ("scale" in nm or "gamma" in nm or "gate" in nm):
                try: out[f"{tag}:{w.name}"] = float(np.array(w))
                except Exception: pass
    return out
probe = _find_hist_scales(runs[SEEDS[0]])
print("candidate history-gate weights found:")
for k, v in probe.items(): print(f"  {k} = {v:.4f}")
print("\nif exactly one scalar matches, extend E14-A's _extract with it and "
      "rerun the 11-fit walk; if several, identify the zero-init one from "
      "cell [52]'s gate_ship_history wiring before tabulating.")

# ----------------------------------------------------------------------
# [notebook cell 154]
# ----------------------------------------------------------------------
# =============================================================================
# H4 (priority 4) -- Lens 2: repetitiveness vs depth (mechanism)
# =============================================================================
# Behavioural regularity vs raw count: entropy of the vessel's PRIOR arrival
# distribution per test segment, in the clustered logistic beside depth.
_arrs = tj_.set_index("seg_id")
def _prior_entropy(sid):
    row = _arrs.loc[int(sid)]
    prior = tj_[(tj_["IMO"] == row["IMO"]) & (tj_["dep_ts"] < row["dep_ts"])]
    if len(prior) < 2: return np.nan
    p = prior[TARGET_COL].value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum())
_ent = {int(s): _prior_entropy(s) for s in np.unique(tseg)}
t_ent = np.array([_ent[int(s)] for s in tseg])
m = np.isfinite(t_ent) & np.isfinite(t_nprior)
X = pd.DataFrame({"hist_entropy": t_ent[m], "log1p_prior": np.log1p(t_nprior[m]),
                  "frac": tfrac[m]})
fit = sm.Logit(t_ok[m].astype(float), sm.add_constant(X)).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": tseg[m]})
print(fit.summary2().tables[1].round(4))
q = pd.qcut(t_ent[m], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print(pd.DataFrame({"ent_quartile": q, "ok": t_ok[m]})
      .groupby("ent_quartile", observed=True)["ok"].agg(["mean", "size"]).round(3))
print("\nreading: entropy dominating depth = the encoder exploits behavioural "
      "regularity, not data volume (shuttle vs tramp, quantified).")

# ----------------------------------------------------------------------
# [notebook cell 155]
# ----------------------------------------------------------------------
# =============================================================================
# H5 (priority 5) -- Lens 5: history-embedding probe (set HISTORY_CHANNEL)
# =============================================================================
# Vessel-level history embeddings: the history channel's step-0 row per
# segment. HISTORY_CHANNEL = the channel index of ship history in the repr
# stack -- VERIFY against cell [52]'s RepresentationLayer construction.
HISTORY_CHANNEL = 3      # e.g. 3 -- set after checking; None refuses to run
assert HISTORY_CHANNEL is not None, "set HISTORY_CHANNEL from cell [52] first"
import torch
from keras import ops as _ops
from sklearn.linear_model import LogisticRegression
r_ = runs[SEEDS[0]]; _vl = r_["val_loader"]
EMB, LBL = [], []
for bi in range(min(60, len(_vl))):
    bs = _vl.batches[bi]
    inputs, n_mask, labels, _ln = _vl[bi]
    if r_.get("eta_channel_lookup") is not None:
        inputs["eta_channel_values"] = _ops.convert_to_tensor(
            eta_progression_for_batch(r_["eta_channel_lookup"], bs,
                                      n_steps=inputs["tau"].shape[1]))
    with torch.no_grad():
        x = r_["repr_layer"](inputs)
    xe = _ops.convert_to_numpy(x[:, HISTORY_CHANNEL, 0])   # step-0 history row
    lb = _ops.convert_to_numpy(labels)
    if lb.ndim > 1: lb = lb[:, 0]
    EMB.append(xe); LBL.append(lb)
E, L = np.concatenate(EMB), np.concatenate(LBL)
pr = LogisticRegression(max_iter=300).fit(E, L)
base = pd.Series(L).value_counts(normalize=True).iloc[0]
print(f"history-embedding probe: {100*pr.score(E, L):.1f}% next-destination "
      f"accuracy from history alone (majority baseline {100*base:.1f}%)")

# ----------------------------------------------------------------------
# [notebook cell 156]
# ----------------------------------------------------------------------
# =============================================================================
# H6 (fixed) -- staleness of history: days since last voyage
# =============================================================================
import numpy as np, pandas as pd, statsmodels.api as sm

neg = np.isfinite(t_stale) & (t_stale < 0)
print(f"negative days_since_last: {neg.sum():,} steps "
      f"({100*neg.mean():.1f}%) -- overlapping/same-day segment bookkeeping; excluded")
m = np.isfinite(t_stale) & (t_stale >= 0) & (t_nprior >= 1)
X = pd.DataFrame({"log1p_stale": np.log1p(t_stale[m]),
                  "log1p_prior": np.log1p(t_nprior[m]),
                  "frac": tfrac[m]})
assert np.isfinite(X.values).all()
fit = sm.Logit(t_ok[m].astype(float), sm.add_constant(X)).fit(
    disp=0, method="bfgs", maxiter=200,
    cov_type="cluster", cov_kwds={"groups": tseg[m]})
print(fit.summary2().tables[1].round(4))
print("\nreading: negative log1p_stale = stale history helps less -- the "
      "temporal analogue of declaration staleness (A4).")

# ----------------------------------------------------------------------
# [notebook cell 157]
# ----------------------------------------------------------------------
# =============================================================================
# H8 (canonical) -- ship-history contribution: length, load region, and cross
# =============================================================================
# Prereqs: H1 (n_ok), E17 arrays (t_ok/tseg/tdur), E18 (t_load).
import numpy as np, pandas as pd
from scipy import stats as _st
from IPython.display import display, HTML


# ---- t_load / t_dest construction (lifted from E18-SLICE) ------------------
import numpy as np
if "tdep" not in dir():
    _tj = data.traj_idx.set_index("seg_id")
    dep_sub_of = {int(s): int(port_to_sub.get(_tj.loc[int(s), "DEP_PORT_ID"], -1))
                  for s in np.unique(tseg)}
    tdep = np.array([dep_sub_of[int(s)] for s in tseg])
subregion_name_map = get_subregion_name_map(data)
t_load = np.array([subregion_name_map.get(int(d), str(d)) for d in tdep])
t_dest = np.array([subregion_name_map.get(int(c), str(c)) for c in ttrue])
print("load regions present:", sorted(set(t_load))[:15])

RNGH = np.random.default_rng(11); NB = 1000

def _paired_group(label, m):
    dmg = 100 * (t_ok[m].mean() - n_ok[m].mean())
    b_ = int((t_ok[m] & ~n_ok[m]).sum()); c_ = int((~t_ok[m] & n_ok[m]).sum())
    z = (abs(b_ - c_) - 1) / np.sqrt(max(b_ + c_, 1))
    per = pd.DataFrame({"s": tseg[m],
                        "d": t_ok[m].astype(float) - n_ok[m].astype(float)}
                       ).groupby("s")["d"].agg(["sum", "count"])
    sv, cv = per["sum"].values, per["count"].values
    boots = [sv[i].sum()/max(cv[i].sum(),1)
             for i in (RNGH.integers(0, len(sv), len(sv)) for _ in range(NB))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(group=label, n_steps=int(m.sum()), n_voyages=len(sv),
                damage_pp=round(dmg, 2),
                ci_lo=round(100*lo, 2), ci_hi=round(100*hi, 2),
                sig=bool(100*lo > 0 or 100*hi < 0),
                mcnemar_p=float(2*_st.norm.sf(z)))

GROUPS8 = [("ALL test steps", np.ones_like(t_ok, bool)),
           ("<= 14 days", tdur <= 14), ("> 14 days", tdur > 14),
           ("USGC loadings", t_load == "USGC"),
           ("ME loadings",   t_load == "ME")]
for load in ("USGC", "ME"):                       # length x load cross
    for lb, lm in [("<= 14d", tdur <= 14), ("> 14d", tdur > 14)]:
        GROUPS8.append((f"{load}, {lb}", (t_load == load) & lm))

h8 = pd.DataFrame([_paired_group(lb, m) for lb, m in GROUPS8])
display

# ----------------------------------------------------------------------
# [notebook cell 158]
# ----------------------------------------------------------------------
# ---- H8 output, print-based ------------------------------------------------
rows = []
for lb, m in GROUPS8:
    if m.sum() == 0:
        print(f"SKIP {lb}: 0 steps (check group labels, e.g. set(t_load))")
        continue
    rows.append(_paired_group(lb, m))
h8 = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print("=" * 100)
print("H8 -- ship-history contribution (with - without, pp), "
      "segment-clustered 95% CIs")
print("=" * 100)
print(h8.to_string(index=False))
h8.to_csv(os.path.join(WORK_DIR, "h8_history_by_slice.csv"), index=False)
print("\nsaved -> h8_history_by_slice.csv")
