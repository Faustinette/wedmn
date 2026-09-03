# E15 — error structure (G-test, taxonomy, ambiguity, empirical ceiling)
# Migrated verbatim from Main_forGitHub.ipynb cells [202, 203, 204, 205, 206, 207, 209, 211, 213, 215].
# Executed by runner.py inside the shared namespace (notebook-kernel style).



# [notebook cell 202]

# E15-0a, LEVEL 0: does error structure exist at all? (global test + MI)
#
# Before naming any error categories, this level asks whether late-voyage
# errors carry structure in the first place, via two tests:
#   (i)  a global G-test of the error confusion matrix against the
#        independence null (errors spread as if predictions were unrelated
#        to the true class);
#   (ii) normalized mutual information between true and predicted class,
#        computed ON ERROR STEPS ONLY. Nonzero NMI among errors means the
#        mistakes themselves are systematic: knowing the true class still
#        tells you something about which wrong class was predicted.

import numpy as np, pandas as pd
from scipy import stats as _st
from sklearn.metrics import normalized_mutual_info_score
LATE = frac > 0.95; ERR = LATE & (pred != true)
cm = pd.crosstab(pd.Series(true[ERR]).map(subregion_names),
                 pd.Series(pred[ERR]).map(subregion_names))
print(f"late errors: {ERR.sum():,} steps -- error confusion (rows=true):")
print(cm.to_string())
prior = pd.Series(true[LATE]).value_counts(normalize=True)
G_total, dof_total = 0.0, 0
for c in cm.index:
    cid = [k for k, v in subregion_names.items() if v == c][0]
    obs = pd.Series(pred[ERR & (true == cid)]).value_counts()
    pri = prior.drop(index=cid, errors="ignore"); pri = pri / pri.sum()
    exp = pri.reindex(obs.index).fillna(1e-6).values * obs.sum()
    G_total += 2 * (obs.values * np.log(obs.values / np.maximum(exp, 1e-9))).sum()
    dof_total += max(len(obs) - 1, 1)
nmi = normalized_mutual_info_score(true[ERR], pred[ERR])
print(f"\nglobal G = {G_total:.0f} (dof {dof_total}), "
      f"p = {_st.chi2.sf(G_total, dof_total):.2e}")
print(f"NMI(true, pred | error) = {nmi:.3f}  (0 = errors carry no structure)")

# [notebook cell 203]

top3 = np.argsort(-probs, axis=1)[:, :3]
in_top3 = (top3 == true[:, None]).any(axis=1)


# [notebook cell 204]

# E12 - Paste and run again
# ===== E12-A2 -- label-track alignment audit (produces `bad`) =====
last = data.steps_idx.loc[data.steps_idx.groupby("SEG_ID")["STEP_IDX"].idxmax(),
                          ["SEG_ID", "GRID_LAT_C", "GRID_LON_C"]]
pl = (last.merge(data.traj_idx[["seg_id", "ARR_PORT_ID"]].dropna(), left_on="SEG_ID",
                 right_on="seg_id").groupby("ARR_PORT_ID")[["GRID_LAT_C", "GRID_LON_C"]].median())
d = last.merge(data.traj_idx[["seg_id", "ARR_PORT_ID", "ARR_SUBREGION_ID"]].dropna(),
               left_on="SEG_ID", right_on="seg_id")
d = d.join(pl, on="ARR_PORT_ID", rsuffix="_port")
d["end_dist_km"] = 111 * np.sqrt((d["GRID_LAT_C"] - d["GRID_LAT_C_port"])**2 +
    (np.cos(np.radians(d["GRID_LAT_C"])) * (d["GRID_LON_C"] - d["GRID_LON_C_port"]))**2)
bad = d[d["end_dist_km"] > 500]
print(f"{len(bad):,} / {len(d):,} segments end >500 km from their labeled arrival port "
      f"({100*len(bad)/len(d):.2f}%)")


# [notebook cell 205]

# ===== E12-B (STS-first) -- one category per error step =====

LATE = frac > 0.95
err = LATE & (true != pred)
sts_id = [k for k,v in subregion_names.items() if v=="STS"][0]
afr_id = [k for k,v in subregion_names.items() if v=="Africa"][0]
bad_ids = set(bad["SEG_ID"].astype(int))
late_counts = pd.Series(true[LATE]).value_counts()
thin = {c for c in subregion_names if late_counts.get(c, 0) < 300}
cat = np.full(len(seg), "", dtype=object)
e = err
cat[e & (true==sts_id)] = "1 STS_structural"                     # STS first
cat[e & (cat=="") & np.isin(seg, list(bad_ids))] = "2 label_misaligned"
cat[e & (cat=="") & np.isin(true, list(thin))] = "3 thin_support"
cat[e & (cat=="") & in_top3] = "4 near_miss_top3"
cat[e & (cat=="") & (true==afr_id)] = "5 overbroad_Africa"
cat[e & (cat=="")] = "6 residual"
tab = pd.Series(cat[e]).value_counts().rename("n errors").to_frame()
tab["% of late errors"] = (100*tab["n errors"]/e.sum()).round(1)
tab["pp of late error rate"] = (100*tab["n errors"]/LATE.sum()).round(2)
print(tab.sort_index().to_string())
ERR_SEGS = {c: pd.Series(seg[e & (cat==c)]).value_counts() for c in sorted(set(cat[e]))}


# [notebook cell 206]


# E15-0b -- LEVEL 0: taxonomy with clustered bootstrap CIs on each share

# The E12-B cascade shares, upgraded to estimates: CI by resampling SEGMENTS.
# Prereq: `cat` from the (STS-first-fixed) E12-B cell.


assert "cat" in dir(), "run E12-B first (produces the per-step category array)"
RNG0 = np.random.default_rng(7); NB = 1000
err_idx = np.where(ERR)[0]
df0 = pd.DataFrame({"s": seg[err_idx], "c": cat[err_idx]})
per_seg = df0.groupby("s")["c"].value_counts().unstack(fill_value=0)
tot_seg = per_seg.sum(axis=1)
rows = []
for c in per_seg.columns:
    kv, nv = per_seg[c].values.astype(float), tot_seg.values.astype(float)
    draws = [kv[d].sum() / max(nv[d].sum(), 1)
             for d in (RNG0.integers(0, len(kv), len(kv)) for _ in range(NB))]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rows.append(dict(category=c, share=round(100 * kv.sum() / nv.sum(), 1),
                     ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1),
                     pp_of_late=round(100 * kv.sum() / LATE.sum(), 2)))
t0b = pd.DataFrame(rows).sort_values("share", ascending=False)
print(t0b.to_string(index=False))
print("\ncascade-order sensitivity: shares are priority-ordered (STS before "
      "misaligned per the audit); reordering moves mass only between those "
      "two structurally-overlapping buckets -- state this in the caption.")


# [notebook cell 207]

# E15-0c -- LEVEL 0: exhaustiveness -- is the residual structureless?

# If the taxonomy captured all systematic error, the residual should show NO
# concentration: E15-A's test applied to the residual bucket alone.
res = ERR & (cat == "6 residual")
print(f"residual: {res.sum():,} steps "
      f"({100*res.sum()/max(ERR.sum(),1):.1f}% of late errors)")
rows = []
for c in sorted(set(true[res])):
    m = res & (true == c)
    if m.sum() < 15: continue
    obs = pd.Series(pred[m]).value_counts()
    pri = prior.drop(index=c, errors="ignore"); pri = pri / pri.sum()
    exp = pri.reindex(obs.index).fillna(1e-6).values * m.sum()
    g = 2 * (obs.values * np.log(obs.values / np.maximum(exp, 1e-9))).sum()
    dof = max(len(obs) - 1, 1)
    rows.append(dict(true=subregion_names[int(c)], n=int(m.sum()),
                     top=subregion_names[int(obs.index[0])],
                     conc=round(obs.iloc[0]/m.sum()/max(pri.get(obs.index[0],1e-6),1e-6), 1),
                     p=round(float(_st.chi2.sf(g, dof)), 3)))
tr_ = pd.DataFrame(rows)
print(tr_.to_string(index=False) if len(tr_) else "  (all classes < 15 residual errors)")
print("\nreading: near-null p / low concentration = taxonomy exhaustive; "
      "significant rows name a candidate SEVENTH category worth inspecting.")


# [notebook cell 209]

# E15-A -- errors are STRUCTURED, not noise: concentration vs prior null

# H0: given an error, the wrong prediction lands on classes ~ their base
# rates. Corridor confusion predicts massive concentration on ONE adjacent
# basin per true class. G-test per class + concentration ratio.
import numpy as np, pandas as pd
from scipy import stats as _st
LATE = frac > 0.95; ERR = LATE & (pred != true)
prior = pd.Series(true[LATE]).value_counts(normalize=True)
rows = []
for c in sorted(set(true[ERR])):
    m = ERR & (true == c)
    if m.sum() < 30: continue
    obs = pd.Series(pred[m]).value_counts()
    pri = prior.drop(index=c, errors="ignore"); pri = pri / pri.sum()
    exp = pri.reindex(obs.index).fillna(1e-6).values * m.sum()
    g = 2 * (obs.values * np.log(obs.values / np.maximum(exp, 1e-9))).sum()
    dof = max(len(obs) - 1, 1)
    top = obs.index[0]
    rows.append(dict(true=subregion_names[int(c)], n_err=int(m.sum()),
        top_confused=subregion_names[int(top)],
        share_obs=round(obs.iloc[0] / m.sum(), 2),
        share_under_H0=round(float(pri.get(top, 0)), 2),
        concentration=round(obs.iloc[0] / m.sum() / max(pri.get(top, 1e-6), 1e-6), 1),
        G=round(float(g), 1), p=float(_st.chi2.sf(g, dof))))
t = pd.DataFrame(rows).sort_values("n_err", ascending=False)
print(t.to_string(index=False))
print("\nreading: share_obs >> share_under_H0 (concentration >> 1, p ~ 0) "
      "= errors target ONE specific adjacent basin, not random classes.")


# [notebook cell 211]

# E15-B -- ambiguity is IN THE DATA, not the fit: cross-seed error agreement

# Three independently initialized models. If late errors were model-specific
# (epistemic), seeds would err on DIFFERENT steps and disagree when wrong.
# Test: pairwise error overlap + same-wrong-prediction rate vs independence.
_lens = data.steps_idx.groupby("SEG_ID").size()
blocks = {}
for sid in np.unique(seg):
    n = int(_lens.get(int(sid), 0)); ia = np.where(seg == sid)[0]
    if n and len(ia) == 3 * n: blocks[sid] = [ia[:n], ia[n:2*n], ia[2*n:]]
print(f"segments in all 3 seed val sets: {len(blocks):,}")
E_, P_, L_ = [], [], []
for sid, (i1, i2, i3) in blocks.items():
    lt = frac[i1] > 0.95
    for arr, tgt in ((E_, [(pred[i]!=true[i])[lt] for i in (i1,i2,i3)]),
                     (P_, [pred[i][lt] for i in (i1,i2,i3)])):
        arr.append(tgt)
e1, e2, e3 = [np.concatenate([b[k] for b in E_]) for k in range(3)]
p1, p2, p3 = [np.concatenate([b[k] for b in P_]) for k in range(3)]
for a, b, la in [(e1,e2,"s42/s123"), (e1,e3,"s42/s7"), (e2,e3,"s123/s7")]:
    both = (a & b).mean(); indep = a.mean() * b.mean()
    print(f"{la}: P(both wrong) = {both:.4f} vs independence {indep:.4f} "
          f"(lift {both/max(indep,1e-9):.1f}x)")
w = e1 & e2
same = (p1[w] == p2[w]).mean()
print(f"\nwhen s42 AND s123 are both wrong: same wrong prediction "
      f"{100*same:.1f}% of the time (chance ~ {100/14:.1f}%)")
print("reading: errors co-locate across independent fits and agree on the "
      "wrong answer -> the ambiguity is a property of the input, not the model.")


# [notebook cell 213]

# E15-C -- the model KNOWS: true-class rank and mass on error steps

# If corridor errors were confusion, the true class would be buried; if they
# are near-miss ambiguity, it sits at rank 2 with substantial mass.
ranks = 1 + (probs[ERR] > probs[ERR, true[ERR]][:, None] - 1e-12).sum(1) - 1
ranks = np.asarray([(np.argsort(-probs[i]) == true[i]).argmax() + 1
                    for i in np.where(ERR)[0]])
p_true = probs[ERR, true[ERR]]
print(f"true-class rank on late errors: median {np.median(ranks):.0f}, "
      f"rank<=2 {100*(ranks<=2).mean():.1f}%, rank<=3 {100*(ranks<=3).mean():.1f}% "
      f"(uniform null: rank<=2 = {100*2/14:.1f}%)")
print(f"probability mass on TRUE class when wrong: median {np.median(p_true):.2f} "
      f"(class prior scale ~ {1/15:.2f})")


# [notebook cell 215]

# E15-D -- empirical ceiling: Bayes-rate estimate from the data alone

# Model-free bound: group LATE steps by (1-deg grid cell, departure
# subregion); the majority-vote accuracy over historical arrivals from that
# state is the best ANY predictor of this information can do. Compare to the
# model. Leakage note: majority computed on the same pooled steps -> this is
# an OPTIMISTIC ceiling (the bound is generous to hypothetical competitors).
_tj = data.traj_idx.set_index("seg_id")
port_to_sub_l = port_to_sub
dep_sub_of = {int(s): int(port_to_sub_l.get(_tj.loc[int(s), "DEP_PORT_ID"], -1))
              for s in np.unique(seg)}
_steps = data.steps_idx.set_index(["SEG_ID", "STEP_IDX"])
occ = pd.DataFrame({"s": seg}).groupby("s").cumcount().values
cell_lat = np.empty(len(seg)); cell_lon = np.empty(len(seg))
for sid, g in data.steps_idx.groupby("SEG_ID"):
    ia = np.where(seg == int(sid))[0]
    la = g.sort_values("STEP_IDX")["GRID_LAT_C"].values
    lo = g.sort_values("STEP_IDX")["GRID_LON_C"].values
    n = len(la)
    if len(ia) % max(n,1) == 0 and n:
        reps = len(ia) // n
        cell_lat[ia] = np.tile(la, reps); cell_lon[ia] = np.tile(lo, reps)
key = pd.DataFrame({"lat": cell_lat[LATE], "lon": cell_lon[LATE],
                    "dep": [dep_sub_of.get(int(s), -1) for s in seg[LATE]],
                    "y": true[LATE]})
maj = key.groupby(["lat", "lon", "dep"])["y"].agg(lambda v: v.value_counts().iloc[0] / len(v))
sizes = key.groupby(["lat", "lon", "dep"])["y"].size()
ceiling = float((maj * sizes).sum() / sizes.sum())
print(f"empirical Bayes ceiling on the late band, given (position cell, load "
      f"basin): {100*ceiling:.2f}%")
print(f"model late accuracy: {100*(pred[LATE]==true[LATE]).mean():.2f}%")
print("reading: the gap to the ceiling bounds what ANY model could add from "
      "this state description; a small gap = the corridor error is mostly "
      "irreducible given position alone.")
