# =============================================================================
# Step 5B — clustered bootstrap, McNemar, calibration/ECE, lock-in, entropy decay, PCA probes, variance decomposition, gate sign tests
# Migrated verbatim from Main_forGitHub.ipynb cells [88, 90, 92, 94, 96, 98, 99, 101, 103].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 88]
# ----------------------------------------------------------------------
# =============================================================================
# E13-A -- clustered bootstrap CIs (segments, not steps) for headline rates
# =============================================================================
# Steps within a voyage are correlated; every CI here resamples SEGMENTS.
# Prereq: pooled arrays seg/true/pred/probs/frac (E12-A cell).
import numpy as np, pandas as pd
RNG = np.random.default_rng(42); N_BOOT = 1000
LATE = frac > 0.95
_ok = (pred == true)

def boot_ci(mask, n_boot=N_BOOT):
    segs_u = np.unique(seg[mask])
    per = pd.DataFrame({"s": seg[mask], "ok": _ok[mask]}).groupby("s")["ok"]
    k, n = per.sum(), per.count()
    idx = {s: i for i, s in enumerate(k.index)}
    kv, nv = k.values.astype(float), n.values.astype(float)
    stats = []
    for _ in range(n_boot):
        draw = RNG.integers(0, len(kv), len(kv))
        stats.append(kv[draw].sum() / max(nv[draw].sum(), 1))
    point = kv.sum() / nv.sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi, len(kv)

for lbl, m in [("overall (val pooled)", np.ones_like(_ok, bool)),
               ("late band (frac>0.95)", LATE)]:
    p, lo, hi, ns = boot_ci(m)
    print(f"{lbl:24s}: {100*p:5.2f}%  [95% CI {100*lo:.2f} -- {100*hi:.2f}]  "
          f"({ns:,} segments, cluster bootstrap)")

# ----------------------------------------------------------------------
# [notebook cell 90]
# ----------------------------------------------------------------------
# =============================================================================
# E13-B -- model vs captain, PAIRED: clustered bootstrap of the delta + McNemar
# =============================================================================
# Pairing: late steps where a declared destination EXISTS. Primary inference =
# segment-clustered bootstrap of (model_acc - captain_acc); McNemar reported
# as the classical step-level companion (with the clustering caveat stated).
import numpy as np, pandas as pd
from scipy import stats as _st

_decl_df = _build_captain_declared_lookup(data, WORK_DIR, subregion_names)
_dmap = {(int(r.SEG_ID), int(r.STEP_IDX)): int(r.declared_subregion)
         for r in _decl_df.itertuples()}
_step_of = {}
for sid, g in data.steps_idx.groupby("SEG_ID"):
    _step_of[int(sid)] = g.sort_values("STEP_IDX")["STEP_IDX"].values
# align pooled rows to STEP_IDX via per-seg occurrence order (3-seed tiling)
_occ = pd.DataFrame({"s": seg}).groupby("s").cumcount().values
_steps_arr = np.array([_step_of[int(s)][o % len(_step_of[int(s)])]
                       for s, o in zip(seg, _occ)])
_cap = np.array([_dmap.get((int(s), int(st)), -1)
                 for s, st in zip(seg, _steps_arr)])
m = LATE & (_cap >= 0)
mod_ok, cap_ok = _ok[m], (_cap[m] == true[m])
print(f"paired late steps with declaration: {m.sum():,}")
print(f"model {100*mod_ok.mean():.2f}%  captain {100*cap_ok.mean():.2f}%")

per = pd.DataFrame({"s": seg[m], "d": mod_ok.astype(float) - cap_ok.astype(float)}
                   ).groupby("s")["d"].agg(["sum", "count"])
sv, cv = per["sum"].values, per["count"].values
deltas = [sv[d].sum() / max(cv[d].sum(), 1)
          for d in (RNG.integers(0, len(sv), len(sv)) for _ in range(N_BOOT))]
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"delta (model - captain): {100*(mod_ok.mean()-cap_ok.mean()):+.2f}pp  "
      f"[95% CI {100*lo:+.2f} -- {100*hi:+.2f}]  (segment-clustered)")

b_ = int((mod_ok & ~cap_ok).sum()); c_ = int((~mod_ok & cap_ok).sum())
z = (abs(b_ - c_) - 1) / np.sqrt(b_ + c_)        # continuity-corrected normal approx
p_mc = 2 * _st.norm.sf(z)
print(f"McNemar (step-level, unclustered caveat): b={b_}, c={c_}, "
      f"z={z:.1f}, p={p_mc:.2e}")

# ----------------------------------------------------------------------
# [notebook cell 92]
# ----------------------------------------------------------------------
# =============================================================================
# E13-C -- calibration: reliability diagram + ECE, overall and per band
# =============================================================================
import matplotlib.pyplot as plt
_conf = probs.max(axis=1)
BINS = np.linspace(0, 1, 11)
def ece(mask):
    bi = np.clip(np.digitize(_conf[mask], BINS) - 1, 0, 9)
    rows = []
    for b in range(10):
        mm = bi == b
        if mm.sum() == 0: rows.append((np.nan, np.nan, 0)); continue
        rows.append((_conf[mask][mm].mean(), _ok[mask][mm].mean(), mm.sum()))
    e = sum(n * abs(c - a) for c, a, n in rows if n) / max(mask.sum(), 1)
    return rows, e

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=True)
for ax, (lbl, mask) in zip(axes, [("Early (<20%)", frac <= 0.2),
                                  ("Mid (20-60%)", (frac > 0.2) & (frac <= 0.6)),
                                  ("Late (>60%)", frac > 0.6)]):
    rows, e = ece(mask)
    xs = [r[0] for r in rows if r[2]]; ys = [r[1] for r in rows if r[2]]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(xs, ys, "o-", color="C0")
    ax.set_title(f"{lbl}   ECE = {e:.3f}"); ax.set_xlabel("confidence")
axes[0].set_ylabel("empirical accuracy")
fig.suptitle("Reliability by progression band (val pooled, 3 seeds)")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e13_calibration.png"), dpi=150,
            bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# [notebook cell 94]
# ----------------------------------------------------------------------
# =============================================================================
# E13-D -- decision lock-in: share of voyages permanently correct by progression
# =============================================================================
# For each (segment, first-seed block): the progression fraction AFTER which
# every prediction is correct (lock-in point). Kaplan-Meier-style curve.
_lens = data.steps_idx.groupby("SEG_ID").size()
lockins, never = [], 0
for sid in np.unique(seg):
    n = int(_lens.get(int(sid), 0))
    idx_all = np.where(seg == sid)[0]
    if n == 0 or len(idx_all) < n: continue
    idx = idx_all[:n]
    okv = _ok[idx]; fr = frac[idx]
    if okv[-1] and okv.all(): lockins.append(0.0); continue
    wrong = np.where(~okv)[0]
    if not okv[-1] or len(wrong) == 0 and not okv[-1]: never += 1; continue
    if len(wrong) == 0: lockins.append(0.0); continue
    li = wrong[-1] + 1
    if li >= n: never += 1
    else: lockins.append(float(fr[li]))
lockins = np.array(lockins)
grid = np.linspace(0, 1, 101)
curve = [(lockins <= g).mean() * len(lockins) / (len(lockins) + never)
         for g in grid]
plt.figure(figsize=(8, 4.5))
plt.plot(grid * 100, np.array(curve) * 100, lw=2)
plt.xlabel("voyage progression (%)"); plt.ylabel("% of voyages locked in")
plt.title("Prediction lock-in curve (val pooled; never-locked = "
          f"{100*never/(len(lockins)+never):.1f}%)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e13_lockin_curve.png"), dpi=150)
plt.show()
for q in (0.25, 0.5, 0.75, 0.9):
    i = np.searchsorted(curve, q)
    print(f"{int(q*100)}% of voyages locked in by "
          f"{grid[min(i,100)]*100:.0f}% progression" if i <= 100 else "")

# ----------------------------------------------------------------------
# [notebook cell 96]
# ----------------------------------------------------------------------
# =============================================================================
# E13-E -- entropy decay: destination uncertainty (bits) vs progression
# =============================================================================
H = -(probs * np.log2(np.maximum(probs, 1e-12))).sum(axis=1)
bands5 = np.clip((frac * 20).astype(int), 0, 19)
mh = [H[bands5 == b].mean() for b in range(20)]
plt.figure(figsize=(8, 4))
plt.plot(np.arange(20) * 5 + 2.5, mh, "o-")
plt.axhline(np.log2(N_CLASSES), color="grey", ls="--",
            label=f"uniform ({np.log2(N_CLASSES):.2f} bits)")
plt.xlabel("voyage progression (%)"); plt.ylabel("mean posterior entropy (bits)")
plt.title("Uncertainty resolution over the voyage (val pooled)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e13_entropy_decay.png"), dpi=150)
plt.show()
print(f"entropy: start {mh[0]:.2f} -> 20% {mh[3]:.2f} -> 60% {mh[11]:.2f} "
      f"-> end {mh[-1]:.2f} bits (uniform = {np.log2(N_CLASSES):.2f})")

# ----------------------------------------------------------------------
# [notebook cell 98]
# ----------------------------------------------------------------------
# =============================================================================
# E13-F -- representation geometry: PCA of designated-channel states + linear probes
# =============================================================================
# 5b: step representations x[:, DESIGNATED_CHANNEL] at three bands, PCA-2D
# colored by true destination; plus per-band linear probes (logistic reg on
# frozen reps) = information-availability curve. One forward pass over a
# sample of val batches (seed-42 model).
import torch
from keras import ops as _ops
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(0)

_r = runs[SEEDS[0]]; _vl = _r["val_loader"]
REPS, LBL, FR = [], [], []
for bi in range(min(60, len(_vl))):
    bs = _vl.batches[bi]
    inputs, n_mask, labels, _ln = _vl[bi]
    if _r.get("eta_channel_lookup") is not None:
        inputs["eta_channel_values"] = _ops.convert_to_tensor(
            eta_progression_for_batch(_r["eta_channel_lookup"], bs,
                                      n_steps=inputs["tau"].shape[1]))
    with torch.no_grad():
        x = _r["repr_layer"](inputs)
    xr = _ops.convert_to_numpy(x[:, DESIGNATED_CHANNEL])
    mk = _ops.convert_to_numpy(n_mask).astype(bool)
    mf = mk.astype(float)
    fr = np.cumsum(mf, 1) / np.maximum(mf.sum(1, keepdims=True), 1)
    lb = _ops.convert_to_numpy(labels)
    if lb.ndim == 1: lb = np.repeat(lb[:, None], xr.shape[1], axis=1)
    REPS.append(xr[mk]); LBL.append(lb[mk]); FR.append(fr[mk])
R, L, F = map(np.concatenate, (REPS, LBL, FR))
print(f"collected {len(R):,} step representations, d={R.shape[1]}")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, (lbl, m) in zip(axes, [("Early <20%", F <= 0.2),
                               ("Mid 20-60%", (F > 0.2) & (F <= 0.6)),
                               ("Late >60%", F > 0.6)]):
    p2 = PCA(2).fit_transform(R[m])
    for c in np.unique(L[m]):
        mm = L[m] == c
        ax.scatter(p2[mm, 0], p2[mm, 1], s=3, alpha=0.4,
                   label=subregion_names.get(int(c), c) if mm.sum() > 300 else None)
    probe = LogisticRegression(max_iter=300, multi_class="multinomial")
    ncv = min(20000, m.sum()); sel = RNG.choice(np.where(m)[0], ncv, replace=False)
    probe.fit(R[sel], L[sel])
    acc = probe.score(R[sel], L[sel])
    ax.set_title(f"{lbl} -- linear probe {100*acc:.1f}%")
axes[0].legend(fontsize=7, markerscale=3)
fig.suptitle("Designated-channel geometry by progression (PCA-2D, "
             "colour = true destination)")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e13_rep_geometry.png"), dpi=150,
            bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# [notebook cell 99]
# ----------------------------------------------------------------------
# ---- probe accuracy vs retained PCA dimensions (per band) ------------------
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
DIMS = [2, 3, 5, 10, 32]
print(f"{'band':12s}" + "".join(f"  k={k:2d}" for k in DIMS))
for lbl, m in [("Early <20%", F <= 0.2), ("Mid 20-60%", (F > 0.2) & (F <= 0.6)),
               ("Late >60%", F > 0.6)]:
    sel = RNG.choice(np.where(m)[0], min(20000, m.sum()), replace=False)
    accs = []
    for k in DIMS:
        Z = PCA(k).fit_transform(R[sel]) if k < 32 else R[sel]
        p = LogisticRegression(max_iter=300).fit(Z, L[sel])
        accs.append(100 * p.score(Z, L[sel]))
    print(f"{lbl:12s}" + "".join(f"  {a:4.1f}" for a in accs))

# ----------------------------------------------------------------------
# [notebook cell 101]
# ----------------------------------------------------------------------
# =============================================================================
# E13-G -- variance decomposition: initialization vs split vs band
# =============================================================================
rows = []
_seed_accs = [runs[s]["metrics"]["overall_acc"] * 100 for s in SEEDS]
rows.append(("initialization (3 seeds, fixed split)",
             np.std(_seed_accs), np.mean(_seed_accs)))
if "CV_RESULTS" in dir() and CV_RESULTS:
    _cv = [CV_RESULTS[k]["overall_acc"] * 100 for k in sorted(CV_RESULTS)]
    rows.append((f"split ({len(_cv)} CV folds, fixed seed)",
                 np.std(_cv), np.mean(_cv)))
else:
    rows.append(("split (CV folds)", np.nan, np.nan))
vt = pd.DataFrame(rows, columns=["variance source", "std (pp)", "mean acc (%)"])
print(vt.round(2).to_string(index=False))
print("\ninterpretation: comparable stds -> split composition contributes no "
      "more uncertainty than initialization; report both alongside the "
      "clustered bootstrap CI (sampling variance).")

# ----------------------------------------------------------------------
# [notebook cell 103]
# ----------------------------------------------------------------------
# =============================================================================
# E13-H -- gate parameters as estimates: per-seed table + sign tests
# =============================================================================
# The identifiable scalars: alt-signal gate scales (eta, historical_avg_port)
# and the content-gate scale, per CASP layer, per seed. Zero-init => the null
# H0: scale = 0 is the untrained state; departure from 0 is learned signal.
rows = []
for s_ in SEEDS:
    mdl = runs[s_]["model"]
    for li, layer in enumerate(mdl.casp_layers):
        sff = getattr(layer, "sff", None)
        if sff is None or getattr(sff, "n_experts", 1) <= 1: continue
        for i, sc in enumerate(getattr(sff, "alt_prog_scales", []) or []):
            nm = (ALT_PROGRESSION_MODES[i]
                  if i < len(ALT_PROGRESSION_MODES) else f"alt{i}")
            rows.append(dict(seed=s_, layer=li, param=f"beta_{nm}",
                             value=float(_ops.convert_to_numpy(sc))))
        cg = getattr(sff, "content_gate_scale", None)
        if cg is not None:
            rows.append(dict(seed=s_, layer=li, param="beta_content",
                             value=float(_ops.convert_to_numpy(cg))))
gp = pd.DataFrame(rows)
summ = gp.groupby(["layer", "param"])["value"].agg(
    mean="mean", std="std", n="count",
    sign_consistent=lambda v: (np.sign(v) == np.sign(v.iloc[0])).all())
print(summ.round(4).to_string())
print("\nn=3 seeds: report mean +/- std and sign consistency; with the 5 CV "
      "folds added (n=8), a sign test against 0 reaches p=0.008 when all "
      "agree -- rerun this cell after CV to upgrade the table.")
gp.to_csv(os.path.join(WORK_DIR, "e13_gate_parameters.csv"), index=False)
