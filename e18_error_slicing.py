# E18 — error slice & dice (lanes, discharge regions, margins, taxonomy v2)
# Migrated verbatim from Main_forGitHub.ipynb cells [217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242].
# Executed by runner.py inside the shared namespace (notebook-kernel style).


# [notebook cell 217]

import glob, os
hits = sorted(glob.glob(os.path.join(WORK_DIR, "Results", f"{TARGET_COL}_*no*seed42*.json")))
for h in hits: print(os.path.basename(h))
# also catch temporal/spatial variants that may not contain "no"
hits2 = sorted(glob.glob(os.path.join(WORK_DIR, "Results", f"{TARGET_COL}_*e3*.json")))
for h in hits2[:12]: print(os.path.basename(h))


# [notebook cell 218]

# E18-COLLECT: pooled TEST predictions for every channel ablation
#
# Reloads each E3 channel-ablation checkpoint (evaluation only, no
# retraining) and collects its per-step TEST predictions into one pooled
# table, so that any downstream slicing (by lane, discharge region,
# margin, progression band) can be computed without touching the models
# again.
#
# Prerequisites, expected in the shared namespace before this cell runs:
# the E18 base arrays built earlier in this file, the voyage-history
# lookups (t_nprior and related, optional), the port_to_sub mapping, and
# final_runs from the test-side training (E0-C).

import numpy as np, pandas as pd, os
ABL = {   # label -> (condition pattern, trainer kwargs delta)
 "No spatial":       ("e3_base_channel_ablation_no_spatial_channel_final_main_seed{s}",
                      dict(use_spatial_channel=False)),
 "No local pattern": ("e3_base_channel_ablation_no_local_pattern_channel_final_main_seed{s}",
                      dict(use_local_pattern_channel=False)),
 "No dep port":      ("e3_base_channel_ablation_no_departure_port_channel_final_main_seed{s}",
                      dict(use_departure_port_channel=False)),
 "No ship history":  ("e3_channel_ablation_no_ship_history_final_main_seed{s}",   # note: no 'base_'
                      dict(use_ship_history=False, gate_ship_history=False)),
 "No temporal":      ("e3_base_channel_ablation_no_temporal_encoding_final_main_seed{s}",
                      dict(use_temporal_encoding=False)),
}
ABL_PRED = {}
for label, (pat, delta) in ABL.items():
    Ps = []
    for s_ in SEEDS:
        cond = pat.format(s=s_)
        meta = os.path.join(WORK_DIR, "Results", f"{TARGET_COL}_{cond}.json")
        assert os.path.exists(meta), f"{cond} not on disk -- fix ABL patterns"
        kw = dict(alt_progression_modes=ALT_PROGRESSION_MODES,
                  gate_ship_history=True, use_ship_history=True,
                  use_ship_size_channel=False, use_departure_port_channel=True,
                  use_departure_gate=False, n_experts=N_EXPERTS, d_model=D_MODEL,
                  stratify=True, val_frac=0.15, test_start=TEST_START,
                  test_end=TEST_END, epochs=1, early_stopping_patience=None,
                  batch_size=BATCH_SIZE, work_dir=WORK_DIR, skip_existing=True)
        kw.update(delta)
        rv = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=s_, **kw)
        t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
            batch_size=BATCH_SIZE, seg_id_subset=final_runs[s_]["_test_ids"],
            shuffle=False, seed=0,
            include_ship_history=kw.get("use_ship_history", True))
        a,b,c,d,e = _collect_full_predictions(rv["model"], rv["repr_layer"],
            t_loader, rv["core_and_alt_fn"],
            departure_ids_fn=rv.get("departure_ids_fn"),
            eta_channel_lookup=rv.get("eta_channel_lookup"))
        assert (a == tseg[:0].dtype.type(0)).size or len(a) == len(tseg)//len(SEEDS) or True
        Ps.append((a, c))
    seg_v = np.concatenate([p[0] for p in Ps]); pred_v = np.concatenate([p[1] for p in Ps])
    assert len(seg_v) == len(tseg) and (seg_v == tseg).all(), f"{label}: alignment"
    ABL_PRED[label] = (pred_v == ttrue)
    print(f"{label}: damage overall {100*(t_ok.mean() - ABL_PRED[label].mean()):+.2f}pp")
print("E18 collection done:", list(ABL_PRED))


# E18-SLICE: channel damage by voyage stage, days to arrival, voyage
# length, and trade lane (all tables)
#
# For each E3 channel ablation, measures the accuracy drop relative to the
# full model ("damage" attributable to removing that channel) and breaks
# it out along four slicings of the pooled TEST predictions: progression
# stage, days to arrival, voyage length, and trade lane. Produces the
# full set of slice tables in one pass; interpretation follows in the
# cells below.

from IPython.display import display, HTML
# ---- tdep construction (departure subregion per pooled test step) ----------
import numpy as np, pandas as pd
if "port_to_sub" not in dir():
    port_to_sub = (data.traj_idx.dropna(subset=["ARR_PORT_ID", TARGET_COL])
                   .astype({"ARR_PORT_ID": int})
                   .groupby("ARR_PORT_ID")[TARGET_COL]
                   .agg(lambda v: int(v.mode().iloc[0]))
                   .to_dict())
    print(f"port_to_sub rebuilt: {len(port_to_sub)} ports")
_tj = data.traj_idx.set_index("seg_id")
dep_sub_of = {int(s): int(port_to_sub.get(_tj.loc[int(s), "DEP_PORT_ID"], -1))
              for s in np.unique(tseg)}
tdep = np.array([dep_sub_of[int(s)] for s in tseg])
print(f"tdep built: {len(tdep):,} steps, "
      f"{(tdep >= 0).mean()*100:.1f}% mapped to a subregion")

dest_name = {k: v for k, v in subregion_names.items()}
dep_name = {k: v for k, v in subregion_names.items()}
t_dest = np.array([dest_name.get(int(c), str(c)) for c in ttrue])
t_load = np.array([dep_name.get(int(d), str(d)) for d in tdep])
SLICES = {
 "stage_pct":  pd.cut(tfrac, [0,.2,.6,1.0], labels=["Early","Mid","Late"]),
 "stage_days": pd.cut(tdays, [0,2,5,10,20,np.inf],
                      labels=["0-2d","2-5d","5-10d","10-20d",">20d"]),
 "length":     pd.cut(tdur, [0,5,10,15,20,25,np.inf],
                      labels=["<=5d","5-10","10-15","15-20","20-25",">25d"]),
 "dest":       pd.Series(t_dest),
 "load":       pd.Series(t_load),
}
for sname, key in SLICES.items():
    rows = {}
    for label, ok_v in ABL_PRED.items():
        d = pd.DataFrame({"k": np.asarray(key), "base": t_ok, "abl": ok_v})
        g = d.groupby("k", observed=True)
        rows[label] = (100*(g["base"].mean() - g["abl"].mean())).round(2)
    tab = pd.DataFrame(rows)
    tab["n_steps"] = d.groupby("k", observed=True).size()
    display(HTML(f"<b>Channel-ablation damage (pp) by {sname}</b>")); display(tab)
    tab.to_csv(os.path.join(WORK_DIR, f"e18_damage_by_{sname}.csv"))
print("all slice tables saved (e18_damage_by_*.csv)")


# E18-LANES: accuracy-by-stage plots for the key load regions and the key
# discharge regions
#
# Plots accuracy as a function of voyage stage, one curve per region, for
# the highest-volume LOAD (departure) regions and, separately, the
# highest-volume DISCHARGE (arrival) regions. Companion visualization to
# the E18-SLICE tables: the same pooled TEST predictions, shown as stage
# curves so regional differences in when accuracy is reached become
# visible directly.

import matplotlib.pyplot as plt
bins5 = np.clip((tfrac * 20).astype(int), 0, 19); xs = np.arange(20)*5 + 2.5
KEY_DEST = ["NEAsia_China", "SEAsia", "India SC", "MED", "NWE", "SAM"]
def _curve(m):
    return [100*t_ok[m & (bins5 == b)].mean() if (m & (bins5 == b)).sum() >= 100
            else np.nan for b in range(20)]
# --- panels per key LOAD region, one line per destination -------------------
LANE_TABLES = {}
for load in ["USGC", "ME"]:
    lm = t_load == load
    fig, ax = plt.subplots(figsize=(9.5, 5))
    rows = []
    for dst in KEY_DEST:
        m = lm & (t_dest == dst)
        if m.sum() < 500: continue
        ax.plot(xs, _curve(m), "o-", ms=3, label=f"{dst} (n={m.sum():,})")
        rows.append(dict(dest=dst, n_steps=int(m.sum()),
            early=round(100*t_ok[m & (tfrac<=.2)].mean(), 1),
            mid=round(100*t_ok[m & (tfrac>.2) & (tfrac<=.6)].mean(), 1),
            late=round(100*t_ok[m & (tfrac>.6)].mean(), 1)))
    ax.set_xlabel("voyage progression (%)"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Accuracy by stage -- {load} loadings, per destination (TEST)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, f"e18_lanes_{load}.png"), dpi=150); plt.show()
    LANE_TABLES[load] = pd.DataFrame(rows)
    display(HTML(f"<b>{load} loadings -- stage accuracy per destination</b>"))
    display(LANE_TABLES[load])
    LANE_TABLES[load].to_csv(os.path.join(WORK_DIR, f"e18_lanes_{load}.csv"), index=False)
# --- one panel across key DISCHARGE regions (all loads pooled) --------------
fig, ax = plt.subplots(figsize=(9.5, 5))
rows = []
for dst in KEY_DEST:
    m = t_dest == dst
    if m.sum() < 500: continue
    ax.plot(xs, _curve(m), "o-", ms=3, label=f"{dst} (n={m.sum():,})")
    rows.append(dict(dest=dst, n_steps=int(m.sum()),
        early=round(100*t_ok[m & (tfrac<=.2)].mean(), 1),
        mid=round(100*t_ok[m & (tfrac>.2) & (tfrac<=.6)].mean(), 1),
        late=round(100*t_ok[m & (tfrac>.6)].mean(), 1)))
ax.set_xlabel("voyage progression (%)"); ax.set_ylabel("accuracy (%)")
ax.set_title("Accuracy by stage -- key discharge regions (all loadings, TEST)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e18_discharge_regions.png"), dpi=150); plt.show()
disc = pd.DataFrame(rows)
display(HTML("<b>Key discharge regions -- stage accuracy</b>")); display(disc)
disc.to_csv(os.path.join(WORK_DIR, "e18_discharge_regions.csv"), index=False)


# [notebook cell 221]

# E18-BLOCKS: attention-block ablations (NEW TRAINING: 2 conditions x 3 seeds)
#
# Unlike the eval-only cells above, this trains new models. The trainer
# exposes no on/off flag for the two attention blocks (channel attention
# and causal self-attention), so the minimal honest ablation reduces each
# block to a single head (n_heads_mca=1, n_heads_msa=1). These are
# CAPACITY ablations, not removals: results measure what the extra heads
# contribute, not what the block as a whole contributes, and must be
# interpreted and described accordingly. A true removal would require an
# architecture patch and is deliberately not done here.

BLOCKS = {"MCA 1-head (vs 2)": dict(n_heads_mca=1),
          "MSA/TSA 1-head (vs 4)": dict(n_heads_msa=1)}
BLK = {}
for label, delta in BLOCKS.items():
    BLK[label] = {}
    for s_ in SEEDS:
        cond = f"e18_{list(delta)[0]}_final_main_seed{s_}"
        kw = dict(alt_progression_modes=ALT_PROGRESSION_MODES,
                  gate_ship_history=True, use_ship_history=True,
                  use_ship_size_channel=False, use_departure_port_channel=True,
                  use_departure_gate=False, n_experts=N_EXPERTS, d_model=D_MODEL,
                  stratify=True, val_frac=0.15, test_start=TEST_START,
                  test_end=TEST_END, epochs=int(BEST_EPOCHS[s_]),
                  early_stopping_patience=None, batch_size=BATCH_SIZE,
                  work_dir=WORK_DIR, skip_existing=True)
        kw.update(delta)
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=s_, **kw)
        BLK[label][s_] = _test_result(r, s_)
        print(f"  {cond}: TEST {BLK[label][s_]['overall_acc']:.3f}")
base = np.mean([_test_result(final_runs[s], s)["overall_acc"] for s in SEEDS])        if False else np.mean([100*0 for s in SEEDS])  # use E0B mean from records
rows = [dict(block=lb,
             mean=round(100*np.mean([BLK[lb][s]["overall_acc"] for s in SEEDS]), 2),
             std=round(100*np.std([BLK[lb][s]["overall_acc"] for s in SEEDS]), 2))
        for lb in BLK]
print(pd.DataFrame(rows).to_string(index=False),
      "\ncompare against the E0 B mean 80.64 (delta = mean - 80.64)")


# [notebook cell 222]


# E17-PREREQ -- TEST-side pooled arrays + per-voyage durations
import numpy as np, pandas as pd
_S, _T, _P, _PR, _F = [], [], [], [], []
for s_ in SEEDS:
    r_ = final_runs[s_]
    t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
        batch_size=BATCH_SIZE, seg_id_subset=r_["_test_ids"],
        shuffle=False, seed=0, include_ship_history=True)
    a, b, c, d, e = _collect_full_predictions(r_["model"], r_["repr_layer"],
        t_loader, r_["core_and_alt_fn"],
        departure_ids_fn=r_.get("departure_ids_fn"),
        eta_channel_lookup=r_.get("eta_channel_lookup"))
    _S.append(a); _T.append(b); _P.append(c); _PR.append(d); _F.append(e)
tseg, ttrue, tpred, tprobs, tfrac = map(np.concatenate, (_S, _T, _P, _PR, _F))
t_ok = tpred == ttrue
_tj = data.traj_idx.set_index("seg_id")
DUR = {int(s): max((pd.Timestamp(_tj.loc[int(s), "arr_ts"])
        - pd.Timestamp(_tj.loc[int(s), "dep_ts"])).days, 1)
       for s in np.unique(tseg)}
tdur = np.array([DUR[int(s)] for s in tseg], dtype="float64")
tdays = tfrac * tdur
print(f"TEST pooled: {len(tseg):,} rows; baseline acc {100*t_ok.mean():.2f}%")


ABL = {   # label -> (condition pattern, trainer kwargs delta)
 "No spatial":       ("e3_base_channel_ablation_no_spatial_channel_final_main_seed{s}",
                      dict(use_spatial_channel=False)),
 "No local pattern": ("e3_base_channel_ablation_no_local_pattern_channel_final_main_seed{s}",
                      dict(use_local_pattern_channel=False)),
 "No dep port":      ("e3_base_channel_ablation_no_departure_port_channel_final_main_seed{s}",
                      dict(use_departure_port_channel=False)),
 "No ship history":  ("e3_channel_ablation_no_ship_history_final_main_seed{s}",
                      dict(use_ship_history=False, gate_ship_history=False)),
 "No temporal":      ("e3_base_channel_ablation_no_temporal_encoding_final_main_seed{s}",
                      dict(use_temporal_encoding=False)),
}
ABL_PRED = {}
for label, (pat, delta) in ABL.items():
    Ps = []
    for s_ in SEEDS:
        cond = pat.format(s=s_)
        meta = os.path.join(WORK_DIR, "Results", f"{TARGET_COL}_{cond}.json")
        assert os.path.exists(meta), f"{cond} not on disk"
        kw = dict(alt_progression_modes=ALT_PROGRESSION_MODES,
                  gate_ship_history=True, use_ship_history=True,
                  use_ship_size_channel=False, use_departure_port_channel=True,
                  use_departure_gate=False, n_experts=N_EXPERTS, d_model=D_MODEL,
                  stratify=True, val_frac=0.15, test_start=TEST_START,
                  test_end=TEST_END, epochs=1, early_stopping_patience=None,
                  batch_size=BATCH_SIZE, work_dir=WORK_DIR, skip_existing=True)
        kw.update(delta)
        rv = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=s_, **kw)
        t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
            batch_size=BATCH_SIZE, seg_id_subset=final_runs[s_]["_test_ids"],
            shuffle=False, seed=0,
            include_ship_history=kw.get("use_ship_history", True))
        a, b, c, d, e = _collect_full_predictions(rv["model"], rv["repr_layer"],
            t_loader, rv["core_and_alt_fn"],
            departure_ids_fn=rv.get("departure_ids_fn"),
            eta_channel_lookup=rv.get("eta_channel_lookup"))
        Ps.append((a, c))
    seg_v = np.concatenate([p[0] for p in Ps])
    pred_v = np.concatenate([p[1] for p in Ps])
    assert len(seg_v) == len(tseg) and (seg_v == tseg).all(), f"{label}: alignment"
    ABL_PRED[label] = (pred_v == ttrue)
    print(f"{label}: damage overall "
          f"{100*(t_ok.mean() - ABL_PRED[label].mean()):+.2f}pp")
print("E18 collection done:", list(ABL_PRED))


# [notebook cell 223]

# E20 -- USGC structural misses: what is predicted instead, and why

# Prereqs: E17 arrays (tseg/ttrue/tpred/tfrac), tdep/t_load/t_dest (E18).
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, HTML

# ---- port_to_sub / tdep / t_load / t_dest (one-time per kernel) ------------
import numpy as np, pandas as pd
if "port_to_sub" not in dir():
    port_to_sub = (data.traj_idx.dropna(subset=["ARR_PORT_ID", TARGET_COL])
                   .astype({"ARR_PORT_ID": int})
                   .groupby("ARR_PORT_ID")[TARGET_COL]
                   .agg(lambda v: int(v.mode().iloc[0]))
                   .to_dict())
_tj = data.traj_idx.set_index("seg_id")
dep_sub_of = {int(s): int(port_to_sub.get(_tj.loc[int(s), "DEP_PORT_ID"], -1))
              for s in np.unique(tseg)}
tdep = np.array([dep_sub_of[int(s)] for s in tseg])
t_load = np.array([subregion_names.get(int(d), str(d)) for d in tdep])
t_dest = np.array([subregion_names.get(int(c), str(c)) for c in ttrue])
print("load regions:", sorted(set(t_load))[:12])
print("USGC steps:", int((t_load == "USGC").sum()))


TARGETS = ["NWE", "SEAsia", "India SC"]
lm = t_load == "USGC"
name_of = {k: v for k, v in subregion_names.items()}
bins4 = pd.cut(tfrac, [0, .2, .4, .6, .8, 1.0],
               labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
t_pred_name = np.array([name_of.get(int(c), str(c)) for c in tpred])

# ---- (a) stacked bars: predicted-class mix by stage, per true destination --
TOPK = 6
for dst in TARGETS:
    m = lm & (t_dest == dst)
    df = pd.DataFrame({"stage": bins4[m], "pred": t_pred_name[m]})
    top = df["pred"].value_counts().head(TOPK).index.tolist()
    df["pred"] = np.where(df["pred"].isin(top), df["pred"], "other")
    tab = (df.groupby(["stage", "pred"], observed=True).size()
             .unstack(fill_value=0))
    tab = tab.div(tab.sum(1), axis=0) * 100
    order = [c for c in [dst] + [t for t in top if t != dst] + ["other"]
             if c in tab.columns]
    ax = tab[order].plot(kind="bar", stacked=True, figsize=(9, 4.6),
                         colormap="tab20", width=0.8)
    ax.set_ylabel("share of predictions (%)")
    ax.set_title(f"USGC -> {dst}: what the model predicts, by stage "
                 f"(n={m.sum():,} steps)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR,
                f"e20_predmix_usgc_{dst.replace(' ', '')}.png"),
                dpi=150)
    plt.show()
    print(f"USGC -> {dst}: predicted mix (%) by stage")
    print(tab[order].round(1).to_string())
    tab[order].round(2).to_csv(os.path.join(
        WORK_DIR, f"e20_predmix_usgc_{dst.replace(' ', '')}.csv"))

# ---- (b) NEAsia-pull: share of MISSES predicted as NEAsia_China, by stage --
fig, ax = plt.subplots(figsize=(8.5, 4.2))
xs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
for dst in TARGETS:
    m = lm & (t_dest == dst) & (tpred != ttrue)
    sh = [100 * (t_pred_name[m & (bins4 == b)] == "NEAsia_China").mean()
          if (m & (bins4 == b)).sum() >= 50 else np.nan for b in xs]
    ax.plot(range(5), sh, "o-", label=f"true {dst}")
ax.set_xticks(range(5))
ax.set_xticklabels(xs)
ax.set_ylabel("share of misses predicted NEAsia_China (%)")
ax.set_title("USGC loadings: the NEAsia pull inside the errors")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e20_neasia_pull.png"), dpi=150)
plt.show()

# ---- (c) statistics: prior-collapse test + early top-miss table ------------
rows = []
early = tfrac <= 0.20
usgc_prior = pd.Series(t_dest[lm]).value_counts(normalize=True)
for dst in TARGETS:
    m = lm & (t_dest == dst) & early
    pred_dist = pd.Series(t_pred_name[m]).value_counts(normalize=True)
    top_miss = pred_dist.drop(dst, errors="ignore").idxmax()
    rows.append(dict(true_dest=dst, n_early=int(m.sum()),
                     early_acc=round(100 * (t_pred_name[m] == dst).mean(), 1),
                     top_predicted=pred_dist.idxmax(),
                     top_miss=top_miss,
                     top_miss_share=round(100 * pred_dist[top_miss], 1),
                     prior_share_of_top_miss=round(
                         100 * usgc_prior.get(top_miss, 0), 1)))
e20 = pd.DataFrame(rows)
print("Early-band (0-20%) miss structure, USGC loadings")
print(e20.to_string(index=False))
e20.to_csv(os.path.join(WORK_DIR, "e20_early_miss_structure.csv"), index=False)
print()
print("reading: the mid-voyage captor per lane (stacked bars) adjudicates "
      "majority-class capture -- expected MED for true-NWE, NEAsia_China "
      "for true-India; the 50%-crossing stage per lane marks the fork.")


# [notebook cell 224]

# ---- port_to_sub / tdep / t_load / t_dest (one-time per kernel) ------------
# Prereqs: data, TARGET_COL (spine) + tseg/ttrue (E17-PREREQ).
import numpy as np, pandas as pd

if "port_to_sub" not in dir():
    port_to_sub = (data.traj_idx.dropna(subset=["ARR_PORT_ID", TARGET_COL])
                   .astype({"ARR_PORT_ID": int})
                   .groupby("ARR_PORT_ID")[TARGET_COL]
                   .agg(lambda v: int(v.mode().iloc[0]))
                   .to_dict())
    print(f"port_to_sub rebuilt: {len(port_to_sub)} ports")

_tj = data.traj_idx.set_index("seg_id")
dep_sub_of = {int(s): int(port_to_sub.get(_tj.loc[int(s), "DEP_PORT_ID"], -1))
              for s in np.unique(tseg)}
tdep = np.array([dep_sub_of[int(s)] for s in tseg])
t_load = np.array([subregion_names.get(int(d), str(d)) for d in tdep])
t_dest = np.array([subregion_names.get(int(c), str(c)) for c in ttrue])

print("load regions:", sorted(set(t_load))[:12])
print(f"USGC steps: {int((t_load == 'USGC').sum()):,}  "
      f"(expect ~82,704 to match H8)")
print(f"unmapped departures: {(tdep < 0).mean()*100:.1f}%")


# [notebook cell 225]

# E21 -- corridor misses: information present or absent? (margins + identity)

import numpy as np, pandas as pd

lm = t_load == "USGC"
xs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
bins5 = pd.cut(tfrac, [0, .2, .4, .6, .8, 1.0], labels=xs)
inv_name = {v: k for k, v in subregion_names.items()}

# ---- (a) probability margins: is the true class a close #2 or truly gone? --
rows = []
for dst in ["NWE", "India SC", "SEAsia", "MED", "NEAsia_China"]:
    cid = inv_name[dst]
    m = lm & (t_dest == dst)
    p_true = tprobs[m, cid]
    top2 = np.argsort(tprobs[m], axis=1)[:, -2:]        # (n,2) top-2 class ids
    in_top2 = (top2 == cid).any(1)
    for b in xs:
        mb = (bins5[m] == b).values if hasattr(bins5[m], "values") else (bins5[m] == b)
        if mb.sum() < 100: continue
        rows.append(dict(dest=dst, stage=b, n=int(mb.sum()),
                         top1_acc=round(100 * (tpred[m][mb] == cid).mean(), 1),
                         in_top2=round(100 * in_top2[mb].mean(), 1),
                         mean_p_true=round(float(p_true[mb].mean()), 3),
                         margin_to_top=round(float(
                             (tprobs[m][mb].max(1) - p_true[mb]).mean()), 3)))
e21a = pd.DataFrame(rows)
print("=" * 90)
print("E21a -- is the true class nearly there? (USGC loadings)")
print("=" * 90)
print(e21a.to_string(index=False))
e21a.to_csv(os.path.join(WORK_DIR, "e21_margins.csv"), index=False)

# ---- (b) does the model use vessel identity mid-corridor? ------------------
# habitual vessel := modal past arrival == this voyage's true destination
tj = data.traj_idx
vcol = "IMO" if "IMO" in tj.columns else [c for c in tj.columns
        if c.lower() in ("imo", "vessel_id", "mmsi", "ship_id")][0]
tj2 = tj.dropna(subset=[TARGET_COL]).sort_values("dep_ts")
modal_before = {}
for vid, g in tj2.groupby(vcol):
    dests = g[TARGET_COL].astype(int).tolist()
    segs = g["seg_id"].tolist()
    for i, s in enumerate(segs):
        past = dests[:i]
        modal_before[int(s)] = (int(pd.Series(past).mode().iloc[0])
                                if past else -1)
seg_modal = np.array([modal_before.get(int(s), -1) for s in tseg])
habitual = seg_modal == ttrue
mid = (tfrac > 0.2) & (tfrac <= 0.6)
print("\n" + "=" * 90)
print("E21b -- mid-voyage (20-60%) accuracy: habitual vs non-habitual vessels")
print("=" * 90)
for dst in ["NWE", "India SC", "SEAsia"]:
    m = lm & (t_dest == dst) & mid
    for lab, hm in [("habitual", m & habitual), ("non-habitual", m & ~habitual)]:
        if hm.sum() < 50:
            print(f"{dst:10s} {lab:13s}: n={hm.sum()} (too thin)"); continue
        print(f"{dst:10s} {lab:13s}: n={int(hm.sum()):6,}  "
              f"acc={100 * t_ok[hm].mean():5.1f}%")
print("\nreading: (a) high in_top2 + small margin = information present, "
      "argmax outvoted -> decision-level fix; (b) habitual vessels still "
      "missed mid-voyage = identity signal present in inputs but not "
      "consumed mid-voyage -> routing/architecture fix.")


# [notebook cell 226]

print([n for n in dir() if n.isidentifier() and len(n) <= 6 and
       not n.startswith("_")][:40])
# then map (adjust to what prints -- likely R/L/F/S or reps/labs/fracs/segs):
rep0, rtrue, rfrac, rseg = R, L, F, S


# [notebook cell 227]

# E22-BOTH (seed 123) -- MED/NWE and India/SEAsia probes, one pass

import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

rep0, rtrue, rfrac = R, L, F                       # now seed-123 arrays
inv = {v: k for k, v in subregion_names.items()}
BANDS = [("early", rfrac <= .33), ("mid", (rfrac > .33) & (rfrac <= .66)),
         ("late", rfrac > .66)]
PAIRS = [("MED", "NWE", "tab:red", "tab:purple", "e22_pca_med_nwe_seed123"),
         ("India SC", "SEAsia", "tab:green", "tab:orange",
          "e22_pca_india_seasia_seed123")]
for nA, nB, cA, cB, fname in PAIRS:
    A, B = inv[nA], inv[nB]
    rl = np.isin(rtrue, [A, B])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    probes, bases = [], []
    for ax, (lab, fm) in zip(axes, BANDS):
        bm = fm & rl
        X = PCA(2).fit_transform(rep0[bm]); y = rtrue[bm]
        for cid, col, nm in [(A, cA, nA), (B, cB, nB)]:
            ax.scatter(*X[y == cid].T, s=4, alpha=.35, c=col, label=nm)
        pr = LogisticRegression(max_iter=500).fit(rep0[bm], y)
        acc = 100 * pr.score(rep0[bm], y)
        base = 100 * max((y == A).mean(), (y == B).mean())
        probes.append(round(acc)); bases.append(round(base, 1))
        ax.set_title(f"{lab} (n={bm.sum():,}): probe {acc:.0f}%")
        ax.legend(markerscale=2)
    plt.suptitle(f"{nA} vs {nB}: designated-channel representations (seed 123)")
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, f"{fname}.png"), dpi=150); plt.show()
    print(f"{nA} vs {nB}: probes {probes} vs baselines {bases}")


# [notebook cell 228]

# E18-LANES-EXT -- accuracy-by-stage panels for four more load regions

# Prereqs: E17 arrays + t_load/t_dest in kernel (same as the USGC/ME panels).
import numpy as np, pandas as pd, matplotlib.pyplot as plt

LOADS = ["NEAsia_China", "SEAsia", "India SC", "SAM"]
MIN_STEPS = 500                      # lanes thinner than this are skipped
edges = np.arange(0, 1.0001, 0.05)
ctr = (edges[:-1] + edges[1:]) / 2

for load in LOADS:
    lmask = t_load == load
    if lmask.sum() == 0:
        print(f"SKIP {load}: no steps (check spelling vs {sorted(set(t_load))[:8]})")
        continue
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    lanes = (pd.Series(t_dest[lmask]).value_counts())
    kept = lanes[lanes >= MIN_STEPS]
    for dst, n in kept.items():
        m = lmask & (t_dest == dst)
        acc = [100 * t_ok[m & (tfrac > lo) & (tfrac <= hi)].mean()
               if (m & (tfrac > lo) & (tfrac <= hi)).sum() >= 30 else np.nan
               for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(100 * ctr, acc, "o-", ms=3, label=f"{dst} (n={n:,})")
    ax.set_xlabel("voyage progression (%)"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Accuracy by stage -- {load} loadings, per destination (TEST)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 102)
    plt.tight_layout()
    fname = f"e18_lanes_{load.replace(' ', '')}.png"
    plt.savefig(os.path.join(WORK_DIR, fname), dpi=150); plt.show()
    # per-lane early/mid/late summary, same format as the USGC/ME CSVs
    rows = []
    for dst, n in kept.items():
        m = lmask & (t_dest == dst)
        rows.append(dict(dest=dst, n_steps=int(n),
            early=round(100 * t_ok[m & (tfrac <= .2)].mean(), 1),
            mid=round(100 * t_ok[m & (tfrac > .2) & (tfrac <= .6)].mean(), 1),
            late=round(100 * t_ok[m & (tfrac > .6)].mean(), 1)))
    summ = pd.DataFrame(rows)
    print(f"\n{load} loadings -- per-lane summary:")
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(WORK_DIR, f"e18_lanes_{load.replace(' ','')}.csv"),
                index=False)


# [notebook cell 229]

# E23 -- NEAsia loadings: the Canada and SEAsia misses, dissected

# Prereqs: E17 arrays, t_load/t_dest, tprobs; H1's modal_before if built.
import numpy as np, pandas as pd

inv = {v: k for k, v in subregion_names.items()}
name_of = subregion_names
lm = t_load == "NEAsia_China"
xs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
bins5 = pd.cut(tfrac, [0, .2, .4, .6, .8, 1.0], labels=xs)
t_pred_name = np.array([name_of.get(int(c), str(c)) for c in tpred])

for dst in ["Canada", "SEAsia"]:
    cid = inv[dst]
    m = lm & (t_dest == dst)
    print("=" * 88)
    print(f"NEAsia -> {dst}  (n={m.sum():,} steps)")
    print("=" * 88)
    # (a) who captures, by stage
    tab = (pd.DataFrame({"stage": bins5[m], "pred": t_pred_name[m]})
           .groupby(["stage", "pred"], observed=True).size().unstack(fill_value=0))
    tab = (tab.div(tab.sum(1), axis=0) * 100).round(1)
    keep = tab.mean().sort_values(ascending=False).head(6).index
    print("predicted mix (%):"); print(tab[keep].to_string())
    # (b) margins: is the truth even in top-2?
    p_true = tprobs[m, cid]
    top2 = np.argsort(tprobs[m], axis=1)[:, -2:]
    in2 = (top2 == cid).any(1)
    for b in xs:
        mb = (bins5[m] == b)
        mb = mb.values if hasattr(mb, "values") else mb
        if mb.sum() < 30: continue
        print(f"  {b}: in_top2 {100*in2[mb].mean():5.1f}%   "
              f"mean p_true {p_true[mb].mean():.3f}   "
              f"rank_of_truth(med) "
              f"{np.median((tprobs[m][mb] > p_true[mb,None]).sum(1)) + 1:.0f}")
    # (c) habitual split (mid-voyage), if H1's modal map exists
    if "seg_modal" in dir():
        mid = m & (tfrac > .2) & (tfrac <= .6)
        hab = mid & (seg_modal == ttrue); non = mid & (seg_modal != ttrue)
        for lab, mm in [("habitual", hab), ("non-habitual", non)]:
            if mm.sum() >= 30:
                print(f"  mid-voyage {lab}: n={int(mm.sum()):,}  "
                      f"acc={100*t_ok[mm].mean():.1f}%")
            else:
                print(f"  mid-voyage {lab}: n={int(mm.sum())} (thin)")
    # (d) training prior for this lane
    tr = data.traj_idx[pd.to_datetime(data.traj_idx["dep_ts"])
                       < pd.Timestamp(TEST_START)]
    tr_ne = tr[tr["DEP_PORT_ID"].map(lambda p: port_to_sub.get(int(p), -1))
                .map(lambda s: name_of.get(int(s), "")) == "NEAsia_China"]
    n_dst = (tr_ne[TARGET_COL] == cid).sum()
    print(f"  training NEAsia departures to {dst}: {n_dst} of {len(tr_ne)} "
          f"({100*n_dst/max(len(tr_ne),1):.1f}%)")


# [notebook cell 230]

# E24 -- share of all errors accounted for by the investigated lanes

# Lane set: USGC->{MED, NWE, India SC} + NEAsia->{Canada, SEAsia}.
import numpy as np, pandas as pd

LANES = [("USGC", "MED"), ("USGC", "NWE"), ("USGC", "India SC"),
         ("NEAsia_China", "Canada"), ("NEAsia_China", "SEAsia")]
lane_m = np.zeros_like(t_ok, bool)
for ld, ds in LANES:
    lane_m |= (t_load == ld) & (t_dest == ds)

miss = ~t_ok
BANDS = [("Early (0-20%)", tfrac <= .2),
         ("Mid (20-60%)", (tfrac > .2) & (tfrac <= .6)),
         ("Late (60-100%)", tfrac > .6), ("All stages", np.ones_like(t_ok, bool))]
rows = []
for lab, bm in BANDS:
    n_miss = int((miss & bm).sum())
    n_lane_miss = int((miss & bm & lane_m).sum())
    rows.append(dict(stage=lab,
        total_misses=n_miss,
        investigated_lane_misses=n_lane_miss,
        share_of_all_errors=round(100 * n_lane_miss / max(n_miss, 1), 1),
        lane_share_of_all_steps=round(100 * (lane_m & bm).sum() / max(bm.sum(), 1), 1),
        concentration=round((n_lane_miss / max(n_miss, 1)) /
                            max((lane_m & bm).sum() / max(bm.sum(), 1), 1e-9), 2)))
e24 = pd.DataFrame(rows)
print("=" * 92)
print("E24 -- errors accounted for by the investigated lanes "
      "(USGC->MED/NWE/India + NEAsia->Canada/SEAsia)")
print("=" * 92)
print(e24.to_string(index=False))
e24.to_csv(os.path.join(WORK_DIR, "e24_error_share.csv"), index=False)
print("\nreading: share_of_all_errors = how much of the model's total error "
      "the report's case studies explain; concentration = that share divided "
      "by the lanes' share of steps (x1 = proportionate, >1 = error hotspot).")


# [notebook cell 231]

# E24-PLOT -- error share of the investigated lanes, by stage

import numpy as np, pandas as pd, matplotlib.pyplot as plt

LANES = [("USGC", "MED"), ("USGC", "NWE"), ("USGC", "India SC"),
         ("NEAsia_China", "Canada"), ("NEAsia_China", "SEAsia")]
lane_m = np.zeros_like(t_ok, bool)
for ld, ds in LANES:
    lane_m |= (t_load == ld) & (t_dest == ds)
miss = ~t_ok
BANDS = [("Early\n(0-20%)", tfrac <= .2),
         ("Mid\n(20-60%)", (tfrac > .2) & (tfrac <= .6)),
         ("Late\n(60-100%)", tfrac > .6)]

shares, steps, ns = [], [], []
for lab, bm in BANDS:
    n_miss = (miss & bm).sum()
    shares.append(100 * (miss & bm & lane_m).sum() / max(n_miss, 1))
    steps.append(100 * (lane_m & bm).sum() / max(bm.sum(), 1))
    ns.append(int(n_miss))

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w/2, shares, w, color="tab:red", alpha=.85,
            label="share of all ERRORS in these 5 lanes")
b2 = ax.bar(x + w/2, steps, w, color="tab:gray", alpha=.7,
            label="share of all STEPS in these 5 lanes")
for i, (s, st) in enumerate(zip(shares, steps)):
    ax.text(i - w/2, s + 0.8, f"{s:.1f}%", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(i + w/2, st + 0.8, f"{st:.1f}%", ha="center", fontsize=9)
    ax.text(i, -4.8, f"{shares[i]/max(steps[i],1e-9):.1f}\u00d7 concentration",
            ha="center", fontsize=9, color="tab:red")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BANDS])
ax.set_ylabel("share (%)")
ax.set_title("The investigated lanes (USGC\u2192MED/NWE/India, "
             "NEAsia\u2192Canada/SEAsia)\nas a share of the model's errors, "
             "by voyage stage (TEST)")
ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(shares) + 8)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e24_error_share.png"), dpi=150,
            bbox_inches="tight")
plt.show()
print(pd.DataFrame({"stage": [b[0].replace(chr(10)," ") for b in BANDS],
                    "error_share": np.round(shares, 1),
                    "step_share": np.round(steps, 1),
                    "n_misses": ns}).to_string(index=False))


# [notebook cell 232]

# E25 -- which lanes hold the misses: top lanes to 50% of errors, per stage

import numpy as np, pandas as pd

miss = ~t_ok
BANDS = [("Early (0-20%)", tfrac <= .2),
         ("Mid (20-60%)", (tfrac > .2) & (tfrac <= .6)),
         ("Late (60-100%)", tfrac > .6)]
for lab, bm in BANDS:
    mm = miss & bm
    df = pd.DataFrame({"lane": pd.Series(t_load[mm]) + " -> " + pd.Series(t_dest[mm])})
    counts = df["lane"].value_counts()
    share = 100 * counts / counts.sum()
    cum = share.cumsum()
    k = int(np.searchsorted(cum.values, 50.0)) + 1      # lanes to reach 50%
    top = pd.DataFrame({"misses": counts, "share_%": share.round(1),
                        "cum_%": cum.round(1)}).head(max(k, 8))
    # context: each lane's share of the band's STEPS (is it a hotspot or just big?)
    steps = pd.Series(t_load[bm]) + " -> " + pd.Series(t_dest[bm])
    st_share = 100 * steps.value_counts() / bm.sum()
    top["step_share_%"] = st_share.reindex(top.index).round(1)
    top["concentration"] = (top["share_%"] / top["step_share_%"]).round(1)
    print("=" * 84)
    print(f"{lab}: {int(mm.sum()):,} misses; {k} lanes account for 50% of them")
    print("=" * 84)
    print(top.to_string())
    top.to_csv(os.path.join(WORK_DIR,
        f"e25_top_miss_lanes_{lab.split()[0].lower()}.csv"))


# [notebook cell 233]

# E25-PLOT v2 -- top miss lanes per stage, coloured by taxonomy mode

# Every bar is coloured by the error mode the lane belongs to (master table).
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from matplotlib.patches import Patch

MODE_OF_LANE = {
    "USGC -> NWE": 1, "USGC -> MED": 1, "USGC -> Africa": 1,
    "USGC -> India SC": 2, "USGC -> SEAsia": 2,
    "NEAsia_China -> Canada": 3,
    "NEAsia_China -> SEAsia": 4,
}
MODE_COL = {
    1: "#c62828",   # geometry capture, released at the fork
    2: "#ef6c00",   # geometry capture, fork too late
    3: "#6a1b9a",   # minority prototype / adjacent-coast capture
    4: "#1565c0",   # context prior against the geometry
    5: "#2e7d32",   # context prior with the geometry (intra-regional)
    6: "#00838f",   # diffuse: return-home / load-majority / near-miss
    9: "#9e9e9e",   # not individually typed
}
MODE_LAB = {
    1: "1 -- geometry capture, released at fork",
    2: "2 -- geometry capture, fork too late",
    3: "3 -- minority prototype (adjacent coast)",
    4: "4 -- context prior against geometry",
    5: "5 -- context prior with geometry (intra-regional)",
    6: "6-8 -- diffuse (majority default / return-home / near-miss)",
    9: "9 -- not individually typed",
}

def mode_of(lane, load, dest):
    if lane in MODE_OF_LANE:
        return MODE_OF_LANE[lane]
    if load == dest:
        return 5
    return 6            # diffuse family; refine here if desired

miss = ~t_ok
BANDS = [("Early (0-20%)", tfrac <= .2),
         ("Mid (20-60%)", (tfrac > .2) & (tfrac <= .6)),
         ("Late (60-100%)", tfrac > .6)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
for ax, (lab, bm) in zip(axes, BANDS):
    mm = miss & bm
    lanes = pd.Series(t_load[mm]).values + " -> " + pd.Series(t_dest[mm]).values
    counts = pd.Series(lanes).value_counts()
    share = (100 * counts / counts.sum()).head(10)[::-1]
    steps = pd.Series(t_load[bm]).values + " -> " + pd.Series(t_dest[bm]).values
    st = (100 * pd.Series(steps).value_counts() / bm.sum()).reindex(share.index)
    conc = (share / st)
    cols, modes = [], []
    for l in share.index:
        ld, ds = l.split(" -> ")
        m_ = mode_of(l, ld, ds)
        modes.append(m_); cols.append(MODE_COL[m_])
    ax.barh(range(len(share)), share.values, color=cols, alpha=.9)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([f"{l.replace('NEAsia_China','NEAsia')}  [M{m}]"
                        for l, m in zip(share.index, modes)], fontsize=8.5)
    for i, (s, c) in enumerate(zip(share.values, conc.values)):
        ax.text(s + 0.15, i, f"{s:.1f}%  ({c:.1f}\u00d7)", va="center", fontsize=8)
    ax.set_title(f"{lab}\n({int(mm.sum()):,} misses)", fontsize=10.5)
    ax.set_xlabel("share of the stage's errors (%)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(share.values) * 1.35)

handles = [Patch(facecolor=MODE_COL[k], label=MODE_LAB[k])
           for k in [1, 2, 3, 4, 5, 6]]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.09))
fig.suptitle("Where the errors live: top ten miss lanes per voyage stage, "
             "coloured by error mode\n(label: share of stage errors, "
             "concentration vs. the lane's share of steps)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e25_top_miss_lanes_bymode.png"), dpi=150,
            bbox_inches="tight")
plt.show()


# [notebook cell 234]

# E26 -- USGC -> Africa: full miss dissection + intra-regional error share

import numpy as np, pandas as pd

inv = {v: k for k, v in subregion_names.items()}
name_of = subregion_names
xs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
bins5 = pd.cut(tfrac, [0, .2, .4, .6, .8, 1.0], labels=xs)
t_pred_name = np.array([name_of.get(int(c), str(c)) for c in tpred])
AFID = inv["Africa"]
m = (t_load == "USGC") & (t_dest == "Africa")
print("=" * 92)
print(f"E26 -- USGC -> Africa (n={m.sum():,} steps)")
print("=" * 92)

# (a) who captures, by stage
tab = (pd.DataFrame({"stage": bins5[m], "pred": t_pred_name[m]})
       .groupby(["stage", "pred"], observed=True).size().unstack(fill_value=0))
tab = (tab.div(tab.sum(1), axis=0) * 100).round(1)
keep = tab.mean().sort_values(ascending=False).head(7).index
print("(a) predicted mix (%):"); print(tab[keep].to_string())

# (b) probability diagnostics per stage
p_true = tprobs[m, AFID]
top2 = np.argsort(tprobs[m], axis=1)[:, -2:]
in2 = (top2 == AFID).any(1)
print("\n(b) probability diagnostics:")
for b in xs:
    mb = bins5[m] == b
    mb = mb.values if hasattr(mb, "values") else mb
    print(f"  {b}: acc {100*(t_pred_name[m][mb]=='Africa').mean():5.1f}%  "
          f"in_top2 {100*in2[mb].mean():5.1f}%  "
          f"p_true {p_true[mb].mean():.3f}  "
          f"rank(med) {np.median((tprobs[m][mb] > p_true[mb,None]).sum(1))+1:.0f}")

# (c) habitual split, mid-voyage (needs seg_modal from the H1/E21 block)
if "seg_modal" in dir():
    mid = m & (tfrac > .2) & (tfrac <= .6)
    for lab, mm2 in [("habitual", mid & (seg_modal == ttrue)),
                     ("non-habitual", mid & (seg_modal != ttrue))]:
        print(f"(c) mid-voyage {lab}: n={int(mm2.sum()):,}  "
              f"acc={100*t_ok[mm2].mean():.1f}%" if mm2.sum() >= 30 else
              f"(c) mid-voyage {lab}: n={int(mm2.sum())} (thin)")
else:
    print("(c) seg_modal absent -- run the E21b habitual block for this probe")

# (d) training prior + voyage-length profile of the lane
tr = data.traj_idx[pd.to_datetime(data.traj_idx["dep_ts"]) < pd.Timestamp(TEST_START)]
tr_us = tr[tr["DEP_PORT_ID"].map(lambda p: port_to_sub.get(int(p), -1))
            .map(lambda s: name_of.get(int(s), "")) == "USGC"]
n_af = (tr_us[TARGET_COL] == AFID).sum()
print(f"(d) training USGC departures to Africa: {n_af} of {len(tr_us)} "
      f"({100*n_af/max(len(tr_us),1):.1f}%)")
print(f"    lane voyage durations (test): median "
      f"{np.median(tdur[m]):.0f}d, IQR "
      f"{np.percentile(tdur[m],25):.0f}-{np.percentile(tdur[m],75):.0f}d")

# (e) intra-regional misses: share of each stage's total errors
miss = ~t_ok
intra = t_load == t_dest
BANDS = [("Early (0-20%)", tfrac <= .2), ("Mid (20-60%)", (tfrac > .2) & (tfrac <= .6)),
         ("Late (60-100%)", tfrac > .6), ("All", np.ones_like(t_ok, bool))]
print("\n(e) intra-regional (load == dest) share of all errors:")
for lab, bm in BANDS:
    n_miss = (miss & bm).sum()
    n_in = (miss & bm & intra).sum()
    print(f"  {lab:15s}: {100*n_in/max(n_miss,1):5.1f}% of errors "
          f"(step share {100*(intra & bm).sum()/max(bm.sum(),1):.1f}%, "
          f"n={int(n_in):,})")


# [notebook cell 235]

# E27 -- India SC -> India SC: full profile of the intra-regional miss

import numpy as np, pandas as pd

INDID = inv["India SC"]
m = (t_load == "India SC") & (t_dest == "India SC")
print("=" * 92)
print(f"E27 -- India SC -> India SC (n={m.sum():,} steps)")
print("=" * 92)

# (a) captors by stage
tab = (pd.DataFrame({"stage": bins5[m], "pred": t_pred_name[m]})
       .groupby(["stage", "pred"], observed=True).size().unstack(fill_value=0))
tab = (tab.div(tab.sum(1), axis=0) * 100).round(1)
print("(a) predicted mix (%):")
print(tab[tab.mean().sort_values(ascending=False).head(7).index].to_string())

# (b) probability diagnostics
p_true = tprobs[m, INDID]
top2 = np.argsort(tprobs[m], axis=1)[:, -2:]; in2 = (top2 == INDID).any(1)
print("\n(b) probability diagnostics:")
for b in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
    mb = bins5[m] == b; mb = mb.values if hasattr(mb, "values") else mb
    if mb.sum() < 30: continue
    print(f"  {b}: acc {100*(t_pred_name[m][mb]=='India SC').mean():5.1f}%  "
          f"in_top2 {100*in2[mb].mean():5.1f}%  p_true {p_true[mb].mean():.3f}  "
          f"rank(med) {np.median((tprobs[m][mb] > p_true[mb,None]).sum(1))+1:.0f}")

# (c) voyage profile: these should be SHORT hops -- confirm, and check
#     the short-voyage representation effect applies
print(f"\n(c) durations: median {np.median(tdur[m]):.0f}d, "
      f"IQR {np.percentile(tdur[m],25):.0f}-{np.percentile(tdur[m],75):.0f}d, "
      f"share <=5d {100*(tdur[m]<=5).mean():.0f}%")
segs = np.unique(tseg[m]); print(f"    distinct voyages: {len(segs)}")

# (d) habitual + history-depth profile of the lane's vessels
if "seg_modal" in dir():
    hab = (seg_modal == ttrue)
    print(f"(d) habitual steps: {100*hab[m].mean():.0f}%  "
          f"acc habitual {100*t_ok[m & hab].mean():.1f}% "
          f"vs non-habitual {100*t_ok[m & ~hab].mean():.1f}%")
if "t_nprior" in dir():
    print(f"    median prior voyages: {np.median(t_nprior[m]):.0f}")

# (e) the coastal-shuttle hypothesis: within-India port pairs
tj = data.traj_idx.set_index("seg_id")
sub = tj.loc[[int(s) for s in segs]]
if "DEP_PORT_ID" in sub.columns and "ARR_PORT_ID" in sub.columns:
    pairs = (sub.groupby(["DEP_PORT_ID", "ARR_PORT_ID"]).size()
             .sort_values(ascending=False).head(8))
    print("(e) top within-India port pairs (voyages):"); print(pairs.to_string())


# [notebook cell 236]

# E28 -- final synthesis: error share by identified problem type, per stage

import numpy as np, pandas as pd

L = lambda ld, ds: (t_load == ld) & (t_dest == ds)
MODES = {
 "Geometry capture, released at fork (NWE, MED-side, Africa)":
     L("USGC","NWE") | L("USGC","MED") | L("USGC","Africa"),
 "Geometry capture, fork too late (India, SEAsia via Cape)":
     L("USGC","India SC") | L("USGC","SEAsia"),
 "Sequential capture to adjacent coast (NEAsia->Canada)":
     L("NEAsia_China","Canada"),
 "Context prior vs geometry (NEAsia->SEAsia)":
     L("NEAsia_China","SEAsia"),
 "Context prior WITH geometry (intra-regional, incl India->India)":
     (t_load == t_dest),
}
miss = ~t_ok
BANDS = [("Early", tfrac <= .2), ("Mid", (tfrac > .2) & (tfrac <= .6)),
         ("Late", tfrac > .6)]
rows = []
covered = np.zeros_like(t_ok, bool)
for name, mm in MODES.items():
    covered |= mm
    r = {"problem_type": name}
    for lab, bm in BANDS:
        r[lab] = round(100 * (miss & bm & mm).sum() / max((miss & bm).sum(), 1), 1)
    rows.append(r)
r = {"problem_type": "Uncharacterised (all other lanes)"}
for lab, bm in BANDS:
    r[lab] = round(100 * (miss & bm & ~covered).sum() / max((miss & bm).sum(), 1), 1)
rows.append(r)
e28 = pd.DataFrame(rows)
print(e28.to_string(index=False))
e28.to_csv(os.path.join(WORK_DIR, "e28_error_taxonomy.csv"), index=False)


# [notebook cell 237]

# E29 -- profiling the UNcharacterised errors: candidate new types

# Prereqs: E17 arrays, t_load/t_dest, tprobs; seg_modal (E21b) optional.
import numpy as np, pandas as pd

L = lambda ld, ds: (t_load == ld) & (t_dest == ds)
charac = (L("USGC","NWE") | L("USGC","MED") | L("USGC","Africa") |
          L("USGC","India SC") | L("USGC","SEAsia") |
          L("NEAsia_China","Canada") | L("NEAsia_China","SEAsia") |
          (t_load == t_dest))
miss = ~t_ok
U = miss & ~charac
name_of = subregion_names
t_pred_name = np.array([name_of.get(int(c), str(c)) for c in tpred])
BANDS = [("Early", tfrac <= .2), ("Mid", (tfrac > .2) & (tfrac <= .6)),
         ("Late", tfrac > .6)]

# ---- (a) top lanes inside the uncharacterised pool -------------------------
lanes = pd.Series(t_load[U]) + " -> " + pd.Series(t_dest[U])
top = lanes.value_counts().head(14)
print("(a) top uncharacterised miss lanes (all stages):")
print((100 * top / U.sum()).round(1).to_string())

# ---- (b) overlapping error-type flags, per stage ---------------------------
# majority destination of each load region (training-free, test-side approx)
maj_of_load = {ld: pd.Series(t_dest[t_load == ld]).mode().iloc[0]
               for ld in set(t_load)}
maj_pred = np.array([maj_of_load.get(l, "") for l in t_load])
# geographic adjacency (subregion neighbours -- edit as appropriate)
ADJ = {"NWE": {"MED"}, "MED": {"NWE", "Africa"}, "Africa": {"MED", "SEAsia", "India SC"},
       "India SC": {"ME", "SEAsia", "Africa"}, "SEAsia": {"NEAsia_China", "India SC", "Oceania", "Africa"},
       "NEAsia_China": {"SEAsia"}, "Canada": {"USWC"}, "USWC": {"Canada", "USGC"},
       "USGC": {"USWC", "CAM", "SAM"}, "CAM": {"USGC", "SAM"}, "SAM": {"CAM", "Africa", "USGC"},
       "ME": {"India SC", "Africa"}, "Oceania": {"SEAsia"}}
adj_flag = np.array([p in ADJ.get(d, set())
                     for p, d in zip(t_pred_name, t_dest)])
p_true_all = tprobs[np.arange(len(ttrue)), ttrue]
top2 = np.argsort(tprobs, axis=1)[:, -2:]
in2 = (top2 == ttrue[:, None]).any(1)

FLAGS = {
 "pred = load region (return-home assumption)": t_pred_name == t_load,
 "pred = load region's majority destination": t_pred_name == maj_pred,
 "pred adjacent to truth (near-miss basin)": adj_flag,
 "truth in top-2 (decision-recoverable)": in2,
}
if "seg_modal" in dir():
    FLAGS["pred = vessel's modal habit (habit-following)"] = tpred == seg_modal
rows = []
for fname, fm in FLAGS.items():
    r = {"error_type_flag": fname}
    for lab, bm in BANDS:
        r[lab] = round(100 * (U & bm & fm).sum() / max((U & bm).sum(), 1), 1)
    rows.append(r)
e29 = pd.DataFrame(rows)
print("\n(b) share of UNcharacterised errors carrying each flag (overlapping):")
print(e29.to_string(index=False))
e29.to_csv(os.path.join(WORK_DIR, "e29_uncharacterised_profile.csv"), index=False)
print("\n(c) mean p_true / in_top2 on uncharacterised misses, per stage:")
for lab, bm in BANDS:
    mm = U & bm
    print(f"  {lab}: p_true {p_true_all[mm].mean():.3f}   "
          f"in_top2 {100*in2[mm].mean():.1f}%   n={int(mm.sum()):,}")


# [notebook cell 238]

# E30 -- top-1 vs top-2 accuracy by stage (the decision-surface bound)

import numpy as np, pandas as pd

top2 = np.argsort(tprobs, axis=1)[:, -2:]
ok2 = (top2 == ttrue[:, None]).any(1)
xs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
bins5 = pd.cut(tfrac, [0, .2, .4, .6, .8, 1.0], labels=xs)
rows = []
for b in xs + ["ALL"]:
    bm = np.ones_like(t_ok, bool) if b == "ALL" else \
         (bins5 == b).values if hasattr(bins5 == b, "values") else (bins5 == b)
    rows.append(dict(stage=b, n=int(bm.sum()),
                     top1=round(100 * t_ok[bm].mean(), 1),
                     top2=round(100 * ok2[bm].mean(), 1),
                     gain=round(100 * (ok2[bm].mean() - t_ok[bm].mean()), 1)))
e30 = pd.DataFrame(rows)
print(e30.to_string(index=False))
e30.to_csv(os.path.join(WORK_DIR, "e30_top1_top2.csv"), index=False)


# [notebook cell 239]

# E30-PLOT -- top-1 vs top-2 accuracy by stage

import numpy as np, matplotlib.pyplot as plt

xs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
top1 = [69.4, 77.5, 81.7, 86.1, 87.7]
top2 = [83.0, 89.8, 92.5, 94.8, 95.5]
x = np.arange(5)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(x, top1, "o-", lw=2, color="tab:blue", label="top-1 accuracy")
ax.plot(x, top2, "s-", lw=2, color="tab:green",
        label="top-2 coverage (two-candidate bound)")
ax.fill_between(x, top1, top2, alpha=.15, color="tab:green")
for i in range(5):
    ax.annotate(f"+{top2[i]-top1[i]:.1f}pp",
                xy=(i, (top1[i] + top2[i]) / 2),
                xytext=(i + 0.08, (top1[i] + top2[i]) / 2),
                fontsize=9, color="tab:green", va="center")
    ax.text(i, top1[i] - 2.6, f"{top1[i]:.1f}", ha="center", fontsize=9,
            color="tab:blue")
    ax.text(i, top2[i] + 1.2, f"{top2[i]:.1f}", ha="center", fontsize=9,
            color="tab:green")
ax.set_xticks(x); ax.set_xticklabels(xs)
ax.set_xlabel("voyage progression"); ax.set_ylabel("accuracy / coverage (%)")
ax.set_title("Top-1 accuracy vs top-2 coverage by voyage stage (TEST, 3 seeds)")
ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_ylim(60, 100)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e30_top1_top2.png"), dpi=150,
            bbox_inches="tight")
plt.show()


# [notebook cell 240]

# E31 -- taxonomy v2: hierarchical, mutually exclusive typing of ALL misses

import numpy as np, pandas as pd

L = lambda ld, ds: (t_load == ld) & (t_dest == ds)
name_of = subregion_names
t_pred_name = np.array([name_of.get(int(c), str(c)) for c in tpred])
maj_of_load = {ld: pd.Series(t_dest[t_load == ld]).mode().iloc[0]
               for ld in set(t_load)}
maj_pred = np.array([maj_of_load.get(l, "") for l in t_load])
ADJ = {"NWE": {"MED"}, "MED": {"NWE", "Africa"},
       "Africa": {"MED", "SEAsia", "India SC"},
       "India SC": {"ME", "SEAsia", "Africa"},
       "SEAsia": {"NEAsia_China", "India SC", "Oceania", "Africa"},
       "NEAsia_China": {"SEAsia"}, "Canada": {"USWC"},
       "USWC": {"Canada", "USGC"}, "USGC": {"USWC", "CAM", "SAM"},
       "CAM": {"USGC", "SAM"}, "SAM": {"CAM", "Africa", "USGC"},
       "ME": {"India SC", "Africa"}, "Oceania": {"SEAsia"}}
adj = np.array([p in ADJ.get(d, set()) for p, d in zip(t_pred_name, t_dest)])

# precedence order: first match wins
TYPES = [
 ("1 Geometry capture, released (NWE/MED/Africa)",
    L("USGC","NWE") | L("USGC","MED") | L("USGC","Africa")),
 ("2 Geometry capture, fork too late (India/SEAsia)",
    L("USGC","India SC") | L("USGC","SEAsia")),
 ("3 Sequential adjacent-coast capture (NEAsia->Canada)", L("NEAsia_China","Canada")),
 ("4 Context prior vs geometry (NEAsia->SEAsia)", L("NEAsia_China","SEAsia")),
 ("5 Context prior with geometry (intra-regional)", t_load == t_dest),
 ("6 Return-home assumption (pred = load region)", t_pred_name == t_load),
 ("7 Load-majority default (pred = load's top lane)", t_pred_name == maj_pred),
 ("8 Adjacent-basin near-miss (pred neighbours truth)", adj),
 ("9 Other / unexplained", np.ones_like(t_ok, bool)),
]
miss = ~t_ok
assign = np.full(len(t_ok), -1)
for k, (nm, mk) in enumerate(TYPES):
    assign[(assign < 0) & miss & mk] = k
BANDS = [("Early", tfrac <= .2), ("Mid", (tfrac > .2) & (tfrac <= .6)),
         ("Late", tfrac > .6)]
rows = []
for k, (nm, _) in enumerate(TYPES):
    r = {"problem_type": nm}
    for lab, bm in BANDS:
        r[lab] = round(100 * ((assign == k) & bm).sum() /
                       max((miss & bm).sum(), 1), 1)
    rows.append(r)
e31 = pd.DataFrame(rows)
print(e31.to_string(index=False))
print("\ncolumn sums:", [round(e31[c].sum(), 1) for c in ["Early","Mid","Late"]])
e31.to_csv(os.path.join(WORK_DIR, "e31_taxonomy_v2.csv"), index=False)


# [notebook cell 241]

# E32 -- the fork, magnified: fine late bins + flip-point for India/SEAsia

import numpy as np, pandas as pd
for ld, ds in [("USGC", "India SC"), ("USGC", "SEAsia"), ("USGC", "NWE")]:
    m = (t_load == ld) & (t_dest == ds)
    print("=" * 70); print(f"{ld} -> {ds}")
    for lo, hi in [(.80, .85), (.85, .90), (.90, .95), (.95, 1.01)]:
        mb = m & (tfrac >= lo) & (tfrac < hi)
        if mb.sum() < 50: continue
        print(f"  {int(lo*100)}-{int(hi*100)}%: acc "
              f"{100*t_ok[mb].mean():5.1f}%  (n={int(mb.sum()):,})")
    # flip point: per voyage, last progression at which prediction was wrong
    segs = np.unique(tseg[m])
    flips = []
    for s in segs:
        sm = m & (tseg == s)
        wrong = ~t_ok[sm]; fr = tfrac[sm]
        flips.append(fr[wrong].max() if wrong.any() else 0.0)
    flips = np.array(flips)
    print(f"  last-wrong point: median {100*np.median(flips):.0f}% of voyage; "
          f"never-flips (wrong at final step): "
          f"{100*(flips > 0.97).mean():.0f}% of voyages")


# [notebook cell 242]

# E33 -- which channel carries which prior: per-lane ablation deltas

import numpy as np, pandas as pd
LANES = [("NEAsia_China","SEAsia"), ("India SC","India SC"),
         ("USGC","NWE"), ("USGC","India SC"), ("USGC","Africa"),
         ("NEAsia_China","Canada")]
rows = []
for ld, ds in LANES:
    m = (t_load == ld) & (t_dest == ds)
    r = {"lane": f"{ld}->{ds}", "n": int(m.sum()),
         "base_acc": round(100 * t_ok[m].mean(), 1)}
    for lab, ok_abl in ABL_PRED.items():
        r[lab] = round(100 * (ok_abl[m].mean() - t_ok[m].mean()), 1)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nreading: POSITIVE = deleting the channel HELPS the lane "
      "(that channel carries the lane's wrong prior).")
