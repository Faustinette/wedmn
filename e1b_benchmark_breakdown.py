# E1-B: BENCHMARK BREAKDOWN BY VOYAGE DURATION (supporting analysis)
#
# Slices the E1 model / captain / combined comparison by voyage duration.
# Four blocks, in order:
#
#   1. EARLYBARS   Early-band (0-20% progression) accuracy bars for the
#                  three series, split at a duration threshold; run for
#                  both 14-day and 10-day thresholds. Figure + CSV.
#   2. TABLES      The same numbers pivoted into one report table per
#                  threshold. CSV per threshold.
#   3. LENCURVE    Overall and early-band accuracy versus voyage duration
#                  (six quantile bins), all three series. Figure + CSV.
#   4. DURDIST     Test-voyage duration distribution: summary statistics,
#                  binned counts, histogram with split boundaries.
#
# ROLE IN THE REPORT: supporting. E1 carries the headline benchmark; this
# file provides the duration-resolved evidence behind the short-versus-long
# voyage discussion (typically one figure or table in the main text, the
# rest appendix material).
#
# Prerequisites from E1 (must run first): final_runs, subregion_name_map,
# CAPTAIN_DEDUP_STRATEGY, SERIES. The duration series _tl is rebuilt here
# if absent.


# [notebook cell 75]

# E1-EARLYBARS -- early-band (0-20%) accuracy bars: model / combined / captain
#                 split at a duration threshold; run for 14d and 10d

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, HTML

THRESHOLDS = [14, 10]
BAR_ORDER = [("model", "Model"),
             ("captain", "Captain declaration alone"),
             ("combined", "Model + Captain declaration")]
_bounds = np.array(sorted(DEFAULT_PROGRESSION_BOUNDARIES))
_early = _bounds <= 0.20

def _early_acc(res):
    bc = np.array(res["band_correct"], dtype="float64")
    bt = np.array(res["band_total"], dtype="float64")
    return 100 * bc[_early].sum() / max(bt[_early].sum(), 1)

def _eval_group(ids, tag):
    out = {}
    for seed in SEEDS:
        r = final_runs[seed]
        r_comb = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES,
            condition_name=f"final_main_lean2_trainval_declared_seed{seed}",
            seed=seed, alt_progression_modes=ALT_PROGRESSION_MODES,
            gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
            use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
            use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
            use_departure_gate=USE_DEPARTURE_GATE, n_experts=N_EXPERTS,
            d_model=D_MODEL, stratify=True, val_frac=0.15,
            test_start=TEST_START, test_end=TEST_END,
            epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None,
            use_declared_destination=True, batch_size=BATCH_SIZE,
            work_dir=WORK_DIR, skip_existing=True)
        t_loader = BucketedWAYDataset(
            data, target_col=TARGET_COL, batch_size=BATCH_SIZE,
            seg_id_subset=ids, shuffle=False, seed=0, include_ship_history=True)
        res = build_model_vs_captain_combined_accuracy(
            r["model"], r["repr_layer"], t_loader, r["core_and_alt_fn"], data,
            WORK_DIR, subregion_name_map,
            departure_ids_fn=r.get("departure_ids_fn"),
            eta_channel_lookup=r.get("eta_channel_lookup"),
            combined_model=r_comb["model"], combined_repr_layer=r_comb["repr_layer"],
            combined_core_and_alt_fn=r_comb["core_and_alt_fn"],
            combined_departure_ids_fn=r_comb.get("departure_ids_fn"),
            combined_eta_channel_lookup=r_comb.get("eta_channel_lookup"),
            set_label=tag, dedup_strategy=CAPTAIN_DEDUP_STRATEGY)
        out[seed] = {k: _early_acc(res[k]) for k, _ in BAR_ORDER}
    return out

# _tl (durations) from E1-SPLIT / E1-DURDIST; rebuild if absent
if "_tl" not in dir():
    _tj = data.traj_idx.set_index("seg_id")
    _tl = pd.Series({int(s): (pd.Timestamp(_tj.loc[int(s), "arr_ts"])
                              - pd.Timestamp(_tj.loc[int(s), "dep_ts"])).days
                     for s in final_runs[SEEDS[0]]["_test_ids"]})

CACHE = {}                                    # (thr, side) -> per-seed dict
rows = []
for thr in THRESHOLDS:
    groups = {f"<= {thr} days": _tl[_tl <= thr].index.tolist(),
              f"> {thr} days":  _tl[_tl > thr].index.tolist()}
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    for gi, (gname, gids) in enumerate(groups.items()):
        key3 = CACHE.setdefault((thr, gname), _eval_group(gids, f"eb-{thr}-{gi}"))
        for bi, ((k, label), col) in enumerate(zip(BAR_ORDER, colors)):
            vals = [key3[s][k] for s in SEEDS]
            m, sd = np.mean(vals), np.std(vals)
            ax.bar(gi + (bi - 1) * width, m, width, yerr=sd, capsize=4,
                   color=col, label=label if gi == 0 else None,
                   edgecolor="white")
            ax.text(gi + (bi - 1) * width, m + sd + 0.5, f"{m:.1f}",
                    ha="center", fontsize=9)
            rows.append(dict(threshold=f"{thr}d", group=gname, series=label,
                             early_acc=round(m, 2), std=round(sd, 2),
                             n_voyages=len(gids)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(groups))
    ax.set_ylabel("Early-band accuracy, 0-20% progression (%)")
    ax.set_title(f"Early-voyage accuracy by duration (threshold {thr} days) -- "
                 "TEST, 3-seed mean \u00b1 std")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, f"e1_earlyband_bars_{thr}d.png"), dpi=150)
    plt.show()

eb_tab = pd.DataFrame(rows)
display(HTML("<b>Early-band accuracy by duration threshold "
             "(3-seed mean ± std)</b>"))
display(eb_tab)
eb_tab.to_csv(os.path.join(WORK_DIR, "e1_earlyband_bars.csv"), index=False)


# [notebook cell 76]

# ---- pivoted report tables: one per threshold -------------------------------
eb_tab["cell"] = eb_tab.apply(
    lambda r: f"{r['early_acc']:.2f} \u00b1 {r['std']:.2f}", axis=1)
for thr in THRESHOLDS:
    sub = eb_tab[eb_tab["threshold"] == f"{thr}d"]
    piv = sub.pivot_table(index="group", columns="series", values="cell",
                          aggfunc="first", sort=False)
    piv = piv[[lb for _, lb in BAR_ORDER]]          # enforce bar order
    ns = sub.drop_duplicates("group").set_index("group")["n_voyages"]
    piv.insert(0, "N voyages", ns)
    display(HTML(f"<b>Early-band accuracy (0–20%), threshold {thr} days</b>"))
    display(piv)
    piv.to_csv(os.path.join(WORK_DIR, f"e1_earlyband_table_{thr}d.csv"))


# [notebook cell 77]

# E1-LENCURVE -- overall & early-band accuracy vs voyage duration (3 series)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, HTML

N_LBINS = 6
_q = _tl.quantile(np.arange(1, N_LBINS) / N_LBINS).astype(int).tolist()
_ledges = [-np.inf] + _q + [np.inf]
LBINS = {}
for i in range(N_LBINS):
    m = (_tl > _ledges[i]) & (_tl <= _ledges[i + 1])
    lab = (f"<= {int(_ledges[1])}" if i == 0 else
           f"> {int(_ledges[i])}" if i == N_LBINS - 1 else
           f"{int(_ledges[i])+1}-{int(_ledges[i+1])}")
    LBINS[lab] = _tl[m].index.tolist()
print({k: len(v) for k, v in LBINS.items()})

_bounds = np.array(sorted(DEFAULT_PROGRESSION_BOUNDARIES))
_early = _bounds <= 0.20
LC = {}          # (bin, series, seed) -> (overall, early)
for lab, ids in LBINS.items():
    for seed in SEEDS:
        r = final_runs[seed]
        r_comb = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES,
            condition_name=f"final_main_lean2_trainval_declared_seed{seed}",
            seed=seed, alt_progression_modes=ALT_PROGRESSION_MODES,
            gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
            use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
            use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
            use_departure_gate=USE_DEPARTURE_GATE, n_experts=N_EXPERTS,
            d_model=D_MODEL, stratify=True, val_frac=0.15,
            test_start=TEST_START, test_end=TEST_END,
            epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None,
            use_declared_destination=True, batch_size=BATCH_SIZE,
            work_dir=WORK_DIR, skip_existing=True)
        t_loader = BucketedWAYDataset(
            data, target_col=TARGET_COL, batch_size=BATCH_SIZE,
            seg_id_subset=ids, shuffle=False, seed=0, include_ship_history=True)
        res = build_model_vs_captain_combined_accuracy(
            r["model"], r["repr_layer"], t_loader, r["core_and_alt_fn"], data,
            WORK_DIR, subregion_name_map,
            departure_ids_fn=r.get("departure_ids_fn"),
            eta_channel_lookup=r.get("eta_channel_lookup"),
            combined_model=r_comb["model"], combined_repr_layer=r_comb["repr_layer"],
            combined_core_and_alt_fn=r_comb["core_and_alt_fn"],
            combined_departure_ids_fn=r_comb.get("departure_ids_fn"),
            combined_eta_channel_lookup=r_comb.get("eta_channel_lookup"),
            set_label=f"lencurve-{lab}", dedup_strategy=CAPTAIN_DEDUP_STRATEGY)
        for key, _lb in SERIES:
            bc = np.array(res[key]["band_correct"], dtype="float64")
            bt = np.array(res[key]["band_total"], dtype="float64")
            overall = 100 * bc.sum() / max(bt.sum(), 1)
            early = 100 * bc[_early].sum() / max(bt[_early].sum(), 1)
            LC[(lab, key, seed)] = (overall, early)
    print(f"{lab}: done")

# ---- two-panel plot + table -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
colors = {"model": "#2ecc71", "captain": "#3498db", "combined": "#9b59b6"}
rows = []
for pi, (panel, title) in enumerate([(0, "Overall accuracy"),
                                     (1, "Early-band accuracy (0-20%)")]):
    for key, label in SERIES:
        means = [np.mean([LC[(lab, key, s)][panel] for s in SEEDS]) for lab in LBINS]
        stds  = [np.std([LC[(lab, key, s)][panel] for s in SEEDS]) for lab in LBINS]
        axes[pi].errorbar(range(len(LBINS)), means, yerr=stds, marker="o",
                          lw=2, capsize=3, color=colors.get(key), label=label)
        if pi == 0:
            for lab, m_, s_ in zip(LBINS, means, stds):
                rows.append(dict(bin=lab, series=label, metric="overall",
                                 acc=round(m_, 2), std=round(s_, 2)))
        else:
            for lab, m_, s_ in zip(LBINS, means, stds):
                rows.append(dict(bin=lab, series=label, metric="early",
                                 acc=round(m_, 2), std=round(s_, 2)))
    axes[pi].set_title(title); axes[pi].grid(alpha=0.3)
    axes[pi].set_xticks(range(len(LBINS)))
    axes[pi].set_xticklabels(list(LBINS), rotation=30, ha="right")
    axes[pi].set_xlabel("Voyage duration (days)")
axes[0].set_ylabel("Accuracy (%)"); axes[0].legend(fontsize=9)
fig.suptitle("Accuracy vs voyage duration -- model, captain, combined "
             "(TEST, 3-seed mean \u00b1 std)")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e1_accuracy_vs_duration.png"), dpi=150)
plt.show()

lc_tab = pd.DataFrame(rows).pivot_table(index="bin", columns=["metric", "series"],
                                        values="acc", sort=False)
display(HTML("<b>Accuracy by voyage duration (3-seed mean)</b>"))
display(lc_tab)
lc_tab.to_csv(os.path.join(WORK_DIR, "e1_accuracy_vs_duration.csv"))


# [notebook cell 78]

# E1-DURDIST -- test-voyage duration distribution: table + histogram

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, HTML

# _tl from E1-SPLIT (duration in days per test voyage); rebuild if absent
if "_tl" not in dir():
    _test_all = list(final_runs[SEEDS[0]]["_test_ids"])
    _tj = data.traj_idx.set_index("seg_id")
    _tl = pd.Series({int(s): (pd.Timestamp(_tj.loc[int(s), "arr_ts"])
                              - pd.Timestamp(_tj.loc[int(s), "dep_ts"])).days
                     for s in _test_all}).sort_values()

# ---- summary table ----------------------------------------------------------
q = _tl.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
summary = pd.DataFrame({
    "Statistic": ["N voyages", "Mean (days)", "Std", "Min", "P5", "P25",
                  "Median", "P75", "P95", "Max"],
    "Value": [len(_tl), round(_tl.mean(), 1), round(_tl.std(), 1),
              int(_tl.min()), int(q[0.05]), int(q[0.25]),
              int(_tl.median()), int(q[0.75]), int(q[0.95]), int(_tl.max())]})
display(HTML("<b>Test-voyage duration distribution "
             f"[{TEST_START} → {TEST_END}]</b>"))
display(summary)

# binned counts (aligned to the terciles when present)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 60, int(_tl.max()) + 1]
binned = pd.cut(_tl, bins=bins, right=True)
dist = binned.value_counts().sort_index().rename("N voyages").to_frame()
dist["% of test"] = (100 * dist["N voyages"] / len(_tl)).round(1)
display(dist)
summary.to_csv(os.path.join(WORK_DIR, "e1_duration_summary.csv"), index=False)
dist.to_csv(os.path.join(WORK_DIR, "e1_duration_bins.csv"))

# ---- histogram with the split boundaries overlaid ---------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(_tl.values, bins=np.arange(0, _tl.max() + 2, 2),
        color="#3498db", alpha=0.75, edgecolor="white")
for qv, lab in [(q[0.5], "median")] + \
               ([(e, "tercile") for e in _qs] if "_qs" in dir() else []):
    ax.axvline(qv, color="#e74c3c" if lab == "median" else "#7f8c8d",
               ls="--" if lab == "median" else ":", lw=1.5)
    ax.text(qv, ax.get_ylim()[1] * 0.95, f" {lab}\n {qv:.0f}d",
            fontsize=8, va="top",
            color="#e74c3c" if lab == "median" else "#555")
ax.set_xlabel("Voyage duration (days)"); ax.set_ylabel("N test voyages")
ax.set_title(f"Test-voyage duration distribution "
             f"({len(_tl):,} voyages, [{TEST_START} → {TEST_END}])")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "e1_duration_distribution.png"), dpi=150)
plt.show()
