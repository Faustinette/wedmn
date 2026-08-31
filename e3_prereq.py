# =============================================================================
# E3-PREREQ — per-seed epochs + E3-CONFIG v2 (TEST-side) + baseline model reload. Needed by E3/E4/E5/E8/H8.
# Migrated verbatim from Main_forGitHub.ipynb cells [126, 127, 128].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 126]
# ----------------------------------------------------------------------
# =============================================================================
# E3 LIB CELL -- results helpers (2 defs, verbatim from live Step4c)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summarize_accuracy_by_stage_multiseed(results_by_seed, boundaries=DEFAULT_PROGRESSION_BOUNDARIES,
                                           early_late_cutoffs=(0.20, 0.60)):
    """Multi-seed companion to summarize_accuracy_by_stage -- for the
    project's standard 3-seed ablation tables (mean/std across
    independent seed replicates), matching the SAME convention the
    existing "Mean Overall Accuracy"/"Std Overall Accuracy" columns
    already use, extended to Early/Mid/Late too.

    results_by_seed: {label: {seed: result_dict}}, where each
    result_dict has "band_correct"/"band_total"/"overall_acc" (from
    evaluate_quartile_accuracy, _evaluate_with_core_and_alt_progression,
    or equivalent).

    Computes each SEED's own stage accuracy independently (weighted by
    that seed's own raw counts, via summarize_accuracy_by_stage), THEN
    takes the mean/std ACROSS seeds for each stage -- treating each seed
    as an independent replicate to be averaged, the same way overall
    accuracy already is elsewhere in this project's ablations, not
    pooling every seed's steps into one combined sample.

    Returns a pandas DataFrame: Configuration, Mean/Std Overall Accuracy
    (%), Mean/Std Early (0-20%) (%), Mean/Std Mid (20-60%) (%), Mean/Std
    Late (60-100%) (%).
    """
    rows = []
    for label, per_seed_results in results_by_seed.items():
        per_seed_tables = []
        for seed, r in per_seed_results.items():
            t = summarize_accuracy_by_stage({label: r}, boundaries=boundaries, early_late_cutoffs=early_late_cutoffs)
            per_seed_tables.append(t.iloc[0])
        stacked = pd.DataFrame(per_seed_tables)

        row = {"Configuration": label}
        for col in ["Overall Accuracy (%)", "Early (0-20%) (%)", "Mid (20-60%) (%)", "Late (60-100%) (%)"]:
            row[f"Mean {col}"] = round(stacked[col].mean(), 1)
            row[f"Std {col}"] = round(stacked[col].std(), 1)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_regime_comparison_with_variance(series_groups, target_col, work_dir, save_name="regime_comparison_variance.png",
                                          progression_labels=None, colors=None, ylabel="Validation Accuracy (%)",
                                          xlabel="Trajectory progression",
                                          title="Val Accuracy - Sub Region destination prediction"):
    """Like plot_regime_comparison, but for MULTI-SEED results — plots the
    MEAN across seeds as a solid line, with a shaded band showing +/- 1
    std dev, so the spread (how much to trust a given config's margin
    over another) is visible directly on the chart, not just implied.

    series_groups: {label: [progression_acc_array_seed1, progression_acc_array_seed2, ...]}
    — a LIST of arrays per label (one per seed), not a single array —
    this is the key difference from plot_regime_comparison.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    default_colors = plt.cm.tab10.colors
    labels = progression_labels or _progression_labels(DEFAULT_PROGRESSION_BOUNDARIES)
    x = range(len(labels))
    colors = colors or {}

    for i, (name, seed_arrays) in enumerate(series_groups.items()):
        stacked = np.array([[v * 100 if v <= 1.0 else v for v in arr] for arr in seed_arrays])  # [n_seeds, n_bands]
        mean_acc = stacked.mean(axis=0)
        std_acc = stacked.std(axis=0)
        color = colors.get(name, default_colors[i % len(default_colors)])
        ax.plot(x, mean_acc, marker="o", linewidth=2, color=color, label=f"{name} (n={len(seed_arrays)})")
        ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)

    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = os.path.join(work_dir, save_name)
    fig.savefig(save_path, dpi=150)
    print(f"Saved -> {save_path}")
    plt.show()
    return save_path

# ----------------------------------------------------------------------
# [notebook cell 127]
# ----------------------------------------------------------------------
# ===== E3-CONFIG v2 -- latest model, TEST-side evaluation =====
abl_arch = dict(alt_progression_modes=ALT_PROGRESSION_MODES, gate_ship_history=True,
                use_ship_history=True, use_departure_gate=USE_DEPARTURE_GATE, stratify=True,
                use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
                use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
                stratify_by_pair=False, val_frac=0.15,
                test_start=TEST_START, test_end=TEST_END,
                epochs=EPOCHS, early_stopping_patience=PATIENCE,
                batch_size=BATCH_SIZE, d_model=D_MODEL, work_dir=WORK_DIR, n_experts=3)

_bounds20 = tuple(sorted(DEFAULT_PROGRESSION_BOUNDARIES))
def _test_result(r, seed):
    """Score a run's model on the TEST window; return the band_correct/
    band_total/overall_acc dict the E3 helpers consume (20 bins)."""
    _, _, test_ids = _make_split(data, TARGET_COL, val_frac=0.15, seed=seed,
                                 stratify=True, test_start=TEST_START, test_end=TEST_END)
    tl = BucketedWAYDataset(data, target_col=TARGET_COL, batch_size=BATCH_SIZE,
                            seg_id_subset=list(test_ids), shuffle=False, seed=0,
                            include_ship_history=True)
    _s, _t, _p, _pr, _f = _collect_full_predictions(
        r["model"], r["repr_layer"], tl, r["core_and_alt_fn"],
        departure_ids_fn=r.get("departure_ids_fn"),
        eta_channel_lookup=r.get("eta_channel_lookup"))
    band = np.clip(np.digitize(_f, _bounds20), 0, len(_bounds20)-1)
    bc = np.array([( (_t==_p) & (band==b) ).sum() for b in range(len(_bounds20))], float)
    bt = np.array([(band==b).sum() for b in range(len(_bounds20))], float)
    return {"band_correct": bc.tolist(), "band_total": bt.tolist(),
            "overall_acc": float((_t==_p).mean()),
            "progression_acc": (bc/np.maximum(bt,1))}

E3 = {"prog": {}, "overall": {}, "full": {}}

def run_ablation(label, condition_stem=None, **overrides):
    E3["prog"][label], E3["overall"][label], E3["full"][label] = {}, {}, {}
    safe = (condition_stem or label.lower().replace(" ","_").replace("(","")
            .replace(")","").replace("+","plus"))
    for seed in SEEDS:
        condition = (f"final_main_lean2_seed{seed}" if label == "Full (final model)"
                     else f"e3_{safe}_final_main_seed{seed}")
        kwargs = dict(abl_arch); kwargs.update(overrides)
        if label != "Full (final model)":
            # fixed-budget protocol: every variant trains exactly the main
            # model's per-seed val-selected optimum, no early stopping
            assert "BEST_EPOCHS" in globals() and BEST_EPOCHS.get(seed), \
                "BEST_EPOCHS missing -- run the E0 A->B bridge cell first"
            kwargs.update(epochs=int(BEST_EPOCHS[seed]),
                          early_stopping_patience=None)
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=condition, seed=seed,
            skip_existing=True, **kwargs)
        tr = _test_result(r, seed)
        print(f"  {condition} (seed {seed}): TEST acc={tr['overall_acc']:.3f}")
        E3["prog"][label][seed] = tr["progression_acc"]
        E3["overall"][label][seed] = tr["overall_acc"]
        E3["full"][label][seed] = tr
# =============================================================================
# E3-TABLE DEF -- results summarizer (reads the E3 dicts)
# =============================================================================
def e3_table(labels, title):
    full_mean = np.mean(list(E3["overall"]["Full (final model)"].values()))
    t = pd.DataFrame([
        {"Configuration": lb,
         "Mean Overall Accuracy": np.mean(list(E3["overall"][lb].values())),
         "Std Overall Accuracy": np.std(list(E3["overall"][lb].values())),
         "Delta vs Full (mean)": np.mean(list(E3["overall"][lb].values())) - full_mean}
        for lb in labels]).sort_values("Delta vs Full (mean)",
                                       ascending=False).reset_index(drop=True)
    print(f"\n--- {title} ---"); print(t.to_string(index=False))
    stage = summarize_accuracy_by_stage_multiseed({lb: E3["full"][lb] for lb in labels})
    print(f"\n--- {title}: accuracy by voyage stage ---")
    print(stage.to_string(index=False))
    return t, stage

# ----------------------------------------------------------------------
# [notebook cell 128]
# ----------------------------------------------------------------------
# ===== E3-BASELINE MODEL  ===== Full (final model), reloaded
run_ablation("Full (final model)")     # reloads final_main_lean2_* + one test pass/seed
