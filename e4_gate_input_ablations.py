# =============================================================================
# E4 — gate-input ablations
# Migrated verbatim from Main_forGitHub.ipynb cells [145, 146, 148].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 145]
# ----------------------------------------------------------------------
# =============================================================================
# E3-A -- No ETA signal
# =============================================================================
run_ablation("No ETA signal", condition_stem="channel_ablation_no_eta_signal", alt_progression_modes=["historical_avg_port"])

# =============================================================================
# E3-A -- No historical-avg signal
# =============================================================================
run_ablation("No historical-avg signal", condition_stem="channel_ablation_no_historical_avg_signal", alt_progression_modes=["eta"])

# =============================================================================
# E3-A RESULTS -- gate-signal and history ablation (Part A)
# =============================================================================
tA, sA = e3_table(["Full (final model)", "No ship history",
                   "No ETA signal", "No historical-avg signal"],
                  "Gate-signal / history ablation (Part A)")
plot_regime_comparison_with_variance(
    {lb: list(E3["prog"][lb].values()) for lb in tA["Configuration"]},
    TARGET_COL, WORK_DIR, save_name="e3_gate_signal_ablation.png",
    title="E3 Part A -- gate signals and history, 3 seeds, mean \u00b1 std")
tA.to_csv(os.path.join(WORK_DIR, "e3_channel_ablation_partA.csv"), index=False)
print("\nSaved all three E3 tables to WORK_DIR")

# ----------------------------------------------------------------------
# [notebook cell 146]
# ----------------------------------------------------------------------
# =============================================================================
# E3-B RESULTS -- additive-channel comparison (Part B)
# =============================================================================
tB, sB = e3_table(["Full (final model)", "No ship history",
                   "+ Declared destination channel",
                   "+ Departure Subregion channel", "+ ETA channel"],
                  "Additive channels (Part B)")
plot_regime_comparison_with_variance(
    {lb: list(E3["prog"][lb].values()) for lb in tB["Configuration"]},
    TARGET_COL, WORK_DIR, save_name="e3_additive_channel_ablation.png",
    title="E3 Part B -- additive channels, 3 seeds, mean \u00b1 std")
tA.to_csv(os.path.join(WORK_DIR, "e3_channel_ablation_partA.csv"), index=False)
print("\nSaved all three E3 tables to WORK_DIR")

# ----------------------------------------------------------------------
# [notebook cell 148]
# ----------------------------------------------------------------------
# NEW Ablation 2: channel-by-channel ablation / NEW / include all channels NEW
# exec(open("ablation_2_channel_ablation.py").read())
