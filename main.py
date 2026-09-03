"""
main.py — single entry point for every experiment in the report.

Usage:
    python main.py --experiment E5
    WORK_DIR=/path/to/data python main.py --experiment H8

key steps - in order :
══════════════════════
You never execute the files in core/ or experiments/ directly. Every
`--experiment` resolves its prerequisites automatically (see runner.STAGES)
and runs the required steps IN THIS FIXED ORDER, each exactly once:

  STEP 1 — "core" stage: data + model + training library
     core/c01_imports.py                  base imports (sets KERAS_BACKEND=torch)
     core/c02_input_checks.py             verifies the 4 input files exist
     core/c10_representation_layers.py    ┐
     core/c11_arrival_labels.py           │ build the dataset from the
     core/c12_subregion_map.py            │ Step-3 input files
     core/c13_ship_history_index.py       │ (notebook section 3.b)
     core/c14_build_channels.py           ┘
     core/c20_split_dataset.py            train/val/test split      (sec 4.1)
     core/c21_input_channels.py           input channels            (sec 4.2)
     core/c22_model_architecture.py       model architecture        (sec 4.3)
     core/c23_bucketed_loader.py          batch loader              (sec 4.4)
     core/c24_ship_history_gnn.py         ship-history GNN          (sec 4.5)
     core/c25_departure_duration_index.py duration index            (sec 4.6)
     core/c30_training_library.py         training functions        (sec 5.1)

  STEP 2 — "train_config" stage:
     core/c31_train_config.py             hyperparameters + sanity tripwires

  STEP 3 — "trained" stage:
     experiments/e0a_main_training.py     3-seed training. If checkpoints are
                                          present in Results/, they are
                                          RELOADED (skip_existing=True) —
                                          nothing is retrained.

  STEP 4 (only for experiments that need it) —
     "e3_prereq"           experiments/e3_prereq.py           (E3/E4/E5/E6/E8/H8)
     "pooled_predictions"  experiments/pooled_predictions.py  (STATS/E15)
     "cv"                  experiments/e0b_cross_validation.py (E3/E6CV)

  STEP 5 — experimentS files - for instance :
     experiments/e5_mixture_vs_shared_ff.py

The console prints a banner for every stage and file as it runs, so the
sequence above is visible live. Steps 1-4 run at most once per invocation
even if several are shared. Each run_* function below declares which STEP-4
prerequisites its experiment needs.
══════════════════════════════════

The repo starts from Step 3b: it consumes the four preprocessed model-input
files (see README) and never touches raw AIS data or the Step 1-3a scripts.
Training stages use the notebook's skip_existing=True guard, so once the
3-seed model exists under $WORK_DIR/Results/ they reload instead of retrain.
"""

import argparse

from runner import run_experiment


# ── E0: training and headline evaluation ─────────────────────────────────────
def run_e0a():
    """Main 3-seed early-stopped training + BEST_EPOCHS bridge + val bins."""
    run_experiment("experiments/e0a_main_training.py", "train_config")


def run_e0b():
    """k-fold cross-validation on the train+val pool."""
    run_experiment("experiments/e0b_cross_validation.py", "trained")


def run_e0c():
    """Test-set evaluation (trained on train+val block) + test bins."""
    run_experiment("experiments/e0c_test_evaluation.py", "trained")


# ── E1: benchmark vs captain-declared destination ────────────────────────────
def run_e1():
    run_experiment("experiments/e1_benchmark_captain.py", "trained")


def run_e1b():
    run_experiment("experiments/e1b_benchmark_breakdown.py", "trained")


def run_e1c():
    run_experiment("experiments/e1c_performance_investigation.py", "trained")


# ── Step 5B/5C: statistics and parameter estimates ───────────────────────────
def run_stats():
    """Clustered bootstrap, McNemar, calibration/ECE, lock-in, entropy,
    PCA probes, variance decomposition, gate sign tests."""
    run_experiment("experiments/s5b_statistics.py", "pooled_predictions")


def run_params():
    run_experiment("experiments/s5c_parameter_estimates.py", "trained")


# ── Step 6: report visualisations ────────────────────────────────────────────
def run_viz():
    run_experiment("experiments/s6_visualisations.py", "trained")


# ── Step 7: ablations ────────────────────────────────────────────────────────
def run_e3():
    """Channel ablations (test-side + part A CV) + E19 interactions."""
    run_experiment("experiments/e3_channel_ablations.py", "e3_prereq", "cv")


def run_e4():
    """Gate-input ablations."""
    run_experiment("experiments/e4_gate_input_ablations.py", "e3_prereq")


def run_h8():
    """Ship-history contribution (H1 damage-by-depth, H8 by slice)."""
    run_experiment("experiments/h8_ship_history.py", "e3_prereq")


def run_e5():
    """Mixture vs shared feed-forward."""
    run_experiment("experiments/e5_mixture_vs_shared_ff.py", "e3_prereq")


def run_e6cv():
    """Expert count, cross-validation."""
    run_experiment("experiments/e6_expert_count_cv.py", "cv")


def run_e6():
    """Expert count on the TEST set (+ E5/E6 paired tests)."""
    run_experiment("experiments/e6_expert_count_test.py", "e3_prereq")


def run_e7():
    """Gate behaviour over the voyage."""
    run_experiment("experiments/e7_gate_behaviour.py", "trained")


def run_e7b():
    """Mixture-expert drivers."""
    run_experiment("experiments/e7_mixture_drivers.py", "trained")


def run_e8():
    """Cold start (tier-2 pooled)."""
    run_experiment("experiments/e8_cold_start.py", "e3_prereq")


def run_e9():
    """Non-stationarity / rigorous out-of-sample event test."""
    run_experiment("experiments/e9_structural_break.py", "trained")


def run_e10():
    """Fleet effect (exp1b final control)."""
    run_experiment("experiments/e10_fleet_effect.py", "trained")


def run_e11():
    """Training regime and regularization."""
    run_experiment("experiments/e11_regularization.py", "train_config")


# ── Step 8: error analysis ───────────────────────────────────────────────────
def run_e15():
    """Error structure: G-test, taxonomy, ambiguity, empirical ceiling."""
    run_experiment("experiments/e15_error_structure.py", "pooled_predictions")


def run_e18():
    """Error slice & dice: lanes, discharge regions, margins, taxonomy v2."""
    run_experiment("experiments/e18_error_slicing.py", "trained")


def run_e34():
    """Cartopy trajectory maps (requires cartopy)."""
    run_experiment("experiments/e34_trajectory_maps.py", "core")


EXPERIMENTS = {
    "E0A": run_e0a,
    "E0B": run_e0b,
    "E0C": run_e0c,
    "E1": run_e1,
    "E1B": run_e1b,
    "E1C": run_e1c,
    "STATS": run_stats,
    "PARAMS": run_params,
    "VIZ": run_viz,
    "E3": run_e3,
    "E4": run_e4,
    "H8": run_h8,
    "E5": run_e5,
    "E6CV": run_e6cv,
    "E6": run_e6,
    "E7": run_e7,
    "E7B": run_e7b,
    "E8": run_e8,
    "E9": run_e9,
    "E10": run_e10,
    "E11": run_e11,
    "E15": run_e15,
    "E18": run_e18,
    "E34": run_e34,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="WEDMN — LPG vessel destination model: run one experiment.")
    p.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    args = p.parse_args()
    EXPERIMENTS[args.experiment]()
