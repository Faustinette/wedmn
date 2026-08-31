"""
runner.py — executes the migrated notebook cells inside ONE shared namespace.

The original notebook relies on kernel-wide globals crossing cell boundaries
(`data`, `BEST_EPOCHS`, `fold_ids`, `E3`, `_test_result`, ...) and on guard
patterns like `if "SEEDS" not in dir()`. To keep every cell byte-for-byte
verbatim (important for the report), we reproduce the kernel: every stage file
is exec'd into the same CTX dict, exactly like running cells top-to-bottom.

Each stage runs at most once per process (like a notebook cell you ran once);
`ensure()` resolves prerequisites recursively.
"""

import os

# The single shared "kernel" namespace.
CTX: dict = {"__name__": "__main__", "__builtins__": __builtins__}

_DONE: set = set()

# Repo root = directory containing this file. Stage paths are relative to it.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _configure_workdir() -> str:
    """WORK_DIR = where data, Results/ and outputs live.

    Priority: $WORK_DIR env var, else current working directory.
    The original notebook hard-coded a Google Drive path (Colab); here it is
    injected once into the shared namespace before any cell runs, and we chdir
    into it so the `exec(open("Step1a_...py"))`-style pipeline scripts and all
    relative reads/writes behave exactly as in the notebook.
    """
    work_dir = os.environ.get("WORK_DIR", os.getcwd())
    work_dir = os.path.abspath(work_dir)
    CTX["WORK_DIR"] = work_dir
    os.makedirs(os.path.join(work_dir, "Results"), exist_ok=True)
    os.chdir(work_dir)
    print(f"[runner] WORK_DIR = {work_dir}")
    return work_dir


def run_file(rel_path: str) -> None:
    """Execute one stage file inside the shared namespace (verbatim cells)."""
    path = os.path.join(REPO_ROOT, rel_path)
    print(f"\n[runner] ── running {rel_path} " + "─" * max(1, 60 - len(rel_path)))
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, path, "exec"), CTX)


# ───────────────────────────── stage registry ────────────────────────────────
# A stage = (list of prerequisite stages, list of files to exec, in order).
STAGES: dict = {
    # -- data + model + training library (== notebook sections 0, 3.b, 4.1-4.6, 5.1)
    "core": (
        [],
        [
            "core/c01_imports.py",
            "core/c02_input_checks.py",
            "core/c10_representation_layers.py",
            "core/c11_arrival_labels.py",
            "core/c12_subregion_map.py",
            "core/c13_ship_history_index.py",
            "core/c14_build_channels.py",
            "core/c20_split_dataset.py",
            "core/c21_input_channels.py",
            "core/c22_model_architecture.py",
            "core/c23_bucketed_loader.py",
            "core/c24_ship_history_gnn.py",
            "core/c25_departure_duration_index.py",
            "core/c30_training_library.py",
        ],
    ),
    # -- training config (guard pattern; safe to run before any experiment)
    "train_config": (["core"], ["core/c31_train_config.py"]),
    # -- main 3-seed training. skip_existing=True → finished runs reload from
    #    Results/ checkpoints, so this is cheap once the model is trained.
    "trained": (["train_config"], ["experiments/e0a_main_training.py"]),
    # -- E3-PREREQ + E3-CONFIG v2 + baseline reload (needed by E3/E4/E5/E8/H8)
    "e3_prereq": (["trained"], ["experiments/e3_prereq.py"]),
    # -- pooled per-step predictions with probabilities (prereq for 5B / E15)
    "pooled_predictions": (["trained"], ["experiments/pooled_predictions.py"]),
    # -- k-fold CV pool (defines fold_ids / CV_SEED; prereq for E6-CV, E3 part A)
    "cv": (["trained"], ["experiments/e0b_cross_validation.py"]),
}


def ensure(*stage_names: str) -> None:
    for name in stage_names:
        if name in _DONE:
            continue
        prereqs, files = STAGES[name]
        ensure(*prereqs)
        if name in _DONE:  # may have been satisfied while recursing
            continue
        print(f"\n[runner] ══ stage: {name} ══")
        for f in files:
            run_file(f)
        _DONE.add(name)


def run_experiment(rel_path: str, *prereq_stages: str) -> None:
    """Resolve prerequisites, then run the experiment file itself."""
    _configure_workdir() if "WORK_DIR" not in CTX else None
    ensure(*prereq_stages)
    run_file(rel_path)


# Configure the working directory as soon as the runner is imported.
_configure_workdir()
