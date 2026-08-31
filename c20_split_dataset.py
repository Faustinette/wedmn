# =============================================================================
# Section 4.1 — Train/Val/Test split
# Migrated verbatim from Main_forGitHub.ipynb cells [41, 42].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 41]
# ----------------------------------------------------------------------
# =============================================================================
# CELL -- _make_split, SELF-CONTAINED (verbatim from the live Step4c_train.py)
# The project's shared split logic: explicit test window [test_start, test_end]
# carved out first (departures after test_end excluded entirely), then a
# stratified 15% validation draw from the remaining pool. Only numpy/pandas.
# =============================================================================
import numpy as np
import pandas as pd

def _make_split(step3data, target_col, val_frac=0.15, test_frac=0.15, seed=42, stratify=True,
                 stratify_by_pair=False, test_start=None, test_end=None):
    """Shared split logic, used identically by train_one_target(),
    backfill_diagnostics(), and custom_progression_report() — critical that
    all three reconstruct EXACTLY the same split given the same
    (target_col, val_frac, test_frac, seed, stratify, stratify_by_pair),
    the same property already relied on for backfilling diagnostics
    without retraining.

    TEST = the most recent test_frac of segments, chronologically (by
    dep_ts) — a genuine forward-in-time holdout, directly targeting the
    concern that a random split can't tell you whether the model
    generalizes across a real regime change (e.g. this fleet's documented
    Iran-conflict-era disruption) rather than just to a random subset of
    all-time data. Never touched by TRAIN/VAL split decisions.

    test_start / test_end (both optional, default None -- both or
    neither): an EXPLICIT date window for TEST instead of "most recent
    test_frac%". When given, TEST = every segment with dep_ts in
    [test_start, test_end] (inclusive), and TRAIN/VAL are built ONLY
    from segments STRICTLY BEFORE test_start — a genuine, point-in-time
    backtest: only data that would actually have been available at the
    time gets used for training, matching real deployment discipline
    rather than a look-ahead-tainted split. Any segment AFTER test_end
    is excluded from this split entirely — not train, not val, not
    test — since including it in train/val would let the model learn
    from data a real, live deployment at that point in time wouldn't
    yet have had. test_frac is ignored when these are set (the window
    itself decides how much data becomes test).

    TRAIN/VAL = a stratified split of the REMAINING (earlier) segments —
    full temporal + class stratification together isn't possible (which
    classes exist at all depends on which period you're drawing from),
    so stratification is applied only within the already-temporally-
    separated train/val pool, where it's actually achievable and where
    it fixes the real problem (rare classes poorly represented in val,
    adding noise to every reported comparison). Classes with <2 examples
    in this pool can't be split at all and are kept entirely in train
    (flagged in the printed diagnostic).

    stratify_by_pair=False (default, unchanged behavior): groups by
    target_col ALONE -- i.e. by ARR_SUBREGION_ID when that's the target,
    the arrival subregion being predicted. This is NOT "by load
    subregion" (confirmed directly, not assumed -- there was never a
    departure-subregion grouping here at all); it's stratified by the
    prediction TARGET class itself, standard stratified-split practice
    (balance the thing being predicted across train/val).

    stratify_by_pair=True: groups by the PAIR (departure subregion,
    target_col) instead -- a strictly finer partition (each existing
    target_col group splits further by where the voyage started), so
    each trade lane (e.g. "USGC -> NEAsia_China" specifically, not just
    "-> NEAsia_China" from anywhere) gets its own proportional train/val
    representation, not just its arrival side. Departure subregion is
    derived via the SAME build_port_to_subregion_map(...)(DEP_PORT_ID)
    lookup already used elsewhere in this project for the departure
    gate/departure-subregion channel — not a new, separately-computed
    mapping. Segments whose departure port never resolves to a subregion
    are dropped from the val-eligible pool the same way a <2-example
    class would be (kept in train only, flagged).
    """
    valid = step3data.traj_idx.dropna(subset=[target_col]).copy()
    if "dep_ts" not in valid.columns:
        raise ValueError("traj_idx has no 'dep_ts' column — needed for the temporal test split.")
    valid["dep_ts"] = pd.to_datetime(valid["dep_ts"])
    valid = valid.sort_values(["dep_ts", "seg_id"]).reset_index(drop=True)

    if (test_start is None) != (test_end is None):
        raise ValueError("test_start and test_end must be given together (both or neither) -- "
                          "an explicit test window needs both boundaries specified.")

    if test_start is not None:
        test_start_ts, test_end_ts = pd.Timestamp(test_start), pd.Timestamp(test_end)
        if test_end_ts < test_start_ts:
            raise ValueError(f"test_end ({test_end_ts}) is before test_start ({test_start_ts})")
        test_mask = (valid["dep_ts"] >= test_start_ts) & (valid["dep_ts"] <= test_end_ts)
        test_ids = set(valid.loc[test_mask, "seg_id"])
        # ONLY segments strictly before test_start -- a genuine,
        # point-in-time backtest. Anything after test_end is excluded
        # from this split entirely (not train, not val, not test), not
        # folded into train/val -- including it there would let the
        # model learn from data a real, live deployment at test_start
        # wouldn't yet have had, undermining the whole point of an
        # explicit temporal holdout.
        before_mask = valid["dep_ts"] < test_start_ts
        remaining = valid.loc[before_mask].reset_index(drop=True)
        n_after = int((valid["dep_ts"] > test_end_ts).sum())
        print(f"    [explicit test window] [{test_start_ts.date()} -> {test_end_ts.date()}]: "
              f"{len(test_ids):,} test segments, {len(remaining):,} remaining for train/val "
              f"(strictly BEFORE {test_start_ts.date()} only)")
        if n_after:
            print(f"    {n_after:,} segment(s) depart AFTER {test_end_ts.date()} -- excluded from "
                  f"this split entirely (not train, not val, not test), not folded into train/val.")
        if len(test_ids) == 0:
            print(f"    WARNING: 0 segments fall in this test window -- check it against this "
                  f"dataset's own date coverage.")
    else:
        n = len(valid)
        n_test = max(1, int(round(n * test_frac)))
        test_ids = set(valid["seg_id"].iloc[-n_test:].tolist())
        remaining = valid.iloc[:-n_test].reset_index(drop=True)

    if stratify_by_pair and not stratify:
        raise ValueError("stratify_by_pair=True requires stratify=True (it's a finer version of the "
                          "same stratified split, not an independent option)")

    if stratify_by_pair:
        port_to_subregion = build_port_to_subregion_map(step3data.traj_idx, subregion_col=target_col, port_col="ARR_PORT_ID")
        remaining = remaining.copy()
        remaining["_dep_subregion"] = remaining["DEP_PORT_ID"].map(port_to_subregion)
        n_unresolved = remaining["_dep_subregion"].isna().sum()
        group_cols = ["_dep_subregion", target_col]
    elif stratify:
        group_cols = [target_col]

    if stratify:
        rng = np.random.default_rng(seed)
        val_ids_list = []
        singleton_classes = 0
        groupable = remaining.dropna(subset=group_cols) if stratify_by_pair else remaining
        for _, grp in groupable.groupby(group_cols):
            grp_ids = grp["seg_id"].values
            if len(grp_ids) < 2:
                singleton_classes += 1
                continue  # can't split a single example; stays in train only
            n_val_cls = max(1, int(round(len(grp_ids) * val_frac)))
            n_val_cls = min(n_val_cls, len(grp_ids) - 1)  # always leave >=1 for train
            perm = rng.permutation(len(grp_ids))
            val_ids_list.extend(grp_ids[perm[:n_val_cls]].tolist())
        val_ids = set(val_ids_list)
        train_ids = set(remaining["seg_id"].tolist()) - val_ids
        if singleton_classes:
            group_desc = "(departure subregion, target) pair(s)" if stratify_by_pair else "class(es)"
            print(f"    [stratified split] {singleton_classes} {group_desc} had <2 examples in the "
                  f"train/val pool -- kept entirely in train (can't be split).")
        if stratify_by_pair and n_unresolved:
            print(f"    [stratified split] {n_unresolved} segment(s) had an unresolvable departure "
                  f"subregion -- kept entirely in train (can't be pair-stratified).")
    else:
        rng = np.random.default_rng(seed)
        seg_ids = remaining["seg_id"].values
        perm = rng.permutation(len(seg_ids))
        n_val = max(1, int(len(seg_ids) * val_frac))
        val_ids = set(seg_ids[perm[:n_val]].tolist())
        train_ids = set(seg_ids[perm[n_val:]].tolist())

    return train_ids, val_ids, test_ids

# ----------------------------------------------------------------------
# [notebook cell 42]
# ----------------------------------------------------------------------
# Split the Dataset into Train, Val and Test Set
TEST_START = "2025-12-01"
TEST_END   = "2026-03-01"

train_ids, val_ids, test_ids = _make_split(
    data, "ARR_SUBREGION_ID", val_frac=0.15, test_frac=0.15, seed=42, stratify=True,
    test_start=TEST_START, test_end=TEST_END)

print(f"window [{TEST_START} -> {TEST_END}]:")
print(f"  train {len(train_ids):,}   val {len(val_ids):,}   test {len(test_ids):,}")
