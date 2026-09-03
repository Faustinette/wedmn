# Section 4.1 — Train/Val/Test split
# Executed by runner.py inside the shared namespace (notebook-kernel style).

# CELL -- _make_split, SELF-CONTAINED (verbatim from the live Step4c_train.py)
# The project's shared split logic: explicit test window [test_start, test_end]
# carved out first (departures after test_end excluded entirely), then a
# stratified 15% validation draw from the remaining pool. Only numpy/pandas.

import numpy as np
import pandas as pd

def _make_split(step3data, target_col, val_frac=0.15, test_frac=0.15, seed=42, stratify=True,
                 stratify_by_pair=False, test_start=None, test_end=None):
        """Shared split logic used identically by train_one_target(),
    backfill_diagnostics(), and custom_progression_report(). All three must
    reconstruct exactly the same split from the same arguments
    (target_col, val_frac, test_frac, seed, stratify, stratify_by_pair);
    backfilling diagnostics without retraining relies on this property.

    TEST: the most recent test_frac of segments, ordered chronologically by
    departure timestamp (dep_ts). This forms a forward-in-time holdout,
    which measures generalization across time rather than to a random
    subset of the data, and is never affected by train/val decisions.

    test_start / test_end (optional, both or neither): an explicit date
    window for TEST instead of the most recent fraction. When given, TEST
    contains every segment with dep_ts in [test_start, test_end] inclusive,
    and TRAIN/VAL are built only from segments strictly before test_start.
    This yields a point-in-time backtest: only data available at test_start
    is used for training. Segments after test_end are excluded from the
    split entirely (not train, not val, not test), since using them for
    training would introduce information unavailable at that point in
    time. test_frac is ignored when the window is set.

    TRAIN/VAL: a stratified split of the remaining (earlier) segments.
    Full temporal and class stratification cannot be combined, because the
    set of classes present depends on the period sampled; stratification
    is therefore applied only within the temporally separated train/val
    pool, where it addresses the practical problem of rare classes being
    underrepresented in validation. Classes with fewer than 2 examples in
    this pool cannot be split and are kept entirely in train (reported in
    the printed diagnostic).

    stratify_by_pair=False (default): groups by target_col alone, i.e. by
    the class being predicted, following standard stratified-split
    practice of balancing the prediction target across train and val.

    stratify_by_pair=True: groups by the pair (departure subregion,
    target_col), a strictly finer partition, so that each trade lane
    receives proportional train/val representation rather than only its
    arrival side. The departure subregion is derived from DEP_PORT_ID via
    the same build_port_to_subregion_map lookup used for the
    departure-subregion channel, keeping the mapping consistent across
    the project. Segments whose departure port does not resolve to a
    subregion are excluded from the val-eligible pool and kept in train,
    reported in the same diagnostic as singleton classes.
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

# Split the Dataset into Train, Val and Test Set
TEST_START = "2025-12-01"
TEST_END   = "2026-03-01"

train_ids, val_ids, test_ids = _make_split(
    data, "ARR_SUBREGION_ID", val_frac=0.15, test_frac=0.15, seed=42, stratify=True,
    test_start=TEST_START, test_end=TEST_END)

print(f"window [{TEST_START} -> {TEST_END}]:")
print(f"  train {len(train_ids):,}   val {len(val_ids):,}   test {len(test_ids):,}")
