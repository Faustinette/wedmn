# =============================================================================
# Section 4.4 — Bucketed segment loader (Step4b)
# Migrated verbatim from Main_forGitHub.ipynb cells [48].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 48]
# ----------------------------------------------------------------------

# CHANGE THE NAME BucketedWAYDataset

# =============================================================================
# LIB CELL L3 -- Step4b_bucketed_loader (verbatim; 1 import stripped)
# =============================================================================
"""
Step4b_bucketed_loader.py
─────────────────────────────────────────────────────────────────────────────
STEP 4b — BUCKETED DATA LOADER

Motivation (from Step 3b's own diagnostic): a random sample of 4 segments
showed N ranging 5-103 within one batch — padding the short ones out to the
batch max wastes up to 20x compute. This loader groups segments by SIMILAR
length before batching (standard "sequence bucketing"), while still
shuffling within buckets and across epochs so training isn't a fixed
shortest-to-longest curriculum.

Design:
  1. Sort all segments by N (real step count) ascending.
  2. Chunk into buckets of size batch_size * bucket_mult.
  3. Shuffle WITHIN each bucket, then slice into actual batches of
     batch_size — batches drawn from the same bucket have similar N, so
     padding waste stays low, but which segments land in which exact batch
     still varies run to run.
  4. Shuffle the order batches are served in.
  5. Rebuild (reshuffle) at the start of every epoch via on_epoch_end().

TARGET LABEL: configurable (`target_col`) — DEST_PORT_ID, DEST_REGION_ID,
or DEST_SUBREGION_ID are all available from Step 3a's enrichment. Per
earlier discussion, region/subregion are the intended PRIMARY targets for
this fleet (port-level ground truth is noisier and this fleet's per-vessel
repeat-destination rate is low, per Step 3a's causal history diagnostic) —
defaults to DEST_REGION_ID, override for other granularities or for the
port-level task.

Segments with a missing/unmapped target label are dropped (a warning is
printed with the count) — can't train against a NaN label.
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras



class BucketedWAYDataset(keras.utils.PyDataset):
    """Yields (inputs, key_padding_mask, labels, lengths) per batch —
    designed for a custom training loop (not model.fit()), since GD
    reweighting (Step4a's gradient_dropout_weights) needs per-batch instance
    lengths, which model.fit()'s standard (x,y) / (x,y,sample_weight)
    interface doesn't naturally expose the way this needs."""

    def __init__(self, step3data: Step3Data, target_col="DEST_REGION_ID",
                 batch_size=32, bucket_mult=8, shuffle=True, seed=0,
                 seg_id_subset=None, max_mk_per_batch=50,
                 include_ship_history=False, history_max_history=None,
                 use_contract_period_feature=False,
                 include_fleet_context=False, normalize_fleet=False,
                 include_candidate_fleet_context=False,
                 include_fixed_horizon_fleet_context=False,
                 include_active_vessel_set_context=False,
                 weight_fixed_horizon_by_similarity=False,
                 fixed_horizon_size_sigma=None, fixed_horizon_draught_sigma=None,
                 fixed_horizon_weight_combination="multiplicative",
                 active_vessel_truncation_mode="nearest",
                 active_vessel_eta_channel_lookup=None, active_vessel_history_index=None,
                 active_vessel_port_to_subregion=None, active_vessel_history_stats_cache=None, **kwargs):
        super().__init__(**kwargs)
        self.data = step3data
        self.target_col = target_col
        self.batch_size = batch_size
        self.bucket_mult = bucket_mult
        self.shuffle = shuffle
        self.include_ship_history = include_ship_history
        self.history_max_history = history_max_history
        self.use_contract_period_feature = use_contract_period_feature
        self.include_fleet_context = include_fleet_context
        self.normalize_fleet = normalize_fleet
        self.include_candidate_fleet_context = include_candidate_fleet_context
        self.include_fixed_horizon_fleet_context = include_fixed_horizon_fleet_context
        self.include_active_vessel_set_context = include_active_vessel_set_context
        self.weight_fixed_horizon_by_similarity = weight_fixed_horizon_by_similarity
        self.fixed_horizon_size_sigma = fixed_horizon_size_sigma
        self.fixed_horizon_draught_sigma = fixed_horizon_draught_sigma
        self.fixed_horizon_weight_combination = fixed_horizon_weight_combination
        self.active_vessel_truncation_mode = active_vessel_truncation_mode
        # active_vessel_history_stats_cache: NOT set here to a fresh {}
        # by default the way prepare_active_vessel_set_batch's own
        # default does -- this loader is reused across MANY batches
        # (potentially a full epoch), so defaulting to None here and
        # letting the CALLER pass a persistent dict (or leaving it None
        # to allocate fresh per __getitem__ call, no cross-batch reuse)
        # is an explicit choice, not an oversight -- silently caching
        # across an entire epoch by default could mask a caller's own
        # intent to isolate calls.
        self.active_vessel_eta_channel_lookup = active_vessel_eta_channel_lookup
        self.active_vessel_history_index = active_vessel_history_index
        self.active_vessel_port_to_subregion = active_vessel_port_to_subregion
        self.active_vessel_history_stats_cache = active_vessel_history_stats_cache
        self.max_mk_per_batch = max_mk_per_batch
        self.rng = np.random.default_rng(seed)

        idx = self.data.traj_idx.copy()
        if seg_id_subset is not None:
            idx = idx[idx["seg_id"].isin(seg_id_subset)]
        n_total = len(idx)
        idx = idx.dropna(subset=[target_col])
        idx = idx[idx["seg_id"].isin(self.data.n_per_seg.index)]
        n_dropped = n_total - len(idx)
        if n_dropped:
            print(f"    [BucketedWAYDataset] dropped {n_dropped:,} / {n_total:,} segments "
                  f"with missing '{target_col}' ({n_dropped/n_total*100:.1f}%)")

        self.seg_ids = idx["seg_id"].values
        self.labels = idx.set_index("seg_id")[target_col].astype(int).to_dict()
        self.lengths = self.data.n_per_seg.to_dict()

        self._build_batches()

    def _build_batches(self):
        seg_ids = np.array(self.seg_ids)
        lengths = np.array([self.lengths[s] for s in seg_ids])
        order = np.argsort(lengths)  # ascending
        sorted_seg_ids = seg_ids[order]

        bucket_size = self.batch_size * self.bucket_mult
        buckets = [sorted_seg_ids[i:i + bucket_size]
                   for i in range(0, len(sorted_seg_ids), bucket_size)]

        batches = []
        for bucket in buckets:
            b = bucket.copy()
            if self.shuffle:
                self.rng.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                batches.append(b[i:i + self.batch_size])

        if self.shuffle:
            self.rng.shuffle(batches)  # shuffle BATCH ORDER, not batch contents

        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        seg_ids = self.batches[idx]
        inputs, n_mask = self.data.prepare_batch(
            list(seg_ids), max_mk_per_batch=self.max_mk_per_batch,
            include_ship_history=self.include_ship_history,
            history_max_history=self.history_max_history,
            use_contract_period_feature=self.use_contract_period_feature,
            include_fleet_context=self.include_fleet_context,
            normalize_fleet=self.normalize_fleet,
            include_candidate_fleet_context=self.include_candidate_fleet_context,
            include_fixed_horizon_fleet_context=self.include_fixed_horizon_fleet_context,
            include_active_vessel_set_context=self.include_active_vessel_set_context,
            weight_fixed_horizon_by_similarity=self.weight_fixed_horizon_by_similarity,
            fixed_horizon_size_sigma=self.fixed_horizon_size_sigma,
            fixed_horizon_draught_sigma=self.fixed_horizon_draught_sigma,
            fixed_horizon_weight_combination=self.fixed_horizon_weight_combination,
            active_vessel_truncation_mode=self.active_vessel_truncation_mode,
            active_vessel_eta_channel_lookup=self.active_vessel_eta_channel_lookup,
            active_vessel_history_index=self.active_vessel_history_index,
            active_vessel_port_to_subregion=self.active_vessel_port_to_subregion,
            active_vessel_history_stats_cache=self.active_vessel_history_stats_cache)
        labels = np.array([self.labels[s] for s in seg_ids], dtype="int32")
        lengths = np.array([self.lengths[s] for s in seg_ids], dtype="float32")
        return inputs, n_mask, labels, lengths

    def on_epoch_end(self):
        self._build_batches()

    def padding_stats(self):
        """Diagnostic: average and worst-case padding waste ratio (max_N in
        batch / mean real N in batch) across all batches for this epoch's
        batch assignment — call after __init__ or on_epoch_end()."""
        ratios = []
        for batch in self.batches:
            lens = np.array([self.lengths[s] for s in batch])
            ratios.append(lens.max() / lens.mean())
        ratios = np.array(ratios)
        return {"mean_waste_ratio": float(ratios.mean()),
                "max_waste_ratio": float(ratios.max()),
                "n_batches": len(self.batches)}
