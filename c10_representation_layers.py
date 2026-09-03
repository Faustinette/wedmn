# Section 3.b.ii — Representation layers
# Executed by runner.py inside the shared namespace (notebook-kernel style).

import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # must precede any keras import this session
# Create Data transformation class


# CELL 3-INLINE -- Step3Data, FULLY SELF-CONTAINED (no project-file imports)
# Replaces:
#     import Step3b_representation_layer
#     data = Step3b_representation_layer.Step3Data(WORK_DIR)
# The class represents the Step3b_representation_layer.py's in the original notebook 


import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from keras import ops

DATA_SUBFOLDER = ""


class Step3Data:
    """Loads Step 3a's outputs, builds local-pattern per-point features, and
    prepares padded batches — reusable from Step 4's data loader (imports
    this class rather than the whole script re-running on import).
    """

    def __init__(self, work_dir):
        work_dir = Path(work_dir) / DATA_SUBFOLDER
        gridded = pd.read_parquet(work_dir / "trajectories_gridded.parquet")
        self.steps_idx = pd.read_parquet(work_dir / "segment_steps_index.parquet")
        # Defensive sort — prepare_batch() relies on each segment's rows
        # being pre-sorted by STEP_IDX to avoid re-sorting per batch (a
        # measured hot-path cost). Step3a saves it sorted already, but a
        # later merge (declared-destination columns) isn't 100% guaranteed
        # to preserve row order across all pandas versions, so don't rely
        # on it silently — sort explicitly, once, here.
        self.steps_idx = self.steps_idx.sort_values(["SEG_ID", "STEP_IDX"]).reset_index(drop=True)
        self.traj_idx = pd.read_csv(work_dir / "trajectories_index_enriched.csv")
        with open(work_dir / "step3_vocabularies.json") as f:
            self.vocab = json.load(f)

        if not pd.api.types.is_datetime64_any_dtype(gridded["TIMESTAMP"]):
            gridded["TIMESTAMP"] = pd.to_datetime(gridded["TIMESTAMP"], errors="coerce")

        self.n_ports = len(self.vocab["port_to_id"])
        self.none_declared_id = self.vocab.get("declared_dest_none_id", self.n_ports)
        self.n_size_classes = len(self.vocab["size_class_to_id"])

        self.has_declared = "DECLARED_DEST_PORT_ID" in gridded.columns
        self.g = self._build_local_features(gridded)

        # N (real step count) per segment — needed by Step4's bucketed loader.
        self.n_per_seg = self.steps_idx.groupby("SEG_ID").size()

        # PERFORMANCE: pre-group once here rather than re-filtering the full
        # table on every prepare_batch() call. Doing `self.g[self.g["SEG_ID"]
        # == sid]` per segment per batch is an O(total_rows) scan each time —
        # with a 1.3M-row table, batch_size=32, and hundreds of batches per
        # epoch, that adds up to tens of thousands of full-table scans per
        # epoch and dominates wall-clock time far more than model compute
        # does. Grouping once up front makes each segment's lookup an O(1)
        # dict access instead.
        self._g_by_seg = dict(tuple(self.g.groupby("SEG_ID")))
        self._steps_by_seg = dict(tuple(self.steps_idx.groupby("SEG_ID")))
        self._empty_g = self.g.iloc[0:0]
        self._empty_steps = self.steps_idx.iloc[0:0]

        # Ship-history index (Model Block 2) is NOT built here — it needs
        # ARR_PORT_ID, which only exists after enrich_arrival_labels(data)
        # has run (either persisted from an updated Step3a, or derived
        # in-memory as a fallback). Call build_ship_history_index()
        # explicitly once that's done, before using use_ship_history=True.
        self.history_index = None
        self.fleet_heading_index = None
        self.candidate_fleet_state_index = None
        self.fixed_horizon_fleet_index = None
        self.fixed_horizon_days = None
        self.similarity_weighted_fleet_index = None
        self.active_vessel_set_index = None
        self.max_active_vessels = None
        self._seg_to_dep = None

    def build_ship_history_index(self):
        """Call once, after enrich_arrival_labels(data) — builds the causal
        per-vessel voyage-history index (Model Block 2) used when
        prepare_batch(..., include_ship_history=True) or
        RepresentationLayer(..., use_ship_history=True) are used."""
        from Step4d_ship_history import VesselHistoryIndex
        self.history_index = VesselHistoryIndex(self.traj_idx)
        return self.history_index

    def build_fleet_heading_index(self, subregion_col="ARR_SUBREGION_ID", arrival_window_days=None,
                                   anchor="arrival", duration_index=None, window_fraction=None,
                                   min_window_days=3, max_window_days=30):
        """Call once, after enrich_arrival_labels(data) — builds the
        leave-one-out fleet-heading index (Model Block 3, oracle signal)
        used when prepare_batch(..., include_fleet_context=True) or
        RepresentationLayer(..., use_fleet_context=True) are used.

        subregion_col: which column to aggregate by — defaults to
        ARR_SUBREGION_ID, but any granularity works (e.g. ARR_COUNTRY_ID
        to match the fleet signal's resolution to the forecast target).
        The vocab used to SIZE the index is looked up to match whichever
        column is actually passed — using the subregion vocab size for a
        non-subregion column would under-size the array and crash with an
        index-out-of-bounds the moment a class ID exceeds that size.

        anchor: "arrival" (default) counts vessels shortly BEFORE they
        arrive — a late-voyage-relevant signal. "departure" counts vessels
        shortly AFTER they depart — the early-voyage-relevant mirror.

        duration_index + window_fraction: optional, both required together
        — DURATION-CONDITIONED WINDOW (see FleetHeadingIndex docstring).
        Pass a DepartureDurationIndex built from this same traj_idx.
        """
        from Step4e_fleet_context import FleetHeadingIndex, DEFAULT_ARRIVAL_WINDOW_DAYS
        vocab_key_by_col = {
            "ARR_SUBREGION_ID": "port_subregion_to_id",
            "ARR_COUNTRY_ID": "port_country_to_id",
            "ARR_REGION_ID": "port_region_to_id",
        }
        vocab_key = vocab_key_by_col.get(subregion_col)
        n_subregions = len(self.vocab[vocab_key]) if vocab_key else None  # None -> FleetHeadingIndex infers from data
        self.fleet_heading_index = FleetHeadingIndex(
            self.traj_idx, subregion_col=subregion_col, n_subregions=n_subregions,
            arrival_window_days=arrival_window_days or DEFAULT_ARRIVAL_WINDOW_DAYS, anchor=anchor,
            duration_index=duration_index, window_fraction=window_fraction,
            min_window_days=min_window_days, max_window_days=max_window_days)
        return self.fleet_heading_index

    def build_fixed_horizon_fleet_index(self, subregion_col="ARR_SUBREGION_ID",
                                         horizons_days=None, weight_by_similarity=False,
                                         eta_channel_lookup=None, history_index=None, use_dep_subregion=False):
        """Call once, after enrich_arrival_labels(data) — builds the
        fixed-horizon RAW OCCUPANCY fleet index used when
        prepare_batch(..., include_fixed_horizon_fleet_context=True) or
        RepresentationLayer(..., use_fixed_horizon_fleet_context=True)
        are used.

        Deliberately a DIFFERENT premise from build_fleet_heading_index's
        own "oracle heading" signal: that one counts each OTHER vessel
        toward a subregion using ITS OWN, true, final arrival subregion,
        in a window anchored to ITS OWN eventual arrival — reflecting the
        fleet AFTER real-world competition/chartering has already
        resolved who goes where. This instead queries ACTUAL, recorded
        vessel positions (nearest-port-assigned to a subregion, not final
        destination) at FIXED horizons from each step's own current date
        — the raw, still-unsettled occupancy at each future moment, not
        the cleared outcome. Still oracle (real recorded positions, not a
        forecast) — a different question asked of the same ground truth.

        weight_by_similarity=False (default): every active vessel counts
        identically (+1.0), same as before. True: ALSO builds
        SimilarityWeightedFleetIndex (a DIFFERENT data source for the
        SAME channel, not a new one — see that class's own module-level
        docstring in Step4e_fleet_context.py) — each vessel's
        contribution is weighted by its similarity to the QUERYING
        vessel across EVERY comparison dimension that index's own
        extended weight formula covers (size, draught, position/
        distance, heading, departure port/subregion, and — when
        eta_channel_lookup/history_index are also given — ETA-based
        progression and the six ship-history statistics too), rather
        than counted flatly. Which of the two actually gets used is
        chosen later, per call, via prepare_batch's own
        weight_fixed_horizon_by_similarity flag — building both here
        just means either is available without rebuilding.

        eta_channel_lookup, history_index (both optional, default None,
        only relevant when weight_by_similarity=True): passed straight
        through to SimilarityWeightedFleetIndex's own constructor —
        when provided, its weight formula ALSO incorporates ETA-
        progression and ship-history similarity; when omitted, those
        two dimensions are simply absent from the weight (neutral, not
        an error). history_index is typically self.history_index (the
        SAME instance build_ship_history_index() already builds, if
        that's been called) — reuse it rather than construct a second
        one. eta_channel_lookup is typically built via
        precompute_eta_progression_lookup(work_dir, self.traj_idx).

        use_dep_subregion=False (default, only relevant when
        weight_by_similarity=True): whether SimilarityWeightedFleetIndex
        also gets a port_to_subregion mapping (via
        build_port_to_subregion_map on DEP_PORT_ID) for its own
        departure-SUBREGION match term specifically. The departure-PORT
        match term is unaffected either way — it needs only DEP_PORT_ID
        directly from traj_idx, already required, not this mapping.
        False leaves the subregion term at a neutral default (every
        candidate's own dep_subregion_id and the query's own both
        default to the same -1 sentinel, so the match check passes
        trivially — no penalty applied, not a hidden always-fail)
        rather than the constructor raising.

        Reuses PortLocationIndex + FleetOccupancyIndex (the same Stage
        1/2 building blocks build_candidate_fleet_state_index's own "full"
        mechanism depends on) but built independently here, not shared
        with that method's own internal instances — kept decoupled
        deliberately, so this can be tested in isolation without also
        requiring the full candidate-conditioned pipeline. The SAME
        PortLocationIndex instance built here IS shared with the
        similarity-weighted index below, though — both need position ->
        subregion assignment and this call has it already computed.
        """
        from Step4e_fleet_context import PortLocationIndex, FleetOccupancyIndex, DEFAULT_FIXED_HORIZONS_DAYS
        vocab_key_by_col = {
            "ARR_SUBREGION_ID": "port_subregion_to_id",
            "ARR_COUNTRY_ID": "port_country_to_id",
            "ARR_REGION_ID": "port_region_to_id",
        }
        vocab_key = vocab_key_by_col.get(subregion_col)
        n_subregions = len(self.vocab[vocab_key]) if vocab_key else int(self.traj_idx[subregion_col].max()) + 1

        port_loc_idx = PortLocationIndex(self.steps_idx, self.traj_idx, subregion_col=subregion_col)
        self.fixed_horizon_fleet_index = FleetOccupancyIndex(self.steps_idx, port_loc_idx, n_subregions=n_subregions)
        self.fixed_horizon_days = tuple(horizons_days) if horizons_days is not None else DEFAULT_FIXED_HORIZONS_DAYS

        if weight_by_similarity:
            from Step4e_fleet_context import SimilarityWeightedFleetIndex
            port_to_subregion = None
            if use_dep_subregion:
                from Step4c_train import build_port_to_subregion_map
                port_to_subregion = build_port_to_subregion_map(
                    self.traj_idx, subregion_col=subregion_col, port_col="DEP_PORT_ID")
            self.similarity_weighted_fleet_index = SimilarityWeightedFleetIndex(
                self.steps_idx, self.g, self.traj_idx, port_loc_idx, n_subregions=n_subregions,
                eta_channel_lookup=eta_channel_lookup, history_index=history_index,
                port_to_subregion=port_to_subregion)

        return self.fixed_horizon_fleet_index

    def build_active_vessel_set_index(self, max_active_vessels=None, use_region_truncation=False,
                                       subregion_col="ARR_SUBREGION_ID", use_dep_subregion=False):
        """Call once, after enrich_arrival_labels(data) — builds the
        active-vessel-set index (Experiment 2: set-pooling over vessel
        embeddings) used when prepare_batch(...,
        include_active_vessel_set_context=True) or RepresentationLayer(...,
        use_active_vessel_set_context=True) are used.

        Deliberately a DIFFERENT mechanism from every other fleet channel:
        instead of aggregating other vessels into counts or deviations,
        this retrieves the actual SET of other active vessels at each
        step's own current moment, each with its own feature vector
        (position + position-diff + geodesic distance, draught, size,
        current declared destination, departure port/subregion, heading
        similarity, journey-time-so-far — see Step4e_fleet_context.py's
        own module-level docstring for the full, current feature list),
        pooled via a permutation-invariant learned-query attention
        (AttentionPool, reused unmodified from Block 2's own
        ShipHistoryGNN). Queried at the CURRENT moment only, not fixed
        future horizons — see ActiveVesselSetIndex's own module-level
        docstring in Step4e_fleet_context.py for the full rationale.

        ETA-progression and ship-history comparison features are NOT
        built here — they need eta_channel_lookup/history_index, which
        this index itself was never constructed with (this index only
        knows steps_idx/gridded/traj_idx). Whatever builds a full
        training batch calls compute_candidate_temporal_history_features
        separately, using the candidate_seg_ids/candidate_step_idx this
        index's own snapshot_for() already returns, and concatenates
        the result onto this index's own 16-column feature array.

        use_region_truncation=False (default, unchanged): when more than
        max_active_vessels are active on a given day, the index truncates
        by raw physical nearest-by-position only. Diagnosed directly on
        real data (diagnostic_active_vessel_counts.py's own "distance to
        Kth-nearest" analysis): for a globally-dispersed fleet, even the
        NEAREST active vessels can be thousands of km away — "nearest"
        doesn't necessarily mean "genuinely local."

        use_region_truncation=True: ALSO builds a PortLocationIndex (the
        same Stage-1 building block FleetOccupancyIndex/
        SimilarityWeightedFleetIndex already depend on) so
        prepare_batch's own active_vessel_truncation_mode=
        "region_then_nearest" becomes available later — prioritizing
        vessels CURRENTLY in the same region as the querying vessel
        (nearest-port-based on CURRENT position, not final destination —
        non-leaky, same convention every other fleet mechanism in this
        project already uses) before falling back to nearest-by-position
        for any remaining capacity. Built fresh here, independently from
        any other method's own PortLocationIndex instance — kept
        decoupled deliberately, same precedent
        build_fixed_horizon_fleet_index already established.

        use_dep_subregion=False (default): whether each active vessel's
        own record ALSO carries its DEPARTURE subregion (from
        DEP_PORT_ID, via build_port_to_subregion_map) — used for the
        dep_subregion_id feature and matched against the querying
        vessel's own. True builds this mapping here, once; False leaves
        every record's own dep_subregion_id as the -1 sentinel
        (unknown), and the feature column stays present but uninformative
        rather than the constructor raising.

        Needs Step3Data's segment_steps_index (self.steps_idx) and local
        gridded features (self.g) — both already built at __init__, no
        extra file needed.
        """
        from Step4e_fleet_context import ActiveVesselSetIndex, DEFAULT_MAX_ACTIVE_VESSELS

        port_loc_idx = None
        if use_region_truncation:
            from Step4e_fleet_context import PortLocationIndex
            port_loc_idx = PortLocationIndex(self.steps_idx, self.traj_idx, subregion_col=subregion_col)

        port_to_subregion = None
        if use_dep_subregion:
            from Step4c_train import build_port_to_subregion_map
            port_to_subregion = build_port_to_subregion_map(
                self.traj_idx, subregion_col=subregion_col, port_col="DEP_PORT_ID")

        self.active_vessel_set_index = ActiveVesselSetIndex(
            self.steps_idx, self.g, self.traj_idx, none_declared_id=self.none_declared_id,
            port_location_index=port_loc_idx, port_to_subregion=port_to_subregion)
        self.max_active_vessels = max_active_vessels if max_active_vessels is not None else DEFAULT_MAX_ACTIVE_VESSELS
        return self.active_vessel_set_index

    def build_candidate_fleet_state_index(self, subregion_col="ARR_SUBREGION_ID", baseline_window_days=7,
                                           min_count_pair=3, min_count_port=5, min_count_subregion=5,
                                           mode="full"):
        """Builds the full candidate-conditioned fleet-state mechanism
        (Stages 1-6): real position tracking (not windows), per-candidate-
        destination projected arrival dates, and seasonal-baseline
        deviation — genuinely different from build_fleet_heading_index,
        which this does NOT replace; both can coexist as separate channels
        (use_fleet_context vs use_candidate_fleet_context on
        RepresentationLayer) so they're directly comparable.

        mode="full" (default): for each candidate, the full cross-subregion
        share and share-deviation picture at that candidate's own
        projected date, not just that candidate's own slice — see
        CandidateFleetStateIndex's docstring for the full layout and
        rationale. mode="compact" or mode="simple" available for
        comparison against earlier, narrower representations.

        Needs Step3Data's segment_steps_index (self.steps_idx) — already
        loaded at __init__, no extra file needed.
        """
        from Step4e_fleet_context import (PortLocationIndex, FleetOccupancyIndex, CandidateDurationIndex,
                                           SeasonalBaselineIndex, SeasonalShareBaselineIndex,
                                           CandidateFleetStateIndex, build_seg_to_dep)
        vocab_key_by_col = {
            "ARR_SUBREGION_ID": "port_subregion_to_id",
            "ARR_COUNTRY_ID": "port_country_to_id",
            "ARR_REGION_ID": "port_region_to_id",
        }
        vocab_key = vocab_key_by_col.get(subregion_col)
        n_subregions = len(self.vocab[vocab_key]) if vocab_key else int(self.traj_idx[subregion_col].max()) + 1

        port_loc_idx = PortLocationIndex(self.steps_idx, self.traj_idx, subregion_col=subregion_col)
        occupancy_idx = FleetOccupancyIndex(self.steps_idx, port_loc_idx, n_subregions=n_subregions)
        candidate_dur_idx = CandidateDurationIndex(
            self.traj_idx, dest_col=subregion_col, min_count_pair=min_count_pair,
            min_count_port=min_count_port, min_count_subregion=min_count_subregion)
        baseline_idx = SeasonalBaselineIndex(occupancy_idx, window_days=baseline_window_days)
        share_baseline_idx = (SeasonalShareBaselineIndex(occupancy_idx, window_days=baseline_window_days)
                               if mode in ("compact", "full") else None)

        self.candidate_fleet_state_index = CandidateFleetStateIndex(
            occupancy_idx, candidate_dur_idx, baseline_idx, n_subregions=n_subregions,
            share_baseline_index=share_baseline_idx, mode=mode)
        self._seg_to_dep = build_seg_to_dep(self.traj_idx)
        return self.candidate_fleet_state_index

    def _build_local_features(self, gridded):
        g = gridded.sort_values(["SEG_ID", "STEP_IDX", "SUBSEQ_STEP_IDX"]).copy()

        g["REL_LON"] = g["GRID_LON_C"] - g["LON"]
        g["REL_LAT"] = g["GRID_LAT_C"] - g["LAT"]

        _first_ts_per_gk = g.groupby(["SEG_ID", "STEP_IDX"])["TIMESTAMP"].transform("min")
        g["REL_TIME_WITHIN_GK_DAYS"] = (g["TIMESTAMP"] - _first_ts_per_gk).dt.total_seconds() / 86400.0

        g["SOG_NORM"] = (g["SOG"].clip(0, 25) / 25.0).astype("float32")
        _cog_rad = np.radians(g["COG"].fillna(0.0).astype("float32"))
        g["COG_SIN"] = np.sin(_cog_rad)
        g["COG_COS"] = np.cos(_cog_rad)
        g["DRAUGHT_NORM"] = ((g["DRAUGHT"].clip(6, 13) - 6.0) / 7.0).astype("float32")

        if self.has_declared:
            g["_DECL_ID"] = g["DECLARED_DEST_PORT_ID"].fillna(self.none_declared_id).astype(int)
            g["_DECL_CONF"] = g["DECLARED_DEST_CONF"].fillna(0.0).astype("float32")
        else:
            g["_DECL_ID"] = self.none_declared_id
            g["_DECL_CONF"] = 0.0

        self.local_numeric_cols = ["REL_LON", "REL_LAT", "REL_TIME_WITHIN_GK_DAYS", "SOG_NORM",
                                    "COG_SIN", "COG_COS", "DRAUGHT_NORM", "_DECL_CONF"]
        return g

    def prepare_batch(self, seg_ids, max_mk_per_batch=50,
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
                       active_vessel_port_to_subregion=None, active_vessel_history_stats_cache=None):
        """Build padded numpy arrays for a batch of segments. Pads to the
        LOCAL max N and max mk within THIS batch — efficient when seg_ids
        are pre-bucketed by similar length (Step 4's data loader); wasteful
        if seg_ids are randomly sampled with very different N (see Step 4b).

        PERFORMANCE: two things matter here, found by profiling a real
        batch — (1) O(1) per-segment lookup via the pre-grouped dicts from
        __init__, not a full-table scan; (2) converting each segment's
        points to numpy ONCE and finding grid-step boundaries with a single
        np.diff() pass, rather than pandas .groupby("STEP_IDX") + repeated
        .sort_values()/.values calls inside the loop. Pandas' per-call
        overhead is large relative to typical group size here (median
        mk=1), so doing this in pandas inside a tight loop dominated total
        wall time far more than the top-level table lookup did.

        max_mk_per_batch: hard cap on mk (points per grid-step) WITHIN one
        batch, independent of Step3a's own MAX_SUBSEQ_LEN cap. Found via a
        real CUDA OOM: the StepwiseGRU reshapes to [batch*N, mk_max, f]
        before running the GRU, so a batch combining a long trajectory
        (large N, from bucketing) with one segment's long dwell (large mk)
        multiplies both together — backprop-through-time memory for that
        one layer scales with batch*N*mk_max*hidden_dim, and a rare
        worst-case batch (N~322, mk~150 in the case that triggered this)
        can spike to ~20GB+ on its own. Bucketing by N alone doesn't bound
        mk. When a step's own points exceed this cap, keeps the LAST
        max_mk_per_batch points (same "most-recent-state-before-moving-on"
        rationale as Step3a's own mk truncation). Set to None to disable
        (not recommended for training — fine for small-scale inspection).
        """
        idx_rows = self.traj_idx[self.traj_idx["seg_id"].isin(seg_ids)].set_index("seg_id").loc[seg_ids]
        steps_rows = {sid: self._steps_by_seg.get(sid, self._empty_steps) for sid in seg_ids}
        max_N = max(len(v) for v in steps_rows.values())

        batch = len(seg_ids)
        grid_lon = np.zeros((batch, max_N), dtype="float32")
        grid_lat = np.zeros((batch, max_N), dtype="float32")
        tau = np.zeros((batch, max_N), dtype="float32")
        n_mask = np.zeros((batch, max_N), dtype="float32")

        # Convert each segment's points to numpy ONCE, find grid-step
        # boundaries via a single diff pass (data is pre-sorted by
        # STEP_IDX, SUBSEQ_STEP_IDX so each step's rows are contiguous).
        per_seg_arrays = {}
        max_mk = 1
        for sid in seg_ids:
            pts = self._g_by_seg.get(sid, self._empty_g)
            step_idx_arr = pts["STEP_IDX"].to_numpy()
            if len(step_idx_arr):
                numeric_arr = pts[self.local_numeric_cols].to_numpy(dtype="float32")
                decl_arr = pts["_DECL_ID"].to_numpy()
                change_pts = np.flatnonzero(np.diff(step_idx_arr)) + 1
                starts = np.concatenate(([0], change_pts))
                ends = np.concatenate((change_pts, [len(step_idx_arr)]))
                step_values = step_idx_arr[starts]
                step_lens = ends - starts
                if max_mk_per_batch is not None:
                    step_lens = np.minimum(step_lens, max_mk_per_batch)
                max_mk = max(max_mk, int(step_lens.max()))
            else:
                numeric_arr = np.zeros((0, len(self.local_numeric_cols)), dtype="float32")
                decl_arr = np.zeros((0,), dtype="int64")
                starts = ends = step_values = np.array([], dtype="int64")
            per_seg_arrays[sid] = (step_values, starts, ends, numeric_arr, decl_arr)

        local_numeric = np.zeros((batch, max_N, max_mk, len(self.local_numeric_cols)), dtype="float32")
        local_decl_id = np.full((batch, max_N, max_mk), self.none_declared_id, dtype="int32")
        local_mask = np.zeros((batch, max_N, max_mk), dtype="float32")

        dep_port_id = np.zeros((batch,), dtype="int32")
        size_class_id = np.zeros((batch,), dtype="int32")

        for b, sid in enumerate(seg_ids):
            srow = idx_rows.loc[sid]
            dep_port_id[b] = int(srow["DEP_PORT_ID"]) if pd.notna(srow["DEP_PORT_ID"]) else self.none_declared_id
            size_class_id[b] = int(srow["SIZE_CLASS_ID"])

            steps = steps_rows[sid]
            n_here = len(steps)
            grid_lon[b, :n_here] = steps["GRID_LON_C"].to_numpy()
            grid_lat[b, :n_here] = steps["GRID_LAT_C"].to_numpy()
            tau[b, :n_here] = steps["TIME_OFFSET_DAYS"].to_numpy()
            n_mask[b, :n_here] = 1.0

            step_values, starts, ends, numeric_arr, decl_arr = per_seg_arrays[sid]
            for sv, s, e in zip(step_values, starts, ends):
                if sv >= max_N:
                    continue
                if max_mk_per_batch is not None and (e - s) > max_mk_per_batch:
                    s = e - max_mk_per_batch  # keep the LAST max_mk_per_batch points
                mk_here = e - s
                local_numeric[b, sv, :mk_here, :] = numeric_arr[s:e]
                local_decl_id[b, sv, :mk_here] = decl_arr[s:e]
                local_mask[b, sv, :mk_here] = 1.0

        result = {
            "grid_lon": ops.convert_to_tensor(grid_lon),
            "grid_lat": ops.convert_to_tensor(grid_lat),
            "tau": ops.convert_to_tensor(tau),
            "dep_port_id": ops.convert_to_tensor(dep_port_id),
            "size_class_id": ops.convert_to_tensor(size_class_id),
            "local_numeric": ops.convert_to_tensor(local_numeric),
            "local_declared_dest_id": ops.convert_to_tensor(local_decl_id),
            "local_mask": ops.convert_to_tensor(local_mask),
        }

        if include_ship_history:
            if self.history_index is None:
                raise ValueError("include_ship_history=True but build_ship_history_index() "
                                  "hasn't been called yet — call it once after "
                                  "enrich_arrival_labels(data).")
            from Step4d_ship_history import prepare_history_batch, MAX_HISTORY
            hist_batch = prepare_history_batch(
                self.history_index, seg_ids, none_port_id=self.none_declared_id,
                max_history=history_max_history or MAX_HISTORY,
                use_contract_period_feature=use_contract_period_feature)
            for k, v in hist_batch.items():
                result[k] = ops.convert_to_tensor(v)

        if include_fleet_context:
            if self.fleet_heading_index is None:
                raise ValueError("include_fleet_context=True but build_fleet_heading_index() "
                                  "hasn't been called yet — call it once after "
                                  "enrich_arrival_labels(data).")
            from Step4e_fleet_context import prepare_fleet_heading_batch
            fleet_counts = prepare_fleet_heading_batch(self.fleet_heading_index, self, seg_ids, max_N,
                                                         normalize=normalize_fleet)
            result["fleet_heading_counts"] = ops.convert_to_tensor(fleet_counts)

        if include_candidate_fleet_context:
            if self.candidate_fleet_state_index is None:
                raise ValueError("include_candidate_fleet_context=True but "
                                  "build_candidate_fleet_state_index() hasn't been called yet — "
                                  "call it once after enrich_arrival_labels(data).")
            from Step4e_fleet_context import prepare_candidate_fleet_batch
            candidate_vecs = prepare_candidate_fleet_batch(
                self.candidate_fleet_state_index, self, seg_ids, max_N, self._seg_to_dep)
            result["candidate_fleet_state"] = ops.convert_to_tensor(candidate_vecs)

        if include_fixed_horizon_fleet_context:
            if weight_fixed_horizon_by_similarity:
                if self.similarity_weighted_fleet_index is None:
                    raise ValueError("include_fixed_horizon_fleet_context=True, "
                                      "weight_fixed_horizon_by_similarity=True but "
                                      "build_fixed_horizon_fleet_index(weight_by_similarity=True) "
                                      "hasn't been called yet -- call it once after "
                                      "enrich_arrival_labels(data), with weight_by_similarity=True.")
                from Step4e_fleet_context import prepare_similarity_weighted_fleet_batch, DEFAULT_SIZE_SIGMA, DEFAULT_DRAUGHT_SIGMA
                fixed_horizon_vecs = prepare_similarity_weighted_fleet_batch(
                    self.similarity_weighted_fleet_index, self, seg_ids, max_N,
                    horizons_days=self.fixed_horizon_days,
                    size_sigma=fixed_horizon_size_sigma if fixed_horizon_size_sigma is not None else DEFAULT_SIZE_SIGMA,
                    draught_sigma=fixed_horizon_draught_sigma if fixed_horizon_draught_sigma is not None else DEFAULT_DRAUGHT_SIGMA,
                    weight_combination=fixed_horizon_weight_combination)
            else:
                if self.fixed_horizon_fleet_index is None:
                    raise ValueError("include_fixed_horizon_fleet_context=True but "
                                      "build_fixed_horizon_fleet_index() hasn't been called yet — "
                                      "call it once after enrich_arrival_labels(data).")
                from Step4e_fleet_context import prepare_fixed_horizon_fleet_batch
                fixed_horizon_vecs = prepare_fixed_horizon_fleet_batch(
                    self.fixed_horizon_fleet_index, self, seg_ids, max_N, horizons_days=self.fixed_horizon_days)
            # SAME input key either way -- RepresentationLayer's Channel 7
            # (FixedHorizonFleetEncoder) doesn't know or care which data
            # source produced this tensor, only its shape.
            result["fixed_horizon_fleet_occupancy"] = ops.convert_to_tensor(fixed_horizon_vecs)

        if include_active_vessel_set_context:
            if self.active_vessel_set_index is None:
                raise ValueError("include_active_vessel_set_context=True but "
                                  "build_active_vessel_set_index() hasn't been called yet — "
                                  "call it once after enrich_arrival_labels(data).")
            if active_vessel_truncation_mode == "region_then_nearest" and not self.active_vessel_set_index.has_region:
                raise ValueError("active_vessel_truncation_mode='region_then_nearest' but "
                                  "build_active_vessel_set_index() was called without "
                                  "use_region_truncation=True — call it again with that flag set.")
            # NOTE (not enforced here): whether active_vessel_eta_channel_
            # lookup/active_vessel_history_index/active_vessel_port_to_
            # subregion are ALL given determines whether this produces 16
            # or 33 columns -- this MUST match whatever
            # include_temporal_history the caller's own
            # ActiveVesselSetEncoder was actually built with, or the
            # model hits a shape mismatch inside its own forward pass.
            # Step3Data (this method's own class) has no reference to
            # that encoder object at all -- it's owned by a separately-
            # constructed RepresentationLayer, not by Step3Data -- so
            # this genuinely can't be validated from here; keeping this
            # consistent is the caller's own responsibility.
            from Step4e_fleet_context import prepare_active_vessel_set_batch
            # candidate_seg_ids/candidate_step_idx (the 3rd/4th return

            # values) aren't needed here -- they exist for a caller that
            # wants to layer FURTHER per-candidate lookups on top of
            # what this function already produces; prepare_active_vessel_
            # set_batch's own eta/history parameters below already cover
            # that need for THIS caller, via concatenation internally.
            vessel_features, vessel_mask, _, _ = prepare_active_vessel_set_batch(
                self.active_vessel_set_index, self, seg_ids, max_N, max_vessels=self.max_active_vessels,
                truncation_mode=active_vessel_truncation_mode,
                eta_channel_lookup=active_vessel_eta_channel_lookup,
                history_index=active_vessel_history_index,
                port_to_subregion=active_vessel_port_to_subregion,
                history_stats_cache=active_vessel_history_stats_cache)
            result["active_vessel_features"] = ops.convert_to_tensor(vessel_features)
            result["active_vessel_mask"] = ops.convert_to_tensor(vessel_mask)

        return result, n_mask
