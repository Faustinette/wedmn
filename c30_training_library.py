# =============================================================================
# Section 5.1 — Training library (key functions x24)
# Migrated verbatim from Main_forGitHub.ipynb cells [55].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 55]
# ----------------------------------------------------------------------
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # must precede any keras import this session
# =============================================================================
# LIB CELL L5-MIN -- Step4c_train, MINIMAL SUBSET (24 definitions, verbatim)
# Only what the main-model training + evaluation path needs and the notebook
# does not already define. Computed as the code-only transitive closure of:
#     train_residual_progression_variant, evaluate_full_report_metrics
# Groups: checkpoint I/O (save/load weights+model, per-epoch checkpoints,
# fast-result cache paths) | gate-signal machinery (ETA parse/compare/lookup,
# hist-avg via L4e's duration indices, alternative-progression compute) |
# evaluation internals (predictions collect, full metrics incl. per-band,
# macro-AUC) | the trainer itself. Every def is verbatim from the live file.
# =============================================================================
import os, json, math, time, pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import keras
from keras import ops

# shim replacing the stripped 'import Step3b_representation_layer' module ref
import types as _types
Step3b_representation_layer = _types.SimpleNamespace(DATA_SUBFOLDER="")

def _progression_labels(boundaries):
    return [f"<={int(round(b*100))}%" for b in boundaries]


def save_trained_weights(path, repr_layer, model):
    """Saves raw parameter tensors only (not architecture). Pair with
    load_trained_weights() using the SAME hyperparameters used to build
    repr_layer/model in the first place."""
    all_vars = repr_layer.trainable_variables + model.trainable_variables
    torch.save([v.value.detach().cpu() for v in all_vars], path)


def load_trained_weights(path, repr_layer, model):
    """Loads parameter tensors saved by save_trained_weights() into an
    already-constructed repr_layer/model pair (must already be built — i.e.
    called once on a dummy batch — so trainable_variables exist with the
    right shapes to check against)."""
    saved_tensors = torch.load(path, map_location="cpu")
    all_vars = repr_layer.trainable_variables + model.trainable_variables
    if len(saved_tensors) != len(all_vars):
        raise ValueError(f"Checkpoint has {len(saved_tensors)} tensors but the "
                          f"reconstructed model has {len(all_vars)} trainable variables — "
                          f"architecture hyperparameters don't match what was used to train.")
    for i, (v, t) in enumerate(zip(all_vars, saved_tensors)):
        if tuple(v.value.shape) != tuple(t.shape):
            raise ValueError(f"Shape mismatch at variable {i} ({v.path}): "
                              f"checkpoint has {tuple(t.shape)}, model expects {tuple(v.value.shape)} — "
                              f"architecture hyperparameters don't match what was used to train.")
        v.assign(t)


def load_trained_model(weights_path, n_ports, n_size_classes, n_classes,
                        d_model=128, gru_layers=1, n_casp_layers=2,
                        n_heads_mca=2, n_heads_msa=4, d_ff=None,
                        use_spatial_channel=True, use_local_pattern_channel=True,
                        use_departure_port_channel=True, use_ship_size_channel=True,
                        use_temporal_encoding=True,
                        use_declared_destination=True, use_ship_history=False,
                        history_gat_layers=2, history_gat_heads=4, gate_ship_history=False,
                        ship_history_attention=False, use_recency_bias=False, use_contract_period_feature=False,
                        use_departure_subregion_channel=False, n_subregions_departure=None, dep_port_to_subregion_lookup=None,
                        use_eta_channel=False,
                        use_fleet_context=False, n_subregions_fleet=None,
                        use_fixed_horizon_fleet_context=False, n_subregions_fixed_horizon=None,
                        use_active_vessel_set_context=False,
                        use_moe_ffn=False, n_experts=2, gate_uses_content=False, content_code_dim=8,
                        moe_last_layer_only=False, n_alt_progression_signals=0, use_departure_gate=False,
                        n_departure_subregions=None, departure_embed_dim=8):
    """Rebuilds a fresh RepresentationLayer + WAYModel with the given
    architecture, forces weight-building with a dummy forward pass, then
    loads trained weights from a checkpoint saved by save_trained_weights()
    (or the notebook's save_checkpoint()). ALL architecture arguments must
    match what was used at training time, INCLUDING use_spatial_channel,
    use_local_pattern_channel, use_departure_port_channel,
    use_ship_size_channel, use_temporal_encoding, use_declared_destination,
    use_ship_history, use_fleet_context, use_recency_bias,
    use_contract_period_feature, use_departure_subregion_channel,
    use_eta_channel, AND use_moe_ffn/n_experts — these aren't stored in
    the checkpoint file itself, only the raw tensors are (a mismatch
    changes input shapes or parameter counts, so it'll raise the same
    shape/count-mismatch error as any other wrong architecture argument
    — EXCEPT use_temporal_encoding, which doesn't change parameter
    shapes at all (self.te's own weights are unaffected either way, only
    whether its output gets used), so a mismatch there would silently
    load successfully but not reproduce the trained model's actual
    behavior — still worth passing correctly, just won't be caught by a
    shape-mismatch error the way the others would be).
    Returns (repr_layer, model), ready for inference — no training needed.
    """
    d_ff = d_ff or d_model * 2
    repr_layer = RepresentationLayer(d_model, n_ports, n_size_classes, gru_layers=gru_layers,
                                      use_spatial_channel=use_spatial_channel,
                                      use_local_pattern_channel=use_local_pattern_channel,
                                      use_departure_port_channel=use_departure_port_channel,
                                      use_ship_size_channel=use_ship_size_channel,
                                      use_temporal_encoding=use_temporal_encoding,
                                      use_declared_destination=use_declared_destination,
                                      use_ship_history=use_ship_history,
                                      history_gat_layers=history_gat_layers,
                                      history_gat_heads=history_gat_heads,
                                      gate_ship_history=gate_ship_history,
                                      ship_history_attention=ship_history_attention,
                                      use_recency_bias=use_recency_bias,
                                      use_contract_period_feature=use_contract_period_feature,
                                      use_departure_subregion_channel=use_departure_subregion_channel,
                                      n_subregions_departure=n_subregions_departure,
                                      dep_port_to_subregion_lookup=dep_port_to_subregion_lookup,
                                      use_eta_channel=use_eta_channel,
                                      use_fleet_context=use_fleet_context,
                                      n_subregions_fleet=n_subregions_fleet,
                                      use_fixed_horizon_fleet_context=use_fixed_horizon_fleet_context,
                                      n_subregions_fixed_horizon=n_subregions_fixed_horizon,
                                      use_active_vessel_set_context=use_active_vessel_set_context)
    model = WAYModel(d_model, n_classes, n_layers=n_casp_layers, n_heads_mca=n_heads_mca,
                      n_heads_msa=n_heads_msa, d_ff=d_ff, use_moe_ffn=use_moe_ffn, n_experts=n_experts,
                      gate_uses_content=gate_uses_content, content_code_dim=content_code_dim,
                      moe_last_layer_only=moe_last_layer_only, n_alt_progression_signals=n_alt_progression_signals,
                      use_departure_gate=use_departure_gate, n_departure_subregions=n_departure_subregions,
                      departure_embed_dim=departure_embed_dim)

    dummy_inputs = {
        "grid_lon": np.zeros((1, 2), dtype="float32"),
        "grid_lat": np.zeros((1, 2), dtype="float32"),
        "tau": np.zeros((1, 2), dtype="float32"),
        "dep_port_id": np.zeros((1,), dtype="int32"),
        "size_class_id": np.zeros((1,), dtype="int32"),
        "local_numeric": np.zeros((1, 2, 1, 8), dtype="float32"),
        "local_declared_dest_id": np.zeros((1, 2, 1), dtype="int32"),
        "local_mask": np.ones((1, 2, 1), dtype="float32"),
    }
    if use_ship_history:
        n_numeric = 3 if use_contract_period_feature else 2
        dummy_inputs.update({
            "node_dep_port_id": np.zeros((1, 1), dtype="int32"),
            "node_arr_port_id": np.zeros((1, 1), dtype="int32"),
            "node_numeric": np.zeros((1, 1, n_numeric), dtype="float32"),
            "edge_mask": np.zeros((1, 1, 1), dtype="float32"),
            "node_mask": np.zeros((1, 1), dtype="float32"),  # all-zero: exercises cold-start path too
        })
    if use_eta_channel:
        dummy_inputs["eta_channel_values"] = np.zeros((1, 2), dtype="float32")
    if use_fleet_context:
        dummy_inputs["fleet_heading_counts"] = np.zeros((1, 2, n_subregions_fleet), dtype="float32")
    if use_fixed_horizon_fleet_context:
        dummy_inputs["fixed_horizon_fleet_occupancy"] = np.zeros((1, 2, n_subregions_fixed_horizon), dtype="float32")
    if use_active_vessel_set_context:
        # AttentionPool/ActiveVesselSetEncoder have no weights that depend
        # on K (the vessel-set size) at all -- the mechanism is
        # permutation-invariant and size-agnostic by construction, so
        # this dummy K=2 doesn't need to match whatever max_vessels was
        # actually used at training time, only be non-degenerate enough
        # to force weight-building. Feature dim is 8 (lat, lon,
        # draught_norm, size_class_id, decl_id, decl_conf, size_diff,
        # draught_diff) -- not 6 -- since the size/draught similarity-
        # difference features were added on top of the original set.
        dummy_inputs["active_vessel_features"] = np.zeros((1, 2, 2, 8), dtype="float32")
        dummy_inputs["active_vessel_mask"] = np.zeros((1, 2, 2), dtype="float32")
    dummy_mask = np.ones((1, 2), dtype="float32")
    dummy_alt = [np.zeros((1, 2), dtype="float32") for _ in range(n_alt_progression_signals)] if n_alt_progression_signals > 0 else None
    dummy_dep_subregion = np.zeros((1,), dtype="int32") if use_departure_gate else None
    _ = model(repr_layer(dummy_inputs), key_padding_mask=dummy_mask, alt_progression_fracs=dummy_alt,
              departure_subregion_ids=dummy_dep_subregion)

    load_trained_weights(weights_path, repr_layer, model)
    return repr_layer, model


# ═════════════════════════════════════════════════════════════════════════════
# [3b] FEATURE / CHANNEL CONTRIBUTION (diagnostic, not in the paper)
#
# Two complementary views, both computed from an ALREADY-TRAINED model (no
# retraining needed, so this is cheap regardless of how long training took):
#
#  1. Channel-attention summary — reads out the MCA module's own learned
#     attention weights (Section IV-B1) after a forward pass. This is
#     exactly what the model itself decided to emphasize among the 4
#     channels (spatial / local-pattern / departure-port / ship-type),
#     already computed internally — free to extract, nothing extra to run.
#
#  2. Permutation importance — for each individual input feature, shuffle
#     its values across the batch (breaking that feature's association
#     with its trajectory while leaving everything else intact) and measure
#     the drop in final-step accuracy. Caveat: shuffling grid_lon/grid_lat/
#     tau/local_numeric across instances of different lengths means some
#     "real" positions of one instance get paired with what were padding
#     values of another — a minor confound, acceptable for a first-pass
#     importance read but not a rigorous measure. Requires N_repeat extra
#     full validation passes per feature — cheap relative to training, but
#     not free; scales with len(feature_list) * n_repeats * len(val_loader).
# ═════════════════════════════════════════════════════════════════════════════


def _fast_fleet_result_path(work_dir, target_col, condition_name):
    ckpt_dir = os.path.join(work_dir, RESULTS_SUBFOLDER)
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{target_col}_{condition_name}.json")


def load_fast_fleet_result(work_dir, target_col, condition_name):
    """Loads a single condition's result (history/val_history/metrics) back
    from disk, in the exact shape run_fast_fleet_comparison's internal
    _train() produces — usable directly as base_result=... in a later call,
    surviving a Colab session restart, unlike keeping it in memory."""
    path = _fast_fleet_result_path(work_dir, target_col, condition_name)
    with open(path) as f:
        saved = json.load(f)
    saved["metrics"]["progression_acc"] = np.array(saved["metrics"]["progression_acc"])
    if "channel_attention" in saved:
        saved["channel_attention"] = np.array(saved["channel_attention"])
    return saved


def _epoch_checkpoint_paths(work_dir, target_col, condition_name):
    ckpt_dir = os.path.join(work_dir, RESULTS_SUBFOLDER)
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"{target_col}_{condition_name}_epochckpt"
    return os.path.join(ckpt_dir, f"{tag}.pt"), os.path.join(ckpt_dir, f"{tag}_meta.json")


def _save_epoch_checkpoint(work_dir, target_col, condition_name, repr_layer, model,
                            completed_epochs, train_hist, val_hist):
    """Saves progress WITHIN a single condition's training run — after
    every epoch, not just when the whole run finishes. If the session is
    interrupted mid-run, the next attempt resumes from the last completed
    epoch instead of retraining from scratch. Separate from
    _save_fast_fleet_result (the FINAL result for a fully-completed
    condition) — this is deliberately more frequent, and gets deleted once
    the condition finishes normally (see _clear_epoch_checkpoint)."""
    weights_path, meta_path = _epoch_checkpoint_paths(work_dir, target_col, condition_name)
    tmp_weights = weights_path + ".tmp"
    save_trained_weights(tmp_weights, repr_layer, model)
    os.replace(tmp_weights, weights_path)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump({"completed_epochs": completed_epochs, "train_hist": train_hist, "val_hist": val_hist}, f)
    os.replace(tmp_meta, meta_path)


def _load_epoch_checkpoint(work_dir, target_col, condition_name, repr_layer, model):
    """Returns (completed_epochs, train_hist, val_hist) if a resumable
    checkpoint exists and its weights loaded successfully into the
    ALREADY-BUILT repr_layer/model (same shapes required — mismatched
    architecture is treated as "no usable checkpoint", not a crash);
    returns None if nothing to resume from."""
    weights_path, meta_path = _epoch_checkpoint_paths(work_dir, target_col, condition_name)
    if not (os.path.exists(weights_path) and os.path.exists(meta_path)):
        return None
    try:
        load_trained_weights(weights_path, repr_layer, model)
    except ValueError as e:
        print(f"    [epoch checkpoint] found but couldn't load ({e}) — starting fresh instead.")
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["completed_epochs"], meta["train_hist"], meta["val_hist"]


def _clear_epoch_checkpoint(work_dir, target_col, condition_name):
    weights_path, meta_path = _epoch_checkpoint_paths(work_dir, target_col, condition_name)
    for p in (weights_path, meta_path):
        if os.path.exists(p):
            os.remove(p)


def _save_fast_fleet_result(work_dir, target_col, condition_name, result, max_progression_frac=None,
                             min_progression_frac=None, warm_start_from=None):
    path = _fast_fleet_result_path(work_dir, target_col, condition_name)
    to_save = {
        "history": result["history"],
        "val_history": result["val_history"],
        "metrics": {**result["metrics"], "progression_acc": result["metrics"]["progression_acc"].tolist()
                     if hasattr(result["metrics"]["progression_acc"], "tolist")
                     else result["metrics"]["progression_acc"]},
        "max_progression_frac": max_progression_frac,
        "min_progression_frac": min_progression_frac,
        "warm_start_from": warm_start_from,
    }
    if "channel_attention" in result:
        ca = result["channel_attention"]
        to_save["channel_attention"] = ca.tolist() if hasattr(ca, "tolist") else ca
    if "train_history_by_band" in result:
        to_save["train_history_by_band"] = result["train_history_by_band"]
        to_save["val_history_by_band"] = result["val_history_by_band"]
    # Atomic write: build the full string in memory and write to a temp
    # file first, THEN rename into place — if json.dump or anything else
    # fails partway, the real target path is never touched, so a failed
    # save can't leave a corrupt/empty file that skip_existing would later
    # "find" and fail to load with a much more confusing error.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(to_save, f)
    os.replace(tmp_path, path)
    print(f"    saved -> {path}")


def _regime_weights_path(work_dir, target_col, condition_name):
    ckpt_dir = os.path.join(work_dir, RESULTS_SUBFOLDER)
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{target_col}_{condition_name}_weights.pt")


def parse_ais_eta(eta_str, reference_ts, wraparound_tolerance_days=60):
    """Parses one AIS ETA string ("MM-DD HH:MM") into an absolute
    timestamp, using reference_ts (the TIMESTAMP of the position report
    that reported this ETA) to infer the year. If the same-year candidate
    would fall MORE than wraparound_tolerance_days before reference_ts,
    assumes the ETA crosses a year boundary and uses next year instead
    (e.g. a "01-15" ETA reported in December must mean next January, not
    one that already passed) — the tolerance exists so a merely-overdue,
    not-yet-updated ETA (a few days in the past, common and legitimate)
    isn't mistakenly wrapped into the wrong year. Returns pd.NaT for
    missing/invalid/sentinel values (including AIS's "00-00 00:00").
    """
    if pd.isna(eta_str):
        return pd.NaT
    eta_str = str(eta_str).strip()
    try:
        month_str, rest = eta_str.split("-")
        day_str, time_part = rest.split(" ")
        month, day = int(month_str), int(day_str)
        if month == 0 or day == 0 or month > 12 or day > 31:
            return pd.NaT
        hour_str, minute_str = time_part.split(":")
        hour, minute = int(hour_str), int(minute_str)
        reference_ts = pd.Timestamp(reference_ts)
        candidate = pd.Timestamp(year=reference_ts.year, month=month, day=day, hour=hour, minute=minute)
        if candidate < reference_ts - pd.Timedelta(days=wraparound_tolerance_days):
            candidate = candidate.replace(year=reference_ts.year + 1)
        return candidate
    except (ValueError, IndexError):
        return pd.NaT


def build_eta_comparison(work_dir, traj_idx, target_col="ARR_SUBREGION_ID"):
    """Loads the raw gridded AIS data, parses every ETA entry, and joins
    against each segment's actual departure/arrival (from traj_idx) to
    build a per-position-report comparison: how far off was the captain's
    ETA from the true arrival, and how does the ETA-INFERRED progression
    fraction compare to the TRUE progression fraction the MoE currently
    trains on. Returns a DataFrame with one row per valid (non-sentinel,
    parseable) ETA entry — the raw material for both parts of the
    analysis (reliability check and progression-deviation check).
    """
    gridded = pd.read_parquet(os.path.join(work_dir, Step3b_representation_layer.DATA_SUBFOLDER, "trajectories_gridded.parquet"))
    gridded = gridded[["SEG_ID", "STEP_IDX", "TIMESTAMP", "ETA"]].copy()
    gridded["TIMESTAMP"] = pd.to_datetime(gridded["TIMESTAMP"])

    traj = traj_idx.set_index("seg_id")[["dep_ts", "arr_ts"]].copy()
    traj["dep_ts"] = pd.to_datetime(traj["dep_ts"])
    traj["arr_ts"] = pd.to_datetime(traj["arr_ts"])

    merged = gridded.merge(traj, left_on="SEG_ID", right_index=True, how="inner")

    merged["eta_parsed"] = [
        parse_ais_eta(eta, ts) for eta, ts in zip(merged["ETA"], merged["TIMESTAMP"])
    ]
    valid = merged.dropna(subset=["eta_parsed"]).copy()

    valid["eta_error_hours"] = (valid["arr_ts"] - valid["eta_parsed"]).dt.total_seconds() / 3600.0
    valid["elapsed_hours"] = (valid["TIMESTAMP"] - valid["dep_ts"]).dt.total_seconds() / 3600.0
    valid["eta_total_duration_hours"] = (valid["eta_parsed"] - valid["dep_ts"]).dt.total_seconds() / 3600.0
    valid["true_total_duration_hours"] = (valid["arr_ts"] - valid["dep_ts"]).dt.total_seconds() / 3600.0

    # Guard against a degenerate/contradictory ETA (e.g. parsed as BEFORE
    # departure, or before the reporting timestamp itself) rather than
    # silently producing a nonsensical negative or >1 progression fraction.
    plausible = (valid["eta_total_duration_hours"] > 0) & (valid["elapsed_hours"] >= 0)
    valid = valid[plausible].copy()

    valid["eta_inferred_progression"] = (valid["elapsed_hours"] / valid["eta_total_duration_hours"]).clip(0, 1)
    valid["true_progression"] = (valid["elapsed_hours"] / valid["true_total_duration_hours"]).clip(0, 1)
    valid["progression_deviation"] = valid["eta_inferred_progression"] - valid["true_progression"]

    return valid


def build_port_to_subregion_map(traj_idx, subregion_col="ARR_SUBREGION_ID", port_col="ARR_PORT_ID"):
    """Derives a PORT_ID -> SUBREGION_ID mapping purely from traj_idx's own
    arrival records — every port that's ever an arrival destination has a
    known subregion from that row (taking the mode, in case of any rare
    data inconsistency). Reusable for ANY port-id column, including
    DEP_PORT_ID — departure ports are, in general, also ports that appear
    as arrival destinations elsewhere in the same dataset, since the fleet
    cycles continuously between the same set of ports. Drops rows with a
    missing port or subregion FIRST — a group with zero valid values
    would otherwise make .mode() return an empty Series, and indexing
    [0] into that crashes rather than just skipping the unmappable port."""
    valid = traj_idx.dropna(subset=[port_col, subregion_col])

    def _safe_mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else None

    return valid.groupby(port_col)[subregion_col].agg(_safe_mode).to_dict()


def compute_alternative_progression(mode, tau, dep_port_ids=None, size_class_ids=None, dep_months=None,
                                     duration_index=None, fine_duration_index=None):
    """Computes an alternative progression_frac [batch, N] array to pass as
    WAYModel's external_progression_frac. tau: [batch, N] elapsed DAYS
    since departure (this is inputs["tau"] — already a standard model
    input, TIME_OFFSET_DAYS, no new data threading needed for any of
    these modes).

    mode="historical_avg_port": elapsed / historical average duration for
    this segment's departure port (via DepartureDurationIndex — causally
    valid, since it's built from OTHER segments' history, never this
    segment's own outcome).

    mode="historical_avg_finer": same idea, via FineDurationIndex
    (bucketed by port + size class + month, falling back to the coarser
    port-level estimate when a specific bucket lacks enough history).

    mode="raw_elapsed": elapsed days directly, scaled by a FIXED constant
    (not this voyage's own duration) purely to keep values in a
    reasonable range for the gate's Dense layers — always causally valid
    by construction, since elapsed time is always known.

    mode="none": returns a CONSTANT array (0.5 everywhere), NOT None — this
    is the actual fix for a real bug: WAYModel.call() treats
    external_progression_frac=None as "no override provided, fall back to
    computing the TRUE internal signal" — so returning None here would
    have silently made mode="none" train on the exact same true signal as
    mode="true", not genuinely test anything. A constant carries ZERO
    positional information (identical at every step, so the gate's
    progression-based input can never vary by position), while still
    being a real, valid tensor the override mechanism actually uses —
    the gate's only remaining way to vary its output by step is
    content_code, if gate_uses_content=True.
    """
    if mode == "none":
        tau_np = tau if isinstance(tau, np.ndarray) else ops.convert_to_numpy(tau)
        return np.full_like(tau_np, 0.5, dtype="float32")

    tau_np = tau if isinstance(tau, np.ndarray) else ops.convert_to_numpy(tau)

    if mode == "raw_elapsed":
        RAW_ELAPSED_SCALE_DAYS = 30.0  # fixed constant, NOT this voyage's own duration
        return (tau_np / RAW_ELAPSED_SCALE_DAYS).astype("float32")

    if mode == "historical_avg_port":
        if duration_index is None or dep_port_ids is None:
            raise ValueError("mode='historical_avg_port' requires duration_index and dep_port_ids")
        dep_port_np = dep_port_ids if isinstance(dep_port_ids, np.ndarray) else ops.convert_to_numpy(dep_port_ids)
        expected_hours = np.array([duration_index.expected(int(p))[0] for p in dep_port_np])  # [batch]
        expected_days = np.maximum(expected_hours / 24.0, 0.5)
        progression = tau_np / expected_days[:, None]
        return np.clip(progression, 0.0, 1.5).astype("float32")

    if mode == "historical_avg_finer":
        if fine_duration_index is None or dep_port_ids is None or size_class_ids is None or dep_months is None:
            raise ValueError("mode='historical_avg_finer' requires fine_duration_index, dep_port_ids, "
                              "size_class_ids, dep_months")
        dep_port_np = dep_port_ids if isinstance(dep_port_ids, np.ndarray) else ops.convert_to_numpy(dep_port_ids)
        size_np = size_class_ids if isinstance(size_class_ids, np.ndarray) else ops.convert_to_numpy(size_class_ids)
        expected_hours = np.array([
            fine_duration_index.expected(int(p), int(s), int(m))[0]
            for p, s, m in zip(dep_port_np, size_np, dep_months)
        ])
        expected_days = np.maximum(expected_hours / 24.0, 0.5)
        progression = tau_np / expected_days[:, None]
        return np.clip(progression, 0.0, 1.5).astype("float32")

    raise ValueError(f"unknown mode: {mode!r}")


def _evaluate_with_core_and_alt_progression(model, repr_layer, val_loader, core_and_alt_fn,
                                             departure_ids_fn=None, boundaries=DEFAULT_PROGRESSION_BOUNDARIES,
                                             eta_channel_lookup=None):
    """Same as _evaluate_with_external_progression, but for models with
    n_alt_progression_signals>0 — core_and_alt_fn(inputs, seg_ids) must
    return (core, alts) where alts is a list matching the model's own
    n_alt_progression_signals count (or None if the model has none).
    Needed because passing external_progression_frac alone, with no
    alt_progression_fracs, would fail MixtureOfExpertsFeedForward's own
    validation whenever the model actually has residual-gated alt signals.
    departure_ids_fn(seg_ids) -> [batch] int array, only needed if the
    model has use_departure_gate=True.
    eta_channel_lookup: only needed if repr_layer has use_eta_channel=True — the
    same {(seg_id, step_idx): eta_progression} dict from
    precompute_eta_progression_lookup, used to inject inputs["eta_channel_values"]
    (a genuinely different mechanism from alt_progression_fracs' own ETA
    signal, which only ever feeds the mixture gate)."""
    boundaries = tuple(sorted(boundaries))
    n_bands = len(boundaries)
    band_correct = np.zeros(n_bands)
    band_total = np.zeros(n_bands)
    final_correct, final_total = 0, 0
    overall_correct, overall_total = 0, 0

    for i in tqdm(range(len(val_loader)), desc="validating (core+alt progression)", unit="batch", leave=False):
        batch_seg_ids = val_loader.batches[i]
        inputs, n_mask, labels, lengths = val_loader[i]
        if eta_channel_lookup is not None:
            n_steps = inputs["tau"].shape[1]
            inputs["eta_channel_values"] = ops.convert_to_tensor(
                eta_progression_for_batch(eta_channel_lookup, batch_seg_ids, n_steps=n_steps))
        core, alts = core_and_alt_fn(inputs, batch_seg_ids)
        dep_ids = departure_ids_fn(batch_seg_ids) if departure_ids_fn is not None else None
        with torch.no_grad():
            x = repr_layer(inputs)
            logits = model(x, key_padding_mask=n_mask, external_progression_frac=core, alt_progression_fracs=alts,
                            departure_subregion_ids=dep_ids)
        preds = ops.convert_to_numpy(ops.argmax(logits, axis=-1))
        lengths_int = lengths.astype(int)

        for b in range(preds.shape[0]):
            L = lengths_int[b]
            if L <= 0:
                continue
            for t in range(L):
                frac = (t + 1) / L
                correct = bool(preds[b, t] == labels[b])
                overall_correct += int(correct)
                overall_total += 1
                if t == L - 1:
                    final_correct += int(correct)
                    final_total += 1
                band = next(bi for bi, bnd in enumerate(boundaries) if frac <= bnd)
                band_total[band] += 1
                if correct:
                    band_correct[band] += 1

    return {
        "overall_acc": overall_correct / max(1, overall_total),
        "progression_boundaries": list(boundaries),
        "progression_labels": _progression_labels(boundaries),
        "progression_acc": band_correct / np.maximum(band_total, 1),
        "band_correct": band_correct.tolist(),
        "band_total": band_total.tolist(),
        "final_step_acc": final_correct / max(1, final_total),
        "n_val_segments": final_total,
    }


def _macro_auc_roc_partial(y_true, y_probs, labels_range):
    """One-vs-rest AUC-ROC, macro-averaged over only the classes that
    CAN actually be computed for this specific evaluation set.

    sklearn's own roc_auc_score(multi_class="ovr", average="macro")
    fails ENTIRELY -- returns NaN for the WHOLE metric, silently, no
    exception raised -- if even ONE of the given labels has zero true
    positive (or zero true negative) examples in y_true. Confirmed
    directly: a single missing class was enough to turn AUC-ROC into
    NaN for every reported band on a real evaluation, discarding valid
    information from every OTHER class that had plenty of data to
    support its own AUC. A narrow explicit test window, or the Early
    band specifically, are the most likely places for one class out of
    many to genuinely have zero examples.

    This computes each class's own binary (that class vs. rest) AUC
    SEPARATELY, silently skips any class that genuinely can't be
    computed for THIS evaluation set (no positive or no negative
    examples for that one class), and averages over whatever remains.
    Returns (auc, n_classes_used) -- auc is None only if NO class could
    be computed at all (the whole band is empty or degenerately
    single-class), not merely because one class out of many was thin.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    aucs = []
    for cls in labels_range:
        y_true_binary = (y_true == cls).astype(int)
        n_pos = int(y_true_binary.sum())
        n_neg = len(y_true_binary) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue  # can't compute AUC for this one class in this set -- skip it, don't fail the whole metric
        try:
            aucs.append(roc_auc_score(y_true_binary, y_probs[:, cls]))
        except ValueError:
            continue
    if not aucs:
        return None, 0
    return float(np.mean(aucs)), len(aucs)


def _compute_full_metrics(y_true, y_pred, y_probs, n_classes, top_k=(3, 5)):
    """sklearn-based metric bundle for one (y_true, y_pred, y_probs)
    triplet — confusion matrix, macro precision/recall/F1, standard AND
    balanced accuracy (deliberately BOTH -- see below), top-k accuracy
    for each k, and AUC-ROC (one-vs-rest, macro, computed per-class and
    partially averaged -- see _macro_auc_roc_partial above). Called
    once per voyage-stage band by evaluate_full_report_metrics below,
    not directly.

    Returns None for an empty band (nothing to compute) rather than
    raising, so a caller iterating over several bands doesn't need its
    own empty-check first.

    STANDARD vs BALANCED accuracy are genuinely different metrics, not
    two names for the same thing -- confirmed a real point of confusion
    directly: standard accuracy (sklearn's accuracy_score) is the same
    frequency-weighted "fraction correct" reported everywhere ELSE in
    this project's own ablation tables (train_residual_progression_
    variant's own metrics["overall_acc"]); balanced accuracy is macro-
    averaged PER-CLASS recall (every class weighted equally, regardless
    of how common it is). Given this project's own established class
    imbalance, balanced accuracy will almost always read LOWER than
    standard accuracy whenever the model does worse on rare classes --
    that's an expected, real property of the two metrics differing, NOT
    a sign of a bug or a worse model. Both are reported here,
    side-by-side, on the SAME evaluation set, specifically so this
    doesn't need to be re-diagnosed each time it comes up.

    AUC-ROC: see _macro_auc_roc_partial's own docstring -- computed
    per-class, only the classes with zero examples in THIS specific
    evaluation set get skipped, not the whole metric. "auc_roc_n_classes"
    reports how many of n_classes actually contributed, so a report
    reader can see directly whether the number reflects all classes or
    a partial subset. None only if literally no class could be scored.
    Same "skip, don't fail everything" treatment for top-k accuracy
    when k >= n_classes (trivially 1.0, not an error) or when too few
    classes are present for sklearn's own check.
    """
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                  accuracy_score, balanced_accuracy_score, top_k_accuracy_score)
    if len(y_true) == 0:
        return None

    labels_range = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average="macro", zero_division=0)
    std_acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    top_k_results = {}
    for k in top_k:
        if k >= n_classes:
            top_k_results[k] = 1.0  # trivially true once k covers every class
            continue
        try:
            top_k_results[k] = top_k_accuracy_score(y_true, y_probs, k=k, labels=labels_range)
        except ValueError:
            top_k_results[k] = None

    auc_roc, auc_roc_n_classes = _macro_auc_roc_partial(y_true, y_probs, labels_range)

    return {
        "n_samples": len(y_true),
        "confusion_matrix": cm,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "standard_accuracy": std_acc,
        "balanced_accuracy": bal_acc,
        "top_k_accuracy": top_k_results,
        "auc_roc_ovr_macro": auc_roc,
        "auc_roc_n_classes": auc_roc_n_classes,
    }


def _collect_full_predictions(model, repr_layer, val_loader, core_and_alt_fn,
                               departure_ids_fn=None, eta_channel_lookup=None):
    """Shared forward-pass-and-collect step for evaluate_full_report_metrics
    and evaluate_cold_start_stratified — runs the model over every batch
    in val_loader ONCE, collecting per-step (seg_id, true_label,
    predicted_label, softmax_probability_vector, progression_frac) for
    every step of every segment. Pure collection, no bucketing or
    metrics of its own — callers bucket however they need (voyage
    stage, prior-history count, or both) without re-running the model,
    so adding a second stratification never costs a second forward
    pass.

    Returns (all_seg_id, all_true, all_pred, all_probs, all_frac), all
    numpy arrays, same length (one entry per step of every segment).
    """
    all_seg_id, all_true, all_pred, all_probs, all_frac = [], [], [], [], []

    for i in tqdm(range(len(val_loader)), desc="collecting predictions", unit="batch", leave=False):
        batch_seg_ids = val_loader.batches[i]
        inputs, n_mask, labels, lengths = val_loader[i]
        if eta_channel_lookup is not None:
            n_steps = inputs["tau"].shape[1]
            inputs["eta_channel_values"] = ops.convert_to_tensor(
                eta_progression_for_batch(eta_channel_lookup, batch_seg_ids, n_steps=n_steps))
        core, alts = core_and_alt_fn(inputs, batch_seg_ids)
        dep_ids = departure_ids_fn(batch_seg_ids) if departure_ids_fn is not None else None
        with torch.no_grad():
            x = repr_layer(inputs)
            logits = model(x, key_padding_mask=n_mask, external_progression_frac=core, alt_progression_fracs=alts,
                            departure_subregion_ids=dep_ids)
            probs = ops.softmax(logits, axis=-1)
        preds = ops.convert_to_numpy(ops.argmax(logits, axis=-1))
        probs_np = ops.convert_to_numpy(probs)
        lengths_int = lengths.astype(int)

        for b in range(preds.shape[0]):
            L = lengths_int[b]
            if L <= 0:
                continue
            sid = batch_seg_ids[b]
            for t in range(L):
                frac = (t + 1) / L
                all_seg_id.append(sid)
                all_true.append(int(labels[b]))
                all_pred.append(int(preds[b, t]))
                all_probs.append(probs_np[b, t])
                all_frac.append(frac)

    return (np.array(all_seg_id), np.array(all_true), np.array(all_pred),
            np.array(all_probs), np.array(all_frac))


def evaluate_full_report_metrics(model, repr_layer, val_loader, core_and_alt_fn, n_classes,
                                  departure_ids_fn=None,
                                  progression_bands=(("Early (0-20%)", None, 0.2),
                                                      ("Mid (20-60%)", 0.2, 0.6),
                                                      ("Late (60-100%)", 0.6, None)),
                                  eta_channel_lookup=None, top_k=(3, 5)):
    """Runs the model over val_loader once (via _collect_full_predictions),
    then buckets into progression_bands — the SAME (label, lower_exclusive,
    upper_inclusive) convention train_residual_progression_variant's own
    track_progression_band_losses parameter already uses, defaulting to
    the established Early/Mid/Late split this project's accuracy-by-stage
    tables already use throughout (PLUS an "Overall" band spanning
    everything) — deliberately NOT DEFAULT_PROGRESSION_BOUNDARIES' own
    fine-grained 20-bucket convention, which exists for the
    progression-accuracy PLOTS' own x-axis, a different purpose from
    this function's own per-band metric bundle. Computes, PER BAND, the
    full metric bundle from _compute_full_metrics: confusion matrix,
    precision/recall/F1 (macro), balanced accuracy, top-k accuracy, and
    AUC-ROC (one-vs-rest, macro).

    Intended for a DELIBERATE final report, not routine ablation
    iteration — run this on the TEST set (val_loader built over
    result["test_ids"]), not val, matching evaluate_on_test's own
    reasoning: the value of a held-out test set comes specifically from
    not looking at it repeatedly while iterating.

    Returns {band_label: metrics_dict_or_None} — "Overall" is always
    the last key. A band with zero examples in this val_loader (e.g. no
    Early-stage steps at all in a very short test window) gets None,
    handled explicitly downstream, not silently skipped.
    """
    _, all_true, all_pred, all_probs, all_frac = _collect_full_predictions(
        model, repr_layer, val_loader, core_and_alt_fn,
        departure_ids_fn=departure_ids_fn, eta_channel_lookup=eta_channel_lookup)

    def _band_for_frac(frac):
        for label, lower, upper in progression_bands:
            lower_ok = (lower is None) or (frac > lower)
            upper_ok = (upper is None) or (frac <= upper)
            if lower_ok and upper_ok:
                return label
        return None  # shouldn't happen if bands fully cover [0,1], but don't silently misassign if they don't

    band_per_step = np.array([_band_for_frac(f) for f in all_frac]) if len(all_frac) else np.array([], dtype=object)

    results = {}
    for label, _lower, _upper in progression_bands:
        mask = band_per_step == label
        results[label] = _compute_full_metrics(all_true[mask], all_pred[mask], all_probs[mask],
                                                 n_classes, top_k=top_k)
    results["Overall"] = _compute_full_metrics(all_true, all_pred, all_probs, n_classes, top_k=top_k)
    return results


def train_residual_progression_variant(step3data, target_col, n_classes, condition_name,
                                        alt_progression_modes, core_progression_mode="none",
                                        gate_uses_content=True, gate_ship_history=False, ship_history_attention=False,
                                        use_recency_bias=False, use_contract_period_feature=False,
                                        use_spatial_channel=True, use_local_pattern_channel=True,
                                        use_departure_port_channel=True, use_ship_size_channel=True,
                                        use_temporal_encoding=True,
                                        use_departure_subregion_channel=False, use_eta_channel=False,
                                        n_experts=3,
                                        use_departure_gate=False, departure_embed_dim=8,
                                        use_ship_history=True, stratify=True, stratify_by_pair=False, use_declared_destination=False,
                                        epochs=6, batch_size=32, d_model=128, val_frac=0.15, test_frac=0.15, seed=42,
                                        n_casp_layers=2, n_heads_mca=2, n_heads_msa=4, d_ff=None,
                                        history_gat_layers=2, history_gat_heads=4, work_dir=None, skip_existing=True,
                                        track_progression_band_losses=False,
                                        progression_bands=(("Early (0-20%)", None, 0.2),
                                                            ("Mid (20-60%)", 0.2, 0.6),
                                                            ("Late (60-100%)", 0.6, None)),
                                        dropout_rate=0.0, weight_decay=0.0, early_stopping_patience=None,
                                        train_ids_override=None, val_ids_override=None,
                                        use_fixed_horizon_fleet_context=False,
                                        use_active_vessel_set_context=False,
                                        weight_fixed_horizon_by_similarity=False,
                                        fixed_horizon_size_sigma=None, fixed_horizon_draught_sigma=None,
                                        fixed_horizon_weight_combination="multiplicative",
                                        test_start=None, test_end=None,
                                        active_vessel_truncation_mode="nearest",
                                        n_subregions_active_vessel=None,
                                        active_vessel_include_temporal_history=True,
                                        active_vessel_use_similarity_bias=False,
                                        active_vessel_size_sigma=None, active_vessel_draught_sigma=None,
                                        active_vessel_distance_sigma_km=None, active_vessel_duration_sigma_days=None,
                                        active_vessel_history_sigmas=None,
                                        active_vessel_eta_channel_lookup=None, active_vessel_history_index=None,
                                        active_vessel_port_to_subregion=None, active_vessel_history_stats_cache=None):
    """Trains a MoE model where each alternative progression signal in
    alt_progression_modes gets its OWN residual-gated path (learned scale
    starting at exactly 0 — see MixtureOfExpertsFeedForward), rather than
    being fed in unconditionally the way train_progression_signal_variant's
    single-signal experiment did. Directly targets the specific weakness
    found there: historical_avg_port actively hurt accuracy in the
    earliest bands, plausibly because it was forced in with no way for
    the model to discount it exactly where it's least reliable.

    core_progression_mode: the "always-on" signal (default "none" — the
    constant, proven-safest option from the single-signal experiment,
    carrying zero positional information on its own). alt_progression_modes:
    list of modes (e.g. ["historical_avg_port"] for a single residual-gated
    alternative, ["historical_avg_port", "raw_elapsed"] to let the model
    weigh multiple candidates simultaneously, each independently). Modes
    are the same as compute_alternative_progression's: "historical_avg_port",
    "historical_avg_finer", "raw_elapsed" (not "true" or "none" — those
    don't make sense as an alt-signal choice; use core_progression_mode
    for "none").

    gate_ship_history=True combines with the above to also let the model
    learn how much to trust ship history (same residual mechanism, applied
    to the input channel rather than the gate) — testing whichever
    progression setup wins alongside that fix, not as a separate
    standalone experiment.

    track_progression_band_losses=False (default, zero cost/behavior
    change for every existing caller): when True, ALSO tracks train and
    val loss per epoch, restricted to each of progression_bands (default
    Early 0-20% / Mid 20-60% / Late 60-100%, matching this project's
    established zone convention) — via way_loss's own existing
    max_progression_frac/min_progression_frac masking, reusing the SAME
    logits already computed each batch (no extra forward pass, just a
    few cheap extra loss evaluations). Returned as
    result["train_history_by_band"]/["val_history_by_band"], each
    {band_label: [loss_per_epoch]} — feed directly to
    plot_train_val_loss_by_band for an overfitting-risk-by-voyage-stage
    view, distinct from the single aggregate train/val curve
    history/val_history already provides.

    stratify_by_pair=False (default, unchanged behavior): see _make_split's
    own docstring for the full explanation. False stratifies by
    target_col alone (the arrival subregion being predicted) -- NOT "by
    load subregion", confirmed directly from the code, a common
    misreading of what stratify=True actually does here. True stratifies
    by the (departure subregion, target_col) PAIR instead, a strictly
    finer partition -- each trade lane gets its own proportional
    train/val split, not just its arrival side. Requires stratify=True.

    train_ids_override / val_ids_override (default None, None -- unchanged
    behavior for every existing caller): both or neither. When given,
    BYPASSES _make_split entirely -- no recency-based test cutoff, no
    stratification, just uses these two sets of seg_ids directly as
    train/val. For a genuinely custom split, e.g. a regime-shift holdout
    (train on everything strictly before some real-world event, evaluate
    on segments during/after it) -- see build_event_slice_ids for
    constructing exactly this kind of split. test_frac/val_frac/stratify/
    stratify_by_pair are all ignored when these are set.

    REGULARIZATION — none of the below were present at all before these
    three were added (confirmed directly: no Dropout layer anywhere in
    RepresentationLayer/CASPLayer, no weight_decay on any optimizer, no
    early stopping, no label smoothing — gradient_dropout_weights is a
    deterministic sequence-length loss-reweighting scheme despite its
    name, not a dropout mechanism at all). All three default to their
    off/no-op value, so nothing changes for any existing caller unless
    explicitly requested:

    dropout_rate=0.0: standard Transformer dropout placement (Vaswani et
    al.) — applied to each CASP sub-layer's OUTPUT (MCA/MSA/feed-forward)
    before it's added into the residual stream. keras Dropout at rate
    0.0 is a no-op, confirmed directly. Propagated through WAYModel to
    every CASPLayer. A reasonable starting value if you turn this on:
    0.1.

    weight_decay=0.0: L2-style regularization via the optimizer.
    weight_decay=0.0 (default) keeps plain torch.optim.Adam, exactly as
    before this option existed. weight_decay>0 switches the optimizer
    CLASS itself to torch.optim.AdamW (not just passing weight_decay to
    Adam, which implements L2 differently and less effectively than
    AdamW's decoupled version) — a reasonable starting value: 1e-4 to
    1e-5.

    early_stopping_patience=None: no early stopping (default, current
    behavior — always runs the full, fixed epochs count). Set to an
    integer N to stop once N consecutive epochs pass with no new best
    val_loss. Does NOT roll back to the best epoch's own weights — the
    model at stop time is whatever the LAST epoch trained (patience
    epochs past the best one). Combine with track_progression_band_losses
    freely; per-band history simply ends at the same, possibly-earlier
    epoch as the aggregate one.
    """
    d_ff = d_ff or d_model * 2
    n_alt = len(alt_progression_modes)
    n_departure_subregions = len(step3data.vocab["port_subregion_to_id"]) if use_departure_gate else None
    if gate_ship_history and not use_ship_history:
        raise ValueError("gate_ship_history=True requires use_ship_history=True — there's no ship-history "
                          "channel to gate if the channel itself is disabled")
    if use_recency_bias and not use_ship_history:
        raise ValueError("use_recency_bias=True requires use_ship_history=True — there's no ship-history "
                          "pooling step to bias if the channel itself is disabled")
    if use_contract_period_feature and not use_ship_history:
        raise ValueError("use_contract_period_feature=True requires use_ship_history=True — there's no "
                          "ship-history node feature to add if the channel itself is disabled")
    if use_eta_channel and work_dir is None:
        raise ValueError("use_eta_channel=True requires work_dir (used to read the raw ETA field from "
                          "trajectories_gridded.parquet)")

    # Built once, needed by BOTH the load path and the fresh-training path
    # below.
    n_subregions_departure = None
    dep_port_to_subregion_lookup = None
    if use_departure_subregion_channel:
        subregion_map = build_port_to_subregion_map(step3data.traj_idx)  # {port_id: subregion_id}
        n_subregions_departure = len(step3data.vocab["port_subregion_to_id"])
        # Dense array indexed by port id, length n_ports+1 (matches
        # port_embed's own +1 for the NONE_DECLARED id) -- any port never
        # seen as an arrival (so absent from subregion_map) falls back to
        # subregion 0, a minor, rare edge case rather than a crash.
        dep_port_to_subregion_lookup = [
            subregion_map.get(p, 0) for p in range(step3data.n_ports + 1)
        ]

    eta_channel_lookup = precompute_eta_progression_lookup(work_dir, step3data.traj_idx) if use_eta_channel else None

    # Moved here (before the load-from-disk check below) so BOTH the
    # load path and the fresh-training path end up with the SAME
    # objects available for result["core_and_alt_fn"]/["departure_ids_fn"]/
    # ["val_loader"] -- previously these were only built in the
    # fresh-training branch, so a caller wanting plot_moe_gate_weights_full
    # on an ALREADY-TRAINED, loaded-from-disk model (the common case once
    # any ablation has been run once) would silently not have them. The
    # per-call cost added to the load path is index construction only
    # (groupby/aggregation over traj_idx), not training -- modest even
    # for this project's dataset sizes.
    # [inlined -- L4e defines these] from Step4e_fleet_context import DepartureDurationIndex, FineDurationIndex
    needs_coarse = "historical_avg_port" in alt_progression_modes or "historical_avg_finer" in alt_progression_modes
    duration_index = DepartureDurationIndex(step3data.traj_idx) if needs_coarse else None
    fine_duration_index = (FineDurationIndex(step3data.traj_idx, duration_index)
                            if "historical_avg_finer" in alt_progression_modes else None)
    dep_month_lookup = pd.to_datetime(step3data.traj_idx.set_index("seg_id")["dep_ts"]).dt.month.to_dict()

    eta_lookup = None
    if "eta" in alt_progression_modes or core_progression_mode == "eta":
        if work_dir is None:
            raise ValueError("mode='eta' needs work_dir (ETA lives in trajectories_gridded.parquet on disk, "
                              "not in traj_idx)")
        eta_lookup = precompute_eta_progression_lookup(work_dir, step3data.traj_idx)

    def _compute_signal(mode, inputs, seg_ids):
        if mode == "eta":
            tau_shape = ops.convert_to_numpy(inputs["tau"]).shape
            return eta_progression_for_batch(eta_lookup, seg_ids, n_steps=tau_shape[1])
        tau = ops.convert_to_numpy(inputs["tau"])
        dep_port_ids = ops.convert_to_numpy(inputs["dep_port_id"])
        size_class_ids = ops.convert_to_numpy(inputs["size_class_id"])
        dep_months = np.array([dep_month_lookup.get(s, 1) for s in seg_ids])
        return compute_alternative_progression(
            mode, tau, dep_port_ids=dep_port_ids, size_class_ids=size_class_ids, dep_months=dep_months,
            duration_index=duration_index, fine_duration_index=fine_duration_index)

    def _core_and_alt_for_batch(inputs, seg_ids):
        core = _compute_signal(core_progression_mode, inputs, seg_ids) if core_progression_mode != "true" else None
        alts = [_compute_signal(m, inputs, seg_ids) for m in alt_progression_modes] if n_alt > 0 else None
        return core, alts

    seg_to_dep_subregion = None
    if use_departure_gate:
        port_to_subregion = build_port_to_subregion_map(step3data.traj_idx, subregion_col=target_col, port_col="ARR_PORT_ID")
        seg_to_dep_subregion = step3data.traj_idx.set_index("seg_id")["DEP_PORT_ID"].map(port_to_subregion)

    def _departure_ids_for_batch(seg_ids):
        out = []
        for s in seg_ids:
            val = seg_to_dep_subregion.get(s, np.nan)
            out.append(int(val) if pd.notna(val) else 0)  # 0 as a safe fallback for an unmappable port
        return np.array(out, dtype="int32")

    if (train_ids_override is None) != (val_ids_override is None):
        raise ValueError("train_ids_override and val_ids_override must be given together (both or neither) -- "
                          "a custom split needs both sides specified explicitly, not just one.")
    if use_fixed_horizon_fleet_context and step3data.fixed_horizon_fleet_index is None:
        raise ValueError("use_fixed_horizon_fleet_context=True but build_fixed_horizon_fleet_index() "
                          "hasn't been called yet -- call it once after enrich_arrival_labels(data).")
    if use_active_vessel_set_context and step3data.active_vessel_set_index is None:
        raise ValueError("use_active_vessel_set_context=True but build_active_vessel_set_index() "
                          "hasn't been called yet -- call it once after enrich_arrival_labels(data).")
    n_subregions_fixed_horizon = (
        step3data.fixed_horizon_fleet_index.n_subregions * len(step3data.fixed_horizon_days)
        if use_fixed_horizon_fleet_context else None)
    if train_ids_override is not None:
        # Bypasses _make_split's own recency/stratification logic entirely
        # -- for a genuinely custom split (e.g. a regime-shift holdout:
        # train on everything strictly before some event, evaluate on
        # segments during/after it). No temporal test set is carved out
        # here since that's not this call's job -- the caller already
        # decided what train/val should be.
        train_ids, val_ids, test_ids = set(train_ids_override), set(val_ids_override), set()
    else:
        train_ids, val_ids, test_ids = _make_split(
            step3data, target_col, val_frac=val_frac, test_frac=test_frac, seed=seed, stratify=stratify,
            stratify_by_pair=stratify_by_pair, test_start=test_start, test_end=test_end)
    val_loader = BucketedWAYDataset(
        step3data, target_col=target_col, batch_size=batch_size, seg_id_subset=val_ids,
        shuffle=False, seed=seed, include_ship_history=use_ship_history,
        use_contract_period_feature=use_contract_period_feature,
        include_fixed_horizon_fleet_context=use_fixed_horizon_fleet_context,
        include_active_vessel_set_context=use_active_vessel_set_context,
        weight_fixed_horizon_by_similarity=weight_fixed_horizon_by_similarity,
        fixed_horizon_size_sigma=fixed_horizon_size_sigma,
        fixed_horizon_weight_combination=fixed_horizon_weight_combination,
        fixed_horizon_draught_sigma=fixed_horizon_draught_sigma,
        active_vessel_truncation_mode=active_vessel_truncation_mode,
        active_vessel_eta_channel_lookup=active_vessel_eta_channel_lookup,
        active_vessel_history_index=active_vessel_history_index,
        active_vessel_port_to_subregion=active_vessel_port_to_subregion,
        active_vessel_history_stats_cache=active_vessel_history_stats_cache)
    departure_ids_fn = _departure_ids_for_batch if use_departure_gate else None

    if work_dir is not None and skip_existing and os.path.exists(_fast_fleet_result_path(work_dir, target_col, condition_name)):
        print(f"{condition_name} — found on disk, loading weights instead of retraining")
        result = load_fast_fleet_result(work_dir, target_col, condition_name)
        repr_layer, model = load_trained_model(
            _regime_weights_path(work_dir, target_col, condition_name),
            n_ports=step3data.n_ports, n_size_classes=step3data.n_size_classes, n_classes=n_classes,
            d_model=d_model, n_casp_layers=n_casp_layers, n_heads_mca=n_heads_mca, n_heads_msa=n_heads_msa,
            d_ff=d_ff, use_spatial_channel=use_spatial_channel, use_local_pattern_channel=use_local_pattern_channel,
            use_departure_port_channel=use_departure_port_channel, use_ship_size_channel=use_ship_size_channel,
            use_temporal_encoding=use_temporal_encoding,
            use_declared_destination=use_declared_destination, use_ship_history=use_ship_history, gate_ship_history=gate_ship_history,
            ship_history_attention=ship_history_attention, use_recency_bias=use_recency_bias,
            use_contract_period_feature=use_contract_period_feature,
            use_departure_subregion_channel=use_departure_subregion_channel,
            n_subregions_departure=n_subregions_departure,
            dep_port_to_subregion_lookup=dep_port_to_subregion_lookup,
            use_eta_channel=use_eta_channel,
            use_fixed_horizon_fleet_context=use_fixed_horizon_fleet_context,
            n_subregions_fixed_horizon=n_subregions_fixed_horizon,
            use_active_vessel_set_context=use_active_vessel_set_context,
            history_gat_layers=history_gat_layers, history_gat_heads=history_gat_heads,
            use_moe_ffn=True, n_experts=n_experts, gate_uses_content=gate_uses_content,
            n_alt_progression_signals=n_alt, use_departure_gate=use_departure_gate,
            n_departure_subregions=n_departure_subregions, departure_embed_dim=departure_embed_dim)
        if "band_correct" not in result["metrics"]:
            # This condition_name's saved JSON predates band_correct/
            # band_total being added to _evaluate_with_core_and_alt_
            # progression's own return dict -- e.g. the real, deployed
            # final model (r7), trained long before this project's
            # accuracy-by-stage tooling existed. Confirmed directly this
            # actually happens, not just a theoretical edge case: loading
            # r7 for a stage-table summary raised exactly this KeyError.
            # The model's own WEIGHTS are unaffected (nothing here
            # retrains) -- only metrics needs a cheap refresh, a single
            # forward pass over val_loader, not full training. Re-saved
            # once so every later load of this same condition_name is
            # already up to date, no repeated re-evaluation cost.
            print(f"    {condition_name} — saved metrics predate band_correct/band_total, refreshing "
                  f"(re-evaluating on val_loader, NOT retraining) ...")
            result["metrics"] = _evaluate_with_core_and_alt_progression(
                model, repr_layer, val_loader, _core_and_alt_for_batch,
                departure_ids_fn=departure_ids_fn, eta_channel_lookup=eta_channel_lookup)
            _save_fast_fleet_result(work_dir, target_col, condition_name, result)
        result["repr_layer"], result["model"] = repr_layer, model
        result["val_loader"] = val_loader
        result["core_and_alt_fn"] = _core_and_alt_for_batch
        result["departure_ids_fn"] = departure_ids_fn
        result["eta_channel_lookup"] = eta_channel_lookup
        result["test_ids"] = test_ids
        return result

    train_loader = BucketedWAYDataset(
        step3data, target_col=target_col, batch_size=batch_size, seg_id_subset=train_ids,
        shuffle=True, seed=seed, include_ship_history=use_ship_history,
        use_contract_period_feature=use_contract_period_feature,
        include_fixed_horizon_fleet_context=use_fixed_horizon_fleet_context,
        include_active_vessel_set_context=use_active_vessel_set_context,
        weight_fixed_horizon_by_similarity=weight_fixed_horizon_by_similarity,
        fixed_horizon_size_sigma=fixed_horizon_size_sigma,
        fixed_horizon_weight_combination=fixed_horizon_weight_combination,
        fixed_horizon_draught_sigma=fixed_horizon_draught_sigma,
        active_vessel_truncation_mode=active_vessel_truncation_mode,
        active_vessel_eta_channel_lookup=active_vessel_eta_channel_lookup,
        active_vessel_history_index=active_vessel_history_index,
        active_vessel_port_to_subregion=active_vessel_port_to_subregion,
        active_vessel_history_stats_cache=active_vessel_history_stats_cache)

    # seed controls the split above (via _make_split/loaders) — but NOT,
    # until now, the model's own weight initialization, since nothing in
    # this project ever called any seeding function before model
    # construction. That meant "same seed" only ever guaranteed "same
    # data partition" — two runs with identical seed still got two
    # different, uncontrolled random initializations, which could
    # plausibly explain a real, non-trivial accuracy gap on its own with
    # only a few epochs of training. keras.utils.set_random_seed(seed) —
    # NOT torch.manual_seed(seed) alone, which was tried first and
    # verified insufficient: two models built with identical
    # torch.manual_seed() calls still had DIFFERENT initial weights,
    # since Keras 3's own lazy weight-building draws on more than just
    # PyTorch's own RNG internally. keras.utils.set_random_seed() is
    # Keras's own comprehensive seeding utility and was verified directly
    # (not assumed) to produce bit-identical initial weights across
    # independently-constructed models before adopting it here. Placed
    # HERE (after the split/loaders, right before construction) so it
    # controls ONLY weight init, not data partitioning or shuffle order.
    keras.utils.set_random_seed(seed)

    active_vessel_sigma_kwargs = {}
    if active_vessel_size_sigma is not None:
        active_vessel_sigma_kwargs["active_vessel_size_sigma"] = active_vessel_size_sigma
    if active_vessel_draught_sigma is not None:
        active_vessel_sigma_kwargs["active_vessel_draught_sigma"] = active_vessel_draught_sigma
    if active_vessel_distance_sigma_km is not None:
        active_vessel_sigma_kwargs["active_vessel_distance_sigma_km"] = active_vessel_distance_sigma_km
    if active_vessel_duration_sigma_days is not None:
        active_vessel_sigma_kwargs["active_vessel_duration_sigma_days"] = active_vessel_duration_sigma_days
    if active_vessel_history_sigmas is not None:
        active_vessel_sigma_kwargs["active_vessel_history_sigmas"] = active_vessel_history_sigmas
    repr_layer = RepresentationLayer(
        d_model, step3data.n_ports, step3data.n_size_classes,
        use_spatial_channel=use_spatial_channel, use_local_pattern_channel=use_local_pattern_channel,
        use_departure_port_channel=use_departure_port_channel, use_ship_size_channel=use_ship_size_channel,
        use_temporal_encoding=use_temporal_encoding,
        use_declared_destination=use_declared_destination, use_ship_history=use_ship_history, gate_ship_history=gate_ship_history,
        ship_history_attention=ship_history_attention, use_recency_bias=use_recency_bias,
        use_contract_period_feature=use_contract_period_feature,
        use_departure_subregion_channel=use_departure_subregion_channel,
        n_subregions_departure=n_subregions_departure,
        dep_port_to_subregion_lookup=dep_port_to_subregion_lookup,
        use_eta_channel=use_eta_channel,
        use_fixed_horizon_fleet_context=use_fixed_horizon_fleet_context,
        n_subregions_fixed_horizon=n_subregions_fixed_horizon,
        use_active_vessel_set_context=use_active_vessel_set_context,
        n_subregions_active_vessel=n_subregions_active_vessel,
        active_vessel_include_temporal_history=active_vessel_include_temporal_history,
        active_vessel_use_similarity_bias=active_vessel_use_similarity_bias,
        history_gat_layers=history_gat_layers, history_gat_heads=history_gat_heads,
        **active_vessel_sigma_kwargs)
    model = WAYModel(d_model, n_classes, n_layers=n_casp_layers, n_heads_mca=n_heads_mca,
                      n_heads_msa=n_heads_msa, d_ff=d_ff, use_moe_ffn=True, n_experts=n_experts,
                      gate_uses_content=gate_uses_content, n_alt_progression_signals=n_alt,
                      use_departure_gate=use_departure_gate, n_departure_subregions=n_departure_subregions,
                      departure_embed_dim=departure_embed_dim, dropout_rate=dropout_rate)

    dummy_ids = list(train_ids)[:2]
    dummy_inputs, dummy_mask = step3data.prepare_batch(dummy_ids, include_ship_history=use_ship_history,
                                                         use_contract_period_feature=use_contract_period_feature,
                                                         include_fixed_horizon_fleet_context=use_fixed_horizon_fleet_context,
                                                         include_active_vessel_set_context=use_active_vessel_set_context,
                                                         weight_fixed_horizon_by_similarity=weight_fixed_horizon_by_similarity,
                                                         fixed_horizon_size_sigma=fixed_horizon_size_sigma,
                                                         fixed_horizon_weight_combination=fixed_horizon_weight_combination,
                                                         fixed_horizon_draught_sigma=fixed_horizon_draught_sigma,
                                                         active_vessel_truncation_mode=active_vessel_truncation_mode,
                                                         active_vessel_eta_channel_lookup=active_vessel_eta_channel_lookup,
                                                         active_vessel_history_index=active_vessel_history_index,
                                                         active_vessel_port_to_subregion=active_vessel_port_to_subregion,
                                                         active_vessel_history_stats_cache=active_vessel_history_stats_cache)
    if use_eta_channel:
        dummy_inputs["eta_channel_values"] = ops.convert_to_tensor(
            eta_progression_for_batch(eta_channel_lookup, dummy_ids, n_steps=dummy_inputs["tau"].shape[1]))
    dummy_core, dummy_alts = _core_and_alt_for_batch(dummy_inputs, dummy_ids)
    dummy_dep_ids = _departure_ids_for_batch(dummy_ids) if use_departure_gate else None
    _ = model(repr_layer(dummy_inputs), key_padding_mask=dummy_mask,
              external_progression_frac=dummy_core, alt_progression_fracs=dummy_alts,
              departure_subregion_ids=dummy_dep_ids)
    # weight_decay=0.0 (default): plain Adam, EXACTLY as before this
    # option existed -- torch.optim.AdamW with weight_decay=0.0 is close
    # to but not guaranteed bit-identical to torch.optim.Adam in every
    # edge case, so the optimizer CLASS itself only changes when
    # weight_decay is actually requested, guaranteeing zero behavior
    # change for every existing caller rather than relying on the two
    # classes agreeing at the zero-decay limit.
    params = [v.value for v in repr_layer.trainable_variables + model.trainable_variables]
    if weight_decay > 0:
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=1e-3)

    start_epoch, train_hist, val_hist = 0, [], []
    train_hist_by_band = {label: [] for label, _, _ in progression_bands} if track_progression_band_losses else None
    val_hist_by_band = {label: [] for label, _, _ in progression_bands} if track_progression_band_losses else None
    if work_dir is not None:
        resumed = _load_epoch_checkpoint(work_dir, target_col, condition_name, repr_layer, model)
        if resumed is not None:
            start_epoch, train_hist, val_hist = resumed
            print(f"    [epoch checkpoint] resuming from epoch {start_epoch+1}/{epochs} ({start_epoch} already completed)")
            # NOTE: per-band history does NOT resume from an epoch
            # checkpoint (only the aggregate train_hist/val_hist does,
            # via the existing mechanism) -- resuming with
            # track_progression_band_losses=True starts per-band
            # tracking fresh from start_epoch, so its own history will
            # be shorter than train_hist/val_hist in that specific case.
            # Uncommon in practice (this flag is for one-off diagnostic
            # runs, not routine training), but worth knowing.

    # early_stopping_patience=None (default): no early stopping, EXACTLY
    # current behavior -- runs the full, fixed epochs count regardless of
    # val_hist. When set to an integer N, stops (breaks out of the loop
    # early, does NOT retroactively undo already-completed epochs) once
    # N consecutive epochs have passed with no new best val loss. Does
    # NOT restore the best epoch's own weights automatically -- the
    # model's weights at break time are whatever the LAST epoch trained
    # (patience epochs past the best one), same convention the epoch
    # checkpoint mechanism already uses elsewhere (most recent, not
    # best). Track_progression_band_losses interacts fine with this --
    # per-band history simply ends at the same, possibly-earlier epoch
    # as the aggregate history.
    best_val_loss = min(val_hist) if val_hist else float("inf")
    epochs_since_best = 0

    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        epoch_losses_by_band = {label: [] for label, _, _ in progression_bands} if track_progression_band_losses else None
        for i in tqdm(range(len(train_loader)), desc=f"  {condition_name} epoch {epoch+1}/{epochs}", leave=False):
            batch_seg_ids = train_loader.batches[i]
            inputs, n_mask, labels, lengths = train_loader[i]
            if use_eta_channel:
                inputs["eta_channel_values"] = ops.convert_to_tensor(
                    eta_progression_for_batch(eta_channel_lookup, batch_seg_ids, n_steps=inputs["tau"].shape[1]))
            core, alts = _core_and_alt_for_batch(inputs, batch_seg_ids)
            dep_ids = _departure_ids_for_batch(batch_seg_ids) if use_departure_gate else None
            x = repr_layer(inputs)
            logits = model(x, key_padding_mask=n_mask, external_progression_frac=core, alt_progression_fracs=alts,
                            departure_subregion_ids=dep_ids, training=True)
            gd = gradient_dropout_weights(lengths)
            loss = way_loss(logits, labels, n_mask, gd_weights=gd)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(float(loss.detach()))
            if track_progression_band_losses:
                # Reuses the SAME logits already computed above -- no
                # extra forward pass, just 3 additional cheap way_loss
                # calls with different progression masking. Computed
                # AFTER opt.step() (weights already updated for this
                # batch), same convention as epoch_losses.append just
                # above -- purely diagnostic, not part of optimization.
                with torch.no_grad():
                    for label, min_frac, max_frac in progression_bands:
                        band_loss = way_loss(logits, labels, n_mask, gd_weights=gd,
                                              max_progression_frac=max_frac, min_progression_frac=min_frac)
                        epoch_losses_by_band[label].append(float(band_loss))
        train_hist.append(float(np.mean(epoch_losses)))
        if track_progression_band_losses:
            for label in train_hist_by_band:
                train_hist_by_band[label].append(float(np.mean(epoch_losses_by_band[label])))

        val_losses = []
        val_losses_by_band = {label: [] for label, _, _ in progression_bands} if track_progression_band_losses else None
        for i in range(len(val_loader)):
            batch_seg_ids = val_loader.batches[i]
            inputs, n_mask, labels, lengths = val_loader[i]
            if use_eta_channel:
                inputs["eta_channel_values"] = ops.convert_to_tensor(
                    eta_progression_for_batch(eta_channel_lookup, batch_seg_ids, n_steps=inputs["tau"].shape[1]))
            core, alts = _core_and_alt_for_batch(inputs, batch_seg_ids)
            dep_ids = _departure_ids_for_batch(batch_seg_ids) if use_departure_gate else None
            with torch.no_grad():
                x = repr_layer(inputs)
                logits = model(x, key_padding_mask=n_mask, external_progression_frac=core, alt_progression_fracs=alts,
                                departure_subregion_ids=dep_ids)
                gd = gradient_dropout_weights(lengths)
                val_losses.append(float(way_loss(logits, labels, n_mask, gd_weights=gd)))
                if track_progression_band_losses:
                    for label, min_frac, max_frac in progression_bands:
                        band_loss = way_loss(logits, labels, n_mask, gd_weights=gd,
                                              max_progression_frac=max_frac, min_progression_frac=min_frac)
                        val_losses_by_band[label].append(float(band_loss))
        val_hist.append(float(np.mean(val_losses)))
        if track_progression_band_losses:
            for label in val_hist_by_band:
                val_hist_by_band[label].append(float(np.mean(val_losses_by_band[label])))

        print(f"    {condition_name} epoch {epoch+1}/{epochs}: train_loss={train_hist[-1]:.4f}  val_loss={val_hist[-1]:.4f}")
        if work_dir is not None:
            _save_epoch_checkpoint(work_dir, target_col, condition_name, repr_layer, model,
                                    completed_epochs=epoch + 1, train_hist=train_hist, val_hist=val_hist)

        if early_stopping_patience is not None:
            if val_hist[-1] < best_val_loss:
                best_val_loss = val_hist[-1]
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= early_stopping_patience:
                    print(f"    [early stopping] no val_loss improvement for {early_stopping_patience} epochs "
                          f"(best={best_val_loss:.4f}) -- stopping after epoch {epoch+1}/{epochs}")
                    break

    metrics = _evaluate_with_core_and_alt_progression(model, repr_layer, val_loader, _core_and_alt_for_batch,
                                                       departure_ids_fn=departure_ids_fn, eta_channel_lookup=eta_channel_lookup)

    result = {"history": train_hist, "val_history": val_hist, "metrics": metrics}
    if track_progression_band_losses:
        # Included BEFORE the save below, so a future load of this exact
        # condition_name brings this history back too, not just the
        # aggregate train_hist/val_hist.
        result["train_history_by_band"] = train_hist_by_band
        result["val_history_by_band"] = val_hist_by_band
    if work_dir is not None:
        _clear_epoch_checkpoint(work_dir, target_col, condition_name)
        _save_fast_fleet_result(work_dir, target_col, condition_name, result)
        save_trained_weights(_regime_weights_path(work_dir, target_col, condition_name), repr_layer, model)
        print(f"    saved weights -> {_regime_weights_path(work_dir, target_col, condition_name)}")
    result["repr_layer"], result["model"] = repr_layer, model
    result["val_loader"] = val_loader
    result["core_and_alt_fn"] = _core_and_alt_for_batch
    result["departure_ids_fn"] = departure_ids_fn
    result["eta_channel_lookup"] = eta_channel_lookup
    result["test_ids"] = test_ids
    return result


def precompute_eta_progression_lookup(work_dir, traj_idx):
    """Precomputes a {(seg_id, step_idx): eta_inferred_progression} lookup
    from the raw gridded AIS ETA field — SAME parsing as build_eta_comparison
    (year-inference with wraparound handling, sentinel filtering), just
    keyed by step for direct use inside a training loop rather than
    returned as an analysis DataFrame. Segments/steps with no valid ETA
    reading at all are simply absent from the lookup — callers should
    fall back to a neutral default (0.5, matching the "none" signal) for
    any (seg_id, step_idx) not found here, since a genuinely missing
    reading isn't the same as "the captain reported an ETA equal to the
    departure time" or any other real value.
    """
    eta_df = build_eta_comparison(work_dir, traj_idx)
    return {
        (int(row.SEG_ID), int(row.STEP_IDX)): float(row.eta_inferred_progression)
        for row in eta_df.itertuples()
    }


def eta_progression_for_batch(eta_lookup, seg_ids, n_steps, fallback=0.5):
    """Builds a [len(seg_ids), n_steps] array from eta_lookup for one
    training/eval batch — looks up (seg_id, step_idx) for every real
    position, using fallback (default 0.5, the same neutral constant as
    the "none" signal) for any (seg_id, step_idx) with no valid ETA
    reading (missing entirely, or filtered out as a sentinel/unparseable
    value upstream)."""
    out = np.full((len(seg_ids), n_steps), fallback, dtype="float32")
    for b, seg_id in enumerate(seg_ids):
        for t in range(n_steps):
            val = eta_lookup.get((int(seg_id), t))
            if val is not None:
                out[b, t] = val
    return out
