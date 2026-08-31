# =============================================================================
# E1 — Benchmark vs captain-declared destination (+ final main plot, combined model)
# Migrated verbatim from Main_forGitHub.ipynb cells [67, 68, 69, 70, 71, 72, 73].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 67]
# ----------------------------------------------------------------------
# =============================================================================
# E1 LIB CELL -- captain-benchmark functions (4 defs, verbatim, live Step4c)
# _build_captain_declared_lookup (the dedup_strategy machinery),
# build_model_vs_captain_combined_accuracy, summarize_accuracy_by_stage,
# plot_three_way_benchmark. Everything else they need is already defined
# by earlier notebook cells (verified by code-only closure).
# =============================================================================
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _build_captain_declared_lookup(step3data, work_dir, subregion_name_map, target_col="ARR_SUBREGION_ID",
                                    dedup_strategy="most_recent"):
    """Shared helper: builds the (seg_id, step_idx) -> declared_subregion/
    true_subregion/progression_frac table both
    build_captain_declared_subregion_accuracy and
    build_model_vs_captain_combined_accuracy need — extracted so the two
    stay consistent by construction rather than by two independently
    -written, potentially-diverging copies of the same filtering logic.
    Returns the filtered DataFrame directly (same exclusion rules as
    before: missing declaration, unmappable declared port, and unknown
    true subregion are all excluded, not scored as wrong).

    dedup_strategy — a single grid-step can have more than one raw AIS
    ping (mk > 1). The model's own StepwiseGRU processes ALL of them as
    a sequence (confirmed directly: local_declared_dest_id keeps the
    full mk dimension all the way through its own embedding lookup,
    concatenated with the other per-point features before the GRU —
    the model never reduces to one pre-selected ping either); there's
    no single "the" value it picks. For this simple ground-truth
    metric, FOUR genuinely different, defensible choices exist, each
    answering a different question:

      "most_recent" (default): the LAST declaration by TIMESTAMP within
      each step — the best-informed view BY THE END of that step. Still
      carries a flavor of hindsight: it's not necessarily what was
      actually available at any given MOMENT during the step, only what
      was true by its close. Exactly one row per (SEG_ID, STEP_IDX).

      "first": the EARLIEST declaration by TIMESTAMP within each step —
      a genuinely no-lookahead, "what would I have known checking in
      right as this step began" view. Exactly one row per
      (SEG_ID, STEP_IDX).

      "majority_vote": the declared value with the MOST supporting
      pings within the step — closer in spirit to what a GRU might
      learn to do (weigh the more heavily-repeated signal), though NOT
      the same computation; a GRU's own gating is a learned, weighted,
      nonlinear combination, not a literal vote. Ties (equally-common
      candidate values) are broken by whichever candidate's own most
      recent supporting ping is latest. NOT guaranteed to match
      "most_recent" — a step with 3 early wrong pings and 1 late right
      one has a majority-wrong, most-recent-right split; confirmed
      directly with exactly this construction below. Exactly one row
      per (SEG_ID, STEP_IDX).

      "all_pings": NO deduplication at all — every raw ping scored as
      its own independent observation. Avoids picking a single moment
      within the step entirely; multiple pings in the same step
      naturally share the same progression_frac (same STEP_IDX), so
      they land in the same voyage-stage band and contribute several
      observations to it rather than one. This is the most honest
      answer to "how accurate does a captain's declaration tend to be,
      on average, across whatever moments happen to have a raw ping" —
      neither systematically flattering (like "most_recent") nor
      systematically pessimistic (like "first" would be if declarations
      genuinely firm up over a step).

      None of the four is simply "correct" — they answer different
      questions, and reasonable people could prefer different ones
      depending on what's being asked. Confirmed directly this choice
      is far from cosmetic: on one real test window, "most_recent"
      (via an earlier, implicit, accidental version of this same
      selection) read 84.4% where "all_pings" read 76.4% -- an 8-point
      gap from this choice alone, not from anything else changing.
    """
    if dedup_strategy not in ("most_recent", "first", "majority_vote", "all_pings"):
        raise ValueError(f"dedup_strategy must be 'most_recent', 'first', 'majority_vote', or 'all_pings', "
                          f"got {dedup_strategy!r}")

    gridded_path = os.path.join(work_dir, Step3b_representation_layer.DATA_SUBFOLDER, "trajectories_gridded.parquet")
    if not os.path.exists(gridded_path):
        raise FileNotFoundError(f"{gridded_path} not found -- DECLARED_DEST_PORT_ID lives only in the raw "
                                 f"gridded file, not in traj_idx")

    import pyarrow.parquet as pq
    available_cols = pq.ParquetFile(gridded_path).schema.names
    if "DECLARED_DEST_PORT_ID" not in available_cols:
        raise ValueError(f"{gridded_path} has no DECLARED_DEST_PORT_ID column -- cannot compute captain accuracy")
    has_timestamp = "TIMESTAMP" in available_cols

    read_cols = ["SEG_ID", "STEP_IDX", "DECLARED_DEST_PORT_ID"] + (["TIMESTAMP"] if has_timestamp else [])
    gridded = pd.read_parquet(gridded_path, columns=read_cols)
    gridded = gridded[gridded["SEG_ID"].isin(step3data.traj_idx["seg_id"])]
    gridded = gridded[gridded["DECLARED_DEST_PORT_ID"].notna()]  # excluded, not scored as wrong

    if dedup_strategy == "all_pings":
        pass  # every raw ping kept as its own row, deliberately no dedup
    elif dedup_strategy == "majority_vote":
        n_before = len(gridded)
        if has_timestamp:
            gridded["TIMESTAMP"] = pd.to_datetime(gridded["TIMESTAMP"])
            tie_break_col = "TIMESTAMP"
        else:
            print(f"    WARNING: trajectories_gridded.parquet has no TIMESTAMP column -- majority_vote's "
                  f"own tie-breaking (multiple equally-common declared values within one step) falls "
                  f"back to file row order, not a genuine most-recent tie-break.")
            gridded = gridded.reset_index(drop=True)
            gridded["_row_order"] = gridded.index
            tie_break_col = "_row_order"

        # Each candidate declared value's own vote count within its step,
        # plus (for tie-breaking) how recently that SAME candidate value
        # was last seen -- NOT the same as most_recent's own tie-break
        # (which only looks at the single latest row regardless of value).
        vote_counts = (gridded.groupby(["SEG_ID", "STEP_IDX", "DECLARED_DEST_PORT_ID"])
                        .agg(votes=("DECLARED_DEST_PORT_ID", "size"), tie_break=(tie_break_col, "max"))
                        .reset_index())
        vote_counts = vote_counts.sort_values(["SEG_ID", "STEP_IDX", "votes", "tie_break"],
                                                ascending=[True, True, False, False])
        gridded = vote_counts.drop_duplicates(subset=["SEG_ID", "STEP_IDX"], keep="first")[
            ["SEG_ID", "STEP_IDX", "DECLARED_DEST_PORT_ID"]]
        n_dropped = n_before - len(gridded)
        if n_dropped:
            print(f"    {n_dropped:,} of {n_before:,} declared-destination rows were extra pings within an "
                  f"already-declared step ({100*n_dropped/n_before:.1f}%) -- kept the majority-vote "
                  f"declared value per step (ties broken by most recent).")
    else:
        if has_timestamp:
            gridded["TIMESTAMP"] = pd.to_datetime(gridded["TIMESTAMP"])
            gridded = gridded.sort_values(["SEG_ID", "STEP_IDX", "TIMESTAMP"])
        else:
            print(f"    WARNING: trajectories_gridded.parquet has no TIMESTAMP column -- captain accuracy "
                  f"dedup_strategy={dedup_strategy!r} for multi-ping grid-steps falls back to file row "
                  f"order, not a genuine {dedup_strategy.replace('_', ' ')} selection.")
        n_before = len(gridded)
        gridded = gridded.drop_duplicates(subset=["SEG_ID", "STEP_IDX"], keep=("last" if dedup_strategy == "most_recent" else "first"))
        n_dropped = n_before - len(gridded)
        if n_dropped:
            print(f"    {n_dropped:,} of {n_before:,} declared-destination rows were extra pings within an "
                  f"already-declared step ({100*n_dropped/n_before:.1f}%) -- kept only the "
                  f"{dedup_strategy.replace('_', ' ')} declaration per step.")

    port_to_subregion = build_port_to_subregion_map(step3data.traj_idx, subregion_col=target_col, port_col="ARR_PORT_ID")
    gridded["declared_subregion"] = gridded["DECLARED_DEST_PORT_ID"].map(port_to_subregion)
    gridded = gridded[gridded["declared_subregion"].notna()]  # a declared port never seen as an arrival anywhere -- unmappable, excluded

    true_subregion = step3data.traj_idx.set_index("seg_id")[target_col]
    gridded["true_subregion"] = gridded["SEG_ID"].map(true_subregion)
    gridded = gridded[gridded["true_subregion"].notna()]

    seg_length = gridded.groupby("SEG_ID")["STEP_IDX"].transform("max") + 1  # 0-indexed STEP_IDX -> length
    gridded["progression_frac"] = (gridded["STEP_IDX"] + 1) / seg_length
    gridded["correct"] = gridded["declared_subregion"] == gridded["true_subregion"]
    return gridded


def build_model_vs_captain_combined_accuracy(model, repr_layer, val_loader, core_and_alt_fn, step3data, work_dir,
                                              subregion_name_map, departure_ids_fn=None, eta_channel_lookup=None,
                                              target_col="ARR_SUBREGION_ID", boundaries=DEFAULT_PROGRESSION_BOUNDARIES,
                                              combined_model=None, combined_repr_layer=None, combined_core_and_alt_fn=None,
                                              combined_departure_ids_fn=None, combined_eta_channel_lookup=None,
                                              set_label="val-set", dedup_strategy="most_recent"):
    """THREE accuracy-by-progression curves, all computed on the EXACT
    SAME (seg_id, step_idx) steps -- the model's own validation set --
    so they're genuinely comparable, unlike naively plotting the
    model's val-only accuracy against build_captain_declared_subregion_
    accuracy's whole-dataset accuracy (a different population of steps,
    confirmed to matter: the model's own progression_acc is val-only,
    that function's own is train+val+test combined).

    1) "model": the model's own top-1 prediction, same evaluation the
       model's own metrics use.
    2) "captain": the captain's own declaration -- but ONLY on val-set
       steps that also have a captain declaration (a strict subset of
       (1)'s steps), so this differs somewhat from build_captain_
       declared_subregion_accuracy's own number (whole dataset) by
       design, not by omission.
    3) "combined": what this represents depends on combined_model.

       combined_model=None (default): a simple, deployable POST-HOC RULE
       -- use the captain's declaration where one exists, otherwise fall
       back to the model's own prediction. NOT an oracle/ceiling (which
       would pick whichever of the two happens to be right, unrealistic
       since you can't know that in advance) -- what you could actually
       run in production with no retraining at all.

       combined_model=<a genuinely different, trained model> (e.g. the
       "+ Declared destination channel" variant from
       ablation_2_channel_ablation.py's Part B, use_declared_destination
       =True): "combined" becomes THAT model's own top-1 predictions,
       evaluated on the exact same batches/steps as "model" and
       "captain" above -- a jointly-LEARNED integration of the
       declaration with everything else, not a hard override rule.
       Confirmed directly this matters, not just a theoretical
       distinction: the post-hoc rule and the trained-in channel gave
       substantially different accuracy on the same real data (the
       trained-in channel meaningfully higher) -- the rule blindly
       trusts the declaration 100% whenever present even when it's
       known to be unreliable early-voyage, using a model that never
       saw the declaration at all during training; the trained-in
       channel can learn to discount it exactly when it should.
       combined_repr_layer is required alongside combined_model.
       combined_core_and_alt_fn/combined_departure_ids_fn/
       combined_eta_channel_lookup default to the SAME ones as the main
       model if not given (the common case -- the only difference
       between the two models is normally use_declared_destination
       itself, not the progression setup) but can be overridden if the
       combined model genuinely differs there too.

    Returns {"model": {...}, "captain": {...}, "combined": {...}}, each
    shaped like evaluate_quartile_accuracy's own output (progression_acc/
    progression_labels/overall_acc), directly comparable, all three on
    the same steps.
    """
    if combined_model is not None and combined_repr_layer is None:
        raise ValueError("combined_repr_layer is required when combined_model is given")
    if combined_model is not None:
        combined_core_and_alt_fn = combined_core_and_alt_fn or core_and_alt_fn
        # departure_ids_fn/eta_channel_lookup default to None either way,
        # so "or" would silently swap a real None for the main model's
        # own -- only fall back when the combined-specific one was never
        # passed at all, i.e. still at its own default of None AND the
        # main model actually has one to inherit.
        if combined_departure_ids_fn is None:
            combined_departure_ids_fn = departure_ids_fn
        if combined_eta_channel_lookup is None:
            combined_eta_channel_lookup = eta_channel_lookup

    boundaries = tuple(sorted(boundaries))
    n_bands = len(boundaries)

    if dedup_strategy == "all_pings":
        raise ValueError("dedup_strategy='all_pings' is not usable here -- this function compares the "
                          "model's own SINGLE prediction at each step against ONE captain value, but "
                          "'all_pings' can return several, possibly conflicting, declarations for the "
                          "same step. Use 'most_recent' or 'first' here; 'all_pings' is fully supported "
                          "in build_captain_declared_subregion_accuracy instead, which never compares "
                          "against a model prediction and can score every ping as its own observation.")
    captain_lookup_df = _build_captain_declared_lookup(step3data, work_dir, subregion_name_map, target_col,
                                                        dedup_strategy=dedup_strategy)
    # (seg_id, step_idx) -> declared_subregion, for O(1) per-step lookup
    # inside the batch loop below -- built ONCE, not re-filtered per batch.
    # declared_subregion is ALREADY an integer subregion ID here (from
    # build_port_to_subregion_map's own port->ID mapping, used inside
    # _build_captain_declared_lookup) -- directly comparable against the
    # model's own integer class predictions with NO name/ID conversion
    # needed. Confirmed directly: an earlier version of this function
    # incorrectly treated this as a subregion NAME and tried converting
    # it through a name->ID map, which silently failed every single
    # comparison (always returned None, so captain_correct was always
    # False) -- both captain and combined accuracy came back exactly
    # 0.0 on a test dataset deliberately built to have ~100% captain
    # accuracy late-voyage, which is what caught it.
    captain_by_step = captain_lookup_df.set_index(["SEG_ID", "STEP_IDX"])["declared_subregion"].to_dict()

    # (seg_id, TENSOR POSITION) -> the REAL STEP_IDX at that position --
    # NOT the same thing as STEP_IDX itself whenever a segment's own
    # STEP_IDX sequence has gaps (e.g. [0,2,4,5,6,8,9,10,11,12] --
    # ANY segment with a missing intermediate grid-step, a real
    # possibility with raw AIS data). prepare_batch/BucketedWAYDataset
    # place each segment's own steps into tensor position 0,1,2,... in
    # STEP_IDX-sorted order (step3data._steps_by_seg's own construction,
    # itself built from steps_idx already sorted by ["SEG_ID","STEP_IDX"])
    # -- tensor position is a RANK, not the STEP_IDX value. Confirmed
    # directly this distinction matters, not just theoretically: on a
    # deliberately gapped test segment, treating tensor position AS
    # STEP_IDX silently dropped roughly a fifth of real declared-steps
    # (lookup returned None for a position whose true STEP_IDX simply
    # didn't match its own rank) and, worse, could misattribute one
    # step's own declaration to a DIFFERENT step at a nearby tensor
    # position -- not just an undercount, a genuine step-level
    # misalignment. Built once per segment actually appearing in this
    # loader, not the whole dataset.
    step_idx_at_position = {}
    for seg_id in set(sid for batch in val_loader.batches for sid in batch):
        seg_steps = step3data._steps_by_seg.get(seg_id)
        if seg_steps is None or len(seg_steps) == 0:
            continue
        for pos, real_step_idx in enumerate(seg_steps["STEP_IDX"].tolist()):
            step_idx_at_position[(seg_id, pos)] = real_step_idx

    model_band_correct = np.zeros(n_bands); model_band_total = np.zeros(n_bands)
    captain_band_correct = np.zeros(n_bands); captain_band_total = np.zeros(n_bands)
    combined_band_correct = np.zeros(n_bands); combined_band_total = np.zeros(n_bands)

    for i in tqdm(range(len(val_loader)), desc="model vs. captain vs. combined", unit="batch", leave=False):
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
            pred = ops.convert_to_numpy(ops.argmax(logits, axis=-1))  # [batch, N] -- integer class id per step

        combined_pred = None
        if combined_model is not None:
            # SAME underlying seg_ids/inputs as the main model's batch
            # above -- the raw data doesn't depend on which model
            # processes it, so this is the exact same (seg_id, step_idx)
            # population as "model" and "captain" here, not a
            # separately-fetched batch that might not line up 1:1.
            combined_inputs = dict(inputs)
            if combined_eta_channel_lookup is not None:
                n_steps = inputs["tau"].shape[1]
                combined_inputs["eta_channel_values"] = ops.convert_to_tensor(
                    eta_progression_for_batch(combined_eta_channel_lookup, batch_seg_ids, n_steps=n_steps))
            combined_core, combined_alts = combined_core_and_alt_fn(combined_inputs, batch_seg_ids)
            combined_dep_ids = combined_departure_ids_fn(batch_seg_ids) if combined_departure_ids_fn is not None else None
            with torch.no_grad():
                combined_x = combined_repr_layer(combined_inputs)
                combined_logits = combined_model(combined_x, key_padding_mask=n_mask, external_progression_frac=combined_core,
                                                  alt_progression_fracs=combined_alts, departure_subregion_ids=combined_dep_ids)
                combined_pred = ops.convert_to_numpy(ops.argmax(combined_logits, axis=-1))  # [batch, N]

        mask_np = ops.convert_to_numpy(n_mask).astype(bool)  # [batch, N]
        labels_np = ops.convert_to_numpy(labels)  # [batch] -- same true class id every step
        mask_f = mask_np.astype("float32")
        cum_steps = np.cumsum(mask_f, axis=1)
        real_length = np.maximum(mask_f.sum(axis=1, keepdims=True), 1.0)
        progression_frac = cum_steps / real_length  # [batch, N]

        for b, seg_id in enumerate(batch_seg_ids):
            true_id = int(labels_np[b])
            for n in range(pred.shape[1]):
                if not mask_np[b, n]:
                    continue
                frac = progression_frac[b, n]
                band = next((k for k, bnd in enumerate(boundaries) if frac <= bnd), n_bands - 1)

                model_correct = int(pred[b, n]) == true_id
                model_band_total[band] += 1
                if model_correct:
                    model_band_correct[band] += 1

                real_step_idx = step_idx_at_position.get((seg_id, n))
                declared_id = captain_by_step.get((seg_id, real_step_idx)) if real_step_idx is not None else None  # None if not declared -- already an int ID
                if declared_id is not None:
                    captain_correct = int(declared_id) == true_id
                    captain_band_total[band] += 1
                    if captain_correct:
                        captain_band_correct[band] += 1

                if combined_pred is not None:
                    combined_correct = int(combined_pred[b, n]) == true_id  # the trained-in-channel model's OWN prediction
                elif declared_id is not None:
                    combined_correct = captain_correct  # post-hoc rule: captain available -> use it
                else:
                    combined_correct = model_correct  # post-hoc rule: no declaration -> fall back to the model

                combined_band_total[band] += 1
                if combined_correct:
                    combined_band_correct[band] += 1

    def _pack(correct, total):
        return {
            "progression_boundaries": list(boundaries),
            "progression_labels": _progression_labels(boundaries),
            "progression_acc": correct / np.maximum(total, 1),
            "band_correct": correct.tolist(),
            "band_total": total.tolist(),
            "overall_acc": correct.sum() / max(1, total.sum()),
            "n_steps": int(total.sum()),
        }

    print(f"Model-vs-captain-vs-combined: {int(model_band_total.sum()):,} {set_label} steps total, "
          f"{int(captain_band_total.sum()):,} of which also have a captain declaration.")

    return {
        "model": _pack(model_band_correct, model_band_total),
        "captain": _pack(captain_band_correct, captain_band_total),
        "combined": _pack(combined_band_correct, combined_band_total),
    }


def summarize_accuracy_by_stage(results, boundaries=DEFAULT_PROGRESSION_BOUNDARIES, early_late_cutoffs=(0.20, 0.60)):
    """Standard accuracy-by-voyage-stage summary table -- Early/Mid/Late
    (matching this project's established zone convention) plus Overall,
    for any number of named results side by side. Intended as the
    default companion to every progression-accuracy plot going forward,
    not a one-off for this particular comparison.

    results: {label: result_dict}, where each result_dict has
    "band_correct", "band_total" (raw per-fine-band counts -- from
    evaluate_quartile_accuracy, _evaluate_with_core_and_alt_progression,
    build_captain_declared_subregion_accuracy, or
    build_model_vs_captain_combined_accuracy's own sub-dicts, all of
    which return these now) and "overall_acc".

    Early/Mid/Late are computed by SUMMING raw correct/total counts
    across the fine bands that fall in each zone, then dividing --
    NOT by averaging the fine bands' own pre-computed percentages
    together, which would be subtly wrong whenever the fine bands don't
    all have equal step counts (an unweighted mean of percentages
    silently over-weights thin bands and under-weights busy ones; the
    zone's own true accuracy is total-correct/total-steps across
    everything in it, which summing achieves and averaging does not).

    Returns a pandas DataFrame: Configuration, Overall Accuracy (%),
    Early (0-20%) (%), Mid (20-60%) (%), Late (60-100%) (%) -- ready to
    print directly (already percentage-formatted, 1 decimal place).
    """
    boundaries = tuple(sorted(boundaries))
    early_end, mid_end = early_late_cutoffs

    def _zone_of(band_upper_edge):
        if band_upper_edge <= early_end:
            return "early"
        elif band_upper_edge <= mid_end:
            return "mid"
        else:
            return "late"

    zone_of_band = [_zone_of(b) for b in boundaries]

    rows = []
    for label, r in results.items():
        band_correct = np.array(r["band_correct"], dtype="float64")
        band_total = np.array(r["band_total"], dtype="float64")
        if len(band_correct) != len(boundaries):
            raise ValueError(f"{label!r}: band_correct has {len(band_correct)} entries but boundaries has "
                              f"{len(boundaries)} -- these must be computed with the SAME boundaries to summarize together")

        zone_correct = {"early": 0.0, "mid": 0.0, "late": 0.0}
        zone_total = {"early": 0.0, "mid": 0.0, "late": 0.0}
        for zone, c, t in zip(zone_of_band, band_correct, band_total):
            zone_correct[zone] += c
            zone_total[zone] += t

        rows.append({
            "Configuration": label,
            "Overall Accuracy (%)": round(r["overall_acc"] * 100, 1),
            "Early (0-20%) (%)": round(100 * zone_correct["early"] / max(1, zone_total["early"]), 1),
            "Mid (20-60%) (%)": round(100 * zone_correct["mid"] / max(1, zone_total["mid"]), 1),
            "Late (60-100%) (%)": round(100 * zone_correct["late"] / max(1, zone_total["late"]), 1),
        })

    return pd.DataFrame(rows)


def plot_three_way_benchmark(curves, work_dir, colors=None, majority_baseline=None,
                              boundaries=DEFAULT_PROGRESSION_BOUNDARIES, early_late_cutoffs=(0.20, 0.60),
                              title="Model vs. captain vs. combined accuracy",
                              subtitle="Subregion-level match, by voyage progression",
                              save_name="model_captain_combined_benchmark.png"):
    """Same shaded early/mid/late-voyage-zone visual style as
    plot_model_vs_captain_benchmark, generalized to N lines (that
    function's gap annotation is specific to exactly 2 lines, so left
    out here rather than forced to be ambiguous for 3+).

    curves: {label: progression_acc array} -- e.g. from
    build_model_vs_captain_combined_accuracy's own
    {"model": {...}, "captain": {...}, "combined": {...}}, passing each
    sub-dict's "progression_acc" through: {"Final Model":
    result["model"]["progression_acc"], "Captain declared destination":
    result["captain"]["progression_acc"], "Final Model + Captain
    declaration": result["combined"]["progression_acc"]}. Any number of
    curves works, plotted in dict insertion order.
    """
    import matplotlib.pyplot as plt

    boundaries = tuple(sorted(boundaries))
    labels = _progression_labels(boundaries)
    x = list(range(len(labels)))

    default_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]
    colors = colors or default_colors

    def _to_pct(acc):
        return [v * 100 if v <= 1.0 else v for v in acc]

    curves_pct = {label: _to_pct(acc) for label, acc in curves.items()}

    fig, ax = plt.subplots(figsize=(13, 7))

    early_end, mid_end = early_late_cutoffs

    def _frac_to_x(frac):
        for i, b in enumerate(boundaries):
            if frac <= b:
                if i == 0:
                    return i
                prev_b = boundaries[i - 1]
                return (i - 1) + (frac - prev_b) / (b - prev_b)
        return len(boundaries) - 1

    early_x, mid_x = _frac_to_x(early_end), _frac_to_x(mid_end)
    ax.axvspan(-0.5, early_x, color="#e74c3c", alpha=0.08, zorder=0)
    ax.axvspan(early_x, mid_x, color="#f39c12", alpha=0.08, zorder=0)
    ax.axvspan(mid_x, len(x) - 0.5, color="#2ecc71", alpha=0.08, zorder=0)

    if majority_baseline is not None:
        baseline_pct = majority_baseline * 100 if majority_baseline <= 1.0 else majority_baseline
        ax.axhline(baseline_pct, color="#e74c3c", linestyle=":", linewidth=1.2, zorder=1)
        ax.text(len(x) - 0.5, baseline_pct, f"  majority\n  baseline\n  ({baseline_pct:.0f}%)",
                fontsize=8, color="#e74c3c", va="center")

    for (label, pct), color in zip(curves_pct.items(), colors):
        ax.plot(x, pct, marker="o", linewidth=2.5, color=color, label=label, zorder=3)

    all_pct = [v for pct in curves_pct.values() for v in pct]
    data_min, data_max = min(all_pct), max(all_pct)
    data_range = max(data_max - data_min, 1.0)
    label_reserve = 0.15 * data_range
    y_bottom = data_min - label_reserve - 0.05 * data_range
    y_top_lim = data_max + 0.08 * data_range
    ax.set_ylim(y_bottom, y_top_lim)
    label_y = data_min - label_reserve * 0.6

    for x_pos, text, color in [
        (early_x / 2, "Early voyage\n(unreliable)", "#c0392b"),
        ((early_x + mid_x) / 2, "Mid voyage", "#d68910"),
        ((mid_x + len(x) - 1) / 2, "Late voyage", "#1e8449"),
    ]:
        ax.text(x_pos, label_y, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Voyage completion (%)")
    ax.set_ylabel("Subregion match (%)")
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, y=0.98)
    ax.set_title(subtitle, loc="left", fontsize=10, color="gray")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, len(x) - 0.5)

    plt.tight_layout()
    out_path = os.path.join(work_dir, save_name)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved -> {out_path}")

# ----------------------------------------------------------------------
# [notebook cell 68]
# ----------------------------------------------------------------------

# =============================================================================
# E1 RUN CELL -- Model vs Captain vs Model+Declaration, on the TEST set
# =============================================================================
# CAPTAIN SCORING CHOICE -- how a grid-step with multiple AIS pings yields one
# captain "prediction" (real measured spread most_recent vs all_pings: ~8 pts):
#   "most_recent"   best-informed by the END of each step (mild hindsight)
#   "first"         strictly no-lookahead: earliest declaration in the step
#   "majority_vote" modal declaration across the step's pings
#   "all_pings"     every raw ping scored -- ONLY valid for the standalone
#                   captain metric; the three-way benchmark below needs one
#                   value per step and will refuse it (by design).

CAPTAIN_DEDUP_STRATEGY = "most_recent"   # or "first" | "majority_vote" | "all_pings"

MODEL_LABEL    = "Model (no captain declared destination)"
CAPTAIN_LABEL  = f"Captain declared destination only ({CAPTAIN_DEDUP_STRATEGY})"
COMBINED_LABEL = "Model + Captain declared destination"
MAJORITY_BASELINE = None            # e.g. 0.30 to draw the reference line

assert "final_runs" in dir() and all(s in final_runs for s in SEEDS), \
    "final_runs missing -- run E0 B (CELL E0B-1) first: 'Model' reuses those checkpoints"
subregion_name_map = get_subregion_name_map(data)

for seed in SEEDS:
    print(); print("#" * 70); print(f"SEED {seed}"); print("#" * 70)
    r = final_runs[seed]                      # the E0 B final model, reloaded free
    test_ids = r["_test_ids"]

    # ---- "Combined": same protocol as E0 B, + the declaration as a channel
    train_ids, val_ids, _ = _make_split(
        data, TARGET_COL, val_frac=0.15, seed=seed, stratify=True,
        test_start=TEST_START, test_end=TEST_END)
    combined_pool = list(train_ids) + list(val_ids)
    cond = f"final_main_lean2_trainval_declared_seed{seed}"
    print(f"  training/loading {cond}: {BEST_EPOCHS[seed]} fixed epochs, "
          f"+ use_declared_destination=True")
    r_comb = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES, condition_name=cond, seed=seed,
        alt_progression_modes=ALT_PROGRESSION_MODES,
        gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
        use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
        use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
        use_departure_gate=USE_DEPARTURE_GATE,
        n_experts=N_EXPERTS, d_model=D_MODEL,
        stratify=True, val_frac=0.15,
        test_start=TEST_START, test_end=TEST_END,
        epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None,
        train_ids_override=combined_pool, val_ids_override=list(val_ids),
        use_declared_destination=True,
        batch_size=BATCH_SIZE, work_dir=WORK_DIR, skip_existing=True,
    )

    test_loader = BucketedWAYDataset(
        data, target_col=TARGET_COL, batch_size=BATCH_SIZE, seg_id_subset=test_ids,
        shuffle=False, seed=0, include_ship_history=True)

    res = build_model_vs_captain_combined_accuracy(
        r["model"], r["repr_layer"], test_loader, r["core_and_alt_fn"], data, WORK_DIR,
        subregion_name_map, departure_ids_fn=r.get("departure_ids_fn"),
        eta_channel_lookup=r.get("eta_channel_lookup"),
        combined_model=r_comb["model"], combined_repr_layer=r_comb["repr_layer"],
        combined_core_and_alt_fn=r_comb["core_and_alt_fn"],
        combined_departure_ids_fn=r_comb.get("departure_ids_fn"),
        combined_eta_channel_lookup=r_comb.get("eta_channel_lookup"),
        set_label="test-set", dedup_strategy=CAPTAIN_DEDUP_STRATEGY)

    print(f"  Model    overall_acc (TEST): {res['model']['overall_acc']:.3f}")
    print(f"  Captain  overall_acc ({CAPTAIN_DEDUP_STRATEGY}; declared steps only, "
          f"{res['captain']['n_steps']:,}/{res['model']['n_steps']:,}): "
          f"{res['captain']['overall_acc']:.3f}")
    print(f"  Combined overall_acc (TEST): {res['combined']['overall_acc']:.3f}")

    plot_three_way_benchmark({
        MODEL_LABEL: res["model"]["progression_acc"],
        CAPTAIN_LABEL: res["captain"]["progression_acc"],
        COMBINED_LABEL: res["combined"]["progression_acc"],
    }, WORK_DIR, majority_baseline=MAJORITY_BASELINE,
       save_name=f"model_captain_combined_{CAPTAIN_DEDUP_STRATEGY}_seed{seed}.png")

    table = summarize_accuracy_by_stage({
        MODEL_LABEL: res["model"], CAPTAIN_LABEL: res["captain"],
        COMBINED_LABEL: res["combined"]})
    print(f"\n--- Accuracy by voyage stage (seed {seed}, same test-set steps, "
          f"captain={CAPTAIN_DEDUP_STRATEGY}) ---")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(WORK_DIR,
        f"accuracy_by_stage_{CAPTAIN_DEDUP_STRATEGY}_seed{seed}.csv"), index=False)

# ----------------------------------------------------------------------
# [notebook cell 69]
# ----------------------------------------------------------------------
# =============================================================================
# E1-COLLECT -- all three curves, all seeds, retained for averaging
# =============================================================================
# Self-sufficient: reloads Model (E0 B) and Combined (declared) checkpoints
# per seed via skip_existing and re-runs the benchmark, storing each seed's
# three curves. Evaluation-only after the checkpoints exist.
E1_RES = {}
for seed in SEEDS:
    r = final_runs[seed]
    train_ids, val_ids, _ = _make_split(
        data, TARGET_COL, val_frac=0.15, seed=seed, stratify=True,
        test_start=TEST_START, test_end=TEST_END)
    r_comb = train_residual_progression_variant(
        data, TARGET_COL, N_CLASSES,
        condition_name=f"final_main_lean2_trainval_declared_seed{seed}", seed=seed,
        alt_progression_modes=ALT_PROGRESSION_MODES,
        gate_ship_history=GATE_SHIP_HISTORY, use_ship_history=USE_SHIP_HISTORY,
        use_ship_size_channel=USE_SHIP_SIZE_CHANNEL,
        use_departure_port_channel=USE_DEPARTURE_PORT_CHANNEL,
        use_departure_gate=USE_DEPARTURE_GATE,
        n_experts=N_EXPERTS, d_model=D_MODEL, stratify=True, val_frac=0.15,
        test_start=TEST_START, test_end=TEST_END,
        epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None,
        train_ids_override=list(train_ids) + list(val_ids), val_ids_override=list(val_ids),
        use_declared_destination=True,
        batch_size=BATCH_SIZE, work_dir=WORK_DIR, skip_existing=True)
    test_loader = BucketedWAYDataset(
        data, target_col=TARGET_COL, batch_size=BATCH_SIZE,
        seg_id_subset=r["_test_ids"], shuffle=False, seed=0, include_ship_history=True)
    E1_RES[seed] = build_model_vs_captain_combined_accuracy(
        r["model"], r["repr_layer"], test_loader, r["core_and_alt_fn"], data, WORK_DIR,
        subregion_name_map, departure_ids_fn=r.get("departure_ids_fn"),
        eta_channel_lookup=r.get("eta_channel_lookup"),
        combined_model=r_comb["model"], combined_repr_layer=r_comb["repr_layer"],
        combined_core_and_alt_fn=r_comb["core_and_alt_fn"],
        combined_departure_ids_fn=r_comb.get("departure_ids_fn"),
        combined_eta_channel_lookup=r_comb.get("eta_channel_lookup"),
        set_label="test-set", dedup_strategy=CAPTAIN_DEDUP_STRATEGY)
    print(f"seed {seed}: curves stored")
print(f"E1_RES holds {len(E1_RES)} seeds x 3 series")

# =============================================================================
# E1-PLOT-AVG -- 3-seed mean curves with STDEV bands (house benchmark style)
# =============================================================================
import numpy as np

def plot_three_way_benchmark_meanstd(curves_mean_std, work_dir, colors=None,
                                     majority_baseline=None,
                                     boundaries=DEFAULT_PROGRESSION_BOUNDARIES,
                                     early_late_cutoffs=(0.20, 0.60),
                                     title="Model vs. captain vs. combined accuracy",
                                     subtitle="Subregion-level match, by voyage progression",
                                     save_name="model_captain_combined_benchmark_meanstd.png"):
    """plot_three_way_benchmark, verbatim styling, with each curve given as
    (mean_array, std_array) and drawn as line + shaded +-1 std band."""
    import matplotlib.pyplot as plt
    boundaries = tuple(sorted(boundaries))
    labels = _progression_labels(boundaries)
    x = list(range(len(labels)))
    default_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]
    colors = colors or default_colors

    def _to_pct(a):
        a = np.asarray(a, dtype="float64")
        return np.where(a <= 1.0, a * 100, a)

    fig, ax = plt.subplots(figsize=(13, 7))
    early_end, mid_end = early_late_cutoffs

    def _frac_to_x(frac):
        for i, b in enumerate(boundaries):
            if frac <= b:
                if i == 0:
                    return i
                prev_b = boundaries[i - 1]
                return (i - 1) + (frac - prev_b) / (b - prev_b)
        return len(boundaries) - 1

    early_x, mid_x = _frac_to_x(early_end), _frac_to_x(mid_end)
    ax.axvspan(-0.5, early_x, color="#e74c3c", alpha=0.08, zorder=0)
    ax.axvspan(early_x, mid_x, color="#f39c12", alpha=0.08, zorder=0)
    ax.axvspan(mid_x, len(x) - 0.5, color="#2ecc71", alpha=0.08, zorder=0)

    if majority_baseline is not None:
        b_pct = majority_baseline * 100 if majority_baseline <= 1.0 else majority_baseline
        ax.axhline(b_pct, color="#e74c3c", linestyle=":", linewidth=1.2, zorder=1)
        ax.text(len(x) - 0.5, b_pct, f"  majority\n  baseline\n  ({b_pct:.0f}%)",
                fontsize=8, color="#e74c3c", va="center")

    all_vals = []
    for (label, (mean, std)), color in zip(curves_mean_std.items(), colors):
        m, s = _to_pct(mean), _to_pct(std) if np.nanmax(std) <= 1.0 else np.asarray(std)
        s = np.asarray(std, dtype="float64")
        s = np.where(np.asarray(mean) <= 1.0, s * 100, s)
        ax.plot(x, m, marker="o", linewidth=2.5, color=color, label=label, zorder=3)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0, zorder=2)
        all_vals += [v for v in (m - s).tolist() + (m + s).tolist() if np.isfinite(v)]

    data_min, data_max = min(all_vals), max(all_vals)
    data_range = max(data_max - data_min, 1.0)
    label_reserve = 0.15 * data_range
    ax.set_ylim(data_min - label_reserve - 0.05 * data_range, data_max + 0.08 * data_range)
    label_y = data_min - label_reserve * 0.6
    for x_pos, text, color in [
        (early_x / 2, "Early voyage\n(unreliable)", "#c0392b"),
        ((early_x + mid_x) / 2, "Mid voyage", "#d68910"),
        ((mid_x + len(x) - 1) / 2, "Late voyage", "#1e8449"),
    ]:
        ax.text(x_pos, label_y, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Voyage completion (%)"); ax.set_ylabel("Subregion match (%)")
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, y=0.98)
    ax.set_title(subtitle, loc="left", fontsize=10, color="gray")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, len(x) - 0.5)
    plt.tight_layout()
    out_path = os.path.join(work_dir, save_name)
    plt.savefig(out_path, dpi=150); plt.show()
    print(f"Saved -> {out_path}")

# ---- build mean/std per series from E1_RES (band_total-masked) and plot ----
SERIES = [("model", "Model"),
          ("captain", f"Captain declared destination ({CAPTAIN_DEDUP_STRATEGY})"),
          ("combined", "Final Model + Captain declaration")]
curves_ms = {}
for key, label in SERIES:
    C = np.vstack([np.where(np.array(E1_RES[s][key]["band_total"], dtype="float64") > 0,
                            np.array(E1_RES[s][key]["progression_acc"], dtype="float64"),
                            np.nan) for s in SEEDS])
    curves_ms[label] = (np.nanmean(C, axis=0), np.nanstd(C, axis=0))

plot_three_way_benchmark_meanstd(
    curves_ms, WORK_DIR, majority_baseline=MAJORITY_BASELINE,
    title="Accuracy of Model, Captain declared destination, and MOdel combined with declared destination (3-seed mean) \u00b1 1 std",
    subtitle=f"TEST set [{TEST_START} -> {TEST_END}], captain: {CAPTAIN_DEDUP_STRATEGY}",
    save_name=f"e1_threeway_mean_std_{CAPTAIN_DEDUP_STRATEGY}.png")

# =============================================================================
# E1-TABLE-5PCT (fixed) -- averaged accuracy per 5% bin, all three series
# =============================================================================
import numpy as np, pandas as pd

_bounds = np.array(sorted(DEFAULT_PROGRESSION_BOUNDARIES))
labels_5 = _progression_labels(tuple(_bounds))

def _stacked5(key):
    C = np.vstack([np.where(np.array(E1_RES[s][key]["band_total"], dtype="float64") > 0,
                            np.array(E1_RES[s][key]["progression_acc"], dtype="float64"),
                            np.nan) for s in SEEDS])
    return C

tbl = {}
for item in SERIES:                        # works for 2- or 3-tuple SERIES
    key, label = item[0], item[1]
    C = _stacked5(key)
    mean = np.nanmean(C, axis=0) * 100
    std = np.nanstd(C, axis=0) * 100
    tbl[label] = [f"{m:.2f} \u00b1 {s:.2f}" if np.isfinite(m) else "--"
                  for m, s in zip(mean, std)]

e1_5pct = pd.DataFrame(tbl, index=labels_5[:len(_bounds)])
print("=" * 96)
print(f"E1 -- TEST ACCURACY PER 5% PROGRESSION BIN, 3-seed mean \u00b1 std  "
      f"(captain: {CAPTAIN_DEDUP_STRATEGY})")
print("=" * 96)
print(e1_5pct.to_string())
e1_5pct.to_csv(os.path.join(WORK_DIR, f"e1_threeway_5pct_{CAPTAIN_DEDUP_STRATEGY}.csv"))
print("\nSaved -> e1_threeway_5pct_" + CAPTAIN_DEDUP_STRATEGY + ".csv")

# ----------------------------------------------------------------------
# [notebook cell 70]
# ----------------------------------------------------------------------
# FINAL FIGURE Final Main Plot - Model vs Captain vs Combined, split by voyage duration (%)

# =============================================================================
# E1-PLOT-AVG v2 -- 3-seed mean curves + std bands, with tunable text sizes
# =============================================================================
# All text sizes live in FS below: edit one dict, everything scales.
import os
import numpy as np
import matplotlib.pyplot as plt

FS = dict(          # ---- font sizes: adjust freely -----------------------
    title=17,       # main title
    subtitle=12,    # grey line under the title
    axis_label=14,  # "Voyage completion (%)" / "Subregion match (%)"
    tick=11,        # axis tick labels
    legend=12,      # series legend
    stage=13,       # "Early voyage" / "Mid voyage" / "Late voyage"
    baseline=10,    # majority-baseline annotation
)
LW = dict(curve=2.8, marker=7, band_alpha=0.15)
FIGSIZE = (13, 7)   # widen/shorten here; keep ratio ~1.85 for report columns


def plot_three_way_benchmark_meanstd(
        curves_mean_std, work_dir, colors=None, majority_baseline=None,
        boundaries=None, early_late_cutoffs=(0.20, 0.60),
        title="Model vs. captain vs. combined accuracy",
        subtitle="Subregion-level match, by voyage progression",
        save_name="model_captain_combined_benchmark_meanstd.png",
        fs=None, figsize=None):
    """Three-way benchmark curves with +-1 std bands and scalable typography."""
    fs = {**FS, **(fs or {})}
    figsize = figsize or FIGSIZE
    boundaries = tuple(sorted(boundaries if boundaries is not None
                              else DEFAULT_PROGRESSION_BOUNDARIES))
    labels = _progression_labels(boundaries)
    x = list(range(len(labels)))
    colors = colors or ["#2ecc71", "#3498db", "#9b59b6", "#e67e22",
                        "#1abc9c", "#e74c3c"]

    def _to_pct(a):
        a = np.asarray(a, dtype="float64")
        return np.where(a <= 1.0, a * 100, a)

    fig, ax = plt.subplots(figsize=figsize)
    early_end, mid_end = early_late_cutoffs

    def _frac_to_x(frac):
        for i, b in enumerate(boundaries):
            if frac <= b:
                if i == 0:
                    return i
                prev_b = boundaries[i - 1]
                return (i - 1) + (frac - prev_b) / (b - prev_b)
        return len(boundaries) - 1

    early_x, mid_x = _frac_to_x(early_end), _frac_to_x(mid_end)
    ax.axvspan(-0.5, early_x, color="#e74c3c", alpha=0.08, zorder=0)
    ax.axvspan(early_x, mid_x, color="#f39c12", alpha=0.08, zorder=0)
    ax.axvspan(mid_x, len(x) - 0.5, color="#2ecc71", alpha=0.08, zorder=0)

    if majority_baseline is not None:
        b_pct = (majority_baseline * 100 if majority_baseline <= 1.0
                 else majority_baseline)
        ax.axhline(b_pct, color="#e74c3c", linestyle=":", linewidth=1.4, zorder=1)
        ax.text(len(x) - 0.45, b_pct, f"  majority\n  baseline\n  ({b_pct:.0f}%)",
                fontsize=fs["baseline"], color="#e74c3c", va="center")

    all_vals = []
    for (label, (mean, std)), color in zip(curves_mean_std.items(), colors):
        m = _to_pct(mean)
        s = np.asarray(std, dtype="float64")
        s = np.where(np.asarray(mean) <= 1.0, s * 100, s)
        ax.plot(x, m, marker="o", markersize=LW["marker"], linewidth=LW["curve"],
                color=color, label=label, zorder=3)
        ax.fill_between(x, m - s, m + s, color=color, alpha=LW["band_alpha"],
                        linewidth=0, zorder=2)
        all_vals += [v for v in (m - s).tolist() + (m + s).tolist()
                     if np.isfinite(v)]

    data_min, data_max = min(all_vals), max(all_vals)
    data_range = max(data_max - data_min, 1.0)
    label_reserve = 0.15 * data_range
    ax.set_ylim(data_min - label_reserve - 0.05 * data_range,
                data_max + 0.08 * data_range)
    label_y = data_min - label_reserve * 0.6
    for x_pos, text, color in [
        (early_x / 2, "Early voyage\n(declaration unreliable)", "#c0392b"),
        ((early_x + mid_x) / 2, "Mid voyage", "#d68910"),
        ((mid_x + len(x) - 1) / 2, "Late voyage", "#1e8449"),
    ]:
        ax.text(x_pos, label_y, text, ha="center", va="center",
                fontsize=fs["stage"], fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs["tick"])
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.set_xlabel("Voyage completion (% of elapsed-to-total duration)",
                  fontsize=fs["axis_label"])
    ax.set_ylabel("Destination subregion accuracy (%)",
                  fontsize=fs["axis_label"])
    fig.suptitle(title, x=0.02, ha="left", fontsize=fs["title"], y=0.99)
    ax.set_title(subtitle, loc="left", fontsize=fs["subtitle"], color="gray")
    ax.legend(loc="upper left", fontsize=fs["legend"], framealpha=.92)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, len(x) - 0.5)
    plt.tight_layout()
    out_path = os.path.join(work_dir, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")


# ---- build mean/std per series from E1_RES (band_total-masked) and plot -----
SERIES = [("model", "Model"),
          ("captain", f"Captain's declared destination "
                      f"({CAPTAIN_DEDUP_STRATEGY})"),
          ("combined", "Model + declaration combined")]
curves_ms = {}
for key, label in SERIES:
    C = np.vstack([
        np.where(np.array(E1_RES[s][key]["band_total"], dtype="float64") > 0,
                 np.array(E1_RES[s][key]["progression_acc"], dtype="float64"),
                 np.nan)
        for s in SEEDS])
    curves_ms[label] = (np.nanmean(C, axis=0), np.nanstd(C, axis=0))

plot_three_way_benchmark_meanstd(
    curves_ms, WORK_DIR, majority_baseline=MAJORITY_BASELINE,
    title="Model, vs. Captain's declared Destination, vs. Model combining the two approaches",
    subtitle=f"TEST [{TEST_START} \u2192 {TEST_END}], three-seed mean \u00b1 1 std; "
             f"captain: {CAPTAIN_DEDUP_STRATEGY}",
    save_name=f"e1_threeway_mean_std_{CAPTAIN_DEDUP_STRATEGY}.png",
    # fs=dict(title=20, stage=15),      # <- example: bump individual sizes
    # figsize=(15, 8),                  # <- example: larger canvas
)

# ----------------------------------------------------------------------
# [notebook cell 71]
# ----------------------------------------------------------------------
# TEMP TO RUN - performance of combined model
import numpy as np
for key in ("model", "captain", "combined"):
    ov = [100 * np.sum(E1_RES[s][key]["band_correct"]) /
          max(np.sum(E1_RES[s][key]["band_total"]), 1) for s in SEEDS]
    eb = [100 * np.sum(np.array(E1_RES[s][key]["band_correct"])[_early]) /
          max(np.sum(np.array(E1_RES[s][key]["band_total"])[_early]), 1)
          for s in SEEDS]          # _early = boundaries <= 0.20, as in E1
    print(f"{key:9s}: overall {np.mean(ov):5.2f} ± {np.std(ov):.2f}   "
          f"early {np.mean(eb):5.2f} ± {np.std(eb):.2f}")

# ----------------------------------------------------------------------
# [notebook cell 72]
# ----------------------------------------------------------------------
# =============================================================================
# E1-SPLIT -- 3 GROUPS  Model vs Captain vs Combined, split by voyage duration (days)
# =============================================================================
SPLIT_MODE = "days"
N_GROUPS   = 3              # run 1: 3 (short/medium/long) | run 2: set to 2

import numpy as np, pandas as pd
from IPython.display import display, HTML

_test_all = list(final_runs[SEEDS[0]]["_test_ids"])
_tj = data.traj_idx.set_index("seg_id")
_dur = {int(s): (pd.Timestamp(_tj.loc[int(s), "arr_ts"])
                 - pd.Timestamp(_tj.loc[int(s), "dep_ts"])).days
        for s in _test_all}
_tl = pd.Series(_dur).sort_values(); _unit = "days"

_qs = _tl.quantile(np.arange(1, N_GROUPS) / N_GROUPS).tolist()
_names = (["Short", "Long"] if N_GROUPS == 2 else ["Short", "Medium", "Long"])
_edges = [-np.inf] + _qs + [np.inf]
GROUPS = {}
for i, nm in enumerate(_names):
    m = (_tl > _edges[i]) & (_tl <= _edges[i + 1])
    if i == len(_names) - 1:
        label = f"{nm} (> {int(_edges[i])} {_unit})"
    elif i == 0:
        label = f"{nm} (<= {int(_edges[i + 1])} {_unit})"
    else:
        label = f"{nm} ({int(_edges[i])+1}-{int(_edges[i+1])} {_unit})"
    GROUPS[label] = _tl[m].index.tolist()
for k, v in GROUPS.items():
    print(f"{k}: {len(v):,} voyages")

# ---- collect the three series per group per seed (evaluation-only) ---------
E1S_RES = {}
for gname, gids in GROUPS.items():
    E1S_RES[gname] = {}
    for seed in SEEDS:
        r = final_runs[seed]
        r_comb = train_residual_progression_variant(     # reload, never retrain
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
            seg_id_subset=gids, shuffle=False, seed=0, include_ship_history=True)
        E1S_RES[gname][seed] = build_model_vs_captain_combined_accuracy(
            r["model"], r["repr_layer"], t_loader, r["core_and_alt_fn"], data,
            WORK_DIR, subregion_name_map,
            departure_ids_fn=r.get("departure_ids_fn"),
            eta_channel_lookup=r.get("eta_channel_lookup"),
            combined_model=r_comb["model"], combined_repr_layer=r_comb["repr_layer"],
            combined_core_and_alt_fn=r_comb["core_and_alt_fn"],
            combined_departure_ids_fn=r_comb.get("departure_ids_fn"),
            combined_eta_channel_lookup=r_comb.get("eta_channel_lookup"),
            set_label=f"test-{gname}", dedup_strategy=CAPTAIN_DEDUP_STRATEGY)
    print(f"{gname}: 3 seeds collected")

# ---- plots (house style) + combined table -----------------------------------
def _ms(res_by_seed, key):
    C = np.vstack([np.where(np.array(res_by_seed[s][key]["band_total"],
                                     dtype="float64") > 0,
                            np.array(res_by_seed[s][key]["progression_acc"],
                                     dtype="float64"), np.nan) for s in SEEDS])
    return np.nanmean(C, axis=0), np.nanstd(C, axis=0)

labels_5 = _progression_labels(tuple(sorted(DEFAULT_PROGRESSION_BOUNDARIES)))
tables = {}
for gname in GROUPS:
    curves_ms = {label: _ms(E1S_RES[gname], key) for key, label in SERIES}
    plot_three_way_benchmark_meanstd(
        curves_ms, WORK_DIR, majority_baseline=MAJORITY_BASELINE,
        title=f"Model vs captain vs combined -- {gname} (3-seed mean \u00b1 1 std)",
        subtitle=f"TEST [{TEST_START} -> {TEST_END}], captain: {CAPTAIN_DEDUP_STRATEGY}",
        save_name=f"e1_threeway_meanstd_{SPLIT_MODE}{N_GROUPS}_"
                  f"{gname.split(' ')[0].lower()}.png")
    for key, label in SERIES:
        m, s = _ms(E1S_RES[gname], key)
        tables[(gname, label)] = [f"{mm*100:.2f} \u00b1 {ss*100:.2f}"
                                  if np.isfinite(mm) else "--"
                                  for mm, ss in zip(m, s)]
e1_split = pd.DataFrame(tables, index=labels_5)
e1_split.columns = pd.MultiIndex.from_tuples(e1_split.columns,
                                             names=["Voyage length", "Series"])
display(HTML(f"<b>E1 by voyage duration ({N_GROUPS} groups) — accuracy per "
             f"5% bin (3-seed mean ± std)</b>"))
display(e1_split)
e1_split.to_csv(os.path.join(WORK_DIR,
    f"e1_threeway_5pct_by_{SPLIT_MODE}{N_GROUPS}_{CAPTAIN_DEDUP_STRATEGY}.csv"))
print(f"saved plots + CSV for {SPLIT_MODE} x {N_GROUPS}")

# ----------------------------------------------------------------------
# [notebook cell 73]
# ----------------------------------------------------------------------
# =============================================================================
# E1-SPLIT -- 2 GROUPS  Model vs Captain vs Combined, split by voyage duration (days)
# =============================================================================
SPLIT_MODE = "days"
N_GROUPS   = 2              # run 1: 3 (short/medium/long) | run 2: set to 2

import numpy as np, pandas as pd
from IPython.display import display, HTML

_test_all = list(final_runs[SEEDS[0]]["_test_ids"])
_tj = data.traj_idx.set_index("seg_id")
_dur = {int(s): (pd.Timestamp(_tj.loc[int(s), "arr_ts"])
                 - pd.Timestamp(_tj.loc[int(s), "dep_ts"])).days
        for s in _test_all}
_tl = pd.Series(_dur).sort_values(); _unit = "days"

_qs = _tl.quantile(np.arange(1, N_GROUPS) / N_GROUPS).tolist()
_names = (["Short", "Long"] if N_GROUPS == 2 else ["Short", "Medium", "Long"])
_edges = [-np.inf] + _qs + [np.inf]
GROUPS = {}
for i, nm in enumerate(_names):
    m = (_tl > _edges[i]) & (_tl <= _edges[i + 1])
    if i == len(_names) - 1:
        label = f"{nm} (> {int(_edges[i])} {_unit})"
    elif i == 0:
        label = f"{nm} (<= {int(_edges[i + 1])} {_unit})"
    else:
        label = f"{nm} ({int(_edges[i])+1}-{int(_edges[i+1])} {_unit})"
    GROUPS[label] = _tl[m].index.tolist()
for k, v in GROUPS.items():
    print(f"{k}: {len(v):,} voyages")

# ---- collect the three series per group per seed (evaluation-only) ---------
E1S_RES = {}
for gname, gids in GROUPS.items():
    E1S_RES[gname] = {}
    for seed in SEEDS:
        r = final_runs[seed]
        r_comb = train_residual_progression_variant(     # reload, never retrain
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
            seg_id_subset=gids, shuffle=False, seed=0, include_ship_history=True)
        E1S_RES[gname][seed] = build_model_vs_captain_combined_accuracy(
            r["model"], r["repr_layer"], t_loader, r["core_and_alt_fn"], data,
            WORK_DIR, subregion_name_map,
            departure_ids_fn=r.get("departure_ids_fn"),
            eta_channel_lookup=r.get("eta_channel_lookup"),
            combined_model=r_comb["model"], combined_repr_layer=r_comb["repr_layer"],
            combined_core_and_alt_fn=r_comb["core_and_alt_fn"],
            combined_departure_ids_fn=r_comb.get("departure_ids_fn"),
            combined_eta_channel_lookup=r_comb.get("eta_channel_lookup"),
            set_label=f"test-{gname}", dedup_strategy=CAPTAIN_DEDUP_STRATEGY)
    print(f"{gname}: 3 seeds collected")

# ---- plots (house style) + combined table -----------------------------------
def _ms(res_by_seed, key):
    C = np.vstack([np.where(np.array(res_by_seed[s][key]["band_total"],
                                     dtype="float64") > 0,
                            np.array(res_by_seed[s][key]["progression_acc"],
                                     dtype="float64"), np.nan) for s in SEEDS])
    return np.nanmean(C, axis=0), np.nanstd(C, axis=0)

labels_5 = _progression_labels(tuple(sorted(DEFAULT_PROGRESSION_BOUNDARIES)))
tables = {}
for gname in GROUPS:
    curves_ms = {label: _ms(E1S_RES[gname], key) for key, label in SERIES}
    plot_three_way_benchmark_meanstd(
        curves_ms, WORK_DIR, majority_baseline=MAJORITY_BASELINE,
        title=f"Model vs captain vs combined -- {gname} (3-seed mean \u00b1 1 std)",
        subtitle=f"TEST [{TEST_START} -> {TEST_END}], captain: {CAPTAIN_DEDUP_STRATEGY}",
        save_name=f"e1_threeway_meanstd_{SPLIT_MODE}{N_GROUPS}_"
                  f"{gname.split(' ')[0].lower()}.png")
    for key, label in SERIES:
        m, s = _ms(E1S_RES[gname], key)
        tables[(gname, label)] = [f"{mm*100:.2f} \u00b1 {ss*100:.2f}"
                                  if np.isfinite(mm) else "--"
                                  for mm, ss in zip(m, s)]
e1_split = pd.DataFrame(tables, index=labels_5)
e1_split.columns = pd.MultiIndex.from_tuples(e1_split.columns,
                                             names=["Voyage length", "Series"])
display(HTML(f"<b>E1 by voyage duration ({N_GROUPS} groups) — accuracy per "
             f"5% bin (3-seed mean ± std)</b>"))
display(e1_split)
e1_split.to_csv(os.path.join(WORK_DIR,
    f"e1_threeway_5pct_by_{SPLIT_MODE}{N_GROUPS}_{CAPTAIN_DEDUP_STRATEGY}.csv"))
print(f"saved plots + CSV for {SPLIT_MODE} x {N_GROUPS}")
