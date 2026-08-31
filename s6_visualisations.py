# =============================================================================
# Step 6 — report visualisations (routes map, consecutive voyages, port call, correctness map)
# Migrated verbatim from Main_forGitHub.ipynb cells [110, 111, 113, 114, 115, 117, 119, 121, 122].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 110]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-PREREQ -- session-state check (run in a session built like Main_V4)
# =============================================================================
# This section consumes state from earlier Main_V4 sections. Fail loudly here
# rather than mid-plot. Needs: 0.A env -> 3.b (data, labels, subregion_names,
# history) -> 4.1 (_make_split) -> 5.1 L5-MIN -> E0 A (runs).
for _req in ("data", "subregion_names", "runs", "_make_split",
             "build_port_to_subregion_map", "_collect_full_predictions",
             "BucketedWAYDataset", "TEST_START", "TEST_END", "TARGET_COL",
             "BATCH_SIZE", "SEEDS"):
    assert _req in dir(), f"'{_req}' missing -- run the prerequisite sections first"
print("viz prerequisites present")


# =============================================================================
# VIZ-LIB -- plotting suite (7 defs, verbatim live Step4c; dead wrapper removed)
# =============================================================================
# Consumer map: _plot_routes_between_subregions_single -> Viz1-A/B
# (helpers: extract_segment_entries, _resolve_dataset_seg_id_filter);
# plot_consecutive_segments_for_vessel -> Viz2;
# find_consecutive_port_call_pairs + plot_port_traffic -> Viz3;
# get_port_name_map -> setup. Viz4 uses L5-MIN's _collect_full_predictions.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_port_name_map(step3data):
    """Reverse of step3data.vocab["port_to_id"] — port ID -> readable port
    name, for the same reason get_subregion_name_map exists: raw IDs
    aren't something you actually want to read in a segment-level table."""
    return {v: k for k, v in step3data.vocab["port_to_id"].items()}


def extract_segment_entries(model, repr_layer, step3data, seg_id, target_col="ARR_SUBREGION_ID",
                             extra_kwargs_fn=None, include_ship_history=True, subregion_name_map=None):
    """Runs inference on a SINGLE segment (by seg_id) and returns one row
    per actual grid-step of that voyage: the model's prediction made AT
    that step, whether it was correct, and the step's real-world
    coordinates and elapsed time (from step3data.steps_idx). This is the
    raw material for both a full audit trail of one voyage's entire
    prediction history and the map visualization below — unlike every
    other tool in this file, which only captures ONE step per segment,
    this captures ALL of them.

    extra_kwargs_fn: same callback convention as profile_predictions —
    required for models with n_alt_progression_signals>0 or
    use_departure_gate=True (e.g. r7).
    """
    inputs, mask = step3data.prepare_batch([seg_id], include_ship_history=include_ship_history)
    extra_kwargs = extra_kwargs_fn(inputs, [seg_id]) if extra_kwargs_fn is not None else {}
    with torch.no_grad():
        x = repr_layer(inputs)
        logits = model(x, key_padding_mask=mask, **extra_kwargs)
    preds = ops.convert_to_numpy(ops.argmax(logits, axis=-1))[0]  # [N]

    true_label = int(step3data.traj_idx.set_index("seg_id").loc[seg_id, target_col])

    seg_steps = step3data.steps_idx[step3data.steps_idx["SEG_ID"] == seg_id].sort_values("STEP_IDX").reset_index(drop=True)
    n_real = len(seg_steps)

    def _name(class_id):
        if subregion_name_map is None:
            return class_id
        return subregion_name_map.get(class_id, class_id)

    records = []
    for i in range(n_real):
        pred = int(preds[i])
        records.append({
            "seg_id": seg_id, "step_idx": i,
            "grid_lat": float(seg_steps.loc[i, "GRID_LAT_C"]), "grid_lon": float(seg_steps.loc[i, "GRID_LON_C"]),
            "time_offset_days": float(seg_steps.loc[i, "TIME_OFFSET_DAYS"]),
            "predicted_class": pred, "predicted_name": _name(pred),
            "true_class": true_label, "true_name": _name(true_label),
            "correct": pred == true_label,
        })
    return pd.DataFrame(records)


def _resolve_dataset_seg_id_filter(dataset, train_ids, val_ids, test_ids, seg_id_filter):
    """Shared logic behind every tool in this file that offers a
    dataset="train"/"val"/"test"/[combination] parameter. Returns the
    effective seg_id_filter: None if dataset=="all" and no manual filter
    was given, otherwise the union of the requested splits' ID sets (or
    the manual seg_id_filter itself, if that's what was passed instead).
    Centralized here so every caller raises the exact same errors for
    the exact same mistakes, rather than each tool re-implementing (and
    potentially drifting from) this validation independently.
    """
    if dataset != "all" and seg_id_filter is not None:
        raise ValueError("dataset and seg_id_filter are mutually exclusive — pass only one")
    if dataset == "all":
        return seg_id_filter

    requested = [dataset] if isinstance(dataset, str) else list(dataset)
    id_sources = {"train": train_ids, "val": val_ids, "test": test_ids}
    unknown = [r for r in requested if r not in id_sources]
    if unknown:
        raise ValueError(f"dataset must be 'all', or one/more of 'train'/'val'/'test', got {unknown!r}")
    missing = [r for r in requested if id_sources[r] is None]
    if missing:
        raise ValueError(f"dataset includes {missing!r} but the corresponding _ids argument "
                          f"({', '.join(m + '_ids' for m in missing)}) was not passed in")
    resolved = set()
    for r in requested:
        resolved |= set(id_sources[r])
    return resolved


def _plot_routes_between_subregions_single(step3data, subregion_name_map, load_subregion_name=None, arrival_subregion_name=None,
                                    target_col="ARR_SUBREGION_ID", dataset="all",
                                    train_ids=None, val_ids=None, test_ids=None,
                                    seg_id_filter=None, max_segments=None,
                                    window_start=None, window_end=None, window_mode="overlap",
                                    highlight_seg_ids=None, model=None, repr_layer=None, extra_kwargs_fn=None,
                                    include_ship_history=True, title=None, save_path=None):
    """Single-window implementation -- see plot_routes_between_subregions
    (the public entry point) for the full docstring and for MULTIPLE-
    time-window support (window_start/window_end as paired lists ->
    one map per window). This private version always takes window_start/
    window_end as single values or None, never lists.
    """
    import plotly.graph_objects as go

    if dataset != "all" and seg_id_filter is not None:
        raise ValueError("dataset and seg_id_filter are mutually exclusive — pass only one")
    if highlight_seg_ids is not None and (model is None or repr_layer is None):
        raise ValueError("highlight_seg_ids requires model and repr_layer to be passed in")
    if highlight_seg_ids is not None and len(highlight_seg_ids) > 10:
        raise ValueError(f"got {len(highlight_seg_ids)} highlight_seg_ids, max is 10 — pick a smaller set")
    if (load_subregion_name is None) != (arrival_subregion_name is None):
        raise ValueError("load_subregion_name and arrival_subregion_name must both be None (no background "
                          "routes) or both be provided together, paired positionally — got only one of the two")
    if load_subregion_name is None and not highlight_seg_ids:
        raise ValueError("Nothing to plot: load_subregion_name/arrival_subregion_name are both None (no "
                          "background routes) AND highlight_seg_ids is also empty/None. Provide at least one.")
    if (window_start is None) != (window_end is None):
        raise ValueError("window_start and window_end must both be None or both be provided -- got only one of the two")
    if window_mode not in ("overlap", "departure"):
        raise ValueError(f"window_mode must be 'overlap' or 'departure', got {window_mode!r}")

    seg_id_filter = _resolve_dataset_seg_id_filter(dataset, train_ids, val_ids, test_ids, seg_id_filter)

    if load_subregion_name is None:
        load_list, arrival_list = [], []
    else:
        load_list = [load_subregion_name] if isinstance(load_subregion_name, str) else list(load_subregion_name)
        arrival_list = [arrival_subregion_name] if isinstance(arrival_subregion_name, str) else list(arrival_subregion_name)
        if len(load_list) != len(arrival_list):
            raise ValueError(f"load_subregion_name ({len(load_list)}) and arrival_subregion_name "
                              f"({len(arrival_list)}) must have the same length, paired positionally")

    name_to_id = {v: k for k, v in subregion_name_map.items()}
    unknown_names = [n for n in load_list + arrival_list if n not in name_to_id]
    if unknown_names:
        raise ValueError(f"{unknown_names!r} not found. Valid names: {sorted(name_to_id.keys())}")

    # Curated high-contrast palette -- maximally distinguishable, not matplotlib/plotly's default cycle
    route_palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                     "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324"]

    port_to_subregion = build_port_to_subregion_map(step3data.traj_idx, subregion_col=target_col, port_col="ARR_PORT_ID")
    seg_meta = step3data.traj_idx.set_index("seg_id")
    dep_subregion = seg_meta["DEP_PORT_ID"].map(port_to_subregion)

    window_mask = None
    if window_start is not None:
        window_start_ts, window_end_ts = pd.Timestamp(window_start), pd.Timestamp(window_end)
        if window_start_ts > window_end_ts:
            raise ValueError(f"window_start ({window_start_ts}) is after window_end ({window_end_ts})")
        dep_ts = pd.to_datetime(seg_meta["dep_ts"])
        if window_mode == "overlap":
            arr_ts = pd.to_datetime(seg_meta["arr_ts"])
            window_mask = (dep_ts <= window_end_ts) & (arr_ts >= window_start_ts)
        else:
            window_mask = (dep_ts >= window_start_ts) & (dep_ts <= window_end_ts)

    fig = go.Figure()
    route_summaries = []
    for i, (load_name, arrival_name) in enumerate(zip(load_list, arrival_list)):
        load_id, arrival_id = name_to_id[load_name], name_to_id[arrival_name]
        matching = seg_meta[(dep_subregion == load_id) & (seg_meta[target_col] == arrival_id)]
        if window_mask is not None:
            matching = matching[window_mask]
        seg_ids = matching.index.tolist()
        if seg_id_filter is not None:
            seg_id_filter_set = set(seg_id_filter)
            seg_ids = [s for s in seg_ids if s in seg_id_filter_set]
        total_found = len(seg_ids)
        if max_segments is not None and len(seg_ids) > max_segments:
            seg_ids = seg_ids[:max_segments]
        if len(seg_ids) == 0:
            route_summaries.append(f"{load_name}→{arrival_name}: 0 voyages")
            continue

        color = route_palette[i % len(route_palette)]
        route_name = f"{load_name} → {arrival_name} (n={len(seg_ids)})"
        # A dedicated, fully-opaque, thicker-line trace JUST for the
        # legend swatch -- draws nothing on the map itself (no lon/lat
        # data), so it doesn't add visual clutter, but gives the legend
        # entry a clearly visible line instead of inheriting the faint
        # opacity=0.25 used for the actual density lines below. Plotly
        # legend swatches otherwise directly mirror the trace's own
        # opacity/width, which is deliberately low for the map itself
        # (so many overlapping voyages read as density) but was hard to
        # even see as a color swatch in the legend.
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None], mode="lines",
            line=dict(width=4, color=color), opacity=1.0,
            name=route_name, legendgroup=f"route_{i}", showlegend=True, hoverinfo="skip",
        ))
        for j, seg_id in enumerate(seg_ids):
            seg_steps = step3data.steps_idx[step3data.steps_idx["SEG_ID"] == seg_id].sort_values("STEP_IDX")
            if len(seg_steps) == 0:
                continue
            imo_raw = seg_meta.loc[seg_id, "IMO"] if "IMO" in seg_meta.columns else np.nan
            # IMO columns commonly upcast to float64 once any row is NaN (a mix of a
            # real ID and a missing value forces the whole column to float) -- cast
            # back to int for display so it reads "9876543", not "9876543.0"
            imo = int(imo_raw) if pd.notna(imo_raw) else "unknown"
            duration_h = seg_meta.loc[seg_id, "duration_h"] if "duration_h" in seg_meta.columns else np.nan
            duration_str = f"{duration_h / 24.0:.1f} days" if pd.notna(duration_h) else "duration unknown"
            hover = f"seg_id={seg_id}<br>IMO={imo}<br>{duration_str}"
            fig.add_trace(go.Scattergeo(
                lon=seg_steps["GRID_LON_C"], lat=seg_steps["GRID_LAT_C"], mode="lines",
                line=dict(width=1.2, color=color), opacity=0.25,
                showlegend=False, name=route_name,
                legendgroup=f"route_{i}", hoverinfo="text", text=[hover] * len(seg_steps),
            ))
        shown_note = "" if len(seg_ids) == total_found else f" ({len(seg_ids)} of {total_found} shown)"
        route_summaries.append(f"{load_name}→{arrival_name}: {len(seg_ids)} voyages{shown_note}")

    if len(fig.data) == 0 and highlight_seg_ids is None:
        raise ValueError(f"no segments found for any of the requested routes ({list(zip(load_list, arrival_list))})"
                          + (" within the requested dataset/filter" if (seg_id_filter is not None or dataset != "all") else ""))

    if highlight_seg_ids is not None:
        for seg_id in highlight_seg_ids:
            entries = extract_segment_entries(model, repr_layer, step3data, seg_id, target_col=target_col,
                                               extra_kwargs_fn=extra_kwargs_fn, include_ship_history=include_ship_history,
                                               subregion_name_map=subregion_name_map)
            if len(entries) == 0:
                continue
            fig.add_trace(go.Scattergeo(
                lon=entries["grid_lon"], lat=entries["grid_lat"], mode="lines",
                line=dict(width=2, color="rgba(60,60,60,0.8)"), showlegend=False,
            ))
            for correct_flag, color, label in [(True, "#2ca02c", "Correct"), (False, "#d62728", "Incorrect")]:
                sub = entries[entries["correct"] == correct_flag]
                if len(sub) == 0:
                    continue
                hover_text = [f"segment {seg_id} — step {r.step_idx} — predicted: {r.predicted_name}, true: {r.true_name}"
                              for r in sub.itertuples()]
                fig.add_trace(go.Scattergeo(
                    lon=sub["grid_lon"], lat=sub["grid_lat"], mode="markers",
                    marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                    name=label, text=hover_text, hoverinfo="text",
                    showlegend=(seg_id == highlight_seg_ids[0]),
                ))

    dataset_label = "all data" if dataset == "all" else (dataset if isinstance(dataset, str) else "+".join(dataset))
    subtitle = "; ".join(route_summaries)
    if subtitle:
        auto_title = f"{subtitle} ({dataset_label})"
    elif highlight_seg_ids:
        auto_title = f"Highlighted segments: {', '.join(str(s) for s in highlight_seg_ids)}"
    else:
        auto_title = "Route map"  # unreachable given the earlier "nothing to plot" guard, but a safe fallback regardless
    if window_start is not None:
        auto_title += f" [{pd.Timestamp(window_start).strftime('%Y-%m-%d')} → {pd.Timestamp(window_end).strftime('%Y-%m-%d')}]"
    fig.update_layout(
        # x=0.5 would center on the FULL figure, but the legend eats
        # into the right side of that space -- centering on 0.5 made
        # the title look visibly off-center relative to the map itself.
        # geo.domain reserves a known, fixed width for the map (78%,
        # leaving 22% for the legend), and the title's own x is set to
        # the CENTER of that same domain, not the full figure.
        title=dict(text=title or auto_title,
                    x=0.39, xanchor="center", font=dict(size=18), y=0.98, yanchor="top"),
        geo=dict(showland=True, landcolor="rgb(235,235,235)", showocean=True, oceancolor="rgb(225,242,255)",
                  showcountries=True, countrycolor="rgb(210,210,210)", projection_type="natural earth",
                  domain=dict(x=[0, 0.78], y=[0, 1])),
        height=650,
        margin=dict(t=50, b=10, l=10, r=10),  # reduced from 110 -- was leaving a large empty gap above the map
    )
    if save_path:
        fig.write_html(save_path)
        print(f"Saved -> {save_path}")
    fig.show()
    return fig


def plot_consecutive_segments_for_vessel(step3data, subregion_name_map, port_name_map, work_dir, imo,
                                          n_segments=3, start_seg_id=None,
                                          target_col="ARR_SUBREGION_ID", port_radius_km=20,
                                          title=None, save_path=None):
    """Plots N CONSECUTIVE voyages for ONE vessel (by IMO) — a close-up,
    zoomable, per-ping view of a short back-to-back voyage sequence,
    unlike plot_segments_by_imo_and_window (which can span many vessels
    and a whole time window at once, at per-STEP granularity). Built
    for questions like "what did this ship actually do across its last
    3 voyages" — usually 2-3 segments is the useful range, though any
    n_segments >= 1 works.

    n_segments: how many consecutive voyages to show. If the vessel has
    fewer than this many segments available (from start_seg_id onward,
    or in total if start_seg_id is None), shows whatever's available
    and prints a note -- not an error, since this is a real, expected
    data limitation, not a misuse of the function.

    start_seg_id: optional anchor. If given, shows THIS segment plus
    the next (n_segments - 1) segments chronologically AFTER it, for
    this same vessel -- must be one of this IMO's own segments, or
    raises a clear error listing what its real segments are. If None
    (the default), shows the LAST n_segments -- this vessel's most
    recent voyages.

    PER-PING detail: unlike every other map function in this file
    (which plot the per-STEP aggregated position, GRID_LAT_C/GRID_LON_C
    -- one point per grid-step, however many raw AIS pings landed in
    it), this reads trajectories_gridded.parquet's own raw LAT/LON
    columns directly and plots EVERY individual ping as its own marker,
    connected in chronological order (STEP_IDX, then SUBSEQ_STEP_IDX,
    then TIMESTAMP) -- the actual, ping-by-ping track, not a
    simplification of it.

    DISCHARGE PORT + RADIUS: for each segment's own arrival port,
    computes a representative port location the same way
    PortLocationIndex itself does (median of the LAST recorded position
    across every segment that ever arrived there, not a lookup against
    an external port-reference file), places a labeled marker there,
    and draws a circle of port_radius_km around it (default 20km, no
    single correct "port zone" size exists in this data -- adjust to
    taste). Multiple consecutive segments arriving at the SAME port
    only get one marker/circle, not one per segment.

    Renders with a mercator projection (unlike the other route maps'
    natural-earth) and fitbounds="locations" -- mercator handles
    regional zoom/pan far better for a close-up view like this one, and
    fitbounds starts the map already zoomed to the plotted voyages
    rather than showing the whole world by default.
    """
    import plotly.graph_objects as go

    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")

    seg_meta = step3data.traj_idx.set_index("seg_id")
    if "IMO" not in seg_meta.columns:
        raise ValueError("traj_idx has no 'IMO' column -- cannot select segments by vessel")

    imo_segs = seg_meta[seg_meta["IMO"] == imo].copy()
    if len(imo_segs) == 0:
        raise ValueError(f"No segments found for IMO {imo}")
    imo_segs["_dep_ts_parsed"] = pd.to_datetime(imo_segs["dep_ts"])
    imo_segs = imo_segs.sort_values("_dep_ts_parsed")
    ordered_seg_ids = imo_segs.index.tolist()

    if start_seg_id is not None:
        if start_seg_id not in ordered_seg_ids:
            raise ValueError(f"start_seg_id={start_seg_id} is not one of IMO {imo}'s own segments. "
                              f"This vessel's segments (chronological): {ordered_seg_ids}")
        start_pos = ordered_seg_ids.index(start_seg_id)
        selected_seg_ids = ordered_seg_ids[start_pos:start_pos + n_segments]
    else:
        selected_seg_ids = ordered_seg_ids[-n_segments:]

    if len(selected_seg_ids) < n_segments:
        print(f"NOTE: only {len(selected_seg_ids)} of the requested {n_segments} segment(s) available for "
              f"IMO {imo} (this vessel has {len(ordered_seg_ids)} segment(s) total in this dataset).")

    # Raw, per-PING data (LAT/LON) for exactly these segments -- not the
    # per-step aggregated GRID_LAT_C/GRID_LON_C every other map function
    # in this file uses.
    gridded_path = os.path.join(work_dir, Step3b_representation_layer.DATA_SUBFOLDER, "trajectories_gridded.parquet")
    if not os.path.exists(gridded_path):
        raise FileNotFoundError(f"{gridded_path} not found -- raw per-ping LAT/LON lives only in the raw "
                                 f"gridded file, not in steps_idx")
    import pyarrow.parquet as pq
    available_cols = pq.ParquetFile(gridded_path).schema.names
    needed = {"SEG_ID", "STEP_IDX", "LAT", "LON", "TIMESTAMP"}
    missing = needed - set(available_cols)
    if missing:
        raise ValueError(f"trajectories_gridded.parquet is missing columns needed for per-ping plotting: {missing}")
    sort_cols = ["SEG_ID", "STEP_IDX"] + (["SUBSEQ_STEP_IDX"] if "SUBSEQ_STEP_IDX" in available_cols else []) + ["TIMESTAMP"]
    read_cols = list(dict.fromkeys(["SEG_ID", "STEP_IDX", "LAT", "LON", "TIMESTAMP"] +
                                     (["SUBSEQ_STEP_IDX"] if "SUBSEQ_STEP_IDX" in available_cols else [])))
    pings = pd.read_parquet(gridded_path, columns=read_cols)
    pings = pings[pings["SEG_ID"].isin(selected_seg_ids)].sort_values(sort_cols)

    # Port locations, same methodology as PortLocationIndex itself
    # (median of each segment's own LAST step position, grouped by
    # ARR_PORT_ID) -- computed directly here rather than reaching into
    # that class's own private _tree, since it exposes no public
    # port_id -> (lat, lon) lookup.
    last_step = step3data.steps_idx.loc[step3data.steps_idx.groupby("SEG_ID")["STEP_IDX"].idxmax()]
    port_merge = last_step.merge(
        step3data.traj_idx[["seg_id", "ARR_PORT_ID"]].dropna(subset=["ARR_PORT_ID"]),
        left_on="SEG_ID", right_on="seg_id", how="inner")
    port_merge["ARR_PORT_ID"] = port_merge["ARR_PORT_ID"].astype(int)
    port_locations = port_merge.groupby("ARR_PORT_ID")[["GRID_LAT_C", "GRID_LON_C"]].median()

    def _circle_points_km(center_lat, center_lon, radius_km, n_points=60):
        R = 6371.0
        lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
        d_r = radius_km / R
        bearings = np.linspace(0, 2 * np.pi, n_points)
        lat2 = np.arcsin(np.sin(lat1) * np.cos(d_r) + np.cos(lat1) * np.sin(d_r) * np.cos(bearings))
        lon2 = lon1 + np.arctan2(np.sin(bearings) * np.sin(d_r) * np.cos(lat1),
                                   np.cos(d_r) - np.sin(lat1) * np.sin(lat2))
        return np.degrees(lat2), np.degrees(lon2)

    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324"]
    port_to_subregion = build_port_to_subregion_map(step3data.traj_idx, subregion_col=target_col, port_col="ARR_PORT_ID")

    fig = go.Figure()
    legend_rows = []
    ports_drawn = set()
    for i, seg_id in enumerate(selected_seg_ids):
        seg_pings = pings[pings["SEG_ID"] == seg_id]
        if len(seg_pings) == 0:
            print(f"WARNING: seg_id={seg_id} has no raw ping rows in trajectories_gridded.parquet -- skipped")
            continue
        color = palette[i % len(palette)]

        row = seg_meta.loc[seg_id]
        dep_port_id, arr_port_id = row["DEP_PORT_ID"], row["ARR_PORT_ID"]
        load_port = port_name_map.get(dep_port_id, str(dep_port_id))
        disch_port = port_name_map.get(arr_port_id, str(arr_port_id))
        load_subregion = subregion_name_map.get(port_to_subregion.get(dep_port_id), "unknown")
        disch_subregion = subregion_name_map.get(row[target_col], "unknown")
        dep_date = pd.Timestamp(row["dep_ts"]).strftime("%Y-%m-%d") if pd.notna(row["dep_ts"]) else "unknown"
        arr_date = pd.Timestamp(row["arr_ts"]).strftime("%Y-%m-%d") if pd.notna(row.get("arr_ts")) else "unknown"
        duration_h = row["duration_h"] if "duration_h" in seg_meta.columns else np.nan
        journey_length = f"{duration_h / 24.0:.1f} days" if pd.notna(duration_h) else "unknown"
        legend_name = (f"#{seg_id}: {load_port} ({load_subregion}) → {disch_port} ({disch_subregion}) "
                       f"— dep. {dep_date}")
        legend_rows.append({"seg_id": seg_id, "IMO": imo, "load_port": load_port, "load_subregion": load_subregion,
                             "discharge_port": disch_port, "discharge_subregion": disch_subregion,
                             "n_pings": len(seg_pings), "departure_date": dep_date, "arrival_date": arr_date,
                             "journey_length": journey_length})

        hover_text = [f"seg_id={seg_id}<br>ping {j+1}/{len(seg_pings)}<br>{ts}<br>lat={lat:.3f}, lon={lon:.3f}"
                      for j, (ts, lat, lon) in enumerate(zip(seg_pings["TIMESTAMP"], seg_pings["LAT"], seg_pings["LON"]))]
        fig.add_trace(go.Scattergeo(
            lon=seg_pings["LON"], lat=seg_pings["LAT"], mode="lines+markers",
            line=dict(width=2, color=color), marker=dict(size=5, color=color, line=dict(width=1, color="white")),
            opacity=0.9, name=legend_name, legendgroup=f"seg_{seg_id}", showlegend=True,
            hoverinfo="text", text=hover_text,
        ))

        # Segment number label at the ping track's own midpoint.
        mid_idx = len(seg_pings) // 2
        fig.add_trace(go.Scattergeo(
            lon=[seg_pings["LON"].iloc[mid_idx]], lat=[seg_pings["LAT"].iloc[mid_idx]],
            mode="markers+text", text=[str(seg_id)], textposition="top center",
            textfont=dict(size=13, color=color, family="Arial Black"),
            marker=dict(size=6, color=color, line=dict(width=1.5, color="white")),
            showlegend=False, legendgroup=f"seg_{seg_id}", hoverinfo="skip",
        ))

        # Discharge port marker + radius circle -- once per distinct
        # port, even if several consecutive segments share the same one.
        if arr_port_id in ports_drawn or arr_port_id not in port_locations.index:
            continue
        ports_drawn.add(arr_port_id)
        port_lat, port_lon = port_locations.loc[arr_port_id, ["GRID_LAT_C", "GRID_LON_C"]]
        fig.add_trace(go.Scattergeo(
            lon=[port_lon], lat=[port_lat], mode="markers+text",
            text=[disch_port], textposition="bottom center", textfont=dict(size=11, color="#333333"),
            marker=dict(size=10, color="#333333", symbol="square", line=dict(width=1, color="white")),
            name=f"Port: {disch_port}", legendgroup=f"port_{arr_port_id}", showlegend=True,
            hoverinfo="text",
        ))
        circle_lat, circle_lon = _circle_points_km(port_lat, port_lon, port_radius_km)
        fig.add_trace(go.Scattergeo(
            lon=circle_lon, lat=circle_lat, mode="lines",
            line=dict(width=1, color="#333333", dash="dot"), opacity=0.6,
            name=f"{port_radius_km}km radius", legendgroup=f"port_{arr_port_id}", showlegend=False,
            hoverinfo="skip",
        ))

    if len(fig.data) == 0:
        raise ValueError(f"No plottable ping data found for any of the selected segments {selected_seg_ids}")

    auto_title = f"IMO {imo}: {len(selected_seg_ids)} consecutive segment(s) ({', '.join(str(s) for s in selected_seg_ids)})"
    fig.update_layout(
        # Legend sits INSIDE the map, bottom-right corner (x/y in paper
        # coordinates, anchored to that corner) -- not beside it, so
        # the map itself can use the full canvas (geo.domain back to
        # [0,1], title back to full-figure center at x=0.5) rather than
        # reserving a separate strip. bgcolor keeps the legend text
        # readable over whatever's plotted underneath it.
        title=dict(text=title or auto_title, x=0.5, xanchor="center", font=dict(size=18), y=0.98, yanchor="top"),
        geo=dict(showland=True, landcolor="rgb(235,235,235)", showocean=True, oceancolor="rgb(225,242,255)",
                  showcountries=True, countrycolor="rgb(210,210,210)", projection_type="mercator",
                  domain=dict(x=[0, 1], y=[0, 1]), fitbounds="locations"),
        height=650,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom", font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
    )
    if save_path:
        fig.write_html(save_path)
        print(f"Saved -> {save_path}")
    fig.show()

    legend_table = pd.DataFrame(legend_rows)
    print(f"\n{len(legend_rows)} segment(s) plotted for IMO {imo}:")
    print(legend_table.to_string(index=False))
    return fig, legend_table


def find_consecutive_port_call_pairs(step3data, port_name, max_dwell_days=None):
    """Finds (inbound, outbound) segment PAIRS for the SAME vessel where
    it arrived at port_name, then departed from that SAME port on its
    own VERY NEXT segment -- a genuine "arrived, then later left this
    same port" port call, not just any two segments that happen to
    each touch this port independently, possibly with other, unrelated
    voyages in between.

    "Consecutive" is precise here, not approximate: for each vessel's
    own FULL, chronologically-sorted segment history (not just the
    segments touching this port -- the full history is needed to
    correctly tell true adjacency apart from "this vessel visited here
    twice, with an unrelated voyage elsewhere in between"), looks for
    seg[i].arr_port == port_name AND seg[i+1].dep_port == port_name,
    where seg[i+1] is that SAME vessel's own immediately-next segment
    after seg[i] -- not merely some later segment that also happens to
    depart from here.

    max_dwell_days: optional -- excludes pairs where the gap between
    the inbound segment's own arrival and the outbound segment's own
    departure exceeds this many days. None (default): no cap, every
    genuinely consecutive pair is returned regardless of how long the
    vessel appears to have stayed.

    Returns a DataFrame (sorted chronologically by arrival_date), each
    row: IMO, inbound_seg_id, outbound_seg_id, previous_port (where the
    inbound segment came FROM), next_port (where the outbound segment
    goes TO), arrival_date, departure_date, dwell_days (computed
    directly from the two segments' own timestamps, not an estimate).
    Feed inbound_seg_id/outbound_seg_id straight into plot_port_traffic's
    own seg_ids parameter to plot exactly one such pair.
    """
    for col in ("IMO", "dep_port", "arr_port", "seg_id", "dep_ts", "arr_ts"):
        if col not in step3data.traj_idx.columns:
            raise ValueError(f"traj_idx is missing {col!r} -- cannot find port-call pairs")

    seg_meta = step3data.traj_idx.copy()
    seg_meta["dep_ts"] = pd.to_datetime(seg_meta["dep_ts"])
    seg_meta["arr_ts"] = pd.to_datetime(seg_meta["arr_ts"])
    seg_meta = seg_meta.sort_values(["IMO", "dep_ts"]).reset_index(drop=True)

    pairs = []
    for imo, group in seg_meta.groupby("IMO"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            seg_in, seg_out = group.iloc[i], group.iloc[i + 1]
            if seg_in["arr_port"] == port_name and seg_out["dep_port"] == port_name:
                dwell_days = (seg_out["dep_ts"] - seg_in["arr_ts"]).total_seconds() / 86400.0
                pairs.append({
                    "IMO": imo, "inbound_seg_id": seg_in["seg_id"], "outbound_seg_id": seg_out["seg_id"],
                    "previous_port": seg_in["dep_port"], "next_port": seg_out["arr_port"],
                    "arrival_date": seg_in["arr_ts"].strftime("%Y-%m-%d") if pd.notna(seg_in["arr_ts"]) else "unknown",
                    "departure_date": seg_out["dep_ts"].strftime("%Y-%m-%d") if pd.notna(seg_out["dep_ts"]) else "unknown",
                    "dwell_days": round(dwell_days, 2),
                    "_arrival_sort": seg_in["arr_ts"],
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) == 0:
        print(f"No consecutive in/out pairs found for {port_name!r}.")
        return pairs_df

    if max_dwell_days is not None:
        n_before = len(pairs_df)
        pairs_df = pairs_df[pairs_df["dwell_days"] <= max_dwell_days]
        n_dropped = n_before - len(pairs_df)
        if n_dropped:
            print(f"Excluded {n_dropped} pair(s) with dwell_days > {max_dwell_days} (max_dwell_days cap)")

    pairs_df = pairs_df.sort_values("_arrival_sort").drop(columns="_arrival_sort").reset_index(drop=True)
    print(f"{len(pairs_df)} consecutive in/out pair(s) found for {port_name!r}:")
    print(pairs_df.to_string(index=False))
    return pairs_df


def plot_port_traffic(step3data, subregion_name_map, port_name_map, work_dir, port_name,
                       target_col="ARR_SUBREGION_ID", direction="both", seg_ids=None, max_segments=20,
                       show_grid=True, title=None, save_path=None):
    """Plots INBOUND and/or OUTBOUND voyage segments for ONE PORT — the
    port-centric counterpart to plot_consecutive_segments_for_vessel's
    own vessel-centric view. Same per-ping detail, segment-number
    labels, port radius zone, and grid delimitation as before.

    seg_ids: optional -- an explicit list of specific segment IDs to
    plot, instead of every segment matching direction automatically.
    Every ID given must genuinely touch port_name as either its load or
    discharge port, or a clear error names exactly which ones don't --
    silently ignoring a wrong ID would be worse than refusing. When
    given, direction is IGNORED for selection (each segment's own real
    inbound/outbound status is still detected and labeled correctly --
    seg_ids only changes WHICH segments are shown, not how they're
    classified). The natural pairing: find_consecutive_port_call_pairs
    finds genuine "arrived, then departed from here" pairs for a given
    port; feed its own inbound_seg_id/outbound_seg_id straight into
    seg_ids=[...] to plot exactly one such pair, rather than every
    segment ever touching this port at once.

    port_name: must match a canonical_name EXACTLY as it appears in
    lpg_port_reference_fixed.csv, and in traj_idx's own dep_port/
    arr_port columns — these are ALREADY the same canonical_name
    values (confirmed directly in this project's own Step2b
    segmentation stage), not a separate name space needing fuzzy
    matching. A non-matching name raises a clear error listing example
    real names, rather than silently finding nothing.

    direction: "both" (default) — every segment where this port is
    EITHER the load or discharge port. "inbound" — only segments
    arriving here. "outbound" — only segments departing from here.
    Each segment's own legend entry and hover text is labeled with
    which one it is.

    max_segments: hard cap (default 20) — a busy port could easily
    match far more segments than are readable on one map. Raises a
    clear error naming the actual count if exceeded, rather than
    silently truncating.

    PORT LOCATION + RADIUS: read directly from
    lpg_port_reference_fixed.csv (work_dir/lpg_port_reference_fixed.csv
    — this project's own Step 1b output, not the Model_Inputs
    subfolder), using its own lat/lon and radius_nm columns — the real,
    curated port location and approach-zone radius this project's own
    preprocessing pipeline already established, not an approximation
    from AIS pings or a guessed default. radius_nm is genuinely in
    NAUTICAL MILES; converted to km via the exact, standard 1 NM =
    1.852 km for the circle-drawing math, and displayed in NM (not
    converted) in the legend, matching the source data's own units. A
    port with radius_nm <= 0 (a "waypoint" entry in the reference —
    transit-only, not a real port zone, per this project's own Step 1b
    labeling) gets a marker but no circle, with a clear note explaining
    why.

    GRID DELIMITATION, TWO LAYERS: (1) the grid PATH — this project's
    own documented 1° x 1° grid (GRID_CELL_SIZE_DEG), each segment's
    own sequence of distinct grid cells connected in visiting order
    (STEP_IDX), one dashed line per segment, sharing that segment's own
    color and legend entry; (2) grid cell BOUNDARIES — every distinct
    cell any plotted segment passed through, drawn as a light rectangle
    outline (no ordering implied, just which cells were visited at
    all). The cell size itself is this project's own documented
    constant, not derived — but is still cross-checked empirically
    against the actual loaded data (same GRID_LAT_IDX-to-GRID_LAT_C
    relationship, confirmed exact on a controlled test) and prints a
    clear warning if the two disagree, rather than silently trusting a
    constant that might not apply to whatever file actually loaded. Set
    show_grid=False to skip both layers.
    """
    import plotly.graph_objects as go

    if direction not in ("both", "inbound", "outbound"):
        raise ValueError(f"direction must be 'both', 'inbound', or 'outbound', got {direction!r}")

    port_ref_path = os.path.join(work_dir, "lpg_port_reference_fixed.csv")
    if not os.path.exists(port_ref_path):
        raise FileNotFoundError(f"{port_ref_path} not found -- expected directly in work_dir (this project's "
                                 f"own Step 1b output location, not the Model_Inputs subfolder)")
    port_ref = pd.read_csv(port_ref_path)
    for col in ("canonical_name", "lat", "lon", "radius_nm"):
        if col not in port_ref.columns:
            raise ValueError(f"lpg_port_reference_fixed.csv is missing expected column {col!r}")
    port_ref_row = port_ref[port_ref["canonical_name"] == port_name]
    if len(port_ref_row) == 0:
        sample_names = sorted(port_ref["canonical_name"].astype(str).unique())[:15]
        raise ValueError(f"Port {port_name!r} not found in lpg_port_reference_fixed.csv's own canonical_name "
                          f"column. Example available names: {sample_names}")
    port_lat = float(port_ref_row.iloc[0]["lat"])
    port_lon = float(port_ref_row.iloc[0]["lon"])
    port_radius_nm = float(port_ref_row.iloc[0]["radius_nm"])
    if port_radius_nm <= 0:
        print(f"NOTE: {port_name}'s own radius_nm is {port_radius_nm} (a 'waypoint' entry in the port "
              f"reference -- transit-only, not a real port zone) -- no radius circle will be drawn.")

    seg_meta = step3data.traj_idx.set_index("seg_id")
    if "dep_port" not in seg_meta.columns or "arr_port" not in seg_meta.columns:
        raise ValueError("traj_idx has no 'dep_port'/'arr_port' columns -- cannot match segments against this port")

    inbound_mask = seg_meta["arr_port"] == port_name
    outbound_mask = seg_meta["dep_port"] == port_name

    if seg_ids is not None:
        # Explicit selection -- validate every ID genuinely touches
        # this port as either its load or discharge, rather than
        # silently plotting something unrelated to port_name. direction
        # is not used to FILTER here (the caller already made an exact
        # choice); each segment's own real inbound/outbound status is
        # still detected below for labeling.
        missing = [sid for sid in seg_ids if sid not in seg_meta.index]
        if missing:
            raise ValueError(f"seg_ids contains segment(s) not found in traj_idx at all: {missing}")
        not_touching = [sid for sid in seg_ids
                         if not (inbound_mask.get(sid, False) or outbound_mask.get(sid, False))]
        if not_touching:
            raise ValueError(f"seg_ids contains segment(s) that don't touch {port_name!r} as either load or "
                              f"discharge port: {not_touching}. Check these are the right IDs, or the right port.")
        selected_seg_ids = list(seg_ids)
    else:
        if direction == "inbound":
            selection_mask = inbound_mask
        elif direction == "outbound":
            selection_mask = outbound_mask
        else:
            selection_mask = inbound_mask | outbound_mask
        selected = seg_meta[selection_mask]
        if len(selected) == 0:
            raise ValueError(f"No {direction} segments found for port {port_name!r}")
        selected_seg_ids = selected.index.tolist()

    directions_by_seg = {sid: ("inbound" if inbound_mask.get(sid, False) else "outbound") for sid in selected_seg_ids}
    if len(selected_seg_ids) > max_segments:
        raise ValueError(f"{len(selected_seg_ids)} segment(s) selected for {port_name!r} -- exceeds "
                          f"max_segments={max_segments}. Narrow direction/seg_ids, or raise max_segments "
                          f"explicitly if you genuinely want all {len(selected_seg_ids)}.")

    gridded_path = os.path.join(work_dir, Step3b_representation_layer.DATA_SUBFOLDER, "trajectories_gridded.parquet")
    if not os.path.exists(gridded_path):
        raise FileNotFoundError(f"{gridded_path} not found -- raw per-ping LAT/LON lives only in the raw "
                                 f"gridded file, not in steps_idx")
    import pyarrow.parquet as pq
    available_cols = pq.ParquetFile(gridded_path).schema.names
    needed = {"SEG_ID", "STEP_IDX", "LAT", "LON", "TIMESTAMP"}
    missing = needed - set(available_cols)
    if missing:
        raise ValueError(f"trajectories_gridded.parquet is missing columns needed for per-ping plotting: {missing}")
    grid_cols = {"GRID_LAT_IDX", "GRID_LON_IDX", "GRID_LAT_C", "GRID_LON_C"}
    has_grid_cols = show_grid and grid_cols.issubset(available_cols)
    sort_cols = ["SEG_ID", "STEP_IDX"] + (["SUBSEQ_STEP_IDX"] if "SUBSEQ_STEP_IDX" in available_cols else []) + ["TIMESTAMP"]
    read_cols = list(dict.fromkeys(
        ["SEG_ID", "STEP_IDX", "LAT", "LON", "TIMESTAMP"] +
        (["SUBSEQ_STEP_IDX"] if "SUBSEQ_STEP_IDX" in available_cols else []) +
        (list(grid_cols) if has_grid_cols else [])))
    pings = pd.read_parquet(gridded_path, columns=read_cols)
    pings = pings[pings["SEG_ID"].isin(selected_seg_ids)].sort_values(sort_cols)

    def _derive_grid_cell_size(sample):
        # Empirical cross-check against GRID_CELL_SIZE_DEG, not the
        # primary source of truth (that's the documented constant
        # itself) -- confirmed exact on a controlled test with a known
        # cell size before ever being used against real data.
        unique_lat = sample.drop_duplicates(subset=["GRID_LAT_IDX"])[["GRID_LAT_IDX", "GRID_LAT_C"]].sort_values("GRID_LAT_IDX")
        unique_lon = sample.drop_duplicates(subset=["GRID_LON_IDX"])[["GRID_LON_IDX", "GRID_LON_C"]].sort_values("GRID_LON_IDX")
        if len(unique_lat) < 2 or len(unique_lon) < 2:
            return None, None
        lat_size = np.median(np.diff(unique_lat["GRID_LAT_C"]) / np.diff(unique_lat["GRID_LAT_IDX"]))
        lon_size = np.median(np.diff(unique_lon["GRID_LON_C"]) / np.diff(unique_lon["GRID_LON_IDX"]))
        return abs(lat_size), abs(lon_size)

    lat_cell_size = lon_cell_size = None
    if has_grid_cols:
        # GRID_CELL_SIZE_DEG (1° x 1°) is this project's own documented
        # gridding methodology, not derived -- used directly here. Still
        # cross-checked empirically against the actual data as a cheap
        # sanity check; a mismatch would mean something's genuinely off
        # (wrong file, unexpected gridding version) worth knowing about
        # immediately, not silently trusting a constant that might not
        # apply to whatever file actually got loaded.
        lat_cell_size, lon_cell_size = GRID_CELL_SIZE_DEG, GRID_CELL_SIZE_DEG
        empirical_lat, empirical_lon = _derive_grid_cell_size(pings)
        if empirical_lat is not None and (abs(empirical_lat - lat_cell_size) > 0.01 or abs(empirical_lon - lon_cell_size) > 0.01):
            print(f"WARNING: empirically-derived grid cell size ({empirical_lat:.4f}° lat x {empirical_lon:.4f}° lon) "
                  f"does NOT match the documented GRID_CELL_SIZE_DEG={GRID_CELL_SIZE_DEG}° -- using the documented "
                  f"value, but this mismatch is worth investigating (wrong file? different gridding version?).")
        else:
            print(f"Grid cell size: {lat_cell_size}° x {lon_cell_size}° (this project's own documented 1°x1° "
                  f"gridding methodology" + (", confirmed against this data" if empirical_lat is not None else "") + ")")
    elif show_grid:
        print("NOTE: trajectories_gridded.parquet has no GRID_LAT_IDX/GRID_LON_IDX/GRID_LAT_C/GRID_LON_C "
              "columns -- grid delimitation skipped.")

    def _circle_points_km(center_lat, center_lon, radius_km, n_points=60):
        R = 6371.0
        lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
        d_r = radius_km / R
        bearings = np.linspace(0, 2 * np.pi, n_points)
        lat2 = np.arcsin(np.sin(lat1) * np.cos(d_r) + np.cos(lat1) * np.sin(d_r) * np.cos(bearings))
        lon2 = lon1 + np.arctan2(np.sin(bearings) * np.sin(d_r) * np.cos(lat1),
                                   np.cos(d_r) - np.sin(lat1) * np.sin(lat2))
        return np.degrees(lat2), np.degrees(lon2)

    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324"]
    port_to_subregion = build_port_to_subregion_map(step3data.traj_idx, subregion_col=target_col, port_col="ARR_PORT_ID")

    fig = go.Figure()
    legend_rows = []
    grid_cells_drawn = set()
    grid_lat_pts, grid_lon_pts = [], []

    for i, seg_id in enumerate(selected_seg_ids):
        seg_pings = pings[pings["SEG_ID"] == seg_id]
        if len(seg_pings) == 0:
            print(f"WARNING: seg_id={seg_id} has no raw ping rows in trajectories_gridded.parquet -- skipped")
            continue
        color = palette[i % len(palette)]
        seg_direction = directions_by_seg[seg_id]

        row = seg_meta.loc[seg_id]
        load_port_name, disch_port_name = row["dep_port"], row["arr_port"]
        load_subregion = subregion_name_map.get(port_to_subregion.get(row["DEP_PORT_ID"]), "unknown")
        disch_subregion = subregion_name_map.get(row[target_col], "unknown")
        dep_date = pd.Timestamp(row["dep_ts"]).strftime("%Y-%m-%d") if pd.notna(row["dep_ts"]) else "unknown"
        arr_date = pd.Timestamp(row["arr_ts"]).strftime("%Y-%m-%d") if pd.notna(row.get("arr_ts")) else "unknown"
        duration_h = row["duration_h"] if "duration_h" in seg_meta.columns else np.nan
        journey_length = f"{duration_h / 24.0:.1f} days" if pd.notna(duration_h) else "unknown"

        legend_name = f"#{seg_id} [{seg_direction}]: {load_port_name} → {disch_port_name} — dep. {dep_date}"
        legend_rows.append({"seg_id": seg_id, "direction": seg_direction, "load_port": load_port_name,
                             "load_subregion": load_subregion, "discharge_port": disch_port_name,
                             "discharge_subregion": disch_subregion, "n_pings": len(seg_pings),
                             "departure_date": dep_date, "arrival_date": arr_date, "journey_length": journey_length})

        hover_text = [f"seg_id={seg_id} [{seg_direction}]<br>ping {j+1}/{len(seg_pings)}<br>{ts}<br>lat={lat:.3f}, lon={lon:.3f}"
                      for j, (ts, lat, lon) in enumerate(zip(seg_pings["TIMESTAMP"], seg_pings["LAT"], seg_pings["LON"]))]
        fig.add_trace(go.Scattergeo(
            lon=seg_pings["LON"], lat=seg_pings["LAT"], mode="lines+markers",
            line=dict(width=2, color=color), marker=dict(size=5, color=color, line=dict(width=1, color="white")),
            opacity=0.9, name=legend_name, legendgroup=f"seg_{seg_id}", showlegend=True,
            hoverinfo="text", text=hover_text,
        ))

        mid_idx = len(seg_pings) // 2
        fig.add_trace(go.Scattergeo(
            lon=[seg_pings["LON"].iloc[mid_idx]], lat=[seg_pings["LAT"].iloc[mid_idx]],
            mode="markers+text", text=[str(seg_id)], textposition="top center",
            textfont=dict(size=13, color=color, family="Arial Black"),
            marker=dict(size=6, color=color, line=dict(width=1.5, color="white")),
            showlegend=False, legendgroup=f"seg_{seg_id}", hoverinfo="skip",
        ))

        if lat_cell_size is not None:
            # Grid PATH -- the sequence of distinct grid cells this
            # specific segment actually passed through, IN ORDER
            # (STEP_IDX), connecting cell centroids -- distinct from
            # the raw ping track above (every individual AIS
            # observation) and the cell BOUNDARY rectangles below
            # (every cell visited, no ordering/sequence implied). Same
            # per-segment color, dashed + open-circle markers to stay
            # visually distinct from the solid raw-ping line, sharing
            # that segment's own legendgroup so toggling one toggles
            # both rather than adding a redundant legend entry.
            grid_seq = seg_pings.drop_duplicates(subset=["STEP_IDX"])[["GRID_LON_C", "GRID_LAT_C", "STEP_IDX"]].sort_values("STEP_IDX")
            if len(grid_seq) > 0:
                fig.add_trace(go.Scattergeo(
                    lon=grid_seq["GRID_LON_C"], lat=grid_seq["GRID_LAT_C"], mode="lines+markers",
                    line=dict(width=2, color=color, dash="dash"),
                    marker=dict(size=8, color=color, symbol="circle-open", line=dict(width=2, color=color)),
                    opacity=0.75, legendgroup=f"seg_{seg_id}", showlegend=False,
                    hoverinfo="text", text=[f"seg_id={seg_id} — grid step {int(s)}" for s in grid_seq["STEP_IDX"]],
                ))

            cell_keys = seg_pings[["GRID_LAT_IDX", "GRID_LON_IDX", "GRID_LAT_C", "GRID_LON_C"]].drop_duplicates(
                subset=["GRID_LAT_IDX", "GRID_LON_IDX"])
            h, w = lat_cell_size / 2, lon_cell_size / 2
            for _, cell in cell_keys.iterrows():
                key = (cell["GRID_LAT_IDX"], cell["GRID_LON_IDX"])
                if key in grid_cells_drawn:
                    continue
                grid_cells_drawn.add(key)
                clat, clon = cell["GRID_LAT_C"], cell["GRID_LON_C"]
                grid_lat_pts.extend([clat - h, clat - h, clat + h, clat + h, clat - h, None])
                grid_lon_pts.extend([clon - w, clon + w, clon + w, clon - w, clon - w, None])

    if len(fig.data) == 0:
        raise ValueError(f"No plottable ping data found for any of the selected segments {selected_seg_ids}")

    if lat_cell_size is not None and grid_lat_pts:
        fig.add_trace(go.Scattergeo(
            lon=grid_lon_pts, lat=grid_lat_pts, mode="lines",
            line=dict(width=1, color="#999999"), opacity=0.5,
            name=f"Grid cells ({lat_cell_size:.3f}° x {lon_cell_size:.3f}°, {len(grid_cells_drawn)} shown)",
            showlegend=True, hoverinfo="skip",
        ))

    fig.add_trace(go.Scattergeo(
        lon=[port_lon], lat=[port_lat], mode="markers+text",
        text=[port_name], textposition="bottom center", textfont=dict(size=12, color="#333333"),
        marker=dict(size=12, color="#333333", symbol="square", line=dict(width=1.5, color="white")),
        name=f"Port: {port_name}", showlegend=True, hoverinfo="text",
    ))
    if port_radius_nm > 0:
        radius_km = port_radius_nm * 1.852  # exact NM -> km conversion
        circle_lat, circle_lon = _circle_points_km(port_lat, port_lon, radius_km)
        fig.add_trace(go.Scattergeo(
            lon=circle_lon, lat=circle_lat, mode="lines",
            line=dict(width=1.5, color="#333333", dash="dot"), opacity=0.7,
            name=f"{port_radius_nm:.0f} NM radius", showlegend=True, hoverinfo="skip",
        ))

    n_inbound = sum(1 for d in directions_by_seg.values() if d == "inbound")
    n_outbound = sum(1 for d in directions_by_seg.values() if d == "outbound")
    auto_title = f"{port_name}: {n_inbound} inbound, {n_outbound} outbound segment(s)"
    fig.update_layout(
        title=dict(text=title or auto_title, x=0.5, xanchor="center", font=dict(size=18), y=0.98, yanchor="top"),
        geo=dict(showland=True, landcolor="rgb(235,235,235)", showocean=True, oceancolor="rgb(225,242,255)",
                  showcountries=True, countrycolor="rgb(210,210,210)", projection_type="mercator",
                  domain=dict(x=[0, 1], y=[0, 1]), fitbounds="locations"),
        height=650,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom", font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
    )
    if save_path:
        fig.write_html(save_path)
        print(f"Saved -> {save_path}")
    fig.show()

    legend_table = pd.DataFrame(legend_rows)
    print(f"\n{len(legend_rows)} segment(s) plotted for {port_name} ({n_inbound} inbound, {n_outbound} outbound):")
    print(legend_table.to_string(index=False))
    return fig, legend_table

# ----------------------------------------------------------------------
# [notebook cell 111]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-SETUP -- shared session state for every Viz cell
# =============================================================================
port_name_map = get_port_name_map(data)
port_to_sub = build_port_to_subregion_map(data.traj_idx,
                                          subregion_col="ARR_SUBREGION_ID",
                                          port_col="ARR_PORT_ID")
train_ids, val_ids, test_ids = _make_split(
    data, TARGET_COL, val_frac=0.15, seed=42, stratify=True,
    test_start=TEST_START, test_end=TEST_END)

def make_extra_kwargs_fn(r):
    def _fn(inputs, seg_ids):
        core, alts = r["core_and_alt_fn"](inputs, seg_ids)
        kw = {"external_progression_frac": core, "alt_progression_fracs": alts}
        if r.get("departure_ids_fn") is not None:
            kw["departure_subregion_ids"] = r["departure_ids_fn"](seg_ids)
        return kw
    return _fn

VIZ_SEED = 123
viz_r = runs[VIZ_SEED]
extra_kwargs_fn = make_extra_kwargs_fn(viz_r)

name_to_id = {v: k for k, v in subregion_names.items()}
print(f"viz setup ready: {len(port_name_map)} ports, seed {VIZ_SEED} model")

# ----------------------------------------------------------------------
# [notebook cell 113]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-1A -- routes map: arbitrary load->destination pairs, one sized map
# =============================================================================
import plotly.graph_objects as go
from IPython.display import HTML, display

PAIRS = [("USGC", "NEAsia_China"), ("USGC", "India SC"),
         ("USGC", "MED"), ("USGC", "NWE")]     # any (load, destination) pairs
HIGHLIGHT_SEG_IDS = []                          # explicit seg ids, [] for none
DATASET = ["train", "val", "test"]              # any combination
MAP_TITLE = "USGC departures -- four destination basins"

go.Figure.show, _oshow = (lambda *a, **k: None), go.Figure.show
try:
    ret = _plot_routes_between_subregions_single(
        data, subregion_names,
        load_subregion_name=[p[0] for p in PAIRS],
        arrival_subregion_name=[p[1] for p in PAIRS],
        dataset=DATASET, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        highlight_seg_ids=HIGHLIGHT_SEG_IDS or None,
        model=viz_r["model"], repr_layer=viz_r["repr_layer"],
        extra_kwargs_fn=extra_kwargs_fn,
        title=MAP_TITLE, save_path="routes_onemap.html")
finally:
    go.Figure.show = _oshow

fig = ret[0] if isinstance(ret, tuple) else ret
fig.update_layout(height=560, width=1000, margin=dict(t=30, l=0, r=0, b=0),
    title=dict(text=MAP_TITLE, y=0.99, x=0.5, xanchor="center", font=dict(size=18)),
    legend=dict(orientation="h", yanchor="top", y=0.945,
                xanchor="center", x=0.5, font=dict(size=12)))
fig.update_geos(domain=dict(x=[0, 1], y=[0, 0.88]), projection_scale=1.05)
fig.write_html(os.path.join(WORK_DIR, "routes_onemap.html"))
display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))

# ----------------------------------------------------------------------
# [notebook cell 114]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-1A -- TEST - SEGMENT ONLY
# =============================================================================
import plotly.graph_objects as go
from IPython.display import HTML, display

PAIRS = [("USGC", "NEAsia_China"), ("USGC", "India SC"),
         ("USGC", "MED"), ("USGC", "NWE")]     # any (load, destination) pairs
HIGHLIGHT_SEG_IDS = []                          # explicit seg ids, [] for none
DATASET = ["train", "val", "test"]              # any combination
MAP_TITLE = "USGC departures -- four destination basins"

go.Figure.show, _oshow = (lambda *a, **k: None), go.Figure.show
try:
    ret = _plot_routes_between_subregions_single(
        data, subregion_names,
        load_subregion_name=[p[0] for p in PAIRS],
        arrival_subregion_name=[p[1] for p in PAIRS],
        dataset=DATASET, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        highlight_seg_ids=HIGHLIGHT_SEG_IDS or None,
        model=viz_r["model"], repr_layer=viz_r["repr_layer"],
        extra_kwargs_fn=extra_kwargs_fn,
        title=MAP_TITLE, save_path="routes_onemap.html")
finally:
    go.Figure.show = _oshow

fig = ret[0] if isinstance(ret, tuple) else ret
fig.update_layout(height=560, width=1000, margin=dict(t=30, l=0, r=0, b=0),
    title=dict(text=MAP_TITLE, y=0.99, x=0.5, xanchor="center", font=dict(size=18)),
    legend=dict(orientation="h", yanchor="top", y=0.945,
                xanchor="center", x=0.5, font=dict(size=12)))
fig.update_geos(domain=dict(x=[0, 1], y=[0, 0.88]), projection_scale=1.05)
fig.write_html(os.path.join(WORK_DIR, "routes_onemap.html"))
display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))

# ----------------------------------------------------------------------
# [notebook cell 115]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-1B -- same map, one per time period, stacked vertically
# =============================================================================
PAIRS_B = PAIRS
PERIODS = [("2023", "2023-01-01", "2024-01-01"),
           ("2024", "2024-01-01", "2025-01-01"),
           ("2025 pre-conflict", "2025-01-01", "2025-12-01"),
           ("Conflict window", "2025-12-01", "2026-03-01")]
WINDOW_MODE = "overlap"      # voyages touching the window count (labels are
                             # cosmetic; the two DATES do the filtering)
for label, w_start, w_end in PERIODS:
    ptitle = f"{MAP_TITLE} -- {label} ({w_start} -> {w_end})"
    go.Figure.show, _oshow = (lambda *a, **k: None), go.Figure.show
    try:
        ret = _plot_routes_between_subregions_single(
            data, subregion_names,
            load_subregion_name=[p[0] for p in PAIRS_B],
            arrival_subregion_name=[p[1] for p in PAIRS_B],
            dataset=DATASET, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
            window_start=w_start, window_end=w_end, window_mode=WINDOW_MODE,
            highlight_seg_ids=HIGHLIGHT_SEG_IDS or None,
            model=viz_r["model"], repr_layer=viz_r["repr_layer"],
            extra_kwargs_fn=extra_kwargs_fn,
            title=ptitle, save_path=f"routes_{label.replace(' ', '_')}.html")
    finally:
        go.Figure.show = _oshow
    fig = ret[0] if isinstance(ret, tuple) else ret
    fig.update_layout(height=560, width=1000, margin=dict(t=30, l=0, r=0, b=0),
        title=dict(text=ptitle, y=0.99, x=0.5, xanchor="center", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="top", y=0.945,
                    xanchor="center", x=0.5, font=dict(size=12)))
    fig.update_geos(domain=dict(x=[0, 1], y=[0, 0.88]), projection_scale=1.05)
    fig.write_html(os.path.join(WORK_DIR, f"routes_{label.replace(' ', '_')}.html"))
    display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))

# ----------------------------------------------------------------------
# [notebook cell 117]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-2 -- a vessel's consecutive voyages
# =============================================================================
VIZ2_IMO = 9354935          # if it errors: pick from
                            # data.traj_idx["IMO"].value_counts().head(3)
fig2, legend_table = plot_consecutive_segments_for_vessel(
    data, subregion_names, port_name_map, WORK_DIR,
    imo=VIZ2_IMO, n_segments=3,
    title=f"IMO {VIZ2_IMO}: three most recent voyages")
print(legend_table.to_string(index=False))

# ----------------------------------------------------------------------
# [notebook cell 119]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-3 FINAL v8 -- port call: status-coloured paths, berth clusters visible
# =============================================================================
import plotly.graph_objects as go
import pyarrow.parquet as pq
from IPython.display import HTML, display

PORT = "Ruwais"
PAIR_ROW = 0
DIRECTION = "inbound"          # "inbound" | "outbound" | "both"

pairs = find_consecutive_port_call_pairs(data, PORT)
_row = pairs.iloc[PAIR_ROW]
_segs = {"inbound": [int(_row["inbound_seg_id"])],
         "outbound": [int(_row["outbound_seg_id"])],
         "both": [int(_row["inbound_seg_id"]), int(_row["outbound_seg_id"])]}[DIRECTION]

go.Figure.show, _oshow = (lambda *a, **k: None), go.Figure.show
try:
    fig3, legend3 = plot_port_traffic(
        data, subregion_names, port_name_map, WORK_DIR, port_name=PORT,
        seg_ids=_segs, show_grid=True,
        title=f"{PORT} port call ({DIRECTION}) -- IMO {_row['IMO']}")
finally:
    go.Figure.show = _oshow
fig3.data = tuple(t for t in fig3.data
                  if not (t.name and "radius" in str(t.name).lower()))

# ---- ports: black dotted circles; label BELOW the circle, 14pt black -------
_ref = pd.read_csv(os.path.join(WORK_DIR, "lpg_port_reference_fixed.csv"))
_ref["_key"] = _ref["canonical_name"].str.strip().str.lower()
_tj = data.traj_idx.set_index("seg_id")
_ports = sorted({str(_tj.loc[s, c]) for s in _segs for c in ("dep_port", "arr_port")})
theta = np.linspace(0, 2*np.pi, 121)
for j, pname in enumerate(_ports):
    row = _ref[_ref["_key"] == pname.strip().lower()]
    if len(row):
        r_nm = float(row["radius_nm"].iloc[0])
        p_lat, p_lon = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
    else:
        r_nm = 12.0
        _pid = next((k for k, v in port_name_map.items() if v == pname), None)
        _arr = data.traj_idx[data.traj_idx["ARR_PORT_ID"] == _pid]["seg_id"]
        _last = data.steps_idx.loc[data.steps_idx.groupby("SEG_ID")["STEP_IDX"].idxmax()]
        _pl = _last[_last["SEG_ID"].isin(_arr)][["GRID_LAT_C", "GRID_LON_C"]].median()
        p_lat, p_lon = float(_pl["GRID_LAT_C"]), float(_pl["GRID_LON_C"])
    r_km = r_nm * 1.852
    r_lat, r_lon = r_km/111.0, r_km/(111.0*np.cos(np.radians(p_lat)))
    fig3.add_trace(go.Scattergeo(
        lat=p_lat + r_lat*np.sin(theta), lon=p_lon + r_lon*np.cos(theta),
        mode="lines", line=dict(width=2.5, color="black", dash="dot"),
        name="Port radius (reference, NM)", showlegend=(j == 0)))
    fig3.add_trace(go.Scattergeo(
        lat=[p_lat], lon=[p_lon], mode="markers",
        marker=dict(size=10, color="black", symbol="square"),
        showlegend=False, hoverinfo="skip"))
    fig3.add_trace(go.Scattergeo(               # label BELOW the circle
        lat=[p_lat - 1.6*r_lat], lon=[p_lon], mode="text",
        text=[f"{pname}  ({r_nm:.0f} NM)"], textposition="bottom center",
        textfont=dict(size=14, color="black"),
        showlegend=False, hoverinfo="skip"))

# ---- pings + STATUS-COLOURED PATH RUNS -------------------------------------
_gp = os.path.join(WORK_DIR, Step3b_representation_layer.DATA_SUBFOLDER,
                   "trajectories_gridded.parquet")
_have = set(pq.ParquetFile(_gp).schema.names)
_latc = next(c for c in ("LAT", "LATITUDE", "lat") if c in _have)
_lonc = next(c for c in ("LON", "LONGITUDE", "lon") if c in _have)
_stc = next((c for c in ("PORT_STATUS", "STATUS", "NAV_STATUS") if c in _have), None)
_p = pd.read_parquet(_gp, columns=["SEG_ID", "TIMESTAMP", _latc, _lonc]
                                  + ([_stc] if _stc else []))
_p = _p[_p["SEG_ID"].isin(_segs)].sort_values(["SEG_ID", "TIMESTAMP"])

STATUS_COLORS = {"docked": "#aa00ff", "in_port": "#ff6d00",
                 "at_sea": "#2171b5", "waypoint": "#6a51a3"}
STATUS_WIDTHS = {"docked": 6, "in_port": 5, "at_sea": 3, "waypoint": 3}
_seen_status = set()
if _stc:
    for sid, gseg in _p.groupby("SEG_ID"):
        st = gseg[_stc].astype(str).values
        runs = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
        for a, b in zip(runs[:-1], runs[1:]):
            b2 = min(b + 1, len(gseg))          # 1-pt overlap: continuous path
            seg_run = gseg.iloc[a:b2]
            s_name = st[a]
            _is_port_side = s_name in ("docked", "in_port")
            fig3.add_trace(go.Scattergeo(
                lat=seg_run[_latc], lon=seg_run[_lonc],
                mode="lines+markers" if _is_port_side else "lines",
                line=dict(width=STATUS_WIDTHS.get(s_name, 3),
                          color=STATUS_COLORS.get(s_name, "#666666")),
                marker=dict(size=10 if s_name == "docked" else 8,
                            color=STATUS_COLORS.get(s_name, "#666666"),
                            line=dict(width=1, color="white")),
                name=s_name, showlegend=(s_name not in _seen_status),
                hoverinfo="skip"))
            _seen_status.add(s_name)
fig3.add_trace(go.Scattergeo(                   # pings: neutral small stars
    lat=_p[_latc], lon=_p[_lonc], mode="markers",
    marker=dict(size=5, symbol="star", color="rgba(60,60,60,0.55)",
                line=dict(width=0.5, color="white")),
    name=f"AIS pings ({len(_p)})",
    text=[f"#{s}  {t}" + (f"  [{st}]" if _stc else "") for s, t, st in
          zip(_p["SEG_ID"], _p["TIMESTAMP"].astype(str),
              _p[_stc].astype(str) if _stc else [""]*len(_p))],
    hoverinfo="text"))

for nm, dash in [("Raw AIS track (status-coloured, solid)", None),
                 ("Modelled grid-based track (dashed)", "dash")]:
    fig3.add_trace(go.Scattergeo(lat=[None], lon=[None], mode="lines",
        line=dict(width=2, color="grey", dash=dash), name=nm))

# ---- legend: Paths -> Port/Radius/Grid -> Status (path) -> Trajectory ------
def _legend_slot(t):
    n = str(t.name or "")
    if n.startswith("Raw AIS") or n.startswith("Modelled"):
        return "paths", "Paths: Raw vs. Modelled", 100, n, None, None
    if n.startswith("Port:"):
        return "portgrp", "Port, Radius, Grid Cells", 200, "Port", "markers", None
    if n.startswith("Port radius") or n.startswith("Grid cells"):
        return "portgrp", "Port, Radius, Grid Cells", 200, n, None, None
    if n in STATUS_COLORS or n.startswith("AIS pings"):
        return "status", "Status (path colour)", 250, n, None, None
    if n.startswith("#"):     # function's per-segment solid raw track
        return "traj", "Trajectory Detail", 300, n, None, "underlay"
    return "traj", "Trajectory Detail", 300, n, None, None

for t in fig3.data:
    grp, title, rank, name, force_mode, style = _legend_slot(t)
    t.legendgroup = grp; t.legendgrouptitle = dict(text=title)
    t.legendrank = rank; t.name = name
    if force_mode:
        t.mode = force_mode
        if hasattr(t, "text"): t.text = None
    if style == "underlay" and t.line is not None and not t.line.dash:
        t.line.color = "#bbbbbb"; t.line.width = 1   # grey under status runs

fig3.update_layout(
    legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01,
                font=dict(size=11), groupclick="toggleitem",
                grouptitlefont=dict(size=12, color="#333")),
    title=dict(y=0.99, x=0.5, xanchor="center", font=dict(size=18)))
fig3.write_html(os.path.join(WORK_DIR, f"port_call_{PORT}_{DIRECTION}.html"))
display(HTML(fig3.to_html(include_plotlyjs=True, full_html=False)))

# ----------------------------------------------------------------------
# [notebook cell 121]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-2C -- high / medium / low forecast-accuracy exemplars, green/red maps
# =============================================================================
# Prereq: the working VIZ-4 cell (pooled arrays seg/true/pred + the proven
# plot_error_segment) has run in this session.
for _req in ("seg", "true", "pred", "plot_error_segment"):
    assert _req in dir(), f"'{_req}' missing -- run the VIZ-4 (E12-format) cell first"

MIN_STEPS = 10          # exclude short voyages: "average accuracy" needs steps

# ---- per-segment mean forecast accuracy over the pooled predictions --------
_df = pd.DataFrame({"seg": seg, "ok": (pred == true)})
_acc = _df.groupby("seg")["ok"].agg(["mean", "size"]).rename(
        columns={"mean": "acc", "size": "n_rows"})
_len = data.steps_idx.groupby("SEG_ID").size()
_acc["n_steps"] = _acc.index.map(_len)
_eligible = _acc[_acc["n_steps"] >= MIN_STEPS].sort_values("acc")

_lo  = int(_eligible.index[0])
_med = int(_eligible.index[(len(_eligible) - 1) // 2])
_hi  = int(_eligible.index[-1])
CANDIDATES = [("HIGH", _hi), ("MEDIUM", _med), ("LOW", _lo)]
print(f"{len(_eligible):,} eligible segments (>= {MIN_STEPS} steps)")
for lbl, sid in CANDIDATES:
    r = _eligible.loc[sid]
    print(f"  {lbl:6s}: seg {sid}  acc {100*r['acc']:5.1f}%  ({int(r['n_steps'])} steps)")

# ---- one green/red map per candidate (proven E12 format) -------------------
for lbl, sid in CANDIDATES:
    print("=" * 70); print(f"{lbl}-accuracy exemplar -- seg {sid}")
    plot_error_segment(sid)

# ----------------------------------------------------------------------
# [notebook cell 122]
# ----------------------------------------------------------------------
# =============================================================================
# VIZ-2B -- plot a GIVEN set of segments (any vessels), reusing VIZ-2's plotter
# =============================================================================
SEG_IDS = [15341, 15342]        # <- your segments, any mix of vessels

_tj = data.traj_idx.set_index("seg_id")
_missing = [s for s in SEG_IDS if s not in _tj.index]
assert not _missing, f"not in current data: {_missing}"

# group by vessel, preserving your order; consecutive runs plot on ONE map
from collections import OrderedDict
_by_imo = OrderedDict()
for s in SEG_IDS:
    _by_imo.setdefault(_tj.loc[s, "IMO"], []).append(int(s))

VIZ2B = {}
for imo, segs in _by_imo.items():
    ordered = sorted(segs, key=lambda s: pd.Timestamp(_tj.loc[s, "dep_ts"]))
    first, n = ordered[0], len(ordered)
    # contiguity check: the function plots N CONSECUTIVE voyages from start_seg_id;
    # non-adjacent requests fall back to one map per segment
    vessel_order = (_tj[_tj["IMO"] == imo].sort_values("dep_ts").index.astype(int).tolist())
    i0 = vessel_order.index(first)
    contiguous = vessel_order[i0:i0 + n] == ordered
    calls = [(first, n)] if contiguous else [(s, 1) for s in ordered]
    if not contiguous:
        print(f"IMO {imo}: segments {ordered} are not consecutive voyages -- "
              f"plotting {len(calls)} separate map(s)")
    for start, count in calls:
        fig, tbl = plot_consecutive_segments_for_vessel(
            data, subregion_names, port_name_map, WORK_DIR,
            imo=imo, n_segments=count, start_seg_id=start,
            title=f"IMO {imo} -- segment(s) "
                  f"{ordered if contiguous else [start]}")
        VIZ2B[(imo, start)] = tbl
        print(tbl.to_string(index=False))
