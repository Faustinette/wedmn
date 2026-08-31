# =============================================================================
# Section 3.b.iii — Arrival labels + ship-history index (self-contained)
# Migrated verbatim from Main_forGitHub.ipynb cells [31].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 31]
# ----------------------------------------------------------------------
# CELL 3d-INLINE -- ARRIVAL LABELS + SHIP-HISTORY INDEX, SELF-CONTAINED
# Replaces:
#     from Step4c_train import enrich_arrival_labels, get_subregion_name_map
#     enrich_arrival_labels(data); data.build_ship_history_index()
# Contents, all verbatim from the live project files:
#   1. enrich_arrival_labels      (your uploaded Step4c_train.py)
#   2. get_subregion_name_map     (your uploaded Step4c_train.py)
#   3. VesselHistoryIndex + its 3 constants (Step4d_ship_history.py) --
#      so data.build_ship_history_index()'s lazy `from Step4d_...` import
#      is bypassed: we assign data.history_index directly.
# =============================================================================
import numpy as np
import pandas as pd

def enrich_arrival_labels(step3data):
    """Adds ARR_PORT_ID, ARR_REGION_ID, ARR_SUBREGION_ID, ARR_COUNTRY_ID columns to
    step3data.traj_idx in place. Prints coverage diagnostics.

    If Step3a already built these (region/subregion/country vocabularies +
    ARR_* columns merged directly from arr_port, saved to disk) — the
    preferred path, since it persists across sessions instead of being
    recomputed in memory every time — this function detects that and skips
    straight to the coverage diagnostic. Falls back to deriving everything
    itself (the original in-memory approach) only if Step3a's output
    predates that addition, so older trajectories_index_enriched.csv files
    still work.
    """
    ti = step3data.traj_idx
    if "arr_port" not in ti.columns:
        raise ValueError("trajectories_index_enriched.csv has no 'arr_port' column — "
                          "check it survived Step2b/Step2c/Step3a's exports.")

    already_done = ("ARR_COUNTRY_ID" in ti.columns and "port_country_to_id" in step3data.vocab)
    if already_done:
        print("    ARR_* columns + country vocabulary already present from Step3a — "
              "skipping recomputation.")
    else:
        print("    ARR_* columns not found from Step3a (older pipeline output, or Step3a "
              "hasn't been re-run since country support was added) — deriving in memory "
              "instead. Re-run Step3a to persist this to disk and skip this every session.")
        port_to_id = step3data.vocab["port_to_id"]
        region_to_id = step3data.vocab.get("port_region_to_id", {})
        subregion_to_id = step3data.vocab.get("port_subregion_to_id", {})

        ti["ARR_PORT_ID"] = ti["arr_port"].map(port_to_id)

        region_lookup = _build_port_attribute_lookup(ti, "region")
        subregion_lookup = _build_port_attribute_lookup(ti, "subregion")
        ti["ARR_REGION_ID"] = ti["arr_port"].map(region_lookup).map(region_to_id)
        ti["ARR_SUBREGION_ID"] = ti["arr_port"].map(subregion_lookup).map(subregion_to_id)

        country_lookup = _build_port_attribute_lookup(ti, "country")
        if "port_country_to_id" not in step3data.vocab:
            country_names = sorted(set(country_lookup.values()))
            step3data.vocab["port_country_to_id"] = {c: i for i, c in enumerate(country_names)}
        country_to_id = step3data.vocab["port_country_to_id"]
        ti["ARR_COUNTRY_ID"] = ti["arr_port"].map(country_lookup).map(country_to_id)

    n = len(ti)
    for col in ["ARR_PORT_ID", "ARR_REGION_ID", "ARR_SUBREGION_ID", "ARR_COUNTRY_ID"]:
        n_missing = ti[col].isna().sum()
        print(f"    {col}: {n - n_missing:,} / {n:,} resolved "
              f"({n_missing:,} missing, {n_missing/n*100:.1f}%)")
    print(f"    Country vocabulary size: {len(step3data.vocab['port_country_to_id'])}")

    return ti
