# Section 3.b.iv — Subregion name mapping
# Executed by runner.py inside the shared namespace (notebook-kernel style).

def get_subregion_name_map(step3data, subregion_col="ARR_SUBREGION_ID"):
    """Reverse of step3data.vocab's own name->id mapping (e.g.
    port_subregion_to_id), for turning subregion IDs back into readable
    names in any table meant to be actually read, not just IDs. Vocab key
    is derived automatically from subregion_col, matching the same
    convention used elsewhere (e.g. build_candidate_fleet_state_index)."""
    vocab_key_by_col = {
        "ARR_SUBREGION_ID": "port_subregion_to_id",
        "ARR_COUNTRY_ID": "port_country_to_id",
        "ARR_REGION_ID": "port_region_to_id",
    }
    vocab_key = vocab_key_by_col.get(subregion_col, "port_subregion_to_id")
    return {v: k for k, v in step3data.vocab[vocab_key].items()}
