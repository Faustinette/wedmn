# Section 4.6 — Departure duration index
# Executed by runner.py inside the shared namespace (notebook-kernel style).


class DepartureDurationIndex:
    """Fleet-wide expected-duration lookup, keyed by departure port, with a
    subregion-level fallback for ports with too little history of their
    own. Built once from historical traj_idx; expected() is then an O(1)
    lookup."""

    def __init__(self, traj_idx: pd.DataFrame, min_count_port=5, min_count_subregion=5):
        required = {"DEP_PORT_ID", "duration_h", "ARR_PORT_ID", "ARR_SUBREGION_ID"}
        missing = required - set(traj_idx.columns)
        if missing:
            raise ValueError(f"traj_idx is missing columns needed for duration estimation: {missing}")

        valid = traj_idx.dropna(subset=["DEP_PORT_ID", "duration_h"]).copy()
        valid["DEP_PORT_ID"] = valid["DEP_PORT_ID"].astype(int)

        # Derive a DEP_PORT_ID -> subregion lookup from the arrival side
        # of the data: every port that appears as an arrival already has
        # ARR_SUBREGION_ID attached, so the mapping is read off arrivals
        # and applied to departure ports (ports are shared between the
        # two roles). A port that only ever appears as a departure and
        # never as an arrival has no derivable subregion; it falls
        # through to the global fallback below and is logged as such
        # rather than silently assigned a guess.
        
        port_to_subregion = (
            traj_idx.dropna(subset=["ARR_PORT_ID", "ARR_SUBREGION_ID"])
            .groupby("ARR_PORT_ID")["ARR_SUBREGION_ID"].first().astype(int).to_dict()
        )

        # Level 1: exact departure port
        port_counts = valid.groupby("DEP_PORT_ID")["duration_h"].count()
        port_median = valid.groupby("DEP_PORT_ID")["duration_h"].median()
        self._port_median = {p: port_median[p] for p in port_counts.index if port_counts[p] >= min_count_port}

        # Level 2: departure port's subregion (fleet-wide, using the derived mapping)
        valid["_dep_subregion"] = valid["DEP_PORT_ID"].map(port_to_subregion)
        sub_valid = valid.dropna(subset=["_dep_subregion"])
        sub_counts = sub_valid.groupby("_dep_subregion")["duration_h"].count()
        sub_median = sub_valid.groupby("_dep_subregion")["duration_h"].median()
        self._subregion_median = {s: sub_median[s] for s in sub_counts.index if sub_counts[s] >= min_count_subregion}

        self._port_to_subregion = port_to_subregion
        self._global_median = float(valid["duration_h"].median()) if len(valid) else None
        self.min_count_port = min_count_port
        self.min_count_subregion = min_count_subregion

        # Usage-level accounting, for transparency (matches the "N classes
        # kept entirely in train" printed diagnostic already used for the
        # stratified split) — populated lazily as expected() is actually called.
        self._usage_counts = {"port": 0, "subregion": 0, "global": 0}

    def expected(self, dep_port_id):
        """Returns (expected_duration_h, level_used) — level_used is
        "port", "subregion", or "global", so callers can inspect exactly
        which fallback fired for a given lookup, not just get a number
        back with no way to audit it."""
        if dep_port_id in self._port_median:
            self._usage_counts["port"] += 1
            return self._port_median[dep_port_id], "port"
        subregion = self._port_to_subregion.get(dep_port_id)
        if subregion is not None and subregion in self._subregion_median:
            self._usage_counts["subregion"] += 1
            return self._subregion_median[subregion], "subregion"
        self._usage_counts["global"] += 1
        return self._global_median, "global"

    def usage_summary(self):
        """Prints how often each fallback level actually fired — call
        after running expected() over a batch of departure ports, to see
        whether the exact-port level is doing most of the work or whether
        most vessels are falling through to a coarser estimate."""
        total = sum(self._usage_counts.values())
        if total == 0:
            print("No lookups recorded yet.")
            return
        for level, count in self._usage_counts.items():
            print(f"  {level:<10} {count:>6} ({count/total*100:.1f}%)")



class FineDurationIndex:
    """Like DepartureDurationIndex, but bucketed by (departure port, size
    class, departure month) instead of departure port alone — a sharper,
    more specific estimate, at the same data-sparsity risk seen throughout
    this project's narrower regime cutoffs (finer buckets, less data per
    bucket). Falls back to the coarser DepartureDurationIndex (port ->
    subregion -> global) when a specific fine bucket has too little
    history of its own, rather than an unconditional global fallback —
    keeps the estimate as sharp as the DATA actually supports, never
    sharper than that.
    """

    def __init__(self, traj_idx: pd.DataFrame, coarse_index: DepartureDurationIndex, min_count_fine=5):
        required = {"DEP_PORT_ID", "duration_h", "SIZE_CLASS_ID", "dep_ts"}
        missing = required - set(traj_idx.columns)
        if missing:
            raise ValueError(f"traj_idx is missing columns needed for fine duration estimation: {missing}")

        self.coarse_index = coarse_index
        valid = traj_idx.dropna(subset=["DEP_PORT_ID", "duration_h", "SIZE_CLASS_ID", "dep_ts"]).copy()
        valid["DEP_PORT_ID"] = valid["DEP_PORT_ID"].astype(int)
        valid["SIZE_CLASS_ID"] = valid["SIZE_CLASS_ID"].astype(int)
        valid["month"] = pd.to_datetime(valid["dep_ts"]).dt.month

        grouped = valid.groupby(["DEP_PORT_ID", "SIZE_CLASS_ID", "month"])
        group_sizes = grouped.size()
        fine_median = grouped["duration_h"].median()
        self._fine_median = {k: v for k, v in fine_median.items() if group_sizes[k] >= min_count_fine}
        self._usage_counts = {"fine": 0, "fallback": 0}

    def expected(self, dep_port_id, size_class_id, month):
        """Returns (expected_duration_h, level_used) — "fine" if the exact
        (port, size, month) bucket had enough history, else falls through
        to the coarser DepartureDurationIndex entirely (whose own level —
        "port"/"subregion"/"global" — gets reported prefixed with
        "fallback_" so callers can still audit which coarse level fired)."""
        key = (int(dep_port_id), int(size_class_id), int(month))
        if key in self._fine_median:
            self._usage_counts["fine"] += 1
            return self._fine_median[key], "fine"
        self._usage_counts["fallback"] += 1
        coarse_val, coarse_level = self.coarse_index.expected(dep_port_id)
        return coarse_val, f"fallback_{coarse_level}"

    def usage_summary(self):
        total = sum(self._usage_counts.values())
        if total == 0:
            print("No lookups recorded yet.")
            return
        for level, count in self._usage_counts.items():
            print(f"  {level:<10} {count:>6} ({count/total*100:.1f}%)")



# [5] CANDIDATE-CONDITIONED FUTURE FLEET STATE — Stages 1-6
#
# A genuinely different mechanism from everything above: instead of one
# fleet-heading vector per step (built from a fixed window around each
# OTHER vessel's own voyage), this builds ONE VECTOR PER CANDIDATE
# DESTINATION for the query vessel — "if I end up going to subregion C,
# here's the fleet state I'd expect to see when I actually arrive there,
# compared to what's typical."
#
# STAGE 1: position -> subregion assignment, for ANY point (not just
# arrivals). Port locations are DERIVED from the data itself (median
# position of vessels' last recorded step before arrival), then any
# arbitrary (lat, lon) is assigned to its NEAREST known port's subregion.
