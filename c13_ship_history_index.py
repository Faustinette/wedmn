# Section 3.b.v — Ship history index (MAX_HISTORY graphs)
# Executed by runner.py inside the shared namespace (notebook-kernel style).

MAX_HISTORY = 20          # cap on nodes (past voyages) per graph
DURATION_NORM = 500.0     # rough normalizer for duration_h (hours)
RECENCY_NORM_DAYS = 365.0 # rough normalizer for "days since that past voyage"

class VesselHistoryIndex:
    """Precomputes, once, each vessel's own segments sorted chronologically —
    so building any single segment's history graph is an O(1) list-slice
    (by POSITION, not a timestamp re-comparison) rather than re-scanning the
    full traj_idx table per segment per batch."""

    def __init__(self, traj_idx: pd.DataFrame):
        required = {"seg_id", "IMO", "dep_ts", "DEP_PORT_ID", "ARR_PORT_ID"}
        missing = required - set(traj_idx.columns)
        if missing:
            raise ValueError(f"traj_idx is missing columns needed for ship history: {missing}")

        df = traj_idx.copy()
        df["dep_ts"] = pd.to_datetime(df["dep_ts"])
        df["duration_h"] = pd.to_numeric(df.get("duration_h", np.nan), errors="coerce").fillna(0.0)
        # Sort by (IMO, dep_ts, seg_id) — seg_id as a stable tiebreaker for any
        # exact-duplicate timestamps, so position-based indexing is unambiguous.
        df = df.sort_values(["IMO", "dep_ts", "seg_id"]).reset_index(drop=True)

        self._by_imo = {}
        self._seg_position = {}  # seg_id -> (IMO, position within that vessel's list)
        for imo, grp in df.groupby("IMO", sort=False):
            grp = grp.reset_index(drop=True)
            self._by_imo[imo] = grp[["seg_id", "dep_ts", "DEP_PORT_ID", "ARR_PORT_ID", "duration_h"]]
            for pos, seg_id in enumerate(grp["seg_id"].values):
                self._seg_position[seg_id] = (imo, pos)

        self.seg_to_imo = dict(zip(df["seg_id"], df["IMO"]))

    def history_for(self, seg_id, max_history=MAX_HISTORY):
        """Returns a DataFrame of the up-to-max_history most recent PRIOR
        voyages for this segment's vessel — strictly before this segment's
        own position in that vessel's chronological list. Empty DataFrame
        for a cold-start (first-ever) segment or an unknown seg_id."""
        if seg_id not in self._seg_position:
            return self._empty_history()
        imo, pos = self._seg_position[seg_id]
        vessel_segs = self._by_imo[imo]
        prior = vessel_segs.iloc[:pos]  # strictly before position `pos` — never includes seg_id itself
        if len(prior) > max_history:
            prior = prior.iloc[-max_history:]  # keep the MOST RECENT max_history
        return prior.reset_index(drop=True)

    def own_dep_ts(self, seg_id):
        """Returns this segment's OWN departure timestamp (not a prior
        voyage's) — needed for anything that must anchor to the CURRENT
        voyage's own calendar position, e.g. the contract-period feature
        below (which January is "current" depends on the voyage being
        predicted, not on the most recent prior voyage). Returns pd.NaT
        for an unknown seg_id, so a caller can raise its own clear error
        rather than this silently returning something misleading."""
        if seg_id not in self._seg_position:
            return pd.NaT
        imo, pos = self._seg_position[seg_id]
        return self._by_imo[imo].iloc[pos]["dep_ts"]

    @staticmethod
    def _empty_history():
        return pd.DataFrame(columns=["seg_id", "dep_ts", "DEP_PORT_ID", "ARR_PORT_ID", "duration_h"])
