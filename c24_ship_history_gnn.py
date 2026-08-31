# =============================================================================
# Section 4.5 — Ship-history channel (GNN/GAT)
# Migrated verbatim from Main_forGitHub.ipynb cells [50].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 50]
# ----------------------------------------------------------------------
# =============================================================================
# LIB CELL L4 -- Step4d_ship_history (entire, verbatim)
# =============================================================================
"""
Step4d_ship_history.py
─────────────────────────────────────────────────────────────────────────────
STEP 4d — SHIP-SPECIFIC CONTEXT (Model Block 2): DAG + GNN over voyage history

Extends the causal per-vessel history already used in Step 3a (which collapsed
a vessel's past voyages into a few summary statistics) into a genuine LEARNED
representation: each vessel's past voyages become nodes in a directed acyclic
graph, and a graph attention network aggregates them into a "trade-lane
profile" embedding, fed into the main model as a 5th representation channel.

WHY THIS DOESN'T NEED ANY CHANGES TO THE CASP ARCHITECTURE (verified, not
assumed): MultiheadChannelAttention infers its channel count C from the input
shape at build time (`C = input_shape[-2]`), and the channel-replacement /
residual wiring in CASPLayer indexes channels by DESIGNATED_CHANNEL rather
than assuming exactly 4 — confirmed by running a 5-channel tensor through the
existing CASPLayer/WAYModel unchanged before writing any of this file.

GRAPH DESIGN
    Nodes   = a vessel's own past voyages (segments), strictly BEFORE the
              current segment's departure — never itself, never anything
              later. Capped to the most recent MAX_HISTORY voyages.
    Edges (all directed FORWARD in time only — this is what keeps the graph
    acyclic, and is the same causal discipline used everywhere else in this
    pipeline: a voyage can only be influenced by earlier ones):
      1. Chronological backbone: voyage i -> voyage i+1
      2. Same-departure-port:    voyage i -> voyage j (i<j), same dep_port
      3. Same-arrival-port:      voyage i -> voyage j (i<j), same arr_port
    (2) and (3) are what let the graph capture recurring trade-lane patterns
    across NON-adjacent voyages — e.g. a vessel that alternates between two
    routes gets edges connecting same-route voyages directly, not just
    through the chronological chain between them.

CAUSALITY GUARANTEE: graphs are built from a vessel's segment list INDEXED
BY POSITION (not by timestamp comparison), so "prior segments" means
"everything strictly before the current segment's own position in that
vessel's chronological list" — avoids any edge-case ambiguity from duplicate
or missing timestamps. Verified directly in the tests below: a segment's
history graph never contains that segment itself or anything after it,
under adversarial conditions (duplicate timestamps, out-of-order input).

Node features: departure port + arrival port (embedded via the SAME shared
port-embedding table used elsewhere — one "port identity" space, not a
separate table), plus voyage duration and recency (both normalized numeric
features).

Cold start (a vessel's first-ever voyage, zero history): a single LEARNED
fallback vector, not a zero vector — lets the model represent "no history"
as a distinct, trainable state rather than an arbitrary default.
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import keras
from keras import ops

MAX_HISTORY = 20          # cap on nodes (past voyages) per graph
DURATION_NORM = 500.0     # rough normalizer for duration_h (hours)
RECENCY_NORM_DAYS = 365.0 # rough normalizer for "days since that past voyage"


# ═════════════════════════════════════════════════════════════════════════════
# [1] VESSEL HISTORY INDEX — precompute per-vessel chronological segment lists
# ═════════════════════════════════════════════════════════════════════════════

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


def build_edge_mask(history_df):
    """Builds the K x K directed adjacency matrix for one segment's history
    graph, from its (already causal, already time-ordered) prior-voyage
    DataFrame. Rows/cols are in chronological order (row 0 = oldest kept).
    edge_mask[i, j] = 1 means "node i -> node j" (i is a source informing j).
    All edges point from earlier index to later index only — acyclic by
    construction, never the reverse."""
    K = len(history_df)
    mask = np.zeros((K, K), dtype="float32")
    if K == 0:
        return mask

    dep = history_df["DEP_PORT_ID"].values
    arr = history_df["ARR_PORT_ID"].values

    for i in range(K):
        for j in range(i + 1, K):
            is_edge = False
            if j == i + 1:
                is_edge = True  # (1) chronological backbone
            elif dep[i] == dep[j]:
                is_edge = True  # (2) same departure port
            elif arr[i] == arr[j]:
                is_edge = True  # (3) same arrival port
            if is_edge:
                mask[i, j] = 1.0
    return mask


# ═════════════════════════════════════════════════════════════════════════════
# [2] GRAPH ATTENTION LAYER
#
# A GAT layer is structurally "self-attention restricted to graph neighbors"
# — the exact same masked multi-head attention machinery already built and
# tested for MSA (Step4a_CASP.py), just swapping the causal upper-triangular
# mask for an arbitrary edge-adjacency mask. Reimplemented here (rather than
# importing MaskedMultiheadSelfAttention directly) so the mask is a required
# explicit argument, not an internally-computed causal one — makes misuse
# (accidentally getting a causal mask here) impossible by construction.
# ═════════════════════════════════════════════════════════════════════════════

class GraphAttentionLayer(keras.layers.Layer):
    """Input: x [batch, K, d], edge_mask [batch, K, K] (1 = i can send to j).
    Output: [batch, K, d]. A node with no incoming edges (row of zeros in
    edge_mask, i.e. nothing points TO it) still passes through via its own
    residual connection (added by the caller) — it just doesn't aggregate
    anyone else's information at this layer."""

    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def build(self, input_shape):
        h, d, dk = self.n_heads, self.d_model, self.d_k
        self.W_Q = self.add_weight(shape=(h, d, dk), initializer="glorot_uniform", name="W_Q")
        self.W_K = self.add_weight(shape=(h, d, dk), initializer="glorot_uniform", name="W_K")
        self.W_V = self.add_weight(shape=(h, d, dk), initializer="glorot_uniform", name="W_V")
        self.W_out = self.add_weight(shape=(h * dk, d), initializer="glorot_uniform", name="W_out")
        super().build(input_shape)

    def call(self, x, edge_mask):
        K = ops.shape(x)[1]
        Q = ops.einsum("bkd,hdc->bhkc", x, self.W_Q)
        Kx = ops.einsum("bkd,hdc->bhkc", x, self.W_K)
        V = ops.einsum("bkd,hdc->bhkc", x, self.W_V)

        scores = ops.einsum("bhqc,bhkc->bhqk", Q, Kx) / ops.sqrt(float(self.d_k))  # [b,h,K,K]

        # edge_mask[b, i, j] = 1 means i -> j (i sends to j). For query position
        # j attending over key positions i, that means attend where edge_mask[i,j]=1
        # i.e. transpose so the mask aligns as [query, key] = edge_mask^T.
        allow = ops.cast(ops.transpose(edge_mask, (0, 2, 1)), "bool")  # [b, K(query), K(key)]
        # A node always attends to itself too (so isolated nodes get a
        # well-defined, non-degenerate output — otherwise an all-masked row
        # would soften into a uniform, meaningless attention distribution).
        eye = ops.eye(K, dtype="bool")
        allow = ops.logical_or(allow, ops.expand_dims(eye, 0))
        allow = ops.expand_dims(allow, 1)  # [b,1,K,K] broadcast over heads

        neg = -1e9
        scores = ops.where(allow, scores, neg)
        alpha = ops.softmax(scores, axis=-1)

        out = ops.einsum("bhqk,bhkc->bhqc", alpha, V)
        out = ops.transpose(out, (0, 2, 1, 3))
        shp = ops.shape(out)
        out = ops.reshape(out, (shp[0], shp[1], self.n_heads * self.d_k))
        return ops.matmul(out, self.W_out)


# ═════════════════════════════════════════════════════════════════════════════
# [3] ATTENTION POOLING — collapse K node embeddings into one graph embedding
# ═════════════════════════════════════════════════════════════════════════════

class AttentionPool(keras.layers.Layer):
    """A single learned query vector attends over all K (valid) nodes and
    produces one pooled embedding — the same [CLS]-token-style readout idea
    used elsewhere in this pipeline's attention-based design, rather than a
    plain mean/max pool. Input: x [batch,K,d], node_mask [batch,K]
    (1=real node, 0=padding). Output: [batch, d].

    use_recency_bias=True: adds a residual bias to this pool's own
    attention scores, proportional to how old each node's voyage is —
    same pattern as every other optional addition in this project (a
    single learned scale, initialized to EXACTLY 0, so at initialization
    this produces identical output to use_recency_bias=False; training
    can only grow it if doing so actually reduces the loss). Gives the
    model a STRUCTURAL lever to down-weight older voyages in the final
    pooled trade-lane profile specifically — distinct from recency
    already being available as a passive input feature to the GAT layers
    above, which the model would otherwise have to discover the
    usefulness of entirely unaided.
    """

    def __init__(self, use_recency_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.use_recency_bias = use_recency_bias

    def build(self, input_shape):
        d = input_shape[-1]
        self.query = self.add_weight(shape=(1, d), initializer="glorot_uniform", name="pool_query")
        self.W_K = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="pool_W_K")
        self.W_V = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="pool_W_V")
        if self.use_recency_bias:
            self.recency_bias_scale = self.add_weight(
                shape=(), initializer="zeros", trainable=True, name="recency_bias_scale")
        else:
            self.recency_bias_scale = None
        super().build(input_shape)

    def call(self, x, node_mask, recency_norm=None, external_logit_bias=None):
        """external_logit_bias (optional): [batch,K] — ADDED to this
        pool's own attention scores before masking/softmax, exactly
        like the recency-bias term above but computed entirely OUTSIDE
        this layer and handed in pre-computed, rather than derived from
        a single scalar input the way recency_norm is. Lets a caller
        (e.g. ActiveVesselSetEncoder's own similarity-bias mechanism)
        bias this SAME, reused attention mechanism toward candidates
        similar to the querying vessel across whatever comparison
        dimensions IT chooses to compute — without this generic layer
        needing to know anything about what those dimensions are. None
        (default): no bias, output identical to before this parameter
        existed.
        """
        d = ops.shape(x)[-1]
        Kx = ops.matmul(x, self.W_K)          # [batch,K,d]
        Vx = ops.matmul(x, self.W_V)          # [batch,K,d]
        q = self.query                         # [1,d]

        scores = ops.einsum("bkd,qd->bk", Kx, q) / ops.sqrt(ops.cast(d, "float32"))  # [batch,K]

        if self.use_recency_bias:
            if recency_norm is None:
                raise ValueError("use_recency_bias=True requires recency_norm to be passed to call()")
            # recency_norm: larger = OLDER (days-since, normalized). Scale
            # starts at 0 (no effect); if training finds recent voyages
            # more informative, growing this scale structurally lowers
            # older nodes' pooling weight.
            scores = scores + self.recency_bias_scale * (-recency_norm)

        if external_logit_bias is not None:
            scores = scores + external_logit_bias

        valid = ops.cast(node_mask, "bool")
        scores = ops.where(valid, scores, -1e9)
        alpha = ops.softmax(scores, axis=-1)   # [batch,K]

        pooled = ops.einsum("bk,bkd->bd", alpha, Vx)  # [batch,d]
        return pooled


# ═════════════════════════════════════════════════════════════════════════════
# [4] SHIP HISTORY GNN — full module: node embedding -> GAT layers -> pooling
# ═════════════════════════════════════════════════════════════════════════════

class ShipHistoryGNN(keras.layers.Layer):
    """Produces the per-segment "trade-lane profile" embedding from a
    vessel's causal voyage-history DAG. Reuses the shared port-embedding
    table (same one used for departure port / declared destination
    elsewhere) for node departure/arrival ports — one consistent "port
    identity" space throughout the model, not a separate table here.

    Cold start (zero prior voyages): outputs a single LEARNED fallback
    vector rather than zeros — lets "no history yet" be a distinct,
    trainable representation instead of an arbitrary default.
    """

    def __init__(self, d_model, port_embed_layer, n_gat_layers=2, n_heads=4, use_recency_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.port_embed = port_embed_layer  # SHARED table, passed in, not owned here
        self.numeric_proj = keras.layers.Dense(d_model)
        self.gat_layers = [GraphAttentionLayer(d_model, n_heads) for _ in range(n_gat_layers)]
        self.norms = [keras.layers.LayerNormalization() for _ in range(n_gat_layers)]
        self.use_recency_bias = use_recency_bias
        self.pool = AttentionPool(use_recency_bias=use_recency_bias)

    def build(self, input_shape):
        self.cold_start_embed = self.add_weight(
            shape=(1, self.d_model), initializer="glorot_uniform", name="cold_start_embed")
        super().build(input_shape)

    def call(self, node_dep_port_id, node_arr_port_id, node_numeric, edge_mask, node_mask):
        # node_dep_port_id, node_arr_port_id: [batch, K] int
        # node_numeric: [batch, K, 2 or 3] float (duration_norm, recency_norm,
        # [optional] same_contract_period) — column 1 is ALWAYS recency_norm
        # regardless of whether the 3rd column is present.
        # edge_mask: [batch, K, K]; node_mask: [batch, K]
        dep_e = self.port_embed(node_dep_port_id)   # [batch,K,d]
        arr_e = self.port_embed(node_arr_port_id)   # [batch,K,d]
        num_e = self.numeric_proj(node_numeric)      # [batch,K,d]
        x = dep_e + arr_e + num_e                     # [batch,K,d] — combine, keep dim fixed at d

        for gat, norm in zip(self.gat_layers, self.norms):
            gat_out = gat(x, edge_mask)
            x = norm(gat_out + x)

        if self.use_recency_bias:
            recency_norm = node_numeric[:, :, 1]  # [batch, K]
            pooled = self.pool(x, node_mask, recency_norm=recency_norm)  # [batch, d]
        else:
            pooled = self.pool(x, node_mask)  # [batch, d]

        # Cold start: any segment with zero valid nodes gets the learned
        # fallback instead of whatever the (degenerate, all-padding) pool
        # produced.
        has_history = ops.cast(ops.sum(node_mask, axis=-1, keepdims=True) > 0, "float32")  # [batch,1]
        cold = ops.tile(self.cold_start_embed, (ops.shape(pooled)[0], 1))
        return has_history * pooled + (1.0 - has_history) * cold


# ═════════════════════════════════════════════════════════════════════════════
# [5] BATCH PREPARATION — build padded GNN inputs for a batch of segments
# ═════════════════════════════════════════════════════════════════════════════

def prepare_history_batch(history_index: VesselHistoryIndex, seg_ids, none_port_id,
                           max_history=MAX_HISTORY, use_contract_period_feature=False):
    """For each segment in seg_ids, builds its causal history graph via
    VesselHistoryIndex (already tested for causality above), then pads all
    graphs in the batch to the same K (local batch max, capped at
    max_history) — same dynamic-padding-per-batch approach used throughout
    this pipeline (Step3b's prepare_batch), not a fixed global size.

    use_contract_period_feature: if True, adds a 3rd numeric node feature —
    1.0 if that PAST voyage's own departure falls on-or-after the most
    recent January 1st BEFORE the CURRENT segment's own departure (i.e.
    "happened under the same time-charter contract period as the voyage
    being predicted"), else 0.0. Anchored to the CURRENT segment's own
    departure date (via history_index.own_dep_ts), not the most recent
    prior voyage's. Default False keeps node_numeric's shape at
    [batch,K,2], unchanged from before this feature existed.

    Returns a dict of numpy arrays: node_dep_port_id, node_arr_port_id
    [batch,K], node_numeric [batch,K,2] or [batch,K,3], edge_mask
    [batch,K,K], node_mask [batch,K].
    """
    histories = [history_index.history_for(sid, max_history=max_history) for sid in seg_ids]
    K = max(1, max(len(h) for h in histories))  # at least 1 so shapes are never zero-sized
    batch = len(seg_ids)
    n_numeric = 3 if use_contract_period_feature else 2

    node_dep = np.zeros((batch, K), dtype="int32")
    node_arr = np.zeros((batch, K), dtype="int32")
    node_num = np.zeros((batch, K, n_numeric), dtype="float32")
    edge_mask = np.zeros((batch, K, K), dtype="float32")
    node_mask = np.zeros((batch, K), dtype="float32")

    for b, (seg_id, hist) in enumerate(zip(seg_ids, histories)):
        k_here = len(hist)
        if k_here == 0:
            continue
        node_dep[b, :k_here] = hist["DEP_PORT_ID"].fillna(none_port_id).astype(int).values
        node_arr[b, :k_here] = hist["ARR_PORT_ID"].fillna(none_port_id).astype(int).values
        node_num[b, :k_here, 0] = (hist["duration_h"].values / DURATION_NORM)
        if k_here > 0:
            last_dep_ts = hist["dep_ts"].iloc[-1]
            recency_days = (last_dep_ts - hist["dep_ts"]).dt.total_seconds().values / 86400.0
            node_num[b, :k_here, 1] = recency_days / RECENCY_NORM_DAYS
        if use_contract_period_feature and k_here > 0:
            current_dep_ts = history_index.own_dep_ts(seg_id)
            if pd.notna(current_dep_ts):
                contract_start = pd.Timestamp(year=current_dep_ts.year, month=1, day=1)
                same_period = (hist["dep_ts"] >= contract_start).astype("float32").values
                node_num[b, :k_here, 2] = same_period
        node_mask[b, :k_here] = 1.0
        edge_mask[b, :k_here, :k_here] = build_edge_mask(hist)

    return {
        "node_dep_port_id": node_dep,
        "node_arr_port_id": node_arr,
        "node_numeric": node_num,
        "edge_mask": edge_mask,
        "node_mask": node_mask,
    }



# ═════════════════════════════════════════════════════════════════════════════
# [6] VISUALIZATION — draw a vessel's actual voyage-history DAG
# ═════════════════════════════════════════════════════════════════════════════

def plot_vessel_history_graph(history_index, imo, port_names=None, save_path=None,
                               max_history=MAX_HISTORY):
    """Draws one vessel's full voyage sequence as a DAG: nodes = voyages in
    chronological order, edges = the 3 causal relationship types (chrono
    backbone, same departure port, same arrival port), color-coded. Useful
    as a sanity check that the graph construction matches real trade
    patterns (a vessel on a repetitive shuttle route should show dense
    same-port edges; a tramp trader should show mostly just the backbone)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    if imo not in history_index._by_imo:
        raise ValueError(f"IMO {imo} not found in this history index")
    segs = history_index._by_imo[imo].reset_index(drop=True)
    K = len(segs)
    if K < 2:
        print(f"IMO {imo} has only {K} voyage(s) on record — nothing to visualize")
        return

    def pname(pid):
        return port_names[pid] if port_names is not None and pid in port_names else str(pid)

    G = nx.DiGraph()
    for i in range(K):
        label = f"V{i}\n{pname(segs['DEP_PORT_ID'][i])}\u2192{pname(segs['ARR_PORT_ID'][i])}"
        G.add_node(i, label=label)

    edge_mask = build_edge_mask(segs)
    chrono_edges, dep_edges, arr_edges = [], [], []
    for i in range(K):
        for j in range(i + 1, K):
            if edge_mask[i, j] == 0:
                continue
            if j == i + 1:
                chrono_edges.append((i, j))
            elif segs["DEP_PORT_ID"][i] == segs["DEP_PORT_ID"][j]:
                dep_edges.append((i, j))
            elif segs["ARR_PORT_ID"][i] == segs["ARR_PORT_ID"][j]:
                arr_edges.append((i, j))

    pos = {i: (i, 0) for i in range(K)}
    fig, ax = plt.subplots(figsize=(max(8, K * 1.3), 4.5))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1400, node_color="#dbe7f5", edgecolors="#1f3864")
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), ax=ax, font_size=8)

    nx.draw_networkx_edges(G, pos, edgelist=chrono_edges, ax=ax, edge_color="#444444",
                            arrows=True, arrowsize=14, width=1.5, connectionstyle="arc3,rad=0.0")
    nx.draw_networkx_edges(G, pos, edgelist=dep_edges, ax=ax, edge_color="#2e75b6",
                            arrows=True, arrowsize=12, width=1.3, connectionstyle="arc3,rad=0.35")
    nx.draw_networkx_edges(G, pos, edgelist=arr_edges, ax=ax, edge_color="#c55a11",
                            arrows=True, arrowsize=12, width=1.3, connectionstyle="arc3,rad=-0.35")

    legend = [
        mpatches.Patch(color="#444444", label="chronological (i \u2192 i+1)"),
        mpatches.Patch(color="#2e75b6", label="same departure port"),
        mpatches.Patch(color="#c55a11", label="same arrival port"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    ax.set_title(f"Voyage-history DAG \u2014 IMO {imo}  ({K} voyages)")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {save_path}")
    plt.close()
