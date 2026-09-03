# Section 4.2 — Input channel creation
# Executed by runner.py inside the shared namespace (notebook-kernel style).

# Reimport libraries - in case rerunning from this checkpoint only
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from keras import ops
warnings.filterwarnings("ignore")
# WORK_DIR = Path.cwd()

# CONFIGURATION

D_MODEL = 32           # kept small for faster trainning and efficiency
GRU_LAYERS = 1         # depth of the Stepwise GRU stack
EXAMPLE_SEG_ID = None  # None = reuse the same auto-picked example as Step3a's
                       # own logic (moderate length); set explicitly to match

# Demo-only input paths removed; file paths are handled by the input check
# (core/c02_input_checks.py) and the dataset build (core/c14_build_channels.py).

# [1] LAYERS

class SpatialEncoding(keras.layers.Layer):
    """Turns (longitude, latitude) coordinate into a
    d_model-dimensional vector, the same way Transformer positional encoding
    turns an integer position into a vector, so that a downstream model can
    use spatial location as a normal continuous feature instead of a raw
    (lat, lon) pair.

    Input: lam_deg, phi_deg (grid-cell centroid lon/lat, in DEGREES), any
    matching shape [...]. Output: [..., d_model]. d_model must be divisible
    by 4 (the formula produces output in interleaved GROUPS OF 4 dimensions,
    see call() below).

    Confirmed constant: (log(pi))**2 multiplies dims 4i+1 (positive) and
    4i+3 (negative). Earlier PDF text-extraction rendered this ambiguously
    as "(log pi) 2 * sin(...)"; confirmed against the original as squared,
    not halved.
    """

    def __init__(self, d_model, phi_scale=None, **kwargs):
        super().__init__(**kwargs)
        assert d_model % 4 == 0, "SpatialEncoding requires d_model divisible by 4"
        self.d_model = d_model
        # (log(pi))^2 \u2248 1.3104 \u2014 a fixed scalar, not learned, applied only to
        # the two "latitude-only" dimensions (see call() for what that means).
        self.phi_scale = phi_scale if phi_scale is not None else float(np.log(np.pi) ** 2)

    def call(self, lam_deg, phi_deg):
        # Formula operates in RADIANS, so convert from degrees first.
        lam = lam_deg * (np.pi / 180.0)   # longitude, radians
        phi = phi_deg * (np.pi / 180.0)   # latitude, radians

        # The output is built in GROUPS OF 4 dimensions. n_quad = how many
        # groups of 4 fit in d_model (e.g. d_model=32 -> 8 groups -> i=0..7).
        n_quad = self.d_model // 4
        idx4 = ops.arange(n_quad, dtype="float32") * 4.0   # [0,4,8,...] one per group
        exponent = idx4 / float(self.d_model)
        # denom grows with the group index, exactly like the 10000^(2i/d)
        # denominator in standard Transformer positional encoding, but with
        # base 2*pi instead of 10000. This means: EARLY groups (small i) use
        # a SHORT wavelength (change fast as lon/lat changes -> fine detail),
        # LATE groups (large i) use a LONG wavelength (change slowly ->
        # coarse, large-scale position). Stacking many groups together lets
        # the model reconstruct position at multiple spatial resolutions
        # simultaneously, the same trick that makes Transformer PE work.
        denom = ops.power(2.0 * np.pi, exponent)

        # g_lam / g_phi = the longitude/latitude, "slowed down" by each
        # group's wavelength -> the raw angle fed into sin/cos below.
        g_lam = ops.expand_dims(lam, -1) / denom   # [..., n_quad]
        g_phi = ops.expand_dims(phi, -1) / denom   # [..., n_quad]

        # The 4 dimensions within each group are NOT independent random
        # features — they're built from the polar-to-Cartesian conversion
        # (x=r*cos(theta), y=r*sin(theta)), applied twice (once combining
        # both lon and lat, once using only lat):
        #   d0 = cos(lat-angle) * sin(lon-angle)   -- "x-like": mixes both
        #   d2 = cos(lat-angle) * cos(lon-angle)   -- "y-like": mixes both
        #     (d0, d2) together behave like a 2D Cartesian point that wraps
        #     smoothly around the globe as longitude increases by 360 deg,
        #     which is exactly why SE captures the CYCLIC nature of
        #     longitude (day-line wraparound) that a raw longitude number
        #     can't represent (e.g. 179 deg and -179 deg are physically
        #     next to each other, but numerically far apart).
        #   d1 =  phi_scale * sin(lat-angle)       -- "latitude-only" term
        #   d3 = -phi_scale * sin(lat-angle)       -- its negation
        #     (d1, d3) depend on latitude ALONE (no longitude), giving the
        #     encoding a way to represent "how far from the equator" that
        #     doesn't wrap around (the poles are NOT cyclic like longitude
        #     is), which is why d1/d3 are a separate, simpler pair, scaled
        #     by the fixed constant (log(pi))^2 to balance their magnitude
        #     against the d0/d2 pair (which are bounded by [-1,1] products
        #     of two sinusoids, i.e. naturally smaller on average).
        d0 = ops.cos(g_phi) * ops.sin(g_lam)
        d1 = self.phi_scale * ops.sin(g_phi)
        d2 = ops.cos(g_phi) * ops.cos(g_lam)
        d3 = -self.phi_scale * ops.sin(g_phi)

        # Interleave [d0,d1,d2,d3] per group so the final vector is ordered
        # dim 0,1,2,3 = group 0's (d0,d1,d2,d3); dim 4,5,6,7 = group 1's; etc.
        # matching the paper's indexing SE(...,4i), SE(...,4i+1), etc.
        stacked = ops.stack([d0, d1, d2, d3], axis=-1)     # [..., n_quad, 4]
        out_shape = ops.shape(stacked)[:-2] + (self.d_model,)
        return ops.reshape(stacked, out_shape)              # [..., d_model]


class TimeEncoding(keras.layers.Layer):
    """WAY Eq. 9. Input: tau (day-unit real-valued time distance), any
    shape [...]. Output: [..., d_model]. d_model must be divisible by 2."""

    def __init__(self, d_model, base=1000.0, **kwargs):
        super().__init__(**kwargs)
        assert d_model % 2 == 0, "TimeEncoding requires d_model divisible by 2"
        self.d_model = d_model
        self.base = base

    def call(self, tau):
        n_pair = self.d_model // 2
        i = ops.arange(n_pair, dtype="float32")
        exponent = (2.0 * i) / float(self.d_model)
        denom = ops.power(self.base, exponent)

        angle = ops.expand_dims(tau, -1) / denom
        even = ops.cos(angle)
        odd = ops.sin(angle)
        stacked = ops.stack([even, odd], axis=-1)
        out_shape = ops.shape(stacked)[:-2] + (self.d_model,)
        return ops.reshape(stacked, out_shape)


class StepwiseGRU(keras.layers.Layer):
    """Local navigational pattern channel (Section IV-A-2). Input:
    local_x [batch, N, mk_max, f], local_mask [batch, N, mk_max] (1=valid,
    0=pad). Runs an independent GRU stack over each grid-step's own raw
    subsequence (folding batch*N into one axis) and returns the FINAL
    hidden state per grid-step -> [batch, N, d_model]."""

    def __init__(self, d_model, n_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layers = n_layers
        self.masking = keras.layers.Masking(mask_value=0.0)
        self.grus = [keras.layers.GRU(d_model, return_sequences=(l < n_layers - 1))
                     for l in range(n_layers)]

    def call(self, local_x, local_mask=None):
        shape = ops.shape(local_x)
        batch, N, mk_max, f = shape[0], shape[1], shape[2], shape[3]
        x = ops.reshape(local_x, (batch * N, mk_max, f))
        if local_mask is not None:
            mask_f = ops.reshape(ops.cast(local_mask, "float32"), (batch * N, mk_max, 1))
            x = x * mask_f
        x = self.masking(x)
        for gru in self.grus:
            x = gru(x)
        return ops.reshape(x, (batch, N, self.d_model))


class RepresentationLayer(keras.layers.Layer):
    """Assembles the 4-channel vector sequence x \u2208 [batch, C=4, N, d_model].

    CHANNEL-COMBINATION DESIGN NOTE (why TE is added into every channel
    rather than being its own 5th channel): "[TE] is added
    to W^Y_x and W^S_x each while duplicating [them] for N steps, as well as
    to the sequence outputs, then "the
    concatenation of such representations yields a four-channel vector
    sequence x \u2208 R^(C\u00d7N\u00d7d), where C=4." If TE were a standalone channel,
    we would have C=5 (SE, local-pattern, Y, S, TE), but we keep 
    C=4 explicitly. C=4 is TE injected
    additively into each of the 4 substantive channels (analogous to
    standard Transformer positional-encoding injection), which is what's
    implemented here.

    Auxiliary declared-destination signal (design addition): folded into the local-pattern channel's per-point feature
    vector, not as a separate channel — see module docstring.
    """

    def __init__(self, d_model, n_ports, n_size_classes, gru_layers=1,
                 use_spatial_channel=True, use_local_pattern_channel=True,
                 use_departure_port_channel=True, use_ship_size_channel=True,
                 use_temporal_encoding=True,
                 use_declared_destination=True, use_ship_history=False,
                 history_gat_layers=2, history_gat_heads=4, gate_ship_history=False,
                 ship_history_attention=False, use_recency_bias=False, use_contract_period_feature=False,
                 use_departure_subregion_channel=False, n_subregions_departure=None, dep_port_to_subregion_lookup=None,
                 use_eta_channel=False,
                 use_fleet_context=False, n_subregions_fleet=None,
                 use_candidate_fleet_context=False, n_subregions_candidate_fleet=None,
                 candidate_fleet_use_identity_embed=False, candidate_fleet_n_subregions=None,
                 candidate_fleet_identity_embed_dim=8,
                 use_fixed_horizon_fleet_context=False, n_subregions_fixed_horizon=None,
                 n_fixed_horizons=5,
                 use_active_vessel_set_context=False, n_subregions_active_vessel=None,
                 active_vessel_include_temporal_history=True, active_vessel_use_similarity_bias=False,
                 active_vessel_size_sigma=None, active_vessel_draught_sigma=None,
                 active_vessel_distance_sigma_km=None, active_vessel_duration_sigma_days=None,
                 active_vessel_history_sigmas=None, **kwargs):
        super().__init__(**kwargs)
        if gate_ship_history and ship_history_attention:
            raise ValueError("gate_ship_history and ship_history_attention are two different mechanisms "
                              "for the same purpose (how much to trust ship history) — combining both would "
                              "be redundant and ambiguous. Choose one.")
        if use_departure_subregion_channel and (n_subregions_departure is None or dep_port_to_subregion_lookup is None):
            raise ValueError("use_departure_subregion_channel=True requires BOTH n_subregions_departure and "
                              "dep_port_to_subregion_lookup (a length-n_ports array mapping port id -> subregion id)")
        if use_fixed_horizon_fleet_context and n_subregions_fixed_horizon is None:
            raise ValueError("use_fixed_horizon_fleet_context=True requires n_subregions_fixed_horizon")
        self.d_model = d_model
        # Base-channel ablation flags — ALL default True, matching the
        # architecture's original, unconditional behavior exactly (these
        # 4 channels had NO on/off switch at all before this — every
        # existing checkpoint/caller is completely unaffected). Added
        # specifically so a leave-one-out ablation can assess these 4
        # the same way the optional channels below already could.
        self.use_spatial_channel = use_spatial_channel
        self.use_local_pattern_channel = use_local_pattern_channel
        self.use_departure_port_channel = use_departure_port_channel
        self.use_ship_size_channel = use_ship_size_channel
        # NOT a channel-ablation flag in the same sense as the 4 above —
        # temporal encoding isn't its own entry in the channels list at
        # all, it's a shared additive term folded into EVERY channel
        # (base and optional alike, 9 separate addition sites). Default
        # True, zero behavior change unless explicitly set.
        self.use_temporal_encoding = use_temporal_encoding
        self.use_declared_destination = use_declared_destination
        self.use_ship_history = use_ship_history
        self.gate_ship_history = gate_ship_history
        self.ship_history_attention = ship_history_attention
        self.use_recency_bias = use_recency_bias
        self.use_contract_period_feature = use_contract_period_feature
        self.use_departure_subregion_channel = use_departure_subregion_channel
        self.use_eta_channel = use_eta_channel
        self.use_fleet_context = use_fleet_context
        self.use_candidate_fleet_context = use_candidate_fleet_context
        self.use_fixed_horizon_fleet_context = use_fixed_horizon_fleet_context
        self.use_active_vessel_set_context = use_active_vessel_set_context
        self.n_subregions_active_vessel = n_subregions_active_vessel
        self.active_vessel_include_temporal_history = active_vessel_include_temporal_history
        self.active_vessel_use_similarity_bias = active_vessel_use_similarity_bias
        self.active_vessel_size_sigma = active_vessel_size_sigma
        self.active_vessel_draught_sigma = active_vessel_draught_sigma
        self.active_vessel_distance_sigma_km = active_vessel_distance_sigma_km
        self.active_vessel_duration_sigma_days = active_vessel_duration_sigma_days
        self.active_vessel_history_sigmas = active_vessel_history_sigmas
        self.se = SpatialEncoding(d_model)
        self.te = TimeEncoding(d_model)
        self.local_gru = StepwiseGRU(d_model, n_layers=gru_layers)
        # Shared port-identity embedding: used for the departure-port
        # semantic channel (W^Y), the auxiliary declared-destination
        # feature, AND (if enabled) the ship-history graph's node features
        # — one consistent "port identity" space throughout, not separate
        # tables per use.
        self.port_embed = keras.layers.Embedding(n_ports + 1, d_model)  # +1 = NONE_DECLARED id
        self.size_embed = keras.layers.Embedding(n_size_classes, d_model)
        if use_departure_subregion_channel:
            # Genuinely NEW main-pathway channel (Multi-Signal Fusion Blocks (MSF)
            # sees this alongside spatial/motion/port/size), distinct from
            # the EXISTING use_departure_gate mechanism elsewhere in this
            # project, which only ever feeds the mixture-of-experts GATE,
            # never the core representation itself. dep_port_to_subregion_lookup
            # is precomputed OUTSIDE this class (avoids importing
            # build_port_to_subregion_map from Step4c_train, which would
            # create a circular import) and passed in as a plain array —
            # static data, not a learnable parameter, so it's stored
            # directly rather than via add_weight.
            self.departure_subregion_embed = keras.layers.Embedding(n_subregions_departure, d_model)
            self._dep_port_to_subregion_lookup = ops.convert_to_tensor(dep_port_to_subregion_lookup, dtype="int32")
        if use_eta_channel:
            # NEW main-pathway channel, distinct from the
            # EXISTING alt_progression_modes=["eta"] mechanism elsewhere
            # in this project, which only ever feeds the mixture-of-experts
            # GATE. The actual per-step ETA-derived progression VALUES are
            # computed OUTSIDE this class (same reason as above — avoids
            # importing precompute_eta_progression_lookup from
            # Step4c_train) and are expected as inputs["eta_channel_values"]
            # at call() time — this class only owns the projection layer.
            self.eta_channel_proj = keras.layers.Dense(d_model)
        if use_ship_history:
            # [inlined -- name defined by a library cell] from Step4d_ship_history import ShipHistoryGNN
            self.ship_history_gnn = ShipHistoryGNN(
                d_model, port_embed_layer=self.port_embed,
                n_gat_layers=history_gat_layers, n_heads=history_gat_heads,
                use_recency_bias=use_recency_bias)
            # gate_ship_history=False (default — matches original,
            # unconditional behavior, so existing checkpoints trained
            # before this option existed stay loadable without retraining):
            # ch4 = h_bcast + te_out always, same as before this was ever
            # added. gate_ship_history=True: same residual pattern that
            # fixed the MoE content-aware gate — a single learned scale,
            # initialized to EXACTLY 0, applied to the history embedding
            # specifically (NOT the time encoding — every channel gets
            # +te_out regardless, so at scale=0 this channel is still a
            # valid, neutral, positionally-informative channel, not a
            # degenerate all-zero one; only the ship-history CONTENT
            # starts at zero). At initialization this channel then
            # contributes nothing beyond position, exactly as if
            # use_ship_history were False — MSA never has to reconcile a
            # noisy, untrained signal against the other channels from step
            # one. The model can only grow this scale if doing so actually
            # reduces the loss, giving it a real, gradient-driven way to
            # decide how much to trust vessel history — rather than that
            # decision being all-or-nothing, set once in advance by
            # use_ship_history=True/False. Kept OPTIONAL specifically so
            # the original, ungated behavior remains directly comparable
            # against this one, same as every other toggle in this project.
            if gate_ship_history:
                self.history_channel_scale = self.add_weight(
                    shape=(), initializer="zeros", trainable=True, name="history_channel_scale")
            else:
                self.history_channel_scale = None

            # ship_history_attention=True: a GENUINELY different mechanism
            # from gate_ship_history above — that scale is a single global
            # number, the SAME trust level applied uniformly to every step
            # of every voyage. This instead computes a PER-STEP trust
            # weight, conditioned on that step's own spatial+time context
            # (ch0, already computed by this point) — e.g. the model could
            # learn to lean on a vessel's history heavily right after
            # departure (when position alone is most ambiguous) and
            # discount it once position evidence becomes decisive, or
            # weight it differently depending on WHERE the vessel
            # currently is, not just a single fixed "how much do we trust
            # history overall" number. Small MLP: Dense(16, relu) ->
            # Dense(1), with the FINAL layer's kernel and bias explicitly
            # zero-initialized — not the default initializer — so the very
            # first forward pass produces a gate output of EXACTLY 0
            # regardless of what ch0 contains, matching the same
            # "identical to the ungated baseline at step one" guarantee
            # used everywhere else in this project, just enforced via
            # layer initialization here instead of a standalone add_weight
            # scalar, since the gate's output now varies by input rather
            # than being a single learned number.
            if ship_history_attention:
                self.history_attn_dense = keras.layers.Dense(16, activation="relu")
                self.history_attn_out = keras.layers.Dense(
                    1, kernel_initializer="zeros", bias_initializer="zeros")
            else:
                self.history_attn_dense = None
                self.history_attn_out = None
        if use_fleet_context:
            # [inlined -- name defined by a library cell] from Step4e_fleet_context import FleetHeadingEncoder
            if n_subregions_fleet is None:
                raise ValueError("use_fleet_context=True requires n_subregions_fleet "
                                  "(the fleet-heading vector width) to be specified.")
            self.fleet_heading_encoder = FleetHeadingEncoder(d_model)
        if use_candidate_fleet_context:
            if n_subregions_candidate_fleet is None:
                raise ValueError("use_candidate_fleet_context=True requires n_subregions_candidate_fleet "
                                  "(the candidate-vector width) to be specified.")
            if candidate_fleet_use_identity_embed:
                # [inlined -- name defined by a library cell] from Step4e_fleet_context import CandidateFleetEncoderWithIdentity
                if candidate_fleet_n_subregions is None:
                    raise ValueError("candidate_fleet_use_identity_embed=True requires "
                                      "candidate_fleet_n_subregions (the raw subregion count, not "
                                      "the flattened vector width) to correctly reshape per candidate.")
                per_candidate_width = n_subregions_candidate_fleet // candidate_fleet_n_subregions
                self.candidate_fleet_encoder = CandidateFleetEncoderWithIdentity(
                    d_model, n_subregions=candidate_fleet_n_subregions,
                    per_candidate_width=per_candidate_width, embed_dim=candidate_fleet_identity_embed_dim)
            else:
                # from Step4e_fleet_context import CandidateFleetEncoder
                # NOTE: this is NOT FleetHeadingEncoder — that one applies
                # plain log1p, which is undefined for the negative values
                # this channel's input routinely has (it's a signed
                # deviation, not a non-negative count). CandidateFleetEncoder
                # uses signed-log compression instead, specifically to
                # handle that correctly.
                self.candidate_fleet_encoder = CandidateFleetEncoder(d_model)
        if use_fixed_horizon_fleet_context:
            # from Step4e_fleet_context import FixedHorizonFleetEncoder
            # n_subregions_fixed_horizon is the FLATTENED width
            # (n_horizons * n_subregions) -- matches the shape
            # prepare_fixed_horizon_fleet_batch actually produces, same
            # convention as n_subregions_candidate_fleet above (flattened,
            # not the raw per-horizon subregion count).
            self.fixed_horizon_fleet_encoder = FixedHorizonFleetEncoder(d_model)
        if use_active_vessel_set_context:
            # [inlined -- name defined by a library cell] from Step4e_fleet_context import ActiveVesselSetEncoder
            if n_subregions_active_vessel is None:
                raise ValueError("use_active_vessel_set_context=True requires n_subregions_active_vessel -- "
                                  "needed for ActiveVesselSetEncoder's own departure-subregion embedding table.")
            # Shares self.se/self.size_embed/self.port_embed (already
            # constructed above) rather than owning separate tables --
            # one consistent position/size/port-identity space
            # throughout the model, same principle ShipHistoryGNN
            # already establishes for its own shared port_embed.
            # Sigma overrides only passed through when explicitly given
            # (not None) -- otherwise ActiveVesselSetEncoder's own
            # already-calibrated defaults (Step4e_fleet_context.py's
            # DEFAULT_* constants) apply, not a silently-overridden None.
            sigma_kwargs = {}
            if active_vessel_size_sigma is not None:
                sigma_kwargs["size_sigma"] = active_vessel_size_sigma
            if active_vessel_draught_sigma is not None:
                sigma_kwargs["draught_sigma"] = active_vessel_draught_sigma
            if active_vessel_distance_sigma_km is not None:
                sigma_kwargs["distance_sigma_km"] = active_vessel_distance_sigma_km
            if active_vessel_duration_sigma_days is not None:
                sigma_kwargs["duration_sigma_days"] = active_vessel_duration_sigma_days
            if active_vessel_history_sigmas is not None:
                sigma_kwargs["history_sigmas"] = active_vessel_history_sigmas
            self.active_vessel_set_encoder = ActiveVesselSetEncoder(
                d_model, spatial_encoding_layer=self.se, size_embed_layer=self.size_embed,
                port_embed_layer=self.port_embed, n_subregions=n_subregions_active_vessel,
                include_temporal_history=active_vessel_include_temporal_history,
                use_similarity_bias=active_vessel_use_similarity_bias, **sigma_kwargs)

    def call(self, inputs):
        grid_lon = inputs["grid_lon"]       # [batch, N]
        grid_lat = inputs["grid_lat"]       # [batch, N]
        tau = inputs["tau"]                 # [batch, N]  (TIME_OFFSET_DAYS)
        dep_port_id = inputs["dep_port_id"]  # [batch]
        size_class_id = inputs["size_class_id"]  # [batch]
        local_numeric = inputs["local_numeric"]      # [batch, N, mk_max, f_numeric]
        local_declared_dest_id = inputs["local_declared_dest_id"]  # [batch, N, mk_max]
        local_mask = inputs["local_mask"]   # [batch, N, mk_max]

        N = ops.shape(grid_lon)[1]

        te_out = self.te(tau)  # [batch, N, d]
        if not self.use_temporal_encoding:
            # Zeroed HERE, once -- every "+ te_out" addition below (9
            # separate sites: every channel, base and optional alike)
            # becomes a no-op automatically, rather than needing each
            # site touched individually. TE isn't a standalone channel
            # like the other 4 base ones (it's not its own entry in the
            # channels list at all) -- it's a shared additive term folded
            # into EVERY channel's own representation, so ablating it
            # means stripping time information from all of them at once,
            # not removing one channel from the stack.
            te_out = ops.zeros_like(te_out)

        channels = []

        # Channel 0: Spatial Encoding + TE
        if self.use_spatial_channel:
            se_out = self.se(grid_lon, grid_lat)  # [batch, N, d]
            channels.append(se_out + te_out)

        # Local pattern feature vector = numeric features concat with the
        # SHARED port embedding of the declared destination at each ping —
        # UNLESS use_declared_destination=False (ablation), in which case
        # BOTH the categorical embedding AND the numeric confidence column
        # (_DECL_CONF, the last column of local_numeric — see Step3Data's
        # LOCAL_NUMERIC_COLS order) are excluded, since both derive from
        # the same declared-destination signal and either alone would leak
        # it back in. Skipped entirely (no GRU call at all) when
        # use_local_pattern_channel=False -- this flag being off makes
        # use_declared_destination moot, since there is no channel left
        # for it to be folded into.
        if self.use_local_pattern_channel:
            if self.use_declared_destination:
                declared_embed = self.port_embed(local_declared_dest_id)  # [batch,N,mk_max,d]
                local_x = ops.concatenate([local_numeric, declared_embed], axis=-1)
            else:
                local_x = local_numeric[..., :-1]  # drop _DECL_CONF, no embedding concat

            # Channel 1: Stepwise GRU (local pattern) + TE
            local_out = self.local_gru(local_x, local_mask)  # [batch, N, d]
            channels.append(local_out + te_out)

        # Channel 2: departure port semantic embedding (broadcast) + TE
        if self.use_departure_port_channel:
            y_embed = self.port_embed(dep_port_id)          # [batch, d]
            y_bcast = ops.repeat(ops.expand_dims(y_embed, 1), N, axis=1)  # [batch,N,d]
            channels.append(y_bcast + te_out)

        # Channel 3: ship size-class semantic embedding (broadcast) + TE
        if self.use_ship_size_channel:
            s_embed = self.size_embed(size_class_id)         # [batch, d]
            s_bcast = ops.repeat(ops.expand_dims(s_embed, 1), N, axis=1)
            channels.append(s_bcast + te_out)

        # Channel (optional): departure subregion, as a genuine main-
        # pathway semantic channel — same broadcast-static pattern as
        # channels 2/3 (departure port, ship size). Subregion is derived
        # from dep_port_id via the precomputed lookup passed in at
        # construction, entirely inside this call, no external dependency.
        if self.use_departure_subregion_channel:
            dep_subregion_id = ops.take(self._dep_port_to_subregion_lookup, dep_port_id)  # [batch] int
            ds_embed = self.departure_subregion_embed(dep_subregion_id)  # [batch, d]
            ds_bcast = ops.repeat(ops.expand_dims(ds_embed, 1), N, axis=1)  # [batch, N, d]
            channels.append(ds_bcast + te_out)

        # Channel (optional): ETA-derived progression, as a genuine main-
        # pathway channel — PER STEP (not broadcast-static), same
        # treatment as fleet-context channels 5/6. inputs["eta_channel_values"]
        # is computed and injected by the caller BEFORE this call (see
        # train_residual_progression_variant), not by this class or
        # prepare_batch — kept that way specifically to avoid a circular
        # import (the ETA lookup logic lives in Step4c_train.py, which
        # itself imports from this module).
        if self.use_eta_channel:
            eta_values = inputs["eta_channel_values"]  # [batch, N] float
            eta_proj = self.eta_channel_proj(ops.expand_dims(eta_values, -1))  # [batch, N, d]
            channels.append(eta_proj + te_out)

        # Channel 4 (optional, Model Block 2 — "ship-specific context"):
        # trade-lane profile embedding from the vessel's causal voyage-
        # history DAG, broadcast across all N steps + TE, same pattern as
        # channels 2/3. Architecturally just a 5th static semantic channel
        # — CASP's channel attention/self-attention/feed-forward already
        # generalize to any channel count (verified directly, not assumed),
        # so nothing downstream needs to change to accommodate this.
        if self.use_ship_history:
            history_embed = self.ship_history_gnn(
                inputs["node_dep_port_id"], inputs["node_arr_port_id"],
                inputs["node_numeric"], inputs["edge_mask"], inputs["node_mask"])  # [batch,d]
            h_bcast = ops.repeat(ops.expand_dims(history_embed, 1), N, axis=1)
            if self.gate_ship_history:
                ch4 = self.history_channel_scale * h_bcast + te_out
            elif self.ship_history_attention:
                # Spatial position + time, as the per-step context the
                # gate conditions on, so trust in vessel history can vary
                # by WHERE/WHEN the vessel currently is, not just one
                # fixed global amount. Recomputed HERE independently of
                # use_spatial_channel -- that flag controls whether
                # spatial position is included as one of the model's own
                # main channels, a separate question from whether THIS
                # attention mechanism gets to see it; se_out/te_out are
                # cheap, so recomputing costs nothing and keeps this
                # correct even when use_spatial_channel=False (an
                # ablation of the base channels shouldn't silently break
                # ship_history_attention's own signal).
                se_for_attn = self.se(grid_lon, grid_lat) + te_out
                attn_weight = self.history_attn_out(self.history_attn_dense(se_for_attn))  # [batch, N, 1]
                ch4 = attn_weight * h_bcast + te_out
            else:
                ch4 = h_bcast + te_out
            channels.append(ch4)

        # Channel 5 (optional, Model Block 3 — "fleet-specific context",
        # oracle heading signal): UNLIKE channels 2/3/4 above, this is NOT
        # broadcast-static — it's computed fresh PER STEP from that step's
        # own real calendar date (a voyage can span weeks; the fleet looks
        # different at the start than the end), closer in spirit to
        # grid_lon/grid_lat/tau than to a static per-segment fact.
        if self.use_fleet_context:
            fleet_embed = self.fleet_heading_encoder(inputs["fleet_heading_counts"])  # [batch,N,d]
            ch5 = fleet_embed + te_out
            channels.append(ch5)

        # Channel 6 (optional — candidate-conditioned future fleet state):
        # a genuinely different mechanism from channel 5 — one deviation
        # value per CANDIDATE destination (real position tracking,
        # projected to each candidate's own expected arrival date, vs.
        # seasonal baseline), not one flat "who's converging right now"
        # vector. Same per-step, non-broadcast treatment as channel 5.
        if self.use_candidate_fleet_context:
            cand_embed = self.candidate_fleet_encoder(inputs["candidate_fleet_state"])  # [batch,N,d]
            ch6 = cand_embed + te_out
            channels.append(ch6)

        # Channel 7 (optional — fixed-horizon fleet context): a
        # DIFFERENT premise from channel 5's own "oracle heading" signal
        # — that one uses each OTHER vessel's own TRUE, final arrival
        # subregion in a window anchored to ITS OWN eventual arrival,
        # reflecting the fleet AFTER real-world competition/chartering
        # has already resolved who goes where. This instead queries
        # ACTUAL recorded positions (not final destination) at FIXED
        # horizons (e.g. 10/20/30/40/50 days) from each step's own
        # current date — the raw, still-unsettled occupancy at each
        # future moment, not the cleared outcome. Same per-step,
        # non-broadcast treatment as channels 5/6.
        if self.use_fixed_horizon_fleet_context:
            fh_embed = self.fixed_horizon_fleet_encoder(inputs["fixed_horizon_fleet_occupancy"])  # [batch,N,d]
            ch7 = fh_embed + te_out
            channels.append(ch7)

        # Channel 8 (optional — active vessel set, set-pooling): a
        # genuinely different mechanism from every other fleet channel
        # — instead of aggregating other vessels into counts or per-
        # candidate deviations, retrieves the actual SET of other
        # active vessels at each step's own current moment, each with
        # its own small feature vector (position, draught, size,
        # current declared destination), pooled via permutation-
        # invariant attention (AttentionPool, reused from Block 2's own
        # ShipHistoryGNN). Same per-step, non-broadcast treatment as
        # channels 5/6/7.
        if self.use_active_vessel_set_context:
            av_embed = self.active_vessel_set_encoder(
                inputs["active_vessel_features"], inputs["active_vessel_mask"])  # [batch,N,d]
            ch8 = av_embed + te_out
            channels.append(ch8)

        x = ops.stack(channels, axis=1)  # [batch, C, N, d]  C=4 through 9
        return x


# Subfolder name (under work_dir) where Step3Data loads its 4 required
# preprocessing-output files (trajectories_gridded.parquet,
# segment_steps_index.parquet, trajectories_index_enriched.csv,
# step3_vocabularies.json). Default "" loads directly from work_dir
# itself (this project's original convention). Change this ONE value
# (e.g. Step3b_representation_layer.DATA_SUBFOLDER = "Model_Inputs") if a
# given work_dir keeps its preprocessing outputs in a named subfolder
# instead — this constant applies wherever Step3Data is constructed, no
# need to pass it to each call individually.
