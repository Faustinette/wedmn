# Section 4.3 — Model architecture (Step4a, verbatim)
# Executed by runner.py inside the shared namespace (notebook-kernel style).

# Step4a (entire, verbatim)

"""
STEP 4a —

Implements MSF channels and MoEFF

Consumes Step 3b's representation-layer output x \u2208 [batch, C=4, N, d].

ARCHITECTURE

"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from keras import ops

DESIGNATED_CHANNEL = 0  # index of the channel repeatedly replaced (= SE
                         # channel's slot, per Step3b's channel ordering:
                         # 0=SE, 1=local-pattern, 2=departure, 3=ship-type)



# [1] MULTIHEAD CHANNEL ATTENTION (MCA) — Eq. 10

class MultiheadChannelAttention(keras.layers.Layer):
    """Eq. 10. Input: x [batch, N, C, d]. Output: [batch, N, d].

    CBAM-style channel attention (avg-pool AND max-pool branches through a
    SHARED bottleneck MLP, summed, sigmoid), extended to h independent
    heads, each with its own linear projection W^tr_i and its own
    bottleneck weights (W_sq, W_ex) — heads are NOT weight-shared with each
    other, only the avg/max branches share weights within a head.
    """

    def __init__(self, d_model, n_heads, gamma=2, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.gamma = gamma

    def build(self, input_shape):
        C = input_shape[-2]
        h, d, dk, gamma = self.n_heads, self.d_model, self.d_k, self.gamma
        C_r = max(1, C // gamma)
        # W^tr_i: per-head projection from d_model -> d_k
        self.W_tr = self.add_weight(shape=(h, d, dk), initializer="glorot_uniform", name="W_tr")
        # Shared (within a head, across avg/max branches) bottleneck weights
        self.W_sq = self.add_weight(shape=(h, C, C_r), initializer="glorot_uniform", name="W_sq")
        self.W_ex = self.add_weight(shape=(h, C_r, C), initializer="glorot_uniform", name="W_ex")
        # Output projection back to d_model, from concatenated heads
        self.W_out = self.add_weight(shape=(h * dk, d), initializer="glorot_uniform", name="W_out")
        super().build(input_shape)

    def _bottleneck(self, z):
        # z: [batch, N, h, C] -> same shape out
        t = ops.einsum("bnhc,hcr->bnhr", z, self.W_sq)
        t = ops.relu(t)
        out = ops.einsum("bnhr,hrc->bnhc", t, self.W_ex)
        return out

    def call(self, x):
        # x: [batch, N, C, d]
        x_h = ops.einsum("bncd,hdk->bnhck", x, self.W_tr)  # [batch,N,h,C,dk]

        z_avg = ops.mean(x_h, axis=-1)   # [batch,N,h,C]
        z_max = ops.max(x_h, axis=-1)

        zp_avg = self._bottleneck(z_avg)
        zp_max = self._bottleneck(z_max)
        alpha = ops.sigmoid(zp_avg + zp_max)          # [batch,N,h,C]

        # Stored (not returned) for introspection — e.g. Step4c_train.py's
        # get_channel_attention_summary() reads this after a forward pass to
        # report how much the model emphasizes each of the 4 channels. Not
        # part of the paper's formula; a diagnostic side-channel only.
        self.last_alpha = alpha

        x_tilde = ops.expand_dims(alpha, -1) * x_h     # [batch,N,h,C,dk]

        # headwise concat: [batch,N,h,C,dk] -> [batch,N,C,h,dk] -> [batch,N,C,h*dk]
        x_tilde = ops.transpose(x_tilde, (0, 1, 3, 2, 4))
        shp = ops.shape(x_tilde)
        x_tilde = ops.reshape(x_tilde, (shp[0], shp[1], shp[2], self.n_heads * self.d_k))

        x_pooled = ops.max(x_tilde, axis=2)             # [batch,N,h*dk] — max over channels
        out = ops.matmul(x_pooled, self.W_out)          # [batch,N,d]
        return out



# [2] CFA — Eq. 11

class MaskedMultiheadSelfAttention(keras.layers.Layer):
    """Eq. 11. Standard causal Transformer-decoder self-attention. Input:
    x [batch, N, d], optional key_padding_mask [batch, N] (1=real step,
    0=padding — needed on top of the paper's causal mask since we train in
    padded batches with variable N per segment, which the paper's math
    doesn't have to address for a single fixed-length trajectory).
    Output: [batch, N, d].
    """

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

    def call(self, x, key_padding_mask=None):
        N = ops.shape(x)[1]
        Q = ops.einsum("bnd,hdk->bhnk", x, self.W_Q)
        K = ops.einsum("bnd,hdk->bhnk", x, self.W_K)
        V = ops.einsum("bnd,hdk->bhnk", x, self.W_V)

        scores = ops.einsum("bhnk,bhmk->bhnm", Q, K) / ops.sqrt(float(self.d_k))  # [b,h,N,N]

        # Causal mask: query position n may only attend to key positions m<=n.
        row = ops.reshape(ops.arange(N), (N, 1))
        col = ops.reshape(ops.arange(N), (1, N))
        causal_ok = ops.cast(col <= row, "bool")  # [N,N], True = allowed

        neg = -1e9
        scores = ops.where(causal_ok, scores, neg)

        if key_padding_mask is not None:
            # block attending to PADDING key positions (independent of causal mask)
            kp = ops.cast(key_padding_mask, "bool")           # [batch,N]
            kp = ops.reshape(kp, (ops.shape(kp)[0], 1, 1, N))  # broadcast over heads & query pos
            scores = ops.where(kp, scores, neg)

        alpha = ops.softmax(scores, axis=-1)  # [b,h,N,N] — safe: finite neg, never all -inf
        out = ops.einsum("bhnm,bhmk->bhnk", alpha, V)  # [b,h,N,dk]

        out = ops.transpose(out, (0, 2, 1, 3))            # [b,N,h,dk]
        shp = ops.shape(out)
        out = ops.reshape(out, (shp[0], shp[1], self.n_heads * self.d_k))
        out = ops.matmul(out, self.W_out)                 # [b,N,d]
        return out



# [3] SHARED FEED-FORWARD (SFF)

class SharedFeedForward(keras.layers.Layer):
    """2-layer FFN with ReLU, weights shared across BOTH position (N) and
    channel (C) — applied identically to every (channel, step) vector.
    Input/output: [batch, C, N, d]."""

    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(d_ff, activation="relu")
        self.dense2 = keras.layers.Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))


class MixtureOfExpertsFeedForward(keras.layers.Layer):
    """DRAFT — not yet used by default anywhere, opt-in via CASPLayer's
    use_moe_ffn flag. Drop-in alternative to SharedFeedForward: K
    INDEPENDENT feed-forward "experts" (own weights each, same 2-layer
    ReLU structure as SharedFeedForward), SOFT-blended per step by a small
    learned gate conditioned on that step's own trajectory progression
    fraction — not hard top-1 routing (the industrial MoE default, built
    for scaling capacity across thousands of experts, needing auxiliary
    load-balancing losses to stop the gate collapsing onto one expert).
    At K=2-3, purely for regime specialization rather than raw capacity,
    soft blending sidesteps that instability entirely — closer to "a
    slightly larger effective feed-forward, smoothly combined" than to
    real distributed MoE.

    Motivation: the regime-routing experiments (train_multi_regime_models)
    showed real gains from training SEPARATE models per progression
    range, at the cost of each regime needing its own full backbone
    trained from a narrower data slice. This targets the same underlying
    idea — "behave differently depending on progression" — while keeping
    everything BEFORE the feed-forward (channel attention, causal
    self-attention) fully shared across the whole trajectory, only the
    feed-forward itself specializes. Cheaper in parameter count than
    K separate full models; the cost here is engineering risk (new
    architecture requiring its own correctness testing), not size.

    Input: x [batch, C, N, d], progression_frac [batch, N] (each step's own
    (t+1)/length — same definition used throughout this project's
    progression bands). Output: [batch, C, N, d], same shape as
    SharedFeedForward.
    """

    def __init__(self, d_model, d_ff, n_experts=2, gate_uses_content=False, content_code_dim=8,
                 n_alt_progression_signals=0, use_departure_gate=False, n_departure_subregions=None,
                 departure_embed_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.gate_uses_content = gate_uses_content
        self.n_alt_progression_signals = n_alt_progression_signals
        self.experts = [
            (keras.layers.Dense(d_ff, activation="relu"), keras.layers.Dense(d_model))
            for _ in range(n_experts)
        ]
        # Position gate: the ORIGINAL, already-proven design — progression_frac
        # alone -> softmax over n_experts. Always present, unconditionally,
        # regardless of gate_uses_content.
        self.gate_dense = keras.layers.Dense(16, activation="relu")
        self.gate_out = keras.layers.Dense(n_experts)

        # gate_uses_content=True: a SEPARATE small MLP over content, added
        # to the position gate's logits as a RESIDUAL, scaled by a learned
        # scalar initialized to EXACTLY 0. At the start of training this
        # means gate_logits = position_logits + 0*content_logits =
        # position_logits — bit-for-bit identical to the position-only
        # gate's own behavior on the very first forward pass, before any
        # training happens at all. Gradient descent can only grow this
        # scale if doing so genuinely reduces the loss; the model is never
        # forced to reconcile two competing, differently-scaled signals
        # from epoch 1 the way a straight concatenation does. This directly
        # replaces an earlier version that concatenated content into the
        # SAME gate input as progression_frac — that version measurably
        # underperformed the position-only gate, especially in later
        # trajectory bands, most likely because content's noisy,
        # not-yet-useful signal competed with an already-good position
        # signal from the very first step of training, not just early on.
        if gate_uses_content:
            self.content_proj = keras.layers.Dense(content_code_dim, activation="relu")
            self.content_norm = keras.layers.LayerNormalization()
            self.content_gate_dense = keras.layers.Dense(16, activation="relu")
            self.content_gate_out = keras.layers.Dense(n_experts)
            self.content_gate_scale = self.add_weight(
                shape=(), initializer="zeros", trainable=True, name="content_gate_scale")
        else:
            self.content_proj = None
            self.content_norm = None
            self.content_gate_dense = None
            self.content_gate_out = None
            self.content_gate_scale = None

        # n_alt_progression_signals>0: SAME residual pattern, applied to
        # one or more ADDITIONAL progression-like signals (e.g. a
        # historical-average duration estimate) instead of (or alongside)
        # the "core" progression_frac. Motivation: historical-average
        # signals were found to actively HURT accuracy in the earliest
        # bands specifically — plausibly because they were fed in
        # unconditionally, forcing the model to trust a signal that's
        # least reliable exactly where voyage duration is hardest to
        # estimate (right after departure). Each alt signal gets its OWN
        # small gate MLP and its OWN learned scale, starting at exactly 0
        # — the model can only lean on a given alt signal if doing so
        # genuinely helps, and different alt signals can end up trusted
        # to different degrees (or not at all), rather than forcing one
        # single choice of "the" progression signal.
        self.alt_prog_gate_dense = [keras.layers.Dense(16, activation="relu") for _ in range(n_alt_progression_signals)]
        self.alt_prog_gate_out = [keras.layers.Dense(n_experts) for _ in range(n_alt_progression_signals)]
        self.alt_prog_scales = [
            self.add_weight(shape=(), initializer="zeros", trainable=True, name=f"alt_prog_scale_{i}")
            for i in range(n_alt_progression_signals)
        ]

        # use_departure_gate=True: a GENUINELY DIFFERENT mechanism from
        # every residual signal above — those are each processed
        # INDEPENDENTLY then simply ADDED, so the model can weigh each
        # signal's overall importance but can never learn that one
        # signal's meaning should change DEPENDING ON another (e.g. "trust
        # early position evidence sooner, specifically for USGC
        # departures, but not for other departure regions" — an
        # interaction between progression AND departure region, not
        # something either one can express alone). This gate instead
        # embeds the departure subregion, CONCATENATES it with
        # progression_frac (not adds — concatenation lets a single small
        # MLP learn a genuinely joint function of both, rather than two
        # separate linear opinions), and only THEN produces gate logits.
        # Still wrapped in the same proven residual pattern — a learned
        # scale starting at exactly 0 — so this joint interaction can only
        # ever ADD capability on top of the already-working gate, never
        # destabilize it from step one.
        if use_departure_gate:
            if n_departure_subregions is None:
                raise ValueError("use_departure_gate=True requires n_departure_subregions")
            self.departure_embed = keras.layers.Embedding(n_departure_subregions, departure_embed_dim)
            self.joint_gate_dense = keras.layers.Dense(16, activation="relu")
            self.joint_gate_out = keras.layers.Dense(n_experts)
            self.joint_gate_scale = self.add_weight(
                shape=(), initializer="zeros", trainable=True, name="joint_gate_scale")
        else:
            self.departure_embed = None
            self.joint_gate_dense = None
            self.joint_gate_out = None
            self.joint_gate_scale = None
        self.use_departure_gate = use_departure_gate

    def call(self, x, progression_frac, alt_progression_fracs=None, departure_subregion_ids=None):
        # x: [batch, C, N, d], progression_frac: [batch, N]
        # alt_progression_fracs: optional list of [batch, N] tensors, one
        # per residual-gated alternative signal (length must match
        # n_alt_progression_signals if given).
        # departure_subregion_ids: optional [batch] int tensor, one
        # subregion id per segment (same for every step within a segment,
        # since departure region doesn't change mid-voyage) — required
        # if use_departure_gate=True.
        expert_outputs = []
        for dense1, dense2 in self.experts:
            expert_outputs.append(dense2(dense1(x)))  # each: [batch, C, N, d]
        stacked = ops.stack(expert_outputs, axis=-1)  # [batch, C, N, d, n_experts]

        gate_in_position = ops.expand_dims(progression_frac, axis=-1)   # [batch, N, 1]
        gate_logits = self.gate_out(self.gate_dense(gate_in_position))  # [batch, N, n_experts]

        if self.gate_uses_content:
            designated_channel = x[:, DESIGNATED_CHANNEL]              # [batch, N, d] -- the FRESH channel, not a stale-diluted mean
            content_code = self.content_proj(designated_channel)        # [batch, N, content_code_dim]
            content_code = self.content_norm(content_code)              # normalize scale before its own gate MLP
            content_logits = self.content_gate_out(self.content_gate_dense(content_code))  # [batch, N, n_experts]
            gate_logits = gate_logits + self.content_gate_scale * content_logits

        if self.n_alt_progression_signals > 0:
            if alt_progression_fracs is None or len(alt_progression_fracs) != self.n_alt_progression_signals:
                raise ValueError(f"n_alt_progression_signals={self.n_alt_progression_signals} requires exactly "
                                  f"that many tensors passed as alt_progression_fracs, got "
                                  f"{0 if alt_progression_fracs is None else len(alt_progression_fracs)}")
            for i, alt_frac in enumerate(alt_progression_fracs):
                alt_in = ops.expand_dims(alt_frac, axis=-1)  # [batch, N, 1]
                alt_logits = self.alt_prog_gate_out[i](self.alt_prog_gate_dense[i](alt_in))  # [batch, N, n_experts]
                gate_logits = gate_logits + self.alt_prog_scales[i] * alt_logits

        if self.use_departure_gate:
            if departure_subregion_ids is None:
                raise ValueError("use_departure_gate=True requires departure_subregion_ids to be passed to call()")
            N = ops.shape(progression_frac)[1]
            dep_embed = self.departure_embed(departure_subregion_ids)        # [batch, departure_embed_dim]
            dep_embed_bcast = ops.repeat(ops.expand_dims(dep_embed, 1), N, axis=1)  # [batch, N, departure_embed_dim]
            joint_in = ops.concatenate([gate_in_position, dep_embed_bcast], axis=-1)  # [batch, N, 1+departure_embed_dim]
            joint_logits = self.joint_gate_out(self.joint_gate_dense(joint_in))  # [batch, N, n_experts]
            gate_logits = gate_logits + self.joint_gate_scale * joint_logits

        gate_weights = ops.softmax(gate_logits, axis=-1)           # [batch, N, n_experts]
        # Broadcast to match stacked's [batch, C, N, d, n_experts] —
        # gate is the SAME across channels (C) and feature dims (d), only
        # varies by batch/step/expert.
        gate_weights = ops.reshape(gate_weights, (ops.shape(gate_weights)[0], 1, ops.shape(gate_weights)[1], 1, self.n_experts))

        blended = ops.sum(stacked * gate_weights, axis=-1)  # [batch, C, N, d]
        return blended



# [4] MSF LAYER — assembles CFA -> TSA -> MoEFF


class CASPLayer(keras.layers.Layer):
    """Input/output: x [batch, C=4, N, d]. One stacked CASP layer.

    Wiring (re-derived carefully from the paper's residual-connection
    section, since it's NOT the same as a standard Transformer block):

      designated = x[:, DESIGNATED_CHANNEL]              # [batch,N,d]
      mca_out    = CFA(x.transpose to [batch,N,C,d])      # [batch,N,d]
      z1         = LayerNorm(mca_out + designated)        # residual is
                   # against the DESIGNATED channel specifically, not the
                   # whole C-channel tensor — MCA already collapses C->1,
                   # so there's nothing else of matching shape to add to.
      msa_out    = CFA(z1, causal + padding mask)
      z2         = LayerNorm(msa_out + z1)                # standard residual,
                   # shapes match (both [batch,N,d])
      x'         = x with x[:, DESIGNATED_CHANNEL] replaced by z2 — the
                   # other C-1 channels pass through UNCHANGED at this stage
      sff_out    = SMOeFF(x')                                # applied to all
                   # C channels at every step
      x_out      = LayerNorm(sff_out + x')                # residual over
                   # the whole C-channel tensor now, shapes match
    """

    def __init__(self, d_model, n_heads_mca, n_heads_msa, d_ff, mca_gamma=2,
                 use_moe_ffn=False, n_experts=2, gate_uses_content=False, content_code_dim=8,
                 n_alt_progression_signals=0, use_departure_gate=False, n_departure_subregions=None,
                 departure_embed_dim=8, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mca = MultiheadChannelAttention(d_model, n_heads_mca, gamma=mca_gamma)
        self.msa = MaskedMultiheadSelfAttention(d_model, n_heads_msa)
        self.use_moe_ffn = use_moe_ffn
        self.sff = (MixtureOfExpertsFeedForward(d_model, d_ff, n_experts=n_experts,
                                                 gate_uses_content=gate_uses_content, content_code_dim=content_code_dim,
                                                 n_alt_progression_signals=n_alt_progression_signals,
                                                 use_departure_gate=use_departure_gate,
                                                 n_departure_subregions=n_departure_subregions,
                                                 departure_embed_dim=departure_embed_dim)
                    if use_moe_ffn else SharedFeedForward(d_model, d_ff))
        self.ln1 = keras.layers.LayerNormalization()
        self.ln2 = keras.layers.LayerNormalization()
        self.ln3 = keras.layers.LayerNormalization()
        # Standard Transformer dropout placement (Vaswani et al.): applied
        # to each sub-layer's OUTPUT, before it's added into the residual
        # stream -- NOT the same thing as gradient_dropout_weights above
        # (a deterministic loss-reweighting scheme, not a dropout layer at
        # all, despite the shared name). dropout_rate=0.0 (default): keras
        # Dropout at rate 0.0 is a no-op (confirmed directly, not
        # assumed) -- zero behavior change for every existing caller
        # unless this is explicitly set.
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dropout3 = keras.layers.Dropout(dropout_rate)

    def call(self, x, key_padding_mask=None, progression_frac=None, alt_progression_fracs=None,
              departure_subregion_ids=None, training=False):
        # x: [batch, C, N, d]
        designated = x[:, DESIGNATED_CHANNEL]              # [batch, N, d]

        x_for_mca = ops.transpose(x, (0, 2, 1, 3))          # [batch, N, C, d]
        mca_out = self.mca(x_for_mca)                       # [batch, N, d]
        z1 = self.ln1(self.dropout1(mca_out, training=training) + designated)

        msa_out = self.msa(z1, key_padding_mask=key_padding_mask)
        z2 = self.ln2(self.dropout2(msa_out, training=training) + z1)  # [batch, N, d]

        # Replace the designated channel slot with z2, leave others as-is.
        chans = ops.unstack(x, axis=1)                       # list of C tensors [batch,N,d]
        chans = list(chans)
        chans[DESIGNATED_CHANNEL] = z2
        x_replaced = ops.stack(chans, axis=1)                 # [batch, C, N, d]

        if self.use_moe_ffn:
            if progression_frac is None:
                raise ValueError("use_moe_ffn=True requires progression_frac to be passed to call() "
                                  "— WAYModel computes this automatically from key_padding_mask when "
                                  "any layer has use_moe_ffn=True.")
            sff_out = self.sff(x_replaced, progression_frac, alt_progression_fracs=alt_progression_fracs,
                                departure_subregion_ids=departure_subregion_ids)
        else:
            sff_out = self.sff(x_replaced)
        x_out = self.ln3(self.dropout3(sff_out, training=training) + x_replaced)
        return x_out



# [5] STACK OF L MSF LAYERS + PREDICTION HEAD — Eq. 12



class WAYModel(keras.layers.Layer):
    """Stacks L CASPLayers, then converts the DESIGNATED channel of the
    LAST layer's output, at EVERY step t=1..N, into a probability
    distribution over the Y port classes (many-to-many prediction — the
    same true destination label is the target at every step, per the
    paper's problem definition in Eq. 1/12).

    Input: x [batch, C=4, N, d]  (Step 3b's representation-layer output)
    Output: logits [batch, N, n_ports]  (apply softmax / cross-entropy
    outside this layer, e.g. via keras.losses.SparseCategoricalCrossentropy
    with from_logits=True, for numerical stability).
    """

    def __init__(self, d_model, n_ports, n_layers, n_heads_mca, n_heads_msa, d_ff,
                 mca_gamma=2, use_moe_ffn=False, n_experts=2, gate_uses_content=False, content_code_dim=8,
                 moe_last_layer_only=False, n_alt_progression_signals=0, use_departure_gate=False,
                 n_departure_subregions=None, departure_embed_dim=8, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_alt_progression_signals = n_alt_progression_signals
        # moe_last_layer_only=True: only the LAST CASP layer gets the MoE
        # feed-forward; every earlier layer keeps the plain shared one.
        # Tests the "late fusion" idea directly — let the shared backbone
        # (channel attention, self-attention, AND every earlier layer's
        # feed-forward) fully process the trajectory first, and only
        # specialize by progression right before the output, rather than
        # letting regime-specialization shape every layer's
        # representations from the very first one. Cheaper too — fewer
        # duplicated feed-forwards, same 1x-attention-cost savings as full
        # MoE, just even less duplicated compute.
        if use_moe_ffn and moe_last_layer_only:
            per_layer_moe = [False] * (n_layers - 1) + [True]
        else:
            per_layer_moe = [use_moe_ffn] * n_layers
        self.use_moe_ffn = any(per_layer_moe)  # controls whether progression_frac gets computed at all
        self.casp_layers = [
            CASPLayer(d_model, n_heads_mca, n_heads_msa, d_ff, mca_gamma=mca_gamma,
                      use_moe_ffn=per_layer_moe[i], n_experts=n_experts,
                      gate_uses_content=gate_uses_content, content_code_dim=content_code_dim,
                      n_alt_progression_signals=n_alt_progression_signals if per_layer_moe[i] else 0,
                      use_departure_gate=use_departure_gate if per_layer_moe[i] else False,
                      n_departure_subregions=n_departure_subregions, departure_embed_dim=departure_embed_dim,
                      dropout_rate=dropout_rate)
            for i in range(n_layers)
        ]
        self.out_proj = keras.layers.Dense(n_ports)  # W in Eq. 12; logits, not softmax

    def call(self, x, key_padding_mask=None, external_progression_frac=None, alt_progression_fracs=None,
             departure_subregion_ids=None, training=False):
        # external_progression_frac: [batch, N] optional override — if
        # given, used AS-IS instead of the internally-computed TRUE
        # progression fraction below. This is how every alternative
        # progression-signal variant (historical-average duration, raw
        # elapsed time, etc.) plugs in — compute the alternative signal
        # OUTSIDE the model (at the training-loop level, where seg_ids and
        # historical lookups are available) and pass it straight in here,
        # rather than needing WAYModel itself to know about departure
        # ports, duration indices, or any of that machinery. None
        # (default): unchanged, original TRUE-progression behavior.
        #
        # alt_progression_fracs: optional list of [batch, N] tensors, ONE
        # OR MORE additional progression-like signals, each residual-gated
        # independently inside the MoE feed-forward (see
        # MixtureOfExpertsFeedForward) — the model can learn to trust each
        # one to a different degree, or not at all, rather than committing
        # to a single choice of "the" progression signal in advance.
        #
        # departure_subregion_ids: optional [batch] int tensor — required
        # if any layer has use_departure_gate=True. Lets the gate learn a
        # JOINT function of progression AND departure region (e.g. "trust
        # position evidence sooner for USGC departures specifically"),
        # rather than the independent, additive combination every other
        # residual signal here uses.
        progression_frac = None
        if self.use_moe_ffn:
            if external_progression_frac is not None:
                progression_frac = external_progression_frac
            else:
                if key_padding_mask is None:
                    raise ValueError("use_moe_ffn=True requires key_padding_mask to be passed to "
                                      "WAYModel.call() — progression_frac is derived from it.")
                mask_f = ops.cast(key_padding_mask, "float32")               # [batch, N]
                cum_steps = ops.cumsum(mask_f, axis=1)                        # [batch, N] — running count of real steps so far
                real_length = ops.sum(mask_f, axis=1, keepdims=True)          # [batch, 1]
                real_length_safe = ops.maximum(real_length, 1.0)              # avoid div-by-zero on an all-padding row
                progression_frac = cum_steps / real_length_safe               # [batch, N], each real step's own (t+1)/length

        for layer in self.casp_layers:
            x = layer(x, key_padding_mask=key_padding_mask, progression_frac=progression_frac,
                      alt_progression_fracs=alt_progression_fracs, departure_subregion_ids=departure_subregion_ids,
                      training=training)
        designated = x[:, DESIGNATED_CHANNEL]          # [batch, N, d]
        logits = self.out_proj(designated)             # [batch, N, n_ports]
        return logits



# [6] LENGTH-BALANCED LOSS WEIGHTING
#
# In many-to-many training, longer trajectories contribute more summed
# per-step cross-entropy terms purely because they have more steps,
# regardless of how informative each step is. To compensate, every
# instance k in a mini-batch of lengths N_1..N_B receives a loss weight
#
#   delta_k = 1 + log(max(N_1..N_B) / min(N_1..N_B)) / N_k
#
# a small positive boost that is larger for shorter instances (smaller
# N_k in the denominator), calibrated by the length spread of the batch
# (the log ratio of the longest to the shortest instance).
#
# Implemented as a deterministic per-instance loss-weight multiplier.
# A stochastic variant of the same idea exists, in which delta_k instead
# calibrates a per-step random dropout of loss terms; the deterministic
# weighting is used here as a simpler, well-defined mechanism that serves
# the same purpose of balancing loss updates across instance lengths.

def gradient_dropout_weights(lengths):
    """lengths: 1D array of N_k (real step count) for each instance in a
    mini-batch. Returns delta_k, one weight per instance, to multiply that
    instance's total (summed-over-steps) many-to-many loss before the
    batch-mean is taken."""
    lengths = np.asarray(lengths, dtype="float64")
    n_max, n_min = lengths.max(), lengths.min()
    if n_min <= 0:
        n_min = 1.0
    ratio = np.log(n_max / n_min) if n_max > n_min else 0.0
    delta = 1.0 + ratio / lengths
    return delta.astype("float32")


def restrict_mask_to_progression(step_mask, max_progression_frac=None, min_progression_frac=None):
    """Restricts step_mask to a PROGRESSION RANGE, relative to EACH
    example's own real trajectory length — same "relative to its own
    voyage" convention used throughout this project's progression bands
    (frac = (t+1)/length, the exact definition evaluate_quartile_accuracy
    also uses, so the two stay consistent).

    max_progression_frac (upper bound, e.g. 0.25 -> keep steps with
    frac <= 0.25, an "early specialist"): count of kept steps is
    floor(max_progression_frac * length) — NOT ceil, which would
    over-include by up to one step whenever max_progression_frac * length
    isn't a whole number. Always keeps at least 1 step even for very short
    trajectories.

    min_progression_frac (lower bound, e.g. 0.25 -> keep steps with
    frac > 0.25, a genuine "late specialist" — the complement of the
    max_progression_frac=0.25 case above, EXACTLY: no gap, no overlap,
    verified directly). Always keeps at least the LAST step even for very
    short trajectories where every step's own fraction falls at or below
    the threshold.

    Both together (e.g. min=0.25, max=0.75) keep a middle band. Combines
    with (does not replace) the original padding mask — padding stays
    excluded regardless.
    """
    mask_f = ops.cast(step_mask, "float32")
    real_length = ops.sum(mask_f, axis=1, keepdims=True)  # [batch, 1]
    N = ops.shape(step_mask)[1]
    position_index = ops.arange(N, dtype="float32")
    position_index = ops.reshape(position_index, (1, N))  # broadcasts against [batch,1]

    keep = ops.ones_like(mask_f)

    if max_progression_frac is not None:
        # keep t where (t+1)/length <= max_progression_frac
        # <=> count of kept positions = floor(max_progression_frac * length)
        cutoff = ops.floor(max_progression_frac * real_length)
        cutoff = ops.maximum(cutoff, 1.0)  # always keep at least the FIRST step
        keep = keep * ops.cast(position_index < cutoff, "float32")

    if min_progression_frac is not None:
        # keep t where (t+1)/length > min_progression_frac
        # <=> t >= floor(min_progression_frac * length)  (see docstring derivation)
        floor_cutoff = ops.floor(min_progression_frac * real_length)
        floor_cutoff = ops.minimum(floor_cutoff, real_length - 1.0)  # always keep at least the LAST step
        keep = keep * ops.cast(position_index >= floor_cutoff, "float32")

    return mask_f * keep


def way_loss(logits, labels, step_mask, gd_weights=None, max_progression_frac=None, min_progression_frac=None):
       """Many-to-many cross-entropy loss over trajectory steps.

    The same label is compared against the prediction at every real
    (non-padded) step, summed per instance, optionally reweighted per
    instance, then averaged over the batch.

    Args:
      logits: [batch, N, n_ports] per-step predictions.
      labels: [batch] true destination port id, identical target at every
        step of an instance.
      step_mask: [batch, N], 1 for real steps, 0 for padding.
      gd_weights: optional [batch] per-instance loss-weight multipliers
        (see the length-balanced loss weighting note above).
      max_progression_frac / min_progression_frac: optional, restrict
        which steps are scored to a progression range within each
        example's own trajectory (see restrict_mask_to_progression).
        max_progression_frac=0.25 alone scores only the first 25% of each
        trajectory (early specialist); min_progression_frac=0.25 alone
        scores everything after 25% (late specialist, the exact
        complement, no gap or overlap); both together score a middle
        band; both None (default) scores all steps.
        This restriction changes only which steps are scored and
        backpropagated, not what the model computes or sees in the
        forward pass; causal self-attention already prevents later steps
        from influencing earlier ones.
    """
    batch, N, n_ports = ops.shape(logits)[0], ops.shape(logits)[1], ops.shape(logits)[2]
    labels_b = ops.reshape(labels, (batch, 1))
    labels_b = ops.repeat(labels_b, N, axis=1)  # [batch,N] — same label every step

    ce = keras.losses.sparse_categorical_crossentropy(
        labels_b, logits, from_logits=True
    )  # [batch, N]

    restrict = max_progression_frac is not None or min_progression_frac is not None
    mask_f = (restrict_mask_to_progression(step_mask, max_progression_frac, min_progression_frac)
              if restrict else ops.cast(step_mask, "float32"))
    ce_masked = ce * mask_f
    per_instance_sum = ops.sum(ce_masked, axis=1)  # [batch] — many-to-many summed loss

    if gd_weights is not None:
        per_instance_sum = per_instance_sum * ops.convert_to_tensor(gd_weights)

    return ops.mean(per_instance_sum)
