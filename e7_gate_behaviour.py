# =============================================================================
# E7 — gate behaviour over the voyage (+ E4 gate-input contrasts table)
# Migrated verbatim from Main_forGitHub.ipynb cells [172, 173, 174, 175, 176, 177].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 172]
# ----------------------------------------------------------------------
# =============================================================================
# E7-COLLECT -- gate decomposition: per CASP layer, per expert, PER DRIVER
# =============================================================================
# Extends plot_moe_gate_weights_full's own wrapper mechanism (same forward
# pass, same true-progression binning): instead of only the final softmax
# weights w_t, it captures each gate INPUT DRIVER's logit contribution
# separately -- core G(0.5), each alt signal beta_i*G_i(p_i) (order =
# ALT_PROGRESSION_MODES: eta, historical_avg_port), and the content code
# beta_c*G_c(c_t) -- per CASP layer, per expert, binned by true progression.
import numpy as np, torch
from keras import ops

def collect_gate_driver_decomposition(model, repr_layer, val_loader, core_and_alt_fn,
                                      departure_ids_fn=None, eta_channel_lookup=None,
                                      boundaries=DEFAULT_PROGRESSION_BOUNDARIES,
                                      max_batches=None):
    n_bands = len(boundaries)
    moe_layers = [(i, l) for i, l in enumerate(model.casp_layers)
                  if getattr(l.sff, "n_experts", 1) > 1]
    driver_names = None
    sums, counts, wsums = {}, {}, {}

    def _make_wrapper(sff, holder):
        orig_call = sff.call
        def wrapped_call(x, progression_frac, alt_progression_fracs=None, departure_subregion_ids=None):
            result = orig_call(x, progression_frac, alt_progression_fracs=alt_progression_fracs,
                               departure_subregion_ids=departure_subregion_ids)
            comps = {}
            gate_in = ops.expand_dims(progression_frac, axis=-1)
            comps["core G(0.5)"] = sff.gate_out(sff.gate_dense(gate_in))
            if sff.n_alt_progression_signals > 0:
                for i, alt_frac in enumerate(alt_progression_fracs):
                    name = (ALT_PROGRESSION_MODES[i] if i < len(ALT_PROGRESSION_MODES)
                            else f"alt[{i}]")
                    alt_in = ops.expand_dims(alt_frac, axis=-1)
                    comps[f"{name} (beta={float(ops.convert_to_numpy(sff.alt_prog_scales[i])):.3f})"] = \
                        sff.alt_prog_scales[i] * sff.alt_prog_gate_out[i](sff.alt_prog_gate_dense[i](alt_in))
            if sff.gate_uses_content:
                designated = x[:, DESIGNATED_CHANNEL]
                content_code = sff.content_norm(sff.content_proj(designated))
                comps[f"content (beta={float(ops.convert_to_numpy(sff.content_gate_scale)):.3f})"] = \
                    sff.content_gate_scale * sff.content_gate_out(sff.content_gate_dense(content_code))
            total = None
            for v in comps.values():
                total = v if total is None else total + v
            holder["comps"] = {k: ops.convert_to_numpy(v) for k, v in comps.items()}
            holder["w"] = ops.convert_to_numpy(ops.softmax(total, axis=-1))
            return result
        return wrapped_call, orig_call

    captured = {i: {} for i, _ in moe_layers}
    originals = []
    for i, layer in moe_layers:
        wrapped, orig = _make_wrapper(layer.sff, captured[i])
        layer.sff.call = wrapped
        originals.append((layer, orig))
    try:
        n_b = len(val_loader) if max_batches is None else min(max_batches, len(val_loader))
        for bi in range(n_b):
            batch_seg_ids = val_loader.batches[bi]
            inputs, n_mask, _labels, _lengths = val_loader[bi]
            if eta_channel_lookup is not None:
                inputs["eta_channel_values"] = ops.convert_to_tensor(
                    eta_progression_for_batch(eta_channel_lookup, batch_seg_ids,
                                              n_steps=inputs["tau"].shape[1]))
            core, alts = core_and_alt_fn(inputs, batch_seg_ids)
            dep_ids = departure_ids_fn(batch_seg_ids) if departure_ids_fn is not None else None
            with torch.no_grad():
                x = repr_layer(inputs)
                _ = model(x, key_padding_mask=n_mask, external_progression_frac=core,
                          alt_progression_fracs=alts, departure_subregion_ids=dep_ids)
            mask_np = ops.convert_to_numpy(n_mask).astype(bool)
            mask_f = mask_np.astype("float32")
            core_np = np.cumsum(mask_f, axis=1) / np.maximum(mask_f.sum(axis=1, keepdims=True), 1.0)
            band = np.clip(np.digitize(core_np, boundaries), 0, n_bands - 1)
            for li, _ in moe_layers:
                comps, w = captured[li]["comps"], captured[li]["w"]
                if driver_names is None: driver_names = list(comps)
                K = w.shape[-1]
                if li not in sums:
                    sums[li] = {d: np.zeros((n_bands, K)) for d in comps}
                    wsums[li] = np.zeros((n_bands, K)); counts[li] = np.zeros(n_bands)
                m = mask_np
                np.add.at(counts[li], band[m], 1.0)
                np.add.at(wsums[li], band[m], w[m])
                for d, v in comps.items():
                    np.add.at(sums[li][d], band[m], v[m])
    finally:
        for layer, orig in originals:
            layer.sff.call = orig
    out = {}
    for li in sums:
        c = np.maximum(counts[li], 1.0)[:, None]
        out[li] = {"w_mean": wsums[li] / c,
                   "drivers": {d: sums[li][d] / c for d in sums[li]},
                   "counts": counts[li]}
    return out, driver_names

print("E7 collector defined")

# ----------------------------------------------------------------------
# [notebook cell 173]
# ----------------------------------------------------------------------
# =============================================================================
# E7-RUN -- collect on the main model (K = 3), one seed
# =============================================================================
E7_SEED = 123                      # collection seed (one is representative)
_r = runs[E7_SEED] if "runs" in dir() and E7_SEED in runs else None
assert _r is not None, "runs[E7_SEED] missing -- run E0 A first (or reload via CELL 6-RELOAD)"
E7_DECOMP, E7_DRIVERS = collect_gate_driver_decomposition(
    _r["model"], _r["repr_layer"], _r["val_loader"], _r["core_and_alt_fn"],
    departure_ids_fn=_r.get("departure_ids_fn"),
    eta_channel_lookup=_r.get("eta_channel_lookup"))
print(f"collected: layers {sorted(E7_DECOMP)}  drivers: {E7_DRIVERS}")

# ----------------------------------------------------------------------
# [notebook cell 174]
# ----------------------------------------------------------------------
# =============================================================================
# E7-PLOT-A -- gate weights w_t per MSA layer (the standard view)
# =============================================================================
import matplotlib.pyplot as plt
_bounds = list(DEFAULT_PROGRESSION_BOUNDARIES)
_x = [b for b in _bounds]
fig, axes = plt.subplots(1, len(E7_DECOMP), figsize=(6.2 * len(E7_DECOMP), 4.2), squeeze=False)
for ax, li in zip(axes[0], sorted(E7_DECOMP)):
    W = E7_DECOMP[li]["w_mean"]
    for k in range(W.shape[1]):
        ax.plot(_x, W[:, k], marker="o", ms=3, label=f"expert {k+1}")
    ax.set_title(f"MSF layer {li+1} -- mean gate weight w_t")
    ax.set_xlabel("trajectory progression"); ax.set_ylabel("mean weight")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, f"e7_gate_weights_seed{E7_SEED}.png"), dpi=150)
plt.show()

# ----------------------------------------------------------------------
# [notebook cell 175]
# ----------------------------------------------------------------------
# =============================================================================
# E7-PLOT-B (fixed) -- per-driver panels, per-layer beta-suffixed keys resolved
# =============================================================================
def _prefix(d): return d.split(" (beta")[0]
_canon = [_prefix(d) for d in E7_DECOMP[sorted(E7_DECOMP)[0]]["drivers"]]

n_layers = len(E7_DECOMP); n_drv = len(_canon)
fig, axes = plt.subplots(n_layers, n_drv, figsize=(4.6 * n_drv, 3.6 * n_layers),
                         squeeze=False)
for r_i, li in enumerate(sorted(E7_DECOMP)):
    layer_keys = {_prefix(d): d for d in E7_DECOMP[li]["drivers"]}
    for c_i, pfx in enumerate(_canon):
        ax = axes[r_i][c_i]
        key = layer_keys.get(pfx)
        if key is None:
            ax.set_axis_off(); continue
        V = E7_DECOMP[li]["drivers"][key]
        for k in range(V.shape[1]):
            ax.plot(_x, V[:, k], marker="o", ms=2.5, label=f"expert {k+1}")
        ax.set_title(f"L{li+1} -- {key}", fontsize=9)     # title keeps the beta
        ax.set_xlabel("progression"); ax.set_ylabel("logit contribution")
        ax.axhline(0, color="grey", lw=0.6)
        if r_i == 0 and c_i == 0: ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,
            f"e7_gate_driver_decomposition_seed{E7_SEED}.png"), dpi=150)
plt.show()
print("Saved E7 driver-decomposition figure")

# ----------------------------------------------------------------------
# [notebook cell 176]
# ----------------------------------------------------------------------
# =============================================================================
# E4 -- gate-input ablation: alt progression signals {both | eta | hist | none}
# =============================================================================
sweep_arch = dict(
    d_model=D_MODEL, n_experts=N_EXPERTS,
    gate_ship_history=True, use_ship_history=True,
    use_spatial_channel=True, use_local_pattern_channel=True,
    use_departure_port_channel=True, use_temporal_encoding=True,
    use_ship_size_channel=False, use_departure_gate=False,
    alt_progression_modes=ALT_PROGRESSION_MODES,
    stratify=True, val_frac=0.15,
    test_start=TEST_START, test_end=TEST_END,
    batch_size=BATCH_SIZE, work_dir=WORK_DIR,
)
print({k: v for k, v in sweep_arch.items() if k != "work_dir"})

E4_ARMS = {"both (main)": ALT_PROGRESSION_MODES,          # reloads baseline
           "eta only": ["eta"],
           "hist only": ["historical_avg_port"],
           "none": []}
E4 = {}
for label, modes in E4_ARMS.items():
    E4[label] = {}
    for seed in SEEDS:
        cond = (f"final_main_lean2_seed{seed}" if label.startswith("both")
                else f"e4_gate_{label.split(' ')[0]}_final_main_seed{seed}")
        kw = (dict(epochs=EPOCHS, early_stopping_patience=PATIENCE)
              if label.startswith("both")
              else dict(epochs=int(BEST_EPOCHS[seed]), early_stopping_patience=None))
        r = train_residual_progression_variant(
            data, TARGET_COL, N_CLASSES, condition_name=cond, seed=seed,
            alt_progression_modes=modes, skip_existing=True,
            **{k: v for k, v in sweep_arch.items()
               if k != "alt_progression_modes"}, **kw)
        E4[label][seed] = _test_result(r, seed)
        print(f"  {cond}: TEST {E4[label][seed]['overall_acc']:.3f}")

# ----------------------------------------------------------------------
# [notebook cell 177]
# ----------------------------------------------------------------------
# =============================================================================
# E4-RESULTS -- one consolidated table: arms + the four informative contrasts
# =============================================================================
import numpy as np, pandas as pd
from scipy import stats as _st
from IPython.display import display, HTML

_bounds = np.array(sorted(DEFAULT_PROGRESSION_BOUNDARIES))
_early = _bounds <= 0.20
def _ea(d):
    bc = np.array(d["band_correct"], float); bt = np.array(d["band_total"], float)
    return bc[_early].sum() / max(bt[_early].sum(), 1)

# ---- arm summary ------------------------------------------------------------


arms = []
for label in E4:
    ov = [100*E4[label][s]["overall_acc"] for s in SEEDS]
    eb = [100*_ea(E4[label][s]) for s in SEEDS]
    arms.append(dict(arm=label,
                     overall=f"{np.mean(ov):.2f} \u00b1 {np.std(ov):.2f}",
                     early=f"{np.mean(eb):.2f} \u00b1 {np.std(eb):.2f}"))
display(HTML("<b>E4 — gate timing-signal ablation (TEST, 3 seeds)</b>"))
display(pd.DataFrame(arms))

# ---- paired contrasts -------------------------------------------------------
def _delta(A, B, f):
    return np.array([f(E4[A][s]) - f(E4[B][s]) for s in SEEDS]) * 100
CONTRASTS = [
    ("both vs eta-only",  "history's ADDED value given ETA"),
    ("both vs hist-only", "ETA's ADDED value given history"),
    ("eta-only vs none",  "ETA standalone"),
    ("hist-only vs none", "history standalone"),
    ("both vs none",      "joint value of both signals"),
]
_key = {a.split(" vs ")[0]: a.split(" vs ")[0] for a, _ in CONTRASTS}
def _arm(name):                     # tolerant: "eta-only" -> "eta only"
    probe = name.replace("-", " ").strip().lower()
    for lab in E4:
        if lab.lower().startswith(probe.split()[0]):
            return lab
    raise KeyError(f"{name} not in {list(E4)}")
rows = []
for pair, meaning in CONTRASTS:
    a, b = pair.split(" vs ")
    for metric, f in [("overall", lambda d: d["overall_acc"]), ("early", _ea)]:
        d = _delta(_arm(a), _arm(b), f)
        t, p = _st.ttest_1samp(d, 0.0)
        rows.append(dict(contrast=pair, tests=meaning, metric=metric,
                         delta_pp=round(d.mean(), 2),
                         per_seed=[round(x, 2) for x in d],
                         t=round(float(t), 2), p=round(float(p), 3),
                         sign_consistent=bool((d > 0).all() or (d < 0).all())))
e4t = pd.DataFrame(rows)
display(HTML("<b>E4 — paired contrasts (same-seed pairing)</b>"))
display(e4t)
e4t.to_csv(os.path.join(WORK_DIR, "e4_gate_input_contrasts.csv"), index=False)
