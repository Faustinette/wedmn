# =============================================================================
# E7 — mixture-expert drivers
# Migrated verbatim from Main_forGitHub.ipynb cells [179, 180, 181].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 179]
# ----------------------------------------------------------------------
# =============================================================================
# E7-CHANNELS -- channel reliance per gate regime (occlusion x dominant expert)
# =============================================================================
# Seed 42, validation. Two-part pass per batch: (1) baseline forward with the
# E7 gate wrapper capturing layer-1 mixture weights -> per-step dominant
# expert; (2) one forward per occluded channel (x[:, c] zeroed) -> damage.
# Damage is then stratified by the BASELINE dominant expert. Indirect by
# design (experts consume fused features); permutation caveat applies:
# expert columns are this fit's labels, only the PATTERN generalises.
import numpy as np, pandas as pd, torch
from keras import ops as _ops
from IPython.display import display, HTML

E7C_SEED = 42
r_ = runs[E7C_SEED]
_vl = r_["val_loader"]
N_BATCH = min(80, len(_vl))

# channel names: VERIFY against the repr-layer construction order for lean2.
CHANNEL_NAMES = None      # None -> auto "ch0..chN"; else list matching x.shape[1]

# gate capture on layer 1 (index 0): reuse the E7 wrapper mechanics
_layer = r_["model"].casp_layers[0]
_holder = {}
_orig_call = _layer.sff.call
def _wrapped(x, progression_frac, alt_progression_fracs=None,
             departure_subregion_ids=None):
    out = _orig_call(x, progression_frac,
                     alt_progression_fracs=alt_progression_fracs,
                     departure_subregion_ids=departure_subregion_ids)
    gate_in = _ops.expand_dims(progression_frac, axis=-1)
    total = _layer.sff.gate_out(_layer.sff.gate_dense(gate_in))
    if _layer.sff.n_alt_progression_signals > 0:
        for i, af in enumerate(alt_progression_fracs):
            total = total + _layer.sff.alt_prog_scales[i] * \
                _layer.sff.alt_prog_gate_out[i](
                    _layer.sff.alt_prog_gate_dense[i](_ops.expand_dims(af, -1)))
    if _layer.sff.gate_uses_content:
        cc = _layer.sff.content_norm(_layer.sff.content_proj(x[:, DESIGNATED_CHANNEL]))
        total = total + _layer.sff.content_gate_scale * \
            _layer.sff.content_gate_out(_layer.sff.content_gate_dense(cc))
    _holder["dom"] = _ops.convert_to_numpy(_ops.argmax(total, axis=-1))
    return out

DOM, OK0, OKC = [], [], {}
_layer.sff.call = _wrapped
try:
    for bi in range(N_BATCH):
        bs = _vl.batches[bi]
        inputs, n_mask, labels, _ln = _vl[bi]
        if r_.get("eta_channel_lookup") is not None:
            inputs["eta_channel_values"] = _ops.convert_to_tensor(
                eta_progression_for_batch(r_["eta_channel_lookup"], bs,
                                          n_steps=inputs["tau"].shape[1]))
        core, alts = r_["core_and_alt_fn"](inputs, bs)
        dep = (r_["departure_ids_fn"](bs)
               if r_.get("departure_ids_fn") is not None else None)
        with torch.no_grad():
            x = r_["repr_layer"](inputs)
            n_ch = x.shape[1]
            if CHANNEL_NAMES is None:
                CHANNEL_NAMES = [f"ch{c}" for c in range(n_ch)]
            logits = r_["model"](x, key_padding_mask=n_mask,
                                 external_progression_frac=core,
                                 alt_progression_fracs=alts,
                                 departure_subregion_ids=dep)
        mk = _ops.convert_to_numpy(n_mask).astype(bool)
        lb = _ops.convert_to_numpy(labels)
        if lb.ndim == 1: lb = np.repeat(lb[:, None], x.shape[2], axis=1)
        pr = _ops.convert_to_numpy(_ops.argmax(logits, axis=-1))
        DOM.append(_holder["dom"][mk]); OK0.append((pr == lb)[mk])
        for c in range(n_ch):                       # occlusion passes
            xz = torch.clone(x); xz[:, c] = 0.0
            with torch.no_grad():
                lz = r_["model"](xz, key_padding_mask=n_mask,
                                 external_progression_frac=core,
                                 alt_progression_fracs=alts,
                                 departure_subregion_ids=dep)
            pz = _ops.convert_to_numpy(_ops.argmax(lz, axis=-1))
            OKC.setdefault(c, []).append((pz == lb)[mk])
finally:
    _layer.sff.call = _orig_call

dom = np.concatenate(DOM); ok0 = np.concatenate(OK0)
rows = []
for c in sorted(OKC):
    okc = np.concatenate(OKC[c])
    row = {"channel": CHANNEL_NAMES[c],
           "overall damage (pp)": round(100 * (ok0.mean() - okc.mean()), 2)}
    for k in sorted(set(dom)):
        m = dom == k
        row[f"expert {k+1} (n={m.sum():,})"] = \
            round(100 * (ok0[m].mean() - okc[m].mean()), 2)
    rows.append(row)
e7c = pd.DataFrame(rows)
display(HTML(f"<b>E7-CHANNELS — occlusion damage by layer-1 dominant expert "
             f"(seed {E7C_SEED}, {len(dom):,} steps)</b>"))
display(e7c)
e7c.to_csv(os.path.join(WORK_DIR, "e7_channel_by_expert.csv"), index=False)
print("note: verify CHANNEL_NAMES ordering against the repr-layer stack "
      "before quoting channels by name in the report.")

# ----------------------------------------------------------------------
# [notebook cell 180]
# ----------------------------------------------------------------------
# TEMP E7 dependencies (run ealrier supositely)
# =============================================================================
# E7-LEAD -- per-step MSF-1 gate weights on TEST, pooled over seeds
# =============================================================================
# Produces: gw1 (n_steps, K) and lead_expert (n_steps,), aligned to tseg/tfrac.
import numpy as np, torch

GW, SEG = [], []
for s_ in SEEDS:
    r_ = final_runs[s_]
    model, repr_layer = r_["model"], r_["repr_layer"]
    store = {}
    # MSF block 1's MoEFF gating network -- adjust the attribute path if your
    # module names differ (print(model) once to confirm).
    target = model.blocks[0].moeff.gate
    h = target.register_forward_hook(
        lambda m, i, o: store.setdefault("w", []).append(
            torch.softmax(o.detach(), dim=-1).float().cpu().numpy()))
    t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
        batch_size=BATCH_SIZE, seg_id_subset=r_["_test_ids"],
        shuffle=False, seed=0, include_ship_history=True)
    a, b, c, d, e = _collect_full_predictions(model, repr_layer, t_loader,
        r_["core_and_alt_fn"], departure_ids_fn=r_.get("departure_ids_fn"),
        eta_channel_lookup=r_.get("eta_channel_lookup"))
    h.remove()
    W = np.concatenate([w.reshape(-1, w.shape[-1]) for w in store["w"]], axis=0)
    GW.append(W); SEG.append(a)
    print(f"seed {s_}: gate weights {W.shape}, steps {len(a)}")

gw1 = np.concatenate(GW, axis=0)
seg_chk = np.concatenate(SEG)
assert len(gw1) == len(tseg), f"{len(gw1)} vs {len(tseg)} -- masking mismatch"
assert (seg_chk == tseg).all(), "row alignment differs from E17 arrays"
lead_expert = gw1.argmax(1)
print("lead-expert shares:", np.round(np.bincount(lead_expert) / len(lead_expert), 3))

# ----------------------------------------------------------------------
# [notebook cell 181]
# ----------------------------------------------------------------------
# E7-DISENTANGLE -- expert effect within progression band
import numpy as np, pandas as pd
band = pd.cut(tfrac, [0,.2,.6,1.0], labels=["Early","Mid","Late"])
rows=[]
for b in ["Early","Mid","Late"]:
    for e in np.unique(lead_expert):                 # argmax gate, MSF 1
        m = (band==b).values & (lead_expert==e)
        if m.sum() < 300: continue
        for ch in ["No ship history","No dep port"]:
            rows.append(dict(band=b, expert=int(e), channel=ch, n=int(m.sum()),
                damage_pp=round(100*(t_ok[m].mean()-ABL_PRED[ch][m].mean()),2)))
print(pd.DataFrame(rows).pivot_table(index=["band","expert","n"],
      columns="channel", values="damage_pp").to_string())
