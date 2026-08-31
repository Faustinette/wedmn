# =============================================================================
# E12-A — pooled predictions with probabilities (prereq for stats / E15)
# Migrated verbatim from Main_forGitHub.ipynb cells [86].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 86]
# ----------------------------------------------------------------------
# PRE-REQUISITS ===== E12-A -- pooled predictions with probabilities (prereq for E13/E15) =====
import numpy as np, pandas as pd
LATE = 0.95
_S,_T,_P,_PR,_F = [],[],[],[],[]
for s_ in SEEDS:
    r_ = runs[s_]
    a,b,c,d,e = _collect_full_predictions(r_["model"], r_["repr_layer"],
        r_["val_loader"], r_["core_and_alt_fn"],
        departure_ids_fn=r_.get("departure_ids_fn"),
        eta_channel_lookup=r_.get("eta_channel_lookup"))
    _S.append(a);_T.append(b);_P.append(c);_PR.append(d);_F.append(e)
seg, true, pred, probs, frac = map(np.concatenate, (_S,_T,_P,_PR,_F))
print(f"pooled: {len(seg):,} rows across {len(SEEDS)} seeds")# ===== E12-A -- pooled predictions with probabilities (prereq for E13/E15) =====
import numpy as np, pandas as pd
LATE = 0.95
_S,_T,_P,_PR,_F = [],[],[],[],[]
for s_ in SEEDS:
    r_ = runs[s_]
    a,b,c,d,e = _collect_full_predictions(r_["model"], r_["repr_layer"],
        r_["val_loader"], r_["core_and_alt_fn"],
        departure_ids_fn=r_.get("departure_ids_fn"),
        eta_channel_lookup=r_.get("eta_channel_lookup"))
    _S.append(a);_T.append(b);_P.append(c);_PR.append(d);_F.append(e)
seg, true, pred, probs, frac = map(np.concatenate, (_S,_T,_P,_PR,_F))
print(f"pooled: {len(seg):,} rows across {len(SEEDS)} seeds")
