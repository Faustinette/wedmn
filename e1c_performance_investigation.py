# E1-C — Performance investigation
# Migrated verbatim from Main_forGitHub.ipynb cells [80, 81, 82, 83, 84].
# Executed by runner.py inside the shared namespace (notebook-kernel style).


# E1-C: SHORT-VERSUS-LONG VOYAGE GAP, DIAGNOSTIC INVESTIGATION
#
# Five analyses, in priority order, all on pooled TEST predictions, asking
# why early-voyage accuracy differs between short and long voyages:
#
#   PREREQ  Pooled per-step TEST arrays across seeds + per-voyage
#           durations; short/long split at 14 days.
#   A1      Re-bin accuracy by ABSOLUTE elapsed days instead of
#           progression fraction. If short and long curves coincide, the
#           early-band gap is a fraction-binning artifact.
#   A4      Declaration staleness: captain accuracy versus elapsed days.
#           Tests whether departure declarations aging on long voyages
#           explains the captain's mirror-image pattern.
#   A2      Kitagawa/Oaxaca decomposition of the early-band gap into
#           class-mix versus within-class performance.
#   A3      Per-group empirical ceiling (Bayes-rate estimate given grid
#           cell and departure basin): do the groups differ in problem
#           hardness rather than model quality?
#   A5      Formal wrapper: step-level logistic regression of correctness
#           on short/long with elapsed-time and class controls,
#           segment-clustered standard errors.
#
# ROLE IN THE REPORT: optional / appendix. None of these produce headline
# numbers; they exist to justify the INTERPRETATION of the E1/E1-B
# duration gap (artifact versus staleness versus composition versus
# hardness). If the report states a cause for the gap, this file is the
# evidence; otherwise it can be summarized in one or two sentences.
#
# Prerequisites from E1: final_runs and the loaded test-side models.




# [notebook cell 80]

# E17-PREREQ -- TEST-side pooled arrays + per-voyage durations

# All five analyses run on TEST predictions (where the short/long gap lives).
import numpy as np, pandas as pd
_S,_T,_P,_PR,_F = [],[],[],[],[]
for s_ in SEEDS:
    r_ = final_runs[s_]
    t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
        batch_size=BATCH_SIZE, seg_id_subset=r_["_test_ids"],
        shuffle=False, seed=0, include_ship_history=True)
    a,b,c,d,e = _collect_full_predictions(r_["model"], r_["repr_layer"],
        t_loader, r_["core_and_alt_fn"],
        departure_ids_fn=r_.get("departure_ids_fn"),
        eta_channel_lookup=r_.get("eta_channel_lookup"))
    _S.append(a);_T.append(b);_P.append(c);_PR.append(d);_F.append(e)
tseg, ttrue, tpred, tprobs, tfrac = map(np.concatenate,(_S,_T,_P,_PR,_F))
t_ok = tpred == ttrue
_tj = data.traj_idx.set_index("seg_id")
DUR = {int(s): max((pd.Timestamp(_tj.loc[int(s),"arr_ts"])
        - pd.Timestamp(_tj.loc[int(s),"dep_ts"])).days, 1)
       for s in np.unique(tseg)}
tdur = np.array([DUR[int(s)] for s in tseg], dtype="float64")
tdays = tfrac * tdur                       # absolute elapsed days (frac x dur)
THR = 14
short_m, long_m = tdur <= THR, tdur > THR
print(f"TEST pooled: {len(tseg):,} rows; short {short_m.sum():,} / long {long_m.sum():,} steps")


# A1+A4 POOLED v2 -- model & declaration accuracy vs elapsed days, by length

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODEL_COL, CAPTAIN_COL = "#2ecc71", "#3498db"          # house benchmark colors

fig, ax = plt.subplots(figsize=(11, 5.5))
for lab, ls in [("<= 14d", "-"), ("> 14d", "--")]:
    g1 = a1[a1["group"] == lab]
    ax.plot(g1["mid"], g1["acc"], ls, marker="o", color=MODEL_COL, lw=2.2)
    g4 = a4[a4["group"] == lab]
    ax.plot(g4["mid"], g4["cap_acc"], ls, marker="s", color=CAPTAIN_COL, lw=2.2)

ax.set_xlabel("Days since departure")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model vs declared destination by ABSOLUTE elapsed time -- "
             "TEST, short (solid) vs long (dashed) voyages")

# two-part legend: colour = series, linestyle = voyage length
leg1 = ax.legend(handles=[
    Line2D([], [], color=MODEL_COL, lw=2.5, marker="o", label="Model"),
    Line2D([], [], color=CAPTAIN_COL, lw=2.5, marker="s",
           label="Captain declaration")],
    loc="lower right", fontsize=11, title="Series", title_fontsize=10)
ax.add_artist(leg1)
ax.legend(handles=[
    Line2D([], [], color="#444444", lw=2.5, ls="-", label="$\\leq$ 14 days"),
    Line2D([], [], color="#444444", lw=2.5, ls="--", label="> 14 days")],
    loc="center right", fontsize=11, title="Voyage length", title_fontsize=10,
    handlelength=3.5)

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "a1a4_pooled_model_captain_days.png"), dpi=150)
plt.show()

# [notebook cell 81]

# E17-PREREQ -- TEST-side pooled arrays + per-voyage durations

# All five analyses run on TEST predictions (where the short/long gap lives).
import numpy as np, pandas as pd
_S,_T,_P,_PR,_F = [],[],[],[],[]
for s_ in SEEDS:
    r_ = final_runs[s_]
    t_loader = BucketedWAYDataset(data, target_col=TARGET_COL,
        batch_size=BATCH_SIZE, seg_id_subset=r_["_test_ids"],
        shuffle=False, seed=0, include_ship_history=True)
    a,b,c,d,e = _collect_full_predictions(r_["model"], r_["repr_layer"],
        t_loader, r_["core_and_alt_fn"],
        departure_ids_fn=r_.get("departure_ids_fn"),
        eta_channel_lookup=r_.get("eta_channel_lookup"))
    _S.append(a);_T.append(b);_P.append(c);_PR.append(d);_F.append(e)
tseg, ttrue, tpred, tprobs, tfrac = map(np.concatenate,(_S,_T,_P,_PR,_F))
t_ok = tpred == ttrue
_tj = data.traj_idx.set_index("seg_id")
DUR = {int(s): max((pd.Timestamp(_tj.loc[int(s),"arr_ts"])
        - pd.Timestamp(_tj.loc[int(s),"dep_ts"])).days, 1)
       for s in np.unique(tseg)}
tdur = np.array([DUR[int(s)] for s in tseg], dtype="float64")
tdays = tfrac * tdur                       # absolute elapsed days (frac x dur)
THR = 14
short_m, long_m = tdur <= THR, tdur > THR
print(f"TEST pooled: {len(tseg):,} rows; short {short_m.sum():,} / long {long_m.sum():,} steps")



# A1 (priority 1) -- re-bin by ABSOLUTE elapsed days: artifact test

# If short/long model curves coincide vs elapsed DAYS, the early-band gap is
# a fraction-binning artifact (unequal calendar information per band).
import matplotlib.pyplot as plt
day_edges = [0,1,2,3,4,5,7,10,14,21,30,60]
rows = []
for lab, m in [("<= 14d", short_m), ("> 14d", long_m)]:
    bi = np.digitize(tdays[m], day_edges) - 1
    for b in range(len(day_edges)-1):
        mm = bi == b
        if mm.sum() < 200: continue
        rows.append(dict(group=lab, day_bin=f"{day_edges[b]}-{day_edges[b+1]}d",
                         mid=(day_edges[b]+day_edges[b+1])/2,
                         acc=100*t_ok[m][mm].mean(), n=int(mm.sum())))
a1 = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9,4.5))
for lab, g in a1.groupby("group"):
    ax.plot(g["mid"], g["acc"], "o-", label=lab)
ax.set_xlabel("days since departure"); ax.set_ylabel("model accuracy (%)")
ax.set_title("A1 -- model accuracy vs ABSOLUTE elapsed time, short vs long")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,"a1_daybinned_accuracy.png"), dpi=150); plt.show()
print(a1.pivot_table(index="day_bin", columns="group", values="acc").round(1))
print("\nreading: coinciding curves = fraction-binning artifact explains the gap.")



# A4 (priority 2) -- declaration staleness: captain accuracy vs elapsed days

# Captain side of the mirror: departure declarations age on long voyages.
# Elapsed days proxies days-since-declaration for early-voyage declarations.
_decl_df = _build_captain_declared_lookup(data, WORK_DIR, subregion_names)
_dmap = {(int(r.SEG_ID), int(r.STEP_IDX)): int(r.declared_subregion)
         for r in _decl_df.itertuples()}
_step_of = {int(sid): g.sort_values("STEP_IDX")["STEP_IDX"].values
            for sid, g in data.steps_idx.groupby("SEG_ID")}
_occ = pd.DataFrame({"s": tseg}).groupby("s").cumcount().values
_steps_arr = np.array([_step_of[int(s)][o % len(_step_of[int(s)])]
                       for s, o in zip(tseg, _occ)])
tcap = np.array([_dmap.get((int(s), int(st)), -1)
                 for s, st in zip(tseg, _steps_arr)])
has_d = tcap >= 0
cap_ok = tcap == ttrue
rows = []
for lab, m in [("<= 14d", short_m & has_d), ("> 14d", long_m & has_d)]:
    bi = np.digitize(tdays[m], day_edges) - 1
    for b in range(len(day_edges)-1):
        mm = bi == b
        if mm.sum() < 200: continue
        rows.append(dict(group=lab, mid=(day_edges[b]+day_edges[b+1])/2,
                         cap_acc=100*cap_ok[m][mm].mean(), n=int(mm.sum())))
a4 = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9,4.5))
for lab, g in a4.groupby("group"):
    ax.plot(g["mid"], g["cap_acc"], "s--", label=f"captain {lab}")
ax.set_xlabel("days since departure"); ax.set_ylabel("declaration accuracy (%)")
ax.set_title("A4 -- declaration accuracy vs elapsed days (staleness curve)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,"a4_declaration_staleness.png"), dpi=150); plt.show()
print("reading: monotone decay + lower long-haul curve = staleness explains "
      "the captain mirror-image; note: early-band declarations on long hauls "
      "were made further from arrival (in days) than any short-haul one can be.")


# [notebook cell 82]

# A2 (priority 3) -- composition: Kitagawa/Oaxaca decomposition of the gap

# Split the early-band gap into class-mix vs within-class performance.
early = tfrac <= 0.20
def _cls_acc(m):
    df = pd.DataFrame({"y": ttrue[m], "ok": t_ok[m]})
    return df.groupby("y")["ok"].agg(["mean","size"])
S, L = _cls_acc(early & short_m), _cls_acc(early & long_m)
mix_S = S["size"] / S["size"].sum(); mix_L = L["size"] / L["size"].sum()
accS = (S["mean"]*mix_S).sum(); accL = (L["mean"]*mix_L).sum()
cf = (L["mean"].reindex(mix_S.index).fillna(L["mean"].mean()) * mix_S).sum()
print(f"early-band acc: short {100*accS:.1f}%  long {100*accL:.1f}%  gap {100*(accL-accS):.1f}pp")
print(f"counterfactual (long per-class accs on SHORT class mix): {100*cf:.1f}%")
print(f"  -> composition (class mix) explains {100*(accL-cf):.1f}pp of the gap")
print(f"  -> within-class performance explains {100*(cf-accS):.1f}pp")
from scipy.stats import chi2_contingency
ct = pd.crosstab(ttrue[early & (short_m|long_m)], short_m[early & (short_m|long_m)])
chi2, p, dof, _ = chi2_contingency(ct)
print(f"class-mix difference: chi2={chi2:.0f} (dof {dof}), p={p:.2e}")


# [notebook cell 83]

# A3 (priority 4) -- per-group empirical ceiling (E15-D machinery)

# Bayes-rate estimate per group on the early band, given (cell, dep basin).
# ===== port_to_sub -- port id -> arrival subregion id (from traj_idx) =====
port_to_sub = (data.traj_idx.dropna(subset=["ARR_PORT_ID", TARGET_COL])
               .astype({"ARR_PORT_ID": int})
               .groupby("ARR_PORT_ID")[TARGET_COL]
               .agg(lambda v: int(v.mode().iloc[0]))
               .to_dict())
print(f"port_to_sub: {len(port_to_sub)} ports mapped")

cell_lat = np.empty(len(tseg)); cell_lon = np.empty(len(tseg))
for sid, g in data.steps_idx.groupby("SEG_ID"):
    ia = np.where(tseg == int(sid))[0]
    if not len(ia): continue
    la = g.sort_values("STEP_IDX")["GRID_LAT_C"].values
    lo = g.sort_values("STEP_IDX")["GRID_LON_C"].values
    n = len(la)
    if n and len(ia) % n == 0:
        reps = len(ia)//n
        cell_lat[ia] = np.tile(la, reps); cell_lon[ia] = np.tile(lo, reps)
dep_sub_of = {int(s): int(port_to_sub.get(_tj.loc[int(s),"DEP_PORT_ID"], -1))
              for s in np.unique(tseg)}
tdep = np.array([dep_sub_of[int(s)] for s in tseg])
for lab, m in [("<= 14d", early & short_m), ("> 14d", early & long_m)]:
    key = pd.DataFrame({"lat": cell_lat[m], "lon": cell_lon[m],
                        "dep": tdep[m], "y": ttrue[m]})
    maj = key.groupby(["lat","lon","dep"])["y"].agg(
        lambda v: v.value_counts().iloc[0]/len(v))
    sz = key.groupby(["lat","lon","dep"])["y"].size()
    ceil = float((maj*sz).sum()/sz.sum())
    print(f"{lab}: early-band ceiling {100*ceil:.1f}%  vs model "
          f"{100*t_ok[m].mean():.1f}%  (gap {100*(ceil-t_ok[m].mean()):.1f}pp)")
print("reading: similar model-to-ceiling gaps = the groups differ in problem "
      "hardness, not model quality; a LOW short-group ceiling closes the case.")


# [notebook cell 84]

# A5 (priority 5) -- formal wrapper: logistic with cluster-robust SEs

# Does 'short' survive controls? Step-level logit, segment-clustered SEs.
import statsmodels.api as sm
m5 = early & (short_m | long_m)
X = pd.DataFrame({
    "short": short_m[m5].astype(float),
    "log_days_elapsed": np.log1p(tdays[m5]),
    "frac": tfrac[m5],
    "dep": tdep[m5].astype("int32"), "y_cls": ttrue[m5].astype("int32")})
X = pd.get_dummies(X, columns=["dep","y_cls"], drop_first=True, dtype=float)
Xc = sm.add_constant(X)
fit = sm.Logit(t_ok[m5].astype(float), Xc).fit(disp=0,
        cov_type="cluster", cov_kwds={"groups": tseg[m5]})
keep = [i for i in fit.params.index if i in
        ("const","short","log_days_elapsed","frac")]
print(fit.summary2().tables[1].loc[keep].round(4))
print("\nreading: if 'short' shrinks toward 0 once log_days_elapsed and class "
      "controls enter, A1/A2 explained the gap; a surviving negative "
      "coefficient names residual model behaviour worth one report sentence.")
