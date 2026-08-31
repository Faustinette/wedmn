# =============================================================================
# E34 — cartopy trajectory maps (requires cartopy)
# Migrated verbatim from Main_forGitHub.ipynb cells [244, 245, 246, 247].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 244]
# ----------------------------------------------------------------------
# NEW =============================================================================
# E34-MAPS -- one geographic diagram per error mode (5 figures)
# =============================================================================
# Self-contained: needs only matplotlib (+ cartopy if available, else fallback
# to a plain lat/lon canvas with hand-drawn context). Run in Colab:
# [notebook magic removed] !pip -q install cartopy
import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    print("cartopy unavailable -- using plain axes (install for coastlines)")

W = globals().get("WORK_DIR", ".")

# ---- waypoints (lon, lat), coarse but recognisable --------------------------
HOUSTON   = (-95.0, 29.5); GIBRALTAR = (-5.5, 36.0); ROTTERDAM = (4.5, 52.0)
FOS       = (5.0, 43.3);  CAPE      = (19.0, -34.8); SRILANKA  = (81.0, 5.5)
SIKKA     = (69.8, 22.4); SINGAPORE = (103.8, 1.3);  CHIBA     = (140.0, 35.5)
RASTANURA = (50.1, 26.6); MOMBASA   = (39.7, -4.0);  YOKKAICHI = (136.6, 34.9)
KITIMAT   = (-128.7, 54.0); LA      = (-118.3, 33.7); PANAMA   = (-79.6, 8.9)
KOCHI     = (76.2, 9.9);  HORMUZ    = (56.5, 26.5)

def _seg(*pts, n=40):
    pts = np.array(pts, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, n)[:, None]
        out.append(a * (1 - t) + b * t)
    return np.vstack(out)

def _ax(extent, title, figsize=(9.5, 5.4)):
    if HAS_CARTOPY:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
        ax.add_feature(cfeature.OCEAN, facecolor="#eaf4fb")
        ax.add_feature(cfeature.COASTLINE, lw=.6, edgecolor="#888")
        ax.add_feature(cfeature.BORDERS, lw=.3, edgecolor="#bbb")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_facecolor("#eaf4fb")
        ax.set_aspect("auto")
    ax.set_title(title, fontsize=11)
    return fig, ax

def _route(ax, pts, color, label, lw=2.6, ls="-"):
    xy = _seg(*pts)
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, label=label,
            solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=lw + 1.6, foreground="white"),
                          pe.Normal()], **kw)

def _mark(ax, pt, txt, color="k", star=False, dy=1.8):
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    ax.plot(*pt, "*" if star else "o", color=color,
            ms=15 if star else 7, mec="white", mew=1.2, **kw)
    ax.text(pt[0], pt[1] + dy, txt, fontsize=9, ha="center",
            fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            **kw)

def _note(ax, x, y, txt, color="#b30000"):
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    ax.text(x, y, txt, fontsize=8.6, color=color, style="italic",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            **kw)

def _save(fig, name, legend_loc="lower center", ncol=2):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_loc, fontsize=8.6, framealpha=.95,
               ncol=ncol, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(os.path.join(W, name), dpi=170, bbox_inches="tight")
    plt.show()

# ---- Mode 1: geometry capture, released at the fork (USGC->NWE vs MED) -----
fig, ax = _ax([-100, 20, 15, 58], "Mode 1 -- Geometry capture, released at "
              "the fork: USGC\u2192NWE read as MED until Gibraltar")
_route(ax, [HOUSTON, (-60, 27), (-30, 33), GIBRALTAR], "#777777",
       "shared Atlantic corridor (identical for both)", lw=5)
_route(ax, [GIBRALTAR, FOS], "#d62728", "MED branch (majority read)")
_route(ax, [GIBRALTAR, (-8, 44), ROTTERDAM], "#7b52ab", "NWE branch (truth)")
_mark(ax, HOUSTON, "Houston"); _mark(ax, GIBRALTAR, "Gibraltar FORK",
      color="#b30000", star=True); _mark(ax, ROTTERDAM, "Rotterdam",
      color="#7b52ab"); _mark(ax, FOS, "Fos (MED)", color="#d62728", dy=-3.2)
_note(ax, -62, 22, "model output mid-corridor: MED 50.5% vs NWE 45.2%\n"
      "released after the fork: NWE 85.9% in the final fifth")
_save(fig, "e34_mode1_nwe_med.png")

# ---- Mode 2: fork too late (USGC->India vs SEAsia via Cape) ----------------
fig, ax = _ax([-100, 145, -42, 42], "Mode 2 -- Fork too late: USGC\u2192India "
              "read as SEAsia; fork only near Sri Lanka")
_route(ax, [HOUSTON, (-40, 5), (-10, -20), CAPE, (55, -25), SRILANKA],
       "#777777", "shared Cape corridor (identical for both)", lw=5)
_route(ax, [SRILANKA, KOCHI, SIKKA], "#2ca02c", "India branch (truth)")
_route(ax, [SRILANKA, SINGAPORE], "#ff7f0e", "SEAsia branch (majority read)")
_mark(ax, HOUSTON, "Houston"); _mark(ax, CAPE, "Cape of Good Hope", dy=-4)
_mark(ax, SRILANKA, "Sri Lanka FORK (~90% of voyage)", color="#b30000",
      star=True); _mark(ax, SIKKA, "Sikka (India)", color="#2ca02c")
_mark(ax, SINGAPORE, "Singapore", color="#ff7f0e", dy=-4)
_note(ax, -35, -35, "SEAsia holds 33-38% of predictions THROUGHOUT;\n"
      "India reaches only 61.5% even in the final fifth")
_save(fig, "e34_mode2_india_seasia.png")

# ---- Mode 3: sequential capture to adjacent coast (NEAsia->Canada) ---------
# NOTE: drawn in 0-360 longitude so the Pacific is contiguous (no dateline cut).

PROJ = ccrs.PlateCarree(central_longitude=200)   # Pacific in the middle
DAT  = ccrs.PlateCarree()

# continuous 0-360 longitudes: the path never crosses the dateline seam
CHIBA   = (140.0, 35.5)
FORK    = (225.0, 52.0)
KITIMAT = (231.3, 54.0)
LA      = (241.7, 33.7)
PANAMA  = (280.4,  8.9)
HOUSTON = (265.0, 29.5)

fig = plt.figure(figsize=(12, 7.5))
ax = plt.axes(projection=PROJ)
ax.set_extent([125, 265, -5, 66], crs=DAT)
ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
ax.add_feature(cfeature.OCEAN, facecolor="#eaf4fb")
ax.add_feature(cfeature.COASTLINE, lw=.6, edgecolor="#888")
ax.add_feature(cfeature.BORDERS, lw=.3, edgecolor="#bbb")

def line(pts, color, label, lw=3.0, ls="-"):
    xy = np.array(pts, float)
    ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, label=label,
            transform=DAT, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=lw + 1.8, foreground="white"),
                          pe.Normal()])

def mark(pt, txt, color="k", star=False, dy=2.4):
    ax.plot(*pt, "*" if star else "o", color=color, ms=17 if star else 8,
            mec="white", mew=1.3, transform=DAT, zorder=5)
    ax.text(pt[0], pt[1] + dy, txt, fontsize=11, ha="center",
            fontweight="bold", color=color, transform=DAT,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

line([CHIBA, (165, 43), (190, 49), (210, 52), FORK], "#777777",
     "shared trans-Pacific corridor", lw=6)
line([FORK, (228, 53.5), KITIMAT], "#2ca02c",
     "Canada branch (truth, 2.7% of training)")
line([FORK, (234, 45), LA], "#1f77b4", "USWC branch (adjacent-coast captor)")
line([LA, (255, 20), PANAMA, (276, 19), HOUSTON], "#9467bd",
     "USGC-return (early captor, 20x traffic)", ls="--", lw=2.4)

mark(CHIBA, "Chiba (NEAsia)")
mark(KITIMAT, "Kitimat (truth)", color="#2ca02c")
mark(LA, "US West Coast", color="#1f77b4", dy=-4.5)
mark(FORK, "coastal FORK", color="#b30000", star=True, dy=3.0)

ax.text(150, 4, "early: USGC-return holds 57% of predictions;\n"
                "mid/late: USWC 65-80%; Canada never above 13% top-1\n"
                "-- yet median RANK 2 from mid-voyage",
        transform=DAT, fontsize=10.5, color="#b30000", style="italic",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

ax.set_title("Mode 3 -- Sequential capture: NEAsia\u2192Canada read as "
             "USGC-return, then as USWC", fontsize=13)
ax.legend(loc="lower right", fontsize=10, framealpha=.95)
plt.savefig(os.path.join(WORK_DIR, "e34_mode3_canada.png"), dpi=170,
            bbox_inches="tight")

# ---- Mode 4: context prior vs geometry (NEAsia->SEAsia) --------------------
fig, ax = _ax([95, 152, -10, 46], "Mode 4 -- Context prior against the "
              "geometry: NEAsia\u2192SEAsia read as a reload leg")
_route(ax, [YOKKAICHI, (125, 22), (112, 10), SINGAPORE], "#ff7f0e",
       "actual southbound track (truth: SEAsia)", lw=3)
_route(ax, [YOKKAICHI, (150, 20)], "#9467bd",
       "expected: USGC return (context prior)", ls="--", lw=2)
_route(ax, [(112, 10), (101, 3)], "#8c564b",
       "expected: ME reload via Malacca (context prior)", ls="--", lw=2)
_mark(ax, YOKKAICHI, "Yokkaichi (NEAsia)")
_mark(ax, SINGAPORE, "Singapore (truth)", color="#ff7f0e", dy=-2.8)
_note(ax, 97, 41.5, "track points SOUTH from day one, yet USGC+ME hold\n"
      "75-83% of predictions through mid-voyage; truth rank 3;\n"
      "habitual SEAsia traders: 0/138 mid-voyage steps correct")
_save(fig, "e34_mode4_neasia_seasia.png")

# ---- Mode 5: context prior WITH geometry (India->India) --------------------
fig, ax = _ax([45, 90, 0, 32], "Mode 5 -- Context prior riding an agreeing "
              "corridor: India\u2192India read as ME reload")
_route(ax, [KOCHI, (72, 15), SIKKA], "#2ca02c",
       "actual coastal track (truth: Sikka, NW India)", lw=3)
_route(ax, [(72, 15), (63, 22), HORMUZ, RASTANURA], "#8c564b",
       "expected continuation: Hormuz reload (context prior)", ls="--", lw=2)
_mark(ax, KOCHI, "Kochi"); _mark(ax, SIKKA, "Sikka FORK-at-berth",
      color="#b30000", star=True)
_mark(ax, RASTANURA, "Ras Tanura (ME)", color="#8c564b", dy=-2.8)
_note(ax, 46, 3, "ME holds 58% of predictions in the FIRST bin (before any\n"
      "track) and 36% even at 80-100%: the NW-India course and the\n"
      "Gulf course are the same line until the berth")
_save(fig, "e34_mode5_india_india.png")
print("5 map diagrams saved to", W)

# ----------------------------------------------------------------------
# [notebook cell 245]
# ----------------------------------------------------------------------
# ---- Mode 3: sequential capture (NEAsia->Canada), Pacific-centred ----------
import matplotlib.pyplot as plt, matplotlib.patheffects as pe, numpy as np, os
import cartopy.crs as ccrs, cartopy.feature as cfeature

PROJ = ccrs.PlateCarree(central_longitude=200)   # Pacific in the middle
DAT = ccrs.PlateCarree()                          # data given in -180..180

CHIBA = (140.0, 35.5); FORK = (-135.0, 52.0); KITIMAT = (-128.7, 54.0)
LA = (-118.3, 33.7); PANAMA = (-79.6, 8.9); HOUSTON = (-95.0, 29.5)

fig = plt.figure(figsize=(12, 7.5))
ax = plt.axes(projection=PROJ)
ax.set_extent([125, 265, -5, 66], crs=DAT)        # Asia left, Americas right
ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
ax.add_feature(cfeature.OCEAN, facecolor="#eaf4fb")
ax.add_feature(cfeature.COASTLINE, lw=.6, edgecolor="#888")
ax.add_feature(cfeature.BORDERS, lw=.3, edgecolor="#bbb")

def line(pts, color, label, lw=3.0, ls="-"):
    xy = np.array(pts, float)
    ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, label=label,
            transform=DAT, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=lw + 1.8, foreground="white"),
                          pe.Normal()])

def mark(pt, txt, color="k", star=False, dy=2.4):
    ax.plot(*pt, "*" if star else "o", color=color, ms=17 if star else 8,
            mec="white", mew=1.3, transform=DAT, zorder=5)
    ax.text(pt[0], pt[1] + dy, txt, fontsize=11, ha="center", fontweight="bold",
            color=color, transform=DAT,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# 0-360 continuous longitudes: no dateline jump
CHIBA   = (140.0, 35.5)
FORK    = (225.0, 52.0)     # = -135
KITIMAT = (231.3, 54.0)     # = -128.7
LA      = (241.7, 33.7)     # = -118.3
PANAMA  = (280.4,  8.9)     # =  -79.6
HOUSTON = (265.0, 29.5)     # =  -95.0

line([CHIBA, (165, 43), (190, 49), (210, 52), FORK], "#777777",
     "shared trans-Pacific corridor", lw=6)
line([FORK, (228, 53.5), KITIMAT], "#2ca02c",
     "Canada branch (truth, 2.7% of training)")
line([FORK, (234, 45), LA], "#1f77b4", "USWC branch (adjacent-coast captor)")
line([LA, (255, 20), PANAMA, (276, 19), HOUSTON], "#9467bd",
     "USGC-return (early captor, 20x traffic)", ls="--", lw=2.4)

mark(CHIBA, "Chiba (NEAsia)")
mark(KITIMAT, "Kitimat (truth)", color="#2ca02c")
mark(LA, "US West Coast", color="#1f77b4", dy=-4.5)
mark(FORK, "coastal FORK", color="#b30000", star=True, dy=3.0)
ax.text(150, 4, "early: USGC-return holds 57% of predictions;\n"
        "mid/late: USWC 65-80%; Canada never above 13% top-1\n"
        "-- yet median RANK 2 from mid-voyage", transform=DAT, fontsize=10.5,
        color="#b30000", style="italic",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

ax.set_title("Mode 3 -- Sequential capture: NEAsia\u2192Canada read as "
             "USGC-return, then as USWC", fontsize=13)
ax.legend(loc="lower right", fontsize=10, framealpha=.95)
plt.savefig(os.path.join(WORK_DIR, "e34_mode3_canada.png"), dpi=170,
            bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# [notebook cell 246]
# ----------------------------------------------------------------------
# OLD =============================================================================
# E34-MAPS -- one geographic diagram per error mode (5 figures)
# =============================================================================
# Self-contained: needs only matplotlib (+ cartopy if available, else fallback
# to a plain lat/lon canvas with hand-drawn context). Run in Colab:
#   !pip -q install cartopy
import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    print("cartopy unavailable -- using plain axes (install for coastlines)")

W = globals().get("WORK_DIR", ".")

# ---- waypoints (lon, lat), coarse but recognisable --------------------------
HOUSTON   = (-95.0, 29.5); GIBRALTAR = (-5.5, 36.0); ROTTERDAM = (4.5, 52.0)
FOS       = (5.0, 43.3);  CAPE      = (19.0, -34.8); SRILANKA  = (81.0, 5.5)
SIKKA     = (69.8, 22.4); SINGAPORE = (103.8, 1.3);  CHIBA     = (140.0, 35.5)
RASTANURA = (50.1, 26.6); MOMBASA   = (39.7, -4.0);  YOKKAICHI = (136.6, 34.9)
KITIMAT   = (-128.7, 54.0); LA      = (-118.3, 33.7); PANAMA   = (-79.6, 8.9)
KOCHI     = (76.2, 9.9);  HORMUZ    = (56.5, 26.5)

def _seg(*pts, n=40):
    pts = np.array(pts, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, n)[:, None]
        out.append(a * (1 - t) + b * t)
    return np.vstack(out)

def _ax(extent, title):
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(9.5, 5.4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
        ax.add_feature(cfeature.OCEAN, facecolor="#eaf4fb")
        ax.add_feature(cfeature.COASTLINE, lw=.6, edgecolor="#888")
        ax.add_feature(cfeature.BORDERS, lw=.3, edgecolor="#bbb")
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_facecolor("#eaf4fb")
    ax.set_title(title, fontsize=11)
    return fig, ax

def _route(ax, pts, color, label, lw=2.6, ls="-"):
    xy = _seg(*pts)
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, label=label,
            solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=lw + 1.6, foreground="white"),
                          pe.Normal()], **kw)

def _mark(ax, pt, txt, color="k", star=False, dy=1.8):
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    ax.plot(*pt, "*" if star else "o", color=color,
            ms=15 if star else 7, mec="white", mew=1.2, **kw)
    ax.text(pt[0], pt[1] + dy, txt, fontsize=9, ha="center",
            fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            **kw)

def _note(ax, x, y, txt, color="#b30000"):
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    ax.text(x, y, txt, fontsize=8.6, color=color, style="italic",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            **kw)

def _save(fig, name):
    fig.legend(loc="lower left", fontsize=8.6, framealpha=.9,
               bbox_to_anchor=(0.12, 0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(W, name), dpi=170, bbox_inches="tight")
    plt.show()

# ---- Mode 1: geometry capture, released at the fork (USGC->NWE vs MED) -----
fig, ax = _ax([-100, 20, 15, 58], "Mode 1 -- Geometry capture, released at "
              "the fork: USGC\u2192NWE read as MED until Gibraltar")
_route(ax, [HOUSTON, (-60, 27), (-30, 33), GIBRALTAR], "#777777",
       "shared Atlantic corridor (identical for both)", lw=5)
_route(ax, [GIBRALTAR, FOS], "#d62728", "MED branch (majority read)")
_route(ax, [GIBRALTAR, (-8, 44), ROTTERDAM], "#7b52ab", "NWE branch (truth)")
_mark(ax, HOUSTON, "Houston"); _mark(ax, GIBRALTAR, "Gibraltar FORK",
      color="#b30000", star=True); _mark(ax, ROTTERDAM, "Rotterdam",
      color="#7b52ab"); _mark(ax, FOS, "Fos (MED)", color="#d62728", dy=-3.2)
_note(ax, -62, 22, "model output mid-corridor: MED 50.5% vs NWE 45.2%\n"
      "released after the fork: NWE 85.9% in the final fifth")
_save(fig, "e34_mode1_nwe_med.png")

# ---- Mode 2: fork too late (USGC->India vs SEAsia via Cape) ----------------
fig, ax = _ax([-100, 145, -42, 42], "Mode 2 -- Fork too late: USGC\u2192India "
              "read as SEAsia; fork only near Sri Lanka")
_route(ax, [HOUSTON, (-40, 5), (-10, -20), CAPE, (55, -25), SRILANKA],
       "#777777", "shared Cape corridor (identical for both)", lw=5)
_route(ax, [SRILANKA, KOCHI, SIKKA], "#2ca02c", "India branch (truth)")
_route(ax, [SRILANKA, SINGAPORE], "#ff7f0e", "SEAsia branch (majority read)")
_mark(ax, HOUSTON, "Houston"); _mark(ax, CAPE, "Cape of Good Hope", dy=-4)
_mark(ax, SRILANKA, "Sri Lanka FORK (~90% of voyage)", color="#b30000",
      star=True); _mark(ax, SIKKA, "Sikka (India)", color="#2ca02c")
_mark(ax, SINGAPORE, "Singapore", color="#ff7f0e", dy=-4)
_note(ax, -35, -35, "SEAsia holds 33-38% of predictions THROUGHOUT;\n"
      "India reaches only 61.5% even in the final fifth")
_save(fig, "e34_mode2_india_seasia.png")

# ---- Mode 3: sequential capture to adjacent coast (NEAsia->Canada) ---------
fig, ax = _ax([125, -60, 15, 62], "Mode 3 -- Sequential capture: "
              "NEAsia\u2192Canada read as USGC-return, then as USWC")
if HAS_CARTOPY:  # dateline-crossing extent
    ax.set_extent([125, 300, 15, 62], ccrs.PlateCarree())
_route(ax, [CHIBA, (170, 45), (200 - 360 if not HAS_CARTOPY else 200, 50),
            (225 if HAS_CARTOPY else -135, 52)], "#777777",
       "shared trans-Pacific corridor", lw=5)
_route(ax, [(225 if HAS_CARTOPY else -135, 52), KITIMAT if HAS_CARTOPY else
            (-128.7, 54)], "#2ca02c", "Canada branch (truth, 2.7% of training)")
_route(ax, [(225 if HAS_CARTOPY else -135, 52), (232 if HAS_CARTOPY else -128, 44),
            LA if HAS_CARTOPY else (-118.3, 33.7)], "#1f77b4",
       "USWC branch (adjacent-coast captor)")
_route(ax, [(232 if HAS_CARTOPY else -128, 44), (255 if HAS_CARTOPY else -105, 25),
            PANAMA if HAS_CARTOPY else (-79.6, 8.9),
            HOUSTON if HAS_CARTOPY else (-95, 29.5)], "#9467bd",
       "USGC-return (early captor, 20x traffic)", ls="--", lw=2)
_mark(ax, CHIBA, "Chiba (NEAsia)")
_mark(ax, (225 if HAS_CARTOPY else -135, 52), "coastal FORK", color="#b30000",
      star=True)
_note(ax, 150, 22, "early: USGC 57% of predictions; mid/late: USWC 65-80%;\n"
      "Canada never above 13% top-1 -- but median RANK 2 from mid-voyage")
_save(fig, "e34_mode3_canada.png")

# ---- Mode 4: context prior vs geometry (NEAsia->SEAsia) --------------------
fig, ax = _ax([95, 150, -8, 45], "Mode 4 -- Context prior against the "
              "geometry: NEAsia\u2192SEAsia read as a reload leg")
_route(ax, [YOKKAICHI, (125, 22), (112, 10), SINGAPORE], "#ff7f0e",
       "actual southbound track (truth: SEAsia)", lw=3)
_route(ax, [YOKKAICHI, (150, 20)], "#9467bd",
       "expected: USGC return (context prior)", ls="--", lw=2)
_route(ax, [(112, 10), (101, 3)], "#8c564b",
       "expected: ME reload via Malacca (context prior)", ls="--", lw=2)
_mark(ax, YOKKAICHI, "Yokkaichi (NEAsia)")
_mark(ax, SINGAPORE, "Singapore (truth)", color="#ff7f0e", dy=-2.8)
_note(ax, 96, 40, "track points SOUTH from day one, yet USGC+ME hold\n"
      "75-83% of predictions through mid-voyage; truth rank 3;\n"
      "habitual SEAsia traders: 0/138 mid-voyage steps correct")
_save(fig, "e34_mode4_neasia_seasia.png")

# ---- Mode 5: context prior WITH geometry (India->India) --------------------
fig, ax = _ax([45, 90, 0, 32], "Mode 5 -- Context prior riding an agreeing "
              "corridor: India\u2192India read as ME reload")
_route(ax, [KOCHI, (72, 15), SIKKA], "#2ca02c",
       "actual coastal track (truth: Sikka, NW India)", lw=3)
_route(ax, [(72, 15), (63, 22), HORMUZ, RASTANURA], "#8c564b",
       "expected continuation: Hormuz reload (context prior)", ls="--", lw=2)
_mark(ax, KOCHI, "Kochi"); _mark(ax, SIKKA, "Sikka FORK-at-berth",
      color="#b30000", star=True)
_mark(ax, RASTANURA, "Ras Tanura (ME)", color="#8c564b", dy=-2.8)
_note(ax, 46, 3, "ME holds 58% of predictions in the FIRST bin (before any\n"
      "track) and 36% even at 80-100%: the NW-India course and the\n"
      "Gulf course are the same line until the berth")
_save(fig, "e34_mode5_india_india.png")
print("5 map diagrams saved to", W)

# ----------------------------------------------------------------------
# [notebook cell 247]
# ----------------------------------------------------------------------
# =============================================================================
# E34-DIAGRAMS -- model-mechanism schematic per error mode (5 figures)
# =============================================================================
# Pure matplotlib. Each figure: the pipeline (channels -> MSF -> readout),
# the failing link in red, the proposed fix in green.
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W = globals().get("WORK_DIR", ".")
GREY, RED, GREEN, BLUE = "#666666", "#c62828", "#2e7d32", "#1565c0"

def _box(ax, x, y, w, h, txt, fc="#f5f5f5", ec=GREY, fs=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)

def _arrow(ax, p, q, color=GREY, lw=1.8, ls="-", txt=None, ty=0.02):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))
    if txt:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + ty, txt,
                fontsize=8, color=color, ha="center", style="italic")

def _base(ax, spatial_c=GREY, local_c=GREY, ident_c=GREY, read_c=GREY):
    _box(ax, .02, .62, .17, .10, "Spatial channel\n(track so far)", ec=spatial_c)
    _box(ax, .02, .48, .17, .10, "Local pattern\n(GRU: speed/turns)", ec=local_c)
    _box(ax, .02, .34, .17, .10, "Departure port +\nShip history", ec=ident_c)
    _box(ax, .02, .20, .17, .10, "Temporal encoding", ec=GREY)
    _box(ax, .27, .34, .20, .38, "2\u00d7 MSF blocks\n(CFA \u2192 TSA \u2192 MoEFF\nwith gating)")
    _box(ax, .55, .38, .20, .30, "15-class readout\n(class prototypes\n+ softmax)", ec=read_c)
    _box(ax, .83, .44, .14, .18, "argmax\nprediction")
    for y in (.67, .53, .39, .25):
        _arrow(ax, (.19, y), (.27, min(max(y, .40), .66)))
    _arrow(ax, (.47, .53), (.55, .53)); _arrow(ax, (.75, .53), (.83, .53))

def _fig(title):
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=11)
    return fig, ax

def _fail(ax, x, y, txt):
    ax.text(x, y, "PROBLEM: " + txt, fontsize=9, color=RED, fontweight="bold")

def _fixes(ax, x, y, lines):
    ax.text(x, y, "FIX:", fontsize=9, color=GREEN, fontweight="bold")
    for i, l in enumerate(lines):
        ax.text(x + .035, y - .05 * (i + 1) + .01, "\u2713 " + l,
                fontsize=8.6, color=GREEN)

# ---- Mode 1 -----------------------------------------------------------------
fig, ax = _fig("Mode 1 -- Geometry capture, released: information present, "
               "readout outvoted until the fork")
_base(ax, spatial_c=RED, read_c=RED)
_fail(ax, .02, .10, "shared corridor \u21d2 spatial evidence identical for both "
      "classes; readout allocates the mass to the corridor majority "
      "(MED) although the pair is 88% linearly separable in the representation")
_fixes(ax, .02, .93, [
    "two-candidate product surface (top-2 covers 84\u201393% of these steps)",
    "corridor-pair auxiliary loss: force the readout to spend the pair information",
])
fig.tight_layout(); fig.savefig(os.path.join(W, "e34_diag_mode1.png"), dpi=170)
plt.show()

# ---- Mode 2 -----------------------------------------------------------------
fig, ax = _fig("Mode 2 -- Fork too late: the discriminating steps arrive when "
               "accumulated evidence already dominates")
_base(ax, local_c=RED, read_c=RED)
_fail(ax, .02, .10, "the GRU sees the late turn, but a handful of fork-zone "
      "steps must overturn 100+ accumulated corridor steps in the pooled "
      "representation; fork-zone data is also the scarcest slice of training")
_fixes(ax, .02, .93, [
    "recency-weighted auxiliary head (fast local route to the logits)",
    "late-step / fork-zone upweighting in the loss",
    "top-2 surface as the deployable stopgap",
])
fig.tight_layout(); fig.savefig(os.path.join(W, "e34_diag_mode2.png"), dpi=170)
plt.show()

# ---- Mode 3 -----------------------------------------------------------------
fig, ax = _fig("Mode 3 -- Sequential adjacent-coast capture: a 125-voyage "
               "prototype cannot win anywhere")
_base(ax, read_c=RED)
_fail(ax, .02, .10, "Canada's readout prototype is estimated from 125 training "
      "voyages (2.7%): high variance, weak pull; the decision boundary sits "
      "inside Canada's territory, so the neighbouring USWC prototype wins "
      "even post-fork (truth stuck at rank 2)")
_fixes(ax, .02, .93, [
    "logit adjustment (per-departure class-prior correction, dial \u03c4)",
    "top-2 surface: covers 80\u201387% of late steps on this lane",
])
fig.tight_layout(); fig.savefig(os.path.join(W, "e34_diag_mode3.png"), dpi=170)
plt.show()

# ---- Mode 4 -----------------------------------------------------------------
fig, ax = _fig("Mode 4 -- Context prior against the geometry: identity "
               "channels overrule a contradicting track")
_base(ax, ident_c=RED, read_c=RED)
_fail(ax, .02, .10, "departure-context prior (\u2018from NEAsia, ships reload at "
      "USGC/ME\u2019) carries 75\u201383% of the prediction mass although the track "
      "points south from day one; truth rank 3; habit does not rescue "
      "(0/138 habitual steps)")
_fixes(ax, .02, .93, [
    "context-channel dropout in training (force a competent track-only path)",
    "logit adjustment conditioned on departure region",
    "recency-weighted head (believe the track when it contradicts the habit)",
])
fig.tight_layout(); fig.savefig(os.path.join(W, "e34_diag_mode4.png"), dpi=170)
plt.show()

# ---- Mode 5 -----------------------------------------------------------------
fig, ax = _fig("Mode 5 -- Context prior riding an agreeing corridor: wrong "
               "before departure, never contradicted")
_base(ax, ident_c=RED, spatial_c=RED, read_c=RED)
_fail(ax, .02, .10, "the reload prior (ME, 58% in the first bin) AND the "
      "coastal course toward the Gulf agree with each other and against the "
      "truth until the berth; no single channel can be blamed \u2014 the "
      "conjunction is the problem")
_fixes(ax, .02, .93, [
    "both remedies at once: context dropout / logit adjustment + fork-zone head",
    "or new information (fixtures, cargo data) \u2014 the honest ceiling for this mode",
])
fig.tight_layout(); fig.savefig(os.path.join(W, "e34_diag_mode5.png"), dpi=170)
plt.show()
print("5 mechanism diagrams saved to", W)
