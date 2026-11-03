"""Experiment 4: is the uncertainty honest, and is it useful to a robot?

Run:  python 04_calibration.py      (takes about a minute)
Saves:
  figures/04a_accuracy_and_coverage_grid.png
  figures/04b_robot_decision_test.png
  figures/04c_calibration_curves.png

Three questions
---------------
(a) Across all conditions: how accurate is each method, and does its 95%
    interval really contain the truth 95% of the time ("coverage")?
(b) Robot decision test: if a robot ignores the pixels it is least sure
    about, do the errors disappear? If yes, the uncertainty is USEFUL.
(c) Where does honesty break? Calibration curves for interior pixels,
    object edges, and a sensor whose pulse is wider than the model assumes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import maximum_filter, minimum_filter
import plotstyle as ps
from simulator import make_scene, simulate, background_per_bin
from bayes import bayes_pixelwise, bayes_neighbourhood, summarise, coverage
from estimators import fraction_correct

ps.apply()
depth, refl = make_scene()
METHODS = {"Pixelwise Bayes": bayes_pixelwise,
           "Neighbourhood Bayes (3x3)": bayes_neighbourhood}

# ---------------------------------------------------------------------
# (a) Accuracy and coverage across all conditions
# ---------------------------------------------------------------------
PPP_LIST = [1, 2, 5, 10, 20, 50, 100]
SBR_LIST = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
acc = {k: np.zeros((len(PPP_LIST), len(SBR_LIST))) for k in METHODS}
cov = {k: np.zeros((len(PPP_LIST), len(SBR_LIST))) for k in METHODS}
for i, ppp in enumerate(PPP_LIST):
    for j, sbr in enumerate(SBR_LIST):
        y = simulate(depth, refl, ppp, sbr, seed=100 * i + j)
        b = background_per_bin(ppp, sbr)
        for name, f in METHODS.items():
            s = summarise(f(y, b))
            acc[name][i, j] = fraction_correct(s["depth"], depth)
            cov[name][i, j] = coverage(s, depth)


def heatmap(ax, grid, title, cmap, norm, dark_if):
    im = ax.imshow(grid * 100, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if dark_if(v) else ps.INK)
    ax.set_xticks(range(len(SBR_LIST)), [str(s) for s in SBR_LIST])
    ax.set_yticks(range(len(PPP_LIST)), [str(p) for p in PPP_LIST])
    ax.set_xlabel("SBR  →  clearer"); ax.set_ylabel("PPP  →  more light")
    ax.set_title(title, loc="left", weight="bold", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), layout="constrained")
for col, name in enumerate(METHODS):
    im1 = heatmap(axes[0, col], acc[name], f"{name}: accuracy (% within 3 cm)",
                  "Blues", plt.Normalize(0, 100), lambda v: v > 60)
    im2 = heatmap(axes[1, col], cov[name],
                  f"{name}: 95% coverage (ideal 95)", "RdBu",
                  TwoSlopeNorm(vmin=75, vcenter=95, vmax=100),
                  lambda v: v < 80)
fig.colorbar(im1, ax=axes[0, :], shrink=0.9, label="% within 3 cm")
fig.colorbar(im2, ax=axes[1, :], shrink=0.9,
             label="coverage %   (red = overconfident, blue = cautious)")
fig.suptitle("Accuracy and honesty of uncertainty across conditions",
             x=0.06, ha="left", weight="bold", fontsize=11)
fig.savefig("figures/04a_accuracy_and_coverage_grid.png", dpi=150,
            bbox_inches="tight")
for name in METHODS:
    print(f"(a) {name:<26} mean accuracy {acc[name].mean() * 100:4.0f}%   "
          f"mean coverage {cov[name].mean() * 100:4.0f}%   "
          f"worst coverage {cov[name].min() * 100:4.0f}%")

# ---------------------------------------------------------------------
# (b) Robot decision test, in dense fog, averaged over 5 random repeats
# ---------------------------------------------------------------------
PPP, SBR = 2, 0.2
b = background_per_bin(PPP, SBR)
keep = np.linspace(0.05, 1.0, 40)                    # fraction of pixels kept
curves = {k: np.zeros_like(keep) for k in METHODS}
for seed in range(5):
    y = simulate(depth, refl, PPP, SBR, seed=500 + seed)
    for name, f in METHODS.items():
        s = summarise(f(y, b))
        correct = (np.abs(s["depth"] - depth) < 0.03).ravel()
        order = np.argsort(s["std"].ravel())         # most certain first
        for k, fr in enumerate(keep):
            n = max(1, int(fr * correct.size))
            curves[name][k] += correct[order[:n]].mean() / 5

fig, ax = plt.subplots(figsize=(7, 4.4))
styles = {"Pixelwise Bayes": (ps.BLUE, "o"),
          "Neighbourhood Bayes (3x3)": (ps.ORANGE, "s")}
for name, c in curves.items():
    col, mk = styles[name]
    ax.plot(keep * 100, c * 100, color=col, lw=2, marker=mk, ms=4, markevery=4)
    ax.text(101, c[-1] * 100, f"{name}\n(all pixels: {c[-1] * 100:.0f}%)",
            color=ps.INK2, fontsize=8, va="center")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_xlabel("% of pixels the robot keeps (most certain first)")
ax.set_ylabel("% of kept pixels within 3 cm")
ax.grid(color=ps.GRID, lw=0.6)
ax.set_title(f"Robot decision test in dense fog (PPP={PPP}, SBR={SBR}): "
             "drop the least certain pixels", loc="left", fontsize=9.5, weight="bold")
fig.savefig("figures/04b_robot_decision_test.png", dpi=150, bbox_inches="tight")
for name, c in curves.items():
    k = np.argmin(np.abs(keep - 0.5))
    print(f"(b) {name:<26} keep all: {c[-1] * 100:3.0f}% correct   "
          f"keep most certain 50%: {c[k] * 100:3.0f}% correct")

# ---------------------------------------------------------------------
# (c) Calibration curves: where does honesty break?
# ---------------------------------------------------------------------
LEVELS = np.array([0.7, 0.8, 0.9, 0.95, 0.99])
edge = (maximum_filter(depth, 3) - minimum_filter(depth, 3)) > 0.03
cases = {
    "Pixelwise Bayes (all pixels)": (bayes_pixelwise, 2.0, np.ones_like(edge)),
    "Neighbourhood, flat surfaces": (bayes_neighbourhood, 2.0, ~edge),
    "Neighbourhood, object edges": (bayes_neighbourhood, 2.0, edge),
    "Neighbourhood, wrong pulse model": (bayes_neighbourhood, 3.0, np.ones_like(edge)),
}
calib = {k: np.zeros(len(LEVELS)) for k in cases}
for seed in range(5):
    for name, (f, true_sigma, mask) in cases.items():
        # true_sigma = 3 means the real pulse is 50% wider than the model's
        y = simulate(depth, refl, PPP, SBR, seed=900 + seed, irf_sigma=true_sigma)
        p = f(y, b)
        for k, lev in enumerate(LEVELS):
            s = summarise(p, level=lev)
            inside = (depth >= s["lo"]) & (depth <= s["hi"])
            calib[name][k] += inside[mask].mean() / 5

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([60, 100], [60, 100], color=ps.INK2, lw=1, ls="--")
cstyle = [(ps.BLUE, "o", "-"), (ps.AQUA, "^", "-"), (ps.ORANGE, "s", "-"),
          (ps.YELLOW, "D", "--")]
for (name, vals), (col, mk, ls) in zip(calib.items(), cstyle):
    ax.plot(LEVELS * 100, vals * 100, color=col, marker=mk, ms=6, lw=2, ls=ls,
            label=name)
ax.set_xlim(65, 100); ax.set_ylim(60, 100)
ax.set_xlabel("Stated confidence of the interval (%)")
ax.set_ylabel("Actual % of pixels whose true depth is inside")
ax.grid(color=ps.GRID, lw=0.6)
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
ax.text(66, 99, "dashed line = perfectly honest\nabove it = cautious\nbelow it = overconfident",
        color=ps.INK2, fontsize=8, va="top")
ax.set_title(f"Calibration in dense fog (PPP={PPP}, SBR={SBR})", loc="left",
             weight="bold", fontsize=9.5)
fig.savefig("figures/04c_calibration_curves.png", dpi=150, bbox_inches="tight")
for name, vals in calib.items():
    print(f"(c) {name:<34} 95% interval actually covers {vals[3] * 100:3.0f}%")
