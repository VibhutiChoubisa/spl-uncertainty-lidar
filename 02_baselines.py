"""Experiment 2: accuracy of the two baselines across all conditions.

Run:  python 02_baselines.py
Saves: figures/02_baselines_accuracy_grid.png
"""

import numpy as np
import matplotlib.pyplot as plt
import plotstyle as ps
from simulator import make_scene, simulate
from estimators import naive_depth, matched_filter_depth, fraction_correct

PPP_LIST = [1, 2, 5, 10, 20, 50, 100]
SBR_LIST = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

ps.apply()
depth, refl = make_scene()
methods = {"Naive (biggest bin)": naive_depth, "Matched filter": matched_filter_depth}
acc = {k: np.zeros((len(PPP_LIST), len(SBR_LIST))) for k in methods}

for i, ppp in enumerate(PPP_LIST):
    for j, sbr in enumerate(SBR_LIST):
        y = simulate(depth, refl, ppp, sbr, seed=100 * i + j)
        for name, f in methods.items():
            acc[name][i, j] = fraction_correct(f(y), depth)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (name, grid) in zip(axes, acc.items()):
    im = ax.imshow(grid * 100, cmap="Blues", vmin=0, vmax=100, origin="lower",
                   aspect="auto")
    for i in range(len(PPP_LIST)):
        for j in range(len(SBR_LIST)):
            v = grid[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 60 else ps.INK)
    ax.set_xticks(range(len(SBR_LIST)), [str(s) for s in SBR_LIST])
    ax.set_yticks(range(len(PPP_LIST)), [str(p) for p in PPP_LIST])
    ax.set_xlabel("SBR (signal-to-background)  →  clearer")
    ax.set_ylabel("PPP (signal photons per pixel)  →  more light")
    ax.set_title(name, loc="left", weight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
fig.colorbar(im, ax=axes, fraction=0.025, label="% of pixels within 3 cm")
fig.suptitle("Share of pixels with correct depth (bottom-left = hardest)",
             x=0.06, ha="left", fontsize=11)
fig.savefig("figures/02_baselines_accuracy_grid.png", dpi=150, bbox_inches="tight")

for name, grid in acc.items():
    print(f"{name:<20} mean over grid: {grid.mean() * 100:.0f}%")
