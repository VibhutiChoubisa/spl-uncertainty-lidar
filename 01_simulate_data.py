"""Experiment 1: what single-photon data looks like as conditions degrade.

Run:  python 01_simulate_data.py
Saves: figures/01_data_under_degrading_conditions.png
"""

import numpy as np
import matplotlib.pyplot as plt
import plotstyle as ps
from simulator import make_scene, simulate, DEPTH_PER_BIN, N_BINS
from estimators import naive_depth, fraction_correct

ps.apply()
depth, refl = make_scene()
conditions = [("Clear", 100, 10.0), ("Hazy", 10, 1.0), ("Dense fog", 2, 0.2)]
px = (24, 44)                       # a pixel at the centre of the dark sphere
depth_axis = np.arange(N_BINS) * DEPTH_PER_BIN

fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.8, 1], hspace=0.55, wspace=0.3)

# Row 1: the true scene
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(depth, cmap="Blues_r", vmin=1.5, vmax=3.6)
ax.plot(px[1], px[0], "o", ms=7, mfc="none", mec=ps.ORANGE, mew=2)
ax.set_title("True depth (m)", loc="left"); ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(refl, cmap="Greys_r", vmin=0, vmax=1)
ax.set_title("True reflectivity", loc="left"); ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
ax.text(0, 0.95, "The scene", weight="bold", va="top")
ax.text(0, 0.78, "Back wall at 3.5 m\nBright box at 2.0 m\nDark sphere at ~2.3-2.6 m\n\n"
        "Orange circle = the pixel whose\nhistogram is shown below.",
        va="top", color=ps.INK2, linespacing=1.5)

for i, (name, ppp, sbr) in enumerate(conditions):
    y = simulate(depth, refl, ppp, sbr, seed=i)

    # Row 2: one pixel's photon histogram
    ax = fig.add_subplot(gs[1, i])
    ax.bar(depth_axis, y[px], width=DEPTH_PER_BIN, color=ps.BLUE, linewidth=0)
    ax.plot(depth[px], 1.04, marker="v", color=ps.ORANGE, ms=8,
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_title(f"{name}: PPP={ppp}, SBR={sbr}", loc="left", pad=16)
    ax.set_xlabel("Depth (m)"); ax.set_ylabel("Photon count")
    ax.set_xlim(0, N_BINS * DEPTH_PER_BIN)
    if i == 0:
        ax.text(depth[px] + 0.12, 0.88, "true depth (orange)", color=ps.INK2,
                fontsize=8, transform=ax.get_xaxis_transform())

    # Row 3: naive depth map
    est = naive_depth(y)
    good = fraction_correct(est, depth) * 100
    ax = fig.add_subplot(gs[2, i])
    ax.imshow(est, cmap="Blues_r", vmin=1.5, vmax=3.6)
    ax.set_title(f"Naive estimate: {good:.0f}% of pixels within 3 cm",
                 loc="left", fontsize=8.5)
    ax.axis("off")
    print(f"{name:<10} PPP={ppp:<4} SBR={sbr:<5} photons/pixel="
          f"{y.sum(axis=2).mean():5.0f}  naive within 3 cm: {good:.0f}%")

fig.suptitle("Single-photon LiDAR: the same scene under worsening conditions",
             x=0.06, ha="left", weight="bold", fontsize=11)
fig.savefig("figures/01_data_under_degrading_conditions.png", dpi=150,
            bbox_inches="tight")
