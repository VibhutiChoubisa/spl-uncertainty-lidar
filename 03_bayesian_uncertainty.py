"""Experiment 3: Bayesian depth estimates WITH uncertainty, in dense fog.

Run:  python 03_bayesian_uncertainty.py
Saves: figures/03_bayesian_estimate_and_uncertainty.png

For each method we show four maps:
  1. estimated depth
  2. uncertainty (posterior standard deviation) - does it know when it doesn't know?
  3. where the estimate is wrong (more than 3 cm off)
  4. where the method is OVERCONFIDENT: the true depth lies outside its
     95% credible interval
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import plotstyle as ps
from simulator import make_scene, simulate, background_per_bin
from bayes import bayes_pixelwise, bayes_neighbourhood, summarise, coverage
from estimators import fraction_correct

PPP, SBR = 2, 0.2                      # "dense fog"
ps.apply()
depth, refl = make_scene()
y = simulate(depth, refl, PPP, SBR, seed=7)
b = background_per_bin(PPP, SBR)

flag = ListedColormap(["#efeee9", ps.ORANGE])   # grey = fine, orange = flagged
methods = {"Pixelwise Bayes": bayes_pixelwise,
           "Neighbourhood Bayes (3x3)": bayes_neighbourhood}

fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
for row, (name, f) in zip(axes, methods.items()):
    s = summarise(f(y, b))
    wrong = np.abs(s["depth"] - depth) >= 0.03
    missed = ~((depth >= s["lo"]) & (depth <= s["hi"]))
    acc = fraction_correct(s["depth"], depth) * 100
    cov = coverage(s, depth) * 100

    row[0].imshow(s["depth"], cmap="Blues_r", vmin=1.5, vmax=3.6)
    row[0].set_title(f"{name}\nDepth estimate: {acc:.0f}% within 3 cm",
                     loc="left", fontsize=9)
    im = row[1].imshow(s["std"] * 100, cmap="Purples", norm=LogNorm(1, 150))
    row[1].set_title("Uncertainty (std, cm)\ndark = unsure", loc="left", fontsize=9)
    fig.colorbar(im, ax=row[1], fraction=0.046)
    row[2].imshow(wrong, cmap=flag, vmin=0, vmax=1)
    row[2].set_title(f"Wrong by > 3 cm (orange)\n{wrong.mean() * 100:.0f}% of pixels",
                     loc="left", fontsize=9)
    row[3].imshow(missed, cmap=flag, vmin=0, vmax=1)
    row[3].set_title(f"Overconfident (orange): truth outside\n95% interval. "
                     f"Coverage {cov:.0f}% (ideal 95%)", loc="left", fontsize=9)
    for ax in row:
        ax.axis("off")
    print(f"{name:<27} accuracy {acc:.0f}%   95% interval coverage {cov:.0f}%")

fig.suptitle(f"Bayesian estimates with uncertainty in dense fog (PPP={PPP}, SBR={SBR})",
             x=0.04, ha="left", weight="bold", fontsize=11)
fig.savefig("figures/03_bayesian_estimate_and_uncertainty.png", dpi=150,
            bbox_inches="tight")
