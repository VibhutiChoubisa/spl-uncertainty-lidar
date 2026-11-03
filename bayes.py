"""
Step 3 - Bayesian depth estimation with uncertainty
===================================================

The idea in one sentence
------------------------
Instead of giving ONE depth per pixel, we compute the probability of EVERY
possible depth, given the photons we measured. That probability curve is
called the POSTERIOR. Its peak is our best guess; its spread is our
uncertainty.

Bayes' rule
-----------
    posterior(depth | photons)  ∝  likelihood(photons | depth) × prior(depth)

  - likelihood: "if the true depth were d, how likely are the photons we saw?"
                We get it from the physics model of Step 1 (Poisson counts).
  - prior:      what we believe before measuring. Here we use a flat prior:
                every depth from 0 to 4.5 m is equally likely.

The likelihood for one pixel
----------------------------
Expected counts:  lambda_t(d) = s * h(t - tau(d)) + b      (Step 1)
Poisson model:    P(y | d) = prod_t  lambda_t^y_t * exp(-lambda_t) / y_t!

Taking the log and dropping terms that do not depend on d:

    log P(y | d) = sum_t  y_t * log(1 + (s/b) * h(t - tau(d)))  + const

Read it as: every photon "votes" for the depths that could have produced it,
and a photon near the expected pulse position votes strongly. This is a
"log-matched filter" - the matched filter of Step 2, done properly.

What we need to know
--------------------
  b - background per bin. Measured beforehand with the laser switched off.
  s - signal photons in this pixel. Estimated as total counts minus
      expected background (never below a small positive number).
  h - pulse shape (IRF). Known from sensor calibration.

Two versions
------------
  pixelwise      - each pixel on its own.
  neighbourhood  - each pixel also uses the photons of its 3x3 neighbours,
                   assuming they are at the same depth. This "borrows
                   strength" (more photons per decision) but the assumption
                   is wrong at object edges. Step 4 tests what that does to
                   the honesty of the uncertainty.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from simulator import irf, DEPTH_PER_BIN, IRF_SIGMA_BINS


def log_likelihood(hist, b, sigma=IRF_SIGMA_BINS):
    """Log-likelihood of every candidate depth bin, for every pixel.

    hist : (H, W, N) photon counts
    b    : background photons per bin (a number)
    Returns L of shape (H, W, N): L[i, j, k] = log P(photons | depth bin k)
    """
    n_bins = hist.shape[2]
    h = irf(n_bins, sigma)                                   # (N,), centred at 0

    # Signal estimate per pixel: total counts minus expected background
    s = np.maximum(hist.sum(axis=2) - b * n_bins, 1e-3)      # (H, W)

    # Each photon's "vote" curve: log(1 + (s/b) h(t - tau)), one per pixel
    g = np.log1p((s / b)[:, :, None] * h[None, None, :])     # (H, W, N)

    # Sum the votes of all photons for every candidate tau at once.
    # This is a cross-correlation of the counts with g, done with the FFT.
    Y = np.fft.rfft(hist, axis=2)
    G = np.fft.rfft(g, axis=2)
    return np.fft.irfft(Y * np.conj(G), n=n_bins, axis=2)


def posterior(L):
    """Turn log-likelihoods into probabilities that add up to 1 per pixel.

    With a flat prior, the posterior is just the normalised likelihood.
    We subtract the maximum first so exp() never overflows.
    """
    p = np.exp(L - L.max(axis=2, keepdims=True))
    return p / p.sum(axis=2, keepdims=True)


def summarise(p, level=0.95):
    """Best guess, uncertainty and credible interval from the posterior.

    Returns a dict of maps (H, W), all in metres:
      depth  - MAP estimate (the peak of the posterior)
      std    - posterior standard deviation (our uncertainty)
      lo, hi - central credible interval containing `level` probability
    """
    n_bins = p.shape[2]
    t = np.arange(n_bins)
    mean = (p * t).sum(axis=2)
    std = np.sqrt((p * (t - mean[:, :, None]) ** 2).sum(axis=2))
    cdf = np.cumsum(p, axis=2)
    a = (1 - level) / 2
    lo = np.argmax(cdf >= a, axis=2)
    hi = np.argmax(cdf >= 1 - a, axis=2)
    return {"depth": np.argmax(p, axis=2) * DEPTH_PER_BIN,
            "std": std * DEPTH_PER_BIN,
            "lo": (lo - 0.5) * DEPTH_PER_BIN,      # half-bin padding: depth
            "hi": (hi + 0.5) * DEPTH_PER_BIN}      # is continuous, bins are not


def bayes_pixelwise(hist, b, sigma=IRF_SIGMA_BINS):
    """Posterior for each pixel using only its own photons."""
    return posterior(log_likelihood(hist, b, sigma))


def bayes_neighbourhood(hist, b, sigma=IRF_SIGMA_BINS, size=3):
    """Posterior for each pixel using its own photons and its neighbours'.

    Adding log-likelihoods = multiplying likelihoods = treating the
    neighbours' photons as extra evidence about THIS pixel's depth.
    """
    L = log_likelihood(hist, b, sigma)
    L_sum = uniform_filter(L, size=(size, size, 1), mode="nearest") * size ** 2
    return posterior(L_sum)


def coverage(summary, true_depth):
    """Share of pixels whose TRUE depth lies inside the credible interval.

    For an honest (calibrated) 95% interval this should be close to 0.95.
    Much lower = overconfident (dangerous for a robot).
    """
    return np.mean((true_depth >= summary["lo"]) & (true_depth <= summary["hi"]))
