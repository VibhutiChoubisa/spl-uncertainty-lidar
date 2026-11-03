"""
Step 2 - Depth estimators
=========================

Each function takes the photon histograms, shape (H, W, N_BINS), and returns
a depth map in metres, shape (H, W).

1. naive_depth            - pick the single bin with the most photons
2. matched_filter_depth   - slide the known pulse shape along the histogram,
                            and pick the position where it fits best

Why the matched filter is better
--------------------------------
The naive method looks at ONE bin at a time. But a real return is spread
over several neighbouring bins (the pulse has a width of a few bins).
Background photons, on the other hand, land in random, isolated bins.

The matched filter adds up the photons in a window shaped like the pulse.
A real return adds up to a bump; a random background photon on its own
does not. In other words, it uses what we KNOW about the sensor (the pulse
shape) to separate signal from noise.

Mathematically it is a cross-correlation:

    score(tau) = sum over t of  y_t * h(t - tau)

and the estimated return time is the tau with the highest score.
It is also the classic textbook detector used in radar and sonar.

Its limit: every pixel is still treated on its own, and it gives a single
answer with no idea of how reliable that answer is. Step 3 fixes both.
"""

import numpy as np
from simulator import irf, DEPTH_PER_BIN


def naive_depth(hist):
    """Bin with the most photons. Ties go to the earliest bin."""
    return np.argmax(hist, axis=2) * DEPTH_PER_BIN


def matched_filter_depth(hist):
    """Cross-correlate each histogram with the pulse shape, take the peak.

    We compute the cross-correlation for all shifts at once with the FFT
    (a fast way to do 'slide and multiply' for every position).
    """
    n_bins = hist.shape[2]
    h = irf(n_bins)                                   # pulse centred at bin 0
    H = np.fft.rfft(h)
    Y = np.fft.rfft(hist, axis=2)
    score = np.fft.irfft(Y * np.conj(H), n=n_bins, axis=2)
    return np.argmax(score, axis=2) * DEPTH_PER_BIN


def fraction_correct(est, true, tol=0.03):
    """Share of pixels whose depth is within `tol` metres (default 3 cm)."""
    return np.mean(np.abs(est - true) < tol)
