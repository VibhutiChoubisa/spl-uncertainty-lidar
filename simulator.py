"""
Step 1 - A simple single-photon LiDAR simulator
================================================

What a single-photon LiDAR measures
-----------------------------------
For every pixel, the system fires many laser pulses and records WHEN each
detected photon arrives. Over many pulses this gives a HISTOGRAM:
"how many photons arrived in each tiny time bin".

That histogram has two parts:
  1. SIGNAL     - photons that bounced off the object. They pile up into a
                  peak. The peak's POSITION gives depth, its HEIGHT gives
                  reflectivity (how bright the surface is).
  2. BACKGROUND - random photons (sunlight, light scattered by fog/water,
                  detector noise). They spread evenly over all bins.

The physics model (one pixel)
-----------------------------
Expected photon count in time bin t:

    lambda_t = s * h(t - tau) + b

    s   = number of signal photons for this pixel
    h   = the laser pulse shape (the "instrument response function", IRF),
          normalised so it sums to 1
    tau = the time bin where the pulse returns, set by depth:
          tau = 2 * depth / (speed of light * bin width)
          (factor 2 because light travels there AND back)
    b   = background photons per bin

The actual counts are random, drawn from a Poisson distribution:

    y_t ~ Poisson(lambda_t)

Poisson is the right model for counting independent random arrivals
(photons, raindrops, phone calls...).

The two "difficulty dials"
--------------------------
PPP (photons per pixel)       - average number of SIGNAL photons per pixel.
                                Fewer photons = less information.
SBR (signal-to-background)    - total signal photons / total background
                                photons in a histogram. Low SBR = "foggy".
"""

import numpy as np

# ---------------------------------------------------------------------
# Physical constants and system settings
# ---------------------------------------------------------------------
C = 3e8                  # speed of light (m/s)
BIN_WIDTH = 100e-12      # time-bin width: 100 picoseconds
N_BINS = 300             # number of time bins per histogram
DEPTH_PER_BIN = C * BIN_WIDTH / 2   # 0.015 m = 1.5 cm per bin (there and back)
IRF_SIGMA_BINS = 2.0     # width of the laser pulse (std, in bins) ~ 3 cm


def make_scene(size=64):
    """Create a simple 'true' 3D scene: a back wall, a box and a sphere.

    Returns
    -------
    depth : (size, size) array, depth of each pixel in metres
    refl  : (size, size) array, reflectivity of each pixel (0..1)
    """
    yy, xx = np.mgrid[0:size, 0:size]

    # Back wall at 3.5 m, medium reflectivity
    depth = np.full((size, size), 3.5)
    refl = np.full((size, size), 0.5)

    # A box in front, at 2.0 m, bright surface
    box = (xx > 8) & (xx < 28) & (yy > 30) & (yy < 56)
    depth[box] = 2.0
    refl[box] = 0.9

    # A sphere of radius 14 px, centre at 2.6 m - its depth changes
    # smoothly across the surface, which makes the problem more realistic
    cx, cy, r = 44, 24, 14
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sphere = d2 < r ** 2
    depth[sphere] = 2.6 - 0.3 * np.sqrt(1 - d2[sphere] / r ** 2)
    refl[sphere] = 0.3          # a darker object: fewer photons come back

    return depth, refl


def irf(n_bins=N_BINS, sigma=IRF_SIGMA_BINS):
    """Laser pulse shape: a Gaussian centred at bin 0, normalised to sum to 1.

    We centre it at 0 and later shift it to each pixel's return time.
    """
    t = np.arange(n_bins)
    t = np.minimum(t, n_bins - t)          # wrap-around distance to bin 0
    h = np.exp(-0.5 * (t / sigma) ** 2)
    return h / h.sum()


def expected_counts(depth, refl, ppp, sbr, irf_sigma=IRF_SIGMA_BINS):
    """Build lambda_t for every pixel (the noise-free 'expected' histogram).

    irf_sigma lets us simulate a sensor whose pulse is wider than the one
    the estimator assumes (a "model mismatch" test in Step 4).

    Returns an array of shape (H, W, N_BINS).
    """
    H, W = depth.shape
    t = np.arange(N_BINS)

    # Signal photons per pixel: scale reflectivity so the AVERAGE is ppp
    s = ppp * refl / refl.mean()                        # (H, W)

    # Background per bin: total background = total signal / SBR, spread
    # evenly over all bins. Same for every pixel here (uniform fog).
    b = (ppp / sbr) / N_BINS

    # Return time (in bins) for each pixel
    tau = depth / DEPTH_PER_BIN                         # (H, W)

    # Gaussian pulse centred at tau for each pixel
    diff = t[None, None, :] - tau[:, :, None]
    pulse = np.exp(-0.5 * (diff / irf_sigma) ** 2)
    pulse /= pulse.sum(axis=2, keepdims=True)           # sums to 1

    return s[:, :, None] * pulse + b


def simulate(depth, refl, ppp, sbr, seed=0, irf_sigma=IRF_SIGMA_BINS):
    """Draw random photon counts: y ~ Poisson(lambda)."""
    rng = np.random.default_rng(seed)
    lam = expected_counts(depth, refl, ppp, sbr, irf_sigma)
    return rng.poisson(lam)


def background_per_bin(ppp, sbr):
    """Background photons per bin. In a real system this is measured
    beforehand by recording with the laser switched off."""
    return (ppp / sbr) / N_BINS

