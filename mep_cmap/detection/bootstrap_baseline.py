"""
mep_cmap.detection.bootstrap_baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared pre-stimulus baseline statistics used by both the bootstrap onset
detector and the CSP detector.

  • compute_bootstrap_threshold  — vectorised bootstrap noise threshold
"""

import numpy as np


def compute_bootstrap_threshold(pre_abs, criterion=1.96, n_boot=500, seed=42):
    """
    Compute the bootstrap-calibrated onset threshold from pre-stimulus signal.

    Separated from detect_mep_onset_bootstrap so it can be computed ONCE
    per stim type and reused across all segments, rather than being
    recomputed for every segment independently.

    Parameters
    ----------
    pre_abs   : np.ndarray  absolute pre-stim signal
    criterion : float       z-score multiplier (default 1.96)
    n_boot    : int         bootstrap iterations (default 500)
    seed      : int         RNG seed for reproducibility

    Returns
    -------
    threshold : float  onset detection threshold, or None if invalid
    mu_abs    : float  bootstrap median of |pre-stim|
    sigma_abs : float  std of |pre-stim|
    """
    if len(pre_abs) < 5:
        return None, None, None
    rng          = np.random.default_rng(seed)
    n_pre        = len(pre_abs)
    boot_indices = rng.integers(0, n_pre, size=(n_boot, n_pre))
    boot_med     = np.median(pre_abs[boot_indices], axis=1)
    mu_abs       = float(np.median(boot_med))
    sigma_abs    = float(np.std(pre_abs, ddof=1)) if len(pre_abs) > 1 else mu_abs
    if mu_abs < 1e-9:
        return None, None, None
    threshold = mu_abs + criterion * sigma_abs
    threshold = float(np.clip(threshold, mu_abs * 1.5, mu_abs * 5.0))
    return threshold, mu_abs, sigma_abs
