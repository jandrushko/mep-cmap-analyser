"""
mep_cmap.detection
~~~~~~~~~~~~~~~~~~
Public API for all signal detection and quantification.

This package is the single import point for the rest of the codebase.
No other script needs to know which submodule implements a given function —
they all import from here exactly as before.

Submodules
----------
bootstrap_baseline   : shared pre-stim noise threshold computation
onset_peak_fraction  : peak-fraction + slope backtracking onset detector
onset_bootstrap      : bootstrap peak-anchored backward scan onset detector
onset_bigoni         : derivative-based onset detector (Bigoni et al. 2022)
csp_detection        : cortical silent period bootstrap detector
quantification       : PTP, AUC, pre-stim RMS/PTP scalar metrics

Dispatcher
----------
detect_mep_onset()   : calls whichever onset method is configured in
                       preferences (onset_method), passing through all
                       keyword arguments to the active implementation.
                       Falls back to peak_fraction if the configured
                       method is unrecognised.
"""

# ── Re-export everything so existing imports remain unchanged ─────────────────

from .bootstrap_baseline import compute_bootstrap_threshold          # noqa: F401

from .onset_peak_fraction import detect_mep_onset_peak_fraction      # noqa: F401

from .onset_bootstrap import detect_mep_onset_bootstrap              # noqa: F401

from .onset_bigoni import detect_mep_onset_bigoni                    # noqa: F401
from .onset_bigoni_walkback import detect_mep_onset_bigoni_walkback  # noqa: F401

from .csp_detection import detect_csp_bootstrap                      # noqa: F401

from .quantification import (                                         # noqa: F401
    compute_ptp,
    compute_auc,
    compute_prestim_rms,
    compute_prestim_ptp,
)

# VERSION STAMP
DETECTION_VERSION = "2025-modular-v3"

# ── Method registry ───────────────────────────────────────────────────────────
# Maps preference key → callable
# Add new methods here — the dispatcher and preferences UI pick them up
# automatically via ONSET_METHOD_LABELS.

_METHOD_REGISTRY = {
    "peak_fraction": detect_mep_onset_peak_fraction,
    "bootstrap":     detect_mep_onset_bootstrap,
    "bigoni":        detect_mep_onset_bigoni,
    "bigoni_walkback": detect_mep_onset_bigoni_walkback,
}

# Human-readable labels for the preferences UI
# Keys must match _METHOD_REGISTRY exactly.
ONSET_METHOD_LABELS = {
    "peak_fraction": "Peak Fraction",
    "bootstrap":     "Bootstrap Threshold",
    "bigoni":        "Derivative-based (Bigoni et al. 2022)",
    "bigoni_walkback": "Derivative-based + Walkback (Modified Bigoni)",
}


def detect_mep_onset(signal, fs, method=None, **kwargs):
    """
    Dispatcher: call the configured (or explicitly requested) onset method.

    Parameters
    ----------
    signal  : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs      : float           sampling frequency in Hz
    method  : str or None     override the preference setting for this call.
              One of: 'peak_fraction', 'bootstrap', 'bigoni'.
              If None, reads from preferences.prefs.onset_method.
    **kwargs: passed through to the active detection function.

    Returns
    -------
    latency_ms : float or None
    """
    if method is None:
        try:
            from ..preferences import prefs
            method = prefs.onset_method
        except Exception:
            method = "peak_fraction"

    fn = _METHOD_REGISTRY.get(method, detect_mep_onset_peak_fraction)
    return fn(signal, fs, **kwargs)
