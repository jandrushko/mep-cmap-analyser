"""
mep_cmap.detection
~~~~~~~~~~~~~~~~~~
Public API for all signal detection and quantification.

This package is the single import point for the rest of the codebase.
No other script needs to know which submodule implements a given function —
they all import from here exactly as before.

Submodules
----------
defaults             : canonical parameter defaults (single source of truth)
dispatch             : the single method-name -> detector mapping
bootstrap_baseline   : shared pre-stim noise threshold computation
envelope_stats       : RMS envelope, baseline threshold, run-length calibration
tkeo                 : Teager-Kaiser energy operator preconditioning
onset_peak_fraction  : peak-fraction + slope backtracking onset detector
onset_bootstrap      : bootstrap peak-anchored backward scan onset detector
onset_bigoni         : derivative-based onset detector (Bigoni et al. 2022)
onset_rms_envelope   : RMS envelope + SD threshold, refined on a short window
onset_cusum          : CUSUM change-point onset detector
onset_boyles         : derivative-ratio detector (Boyles et al. 2026)
onset_methods_median : median across detectors + per-trial agreement metrics
offset_detection     : MEP offset (return to baseline) + precedence resolution
csp_detection        : cortical silent period bootstrap detector
quantification       : PTP, RMS, AUC, pre-stim RMS/PTP scalar metrics

Dispatcher
----------
detect_mep_onset()   : calls whichever onset method is configured in
                       preferences (onset_method), passing through all
                       keyword arguments to the active implementation.
                       Falls back to peak_fraction if the configured
                       method is unrecognised.
"""

# ── Re-export everything so existing imports remain unchanged ─────────────────

from .defaults import (                                               # noqa: F401
    DEFAULT_METHODS_MEDIAN_MEMBERS,
    DEFAULT_ONSET_METHOD,
    METHOD_ALIASES,
    DETECTION_DEFAULTS,
    OFFSET_DEFAULTS,
    ONSET_DEFAULTS,
    PTP_ANCHOR_DEFAULTS,
    PREF_KEY_ALIASES,
    TK_BACKED_DETECTION_KEYS,
    DETECTION_DEFAULTS_VERSION,
    SUPERSEDED_DEFAULTS,
    as_pref_defaults,
    migrate_detection_defaults,
    reset_detection_defaults,
    config_detection_kwargs,
    prefs_detection_snapshot,
    config_key_for,
    detector_params,
    pref_key_for,
)

from .bootstrap_baseline import compute_bootstrap_threshold          # noqa: F401

from .envelope_stats import (                                         # noqa: F401
    bootstrap_runlength_criterion,
    compute_envelope_baseline,
    compute_rms_envelope,
    find_sustained_run,
    passes_width_guard,
)

from .tkeo import apply_tkeo                                          # noqa: F401

from .onset_peak_fraction import detect_mep_onset_peak_fraction      # noqa: F401

from .onset_bootstrap import detect_mep_onset_bootstrap              # noqa: F401

from .onset_bigoni import detect_mep_onset_bigoni                    # noqa: F401
from .onset_bigoni_walkback import detect_mep_onset_bigoni_walkback  # noqa: F401

from .onset_rms_envelope import detect_mep_onset_rms_envelope        # noqa: F401
from .onset_cusum import detect_mep_onset_cusum                      # noqa: F401
from .onset_boyles import detect_mep_onset_boyles                    # noqa: F401
from .onset_methods_median import (                                        # noqa: F401
    METHODS_MEDIAN_DEFAULT_MEMBERS,
    compute_onset_agreement,
    detect_mep_onset_methods_median,
)

from .dispatch import dispatch_onset                                  # noqa: F401

from .offset_detection import (                                       # noqa: F401
    OFFSET_SOURCES,
    detect_mep_offset,
    offset_marker_field,
    resolve_mep_offset,
)

from .csp_detection import (                                          # noqa: F401
    CspSettings,
    detect_csp_bootstrap,
    detect_csp_for_trial,
)

from .quantification import (                                         # noqa: F401
    compute_ptp,
    compute_rms,
    compute_auc,
    compute_prestim_rms,
    compute_prestim_ptp,
)

# VERSION STAMP
# Bumped whenever detection behaviour changes, so an output file records which
# implementation produced it. v4 adds the envelope, CUSUM, consensus and offset
# detectors; no existing detector was modified, so v3 results reproduce exactly.
#
# v5 changes cortical silent period detection ONLY. Onset and MEP offset
# detection are untouched and reproduce v4 exactly. cSP durations WILL differ
# from v4 and the v4 values were wrong, in three ways:
#   * min_return_ms is now applied, so breakthrough EMG no longer truncates a
#     silent period at the first burst. Affected durations get LONGER.
#   * the envelope is reflection-padded rather than zero-padded, so the
#     baseline SD is no longer deflated by the segment edge.
#   * the run-length bootstrap resamples in blocks, so the stated significance
#     level is the one applied.
# Re-run any analysis whose cSP values are being reported.
DETECTION_VERSION = "2026-modular-v5"

# ── Method registry ───────────────────────────────────────────────────────────
# Maps preference key → callable
# Add new methods here — the dispatcher and preferences UI pick them up
# automatically via ONSET_METHOD_LABELS.

# Saved sessions and preferences written by v1.3.3 name this method
# "consensus". It was renamed because "consensus" implies the agreed value is
# the correct one, which is precisely the inference the method's own outputs
# warn against. The old key still resolves so those files keep working; it is
# absent from the labels, so it never appears as a choice in the interface.
_METHOD_ALIASES = {"consensus": "methods_median"}


_METHOD_REGISTRY = {
    "peak_fraction":   detect_mep_onset_peak_fraction,
    "bootstrap":       detect_mep_onset_bootstrap,
    "bigoni":          detect_mep_onset_bigoni,
    "bigoni_walkback": detect_mep_onset_bigoni_walkback,
    "rms_envelope":    detect_mep_onset_rms_envelope,
    "cusum":           detect_mep_onset_cusum,
    "boyles":          detect_mep_onset_boyles,
    "methods_median":       detect_mep_onset_methods_median,
}

# Human-readable labels for the preferences UI
# Keys must match _METHOD_REGISTRY exactly.
ONSET_METHOD_LABELS = {
    "peak_fraction":   "Peak Fraction",
    "bootstrap":       "Bootstrap Threshold (legacy)",
    "bigoni":          "Derivative-based (Bigoni et al. 2022)",
    "bigoni_walkback": "Derivative-based + Walkback (Modified Bigoni)",
    "rms_envelope":    "RMS Envelope + SD Threshold",
    "cusum":           "CUSUM Change-point",
    "boyles":          "Derivative Ratio (Boyles et al. 2026)",
    "methods_median":  "Median across methods",
}

# Short names for figures and plot axes. The full labels are too long for an
# axis tick or a legend entry, and the registry keys are not written for a
# reader. The CSV Method column keeps the KEY, which is stable for scripting;
# only figures use these.
ONSET_METHOD_SHORT_LABELS = {
    "peak_fraction":   "Peak fraction",
    "bootstrap":       "Bootstrap (legacy)",
    "bigoni":          "Bigoni",
    "bigoni_walkback": "Bigoni + walkback",
    "rms_envelope":    "RMS envelope",
    "cusum":           "CUSUM",
    "boyles":          "Derivative ratio",
    "methods_median":  "Median across methods",
}

# Shown beneath the method selector in Preferences. Kept next to the labels so
# a new method cannot be registered without a note on when to choose it.
ONSET_METHOD_HINTS = {
    "peak_fraction":
        "Relative amplitude threshold. Clean, high-amplitude MEPs on a quiet "
        "baseline.",
    "bootstrap":
        "Retained so analyses run on v1.3.x and earlier reproduce exactly. Its "
        "threshold is clipped to a multiple of the baseline mean, which places "
        "onsets systematically early; prefer RMS Envelope for new work.",
    "bigoni":
        "Derivative run length; assumes nothing about baseline level. The "
        "safest general-purpose choice, and the default.",
    "bigoni_walkback":
        "As above, with the onset walked back to the point of departure from "
        "baseline. Use when the plain method lands mid-rise.",
    "rms_envelope":
        "Smoothed amplitude against an SD-scaled baseline threshold, refined "
        "on a short window. Most precise on a quiet baseline; like all "
        "threshold methods it degrades when background EMG is high.",
    "cusum":
        "Detects the change in mean rather than the crossing of a level, so "
        "latency is intrinsically unbiased. Tolerant of raised background EMG.",
    "boyles":
        "Compares the slope just after each candidate with the slope just "
        "before it, working back from the first peak. Methodologically "
        "independent of the others, which makes it a useful member method. "
        "It has the most parameters of any method here, two of them scaled by "
        "the trial's own peak-to-trough interval, and it cannot return an onset "
        "earlier than the first peak. Needs a condition average, which the "
        "pipeline supplies.",
    "methods_median":
        "Runs several detectors and reports the median of those that find an "
        "onset. The median is not a verdict on which method is right \u2014 it is "
        "the middle value, chosen because it resists one stray member. Its "
        "value is mostly that the spread between members is reported as "
        "Onset_Disagreement(ms), which flags the trials worth reviewing.",
}


def detect_mep_onset(signal, fs, method=None, **kwargs):
    """
    Dispatcher: call the configured (or explicitly requested) onset method.

    Parameters
    ----------
    signal  : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs      : float           sampling frequency in Hz
    method  : str or None     override the preference setting for this call.
              One of the keys of ONSET_METHOD_LABELS.
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
            method = DEFAULT_ONSET_METHOD

    method = _METHOD_ALIASES.get(method, method)
    fn = _METHOD_REGISTRY.get(method, _METHOD_REGISTRY[DEFAULT_ONSET_METHOD])
    return fn(signal, fs, **kwargs)
