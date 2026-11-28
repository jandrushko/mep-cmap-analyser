"""
mep_cmap.detection.onset_methods_median
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Median across several onset detectors, and per-trial agreement metrics.

Two separate entry points, deliberately not merged:

``detect_mep_onset_methods_median``
    Returns the median of the detectors that found an onset — a single float,
    so it registers as an ordinary onset method alongside the others.

``compute_onset_agreement``
    Returns the full per-method breakdown plus dispersion statistics. This is
    computed independently of which method is SELECTED, so disagreement
    columns are available even when the analysis runs plain Bigoni.

The agreement metrics are arguably the more useful product. The recurring
practical question in MEP analysis is not "what is the single best estimator"
— no estimator wins across resting and active paradigms, SNRs and muscles —
but "which trials must a human actually look at". Trials where independent
detectors built on different principles (derivative run length, envelope
threshold, cumulative change point, relative peak fraction) converge to within
a millisecond need no review; trials where they scatter over 10 ms do. Spread
across methods is a direct, cheap proxy for that, and it is reportable.

Interpretation caveats
----------------------
Low spread means the detectors agree, NOT that they are correct: methods
sharing an assumption can be wrong together. The two Bigoni variants in
particular are not independent — the walkback variant starts from the plain
variant's answer — so their agreement is partly structural. Spread is a
triage signal, not a validation.

Cost
----
Median across methods runs every member detector on every trial. The default member set
excludes ``bootstrap``, which recomputes a 500-iteration bootstrap inside each
call and dominates the runtime; it can be added explicitly where its behaviour
is specifically wanted.

  * detect_mep_onset_methods_median
  * compute_onset_agreement
  * METHODS_MEDIAN_DEFAULT_MEMBERS
"""

from collections import namedtuple

import numpy as np

from .onset_bigoni import detect_mep_onset_bigoni
from .onset_bigoni_walkback import detect_mep_onset_bigoni_walkback
from .onset_bootstrap import detect_mep_onset_bootstrap
from .onset_boyles import detect_mep_onset_boyles
from .onset_cusum import detect_mep_onset_cusum
from .onset_peak_fraction import detect_mep_onset_peak_fraction
from .onset_rms_envelope import detect_mep_onset_rms_envelope
from .artefact import (settled_search_start, on_window_edge, lower_bound_ms,
                       plateau_corrected, artefact_at_start)

# Five fast, methodologically distinct detectors. Odd count so the median is a
# member of the set rather than an average of two.
METHODS_MEDIAN_DEFAULT_MEMBERS = (
    "bigoni",
    "bigoni_walkback",
    "rms_envelope",
    "cusum",
    "peak_fraction",
)

OnsetAgreement = namedtuple(
    "OnsetAgreement",
    "per_method consensus_ms spread_ms iqr_ms n_detected n_attempted")


# ── Adapter layer ─────────────────────────────────────────────────────────────
# Each detector was written with its own keyword names (``poststim_start_ms``
# vs ``search_start_ms`` vs ``peak_search_start_ms``). Rather than renaming
# parameters in the existing detectors — which would change their public
# signatures and break reproducibility of saved sessions — a canonical set of
# common keys is translated here.
#
# Canonical keys:
#   pre_ms, search_start_ms, search_end_ms,
#   min_latency_ms, max_latency_ms, min_peak_amplitude


def _call_bigoni(signal, fs, common, params):
    return detect_mep_onset_bigoni(
        signal, fs,
        pre_ms=common["pre_ms"],
        search_start_ms=common["search_start_ms"],
        search_end_ms=common["search_end_ms"],
        min_latency_ms=common["min_latency_ms"],
        max_latency_ms=common["max_latency_ms"],
        min_peak_amplitude=common["min_peak_amplitude"],
        smooth_window_ms=params.get("onset_bigoni_smooth_ms", 0.5),
        min_run_ms=params.get("onset_bigoni_min_run_ms", 0.5),
    )


def _call_bigoni_walkback(signal, fs, common, params):
    return detect_mep_onset_bigoni_walkback(
        signal, fs,
        pre_ms=common["pre_ms"],
        search_start_ms=common["search_start_ms"],
        search_end_ms=common["search_end_ms"],
        min_latency_ms=common["min_latency_ms"],
        max_latency_ms=common["max_latency_ms"],
        min_peak_amplitude=common["min_peak_amplitude"],
        smooth_window_ms=params.get("onset_bigoni_smooth_ms", 0.5),
        min_run_ms=params.get("onset_bigoni_min_run_ms", 0.5),
        walkback_sd_mult=params.get("onset_bigoni_walkback_sd", 1.0),
    )


def _call_peak_fraction(signal, fs, common, params):
    # This detector has no latency-bound parameters, so the physiological
    # window is enforced here instead. Without this the peak-fraction result
    # would be the only unbounded member and would inflate the spread.
    lat = detect_mep_onset_peak_fraction(
        signal, fs,
        pre_ms=common["pre_ms"],
        poststim_start_ms=common["search_start_ms"],
        poststim_end_ms=common["search_end_ms"],
        peak_frac=params.get("peak_fraction", 0.15),
        min_consecutive=params.get("peak_min_consecutive", 5),
        min_peak_amplitude=common["min_peak_amplitude"],
        slope_threshold=params.get("slope_threshold", 0.05),
    )
    if lat is None:
        return None
    lo = common["min_latency_ms"]
    hi = common["max_latency_ms"]
    if lo is not None and lat < lo:
        return None
    if hi is not None and lat > hi:
        return None
    return lat


def _call_bootstrap(signal, fs, common, params):
    return detect_mep_onset_bootstrap(
        signal, fs,
        pre_ms=common["pre_ms"],
        peak_search_start_ms=common["search_start_ms"],
        peak_search_end_ms=common["search_end_ms"],
        min_latency_ms=common["min_latency_ms"],
        max_latency_ms=common["max_latency_ms"],
        min_peak_amplitude=common["min_peak_amplitude"],
        criterion=params.get("onset_bootstrap_crit", 1.96),
        n_boot=params.get("onset_bootstrap_n", 500),
    )


def _call_rms_envelope(signal, fs, common, params):
    return detect_mep_onset_rms_envelope(
        signal, fs,
        pre_ms=common["pre_ms"],
        search_start_ms=common["search_start_ms"],
        search_end_ms=common["search_end_ms"],
        min_latency_ms=common["min_latency_ms"],
        max_latency_ms=common["max_latency_ms"],
        min_peak_amplitude=common["min_peak_amplitude"],
        env_window_ms=params.get("onset_env_window_ms", 5.0),
        criterion=params.get("onset_env_criterion", 2.5),
        significance=params.get("onset_env_significance", 0.99),
        n_boot=params.get("onset_env_n_boot", 500),
        use_tkeo=params.get("onset_env_tkeo", False),
        causal_window=params.get("onset_env_causal", False),
        refine_on_raw=params.get("onset_env_refine", True),
        refine_window_ms=params.get("onset_env_refine_window_ms", 1.0),
        refine_sd_mult=params.get("onset_env_refine_sd", 1.0),
        refine_sustain_ms=params.get("onset_env_refine_sustain_ms", 1.0),
    )


def _call_boyles(signal, fs, common, params):
    # The only member needing a condition average. It arrives through `common`
    # rather than `params` because it is data, not a setting.
    return detect_mep_onset_boyles(
        signal, fs,
        pre_ms=common["pre_ms"],
        search_start_ms=common["search_start_ms"],
        search_end_ms=common["search_end_ms"],
        min_latency_ms=common["min_latency_ms"],
        max_latency_ms=common["max_latency_ms"],
        min_peak_amplitude=common["min_peak_amplitude"],
        template=common.get("template"),
        block_ms=params.get("onset_boyles_block_ms", 2.5),
        baseline_start_ms=params.get("onset_boyles_baseline_start_ms", 100.0),
        baseline_end_ms=params.get("onset_boyles_baseline_end_ms", 1.0),
        amplitude_gate=params.get("onset_boyles_amplitude_gate", 1.1),
        peak_jitter_ms=params.get("onset_boyles_peak_jitter_ms", 15.0),
        peak_window_length=params.get("onset_boyles_peak_window_length", 1.75),
        ratio_cutoff=params.get("onset_boyles_ratio_cutoff", 0.85),
        boyles_max_latency_ms=params.get("onset_boyles_max_latency_ms", 35.0),
        deriv_check_ms=params.get("onset_boyles_deriv_check_ms", 2.0),
        deriv_check_duty=params.get("onset_boyles_deriv_check_duty", 0.75),
        base_deriv_sds=params.get("onset_boyles_base_deriv_sds", 1.5),
        deriv_check_window_length=params.get("onset_boyles_deriv_window_length", 2.0),
        literal=params.get("onset_boyles_literal", False),
    )


def _call_cusum(signal, fs, common, params):
    return detect_mep_onset_cusum(
        signal, fs,
        pre_ms=common["pre_ms"],
        search_start_ms=common["search_start_ms"],
        search_end_ms=common["search_end_ms"],
        min_latency_ms=common["min_latency_ms"],
        max_latency_ms=common["max_latency_ms"],
        min_peak_amplitude=common["min_peak_amplitude"],
        k_mult=params.get("onset_cusum_k", 0.5),
        h_mult=params.get("onset_cusum_h", 20.0),
        use_tkeo=params.get("onset_cusum_tkeo", False),
    )


_ADAPTERS = {
    "bigoni": _call_bigoni,
    "bigoni_walkback": _call_bigoni_walkback,
    "peak_fraction": _call_peak_fraction,
    "bootstrap": _call_bootstrap,
    "rms_envelope": _call_rms_envelope,
    "cusum": _call_cusum,
    "boyles": _call_boyles,
}


# Members that count as ONE vote towards the quorum. Empty: counting the
# walkback variant with plain Bigoni (it starts from Bigoni's answer) was
# tried and blanked dozens of TMS MEPs found by Bigoni, walkback and one
# other member -- a majority of five, not of four families -- while not
# correcting the M-wave trials it was aimed at. Kept as the place to say so.
_VOTE_FAMILY = {}


def consensus_onset(per_method):
    """Median of the members that found an onset, if a majority of votes did.

    Votes are counted per independent family (see ``_VOTE_FAMILY``); the
    median itself is still taken over every member that found an onset.
    Below a majority of families the trial has no onset: the median of a
    minority is not a consensus. On an FDI recruitment curve three of five
    members reported the window start on trials with no M-wave at all; the
    window-edge rule drops those, and this is the backstop for whatever a
    minority still finds. Returns None or the median, rounded to 0.01 ms.
    """
    fams_all, fams_hit, vals = set(), set(), []
    for name, v in (per_method or {}).items():
        fam = _VOTE_FAMILY.get(name, name)
        fams_all.add(fam)
        if v is not None:
            fams_hit.add(fam)
            vals.append(float(v))
    if not vals or len(fams_hit) < len(fams_all) // 2 + 1:
        return None
    return round(float(np.median(vals)), 2)


def compute_onset_agreement(
        signal, fs, *,
        pre_ms=100,
        search_start_ms=5,
        search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        methods=None,
        params=None,
        template=None):
    """
    Run several onset detectors on one trial and summarise their agreement.

    Parameters
    ----------
    signal, fs, pre_ms, search_start_ms, search_end_ms,
    min_latency_ms, max_latency_ms, min_peak_amplitude
        Canonical detection arguments, translated per detector.
    methods : sequence of str or None
        Member detectors; defaults to ``METHODS_MEDIAN_DEFAULT_MEMBERS``. Unknown
        names are ignored rather than raising, so a stale saved session cannot
        abort a run.
    params : dict or None
        Method-specific tuning values, keyed exactly as the corresponding
        ``PipelineConfig`` fields. Missing keys fall back to each detector's
        own defaults.

    Returns
    -------
    OnsetAgreement(per_method, consensus_ms, spread_ms, iqr_ms,
                   n_detected, n_attempted)

        per_method   : dict method name -> latency_ms or None
        consensus_ms : median of the non-None latencies, or None
        spread_ms    : max - min of the non-None latencies, or None if fewer
                       than two detected
        iqr_ms       : interquartile range of the non-None latencies; more
                       robust than spread when one member is an outlier
        n_detected   : how many members returned a latency
        n_attempted  : how many members ran
    """
    methods = tuple(METHODS_MEDIAN_DEFAULT_MEMBERS if methods is None else methods)
    params = {} if params is None else dict(params)
    # The artefact rules (detection/artefact.py) apply only when an artefact
    # is actually present where the search starts. Everywhere else -- the
    # ordinary TMS trial -- this is exactly the 1.4.14 computation.
    _artefact = artefact_at_start(signal, fs, pre_ms, search_start_ms)
    if _artefact:
        # Past the artefact still settling: the same start for every member.
        search_start_ms = settled_search_start(signal, fs, pre_ms,
                                               search_start_ms, search_end_ms)
        # A flat post-artefact plateau moved to baseline, for the detectors
        # only (a copy; every amplitude is measured on the original).
        sig = plateau_corrected(signal, fs, pre_ms, search_start_ms)
    else:
        sig = signal
    # The edge rule guards the SEARCH START only, where the artefact sits. Not
    # the profile's floor (min_latency_ms): a TMS MEP whose first phase begins
    # right at that floor was found there correctly by several members, and
    # dropping them moved the median a phase late.
    _lower = search_start_ms if _artefact else None

    common = {
        "pre_ms": pre_ms,
        "search_start_ms": search_start_ms,
        "search_end_ms": search_end_ms,
        "min_latency_ms": min_latency_ms,
        "max_latency_ms": max_latency_ms,
        "min_peak_amplitude": min_peak_amplitude,
        "template": template,
    }

    per_method = {}
    attempted = 0
    for name in methods:
        fn = _ADAPTERS.get(name)
        if fn is None:
            continue
        attempted += 1
        try:
            per_method[name] = fn(sig, fs, common, params)
        except Exception:
            # One detector failing on a pathological trial must not lose the
            # information the others provide.
            per_method[name] = None
        # A latency on the search's first sample is the window edge, not an
        # onset (detection/artefact.py): the signal was already above this
        # detector's threshold when the search began.
        if on_window_edge(per_method[name], _lower, fs):
            per_method[name] = None

    vals = [v for v in per_method.values() if v is not None]
    if not vals:
        return OnsetAgreement(per_method, None, None, None, 0, attempted)

    arr = np.asarray(vals, dtype=float)
    # Quorum only where an artefact was present; elsewhere the plain median
    # of whatever was found, as in 1.4.14. A TMS onset found by a minority is
    # kept: n_detected travels with it, so the trial can be flagged for
    # review instead of silently losing a clear MEP.
    methods_median = (consensus_onset(per_method) if _artefact
                      else round(float(np.median(arr)), 2))
    if arr.size < 2:
        return OnsetAgreement(per_method, methods_median, None, None,
                              int(arr.size), attempted)

    spread = round(float(arr.max() - arr.min()), 2)
    iqr = round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 2)
    return OnsetAgreement(per_method, methods_median, spread, iqr,
                          int(arr.size), attempted)


def detect_mep_onset_methods_median(signal, fs, **kwargs):
    """
    Median across methods onset latency: median across member detectors.

    Accepts the same canonical keywords as ``compute_onset_agreement``, plus
    ``methods`` and ``params``. Returns a single float (or None) so that it
    slots into the onset method registry like any other detector.
    """
    return compute_onset_agreement(signal, fs, **kwargs).consensus_ms
