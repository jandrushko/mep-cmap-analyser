"""
mep_cmap.detection.dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The one place that maps an onset method name onto a detector call.

Why this module exists
----------------------
The same dispatch used to be written out three times: once in
``pipeline._detect_onset_dispatch``, once inline in ``inspector.py``, and once
in ``detection.detect_mep_onset``. They drifted, as duplicated logic does, and
the drift was invisible because each copy independently produced plausible
numbers:

* The inspector's copy knew four methods. Selecting a fifth silently fell
  through to peak-fraction, so a re-detection in the inspector could use a
  different algorithm than the pipeline that produced the value it replaced.

* The inspector's copy passed no ``min_peak_amplitude``, so it used the
  detector's built-in 0.05 while the pipeline passed the configured value. An
  analyst who raised the amplitude gate got it honoured during analysis and
  ignored during review.

* The inspector's copy passed no ``peak_frac`` or ``slope_threshold`` either.
  The peak-fraction detector defaults ``slope_threshold`` to 0.05 while
  ``PipelineConfig`` defaults it to 0.08, so the two paths disagreed even at
  factory settings.

None of these raised an error. Routing every caller through one function makes
the class of bug structurally impossible rather than merely fixed, and
``tests/test_dispatch_parity.py`` asserts the pipeline and the inspector return
identical onsets for the same trace.

Parameters travel as a plain dict keyed by ``PipelineConfig`` field name — the
same dict ``detection.defaults.detector_params`` produces — so adding a method
parameter never changes a function signature anywhere.

  * dispatch_onset
"""

from .defaults import (DETECTION_DEFAULTS, DEFAULT_ONSET_METHOD,
                       METHOD_ALIASES as _METHOD_ALIASES)
from .onset_bigoni import detect_mep_onset_bigoni
from .onset_boyles import detect_mep_onset_boyles
from .onset_bigoni_walkback import detect_mep_onset_bigoni_walkback
from .onset_bootstrap import detect_mep_onset_bootstrap
from .onset_methods_median import detect_mep_onset_methods_median
from .onset_cusum import detect_mep_onset_cusum
from .onset_peak_fraction import detect_mep_onset_peak_fraction
from .onset_rms_envelope import detect_mep_onset_rms_envelope
from .artefact import (settled_search_start, on_window_edge, lower_bound_ms,
                       plateau_corrected, artefact_at_start)


def dispatch_onset(signal, fs, params=None, *,
                   pre_ms,
                   search_start_ms,
                   search_end_ms,
                   min_latency_ms=None,
                   max_latency_ms=None,
                   method=None,
                   template=None):
    """Run the configured onset detector, kept out of the stimulus artefact.

    The detector itself is chosen and run by ``_dispatch_onset_raw`` below.
    Around it, for every method alike (detection/artefact.py):

    * the search starts where the trace has settled after the stimulus
      artefact, never earlier than ``search_start_ms``;
    * a latency on the search's first sample is returned as None: the signal
      was already above the detector's threshold when the search began, so
      the window edge is not an onset.

    Same signature and return as before.
    """
    # Only where an artefact is present at the search start (see
    # detection/artefact.artefact_at_start); otherwise exactly as in 1.4.14.
    # The methods-median applies these rules itself, inside
    # compute_onset_agreement, so it goes straight through: applied here as
    # well, its inner pass saw the already-corrected trace, found no artefact,
    # skipped the quorum and edge rule, and let an artefact onset through.
    _p = dict(DETECTION_DEFAULTS)
    if params:
        _p.update({k: v for k, v in params.items() if v is not None})
    _m = _METHOD_ALIASES.get(method or _p.get("onset_method", DEFAULT_ONSET_METHOD),
                             method or _p.get("onset_method", DEFAULT_ONSET_METHOD))
    _artefact = (_m != "methods_median"
                 and artefact_at_start(signal, fs, pre_ms, search_start_ms))
    if not _artefact:
        return _dispatch_onset_raw(signal, fs, params, pre_ms=pre_ms,
                                   search_start_ms=search_start_ms,
                                   search_end_ms=search_end_ms,
                                   min_latency_ms=min_latency_ms,
                                   max_latency_ms=max_latency_ms,
                                   method=method, template=template)
    start = settled_search_start(signal, fs, pre_ms, search_start_ms,
                                 search_end_ms)
    # A flat post-artefact plateau off baseline moved to baseline, for the
    # detector only (a copy; amplitude is never measured on it).
    sig = plateau_corrected(signal, fs, pre_ms, start)
    lat = _dispatch_onset_raw(sig, fs, params, pre_ms=pre_ms,
                              search_start_ms=start,
                              search_end_ms=search_end_ms,
                              min_latency_ms=min_latency_ms,
                              max_latency_ms=max_latency_ms,
                              method=method, template=template)
    # The search start only, not the profile floor: see compute_onset_agreement.
    if on_window_edge(lat, start, fs):
        return None
    return lat


def _dispatch_onset_raw(signal, fs, params=None, *,
                        pre_ms,
                        search_start_ms,
                        search_end_ms,
                        min_latency_ms=None,
                        max_latency_ms=None,
                        method=None,
                        template=None):
    """
    Run the configured onset detector on one trace.

    Parameters
    ----------
    signal          : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs              : float  sampling frequency in Hz
    params          : dict or None  detection parameters keyed by
                      PipelineConfig field name. Missing keys fall back to
                      detection.defaults; None means all defaults.
    pre_ms          : float  ms of pre-stimulus data in ``signal``.

                      This MUST be the pre-stimulus length of the array being
                      passed, not a nominal baseline setting. The pipeline
                      extracts trials with ``cfg.pre_ms`` of lead-in while the
                      inspector uses ``prestim_ms``; passing the wrong one
                      mislocates the stimulus and every detector searches the
                      wrong region, returning None even for an obvious MEP.
    search_start_ms : float  ms post-stim to begin searching
    search_end_ms   : float  ms post-stim to stop searching
    min_latency_ms  : float or None  physiological floor (ms post-stim)
    max_latency_ms  : float or None  physiological ceiling (ms post-stim)
    method          : str or None  override ``params["onset_method"]``
    template        : 1-D np.ndarray or None  condition average/median waveform,
                      the same length as ``signal``.

                      Only the derivative-ratio method uses this, and only for
                      its peak-jitter gate. It is a keyword on the shared
                      dispatch rather than a parameter of that method because
                      it is DATA, not a setting: it cannot travel in the params
                      dict alongside scalars, and adding a second dispatch
                      entry point for one method is how the duplicated dispatch
                      this module replaced came about. Callers with no
                      condition average pass None and the gate is skipped.

    Returns
    -------
    latency_ms : float, or None if no confident onset was found

    Notes
    -----
    An unrecognised method falls through to peak-fraction, matching
    ``detection.detect_mep_onset``. That path is reachable only from a saved
    session naming a method this build does not provide.
    """
    p = dict(DETECTION_DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    if method is None:
        method = p.get("onset_method", DEFAULT_ONSET_METHOD)
    # v1.3.3 called this method "consensus"; saved sessions still say so.
    method = _METHOD_ALIASES.get(method, method)

    amp = p["min_peak_amplitude"]

    if method == "bootstrap":
        return detect_mep_onset_bootstrap(
            signal, fs,
            pre_ms=pre_ms,
            peak_search_start_ms=search_start_ms,
            peak_search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            criterion=p["onset_bootstrap_crit"],
            n_boot=p["onset_bootstrap_n"],
        )

    if method == "bigoni":
        return detect_mep_onset_bigoni(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            smooth_window_ms=p["onset_bigoni_smooth_ms"],
            min_run_ms=p["onset_bigoni_min_run_ms"],
        )

    if method == "bigoni_walkback":
        return detect_mep_onset_bigoni_walkback(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            smooth_window_ms=p["onset_bigoni_smooth_ms"],
            min_run_ms=p["onset_bigoni_min_run_ms"],
            walkback_sd_mult=p["onset_bigoni_walkback_sd"],
        )

    if method == "rms_envelope":
        return detect_mep_onset_rms_envelope(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            env_window_ms=p["onset_env_window_ms"],
            criterion=p["onset_env_criterion"],
            significance=p["onset_env_significance"],
            n_boot=p["onset_env_n_boot"],
            min_run_ms=p["onset_env_min_run_ms"],
            min_response_ms=p["onset_env_min_response_ms"],
            use_tkeo=p["onset_env_tkeo"],
            causal_window=p["onset_env_causal"],
            refine_on_raw=p["onset_env_refine"],
            refine_window_ms=p["onset_env_refine_window_ms"],
            refine_sd_mult=p["onset_env_refine_sd"],
            refine_sustain_ms=p["onset_env_refine_sustain_ms"],
        )

    if method == "cusum":
        return detect_mep_onset_cusum(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            k_mult=p["onset_cusum_k"],
            h_mult=p["onset_cusum_h"],
            max_accum_ms=p["onset_cusum_max_accum_ms"],
            min_response_ms=p["onset_cusum_min_response_ms"],
            use_tkeo=p["onset_cusum_tkeo"],
        )

    if method == "boyles":
        return detect_mep_onset_boyles(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            template=template,
            block_ms=p["onset_boyles_block_ms"],
            baseline_start_ms=p["onset_boyles_baseline_start_ms"],
            baseline_end_ms=p["onset_boyles_baseline_end_ms"],
            amplitude_gate=p["onset_boyles_amplitude_gate"],
            peak_jitter_ms=p["onset_boyles_peak_jitter_ms"],
            peak_window_length=p["onset_boyles_peak_window_length"],
            ratio_cutoff=p["onset_boyles_ratio_cutoff"],
            boyles_max_latency_ms=p["onset_boyles_max_latency_ms"],
            deriv_check_ms=p["onset_boyles_deriv_check_ms"],
            deriv_check_duty=p["onset_boyles_deriv_check_duty"],
            base_deriv_sds=p["onset_boyles_base_deriv_sds"],
            deriv_check_window_length=p["onset_boyles_deriv_window_length"],
            literal=p["onset_boyles_literal"],
        )

    if method == "methods_median":
        return detect_mep_onset_methods_median(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            methods=p["onset_methods_median_members"],
            params=p,
            template=template,
        )

    return detect_mep_onset_peak_fraction(
        signal, fs,
        pre_ms=pre_ms,
        poststim_start_ms=search_start_ms,
        poststim_end_ms=search_end_ms,
        peak_frac=p["peak_fraction"],
        min_consecutive=5,
        min_peak_amplitude=amp,
        slope_threshold=p["slope_threshold"],
    )
