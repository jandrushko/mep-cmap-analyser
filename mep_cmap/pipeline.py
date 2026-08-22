"""
mep_cmap.pipeline
~~~~~~~~~~~~~~~~~
Per-file analysis pipeline.

  • PipelineConfig                   — dataclass bundling all analysis settings
  • pipeline_load_file               — load EMG + stim times, apply crop
  • pipeline_apply_filters           — full filter chain
  • pipeline_extract_segments        — trial windowing
  • pipeline_detect_outliers         — z-score flagging
  • pipeline_review_outliers         — interactive review callback
  • pipeline_quantify_segments       — per-trial PTP / latency / CSP / AUC
  • pipeline_compute_pooled_stats    — pooled z-scores and detrending
  • pipeline_bootstrap_comparisons   — pairwise bootstrap comparisons
  • pipeline_write_outputs           — CSV writing
  • pipeline_generate_plots          — figure generation
  • run_pipeline                     — top-level orchestrator
"""

import os
import gc
import glob
import json
import itertools
import pathlib
import re as _re
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.backends.backend_agg
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy.stats import zscore
from numpy.random import default_rng

from .compat import _np_trapz, _np_ptp
from .bids import _sanitise_bids_label, StudyMetadata
from .utils import _add_time_and_digmark
from .io import extract_emg_waveform_and_fs, extract_stim_times
from .filters import adaptive_mains_cancel, design_notch_sos
from .detection     import (compute_ptp, compute_rms, compute_auc,
                            compute_prestim_rms,
                            detect_mep_onset_peak_fraction,
                             detect_mep_onset_bootstrap,
                             detect_mep_onset_bigoni,
                             detect_mep_onset_bigoni_walkback,
                             detect_mep_onset_rms_envelope,
                             detect_mep_onset_cusum,
                             detect_mep_onset_methods_median,
                             compute_onset_agreement,
                             dispatch_onset,
                             resolve_mep_offset,
                             detector_params,
                             config_detection_kwargs,
                             DEFAULT_ONSET_METHOD,
                             DETECTION_DEFAULTS,
                             compute_bootstrap_threshold)
from .normalisation import (compute_mmax, apply_normalisation,
                            apply_emg_compensation, EXCLUDED_DECISIONS)

def clamp_config_to_epoch_bounds(cfg, bounds):
    """Shrink a config's time windows to what a pre-epoched file contains.

    ``bounds`` is the (pre_ms, post_ms) returned by io.get_epoch_bounds(), or
    None for continuous formats — in which case nothing is changed.

    Why this is mandatory rather than advisory
    ------------------------------------------
    Both failure modes here are silent, which is what makes the clamp
    non-optional.

    On the pre-stimulus side, ``pre_start`` is clamped with max(0, ...), which
    only protects the very first trial.  For every other trial an over-long
    ``prestim_ms`` walks backwards out of its own epoch.  With the 100 ms
    default against a 25 ms epoch lead-in, three quarters of that window falls
    outside the trial — and it is the baseline that sets the bootstrap onset
    threshold and the RMS outlier gate.

    On the post-stimulus side one might expect the completeness guard in
    pipeline_extract_segments to reject an over-long window.  It does not: a
    pre-epoched reader supplies guard-band padding between epochs, so the
    window is filled and every trial passes.  Measured directly, post_ms=400
    against a 100 ms epoch keeps 100/100 trials rather than rejecting them.
    The guard prevents contamination by *neighbouring trials*, but it cannot
    stop an unclamped window from measuring padding and reporting it as
    signal.  Only the clamp does that.

    Returns
    -------
    (cfg, changes) : the config (mutated in place) and a list of
                     (field, old, new) tuples describing what was reduced, for
                     the caller to report to the analyst.
    """
    if not bounds:
        return cfg, []
    pre_avail, post_avail = float(bounds[0]), float(bounds[1])
    changes = []

    # Accept either a PipelineConfig-style object or the plain params dict the
    # GUI snapshots before starting the worker thread.
    is_map = isinstance(cfg, dict)
    _get = (lambda k, d=None: cfg.get(k, d)) if is_map else (
        lambda k, d=None: getattr(cfg, k, d))

    def _set(k, v):
        if is_map:
            cfg[k] = v
        else:
            setattr(cfg, k, v)

    # The pre-stimulus baseline may additionally be pushed earlier by a
    # per-stim-type gap, so the gap has to come out of the available lead-in.
    try:
        max_gap = max([float(v) for v in (_get('gap_ms_map') or {}).values()]
                      or [0.0])
    except Exception:
        max_gap = 0.0

    for field, limit in (('pre_ms', pre_avail),
                         ('post_ms', post_avail),
                         ('prestim_ms', max(pre_avail - max_gap, 0.0))):
        old = _get(field)
        if old is None:
            continue
        new = int(min(float(old), limit))
        if new != old:
            _set(field, new)
            changes.append((field, old, new))

    # Per-stimulus-type windows are clamped too, for exactly the reason the
    # file-wide pair is. A pre-epoched recording contains nothing outside its
    # own epoch, so a window reaching past it draws its baseline from the
    # previous trial's response and reports that as background EMG. Adding a
    # per-type column without this would have made that contamination
    # reachable again through a route the clamp did not know about.
    _wmap = _get('window_map') or {}
    if _wmap:
        _clamped, _wchanges = clamp_window_map(_wmap, pre_avail, post_avail)
        changes.extend(_wchanges)
        _set('window_map', _clamped)
    return cfg, changes


def clamp_window_map(window_map, pre_avail, post_avail):
    """Clamp {stim_type: (pre_ms, post_ms)} to what the file contains.

    Separate from clamp_config_to_epoch_bounds because the same map is held in
    two places: once file-wide in the parameters, and once per channel in the
    per-channel setup. The analysis reads the per-channel copy in preference,
    so clamping only the first left the run using the unclamped window while
    everything else used the clamped one -- and on a stitched pre-epoched
    recording the extra samples are mirror-padded guard band, drawn as a flat
    line and indistinguishable from a quiet trace.

    Returns (clamped_map, changes).
    """
    out, changes = {}, []
    for stim_type, win in (window_map or {}).items():
        try:
            pre, post = win
        except Exception:
            out[stim_type] = win
            continue
        new_pre = None if pre in (None, "") else min(float(pre), pre_avail)
        new_post = None if post in (None, "") else min(float(post), post_avail)
        if pre not in (None, "") and new_pre != float(pre):
            changes.append((f"window_map[{stim_type}].pre", pre, new_pre))
        if post not in (None, "") and new_post != float(post):
            changes.append((f"window_map[{stim_type}].post", post, new_post))
        out[stim_type] = (new_pre, new_post)
    return out, changes


# Short alias so the field list below stays readable.
_DD = DETECTION_DEFAULTS


@dataclass
class PipelineConfig:
    """Bundles all analysis settings so subfunctions share one parameter object."""
    # Time windows
    pre_ms:            int   = 20
    post_ms:           int   = 400
    ptp_start:         int   = 10
    ptp_end:           int   = 50
    prestim_ms:        int   = 100
    # Pre-stimulus baseline (background EMG) quantification
    # Carson (2026) estimated r.m.s. EMG over the 100 ms ending 3 ms before the
    # stimulus. The guard keeps the stimulus artefact out of the window; the
    # effective guard is max(rms_guard_ms, this stim type's gap_ms).
    rms_guard_ms:      float = 3.0
    # Remove the DC offset of the pre-stimulus window before taking its r.m.s.
    # An offset is not motoneurone activity, and carrying it into the RMS adds
    # between-trial variance that masks the association with MEP amplitude.
    prestim_rms_demean: bool = True
    # Filter
    apply_filter:      bool  = True
    apply_bandpass:    bool  = True
    apply_notch:       bool  = False
    apply_humbug:      bool  = False
    highpass:          int   = 20
    lowpass:           int   = 450
    notch_freq:        int   = 50
    notch_q:           int   = 30
    filter_order:      int   = 2
    filter_harmonics:  bool  = False
    flexible_bandpass: bool  = False
    hp_order:          int   = 2
    lp_order:          int   = 2
    humbug_harmonics:  int   = 6
    filter_family:     str   = "butter"
    cheby_ripple:      float = 1.0
    # Onset detection
    #
    # Every default below is read from mep_cmap.detection.defaults rather than
    # written as a literal. That module is the single source of truth shared
    # with preferences.py and the GUI; restating values here is what allowed
    # onset_method to default to "peak_fraction" in this dataclass while
    # preferences defaulted to "bigoni". tests/test_detection_defaults.py
    # fails if the two ever drift apart again.
    peak_fraction:         float = _DD["peak_fraction"]
    min_peak_amplitude:    float = _DD["min_peak_amplitude"]
    slope_threshold:       float = _DD["slope_threshold"]
    # See detection.ONSET_METHOD_LABELS for the full set of keys.
    onset_method:              str   = DEFAULT_ONSET_METHOD
    onset_bootstrap_crit:      float = _DD["onset_bootstrap_crit"]
    onset_bootstrap_n:         int   = _DD["onset_bootstrap_n"]
    onset_bigoni_smooth_ms:    float = _DD["onset_bigoni_smooth_ms"]
    onset_bigoni_min_run_ms:   float = _DD["onset_bigoni_min_run_ms"]
    onset_bigoni_walkback_sd:  float = _DD["onset_bigoni_walkback_sd"]
    # RMS envelope
    onset_env_window_ms:         float = _DD["onset_env_window_ms"]
    onset_env_criterion:         float = _DD["onset_env_criterion"]
    onset_env_significance:      float = _DD["onset_env_significance"]
    onset_env_n_boot:            int   = _DD["onset_env_n_boot"]
    onset_env_min_run_ms:        float = _DD["onset_env_min_run_ms"]
    onset_env_min_response_ms:   float = _DD["onset_env_min_response_ms"]
    onset_env_tkeo:              bool  = _DD["onset_env_tkeo"]
    onset_env_causal:            bool  = _DD["onset_env_causal"]
    onset_env_refine:            bool  = _DD["onset_env_refine"]
    onset_env_refine_window_ms:  float = _DD["onset_env_refine_window_ms"]
    onset_env_refine_sd:         float = _DD["onset_env_refine_sd"]
    onset_env_refine_sustain_ms: float = _DD["onset_env_refine_sustain_ms"]
    # CUSUM
    onset_cusum_k:               float = _DD["onset_cusum_k"]
    onset_cusum_h:               float = _DD["onset_cusum_h"]
    onset_cusum_max_accum_ms:    float = _DD["onset_cusum_max_accum_ms"]
    onset_cusum_min_response_ms: float = _DD["onset_cusum_min_response_ms"]
    onset_cusum_tkeo:            bool  = _DD["onset_cusum_tkeo"]
    # Derivative ratio (Boyles et al. 2026)
    onset_boyles_block_ms:            float = _DD["onset_boyles_block_ms"]
    onset_boyles_baseline_start_ms:   float = _DD["onset_boyles_baseline_start_ms"]
    onset_boyles_baseline_end_ms:     float = _DD["onset_boyles_baseline_end_ms"]
    onset_boyles_amplitude_gate:      float = _DD["onset_boyles_amplitude_gate"]
    onset_boyles_peak_jitter_ms:      float = _DD["onset_boyles_peak_jitter_ms"]
    onset_boyles_peak_window_length:  float = _DD["onset_boyles_peak_window_length"]
    onset_boyles_ratio_cutoff:        float = _DD["onset_boyles_ratio_cutoff"]
    onset_boyles_max_latency_ms:      float = _DD["onset_boyles_max_latency_ms"]
    onset_boyles_deriv_check_ms:      float = _DD["onset_boyles_deriv_check_ms"]
    onset_boyles_deriv_check_duty:    float = _DD["onset_boyles_deriv_check_duty"]
    onset_boyles_base_deriv_sds:      float = _DD["onset_boyles_base_deriv_sds"]
    onset_boyles_deriv_window_length: float = _DD["onset_boyles_deriv_window_length"]
    onset_boyles_literal:             bool  = _DD["onset_boyles_literal"]
    # Consensus / per-trial method agreement
    onset_methods_median_members: list = field(
        default_factory=lambda: list(_DD["onset_methods_median_members"]))
    onset_agreement:             bool  = _DD["onset_agreement"]
    # MEP offset (return of the response to baseline)
    mep_offset_enabled:          bool  = _DD["mep_offset_enabled"]
    mep_offset_min_duration_ms:  float = _DD["mep_offset_min_duration_ms"]
    mep_offset_max_duration_ms:  float = _DD["mep_offset_max_duration_ms"]
    mep_offset_min_return_ms:    float = _DD["mep_offset_min_return_ms"]
    mep_offset_env_window_ms:    float = _DD["mep_offset_env_window_ms"]
    mep_offset_criterion:        float = _DD["mep_offset_criterion"]
    mep_offset_peak_frac:        float = _DD["mep_offset_peak_frac"]
    # PTP measurement window anchored per stimulus type
    ptp_anchor:                  bool  = _DD["ptp_anchor"]
    ptp_anchor_pre_ms:           float = _DD["ptp_anchor_pre_ms"]
    ptp_anchor_duration_ms:      float = _DD["ptp_anchor_duration_ms"]
    ptp_anchor_min_trials:       int   = _DD["ptp_anchor_min_trials"]
    latency_map:           dict  = field(default_factory=dict)
    # Onset search-window anchoring (median-waveform seed; MEP onset only).
    # When enabled, the per-trial onset search window for each stim type is
    # narrowed to (anchor ± halfwidth), where 'anchor' is the onset detected
    # on the sample's median waveform using the selected method. Falls back to
    # the user's latency_map window if the anchor is weak/out-of-window.
    onset_anchor:              bool  = False
    onset_anchor_halfwidth_ms: float = 8.0
    onset_anchor_min_trials:   int   = 8
    # Outlier detection
    outlier_threshold:     float = 1.96
    enable_outlier_review: bool  = True
    # Bootstrap
    bootstrap_iter:        int   = 10000
    # Output labels / colours
    custom_labels:   dict = field(default_factory=dict)
    color_map:       dict = field(default_factory=dict)
    gap_ms_map:      dict = field(default_factory=dict)
    # Per-stimulus-type epoch window, {stim_type: (pre_ms, post_ms)}. A type
    # absent from the map uses the file-wide pre_ms/post_ms, so an empty map
    # reproduces the single-window behaviour exactly -- there is one code path,
    # and the shared window is its degenerate case.
    #
    # It exists because the epoch a response needs is a property of the
    # response: a cortical silent period wants several hundred milliseconds
    # after the pulse, an M-wave a few tens, and forcing both to share a window
    # means either truncating the first or carrying an order of magnitude of
    # unnecessary samples through every trial of the second.
    window_map:      dict = field(default_factory=dict)
    # {group_key: (stim_type, condition)} for keys that carry a condition.
    # The analysis groups by one key, but the trial file reports the two
    # separately; this is what lets it, without parsing the key apart.
    # Absent means the key IS the stimulus type, which is every recording
    # whose conditions were never assigned.
    condition_map:   dict = field(default_factory=dict)
    # Per-stimulus-type correction, in ms, between the file's event marker and
    # the actual stimulus. Negative means the pulse fired BEFORE the marker.
    # Applied when epoching, so every measure defined from t=0 follows.
    delay_ms_map:    dict = field(default_factory=dict)
    # Recording identifiers (from BIDS metadata)
    limb:            str  = ""
    measure:         str  = ""
    # Normalisation
    reference_map:      dict  = field(default_factory=dict)
    mmax_file:          str   = ""
    plateau_tolerance:  float = 0.10
    # Additional visual channels for inspector
    extra_channel_indices: list = field(default_factory=list)
    wide_window_s:      float = 3.0   # seconds either side of stim
    # CSP detection
    csp_types:             set   = field(default_factory=set)
    csp_min_silence_ms:    float = 25.0
    csp_min_return_ms:     float = 40.0
    csp_criterion:         float = 1.96
    csp_significance:      float = 0.99
    csp_n_boot:            int   = 1000
    csp_search_end_ms:     float = 400.0
    csp_max_mep_offset_ms: float = 100.0
    # The Data Inspector already accepted an RMS window and PipelineConfig had
    # no such field, so the two could only agree by both happening to sit on
    # the detector's own default. Nothing currently sets either, so behaviour
    # is unchanged; the field exists so that CspSettings reads one value for
    # both callers and wiring a control to it later cannot desynchronise them.
    csp_rms_window_ms:     float = 10.0
    # Averaged-waveform analysis mode (analyse per-condition mean once)
    average_mode:          bool  = False
    # Column selection for the narrowed COPY of trials.csv. None means no
    # narrowed file is written, which is what every run did before this and
    # what an unconfigured run still does. A list of group keys (see
    # column_groups) means also write <prefix>_trials_selected.csv holding
    # those groups plus the protected columns. trials.csv itself is never
    # affected by this.
    column_selection:      list  = None


def _make_bids_prefix(meta_prefix, file_stem):
    """Build a unique, clean bids_prefix from metadata and file stem.
    Strips sub-/ses- tokens (always redundant) and any token already
    verbatim in the metadata prefix (e.g. limb-left duplication).
    Falls back to full stem if nothing unique remains.
    """
    import re as _re
    if not meta_prefix:
        return file_stem
    if file_stem in meta_prefix:
        return meta_prefix
    unique = []
    for t in file_stem.split("_"):
        if _re.match(r"^sub-", t, _re.I) or _re.match(r"^ses-", t, _re.I):
            continue
        if t in meta_prefix or t.lower() in meta_prefix.lower():
            continue
        unique.append(t)
    suffix = "_".join(unique) if unique else file_stem
    return f"{meta_prefix}_{suffix}"


def pipeline_load_file(file_path, channel_idx, marker_name,
                       crop_ranges=None, crop_start=None, crop_end=None,
                       sources=None, channel_names=None, warn=None,
                       event_rows=None):
    """Load raw EMG, extract stim times, apply crop.

    ``sources`` is a list of EventSource. When given, the stimuli come from
    those rather than from the file's markers by name -- a threshold crossing
    on a trigger channel, a filtered set of comments, or asserted timing.

    They are re-derived here from the file rather than handed in already
    resolved, for the same reason crop ranges are: a run should be reproducible
    from the recording plus the configuration, not from whatever the interface
    happened to be holding. Nothing is passed as timestamps that could be
    recomputed from the file.

    With no sources the marker path is untouched, so every file configured the
    way files are configured today loads through exactly the code it always
    did.

    Returns
    -------
    emg        : np.ndarray  (samples,)
    time       : np.ndarray  absolute time axis in seconds
    fs         : int         sampling frequency
    unit       : str         voltage unit string (e.g. 'mV')
    stim_times : dict        {stim_type: [timestamps_s, ...]}
    """
    emg, fs, unit = extract_emg_waveform_and_fs(file_path, channel_idx)
    time       = np.arange(len(emg)) / fs
    if event_rows is not None:
        # Conditions assigned in the interface, from a BIDS events file. The two
        # columns are composed into group keys HERE, by the one function that
        # does it, so the analysis and the preview cannot disagree about which
        # trials belong together.
        from .conditions import group_events
        stim_times, _decoded = group_events(event_rows)
    elif sources:
        from .io import extract_events
        stim_times, _warnings = extract_events(file_path, sources,
                                               channel_names=channel_names)
        # merge_event_sources reports two sources claiming the same stimulus
        # type, and events from different sources landing closer together than
        # the near-simultaneous window. Both are misconfigurations often enough
        # that swallowing them here would hide the reason a trial count looks
        # wrong -- and the interface already shows them when sources are
        # chosen, so the run must not be quieter than the dialogue.
        for _w in (_warnings or []):
            if warn is not None:
                warn(_w)
    else:
        stim_times = extract_stim_times(file_path, marker_name)

    if crop_ranges:
        keep = np.zeros_like(time, dtype=bool)
        for a, b in crop_ranges:
            keep |= (time >= a) & (time <= b)
        emg  = emg[keep]
        time = time[keep]
    elif crop_start is not None and crop_end is not None:
        keep = (time >= crop_start) & (time <= crop_end)
        emg  = emg[keep]
        time = time[keep]

    # One rule, shared with whatever shows the analyst a trial list.
    stim_times = crop_stim_times(stim_times, crop_ranges, crop_start, crop_end)

    return emg, time, fs, unit, stim_times


#: The cfg fields ``pipeline_apply_filters`` reads, named rather than derived.
#: A caller that filters for display must supply exactly these, and building
#: the config by filtering a params snapshot against PipelineConfig field names
#: is silently wrong: the snapshot says `min_amp` and `enable_out_review` where
#: the fields are `min_peak_amplitude` and `enable_outlier_review`, so a name
#: filter substitutes defaults. Defined here, beside the function, so the
#: preview and the conditions review pane share one list instead of keeping a
#: copy each. test_preview_detection.py holds it to what the filter stage reads.
FILTER_CFG_FIELDS = (
    "apply_filter", "apply_bandpass", "apply_notch", "apply_humbug",
    "highpass", "lowpass", "hp_order", "lp_order", "flexible_bandpass",
    "notch_freq", "notch_q", "filter_order", "filter_harmonics",
    "humbug_harmonics",
)


def pipeline_apply_filters(emg, fs, cfg: PipelineConfig):
    """Apply the enabled filter chain to *emg* and return the filtered signal."""
    if not cfg.apply_filter:
        return emg.copy()

    nyq = 0.5 * fs

    if cfg.apply_humbug:
        emg = adaptive_mains_cancel(emg, fs,
                                    mains_freq=cfg.notch_freq,
                                    n_harmonics=cfg.humbug_harmonics)
    if cfg.apply_notch:
        sos_list = design_notch_sos(fs, cfg.notch_freq, cfg.notch_q,
                                    include_harmonics=cfg.filter_harmonics)
        for b, a in sos_list:
            emg = filtfilt(b, a, emg)
    if cfg.apply_bandpass:
        if cfg.flexible_bandpass:
            sos_hp = butter(cfg.hp_order, cfg.highpass / nyq,
                            btype='highpass', output='sos')
            sos_lp = butter(cfg.lp_order, cfg.lowpass  / nyq,
                            btype='lowpass',  output='sos')
            emg = sosfiltfilt(sos_hp, emg)
            emg = sosfiltfilt(sos_lp, emg)
        else:
            sos = butter(cfg.filter_order,
                         [cfg.highpass / nyq, cfg.lowpass / nyq],
                         btype='band', output='sos')
            emg = sosfiltfilt(sos, emg)
    return emg


def pipeline_prestim_rms(prestim, cfg: PipelineConfig = None, axis=None):
    """r.m.s. of the pre-stimulus window, DC offset removed by default.

    Single definition shared by outlier screening, per-trial quantification and
    averaged mode, so PreStimRMS means the same thing everywhere — including
    when it is used as the regressor in the Carson (2026) compensation. Any
    residual DC offset would otherwise enter the r.m.s. as between-trial
    variance unrelated to the state of the motoneurone pool, which attenuates
    the association the method is designed to remove.

    ``axis=None`` for a single 1-D window; ``axis=1`` for a (trials, samples)
    stack, returning one value per trial.
    """
    # Thin wrapper: this function's job is to read the preference, and
    # detection.quantification.compute_prestim_rms owns the arithmetic. Two
    # implementations of one measurement will always drift apart eventually,
    # and this pair already had, by about ten percent.
    demean = True if cfg is None else bool(getattr(cfg, "prestim_rms_demean", True))
    return compute_prestim_rms(prestim, demean=demean, axis=axis)


def crop_stim_times(stim_times, crop_ranges=None, crop_start=None,
                    crop_end=None):
    """Keep only the stimuli inside the selected range.

    Extracted so that anything showing the analyst a trial list uses the same
    rule the analysis does. It matters more than a display discrepancy would:
    conditions are assigned by trial INDEX, so a list built from the whole
    recording numbers its trials differently from one built after a crop, and
    every assignment made against the wrong numbering lands on the wrong trial.

    An empty type is dropped rather than kept empty, matching what the analysis
    does: a stimulus type with no trials in range is not a group.
    """
    if not crop_ranges and (crop_start is None or crop_end is None):
        return {k: list(v) for k, v in stim_times.items()}

    def _inside(t):
        if crop_ranges:
            return any(a <= t <= b for a, b in crop_ranges)
        return crop_start <= t <= crop_end

    out = {}
    for k, times in stim_times.items():
        kept = [t for t in times if _inside(t)]
        if kept:
            out[k] = kept
    return out


def split_group_key(cfg: PipelineConfig, group_key: str):
    """(stim_type, condition) for an analysis group key.

    The analysis groups by one key because every stage downstream is keyed that
    way; the trial file needs the two apart. This is the only place they are
    separated, so nothing has to know how they were joined -- a key is opaque
    everywhere else, which is what allows the composition to live entirely in
    mep_cmap.conditions.

    A key absent from the map is a stimulus type with no condition, which is
    every recording whose conditions were never assigned.
    """
    pair = (cfg.condition_map or {}).get(group_key)
    if not pair:
        return str(group_key), ""
    stim, cond = pair
    return str(stim), str(cond or "")


def resolve_window(cfg: PipelineConfig, stim_type: str):
    """This stimulus type's (pre_ms, post_ms), falling back to the file-wide pair.

    One accessor, so every stage answers the question the same way. Reading
    cfg.window_map directly in each place is how the pre-stimulus window and
    the amplitude window would come to disagree about where a trial starts.
    """
    win = (cfg.window_map or {}).get(stim_type)
    if not win:
        return float(cfg.pre_ms), float(cfg.post_ms)
    pre, post = win
    pre = float(cfg.pre_ms) if pre in (None, "") else float(pre)
    post = float(cfg.post_ms) if post in (None, "") else float(post)
    return pre, post


def window_samples(cfg: PipelineConfig, stim_type: str, fs: float):
    """(samples_before, samples_after) for one stimulus type."""
    pre, post = resolve_window(cfg, stim_type)
    return int(pre * fs / 1000), int(post * fs / 1000)


def time_axis_for(cfg: PipelineConfig, stim_type: str, fs: float):
    """The latency axis, in ms, for one stimulus type's epochs."""
    pre, post = resolve_window(cfg, stim_type)
    before, after = window_samples(cfg, stim_type, fs)
    return np.linspace(-pre, post, before + after, endpoint=False)


def overlay_groups(cfg: PipelineConfig, group_keys):
    """Which group keys may be drawn on one set of axes, and why not otherwise.

    Returns ``{base_stim_type: (keys, epoch, reason)}``. ``keys`` is every
    group key of that stimulus type; ``epoch`` is the (pre_ms, post_ms) they
    share, or None; ``reason`` is empty when they may be overlaid and
    otherwise names the epochs that differ.

    Overlaying two conditions cut to different epochs would put two time axes
    on one plot. The traces would be drawn against a single axis regardless,
    so a response at 30 ms in one condition and 30 ms in the other would land
    in different places, and the plot would show a latency difference that
    does not exist. Refused rather than rescaled: a rescaled overlay is still
    two different measurements, and the analyst has no way to see which trace
    came from which window.

    Compared on the RESOLVED pair from resolve_window, never on what
    window_map happens to hold. Two conditions can be configured differently
    and resolve identically -- one carrying an explicit copy of the file-wide
    pair -- or be configured identically and differ after a per-condition
    epoch is applied. Comparing the configuration rather than the result is
    the same fault as reading a file-wide value where a per-type one was
    needed.

    An event delay difference does NOT block an overlay. A delay moves each
    type's t=0 onto the actual stimulus, so two conditions with different
    delays are each correctly aligned and comparable; only the extents matter.
    """
    by_type = {}
    for key in group_keys:
        base, _cond = split_group_key(cfg, key)
        by_type.setdefault(base, []).append(key)

    out = {}
    for base, keys in by_type.items():
        keys = sorted(keys)
        epochs = {}
        for key in keys:
            epochs.setdefault(resolve_window(cfg, key), []).append(key)
        if len(epochs) == 1:
            out[base] = (keys, next(iter(epochs)), "")
            continue
        parts = []
        for (pre, post), members in sorted(epochs.items()):
            parts.append(f"{', '.join(members)}: {pre:g} to {post:g} ms")
        out[base] = (keys, None,
                     "these are cut to different epochs, so they cannot share "
                     "a time axis (" + "; ".join(parts) + ")")
    return out


#: How much of the requested pre-stimulus window must exist before a trial is
#: kept. Above this the baseline is trimmed and the trial survives; below it,
#: the window is a baseline in name only and the trial is dropped instead.
MIN_BASELINE_FRACTION = 0.5


def pipeline_extract_segments(time, emg, stim_times, stim_types, fs,
                               cfg: PipelineConfig, log_callback=None):
    """Extract per-trial EMG and pre-stim segments for every stim type.

    Returns
    -------
    dict mapping stim_type -> list of (seg_emg, seg_pre, stim_time_s) tuples.
    Only trials with a complete analysis window are included. Pre-stimulus
    baselines are all the same length within a stim type: where a stimulus sits
    too close to the start of the recording for the full window, every baseline
    of that type is trimmed to the shortest, so the outlier test compares like
    with like. ``log_callback`` is optional and reports any trimming.
    stim_time_s is the stimulus timestamp in seconds (from the raw time axis),
    preserved so downstream stages can reconstruct chronological trial order
    across stim types for session-level detrending and other analyses.
    """
    prestim_samples = int(cfg.prestim_ms * fs / 1000)

    result = {}
    guard_samples = int(round(max(cfg.rms_guard_ms, 0.0) * fs / 1000))

    for stim_type in stim_types:
        valid_times = [t for t in stim_times[stim_type]
                       if time.min() <= t <= time.max()]
        # Per type, alongside the gap and the delay: the window is the same
        # kind of quantity as those, and was the only one still shared.
        samples_before, samples_after = window_samples(cfg, stim_type, fs)
        gap_samples = int(cfg.gap_ms_map.get(stim_type, 0.0) * fs / 1000)
        # The background-EMG window must clear the stimulus artefact. Use the
        # larger of the configured guard and this stim type's artefact gap, so a
        # rig that already needs a long gap keeps it (Carson 2026: 103 ms to
        # 3 ms before the pulse).
        pre_offset = max(gap_samples, guard_samples)
        segs = []
        short_prestim = 0
        # Shift t=0 onto the actual stimulus. One line, but everything
        # downstream is defined relative to this index -- the pre-stimulus
        # window, the amplitude window, onset, offset, AUC and the Inspector's
        # zero -- so correcting it here corrects all of them consistently.
        # Reported latencies move with it, which is the point: a latency
        # measured from a marker known to be late is wrong by that much.
        delay_samples = int(round(
            float(cfg.delay_ms_map.get(stim_type, 0.0)) * fs / 1000.0))

        for stim_time in valid_times:
            idx   = int(np.argmin(np.abs(time - stim_time))) + delay_samples
            if idx < 0 or idx >= len(emg):
                continue          # correction pushed this trial off the record
            start = max(0, idx - samples_before)
            end   = min(len(emg), idx + samples_after)
            if (end - start) != (samples_before + samples_after):
                continue          # incomplete window — skip
            seg_emg = emg[start:end]
            pre_end   = max(0, idx - pre_offset)
            pre_start = max(0, pre_end - prestim_samples)
            seg_pre   = emg[pre_start:pre_end]
            # A short baseline is kept and the rest are trimmed to match, not
            # dropped and not padded.
            #
            # The analysis window above is checked and an incomplete one skips
            # the trial. This one was not, and max(0, ...) clamps it silently at
            # the start of the recording, so a stimulus with less room before it
            # than the baseline needs produced a SHORT array beside full-length
            # ones. np.array() over the mixture raised "inhomogeneous shape"
            # three stages later, where nothing pointed back here.
            #
            # Only the BASELINE is short in this case -- the epoch is complete,
            # and the response is perfectly measurable -- because the baseline
            # additionally clears the artefact gap. A stimulus 100 ms into a
            # recording with a 100 ms baseline and a 5 ms gap is short by 5%.
            # Dropping the trial to protect 10 samples of baseline costs more
            # than it saves, and the trial it costs is systematically the first
            # one, which is not random with respect to anything measured across
            # a session.
            #
            # So the length is recorded and every baseline is trimmed to the
            # shortest AFTER the loop. Equal length is what matters: the outlier
            # test compares one trial's baseline RMS against the others', and an
            # RMS over 190 samples is not strictly comparable with one over 200.
            # Padding would put fabricated signal into that comparison.
            segs.append((seg_emg, seg_pre, stim_time))

        # The floor. Trimming is cheap when a baseline is a few per cent short
        # and unacceptable when it is a tenth of what was asked for, so below
        # half the requested window the trial is dropped after all -- that is a
        # baseline in name only, and silently measuring one would be worse than
        # losing the trial.
        if segs:
            _floor = max(1, int(prestim_samples * MIN_BASELINE_FRACTION))
            _kept = [s for s in segs if len(s[1]) >= _floor]
            short_prestim = len(segs) - len(_kept)
            segs = _kept
        if segs:
            _n_pre = min(len(s[1]) for s in segs)
            if _n_pre < prestim_samples:
                # Trimmed from the FRONT: the end of the baseline is fixed
                # relative to the stimulus, and it is the far end that is
                # missing on the clamped trial.
                segs = [(e, p[len(p) - _n_pre:], t) for e, p, t in segs]
                if log_callback:
                    log_callback(
                        f"   \u2139\ufe0f  '{stim_type}': pre-stimulus window "
                        f"trimmed to {_n_pre * 1000.0 / fs:.0f} ms (asked for "
                        f"{cfg.prestim_ms:g} ms) — one or more stimuli sit too "
                        f"close to the start of the recording. Every trial of "
                        f"this type uses the same window, so baselines stay "
                        f"comparable.")
        if short_prestim and log_callback:
            log_callback(
                f"   \u26a0\ufe0f  '{stim_type}': {short_prestim} trial(s) "
                f"skipped — less than half the requested "
                f"{cfg.prestim_ms:g} ms pre-stimulus window was available.")
        if segs:
            result[stim_type] = segs
    return result


def pipeline_detect_outliers(emg_segments, prestim_segments,
                              ptp_start_idx, ptp_end_idx, cfg: PipelineConfig):
    """Compute per-trial z-scores and return flagged outlier indices.

    Returns
    -------
    ptps, rms_vals, preptp : np.ndarray  per-trial metrics
    rms_z, ptp_z           : np.ndarray  z-scores
    outlier_indices        : list[int]   indices where |z| > threshold
    """
    ptps     = _np_ptp(emg_segments[:, ptp_start_idx:ptp_end_idx], axis=1)
    rms_vals = pipeline_prestim_rms(prestim_segments, cfg, axis=1)
    preptp   = _np_ptp(prestim_segments, axis=1)
    rms_z    = zscore(rms_vals) if len(rms_vals) > 1 else np.zeros_like(rms_vals)
    ptp_z    = zscore(ptps)     if len(ptps)     > 1 else np.zeros_like(ptps)
    thr      = cfg.outlier_threshold
    outlier_indices = [i for i, (zr, zp) in enumerate(zip(rms_z, ptp_z))
                       if abs(zr) > thr or abs(zp) > thr]
    return ptps, rms_vals, preptp, rms_z, ptp_z, outlier_indices


def pipeline_review_outliers(stim_type, name, emg_segments, prestim_segments,
                              outlier_indices, ptps, rms_vals, rms_z, ptp_z,
                              cfg, fs, pre_ms, post_ms, unit,
                              review_cb, log_callback):
    """Run interactive outlier review (if enabled).

    Returns
    -------
    rejected_indices : list[int]   indices the user chose to remove
    log_entries      : list[dict]  bookkeeping rows for the rejected-outlier log
    """
    flagged = [
        {"file": name, "stim_type": stim_type, "index": oi,
         "emg_segment": emg_segments[oi], "prestim_segment": prestim_segments[oi],
         "rms": rms_vals[oi], "ptp": ptps[oi],
         "z_rms": rms_z[oi],  "z_ptp": ptp_z[oi]}
        for oi in outlier_indices
    ]
    if not flagged or not cfg.enable_outlier_review:
        return [], []

    log_callback(f"⚠️  {len(flagged)} potential outliers in {name} – {stim_type}")
    kept = review_cb(flagged, fs, pre_ms, post_ms, unit)
    rejected_indices = [o["index"] for o in flagged if o not in kept]
    log_entries = [
        {"File": o["file"], "StimType": o["stim_type"],
         "SegmentIndex": o["index"] + 1,
         "PreStimRMS": o["rms"], "PTP": o["ptp"],
         "Z_RMS": o["z_rms"],   "Z_PTP": o["z_ptp"]}
        for o in flagged if o not in kept
    ]
    return rejected_indices, log_entries


ARTEFACT_FLOOR_MS = 2.0


def onset_search_window(cfg, min_lat, max_lat):
    """Search bounds (ms post-stim) for onset detection on one stimulus type.

    The onset search window is derived from the physiological latency profile
    widened by the PTP window -- NOT taken from the PTP window, as it was until
    v1.3.2.

    The PTP window is a single per-file setting; the latency profile is per
    stimulus type. Passing ``cfg.ptp_start`` as the search floor therefore let
    an amplitude-measurement setting silently override the physiological bounds
    configured in Stage 1b, and it did so without failing. Measured on a
    deltoid-like case (true onset 8.9 ms, profile 8-16 ms, PTP window
    10-50 ms), every trial returned exactly 10.00 ms: the window edge reported
    as a latency, with a between-trial SD of zero. Implausibly consistent
    numbers are the signature, which makes this far more dangerous than a
    detector that returns None.

    Widening is safe in both directions. Every detector still bounds its result
    by ``min_latency_ms`` / ``max_latency_ms``, so a wider search cannot yield
    an onset outside the profile -- it only stops the profile being clipped.
    The floor is held at the artefact blanking period so widening can never
    drag the peak search onto the stimulus artefact.

    Returns
    -------
    (search_start_ms, search_end_ms)
    """
    lo = float(cfg.ptp_start)
    hi = float(cfg.ptp_end)
    if min_lat is not None:
        lo = min(lo, float(min_lat))
    if max_lat is not None:
        hi = max(hi, float(max_lat))
    lo = max(lo, ARTEFACT_FLOOR_MS)
    if hi <= lo:
        hi = float(cfg.ptp_end)
    return lo, hi


def _detect_onset_dispatch(signal, fs, cfg, min_lat, max_lat,
                           template=None, stim_type=None):
    """Run the configured onset detector on a single trace and return onset (ms).

    Thin wrapper over ``detection.dispatch_onset``, which is the single place
    that maps a method name onto a detector. The inspector calls the same
    function, so the two can no longer disagree about which algorithm ran or
    which parameters it used -- see detection/dispatch.py for the drift this
    replaced.

    ``pre_ms`` MUST match how the trace in front of it was cut: it is what
    tells the detector where the stimulus sits. Get it wrong and the detector
    searches the wrong region and returns None even when a clear MEP is
    present.

    That was ``cfg.pre_ms`` unconditionally, and this note said it had to be.
    It stopped being true when epochs became per stimulus type:
    pipeline_extract_segments cuts with ``window_samples(cfg, stim_type)``, so a
    type given a 100 ms epoch against a file-wide 20 ms has its stimulus at
    100 ms while ``cfg.pre_ms`` says 20. The preview, which trims to the same
    per-type window, then found no onset on ANY trial -- exactly the symptom
    this note predicted -- and reported that the run would find none either.

    Passing ``stim_type`` resolves that type's own pre. Omitting it keeps the
    old behaviour, for a caller whose trace really is cut to the file-wide
    window.
    """
    _pre_ms = (resolve_window(cfg, stim_type)[0] if stim_type is not None
               else cfg.pre_ms)
    _lo, _hi = onset_search_window(cfg, min_lat, max_lat)
    return dispatch_onset(
        signal, fs, detector_params(cfg),
        pre_ms=_pre_ms,
        search_start_ms=_lo,
        search_end_ms=_hi,
        min_latency_ms=min_lat,
        max_latency_ms=max_lat,
        template=template,
    )


def pipeline_detect_onsets(stim_type, segs_all, out_set,
                           ptp_start_idx, ptp_end_idx, fs, cfg,
                           log_callback=print):
    """Single anchored MEP-onset pass for one stim-type sample.

    This is the sole source of automatic onset values: it runs the configured
    detector on every trial in ``segs_all``, optionally within a search window
    anchored to the sample's MEDIAN waveform (onset anchoring feature). Returns
    ``{trial_idx: onset_ms | None}``.

    The result seeds both the interactive inspector and the saved output, so
    what you see and what is written trace to one computation. The median
    anchor is derived from OUTLIER-SCREENED trials only (``out_set``); user
    exclusions are made later in the inspector and deliberately do not feed
    back into the anchor (computed once, no recomputation).
    """
    onsets: dict = {}
    if len(segs_all) == 0:
        return onsets

    # A missing profile is REPORTED, not substituted.
    #
    # The fallback used to be a hardcoded 10-50 ms, and every detector bounds
    # its result by the minimum -- so a stimulus type absent from the map
    # returned exactly 10.00 ms on every trial, with a between-trial SD of
    # zero. That is the same window-edge-as-latency failure the search-window
    # comment above describes at length, reached by a different door: there
    # the profile was overridden, here it was invented.
    _base_lat = cfg.latency_map.get(stim_type)
    if not _base_lat:
        _base_lat = (float(cfg.ptp_start), float(cfg.ptp_end))
        log_callback(
            f"   \u26a0\ufe0f  '{stim_type}' has no latency profile, so onset "
            f"detection is bounded by the amplitude window "
            f"({_base_lat[0]:.0f}-{_base_lat[1]:.0f} ms) instead. Set the "
            f"stimulus type and muscle group for it on tab 1a: onsets pinned "
            f"to a window edge read as real latencies.")
    _min_lat0, _max_lat0 = _base_lat
    _eff_min_lat, _eff_max_lat = _min_lat0, _max_lat0

    # Condition median over outlier-screened trials. Previously computed only
    # when onset anchoring was on; it is now always available because the
    # derivative-ratio detector needs a condition average for its peak-jitter
    # gate, and it is the same waveform the anchor uses. Screened trials only,
    # for the same reason as the anchor: a single aberrant trial should not move
    # a landmark that every trial is then judged against.
    _clean = [segs_all[j] for j in range(len(segs_all))
              if j not in (out_set or set())]
    try:
        _template = (np.median(np.vstack(_clean), axis=0)
                     if len(_clean) >= 2 else None)
    except Exception:
        _template = None

    if getattr(cfg, "onset_anchor", False):
        _min_n = int(getattr(cfg, "onset_anchor_min_trials", 8))
        if len(_clean) >= _min_n:
            _median = _template
            try:
                _med_ptp = (_np_ptp(_median[ptp_start_idx:ptp_end_idx])
                            if _median is not None else 0.0)
            except Exception:
                _median, _med_ptp = None, 0.0
            if _median is not None and _med_ptp >= cfg.min_peak_amplitude:
                try:
                    _anchor = _detect_onset_dispatch(
                        _median, fs, cfg, _min_lat0, _max_lat0,
                        template=_median, stim_type=stim_type)
                except Exception:
                    _anchor = None
                if _anchor is not None and _min_lat0 <= _anchor <= _max_lat0:
                    _hw = float(getattr(cfg, "onset_anchor_halfwidth_ms", 8.0))
                    _eff_min_lat = max(_min_lat0, _anchor - _hw)
                    _eff_max_lat = min(_max_lat0, _anchor + _hw)
                    log_callback(
                        f"🎯 Onset anchor '{stim_type}': median onset {_anchor:.1f} ms "
                        f"→ search window {_eff_min_lat:.1f}–{_eff_max_lat:.1f} ms "
                        f"(from {_min_lat0:.1f}–{_max_lat0:.1f} ms)"
                    )
                else:
                    log_callback(
                        f"🎯 Onset anchor '{stim_type}': no reliable median onset "
                        f"— using full window {_min_lat0:.1f}–{_max_lat0:.1f} ms"
                    )
            else:
                log_callback(
                    f"🎯 Onset anchor '{stim_type}': median MEP below amplitude gate "
                    f"({_med_ptp:.3f} < {cfg.min_peak_amplitude} mV) "
                    f"— using full window {_min_lat0:.1f}–{_max_lat0:.1f} ms"
                )
        else:
            log_callback(
                f"🎯 Onset anchor '{stim_type}': only {len(_clean)} clean trial(s) "
                f"(< {_min_n}) — using full window {_min_lat0:.1f}–{_max_lat0:.1f} ms"
            )

    for idx, seg in enumerate(segs_all):
        onsets[idx] = _detect_onset_dispatch(seg, fs, cfg, _eff_min_lat,
                                             _eff_max_lat, template=_template,
                                             stim_type=stim_type)

    _warn_if_onsets_pinned_to_a_bound(
        stim_type, onsets, fs, _eff_min_lat, _eff_max_lat, log_callback)
    return onsets


def _warn_if_onsets_pinned_to_a_bound(stim_type, onsets, fs,
                                      min_lat, max_lat, log_callback):
    """Flag latencies that have collapsed onto a search-window boundary.

    A detector reporting the edge of its own search window produces a latency
    that looks like a measurement and is not one. It is far more damaging than
    a detector returning None, because the numbers are plausible and their
    between-trial consistency reads as good data rather than as a warning.
    Before the search window was decoupled from the PTP window, a deltoid
    profile against the default PTP window gave exactly 10.00 ms on every
    trial.

    That specific cause is fixed, but the same shape can still arise from a
    latency profile set too narrowly for the muscle. This makes it loud.
    """
    vals = [v for v in onsets.values() if v is not None]
    # Three is enough. On a real file a peripheral condition with only three
    # detected trials sat 3/3 on its lower bound and stayed silent at a
    # threshold of four -- the conditions with fewest usable trials are exactly
    # the ones where a wrong latency profile does most damage.
    if len(vals) < 3:
        return
    tol = 1.5 * 1000.0 / float(fs)          # a sample and a half
    for bound, edge in ((min_lat, "lower"), (max_lat, "upper")):
        if bound is None:
            continue
        n_at = sum(1 for v in vals if abs(v - float(bound)) <= tol)
        frac = n_at / float(len(vals))
        if frac >= 0.5:
            log_callback(
                f"⚠️ '{stim_type}': {n_at}/{len(vals)} onsets sit on the "
                f"{edge} latency bound ({float(bound):g} ms). These are the "
                f"edge of the search window, not measurements - widen the "
                f"latency profile for this stimulus type in Stage 1b."
            )


def ptp_window_for_stim_type(stim_type, onsets, fs, cfg,
                             default_start_idx, default_end_idx,
                             samples_before, log_callback=print):
    """PTP measurement window (sample indices) for one stimulus type.

    Returns the file-wide window unchanged unless ``cfg.ptp_anchor`` is on and
    the condition has enough detected onsets to trust a median.

    Why per stimulus type
    ---------------------
    The PTP window is one setting for the whole file; the latency profile is
    per stimulus type. A recording containing both M-waves and MEPs therefore
    cannot be measured correctly by a single window. On a real mixed file with
    the default 10-50 ms window, the conditions whose M-wave onset was 4 ms had
    the first 6 ms of every response excluded from the amplitude measurement --
    and an M-wave's entire biphasic deflection lasts only 5-15 ms, so what was
    reported as peak-to-peak amplitude was largely the signal AFTER the
    response had finished.

    Anchoring is per stimulus type, not per trial. Per-trial anchoring would
    make amplitude a function of onset-detection error: jitter would propagate
    straight into the primary outcome, trials with no detected onset would have
    no amplitude at all, and within-condition amplitudes would no longer be
    measured over a common window. The condition median is computed from the
    same onsets that seed the inspector, so the window is identical for every
    trial in the condition and the measurement stays comparable.

    The user's PTP window end is kept as a hard ceiling, so an anchored window
    can never extend past what was configured.
    """
    if not getattr(cfg, "ptp_anchor", False):
        return default_start_idx, default_end_idx, None

    vals = [v for v in onsets.values() if v is not None]
    min_n = int(getattr(cfg, "ptp_anchor_min_trials", 4))
    if len(vals) < min_n:
        # Fall back to this stimulus type's own LATENCY PROFILE, not to the
        # file-wide window.
        #
        # The file-wide start is the very thing anchoring exists to replace: on
        # a mixed or peripheral recording it is typically 10 ms, which sits
        # after the peak of an M-wave. Using it as the fallback meant that a
        # condition whose onsets failed to detect -- the condition already in
        # trouble -- was also the one whose amplitude got truncated, while its
        # neighbours measured correctly. Measured on a real recording, the
        # first phase of a 3.8 mV M-wave fell outside the window entirely and
        # peak-to-peak was read from a 2.1 mV shoulder instead.
        #
        # The profile minimum is the analyst's own statement of where a
        # response can begin for this stimulus type, so it is a better floor
        # than a single number shared across every type in the file. The
        # configured end still applies as a ceiling.
        _lat = (getattr(cfg, "latency_map", None) or {}).get(stim_type)
        if _lat:
            try:
                _lo_ms = float(_lat[0]) - float(cfg.ptp_anchor_pre_ms)
                _lo_ms = max(_lo_ms, ARTEFACT_FLOOR_MS)
                _s = int(round(samples_before + _lo_ms * fs / 1000.0))
                _s = max(0, min(_s, default_end_idx - 2))
                if _s < default_start_idx:
                    log_callback(
                        f"   PTP anchor '{stim_type}': only {len(vals)} "
                        f"onset(s) detected (need {min_n}) - using the latency "
                        f"profile instead, window {_lo_ms:.1f}-"
                        f"{(default_end_idx - samples_before) * 1000.0 / fs:.1f}"
                        f" ms.")
                    return _s, default_end_idx, None
            except Exception:
                pass
        log_callback(
            f"   PTP anchor '{stim_type}': only {len(vals)} onset(s) detected "
            f"(need {min_n}) - keeping the file-wide PTP window.")
        return default_start_idx, default_end_idx, None

    median_onset = float(np.median(vals))
    start_ms = median_onset - float(cfg.ptp_anchor_pre_ms)
    end_ms = median_onset + float(cfg.ptp_anchor_duration_ms)

    # Never earlier than the artefact floor, never later than the configured
    # PTP window end.
    start_ms = max(start_ms, ARTEFACT_FLOOR_MS)
    end_ms = min(end_ms, float(cfg.ptp_end))
    if end_ms <= start_ms:
        log_callback(
            f"   ⚠️ PTP anchor '{stim_type}': median onset {median_onset:.1f} ms "
            f"leaves no room below the PTP window end ({cfg.ptp_end} ms) - "
            f"keeping the file-wide window.")
        return default_start_idx, default_end_idx, None

    s_idx = samples_before + int(round(start_ms * fs / 1000.0))
    e_idx = samples_before + int(round(end_ms * fs / 1000.0))
    log_callback(
        f"   📐 PTP anchor '{stim_type}': median onset {median_onset:.1f} ms "
        f"-> PTP window {start_ms:.1f}-{end_ms:.1f} ms "
        f"(file-wide was {cfg.ptp_start}-{cfg.ptp_end} ms)")
    return s_idx, e_idx, (start_ms, end_ms)


def pipeline_quantify_segments(stim_type, segs_all, prestim_all,
                                out_set, excluded_set, segments_metadata,
                                ptp_start_idx, ptp_end_idx,
                                fs, cfg: PipelineConfig,
                                custom_labels, name, auto_onsets,
                                log_callback=print, agreement_out=None):
    """Per-trial quantification of PTP, latency, silent period and AUC.

    Returns
    -------
    auto_rows    : list  rows for the auto-metrics CSV
    manual_rows  : list  rows for the manual-override CSV
    summary_row  : list  one summary row (cleaned trials only)
    with_out_row : list  one summary row (all trials including outliers)
    ptps_array   : np.ndarray  per-trial PTP (all trials)
    """
    rms_all    = pipeline_prestim_rms(prestim_all, cfg, axis=1)
    preptp_all = _np_ptp(prestim_all, axis=1)
    # RMS over the same window PTP uses, filled per trial in the loop below via
    # the shared compute_rms so there is only one definition of it.
    mep_rms_vals = []
    rms_z_full = (zscore(rms_all) if len(rms_all) > 1
                  else np.zeros_like(rms_all))
    ptps = np.empty(len(segs_all))

    # Physiological onset window for this stim type, used by the agreement
    # calculation. Mirrors pipeline_detect_onsets' unnarrowed bounds.
    # Same rule as the detection path above: a missing profile is bounded by
    # the amplitude window the analyst chose, not by a constant from here.
    _ag_lat = (cfg.latency_map.get(stim_type)
               or (float(cfg.ptp_start), float(cfg.ptp_end)))
    _ag_min_lat, _ag_max_lat = _ag_lat
    _agreement_warned = False
    _offset_warned = False
    # Condition median for the derivative-ratio member's peak-jitter gate.
    # Screened trials only, matching pipeline_detect_onsets.
    try:
        _ag_clean = [segs_all[j] for j in range(len(segs_all))
                     if j not in (out_set or set())]
        _agreement_template = (np.median(np.vstack(_ag_clean), axis=0)
                               if len(_ag_clean) >= 2 else None)
    except Exception:
        _agreement_template = None

    auto_rows, manual_rows = [], []
    latencies, silent_durs = [], []
    auc_vals_all, auc_vals_clean = [], []

    # Onset comes exclusively from the single anchored pre-pass
    # (pipeline_detect_onsets), so the automatic values used here are identical
    # to those seeded into the inspector. Hard requirement — no inline re-detect.
    if auto_onsets is None:
        raise ValueError(
            "pipeline_quantify_segments requires auto_onsets from "
            "pipeline_detect_onsets (single source of truth for MEP onset)."
        )

    for idx, seg in enumerate(segs_all):
        # This stim type's OWN epoch, in ms. Every helper below is handed a
        # segment cut to THIS window, so passing cfg.pre_ms/cfg.post_ms -- the
        # file-wide pair -- tells them the epoch begins somewhere it does not.
        #
        # Same fault as the index conversion below, in three more places. On a
        # recording whose conditions used a 100 ms epoch against a file-wide
        # 20 ms, the offset detector was told the segment started at -20 ms and
        # returned 82 ms for a response the Inspector put at 53 ms, with the
        # duration wrong to match. The Inspector was right: it works in the
        # segment's own coordinates.
        _pre_type_ms, _post_type_ms = resolve_window(cfg, stim_type)

        # ── automatic metrics ────────────────────────────────────────────
        auto_ptp = compute_ptp(seg, ptp_start_idx, ptp_end_idx)
        auto_lat = auto_onsets.get(idx)

        # ── manual overrides from inspector ──────────────────────────────
        # Inspector segments start at -prestim_ms; segs_all start at this stim
        # type's OWN pre, which is not necessarily the file-wide one.
        # Convert inspector-space indices to segs_all-space before applying.
        #
        # This used cfg.pre_ms, the file-wide value. That was right while every
        # type shared one window and wrong the moment epochs became per type: a
        # condition given a 100 ms epoch against a file-wide 20 ms shifted every
        # index by 160 samples at 2 kHz, putting the peak markers 80 ms early,
        # in the pre-stimulus baseline. Peak-to-peak across noise reads as
        # roughly zero and, the two indices now being arbitrary points rather
        # than a max and a min, sometimes NEGATIVE -- which is how it was found.
        #
        # Latency hid it: man_lat is computed from _insp_sb alone, so latencies
        # stayed correct while amplitudes collapsed, and a results file with
        # plausible latencies and impossible amplitudes looks like an amplitude
        # problem rather than a coordinate one.
        _insp_sb = int(cfg.prestim_ms * fs / 1000)  # stim @ inspector idx
        _segs_sb = window_samples(cfg, stim_type, fs)[0]  # stim @ segs_all idx
        _offset  = _insp_sb - _segs_sb
        _n       = len(seg)
        def _ci(i): return min(max(0, i - _offset), _n - 1)
        mk = (stim_type, idx)
        if mk in segments_metadata:
            m = segments_metadata[mk]
            # Per-field override. A seed-only entry (from the anchored onset
            # pre-pass) carries onset_idx but NOT the PTP marker indices —
            # those are added by the inspector only for trials the user
            # actually views. Fall back to the auto value for any field the
            # metadata does not supply, so unviewed seeded trials behave
            # exactly as auto (single source of truth) rather than raising.
            if "ptp_max_idx" in m and "ptp_min_idx" in m:
                man_ptp = seg[_ci(m["ptp_max_idx"])] - seg[_ci(m["ptp_min_idx"])]
                # Peak-to-peak is a magnitude and cannot be negative. The other
                # writer of this column already guards it; this one did not, so
                # the coordinate bug above reached trials.csv as -0.01 mV and
                # propagated into z-scores, normalisation and the group table
                # as though it meant something.
                #
                # The magnitude is still WRONG when the indices are wrong. This
                # is a floor, not a repair: it stops an impossible number being
                # published, and the disagreement warning below is what says the
                # value should not be trusted.
                if man_ptp < 0:
                    man_ptp = abs(man_ptp)
            else:
                man_ptp = auto_ptp
            if "onset_idx" in m:
                man_lat = (m["onset_idx"] - _insp_sb) * 1000 / fs
            else:
                man_lat = auto_lat
        else:
            man_ptp, man_lat = auto_ptp, auto_lat

        # A latency at or before the stimulus (≤ 0 ms) is not a physiological
        # onset — it is a "no detection" placeholder (e.g. the inspector leaves
        # the marker at the stim when its detector returns nothing, giving
        # onset_idx == stim → 0.0 ms). Treat as undetected so it is written as
        # "Not Detected" and excluded from mean/SD latency, rather than being
        # counted as a real 0 ms latency. PTP is left untouched.
        if auto_lat is not None and auto_lat <= 0:
            auto_lat = None
        if man_lat is not None and man_lat <= 0:
            man_lat = None

        ptps[idx] = man_ptp

        # ── silent period ────────────────────────────────────────────────
        silent_dur = "Not Marked"
        if mk in segments_metadata and "silent_start_idx" in segments_metadata[mk]:
            md = segments_metadata[mk]
            # Duration is a difference so offset cancels out
            silent_dur = round(
                (md["silent_end_idx"] - md["silent_start_idx"]) * 1000 / fs, 2)
            silent_durs.append(silent_dur)
            # Absolute timepoints relative to stim (using inspector segment offset)
            _insp_sb_sp = int(cfg.prestim_ms * fs / 1000)
            sp_mep_offset_ms = round(
                (md["silent_start_idx"] - _insp_sb_sp) * 1000 / fs, 2)
            sp_emg_return_ms = round(
                (md["silent_end_idx"] - _insp_sb_sp) * 1000 / fs, 2)
        else:
            sp_mep_offset_ms = None
            sp_emg_return_ms = None

        # ── Onset method agreement (opt-in) ──────────────────────────────
        # Computed independently of which method is SELECTED, so the
        # disagreement columns are available even on a plain Bigoni run. Off by
        # default: it runs every member detector on every trial.
        #
        # The bounds come from cfg.latency_map, NOT from any anchor-narrowed
        # window. Narrowing the search window would compress the spread between
        # members artificially and make the disagreement metric look better the
        # more constrained the search was, which is exactly backwards.
        agreement = None
        if getattr(cfg, "onset_agreement", False):
            try:
                _ag_lo, _ag_hi = onset_search_window(
                    cfg, _ag_min_lat, _ag_max_lat)
                agreement = compute_onset_agreement(
                    seg, fs,
                    pre_ms=_pre_type_ms,
                    search_start_ms=_ag_lo,
                    search_end_ms=_ag_hi,
                    min_latency_ms=_ag_min_lat,
                    max_latency_ms=_ag_max_lat,
                    min_peak_amplitude=cfg.min_peak_amplitude,
                    methods=cfg.onset_methods_median_members,
                    params=detector_params(cfg),
                    template=_agreement_template,
                )
            except Exception as _exc:
                # Reported, not swallowed. An early version of this block
                # referenced names that a refactor had moved out of scope; the
                # bare handler turned a NameError into silently blank columns
                # that looked like "agreement was simply off".
                agreement = None
                if not _agreement_warned:
                    log_callback(
                        f"⚠️ Onset agreement failed for '{stim_type}' "
                        f"({type(_exc).__name__}: {_exc}); "
                        f"agreement columns will be blank.")
                    _agreement_warned = True

        # ── MEP offset ───────────────────────────────────────────────────
        # One precedence rule, applied in one place: a manual marker wins;
        # otherwise, when cSP is enabled and detected, the end of the MEP and
        # the start of the silent period are the SAME physical event and are
        # reported as the same number; otherwise the envelope detector finds
        # the return to baseline. MEP_Offset_Source records which branch fired
        # so no value's provenance has to be inferred.
        mep_offset_ms = mep_duration_ms = None
        mep_offset_src = "none"
        if getattr(cfg, "mep_offset_enabled", True):
            # Phase 3 will add a draggable offset marker; reading it here now
            # means that work needs no further pipeline change.
            _man_off = None
            if mk in segments_metadata and "mep_offset_idx" in segments_metadata[mk]:
                _man_off = ((segments_metadata[mk]["mep_offset_idx"] - _insp_sb)
                            * 1000.0 / fs)
            try:
                _res = resolve_mep_offset(
                    seg, fs,
                    onset_ms=man_lat,
                    csp_start_ms=sp_mep_offset_ms,
                    csp_enabled=(stim_type in cfg.csp_types),
                    manual_offset_ms=_man_off,
                    pre_ms=_pre_type_ms,
                    search_end_ms=_post_type_ms,
                    min_duration_ms=cfg.mep_offset_min_duration_ms,
                    max_duration_ms=cfg.mep_offset_max_duration_ms,
                    min_return_ms=cfg.mep_offset_min_return_ms,
                    env_window_ms=cfg.mep_offset_env_window_ms,
                    criterion=cfg.mep_offset_criterion,
                    peak_frac=cfg.mep_offset_peak_frac,
                )
                mep_offset_ms  = _res.offset_ms
                mep_duration_ms = _res.duration_ms
                mep_offset_src = _res.source
            except Exception as _exc:
                mep_offset_ms = mep_duration_ms = None
                mep_offset_src = "none"
                if not _offset_warned:
                    log_callback(
                        f"⚠️ MEP offset detection failed for '{stim_type}' "
                        f"({type(_exc).__name__}: {_exc}); "
                        f"MEP_Offset(ms) will be blank.")
                    _offset_warned = True

        # ── AUC ──────────────────────────────────────────────────────────
        auc_val = None
        if mk in segments_metadata and "auc_start_idx" in segments_metadata[mk]:
            # User-set or inspector-auto AUC window
            a0 = _ci(segments_metadata[mk]["auc_start_idx"])
            a1 = _ci(segments_metadata[mk]["auc_end_idx"])
            auc_val = compute_auc(seg, a0, a1, fs)
            if auc_val is not None:
                auc_vals_all.append(auc_val)
        elif mk in segments_metadata and \
                "silent_start_idx" in segments_metadata[mk] and \
                "onset_idx" in segments_metadata[mk]:
            # Auto-calculate AUC: onset → cSP start (inspector detected both)
            a0 = _ci(segments_metadata[mk]["onset_idx"])
            a1 = _ci(segments_metadata[mk]["silent_start_idx"])
            if a1 > a0:
                auc_val = float(_np_trapz(np.abs(seg[a0:a1]), dx=1 / fs))
                auc_vals_all.append(auc_val)
        elif (man_lat is not None and mep_offset_ms is not None
              and mep_offset_src == "envelope"):
            # No cSP to close the window, but the response has a detected end.
            # This is what gives resting-state recordings a principled AUC
            # instead of requiring the endpoint to be dragged by hand.
            a0 = _segs_sb + int(round(man_lat * fs / 1000.0))
            a1 = _segs_sb + int(round(mep_offset_ms * fs / 1000.0))
            a0 = min(max(0, a0), len(seg) - 1)
            a1 = min(max(0, a1), len(seg))
            if a1 > a0:
                auc_val = compute_auc(seg, a0, a1, fs)
                if auc_val is not None:
                    auc_vals_all.append(auc_val)
        elif mk not in segments_metadata and stim_type in cfg.csp_types:
            # Unreviewed segment with CSP enabled — try auto-detect onset+CSP
            # and compute AUC if both succeed
            from .detection import CspSettings, detect_csp_for_trial
            _ptp_s    = _segs_sb + int(cfg.ptp_start * fs / 1000)
            _ptp_e    = _segs_sb + int(cfg.ptp_end   * fs / 1000)
            if _ptp_e < len(seg) and _ptp_s < _ptp_e:
                _seg_ptp = seg[_ptp_s:_ptp_e]
                _peak2   = _ptp_s + int(max(np.argmin(_seg_ptp),
                                            np.argmax(_seg_ptp)))
                _peak2ms = (_peak2 - _segs_sb) * 1000 / fs
                _csp = detect_csp_for_trial(
                    seg, fs,
                    np.linspace(-_pre_type_ms, _post_type_ms,
                                len(seg), endpoint=False),
                    CspSettings.from_source(cfg),
                    second_peak_ms=_peak2ms,
                    pre_ms=_pre_type_ms)
                if _csp is not None and auto_lat is not None:
                    _onset_samp = _segs_sb + int(auto_lat * fs / 1000)
                    _csp_start  = _csp[0]
                    if _csp_start > _onset_samp:
                        auc_val = float(_np_trapz(
                            np.abs(seg[_onset_samp:_csp_start]), dx=1/fs))
                        auc_vals_all.append(auc_val)

        # ── outlier / exclusion flags ────────────────────────────────────
        is_removed   = idx in out_set
        is_excluded  = segments_metadata.get(mk, {}).get("exclude", False)
        note_txt     = segments_metadata.get(mk, {}).get("note", "")
        if is_removed:  decision = "Removed"
        elif idx in (out_set or set()): decision = "Kept"
        else:           decision = "Not flagged"
        if is_excluded: decision = "Excluded"

        # ── AUC for clean trials ─────────────────────────────────────────
        if not is_removed and not is_excluded and auc_val is not None:
            auc_vals_clean.append(auc_val)

        # ── shared fields ────────────────────────────────────────────────
        # LAT_COLS indices:
        # [0-3] ID (File/StimType/Stim_Label/Segment)
        # [4-6] timing (Segment_Overall/Stim_Time/Time_Since_Last_Stim)
        # [7-8] limb/measure, [9] PTP, [10] Latency
        # [11] SilentPeriod, [12] SP_MEP_Offset, [13] SP_EMG_Return, [14] cSP_MEP_Ratio
        # [15] AUC, [16-17] PreStimRMS/PreStimPTP
        # [18] PTP_per_PreStimRMS
        # [19-21] Z scores, [22-25] detrended, [26] Outlier_Decision
        # [27-31] normalisation (incl Normalised_PTP_per_PreStimRMS)
        # [32-33] Adjusted_PTP_QR / Normalised_Adjusted_PTP_QR
        # [34-42] EMGComp_* diagnostics, [43] Manual_Note
        # cSP/MEP ratio (Orth & Rothwell 2004): cSP duration(ms) / MEP PTP(mV), in ms/mV
        _csp_mep = round(float(silent_dur) / float(man_ptp), 4) \
                   if (isinstance(silent_dur, (int, float)) and silent_dur >= 0
                       and man_ptp is not None and float(man_ptp) > 0) else None
        # StimType reports the stimulus, not the group key: a row for A·pre
        # says StimType=A, Condition=pre. The label still comes from the group,
        # each condition being separately configurable on tab 1a.
        _base_stim, _cond = split_group_key(cfg, stim_type)
        common = [
            name, _base_stim, custom_labels.get(stim_type, ""), idx + 1,  # [0-3]
            None, None, None,                                              # [4-6] timing
            cfg.limb, cfg.measure,                                        # [7-8]
            None, None, None,                                  # PTP / MEP_RMS / Latency
            silent_dur, sp_mep_offset_ms, sp_emg_return_ms, _csp_mep,   # [11-14] SP
            auc_val,                                                      # [15]
            round(rms_all[idx], 4), round(preptp_all[idx], 4),           # [16-17] PreStimRMS/PTP
            None,                                                         # [18] PTP_per_PreStimRMS
            round(rms_z_full[idx], 3), None, None,                       # [19-21] Z_PreStimRMS/Within/Pooled
            None, None, None, None,                                       # [22-25] Detrended x4
            decision,                                                     # [26] Outlier_Decision
            None, None, None, None, None,                                 # [27-31] norm cols (incl Norm_PTP_per_PreStimRMS)
            None, None,                                                   # [32-33] Adjusted_PTP_QR / Normalised_Adjusted_PTP_QR
            None, None, None, None, None, None, None, None, None,         # [34-42] EMGComp_* diagnostics
            note_txt,                                                     # [43] Manual_Note
        ]

        auto_row   = common.copy()
        manual_row = common.copy()
        auto_row  [_C_PTP] = round(auto_ptp, 2)
        auto_row  [_C_LAT] = round(auto_lat, 2) if auto_lat is not None else "Not Detected"
        manual_row[_C_PTP] = round(man_ptp, 2)
        manual_row[_C_LAT] = round(man_lat, 2) if man_lat is not None else "Not Detected"

        # RMS over the PTP window. Identical for the auto and manual rows by
        # design: the manual override moves two peak markers, which is not a
        # window, so there is no manual counterpart to a window statistic.
        _mep_rms = compute_rms(seg, ptp_start_idx, ptp_end_idx)
        mep_rms_vals.append(_mep_rms)
        auto_row  [_C_MEP_RMS] = round(_mep_rms, 4)
        manual_row[_C_MEP_RMS] = round(_mep_rms, 4)

        # PTP / PreStimRMS per trial (indices 30-31 in LAT_COLS)
        _rms_val = rms_all[idx]
        _col_ptp_rms  = LAT_COLS.index("PTP_per_PreStimRMS")
        _col_norm_rms = LAT_COLS.index("Normalised_PTP_per_PreStimRMS")
        auto_row  [_col_ptp_rms] = round(float(auto_ptp) / _rms_val, 4) \
                                   if _rms_val > 0 else None
        manual_row[_col_ptp_rms] = round(float(man_ptp)  / _rms_val, 4) \
                                   if _rms_val > 0 else None
        # Normalised_PTP_per_PreStimRMS filled later by apply_normalisation

        # ── New trailing columns ─────────────────────────────────────────
        # `common` is a positional list sized to the historical row width, so
        # both rows are widened to the full schema before these are written by
        # name-resolved index.  Same reason _trials_frame pads: appending to
        # LAT_COLS must never desynchronise a builder.
        _w = len(LAT_COLS)
        auto_row   = auto_row   + [""] * (_w - len(auto_row))
        manual_row = manual_row + [""] * (_w - len(manual_row))

        for _row in (auto_row, manual_row):
            _row[_C_COND]    = _cond
            _row[_C_OFF]     = mep_offset_ms
            _row[_C_DUR]     = mep_duration_ms
            _row[_C_OFF_SRC] = mep_offset_src

        if agreement is not None:
            # Retain the per-method breakdown for the comparison report. The
            # detector runs have already happened; without this the individual
            # latencies are computed and thrown away.
            if agreement_out is not None:
                agreement_out[(stim_type, idx)] = agreement
            for _row in (auto_row, manual_row):
                _row[_C_AG_CONS] = agreement.consensus_ms
                _row[_C_AG_SPRD] = agreement.spread_ms
                _row[_C_AG_IQR]  = agreement.iqr_ms
                _row[_C_AG_N]    = agreement.n_detected

        auto_rows.append(auto_row)
        manual_rows.append(manual_row)

        if man_lat is not None:
            latencies.append(man_lat)

    ptp_z_full = (zscore(ptps) if len(ptps) > 1 else np.zeros_like(ptps))
    # Addressed by name, not by literal index. The literal 19 used here wrote the
    # within-condition PTP z-score into Z_PreStimRMS, clobbering the baseline
    # z that `common` had already filled, and leaving Z_PTP_Within empty for the
    # later pooled pass to shift into.
    _col_z_within = LAT_COLS.index("Z_PTP_Within")
    for i, (ar, mr) in enumerate(zip(auto_rows, manual_rows)):
        ar[_col_z_within] = mr[_col_z_within] = round(float(ptp_z_full[i]), 3)

    mep_rms_all = np.asarray(mep_rms_vals, dtype=float)

    # ── summary rows ─────────────────────────────────────────────────────────
    lat_pos     = [v for v in latencies if v is not None and v >= 0]
    sil_pos     = [v for v in silent_durs if isinstance(v, (int, float)) and v >= 0]
    mean_lat    = float(np.mean(lat_pos))    if lat_pos  else np.nan
    std_lat     = float(np.std(lat_pos, ddof=1)) if len(lat_pos) > 1 else np.nan
    mean_sil    = float(np.mean(sil_pos))    if sil_pos  else np.nan
    std_sil     = float(np.std(sil_pos, ddof=1)) if len(sil_pos) > 1 else np.nan
    mean_auc_a  = float(np.mean(auc_vals_all))   if auc_vals_all   else np.nan
    std_auc_a   = float(np.std(auc_vals_all, ddof=1)) if len(auc_vals_all) > 1 else np.nan
    mean_auc_c  = float(np.mean(auc_vals_clean)) if auc_vals_clean else np.nan
    std_auc_c   = float(np.std(auc_vals_clean, ddof=1)) if len(auc_vals_clean) > 1 else np.nan

    # Mask for clean trials
    n_all = len(segs_all)
    clean_mask = [j not in out_set and j not in excluded_set for j in range(n_all)]
    clean_segs = segs_all[clean_mask]
    clean_ptps = (_np_ptp(clean_segs[:, ptp_start_idx:ptp_end_idx], axis=1)
                  if len(clean_segs) else np.array([]))

    lbl = custom_labels.get(stim_type, "")
    header_vals = [name, stim_type, lbl, sum(clean_mask),
                   float(clean_ptps.mean()) if len(clean_ptps) else np.nan,
                   float(clean_ptps.std(ddof=1)) if len(clean_ptps) > 1 else np.nan,
                   float(mep_rms_all[clean_mask].mean()) if clean_mask.count(True) else np.nan,
                   float(mep_rms_all[clean_mask].std(ddof=1)) if clean_mask.count(True) > 1 else np.nan,
                   float(rms_all[clean_mask].mean()) if clean_mask.count(True) else np.nan,
                   float(rms_all[clean_mask].std(ddof=1)) if clean_mask.count(True) > 1 else np.nan,
                   float(preptp_all[clean_mask].mean()) if clean_mask.count(True) else np.nan,
                   float(preptp_all[clean_mask].std(ddof=1)) if clean_mask.count(True) > 1 else np.nan,
                   mean_lat, std_lat, mean_sil, std_sil, mean_auc_c, std_auc_c]

    with_out_row = [name, stim_type, lbl, n_all,
                    float(np.mean(_np_ptp(segs_all[:, ptp_start_idx:ptp_end_idx], axis=1))),
                    float(np.std( _np_ptp(segs_all[:, ptp_start_idx:ptp_end_idx], axis=1))),
                    float(mep_rms_all.mean()),
                    float(mep_rms_all.std(ddof=1)) if len(mep_rms_all) > 1 else np.nan,
                    float(rms_all.mean()),    float(rms_all.std(ddof=1)),
                    float(preptp_all.mean()), float(preptp_all.std(ddof=1)),
                    mean_lat, std_lat, mean_sil, std_sil, mean_auc_a, std_auc_a]

    return auto_rows, manual_rows, header_vals, with_out_row, ptps


def pipeline_compute_pooled_stats(ptps_per_stim, stim_times_per_stim,
                                   latency_rows_auto, latency_rows_manual):
    """Compute pooled Z-scores and linear detrending across all stim types.
    Modifies columns 17–21 of each row in-place.

    Two detrending approaches are computed:
      WithinCond — linear trend fit within each stim type independently,
                   using sequential trial index as x.  Captures condition-
                   specific drift (valid for blocked and randomised designs).
      Session    — single linear trend fit across ALL trials in chronological
                   order (sorted by stim timestamp), regardless of condition.
                   Captures session-level drift such as fatigue or potentiation
                   that spans the whole recording.
    """
    if not ptps_per_stim:
        return

    # ── Pooled Z-score (across all stim types, in insertion order) ───────────
    all_ptps   = np.concatenate(list(ptps_per_stim.values()))
    pooled_z   = (zscore(all_ptps) if len(all_ptps) > 1
                  else np.zeros(len(all_ptps)))
    pz_by_type = {}
    pos = 0
    for st, pa in ptps_per_stim.items():
        pz_by_type[st] = pooled_z[pos:pos + len(pa)]
        pos += len(pa)

    # ── Within-condition detrend (per stim type independently) ───────────────
    wc_det_mean_by_type = {}
    wc_det_z_by_type    = {}
    for st, pa in ptps_per_stim.items():
        n = len(pa)
        x = np.arange(n, dtype=float)
        if n >= 2:
            slp, icp = np.polyfit(x, pa, 1)
            resid    = pa - (slp * x + icp)
            det_mean = resid + float(pa.mean())
            sd       = float(resid.std(ddof=1))
            det_z    = resid / sd if sd > 0 else np.zeros(n)
        else:
            resid    = np.zeros(n)
            det_mean = pa.copy().astype(float)
            det_z    = np.zeros(n)
        wc_det_mean_by_type[st] = det_mean
        wc_det_z_by_type[st]    = det_z

    # ── Session detrend (all trials in chronological order) ──────────────────
    # Build a flat list of (stim_time, ptp, stim_type, local_idx) and sort by time.
    all_trials = []
    for st, pa in ptps_per_stim.items():
        ts = stim_times_per_stim.get(st, np.arange(len(pa), dtype=float))
        for i, (t, p) in enumerate(zip(ts, pa)):
            all_trials.append((float(t), float(p), st, i))
    all_trials.sort(key=lambda r: r[0])

    n_sess   = len(all_trials)
    sess_x   = np.arange(n_sess, dtype=float)
    sess_ptp = np.array([r[1] for r in all_trials])
    grand_mean = float(sess_ptp.mean())

    if n_sess >= 2:
        slp_s, icp_s = np.polyfit(sess_x, sess_ptp, 1)
        sess_resid   = sess_ptp - (slp_s * sess_x + icp_s)
        sess_det     = sess_resid + grand_mean
        sd_s         = float(sess_resid.std(ddof=1))
        sess_det_z   = sess_resid / sd_s if sd_s > 0 else np.zeros(n_sess)
    else:
        sess_det   = sess_ptp.copy()
        sess_det_z = np.zeros(n_sess)

    # Map session detrended values back to (stim_type, local_idx)
    sess_det_mean_by_type = {st: np.empty(len(pa)) for st, pa in ptps_per_stim.items()}
    sess_det_z_by_type    = {st: np.empty(len(pa)) for st, pa in ptps_per_stim.items()}
    for chron_i, (_, _, st, local_i) in enumerate(all_trials):
        sess_det_mean_by_type[st][local_i] = sess_det[chron_i]
        sess_det_z_by_type[st][local_i]    = sess_det_z[chron_i]

    # ── Build timing lookups from chronological order ────────────────────────
    # overall_rank: (stim_type, local_idx) → (chron_rank, stim_time_s, time_since_last_s)
    overall_rank = {}
    for chron_i, (t, _p, st, local_i) in enumerate(all_trials):
        tsl = round(t - all_trials[chron_i - 1][0], 4) if chron_i > 0 else None
        overall_rank[(st, local_i)] = (chron_i + 1, round(t, 4), tsl)

    # ── Write all computed values back into latency rows (in-place) ──────────
    cum_off = 0
    for st, pa in reversed(list(ptps_per_stim.items())):
        n = len(pa)
        for off in range(1, n + 1):
            ti    = n - off
            abs_i = cum_off + off
            pz  = round(float(pz_by_type[st][ti]),            3)
            wdv = round(float(wc_det_mean_by_type[st][ti]),   4)
            wdz = round(float(wc_det_z_by_type[st][ti]),      3)
            sdv = round(float(sess_det_mean_by_type[st][ti]), 4)
            sdz = round(float(sess_det_z_by_type[st][ti]),    3)
            seg_ov, stim_t, tsl = overall_rank.get((st, ti), (None, None, None))
            for rows in (latency_rows_auto, latency_rows_manual):
                rows[-abs_i][_C_SEG_OV] = seg_ov
                rows[-abs_i][_C_STIM_T] = stim_t
                rows[-abs_i][_C_TSL]    = tsl
                rows[-abs_i][_C_Z_POOL] = pz
                rows[-abs_i][_C_DET_WC] = wdv
                rows[-abs_i][_C_DET_WZ] = wdz
                rows[-abs_i][_C_DET_SV] = sdv
                rows[-abs_i][_C_DET_SZ] = sdz
        cum_off += n


def pipeline_bootstrap_comparisons(ptp_data, rms_data, preptp_data,
                                    bootstrap_iter, rng):
    """Bootstrap pairwise comparisons between stim types for PTP, RMS, PrePTP.

    Returns list of rows for the bootstrap CSV.
    """
    rows = []
    def _do(metric_dict, label):
        for s1, s2 in itertools.combinations(metric_dict, 2):
            d1, d2 = np.array(metric_dict[s1]), np.array(metric_dict[s2])
            diffs  = np.array([
                np.mean(rng.choice(d1, len(d1), True)) -
                np.mean(rng.choice(d2, len(d2), True))
                for _ in range(bootstrap_iter)
            ])
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
            rows.append([f"{s1} vs {s2}", label,
                         round(float(np.mean(diffs)), 4),
                         round(float(ci_lo), 4),
                         round(float(ci_hi), 4),
                         round(float(p),     4)])
    _do(ptp_data,    "PTP")
    _do(rms_data,    "PreStimRMS")
    _do(preptp_data, "PreStimPTP")
    return rows


# Trial-level CSV column definitions — module-level so all pipeline stages can reference it
#
# Rows are built as positional lists, so any code that writes a single field has
# to know that field's index. Writing literal integers was the source of an
# off-by-one that shifted Z_PTP_Pooled and all four detrended columns one place
# left and left PTP_Detrended_Session_Z permanently empty. Every such write now
# resolves its index from LAT_COLS by name (see the _C_* constants below), so
# inserting or reordering a column can no longer silently corrupt the output.
LAT_COLS = [
    # Identification
    "File", "StimType", "Stim_Label", "Segment",
    "Segment_Overall",           # chronological trial number across all stim types
    "Stim_Time(s)",              # absolute timestamp of the stimulation in the recording
    "Time_Since_Last_Stim(s)",   # ISI from previous stim (any type); blank for first stim
    "Limb", "Measure",
    # Core MEP metrics
    "PTP(mV)",
    "MEP_RMS(mV)",          # RMS over the same window as PTP; integrates the
                            # whole response rather than two extreme samples
    "Latency(ms)",
    "cSP_Duration(ms)",     # duration MEP offset → EMG return
    "cSP_MEP_Offset(ms)",    # time of MEP offset (cSP onset) re: stim
    "cSP_EMG_Return(ms)",    # time of EMG return (cSP offset) re: stim
    "cSP_MEP_Ratio(ms/mV)",  # cSP duration(ms) / PTP(mV), Orth & Rothwell 2004
    "AUC(mV*s)",
    # Pre-stimulus baseline
    "PreStimRMS", "PreStimPTP",
    # PTP normalised to baseline EMG
    "PTP_per_PreStimRMS",            # PTP(mV) / PreStimRMS
    # Z-scores and detrended values
    "Z_PreStimRMS", "Z_PTP_Within", "Z_PTP_Pooled",
    "PTP_Detrended_WithinCond(mV)", "PTP_Detrended_WithinCond_Z",
    "PTP_Detrended_Session(mV)",    "PTP_Detrended_Session_Z",
    # Trial status
    "Outlier_Decision",
    # Normalisation (blank if not configured)
    "Reference_Type",     # which condition was the denominator
    "Reference_Mean(mV)", # mean amplitude of reference (plateau-detected if applicable)
    "Reference_N",        # trials contributing to reference mean
    "Normalised_PTP",     # PTP / Reference_Mean  (raw ratio)
    "Normalised_PTP_per_PreStimRMS", # Normalised_PTP / PreStimRMS (blank until normalised)
    # EMG excitability compensation (Carson 2026, quantile regression).
    # Blank for trials excluded from the sample (Removed / Excluded).
    "Adjusted_PTP_QR(mV)",            # excitability-compensated PTP (QR)
    "Normalised_Adjusted_PTP_QR",     # Adjusted_PTP_QR / raw reference mean
    "EMGComp_Method",                 # "qr" | fallback flag (insufficient_n, degenerate_rms, ...)
    "EMGComp_N",                      # trials contributing to the per-sample fit
    "EMGComp_Slope",                  # QR slope, PTP units per PreStimRMS unit
    "EMGComp_Intercept",              # QR intercept (mV)
    "EMGComp_InterceptWeight",        # Wi in [0, 1]
    "EMGComp_Adjustment(mV)",         # reference − median(fitted): the shift applied
    "EMGComp_PseudoR2",               # Koenker–Machado pseudo-R²
    "EMGComp_Rho_Pre",                # Spearman rho(PTP, RMS) — association removed
    "EMGComp_Rho_Post",               # Spearman rho(adjusted, RMS) — ≈ 0 when adequate
    # Annotations — always last
    "Manual_Note",
    # Acquisition-quality flags.  Appended after Manual_Note so that row
    # builders emitting the older, shorter row remain valid; rows are padded
    # to full width in pipeline_write_outputs.
    "Clipped",          # trial contains ADC-saturated samples -> PTP underestimated
    "Units_Assumed",    # amplitude unit was not declared by the file
    # MEP offset and duration.  Appended, like the flags above, so that any row
    # builder emitting a shorter row stays valid; _trials_frame pads.
    #
    # NOTE ON cSP_MEP_Offset(ms): when cSP detection is enabled for a stimulus
    # type and a silent period is found, MEP_Offset(ms) carries the SAME value,
    # because the end of the MEP and the start of the silent period are one
    # event.  MEP_Offset_Source reads "csp_start" on exactly those rows.  Use
    # MEP_Offset(ms) in new analyses: it is also populated at rest, where there
    # is no silent period and cSP_MEP_Offset(ms) is blank.
    "MEP_Offset(ms)",        # end of the evoked response, re: stim
    "MEP_Duration(ms)",      # MEP_Offset(ms) - Latency(ms)
    "MEP_Offset_Source",     # manual | csp_start | envelope | none
    # Per-trial agreement between onset detectors.  Populated only when
    # cfg.onset_agreement is on; blank otherwise.  High disagreement flags a
    # trial for manual review; it does NOT mean the reported onset is wrong.
    "Onset_MethodsMedian(ms)",   # median across the member method detectors
    "Onset_Disagreement(ms)",# max - min across members
    "Onset_IQR(ms)",         # interquartile range; robust to one stray member
    "Onset_Methods_N",       # members that returned a latency
    # The condition this trial was assigned to, blank where none was. Appended
    # rather than placed beside StimType because `common` above is a positional
    # list of literals padded to this width: inserting mid-list would shift
    # every field after it, which is the six-column misalignment this schema
    # already learned once.
    #
    # A separate column rather than part of StimType, so that a timepoint is a
    # factor the group analysis can model rather than a substring to be parsed
    # out of a name.
    "Condition",
]

# Column indices resolved by name. Any code writing a single field into a row
# must use these rather than a literal, so the row layout and the writers can
# never disagree again.
_C_COND    = LAT_COLS.index("Condition")
_C_SEG_OV  = LAT_COLS.index("Segment_Overall")
_C_STIM_T  = LAT_COLS.index("Stim_Time(s)")
_C_TSL     = LAT_COLS.index("Time_Since_Last_Stim(s)")
_C_PTP     = LAT_COLS.index("PTP(mV)")
_C_MEP_RMS = LAT_COLS.index("MEP_RMS(mV)")
_C_LAT     = LAT_COLS.index("Latency(ms)")
_C_Z_POOL  = LAT_COLS.index("Z_PTP_Pooled")
_C_DET_WC  = LAT_COLS.index("PTP_Detrended_WithinCond(mV)")
_C_DET_WZ  = LAT_COLS.index("PTP_Detrended_WithinCond_Z")
_C_DET_SV  = LAT_COLS.index("PTP_Detrended_Session(mV)")
_C_DET_SZ  = LAT_COLS.index("PTP_Detrended_Session_Z")
_C_OFF     = LAT_COLS.index("MEP_Offset(ms)")
_C_DUR     = LAT_COLS.index("MEP_Duration(ms)")
_C_OFF_SRC = LAT_COLS.index("MEP_Offset_Source")
_C_AG_CONS = LAT_COLS.index("Onset_MethodsMedian(ms)")
_C_AG_SPRD = LAT_COLS.index("Onset_Disagreement(ms)")
_C_AG_IQR  = LAT_COLS.index("Onset_IQR(ms)")
_C_AG_N    = LAT_COLS.index("Onset_Methods_N")


def _trials_frame(rows):
    """Build the trial-level DataFrame, padding rows to the LAT_COLS width.

    Row builders emit positional lists.  Appending a column to the end of
    LAT_COLS therefore desynchronises every builder that has not been updated,
    and pandas raises "N columns passed, passed data had M columns" at write
    time — after the whole analysis has run.  Padding here keeps a schema
    addition from breaking the run, and keeps every construction site
    consistent: build frames through this function, never pd.DataFrame(...,
    columns=LAT_COLS) directly.
    """
    w = len(LAT_COLS)
    padded = [list(r) + [""] * (w - len(r)) if len(r) < w else list(r)[:w]
              for r in rows]
    return pd.DataFrame(padded, columns=LAT_COLS)


def pipeline_write_outputs(latency_manual, results_out, bids_prefix,
                           channel_label=None, column_selection=None,
                           log_callback=None):
    """Write all result CSVs to results_out directory.

    Outputs
    -------
    <prefix>_trials.csv               — trial-level data (all metrics, clean trials)
    <prefix>_trials_with_outliers.csv — same including outlier trials
    <prefix>_summary.csv              — mean ± SD per stim type (clean trials only)
    <prefix>_summary_with_outliers.csv — same including outlier trials
    <prefix>_trials_selected.csv      — column-narrowed COPY of trials.csv,
                                        written only when column_selection is
                                        given

    ``column_selection`` is a list of group keys from :mod:`column_groups`.
    It narrows nothing but the extra copy: trials.csv always carries every
    column, and LAT_COLS is neither read nor reordered here.

    Returns
    -------
    list | None — the group keys actually written to the narrowed file, after
    dependency resolution, or None when no narrowed file was written. The
    caller records this in the sidecar so Stage 2 compares stated intent
    rather than guessing from CSV headers.
    """
    # Summary file headers — mirrors LAT_COLS with mean/SD for every metric,
    # plus trial counts so both files report the same variables.
    SUM_HDR = [
        "File", "StimType", "Stim_Label",
        "N_Total", "N_Included", "N_Outliers",
        # Core MEP
        "Mean_PTP(mV)", "SD_PTP(mV)",
        "Mean_MEP_RMS(mV)", "SD_MEP_RMS(mV)",
        "Mean_Latency(ms)", "SD_Latency(ms)",
        # cSP
        "Mean_cSP_Duration(ms)", "SD_cSP_Duration(ms)",
        "Mean_cSP_MEP_Offset(ms)", "SD_cSP_MEP_Offset(ms)",
        "Mean_cSP_EMG_Return(ms)", "SD_cSP_EMG_Return(ms)",
        "Mean_cSP_MEP_Ratio(ms/mV)", "SD_cSP_MEP_Ratio(ms/mV)",
        # AUC
        "Mean_AUC(mV*s)", "SD_AUC(mV*s)",
        # Baseline
        "Mean_PreStimRMS", "SD_PreStimRMS",
        "Mean_PreStimPTP", "SD_PreStimPTP",
        # PTP normalised to baseline EMG
        "Mean_PTP_per_PreStimRMS", "SD_PTP_per_PreStimRMS",
        # Normalisation
        "Mean_Normalised_PTP", "SD_Normalised_PTP",
        "Reference_Type", "Reference_Mean(mV)", "Reference_N",
        "Mean_Normalised_PTP_per_PreStimRMS", "SD_Normalised_PTP_per_PreStimRMS",
        # Detrended
        "Mean_PTP_Detrended_WithinCond(mV)", "SD_PTP_Detrended_WithinCond(mV)",
        "Mean_PTP_Detrended_Session(mV)",    "SD_PTP_Detrended_Session(mV)",
        # EMG excitability compensation (per-sample diagnostics are constant
        # within a StimType, so the summary reports the single sample value)
        "Mean_Adjusted_PTP_QR(mV)", "SD_Adjusted_PTP_QR(mV)",
        "Mean_Normalised_Adjusted_PTP_QR", "SD_Normalised_Adjusted_PTP_QR",
        "EMGComp_Method", "EMGComp_N", "EMGComp_Slope", "EMGComp_Intercept",
        "EMGComp_InterceptWeight", "EMGComp_Adjustment(mV)",
        "EMGComp_PseudoR2", "EMGComp_Rho_Pre", "EMGComp_Rho_Post",
        # MEP offset / duration
        "Mean_MEP_Offset(ms)", "SD_MEP_Offset(ms)",
        "Mean_MEP_Duration(ms)", "SD_MEP_Duration(ms)",
        # Modal offset provenance across the trials contributing to the mean.
        # Reported rather than averaged because the field is categorical; a
        # sample mixing "csp_start" and "envelope" rows is worth noticing.
        "MEP_Offset_Source_Mode",
        # Onset agreement (blank unless onset_agreement is enabled)
        "Mean_Onset_Disagreement(ms)", "SD_Onset_Disagreement(ms)",
    ]

    def _alpha_sort(df, col):
        cats = sorted(df[col].unique())
        df[col] = pd.Categorical(df[col], categories=cats, ordered=True)
        return df.sort_values([col, "File"]).reset_index(drop=True)

    def _p(name):
        # Routed by family. The NAME is unchanged -- a file has to be
        # identifiable wherever it ends up -- only where it lands.
        from .results_layout import result_path
        return result_path(results_out, f"{bids_prefix}_{name}")

    # ── Trial-level files ─────────────────────────────────────────────────────
    def _tag_channel(df):
        """Name the channel every row came from.

        Written whether or not several channels were analysed. A single-channel
        file that later joins a multi-channel dataset must still be able to say
        which channel it holds, and a column that appears only sometimes is
        worse to work with than one that is always there.
        """
        if df is None or not len(df):
            return df
        if "Channel" not in df.columns and channel_label:
            df.insert(min(1, len(df.columns)), "Channel", str(channel_label))
        return df

    _selected_written = None
    if latency_manual:
        df_all = _alpha_sort(
            _trials_frame(latency_manual),
            "StimType").sort_values(["StimType", "File", "Segment"])
        # trials.csv deliberately carries EVERY trial; Outlier_Decision is the
        # filter column so the analyst keeps control of trial-level modelling.
        # (_trials_with_outliers.csv was retired for this reason.)
        _tag_channel(df_all).to_csv(_p("trials.csv"), index=False)

        # ── Column-narrowed COPY ─────────────────────────────────────────
        # On a copy, at write time, after the full file is already on disk.
        # Nothing upstream knows this exists: the rows were built positionally
        # against LAT_COLS and are untouched, and trials.csv above still holds
        # every column. Narrowing a frame the row builders share would put a
        # variable schema behind ~20 index constants that assert their own
        # positions, which is the one thing this must never do.
        if column_selection is not None:
            from . import column_groups as _cg
            _log = log_callback or (lambda _m: None)
            _keys, _pulled = _cg.resolve(column_selection)
            for _dep, _req, _why in _pulled:
                _log(f"   ↳ '{_cg.GROUP_LABELS.get(_req, _req)}' added: "
                     f"'{_cg.GROUP_LABELS.get(_dep, _dep)}' needs it — {_why}")
            _cols = _cg.select(list(df_all.columns), _keys)
            df_all[_cols].to_csv(_p("trials_selected.csv"), index=False)
            _selected_written = sorted(_keys)
            _log(f"   ↳ Selected-column copy: {len(_cols)} of "
                 f"{len(df_all.columns)} columns → "
                 f"{os.path.basename(_p('trials_selected.csv'))}")

    # ── Summary files — build from trial-level data for consistency ───────────
    # This ensures summary and trial files always report the same variables.
    if latency_manual:
        def _mn(vals):
            try:
                v = pd.to_numeric(vals, errors='coerce')
                v = v.dropna().tolist() if hasattr(v, 'dropna') else [x for x in v if x == x]
                return float(np.nanmean(v)) if v else np.nan
            except Exception:
                return np.nan

        def _sd(vals):
            try:
                v = pd.to_numeric(vals, errors='coerce')
                v = v.dropna().tolist() if hasattr(v, 'dropna') else [x for x in v if x == x]
                return float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
            except Exception:
                return np.nan

        def _col(grp, col):
            """Extract a numeric column safely as a list of floats."""
            return pd.to_numeric(grp[col], errors='coerce').dropna().tolist()

        def _str_col(grp, col):
            """Get first non-null string value from a column."""
            vals = grp[col].dropna()
            return vals.iloc[0] if len(vals) else ""

        def _mode_col(grp, col):
            """
            Most common value of a categorical column.

            Used for MEP_Offset_Source, where averaging is meaningless. Ties
            resolve to whichever value pandas orders first, which is acceptable
            because a tie is itself the signal worth noticing: it means the
            sample mixes provenances.
            """
            if col not in grp.columns:
                return ""
            vals = grp[col].replace("", pd.NA).dropna()
            if not len(vals):
                return ""
            counts = vals.value_counts()
            return "" if not len(counts) else str(counts.index[0])

        def _build_summary(df):
            rows = []
            for (fname, st, lbl), grp in df.groupby(
                    ["File", "StimType", "Stim_Label"], sort=False):
                clean = grp[~grp["Outlier_Decision"].isin(EXCLUDED_DECISIONS)]
                n_tot = len(grp)
                n_inc = len(clean)
                n_out = n_tot - n_inc
                rows.append([
                    fname, st, lbl,
                    n_tot, n_inc, n_out,
                    _mn(_col(clean,"PTP(mV)")),         _sd(_col(clean,"PTP(mV)")),
                    _mn(_col(clean,"MEP_RMS(mV)")),      _sd(_col(clean,"MEP_RMS(mV)")),
                    _mn(_col(clean,"Latency(ms)")),      _sd(_col(clean,"Latency(ms)")),
                    _mn(_col(clean,"cSP_Duration(ms)")), _sd(_col(clean,"cSP_Duration(ms)")),
                    _mn(_col(clean,"cSP_MEP_Offset(ms)")),_sd(_col(clean,"cSP_MEP_Offset(ms)")),
                    _mn(_col(clean,"cSP_EMG_Return(ms)")),_sd(_col(clean,"cSP_EMG_Return(ms)")),
                    _mn(_col(clean,"cSP_MEP_Ratio(ms/mV)")), _sd(_col(clean,"cSP_MEP_Ratio(ms/mV)")),
                    _mn(_col(clean,"AUC(mV*s)")),        _sd(_col(clean,"AUC(mV*s)")),
                    _mn(_col(clean,"PreStimRMS")),       _sd(_col(clean,"PreStimRMS")),
                    _mn(_col(clean,"PreStimPTP")),       _sd(_col(clean,"PreStimPTP")),
                    _mn(_col(clean,"PTP_per_PreStimRMS")), _sd(_col(clean,"PTP_per_PreStimRMS")),
                    _mn(_col(clean,"Normalised_PTP")),   _sd(_col(clean,"Normalised_PTP")),
                    _str_col(clean,"Reference_Type"),
                    _mn(_col(clean,"Reference_Mean(mV)")),
                    _mn(_col(clean,"Reference_N")),
                    _mn(_col(clean,"Normalised_PTP_per_PreStimRMS")),
                    _sd(_col(clean,"Normalised_PTP_per_PreStimRMS")),
                    _mn(_col(clean,"PTP_Detrended_WithinCond(mV)")),
                    _sd(_col(clean,"PTP_Detrended_WithinCond(mV)")),
                    _mn(_col(clean,"PTP_Detrended_Session(mV)")),
                    _sd(_col(clean,"PTP_Detrended_Session(mV)")),
                    # EMG excitability compensation
                    _mn(_col(clean,"Adjusted_PTP_QR(mV)")),
                    _sd(_col(clean,"Adjusted_PTP_QR(mV)")),
                    _mn(_col(clean,"Normalised_Adjusted_PTP_QR")),
                    _sd(_col(clean,"Normalised_Adjusted_PTP_QR")),
                    _str_col(clean,"EMGComp_Method"),
                    _mn(_col(clean,"EMGComp_N")),
                    _mn(_col(clean,"EMGComp_Slope")),
                    _mn(_col(clean,"EMGComp_Intercept")),
                    _mn(_col(clean,"EMGComp_InterceptWeight")),
                    _mn(_col(clean,"EMGComp_Adjustment(mV)")),
                    _mn(_col(clean,"EMGComp_PseudoR2")),
                    _mn(_col(clean,"EMGComp_Rho_Pre")),
                    _mn(_col(clean,"EMGComp_Rho_Post")),
                    # MEP offset / duration
                    _mn(_col(clean,"MEP_Offset(ms)")),
                    _sd(_col(clean,"MEP_Offset(ms)")),
                    _mn(_col(clean,"MEP_Duration(ms)")),
                    _sd(_col(clean,"MEP_Duration(ms)")),
                    _mode_col(clean, "MEP_Offset_Source"),
                    # Onset agreement
                    _mn(_col(clean,"Onset_Disagreement(ms)")),
                    _sd(_col(clean,"Onset_Disagreement(ms)")),
                ])
            return pd.DataFrame(rows, columns=SUM_HDR)

        df_all = _trials_frame(latency_manual)
        df_clean_only = df_all[~df_all["Outlier_Decision"].isin(EXCLUDED_DECISIONS)]

        _build_summary(df_clean_only) \
            .pipe(_tag_channel).to_csv(_p("summary.csv"),               index=False)
        _build_summary(df_all) \
            .pipe(_tag_channel).to_csv(_p("summary_with_outliers.csv"), index=False)

    return _selected_written


def pipeline_generate_plots(trace_stats, segments_metadata,
                             color_map, custom_labels,
                             figures_out, figures_all, bids_prefix, name, unit,
                             enable_individual, cfg: PipelineConfig):
    """Save the combined trace figure and (optionally) per-stim-type figures.

    Each entry of trace_stats carries its own latency axis, because stimulus
    types may be epoched over different windows and one shared axis could not
    describe them. Matplotlib draws lines of differing length on shared axes
    without complaint, and the result is more honest than stretching a short
    window to fill the plot: a type measured over less time occupies less of
    the axis.

    The former plot_included filter is gone. It selected which types appeared
    on the combined figure and did nothing else -- not the analysis, not the
    per-type figures, not any CSV -- and a control that changes one image was
    not worth a column on a table where the epoch window now needs two.
    """
    def _ylab(base="EMG"):
        return f"{base} ({unit})" if unit else base

    # Combined figure
    fig = matplotlib.figure.Figure(figsize=(12, 6))
    ax  = fig.add_subplot(111)
    for stim_type, segments, emg_segments, mean_trace, mean_ptp, t_axis \
            in trace_stats:
        color      = color_map.get(stim_type, "gray")
        label_name = custom_labels.get(stim_type, stim_type)
        for s in emg_segments:
            ax.plot(t_axis, s, color=color, alpha=0.2, linewidth=0.5)
        ax.plot(t_axis, mean_trace, color=color, linewidth=3,
                label=f"{label_name} Mean PTP: {mean_ptp:.2f}")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(f"{name} – EMG Responses")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel(_ylab("EMG Amplitude"))
    ax.legend()
    out_path = os.path.join(figures_out, f"{bids_prefix}_traces.png")
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_figure(out_path, dpi=600)
    fig.clf()

    if enable_individual:
        for stim_type, segments, emg_segments, mean_trace, mean_ptp, t_axis \
                in trace_stats:
            color      = color_map.get(stim_type, "gray")
            label_name = custom_labels.get(stim_type, stim_type)
            fig_i = matplotlib.figure.Figure(figsize=(12, 6))
            ax_i  = fig_i.add_subplot(111)
            for s in emg_segments:
                ax_i.plot(t_axis, s, color=color, alpha=0.2, linewidth=0.5)
            ax_i.plot(t_axis, mean_trace, color=color, linewidth=3,
                      label=f"{label_name} Mean PTP: {mean_ptp:.2f}")
            ax_i.axvline(0, color="black", linestyle="--")
            ax_i.set_title(f"{name} – {label_name} Responses")
            ax_i.set_xlabel("Latency (ms)")
            ax_i.set_ylabel(_ylab("EMG Amplitude"))
            ax_i.legend()
            safe = label_name.replace(" ", "_")
            out_i = os.path.join(figures_out, f"{bids_prefix}_stim-{safe}_traces.png")
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig_i).print_figure(out_i, dpi=600)
            fig_i.clf()

    return out_path   # return combined figure path for auto-open

# ─────────────────────────────────────────────────────────────────────────────
# Averaged-waveform analysis (optional mode) — build step 1 of 4
#
# When average_mode is enabled, each condition (stim type) is collapsed to a
# single mean waveform over its outlier-screened trials and measured ONCE,
# mirroring the marker consumption of pipeline_quantify_segments so a mean is
# measured identically to how a trial is measured. Produces a reduced column
# set (single-waveform measures only); detrend/z-score/bootstrap columns are
# omitted because they are inherently multi-trial.
#
# All arrays here are INSPECTOR-space (full prestim_ms pre-stim), i.e. the
# exact segments fed to the inspector, so any user-adjusted marker indices
# apply directly without offset conversion.
# ─────────────────────────────────────────────────────────────────────────────

AVERAGED_COLS = [
    "File", "StimType", "Stim_Label", "Limb", "Measure",
    "N_Trials_Averaged",
    "PTP(mV)", "MEP_RMS(mV)", "Latency(ms)",
    "cSP_Duration(ms)", "cSP_MEP_Offset(ms)", "cSP_EMG_Return(ms)",
    "cSP_MEP_Ratio(ms/mV)",
    "AUC(mV*s)",
    "PreStimRMS", "PreStimPTP",
    "PTP_per_PreStimRMS",
    # Normalisation (Stage 1a reference_map; blank if not configured)
    "Reference_Type", "Reference_Mean(mV)", "Reference_N",
    "Normalised_PTP", "Normalised_PTP_per_PreStimRMS",
    "Manual_Note",
]


def pipeline_assemble_condition_means(stats_per_type, emg, time, fs, cfg,
                                      log_callback=print):
    """Collapse each condition to a mean waveform over its clean trials.

    Built in INSPECTOR space (full prestim_ms pre-stim) directly from the raw
    recording, indexed by the per-trial stim timestamps in
    stats_per_type[...]["stim_times_s"] (which align 1:1 with segs_all and the
    outlier_set). Outlier screening therefore stays correctly aligned no matter
    how many edge trials the extraction windows drop, so the result never
    depends on a segments_inspector count match.

    Returns
    -------
    {stim_type: {"mean": np.ndarray, "individual": np.ndarray[n_clean, L],
                 "n": int}}   Conditions with no usable clean trials are skipped.
    """
    _insp_sb = int(cfg.prestim_ms * fs / 1000)
    means = {}
    for stim_type, info in stats_per_type.items():
        # Per type, matching the analysis and the Inspector: an average built
        # over a different window from the trials it averages is not their
        # average.
        _insp_sa = window_samples(cfg, stim_type, fs)[1]
        stim_ts = info.get("stim_times_s", [])
        out     = info.get("outlier_set", set()) or set()
        clean   = []
        # Same delay as the analysis and the Inspector, for the same reason:
        # a condition average built from a differently aligned epoch is not the
        # average of the trials being measured.
        _mean_delay = int(round(
            float(cfg.delay_ms_map.get(stim_type, 0.0)) * fs / 1000.0))
        for _i, _t0 in enumerate(stim_ts):
            if _i in out:
                continue
            _ix  = int(np.argmin(np.abs(time - _t0))) + _mean_delay
            if _ix < 0 or _ix >= len(emg):
                continue
            _seg = emg[max(0, _ix - _insp_sb):_ix + _insp_sa]
            if len(_seg) == _insp_sb + _insp_sa:
                clean.append(_seg)
        if not clean:
            log_callback(f"⚠️  Averaged mode: '{stim_type}' has no clean trials "
                         f"with a full pre-stim window — skipped.")
            continue
        stack = np.asarray(clean, dtype=float)
        means[stim_type] = {"mean": stack.mean(axis=0),
                            "individual": stack,
                            "n": stack.shape[0]}
    return means


def pipeline_quantify_averaged(means, segments_metadata, fs, cfg,
                               custom_labels, name, log_callback=print):
    """Measure each condition's mean waveform, returning reduced-column rows.

    Inspector markers (keyed ``(stim_type, 0)``) take precedence; any measure
    not supplied by the inspector is auto-detected on the mean. Indices are
    inspector-space (stim at ``prestim_ms * fs / 1000``).
    """
    # prestim RMS/PTP computed inline (same convention as
    # pipeline_quantify_segments) — no cross-module import needed.

    insp_sb = int(cfg.prestim_ms * fs / 1000)          # stim index in the mean
    ptp_s   = insp_sb + int(cfg.ptp_start * fs / 1000)
    ptp_e   = insp_sb + int(cfg.ptp_end   * fs / 1000)
    meta_all = segments_metadata or {}

    rows = []
    for stim_type, info in means.items():
        seg = np.asarray(info["mean"], dtype=float)
        n_clean = info["n"]
        L = len(seg)
        meta = meta_all.get((stim_type, 0), {})
        label = custom_labels.get(stim_type, stim_type) if custom_labels else stim_type

        def _clamp(i):
            return min(max(0, int(i)), L - 1)

        # ── PTP ──────────────────────────────────────────────────────────
        if "ptp_max_idx" in meta and "ptp_min_idx" in meta:
            ptp = float(seg[_clamp(meta["ptp_max_idx"])]
                        - seg[_clamp(meta["ptp_min_idx"])])
            # Peak-to-peak is a magnitude and cannot be negative. A negative
            # value means the stored "max" sample sits below the stored "min"
            # one -- which happens when marker indices from one channel are
            # applied to another, since an index is a position in one
            # particular waveform. Take the magnitude rather than writing an
            # impossible number: a wrong-but-positive value is still wrong, but
            # a negative one propagates into normalisation and z-scores as
            # though it were meaningful.
            if ptp < 0:
                ptp = abs(ptp)
        else:
            _e = min(ptp_e, L); _s = min(ptp_s, max(_e - 1, 0))
            ptp = compute_ptp(seg, _s, _e) if _e > _s else float("nan")

        # ── MEP RMS over the analysis window ─────────────────────────────
        # Always the window, never the peak markers: RMS is a window statistic
        # and a manual two-marker PTP override has no window equivalent.
        _re = min(ptp_e, L); _rs = min(ptp_s, max(_re - 1, 0))
        mep_rms = compute_rms(seg, _rs, _re) if _re > _rs else float("nan")

        # ── Latency / onset ──────────────────────────────────────────────
        onset_idx = None
        if "onset_idx" in meta:
            onset_idx = int(meta["onset_idx"])
            lat = (onset_idx - insp_sb) * 1000.0 / fs
        else:
            _min, _max = cfg.latency_map.get(stim_type,
                                             (cfg.ptp_start, cfg.ptp_end))
            lat = _detect_onset_dispatch(seg, fs, cfg, _min, _max,
                                         stim_type=stim_type)
            if lat is not None:
                onset_idx = insp_sb + int(round(lat * fs / 1000))
        if lat is not None and lat <= 0:
            lat, onset_idx = None, None

        # ── cSP ──────────────────────────────────────────────────────────
        csp_dur = csp_off = csp_ret = None
        csp_start_idx = None
        if "silent_start_idx" in meta and "silent_end_idx" in meta:
            ss = int(meta["silent_start_idx"]); se = int(meta["silent_end_idx"])
            csp_dur = round((se - ss) * 1000.0 / fs, 2)
            csp_off = round((ss - insp_sb) * 1000.0 / fs, 2)
            csp_ret = round((se - insp_sb) * 1000.0 / fs, 2)
            csp_start_idx = ss
        elif stim_type in cfg.csp_types:
            from .detection import detect_csp_bootstrap as _dcsp
            _e = min(ptp_e, L); _s = min(ptp_s, max(_e - 1, 0))
            if _e > _s:
                _win = seg[_s:_e]
                _peak = _s + int(max(np.argmin(_win), np.argmax(_win)))
                _peak2ms = (_peak - insp_sb) * 1000.0 / fs
                _t = np.linspace(-cfg.prestim_ms,
                                 -cfg.prestim_ms + L * 1000.0 / fs,
                                 L, endpoint=False)
                _csp = _dcsp(seg, fs, _t,
                             pre_ms=cfg.prestim_ms,
                             search_start_ms=_peak2ms,
                             search_end_ms=cfg.csp_search_end_ms,
                             min_silence_ms=cfg.csp_min_silence_ms,
                             min_return_ms=cfg.csp_min_return_ms,
                             criterion=cfg.csp_criterion,
                             significance=cfg.csp_significance,
                             n_boot=cfg.csp_n_boot)
                if _csp is not None:
                    ss, se = int(_csp[0]), int(_csp[1])
                    csp_dur = round((se - ss) * 1000.0 / fs, 2)
                    csp_off = round((ss - insp_sb) * 1000.0 / fs, 2)
                    csp_ret = round((se - insp_sb) * 1000.0 / fs, 2)
                    csp_start_idx = ss

        # ── AUC ──────────────────────────────────────────────────────────
        auc_val = None
        if "auc_start_idx" in meta and "auc_end_idx" in meta:
            a0, a1 = _clamp(meta["auc_start_idx"]), _clamp(meta["auc_end_idx"])
            if a1 > a0:
                auc_val = float(_np_trapz(np.abs(seg[a0:a1]), dx=1 / fs))
        elif onset_idx is not None and csp_start_idx is not None \
                and csp_start_idx > onset_idx:
            auc_val = float(_np_trapz(np.abs(seg[onset_idx:csp_start_idx]),
                                      dx=1 / fs))

        # ── baseline / ratios ────────────────────────────────────────────
        # Clear the stimulus artefact from the baseline window, matching
        # pipeline_extract_segments (Carson 2026: ends 3 ms before the pulse).
        _guard = int(round(max(cfg.rms_guard_ms, 0.0) * fs / 1000))
        _pre_e = max(0, insp_sb - _guard)
        prestim = seg[:_pre_e] if _pre_e > 0 else seg[:1]
        rms = pipeline_prestim_rms(prestim, cfg) if len(prestim) else 0.0
        preptp = float(_np_ptp(prestim)) if len(prestim) else 0.0
        ptp_per_rms = (ptp / rms) if (rms and not np.isnan(ptp)) else None
        csp_ratio = (round(csp_dur / ptp, 4)
                     if (csp_dur is not None and ptp and not np.isnan(ptp)
                         and ptp != 0) else None)

        rows.append({
            "File": name, "StimType": stim_type, "Stim_Label": label,
            "Limb": cfg.limb, "Measure": cfg.measure,
            "N_Trials_Averaged": n_clean,
            "PTP(mV)": round(ptp, 6) if not np.isnan(ptp) else None,
            "MEP_RMS(mV)": round(mep_rms, 6) if not np.isnan(mep_rms) else None,
            "Latency(ms)": round(lat, 2) if lat is not None else "Not Detected",
            "cSP_Duration(ms)": csp_dur if csp_dur is not None else "Not Marked",
            "cSP_MEP_Offset(ms)": csp_off,
            "cSP_EMG_Return(ms)": csp_ret,
            "cSP_MEP_Ratio(ms/mV)": csp_ratio,
            "AUC(mV*s)": round(auc_val, 6) if auc_val is not None else None,
            "PreStimRMS": round(rms, 6),
            "PreStimPTP": round(preptp, 6),
            "PTP_per_PreStimRMS": round(ptp_per_rms, 4)
                                  if ptp_per_rms is not None else None,
            "Reference_Type": "",
            "Reference_Mean(mV)": None,
            "Reference_N": None,
            "Normalised_PTP": None,
            "Normalised_PTP_per_PreStimRMS": None,
            "Manual_Note": meta.get("note", ""),
        })
    return rows


def pipeline_write_averaged(rows, results_out, bids_prefix):
    """Write the reduced-column averaged-waveform CSV.

    Output
    ------
    <prefix>_averaged.csv  — one row per condition (measured on the mean).
    """
    df = pd.DataFrame(rows, columns=AVERAGED_COLS)
    from .results_layout import result_path
    out_path = result_path(results_out, f"{bids_prefix}_averaged.csv")
    df.to_csv(out_path, index=False)
    return out_path


def pipeline_normalise_averaged(rows, cfg, log_callback=print):
    """Fill normalisation columns on averaged rows (Stage 1a reference_map).

    Simplified averaged-mode normalisation: each condition contributes a single
    value — the PTP of its mean waveform — so a referenced condition is
    normalised to the reference condition's mean-waveform PTP (no plateau
    detection, which needs multiple trials):

        Normalised_PTP = PTP(mean of X) / PTP(mean of reference)

    Applied within one file's conditions (references are within-recording).
    Fills the row dicts in place; rows without a configured reference are left
    blank, exactly as the per-trial path leaves them.
    """
    if not rows or not cfg.reference_map:
        return
    ptp_by_type = {}
    n_by_type   = {}
    for r in rows:
        st = r["StimType"]
        try:
            ptp_by_type[st] = float(r["PTP(mV)"])
        except (TypeError, ValueError):
            ptp_by_type[st] = None
        n_by_type[st] = r.get("N_Trials_Averaged")
    for r in rows:
        st  = r["StimType"]
        ref = cfg.reference_map.get(st, "")
        if not ref or ref not in ptp_by_type:
            continue
        ref_ptp = ptp_by_type.get(ref)
        try:
            ptp_f = float(r["PTP(mV)"])
        except (TypeError, ValueError):
            continue
        if ref_ptp is None or ref_ptp <= 0:
            log_callback(f"⚠️  Averaged normalisation: reference '{ref}' "
                         f"for '{st}' has no usable mean — left blank.")
            continue
        _norm = ptp_f / ref_ptp
        r["Reference_Type"]     = f"{ref}_averaged"
        r["Reference_Mean(mV)"] = round(ref_ptp, 4)
        r["Reference_N"]        = n_by_type.get(ref)
        r["Normalised_PTP"]     = round(_norm, 4)
        try:
            _rms = float(r.get("PreStimRMS"))
            r["Normalised_PTP_per_PreStimRMS"] = (round(_norm / _rms, 4)
                                                  if _rms > 0 else None)
        except (TypeError, ValueError):
            r["Normalised_PTP_per_PreStimRMS"] = None
        log_callback(f"📐 Averaged '{st}' normalised to '{ref}' "
                     f"(ref PTP {ref_ptp:.3f} mV): {r['Normalised_PTP']}")


def pipeline_generate_averaged_plots(means, fs, cfg, figures_out, bids_prefix,
                                     name, unit, log_callback=print):
    """Save one figure per condition: individual clean traces faint, the mean
    bold on top, and the stimulus line — the output-file companion to the
    inspector's averaged view. Returns the list of written paths."""
    os.makedirs(figures_out, exist_ok=True)
    paths = []
    for stim_type, info in means.items():
        seg = np.asarray(info["mean"], dtype=float)
        L   = len(seg)
        t   = np.linspace(-cfg.prestim_ms, -cfg.prestim_ms + L * 1000.0 / fs,
                          L, endpoint=False)
        colour = cfg.color_map.get(stim_type, "tab:blue")
        label  = cfg.custom_labels.get(stim_type, stim_type)
        fig = matplotlib.figure.Figure(figsize=(12, 6))
        ax  = fig.add_subplot(111)
        for _tr in info["individual"]:
            if len(_tr) == L:
                ax.plot(t, _tr, color=colour, lw=0.4, alpha=0.20, zorder=1)
        ax.plot(t, seg, color=colour, lw=2.0, zorder=3,
                label=f"{label} mean (n={info['n']}, PTP {_np_ptp(seg):.2f})")
        ax.axvline(0, color="k", ls="--")
        ax.set_title(f"{name} – {label} (averaged waveform)")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel(f"EMG ({unit})" if unit else "EMG")
        ax.legend(loc="upper right", frameon=False)
        _pth = os.path.join(figures_out,
                            f"{bids_prefix}_stim-{stim_type}_averaged.png")
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_figure(_pth, dpi=600)
        fig.clf()
        paths.append(_pth)
    return paths


def pipeline_write_segments_bundle(bundle, results_out, bids_prefix,
                                   log_callback=print):
    """Persist the per-trial waveform bundle (<prefix>_segments.npz) that
    add-ons consume.

    Format-agnostic by construction: the waveforms are the normalised segments
    (post format-reader), so the file carries no trace of the original import
    format — only arrays, fs, unit, and the pre/post window needed to rebuild
    the millisecond time axis (0 ms = stimulus). Index-keyed groups support
    mixed fs / length / unit across files in one run.

    Layout
    ------
    manifest_file / manifest_stim / manifest_unit          (str,   one per group)
    manifest_fs / manifest_pre_ms / manifest_post_ms       (float, one per group)
    wav_{i}   [n_trials, n_samples] float32   per-trial waveforms (segs space)
    out_{i}   [n_trials] bool                 outlier flag per trial
    tidx_{i}  [n_trials] int                  trial index within (file, stim)
    stime_{i} [n_trials] float                stim timestamp (s)
    """
    if not bundle:
        return None
    arrays = {}
    g_file, g_stim, g_unit, g_fs, g_pre, g_post = [], [], [], [], [], []
    for i, grp in enumerate(bundle):
        wav = np.asarray(grp["waveforms"], dtype=np.float32)
        arrays[f"wav_{i}"]   = wav
        arrays[f"out_{i}"]   = np.asarray(grp["outlier"], dtype=bool)
        arrays[f"tidx_{i}"]  = np.arange(wav.shape[0], dtype=int)
        arrays[f"stime_{i}"] = np.asarray(grp["stim_time_s"], dtype=float)
        g_file.append(str(grp["file"]));      g_stim.append(str(grp["stim_type"]))
        g_unit.append(str(grp["unit"]))
        g_fs.append(float(grp["fs"]))
        g_pre.append(float(grp["pre_ms"]));   g_post.append(float(grp["post_ms"]))
    # String arrays use numpy unicode dtype (not object) so the file loads
    # without allow_pickle.
    arrays["manifest_file"]    = np.asarray(g_file)
    arrays["manifest_stim"]    = np.asarray(g_stim)
    arrays["manifest_unit"]    = np.asarray(g_unit)
    arrays["manifest_fs"]      = np.asarray(g_fs, dtype=float)
    arrays["manifest_pre_ms"]  = np.asarray(g_pre, dtype=float)
    arrays["manifest_post_ms"] = np.asarray(g_post, dtype=float)
    from .results_layout import result_path
    out_path = result_path(results_out, f"{bids_prefix}_segments.npz")
    np.savez_compressed(out_path, **arrays)
    log_callback(f"💾 Waveform bundle written: {os.path.basename(out_path)} "
                 f"({len(bundle)} group(s))")
    return out_path


def run_pipeline(input_path,
                 pre_ms,
                 post_ms,
                 ptp_start,
                 ptp_end,
                 *,
                 gap_ms_map=None,
                 delay_ms_map=None,
                 delay_source_map=None,
                 review_outliers_cb=None,
                 show_inspector_cb=None,
                 gui_enable_inspector=False,
                 channel_idx=0,
                 event_sources=None, channel_names=None, event_rows=None,
                 condition_map=None,
                 # Display name for the channel, used in logs and in the
                 # Data Inspector's title so a multi-channel run says which
                 # channel is being reviewed.
                 channel_label=None,
                 # True when this run is one of several channels.
                 multi_channel=False,
                 prestim_ms,
                 apply_humbug,
                 humbug_harmonics=6,
                 apply_filter, apply_bandpass, apply_notch,
                 highpass, lowpass, notch_freq, notch_q,
                 filter_order,
                 filter_family="butter", cheby_ripple=1.0,
                 flexible_bandpass=False, hp_order=2, lp_order=2,
                 custom_labels=None, color_map=None, window_map=None,
                 enable_individual_plots=True,
                 log_callback=print,
                 marker_name="Keyboard",
                 enable_outlier_review=True,
                 outlier_threshold=2.0,
                 progress_callback=None,
                 peak_fraction=0.15,
                 min_peak_amplitude=0.05,
                 slope_threshold=0.08,
                 onset_method="peak_fraction",
                 onset_bootstrap_crit=1.96,
                 onset_bootstrap_n=500,
                 onset_bigoni_smooth_ms=0.5,
                 onset_bigoni_min_run_ms=0.5,
                 onset_bigoni_walkback_sd=1.0,
                 # Envelope / CUSUM / consensus / MEP-offset settings, as one
                 # mapping keyed by PipelineConfig field name. A dict rather
                 # than a keyword per parameter: this signature already runs to
                 # sixty-odd arguments, and every detector added would extend
                 # it here, at the PipelineConfig construction below, and at
                 # the call site in app.py -- with a missing entry silently
                 # substituting a default rather than raising.
                 detection_params=None,
                 latency_map=None,
                 onset_anchor=False,
                 onset_anchor_halfwidth_ms=8.0,
                 onset_anchor_min_trials=8,
                 filter_harmonics=False,
                 enable_inspector=False,
                 gui_root=None,
                 gui_pre_ms=None,
                 gui_post_ms=None,
                 gui_label_map=None,
                 gui_color_map=None,
                 crop_start=None,
                 crop_end=None,
                 crop_ranges=None,
                 study_metadata=None,
                 limb="", measure="",
                 reference_map=None, mmax_file="",
                 plateau_tolerance=0.10,
                 extra_channel_indices=None, wide_window_s=3.0,
                 derivatives_root=None,
                 csp_types=None,
                 csp_min_silence_ms=25.0, csp_min_return_ms=40.0,
                 csp_criterion=1.96, csp_significance=0.99,
                 csp_n_boot=1000, csp_search_end_ms=400.0,
                 csp_max_mep_offset_ms=100.0,
                 average_mode=False,
                 column_selection=None,
                 existing_segments_metadata=None):
    """
    Orchestrate the full per-file MEP/CMAP analysis pipeline.

    This function is intentionally thin: each logical stage is delegated to
    a named module-level subfunction (pipeline_load_file, pipeline_apply_filters,
    etc.) so the pipeline is readable, testable, and easy to extend.
    """
    # ── Build PipelineConfig from keyword arguments ───────────────────────────
    cfg = PipelineConfig(
        pre_ms=pre_ms, post_ms=post_ms,
        ptp_start=ptp_start, ptp_end=ptp_end, prestim_ms=prestim_ms,
        apply_filter=apply_filter, apply_bandpass=apply_bandpass,
        apply_notch=apply_notch, apply_humbug=apply_humbug,
        highpass=highpass, lowpass=lowpass,
        notch_freq=notch_freq, notch_q=notch_q,
        filter_order=filter_order, filter_harmonics=filter_harmonics,
        flexible_bandpass=flexible_bandpass, hp_order=hp_order, lp_order=lp_order,
        humbug_harmonics=humbug_harmonics,
        peak_fraction=peak_fraction,
        min_peak_amplitude=min_peak_amplitude,
        slope_threshold=slope_threshold,
        onset_method=onset_method,
        onset_bootstrap_crit=onset_bootstrap_crit,
        onset_bootstrap_n=onset_bootstrap_n,
        onset_bigoni_smooth_ms=onset_bigoni_smooth_ms,
        onset_bigoni_min_run_ms=onset_bigoni_min_run_ms,
        onset_bigoni_walkback_sd=onset_bigoni_walkback_sd,
        **config_detection_kwargs(detection_params or {}),
        latency_map=latency_map or {},
        onset_anchor=onset_anchor,
        onset_anchor_halfwidth_ms=onset_anchor_halfwidth_ms,
        onset_anchor_min_trials=onset_anchor_min_trials,
        outlier_threshold=outlier_threshold,
        enable_outlier_review=enable_outlier_review,
        custom_labels=custom_labels or {},
        color_map=color_map or {},
        window_map=window_map or {},
        condition_map=condition_map or {},
        gap_ms_map=gap_ms_map or {},
        delay_ms_map=delay_ms_map or {},
        reference_map=reference_map or {},
        mmax_file=mmax_file or "",
        plateau_tolerance=plateau_tolerance,
        extra_channel_indices=extra_channel_indices or [],
        wide_window_s=wide_window_s,
        limb=limb or "",
        measure=measure or "",
        cheby_ripple=cheby_ripple,
        filter_family=filter_family or "butter",
        csp_types=set(csp_types) if csp_types else set(),
        csp_min_silence_ms=csp_min_silence_ms,
        csp_min_return_ms=csp_min_return_ms,
        csp_criterion=csp_criterion,
        csp_significance=csp_significance,
        csp_n_boot=csp_n_boot,
        csp_search_end_ms=csp_search_end_ms,
        csp_max_mep_offset_ms=csp_max_mep_offset_ms,
        average_mode=average_mode,
        # NOT `or None`: an empty list is a real selection (protected columns
        # only) and must stay distinguishable from None, which means no
        # narrowed file at all.
        column_selection=column_selection,
    )

    # ── BIDS output paths ─────────────────────────────────────────────────────
    meta         = study_metadata or StudyMetadata()
    _source_dir  = os.path.dirname(input_path) or "."

    def _make_deriv_base(root):
        """Build the derivatives base path, avoiding derivatives/derivatives."""
        norm = os.path.basename(os.path.normpath(root)).lower()
        if norm == "derivatives":
            # root IS the derivatives folder — don't append another level
            return os.path.join(root, meta.sub_ses_path())
        else:
            return os.path.join(root, "derivatives", meta.sub_ses_path())

    _deriv_base = (_make_deriv_base(derivatives_root)
                   if derivatives_root
                   else os.path.join(_source_dir, "derivatives", meta.sub_ses_path()))
    os.makedirs(_deriv_base, exist_ok=True)
    # Always ensure _bids_prefix is unique per source file.
    # For BIDS-named files the stem is already unique and embedded in the prefix.
    # For non-BIDS files (e.g. LabChart exports) the user enters shared metadata
    # for multiple files, so we append a disambiguating suffix derived from the
    # file stem — but strip any tokens that are already in the metadata prefix
    # to avoid redundancy (e.g. sub-o001 appearing twice).
    _bids_prefix = _make_bids_prefix(meta.bids_prefix(), pathlib.Path(input_path).stem)

    # Tag the outputs with the channel when more than one is being analysed,
    # so the passes do not overwrite each other.
    #
    # Only when more than one: a single-channel analysis keeps the filenames it
    # has always had, so existing derivatives, the group-level merge and any
    # scripts pointing at them go on working untouched. The Channel column
    # below is written either way, so a merged table can always say which
    # channel a row came from.
    if channel_label and multi_channel:
        _chan_token = _sanitise_bids_label(str(channel_label))
        _bids_prefix = f"{_bids_prefix}_channel-{_chan_token}"

    def _bids_path(suffix):
        return os.path.join(_deriv_base, f"{_bids_prefix}_{suffix}")

    def _write_sidecar(csv_path, extra=None):
        filter_cfg = dict(highpass=highpass, lowpass=lowpass,
                          apply_bandpass=apply_bandpass,
                          apply_notch=apply_notch, notch_freq=notch_freq,
                          notch_q=notch_q, apply_humbug=apply_humbug,
                          humbug_harmonics=humbug_harmonics,
                          filter_order=filter_order)
        # Record any marker correction. A delay shifts every latency in the
        # file, so the derivative must carry the value and whether it was
        # measured or typed.
        sidecar = meta.to_sidecar(
            input_path, filter_cfg,
            event_delay_ms={k: v for k, v in (cfg.delay_ms_map or {}).items() if v},
            event_delay_source=delay_source_map or {},
            # How the stimuli were found, so the derivative says where its
            # trials came from rather than only what was measured in them.
            event_sources=[_s.to_dict() for _s in (event_sources or [])])
        if extra:
            sidecar.update(extra)
        json_path = os.path.splitext(csv_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(sidecar, jf, indent=2)

    # ── Output directories ────────────────────────────────────────────────────
    stim_out     = os.path.join(_deriv_base, "trials")
    results_out  = os.path.join(_deriv_base, "results")
    figures_out  = os.path.join(_deriv_base, "figures")
    stim_out_all = os.path.join(_deriv_base, "trials_with_outliers")
    figures_all  = os.path.join(_deriv_base, "figures_with_outliers")
    for _d in (stim_out, results_out, figures_out, stim_out_all, figures_all):
        os.makedirs(_d, exist_ok=True)

    # ── File list ─────────────────────────────────────────────────────────────
    if os.path.isdir(input_path):
        txt_files = [f for f in glob.glob(os.path.join(input_path, "*.txt"))
                     if "spreadsheet" not in f.lower()]
    else:
        txt_files = [input_path]
    if not txt_files:
        log_callback("❌ No valid .txt files found.")
        return

    # ── Accumulators (across files) ───────────────────────────────────────────
    summary_rows, with_out_rows = [], []
    latency_auto, latency_manual = [], []
    _averaged_rows = []          # averaged-mode: one row per (file, condition)
    _segments_bundle = []        # add-on bundle: per (file, stim_type) waveforms
    ptp_data, rms_data, preptp_data, full_ptp_data = {}, {}, {}, {}
    rejected_outlier_log = []
    rng = default_rng(42)

    # ── Per-file loop ─────────────────────────────────────────────────────────
    for file_i, raw_file in enumerate(txt_files):
        name = pathlib.Path(raw_file).stem

        def add_tag(fname):
            stem, ext = os.path.splitext(fname)
            return f"{_bids_prefix}_{stem}{ext}"

        try:
            # ── Stage 1: Load ─────────────────────────────────────────────────
            emg, time, fs, unit, stim_times = pipeline_load_file(
                raw_file, channel_idx, marker_name,
                crop_ranges=crop_ranges,
                crop_start=crop_start, crop_end=crop_end,
                sources=event_sources, channel_names=channel_names,
                event_rows=event_rows,
                warn=lambda m: log_callback(f"   ⚠️  {m}"))
            stim_types = sorted(stim_times)

            log_callback(f"📂 Processing {name}  (fs={fs} Hz, {len(stim_types)} stim type(s))")
            if event_rows is not None:
                _n_cond = sum(1 for k in stim_types
                              if split_group_key(cfg, k)[1])
                log_callback(f"   Groups from assigned conditions: "
                             + ", ".join(sorted(stim_types))
                             + (f"  ({_n_cond} carry a condition)" if _n_cond
                                else ""))
            # Where the stimuli came from is part of what makes a run
            # readable months later: the same file yields different events
            # under a different threshold, and the log is the only place that
            # shows which one ran.
            for _src in (event_sources or []):
                try:
                    log_callback(f"   ⚡ events from {_src.describe()}")
                except Exception:
                    log_callback(f"   ⚡ events from {_src}")

            # ── Stage 2: Filter ───────────────────────────────────────────────
            emg = pipeline_apply_filters(emg, fs, cfg)

            # ── Stage 3: Extract segments ─────────────────────────────────────
            # Windows are per stimulus type now, so there is no single
            # samples_before, time_axis or amplitude index for the file. Each
            # is resolved where it is used, from the type being handled.
            def _win_s(_st):
                return window_samples(cfg, _st, fs)

            def _axis(_st):
                return time_axis_for(cfg, _st, fs)

            def _ptp_idx(_st):
                _b, _a = _win_s(_st)
                return (_b + int(ptp_start * fs / 1000),
                        _b + int(ptp_end   * fs / 1000))

            all_segments = pipeline_extract_segments(
                time, emg, stim_times, stim_types, fs, cfg, log_callback)

            # ── Stage 4: Save "with-outliers" CSVs and figures ────────────────
            for stim_type, segs in all_segments.items():
                samples_before, samples_after = _win_s(stim_type)
                time_axis = _axis(stim_type)
                emg_all  = np.array([s[0] for s in segs])
                mean_all = emg_all.mean(axis=0)
                df_all   = pd.DataFrame(emg_all).T
                df_all   = _add_time_and_digmark(df_all, samples_before, fs)
                df_all.to_csv(os.path.join(stim_out_all,
                              add_tag(f"{stim_type}.csv")), index=False)

                fig_all = matplotlib.figure.Figure(figsize=(12, 6))
                ax_all  = fig_all.add_subplot(111)
                col = cfg.color_map.get(stim_type, "gray")
                lbl = cfg.custom_labels.get(stim_type, stim_type)
                for s in emg_all:
                    ax_all.plot(time_axis, s, color=col, alpha=0.2, linewidth=0.5)
                ax_all.plot(time_axis, mean_all, color=col, linewidth=3,
                            label=f"{lbl} Mean PTP: {_np_ptp(mean_all):.2f}")
                ax_all.axvline(0, color="black", linestyle="--")
                ax_all.set_title(f"{name} – {lbl} (All Traces)")
                ax_all.set_xlabel("Latency (ms)")
                ax_all.set_ylabel(f"EMG ({unit})" if unit else "EMG")
                ax_all.legend()
                matplotlib.backends.backend_agg.FigureCanvasAgg(fig_all).print_figure(
                    os.path.join(figures_all, add_tag(f"stim-{stim_type}_traces_all.png")),
                    dpi=600)
                fig_all.clf()

            # ── Stage 5: Outlier detection and review ─────────────────────────
            stats_per_type  = {}
            segments_final        = defaultdict(list)
            segments_inspector    = defaultdict(list)  # full prestim_ms pre-stim
            trace_stats     = []

            for stim_type, segs in all_segments.items():
                emg_segs  = np.array([s[0] for s in segs])
                pre_segs  = np.array([s[1] for s in segs])
                stim_ts   = np.array([s[2] for s in segs])
                mean_tr   = emg_segs.mean(axis=0)
                mean_ptp  = float(_np_ptp(mean_tr))

                _p0, _p1 = _ptp_idx(stim_type)
                ptps, rms_vals, preptp, rms_z, ptp_z, out_idx = pipeline_detect_outliers(
                    emg_segs, pre_segs, _p0, _p1, cfg)

                rejected, log_entries = pipeline_review_outliers(
                    stim_type, name, emg_segs, pre_segs,
                    out_idx, ptps, rms_vals, rms_z, ptp_z,
                    cfg, fs, pre_ms, post_ms, unit,
                    review_outliers_cb, log_callback)
                rejected_outlier_log.extend(log_entries)

                outlier_set = set(rejected)
                stats_per_type[stim_type] = dict(
                    segs=emg_segs, ptps=ptps, rms_vals=rms_vals, preptp=preptp,
                    segs_all=emg_segs.copy(), prestim_all=pre_segs.copy(),
                    outlier_set=outlier_set, stim_times_s=stim_ts.copy())

                if rejected:
                    keep = np.ones(len(emg_segs), dtype=bool)
                    keep[rejected] = False
                    emg_segs  = emg_segs[keep]
                    pre_segs  = pre_segs[keep]
                    stats_per_type[stim_type]["segs"] = emg_segs

                # Accumulate this condition's per-trial waveforms for the
                # add-on results bundle (<prefix>_segments.npz). segs_all is
                # the normalised segment array — format-agnostic — and is
                # 1:1 with outlier_set and stim_times_s.
                _sb_all = stats_per_type[stim_type]["segs_all"]
                _segments_bundle.append({
                    "file": name, "stim_type": stim_type, "fs": fs,
                    "unit": unit or "", "pre_ms": cfg.pre_ms,
                    "post_ms": cfg.post_ms, "waveforms": _sb_all,
                    "outlier": np.array([_i in outlier_set
                                         for _i in range(len(_sb_all))],
                                        dtype=bool),
                    "stim_time_s": np.asarray(
                        stats_per_type[stim_type]["stim_times_s"], dtype=float),
                })

                # Save clean CSV
                df_clean = pd.DataFrame(emg_segs).T
                df_clean = _add_time_and_digmark(df_clean, samples_before, fs)
                df_clean.to_csv(os.path.join(stim_out,
                                add_tag(f"{stim_type}.csv")), index=False)

                # Always pass ALL segments (segs_all) to the inspector so
                # that segment indices align with pipeline_quantify_segments.
                # Inspector notes/edits are stored by (stim_type, idx) where
                # idx must index into segs_all, not the cleaned subset.
                segments_final[stim_type].extend(
                    stats_per_type[stim_type]["segs_all"])
                # Build inspector segments with full prestim_ms pre-stim.
                #
                # The event delay MUST be applied here too. These segments are
                # re-extracted from the raw time axis rather than reusing the
                # analysed ones, and without the shift the Inspector displays a
                # different epoch from the one the analysis measured. Marker
                # indices returned from it are then offset by exactly the
                # delay, and quantification applies them to the shifted
                # segments -- so a corrected condition came back with its
                # peak-to-peak read from the wrong samples while every
                # uncorrected condition was fine.
                _insp_sb = int(cfg.prestim_ms * fs / 1000)
                # Pre stays prestim_ms -- the inspector deliberately shows a
                # wider lead-in than the analysis window. Post is the analysis
                # window and is now per type, so a type epoched over 500 ms
                # after the pulse is reviewed over 500 ms rather than being
                # cut to whatever the file-wide setting happened to be.
                _insp_sa = window_samples(cfg, stim_type, fs)[1]
                _insp_delay = int(round(
                    float(cfg.delay_ms_map.get(stim_type, 0.0)) * fs / 1000.0))
                for _t0 in [t for t in stim_times.get(stim_type, [])
                            if time.min() <= t <= time.max()]:
                    _ix  = int(np.argmin(np.abs(time - _t0))) + _insp_delay
                    if _ix < 0 or _ix >= len(emg):
                        continue
                    _seg = emg[max(0,_ix-_insp_sb):_ix+_insp_sa]
                    if len(_seg) == _insp_sb + _insp_sa:
                        segments_inspector[stim_type].append(_seg)
                # The axis travels with the traces: the combined figure draws
                # types whose windows differ, so a single shared axis could not
                # describe all of them.
                trace_stats.append((stim_type, segs, emg_segs, mean_tr,
                                    mean_ptp, _axis(stim_type)))

                ptp_data.setdefault(stim_type, []).extend(ptps.tolist())
                rms_data.setdefault(stim_type, []).extend(rms_vals.tolist())
                preptp_data.setdefault(stim_type, []).extend(preptp.tolist())
                # Full-range PTP (entire post-stim window, not just analysis window).
                # Used for reference conditions like M-wave that occur before
                # the PTP analysis window (10-50ms) and would otherwise give ~0.
                _full_ptps = _np_ptp(emg_segs[:, samples_before:], axis=1)
                full_ptp_data.setdefault(stim_type, []).extend(_full_ptps.tolist())

            # ── Stage 5c: Anchored onset pre-pass (single source of truth) ────
            # One automatic onset pass per stim type, computed from
            # outlier-screened trials. Feeds BOTH the inspector's starting
            # markers and the saved output, so what you see and what is written
            # are one computation. auto_onsets_by_type: {stim_type: {idx: ms}}.
            auto_onsets_by_type = {}
            for _st, _info in stats_per_type.items():
                _p0, _p1 = _ptp_idx(_st)
                auto_onsets_by_type[_st] = pipeline_detect_onsets(
                    _st, _info["segs_all"], _info["outlier_set"],
                    _p0, _p1, fs, cfg,
                    log_callback=log_callback)

            # ── Stage 5d: PTP window per stimulus type ────────────────────────
            # Derived AFTER onsets, because the anchor is each condition's own
            # median onset. Onset detection itself no longer depends on the PTP
            # window (see onset_search_window), so there is no circularity: the
            # physiological latency profile governs onset, and onset governs
            # where amplitude is measured.
            ptp_window_by_type = {}
            for _st in stats_per_type:
                _p0, _p1 = _ptp_idx(_st)
                _sb, _sa = _win_s(_st)
                ptp_window_by_type[_st] = ptp_window_for_stim_type(
                    _st, auto_onsets_by_type.get(_st, {}), fs, cfg,
                    _p0, _p1, _sb,
                    log_callback=log_callback)
            # Seed for the inspector, in inspector index space (stim @ _insp_sb).
            _insp_sb_seed = int(round(cfg.prestim_ms * fs / 1000))
            auto_meta = {}
            for _st, _onsets in auto_onsets_by_type.items():
                for _idx, _oms in _onsets.items():
                    if _oms is not None:
                        auto_meta[(_st, _idx)] = {
                            "onset_idx": _insp_sb_seed + int(round(_oms * fs / 1000))
                        }

            # ── Averaged-waveform analysis mode ───────────────────────────────
            # Collapse each condition to its clean-trial mean, inspect/measure
            # the mean once, accumulate reduced rows, then skip this file's
            # per-trial quantify / detrend / bootstrap / plot stages. Rows are
            # written once after the loop (the File column disambiguates files).
            if cfg.average_mode:
                _means = pipeline_assemble_condition_means(
                    stats_per_type, emg, time, fs, cfg,
                    log_callback=log_callback)
                # Seed the inspector with an onset detected on each mean.
                _avg_seed = {}
                for _st, _mi in _means.items():
                    _mn, _mx = cfg.latency_map.get(
                        _st, (cfg.ptp_start, cfg.ptp_end))
                    _oms = _detect_onset_dispatch(
                        _mi["mean"], fs, cfg, _mn, _mx, stim_type=_st)
                    if _oms is not None and _oms > 0:
                        _avg_seed[(_st, 0)] = {
                            "onset_idx": _insp_sb_seed
                            + int(round(_oms * fs / 1000))}
                _avg_meta = (dict(existing_segments_metadata)
                             if existing_segments_metadata else {})
                if (enable_inspector and show_inspector_cb
                        and _means and file_i == 0):
                    _avg_insp_segs = {
                        _st: [_mi["mean"]]
                        for _st, _mi in _means.items()}
                    _avg_meta = show_inspector_cb(
                        _avg_insp_segs, fs, cfg.prestim_ms, post_ms, unit,
                        custom_labels, color_map, prestim_ms,
                        extra_segs={},
                        wide_window_s=cfg.wide_window_s,
                        auto_meta=_avg_seed,
                        underlays={_st: _mi["individual"]
                                   for _st, _mi in _means.items()})
                else:
                    _avg_meta.update(_avg_seed)
                _avg_rows = pipeline_quantify_averaged(
                    _means, _avg_meta, fs, cfg,
                    custom_labels or {}, name, log_callback=log_callback)
                pipeline_normalise_averaged(_avg_rows, cfg,
                                            log_callback=log_callback)
                _averaged_rows.extend(_avg_rows)
                # Output-file companion: per-condition mean + faint underlays.
                if _means:
                    try:
                        pipeline_generate_averaged_plots(
                            _means, fs, cfg, figures_out, _bids_prefix,
                            name, unit, log_callback=log_callback)
                    except Exception as _pe:
                        log_callback(f"⚠️  Averaged plot error: {_pe}")
                if progress_callback:
                    progress_callback(((file_i + 1) / len(txt_files)) * 100)
                continue

            # ── Stage 6: Data Inspector ───────────────────────────────────────
            # Seed with any previously saved metadata so manual edits
            # (adjusted markers, notes, exclusions) survive re-runs.
            segments_metadata = dict(existing_segments_metadata) \
                if existing_segments_metadata else {}
            if enable_inspector and show_inspector_cb and segments_final and file_i == 0:
                # Build extra-channel wide segments for visual inspection
                # {chan_name: {stim_type: [wide_seg_array]}}
                # Pass full raw arrays + stim times to inspector so it can
                # slice on demand when the user adjusts the wide-window spinbox.
                # {chan_name: {"emg": array, "time": array, "fs": float,
                #              "stim_times": {stim_type: [t_sec, ...]}}}
                _extra_segs = {}
                from .io import list_waveform_channels
                try:
                    _chan_names_all = list_waveform_channels(raw_file)
                except Exception as _lc_err:
                    log_callback(f"⚠️  Could not list channels for extra-channel panel: {_lc_err}")
                    _chan_names_all = []
                for _ci in cfg.extra_channel_indices:
                    try:
                        _cname = (_chan_names_all[_ci]
                                  if _ci < len(_chan_names_all)
                                  else f"Ch{_ci+1}")
                        _emg_x, _fs_x, _ = extract_emg_waveform_and_fs(
                            raw_file, channel_idx=_ci)
                        _time_x = np.arange(len(_emg_x)) / _fs_x
                        # Apply crop if set
                        if crop_ranges:
                            _keep = np.zeros(len(_time_x), dtype=bool)
                            for _a, _b in crop_ranges:
                                _keep |= (_time_x >= _a) & (_time_x <= _b)
                            _emg_x  = _emg_x[_keep]
                            _time_x = _time_x[_keep]
                        elif crop_start is not None and crop_end is not None:
                            _keep   = (_time_x >= crop_start) & (_time_x <= crop_end)
                            _emg_x  = _emg_x[_keep]
                            _time_x = _time_x[_keep]
                        # Extra channels are NOT filtered — they may be force,
                        # torque, accelerometer etc. whose frequency content
                        # would be destroyed by the EMG bandpass (e.g. 20-450 Hz
                        # removes all slow force signal). Show them raw.
                        _extra_segs[_cname] = {
                            "emg":        _emg_x,
                            "time":       _time_x,
                            "fs":         _fs_x,
                            "stim_times": {_st: list(_tms)
                                           for _st, _tms in stim_times.items()},
                        }
                    except Exception as _xe:
                        log_callback(f"⚠️  Extra channel {_ci}: {_xe}")

                # Only use segments_inspector if ALL stim types have the same
                # count as segs_all — otherwise fall back to segments_final
                # so that (stim_type, idx) keys always align.
                _counts_match = all(
                    len(segments_inspector.get(st, [])) ==
                    len(stats_per_type.get(st, {}).get("segs_all", []))
                    for st in segments_final
                )
                _insp_segs = segments_inspector if _counts_match else segments_final
                # The amplitude window the analysis actually used, per stimulus
                # type, in ms relative to the stimulus. Without this the
                # Inspector re-seeded its PTP markers from the file-wide 1c
                # window, so with anchoring enabled the review measured a
                # different interval from the analysis -- and the peak-to-peak
                # on screen was not the one in the results file.
                # ptp_window_for_stim_type returns THREE values --
                # (start_idx, end_idx, ms_pair_or_None) -- so the first two are
                # taken by index rather than by unpacking. Destructuring to a
                # pair here raised "too many values to unpack" only once an
                # analysis reached the inspector.
                _ptp_ms_by_type = {}
                for _st, _win in (ptp_window_by_type or {}).items():
                    try:
                        _ptp_ms_by_type[_st] = (
                            (int(_win[0]) - samples_before) * 1000.0 / fs,
                            (int(_win[1]) - samples_before) * 1000.0 / fs,
                        )
                    except Exception:
                        pass
                segments_metadata = show_inspector_cb(
                    _insp_segs, fs, cfg.prestim_ms, post_ms, unit,
                    custom_labels, color_map, prestim_ms,
                    extra_segs=_extra_segs,
                    wide_window_s=cfg.wide_window_s,
                    auto_meta=auto_meta,
                    ptp_windows_by_type=_ptp_ms_by_type)

            # Parse inspector metadata
            excluded_sets = defaultdict(set)
            notes_map     = {}
            for (stype, idx), m in segments_metadata.items():
                if m.get("exclude", False):
                    excluded_sets[stype].add(idx)
                if "note" in m:
                    notes_map[(stype, idx)] = m["note"]

            # ── Stage 7: Quantify all segments ────────────────────────────────
            _ptps_per_stim       = {}
            _stim_times_per_stim = {}
            _agreement_by_trial  = {}
            for stim_type, info in stats_per_type.items():
                auto_r, man_r, sum_r, with_r, ptps_arr = pipeline_quantify_segments(
                    stim_type,
                    info["segs_all"], info["prestim_all"],
                    info["outlier_set"], excluded_sets[stim_type],
                    segments_metadata,
                    *ptp_window_by_type.get(
                        stim_type, _ptp_idx(stim_type) + (None,))[:2],
                    fs, cfg, custom_labels or {}, name,
                    auto_onsets_by_type.get(stim_type, {}),
                    log_callback=log_callback,
                    agreement_out=_agreement_by_trial)

                latency_auto.extend(auto_r)
                latency_manual.extend(man_r)
                summary_rows.append(sum_r)
                with_out_rows.append(with_r)
                _ptps_per_stim[stim_type]       = ptps_arr
                _stim_times_per_stim[stim_type] = info["stim_times_s"]

            # ── Stage 7b: Onset-method comparison report ──────────────────────
            # Follows cfg.onset_agreement, not the selected method: agreement
            # runs the member detectors whatever is selected, and comparing
            # methods while running the one you trust is how a method choice
            # gets justified.
            if getattr(cfg, "onset_agreement", False):
                try:
                    from .onset_methods_report import (
                        collect_agreement_rows, write_onset_method_figures,
                        write_onset_method_tables)
                    _m_rows = collect_agreement_rows(
                        _agreement_by_trial, name, cfg.custom_labels)
                    _written = write_onset_method_tables(
                        _m_rows, results_out, _bids_prefix)
                    _written += write_onset_method_figures(
                        _m_rows, _agreement_by_trial,
                        {st: info["segs_all"] for st, info in stats_per_type.items()},
                        fs, cfg.pre_ms, figures_out, _bids_prefix,
                        unit=unit, custom_labels=cfg.custom_labels,
                        selected_method=cfg.onset_method,
                        log_callback=log_callback)
                    if _written:
                        from .onset_methods_report import (
                            FIGURE_SUBDIR_SUFFIX as _FSUF)
                        log_callback(
                            f"📊 Onset-method comparison: {len(_written)} "
                            f"file(s) written — tables in results/, figures "
                            f"in figures/{_bids_prefix}_{_FSUF}/")
                except Exception as _exc:
                    log_callback(f"⚠️ Onset-method comparison failed "
                                 f"({type(_exc).__name__}: {_exc})")
            elif getattr(cfg, "onset_method", "") == "methods_median":
                # Consensus reports a median but keeps no breakdown unless
                # agreement is on. Saying so beats writing nothing silently.
                log_callback(
                    "ℹ️ Consensus is selected but 'Compare methods on every "
                    "trial' is off, so no method-comparison tables or figures "
                    "were produced. Enable it in Preferences → Detection.")

            # ── Stage 8: Pooled z-scores and detrending ───────────────────────
            pipeline_compute_pooled_stats(
                _ptps_per_stim, _stim_times_per_stim,
                latency_auto, latency_manual)

            # ── Stage 9: Plots ────────────────────────────────────────────────
            # Clean up stale figures from previous runs with old prefix names
            for _old_fig in glob.glob(os.path.join(figures_out, "*_traces.png")):
                try:
                    os.remove(_old_fig)
                except Exception:
                    pass
            for _old_fig in glob.glob(os.path.join(figures_all, "*_traces_all.png")):
                try:
                    os.remove(_old_fig)
                except Exception:
                    pass
            pipeline_generate_plots(
                trace_stats, segments_metadata,
                cfg.color_map, cfg.custom_labels,
                figures_out, figures_all, _bids_prefix, name, unit,
                enable_individual_plots, cfg)

            log_callback(f"✔️  Finished {name}")

        except Exception as e:
            import traceback
            log_callback(f"❌ Error processing {name}: {e}")
            log_callback(traceback.format_exc())

        if progress_callback:
            progress_callback(((file_i + 1) / len(txt_files)) * 100)

    # ── Add-on results bundle: per-trial waveforms (format-agnostic) ──────
    pipeline_write_segments_bundle(_segments_bundle, results_out,
                                   _bids_prefix, log_callback)

    # ── Averaged mode: write the accumulated reduced CSV and return ────────
    # The per-file branch skipped per-trial accumulation, so the aggregate
    # stages below (compensation / normalisation / trial + summary CSVs) do not
    # apply. Write <prefix>_averaged.csv once, then finish.
    if cfg.average_mode:
        if _averaged_rows:
            os.makedirs(results_out, exist_ok=True)
            _avg_path = pipeline_write_averaged(
                _averaged_rows, results_out, _bids_prefix)
            log_callback(f"\u2714\ufe0f  Averaged analysis written: "
                         f"{os.path.basename(_avg_path)}")
        else:
            log_callback("\u26a0\ufe0f  Averaged mode: no rows produced \u2014 nothing written.")
        if progress_callback:
            progress_callback(100)
        log_callback("\u2705 Averaged analysis complete.")
        return

    # ── Stage 9a: EMG excitability compensation (Carson 2026, QR) ─────────
    # Runs unconditionally (independent of normalisation). Fits QR per StimType
    # sample on every trial except those Removed or Excluded, so as much of the
    # design's power is retained as possible (a stated benefit of the method).
    # EVERY stim type is compensated — including a
    # single-pulse condition that serves as a paired-pulse normalisation
    # reference, since the fit is within-stim-type and single-pulse MEPs are a
    # valid target. (A reference stim gets Adjusted_PTP_QR but no
    # Normalised_Adjusted_PTP_QR, because a reference is never normalised.)
    # The one exception is genuine M-wave data — a direct muscle response, not
    # spinally mediated, and typically a multi-intensity recruitment curve — so
    # compensation is skipped when the run is designated 'M-wave'. (External
    # Mmax reference files are never present in these trial rows.)
    if (cfg.measure or "").strip().lower().replace(" ", "-") == "m-wave":
        log_callback("🧮 EMG compensation skipped — measure is 'M-wave' "
                     "(direct muscle response; not spinally mediated)")
    else:
        _emg_col_idx = {c: i for i, c in enumerate(LAT_COLS)}
        apply_emg_compensation(
            latency_manual, _emg_col_idx,
            log_callback=log_callback)
        apply_emg_compensation(
            latency_auto, _emg_col_idx,
            log_callback=lambda _: None)

    # ── Stage 9b: Apply normalisation (Mmax / paired-pulse ratios) ─────────
    if cfg.reference_map:
        _col_idx  = {c: i for i, c in enumerate(LAT_COLS)}
        _stim_ptps: dict = {}
        _ri_st  = _col_idx["StimType"]
        _ri_ptp = _col_idx["PTP(mV)"]
        for _row in latency_manual:
            _st  = _row[_ri_st]
            _ptp = _row[_ri_ptp]
            try:    _stim_ptps.setdefault(_st, []).append(float(_ptp))
            except (TypeError, ValueError): pass
        # For each reference condition, prefer window PTP if it has positive
        # values; otherwise fall back to full-range PTP (catches M-wave which
        # occurs before the 10-50ms PTP analysis window).
        for _ref_st in set(cfg.reference_map.values()):
            if not _ref_st:
                continue
            _win_vals = [v for v in _stim_ptps.get(_ref_st, [])
                         if v is not None and np.isfinite(v) and v > 0]
            if not _win_vals and _ref_st in full_ptp_data:
                _full_vals = [v for v in full_ptp_data[_ref_st]
                              if v is not None and np.isfinite(v) and v > 0]
                if _full_vals:
                    _stim_ptps[_ref_st] = full_ptp_data[_ref_st]
                    log_callback(f"📐 Reference '{_ref_st}': window PTP was ~0"
                                 f" — using full post-stim range PTP instead")


        apply_normalisation(
            latency_manual, _col_idx, _stim_ptps,
            cfg.reference_map,
            plateau_tolerance=cfg.plateau_tolerance,
            log_callback=log_callback)
        apply_normalisation(
            latency_auto, _col_idx, _stim_ptps,
            cfg.reference_map,
            plateau_tolerance=cfg.plateau_tolerance,
            log_callback=lambda _: None)

    # ── Stage 10: Write outputs ───────────────────────────────────────────────
    _selected_groups = pipeline_write_outputs(
        latency_manual,
        results_out, _bids_prefix,
        channel_label=channel_label,
        column_selection=cfg.column_selection,
        log_callback=log_callback)

    # ── Write _trials.json sidecar ────────────────────────────────────────────
    # This is the file Stage 2 scans for. It must be written alongside
    # the trials.csv in the results/ folder.
    from .results_layout import result_path as _rp
    _trials_csv = _rp(results_out, f"{_bids_prefix}_trials.csv",
                      create=False)
    if not os.path.isfile(_trials_csv):
        # A study written before the folders existed keeps its flat
        # layout; the sidecar must still find its own trials file.
        _trials_csv = os.path.join(results_out,
                                   f"{_bids_prefix}_trials.csv")
    if os.path.isfile(_trials_csv):
        _write_sidecar(_trials_csv, extra={
            "trials_csv": f"{_bids_prefix}_trials.csv",
            # What the narrowed copy was ASKED for, not what its header
            # happens to show. Stage 2 compares these across sessions to
            # decide whether they can be merged, and a header comparison
            # cannot tell "this analyst chose not to keep cSP" from "this
            # recording had no cSP data to keep".
            #
            # null means no narrowed file was written. Recorded either way,
            # so a session that predates the feature (key absent) is
            # distinguishable from one that deliberately wrote none.
            "column_selection": _selected_groups,
        })

    # Auto-open combined figure
    if txt_files:
        combined = os.path.join(figures_out, f"{_bids_prefix}_traces.png")
        if os.path.exists(combined):
            webbrowser.open(combined)

    if progress_callback:
        progress_callback(100)
    log_callback("✅ All results saved.")


# ─────────────────────────── End of run_pipeline ─────────────────────────────

