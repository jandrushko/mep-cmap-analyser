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
from .bids import StudyMetadata
from .utils import _add_time_and_digmark
from .io import extract_emg_waveform_and_fs, extract_stim_times
from .filters import adaptive_mains_cancel, design_notch_sos
from .detection     import (detect_mep_onset_peak_fraction,
                             detect_mep_onset_bootstrap,
                             compute_bootstrap_threshold)
from .normalisation import compute_mmax, apply_normalisation

@dataclass
class PipelineConfig:
    """Bundles all analysis settings so subfunctions share one parameter object."""
    # Time windows
    pre_ms:            int   = 20
    post_ms:           int   = 400
    ptp_start:         int   = 10
    ptp_end:           int   = 50
    prestim_ms:        int   = 100
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
    peak_fraction:         float = 0.15
    min_peak_amplitude:    float = 0.05
    slope_threshold:       float = 0.08
    # "peak_fraction" (default) or "bootstrap"
    onset_method:          str   = "peak_fraction"
    onset_bootstrap_crit:  float = 1.96
    onset_bootstrap_n:     int   = 500
    latency_map:           dict  = field(default_factory=dict)
    # Outlier detection
    outlier_threshold:     float = 1.96
    enable_outlier_review: bool  = True
    # Bootstrap
    bootstrap_iter:        int   = 10000
    # Output labels / colours
    custom_labels:   dict = field(default_factory=dict)
    color_map:       dict = field(default_factory=dict)
    plot_included:   dict = field(default_factory=dict)
    gap_ms_map:      dict = field(default_factory=dict)
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


def pipeline_load_file(file_path, channel_idx, marker_name,
                       crop_ranges=None, crop_start=None, crop_end=None):
    """Load raw EMG, extract stim times, apply crop.

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
    stim_times = extract_stim_times(file_path, marker_name)

    if crop_ranges:
        keep = np.zeros_like(time, dtype=bool)
        for a, b in crop_ranges:
            keep |= (time >= a) & (time <= b)
        emg  = emg[keep]
        time = time[keep]
        for k in list(stim_times):
            stim_times[k] = [t for t in stim_times[k]
                             if any(a <= t <= b for a, b in crop_ranges)]
            if not stim_times[k]:
                stim_times.pop(k)
    elif crop_start is not None and crop_end is not None:
        keep = (time >= crop_start) & (time <= crop_end)
        emg  = emg[keep]
        time = time[keep]
        for k in list(stim_times):
            stim_times[k] = [t for t in stim_times[k]
                             if crop_start <= t <= crop_end]
            if not stim_times[k]:
                stim_times.pop(k)

    return emg, time, fs, unit, stim_times


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


def pipeline_extract_segments(time, emg, stim_times, stim_types, fs,
                               cfg: PipelineConfig):
    """Extract per-trial EMG and pre-stim segments for every stim type.

    Returns
    -------
    dict mapping stim_type -> list of (seg_emg, seg_pre) tuples.
    Only complete segments (exact pre+post window) are included.
    """
    samples_before  = int(cfg.pre_ms     * fs / 1000)
    samples_after   = int(cfg.post_ms    * fs / 1000)
    prestim_samples = int(cfg.prestim_ms * fs / 1000)

    result = {}
    for stim_type in stim_types:
        valid_times = [t for t in stim_times[stim_type]
                       if time.min() <= t <= time.max()]
        gap_samples = int(cfg.gap_ms_map.get(stim_type, 0.0) * fs / 1000)
        segs = []
        for stim_time in valid_times:
            idx   = int(np.argmin(np.abs(time - stim_time)))
            start = max(0, idx - samples_before)
            end   = min(len(emg), idx + samples_after)
            if (end - start) != (samples_before + samples_after):
                continue          # incomplete window — skip
            seg_emg = emg[start:end]
            pre_start = max(0, idx - prestim_samples - gap_samples)
            pre_end   = max(0, idx - gap_samples)
            seg_pre   = emg[pre_start:pre_end]
            segs.append((seg_emg, seg_pre))
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
    rms_vals = np.sqrt(np.mean(prestim_segments ** 2, axis=1))
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


def pipeline_quantify_segments(stim_type, segs_all, prestim_all,
                                out_set, excluded_set, segments_metadata,
                                ptp_start_idx, ptp_end_idx,
                                fs, cfg: PipelineConfig,
                                custom_labels, name):
    """Per-trial quantification of PTP, latency, silent period and AUC.

    Returns
    -------
    auto_rows    : list  rows for the auto-metrics CSV
    manual_rows  : list  rows for the manual-override CSV
    summary_row  : list  one summary row (cleaned trials only)
    with_out_row : list  one summary row (all trials including outliers)
    ptps_array   : np.ndarray  per-trial PTP (all trials)
    """
    rms_all    = np.sqrt(np.mean(prestim_all ** 2, axis=1))
    preptp_all = _np_ptp(prestim_all, axis=1)
    rms_z_full = (zscore(rms_all) if len(rms_all) > 1
                  else np.zeros_like(rms_all))
    ptps = np.empty(len(segs_all))

    auto_rows, manual_rows = [], []
    latencies, silent_durs = [], []
    auc_vals_all, auc_vals_clean = [], []

    for idx, seg in enumerate(segs_all):
        # ── automatic metrics ────────────────────────────────────────────
        auto_ptp = _np_ptp(seg[ptp_start_idx:ptp_end_idx])
        if cfg.onset_method == "bootstrap":
            _lat     = cfg.latency_map.get(stim_type, (10.0, 50.0))
            _min_lat, _max_lat = _lat if _lat else (10.0, 50.0)
            auto_lat = detect_mep_onset_bootstrap(
                seg, fs,
                pre_ms=cfg.prestim_ms,
                peak_search_start_ms=cfg.ptp_start,
                peak_search_end_ms=cfg.ptp_end,
                min_latency_ms=_min_lat,
                max_latency_ms=_max_lat,
                min_peak_amplitude=cfg.min_peak_amplitude,
                criterion=cfg.onset_bootstrap_crit,
                n_boot=cfg.onset_bootstrap_n,
            )
        else:
            auto_lat = detect_mep_onset_peak_fraction(
                seg, fs,
                pre_ms=cfg.prestim_ms,
                poststim_start_ms=cfg.ptp_start,
                poststim_end_ms=cfg.ptp_end,
                peak_frac=cfg.peak_fraction,
                min_consecutive=5,
                min_peak_amplitude=cfg.min_peak_amplitude,
                slope_threshold=cfg.slope_threshold,
            )

        # ── manual overrides from inspector ──────────────────────────────
        # Inspector segments start at -prestim_ms; segs_all start at -pre_ms.
        # Convert inspector-space indices to segs_all-space before applying.
        _insp_sb = int(cfg.prestim_ms * fs / 1000)  # stim @ inspector idx
        _segs_sb = int(cfg.pre_ms     * fs / 1000)  # stim @ segs_all idx
        _offset  = _insp_sb - _segs_sb
        _n       = len(seg)
        def _ci(i): return min(max(0, i - _offset), _n - 1)
        mk = (stim_type, idx)
        if mk in segments_metadata:
            m       = segments_metadata[mk]
            man_ptp = seg[_ci(m["ptp_max_idx"])] - seg[_ci(m["ptp_min_idx"])]
            man_lat = (m["onset_idx"] - _insp_sb) * 1000 / fs
        else:
            man_ptp, man_lat = auto_ptp, auto_lat

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

        # ── AUC ──────────────────────────────────────────────────────────
        auc_val = None
        if mk in segments_metadata and "auc_start_idx" in segments_metadata[mk]:
            # User-set or inspector-auto AUC window
            a0 = _ci(segments_metadata[mk]["auc_start_idx"])
            a1 = _ci(segments_metadata[mk]["auc_end_idx"])
            auc_val = float(_np_trapz(np.abs(seg[a0:a1]), dx=1 / fs))
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
        elif mk not in segments_metadata and stim_type in cfg.csp_types:
            # Unreviewed segment with CSP enabled — try auto-detect onset+CSP
            # and compute AUC if both succeed
            from .detection import detect_csp_bootstrap as _dcsp
            _pre_samp = int(cfg.prestim_ms * fs / 1000)
            _ptp_s    = _segs_sb + int(cfg.ptp_start * fs / 1000)
            _ptp_e    = _segs_sb + int(cfg.ptp_end   * fs / 1000)
            if _ptp_e < len(seg) and _ptp_s < _ptp_e:
                _seg_ptp = seg[_ptp_s:_ptp_e]
                _peak2   = _ptp_s + int(max(np.argmin(_seg_ptp),
                                            np.argmax(_seg_ptp)))
                _peak2ms = (_peak2 - _segs_sb) * 1000 / fs
                _csp = _dcsp(seg, fs,
                             np.linspace(-cfg.pre_ms, cfg.post_ms,
                                         len(seg), endpoint=False),
                             pre_ms=cfg.pre_ms,
                             search_start_ms=_peak2ms,
                             search_end_ms=cfg.csp_search_end_ms,
                             min_silence_ms=cfg.csp_min_silence_ms,
                             min_return_ms=cfg.csp_min_return_ms,
                             criterion=cfg.csp_criterion,
                             significance=cfg.csp_significance,
                             n_boot=cfg.csp_n_boot)
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
        # [0-3] ID, [4-5] limb/measure, [6] PTP, [7] Latency
        # [8] SilentPeriod, [9] SP_MEP_Offset, [10] SP_EMG_Return, [11] MEP_cSP_Ratio
        # [12] AUC, [13-14] baseline, [15-16] Z_PreStimRMS/Z_PTP_Within
        # [17-19] pooled/detrend, [20] Outlier_Decision
        # [21-24] normalisation, [25] note
        _mep_csp = round(float(man_ptp) / float(silent_dur), 4) \
                   if (isinstance(silent_dur, (int, float)) and silent_dur > 0
                       and man_ptp is not None) else None
        common = [
            name, stim_type, custom_labels.get(stim_type, ""), idx + 1,  # [0-3]
            cfg.limb, cfg.measure,                                        # [4-5]
            None, None,                                                   # [6-7]  PTP/Lat
            silent_dur, sp_mep_offset_ms, sp_emg_return_ms, _mep_csp,   # [8-11] SP
            auc_val,                                                      # [12]
            round(rms_all[idx], 4), round(preptp_all[idx], 4),           # [13-14]
            round(rms_z_full[idx], 3), None, None, None, None,           # [15-19]
            decision,                                                      # [20]
            None, None, None, None,                                       # [21-24] norm
            note_txt,                                                      # [25]
        ]

        auto_row   = common.copy()
        manual_row = common.copy()
        auto_row  [6] = round(auto_ptp, 2)
        auto_row  [7] = round(auto_lat, 2) if auto_lat is not None else "Not Detected"
        manual_row[6] = round(man_ptp, 2)
        manual_row[7] = round(man_lat, 2) if man_lat is not None else "Not Detected"

        auto_rows.append(auto_row)
        manual_rows.append(manual_row)

        if man_lat is not None:
            latencies.append(man_lat)

    ptp_z_full = (zscore(ptps) if len(ptps) > 1 else np.zeros_like(ptps))
    for i, (ar, mr) in enumerate(zip(auto_rows, manual_rows)):
        ar[16] = mr[16] = round(float(ptp_z_full[i]), 3)  # Z_PTP_Within

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
                   float(rms_all[clean_mask].mean()) if clean_mask.count(True) else np.nan,
                   float(rms_all[clean_mask].std(ddof=1)) if clean_mask.count(True) > 1 else np.nan,
                   float(preptp_all[clean_mask].mean()) if clean_mask.count(True) else np.nan,
                   float(preptp_all[clean_mask].std(ddof=1)) if clean_mask.count(True) > 1 else np.nan,
                   mean_lat, std_lat, mean_sil, std_sil, mean_auc_c, std_auc_c]

    with_out_row = [name, stim_type, lbl, n_all,
                    float(np.mean(_np_ptp(segs_all[:, ptp_start_idx:ptp_end_idx], axis=1))),
                    float(np.std( _np_ptp(segs_all[:, ptp_start_idx:ptp_end_idx], axis=1))),
                    float(rms_all.mean()),    float(rms_all.std(ddof=1)),
                    float(preptp_all.mean()), float(preptp_all.std(ddof=1)),
                    mean_lat, std_lat, mean_sil, std_sil, mean_auc_a, std_auc_a]

    return auto_rows, manual_rows, header_vals, with_out_row, ptps


def pipeline_compute_pooled_stats(ptps_per_stim, latency_rows_auto, latency_rows_manual):
    """Compute pooled Z-scores and linear detrending across all stim types.
    Modifies the last three columns of each row in-place.
    """
    if not ptps_per_stim:
        return
    all_ptps   = np.concatenate(list(ptps_per_stim.values()))
    pooled_z   = (zscore(all_ptps) if len(all_ptps) > 1
                  else np.zeros(len(all_ptps)))
    pz_by_type = {}
    pos = 0
    for st, pa in ptps_per_stim.items():
        pz_by_type[st] = pooled_z[pos:pos + len(pa)]
        pos += len(pa)

    cum_off = 0
    for st, pa in reversed(list(ptps_per_stim.items())):
        n = len(pa)
        x = np.arange(n, dtype=float)
        if n >= 2:
            slp, icp  = np.polyfit(x, pa, 1)
            resid     = pa - (slp * x + icp)
            det_mean  = resid + float(pa.mean())
            sd        = float(resid.std(ddof=1))
            det_z     = resid / sd if sd > 0 else np.zeros(n)
        else:
            resid, det_mean, det_z = np.zeros(n), pa.copy().astype(float), np.zeros(n)
        for off in range(1, n + 1):
            ti  = n - off
            abs_i = cum_off + off
            pz  = round(float(pz_by_type[st][ti]), 3)
            dv  = round(float(det_mean[ti]),        4)
            dz  = round(float(det_z[ti]),            3)
            latency_rows_auto  [-abs_i][17] = pz  # Z_PTP_Pooled
            latency_rows_auto  [-abs_i][18] = dv  # PTP_Detrended(mV)
            latency_rows_auto  [-abs_i][19] = dz  # PTP_Detrended_Z
            latency_rows_manual[-abs_i][17] = pz
            latency_rows_manual[-abs_i][18] = dv
            latency_rows_manual[-abs_i][19] = dz
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
LAT_COLS = [
    # Identification
    "File", "StimType", "Stim_Label", "Segment",
    "Limb", "Measure",
    # Core MEP metrics
    "PTP(mV)", "Latency(ms)",
    "cSP_Duration(ms)",     # duration MEP offset → EMG return
    "cSP_MEP_Offset(ms)",    # time of MEP offset (cSP onset) re: stim
    "cSP_EMG_Return(ms)",    # time of EMG return (cSP offset) re: stim
    "MEP_cSP_Ratio",        # PTP(mV) / cSP duration(ms), Orth & Rothwell 2004
    "AUC(mV*s)",
    # Pre-stimulus baseline
    "PreStimRMS", "PreStimPTP",
    # Z-scores and detrended values
    "Z_PreStimRMS", "Z_PTP_Within", "Z_PTP_Pooled",
    "PTP_Detrended(mV)", "PTP_Detrended_Z",
    # Trial status
    "Outlier_Decision",
    # Normalisation (blank if not configured)
    "Reference_Type",     # which condition was the denominator
    "Reference_Mean(mV)", # mean amplitude of reference (plateau-detected if applicable)
    "Reference_N",        # trials contributing to reference mean
    "Normalised_PTP",     # PTP / Reference_Mean  (raw ratio)
    # Annotations — always last
    "Manual_Note",
]


def pipeline_write_outputs(latency_manual, results_out, bids_prefix):
    """Write all result CSVs to results_out directory.

    Outputs
    -------
    <prefix>_trials.csv               — trial-level data (all metrics, clean trials)
    <prefix>_trials_with_outliers.csv — same including outlier trials
    <prefix>_summary.csv              — mean ± SD per stim type (clean trials only)
    <prefix>_summary_with_outliers.csv — same including outlier trials
    """
    # Summary file headers — mirrors LAT_COLS with mean/SD for every metric,
    # plus trial counts so both files report the same variables.
    SUM_HDR = [
        "File", "StimType", "Stim_Label",
        "N_Total", "N_Included", "N_Outliers",
        # Core MEP
        "Mean_PTP(mV)", "SD_PTP(mV)",
        "Mean_Latency(ms)", "SD_Latency(ms)",
        # cSP
        "Mean_cSP_Duration(ms)", "SD_cSP_Duration(ms)",
        "Mean_cSP_MEP_Offset(ms)", "SD_cSP_MEP_Offset(ms)",
        "Mean_cSP_EMG_Return(ms)", "SD_cSP_EMG_Return(ms)",
        "Mean_MEP_cSP_Ratio", "SD_MEP_cSP_Ratio",
        # AUC
        "Mean_AUC(mV*s)", "SD_AUC(mV*s)",
        # Baseline
        "Mean_PreStimRMS", "SD_PreStimRMS",
        "Mean_PreStimPTP", "SD_PreStimPTP",
        # Normalisation
        "Mean_Normalised_PTP", "SD_Normalised_PTP",
        "Reference_Type", "Reference_Mean(mV)", "Reference_N",
        # Detrended
        "Mean_PTP_Detrended(mV)", "SD_PTP_Detrended(mV)",
    ]

    def _alpha_sort(df, col):
        cats = sorted(df[col].unique())
        df[col] = pd.Categorical(df[col], categories=cats, ordered=True)
        return df.sort_values([col, "File"]).reset_index(drop=True)

    def _p(name): return os.path.join(results_out, f"{bids_prefix}_{name}")

    # ── Trial-level files ─────────────────────────────────────────────────────
    if latency_manual:
        # Separate clean vs all-trials using Outlier_Decision column
        df_all = _alpha_sort(
            pd.DataFrame(latency_manual, columns=LAT_COLS),
            "StimType").sort_values(["StimType", "File", "Segment"])
        df_clean = df_all[df_all["Outlier_Decision"] != "Outlier"]
        df_clean.to_csv(_p("trials.csv"),               index=False)
        df_all.to_csv(  _p("trials_with_outliers.csv"), index=False)

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

        def _build_summary(df):
            rows = []
            for (fname, st, lbl), grp in df.groupby(
                    ["File", "StimType", "Stim_Label"], sort=False):
                clean = grp[grp["Outlier_Decision"] != "Outlier"]
                n_tot = len(grp)
                n_inc = len(clean)
                n_out = n_tot - n_inc
                rows.append([
                    fname, st, lbl,
                    n_tot, n_inc, n_out,
                    _mn(_col(clean,"PTP(mV)")),         _sd(_col(clean,"PTP(mV)")),
                    _mn(_col(clean,"Latency(ms)")),      _sd(_col(clean,"Latency(ms)")),
                    _mn(_col(clean,"cSP_Duration(ms)")), _sd(_col(clean,"cSP_Duration(ms)")),
                    _mn(_col(clean,"cSP_MEP_Offset(ms)")),_sd(_col(clean,"cSP_MEP_Offset(ms)")),
                    _mn(_col(clean,"cSP_EMG_Return(ms)")),_sd(_col(clean,"cSP_EMG_Return(ms)")),
                    _mn(_col(clean,"MEP_cSP_Ratio")),    _sd(_col(clean,"MEP_cSP_Ratio")),
                    _mn(_col(clean,"AUC(mV*s)")),        _sd(_col(clean,"AUC(mV*s)")),
                    _mn(_col(clean,"PreStimRMS")),       _sd(_col(clean,"PreStimRMS")),
                    _mn(_col(clean,"PreStimPTP")),       _sd(_col(clean,"PreStimPTP")),
                    _mn(_col(clean,"Normalised_PTP")),   _sd(_col(clean,"Normalised_PTP")),
                    _str_col(clean,"Reference_Type"),
                    _mn(_col(clean,"Reference_Mean(mV)")),
                    _mn(_col(clean,"Reference_N")),
                    _mn(_col(clean,"PTP_Detrended(mV)")),_sd(_col(clean,"PTP_Detrended(mV)")),
                ])
            return pd.DataFrame(rows, columns=SUM_HDR)

        df_all = pd.DataFrame(latency_manual, columns=LAT_COLS)
        df_clean_only = df_all[df_all["Outlier_Decision"] != "Outlier"]

        _build_summary(df_clean_only) \
            .to_csv(_p("summary.csv"),               index=False)
        _build_summary(df_all) \
            .to_csv(_p("summary_with_outliers.csv"), index=False)


def pipeline_generate_plots(trace_stats, time_axis, segments_metadata,
                             color_map, custom_labels, plot_included,
                             figures_out, figures_all, bids_prefix, name, unit,
                             enable_individual, cfg: PipelineConfig):
    """Save the combined trace figure and (optionally) per-stim-type figures."""
    def _ylab(base="EMG"):
        return f"{base} ({unit})" if unit else base

    # Combined figure
    fig = matplotlib.figure.Figure(figsize=(12, 6))
    ax  = fig.add_subplot(111)
    for stim_type, segments, emg_segments, mean_trace, mean_ptp in trace_stats:
        if plot_included and not plot_included.get(stim_type, True):
            continue
        color      = color_map.get(stim_type, "gray")
        label_name = custom_labels.get(stim_type, stim_type)
        for s in emg_segments:
            ax.plot(time_axis, s, color=color, alpha=0.2, linewidth=0.5)
        ax.plot(time_axis, mean_trace, color=color, linewidth=3,
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
        for stim_type, segments, emg_segments, mean_trace, mean_ptp in trace_stats:
            color      = color_map.get(stim_type, "gray")
            label_name = custom_labels.get(stim_type, stim_type)
            fig_i = matplotlib.figure.Figure(figsize=(12, 6))
            ax_i  = fig_i.add_subplot(111)
            for s in emg_segments:
                ax_i.plot(time_axis, s, color=color, alpha=0.2, linewidth=0.5)
            ax_i.plot(time_axis, mean_trace, color=color, linewidth=3,
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

def run_pipeline(input_path,
                 pre_ms,
                 post_ms,
                 ptp_start,
                 ptp_end,
                 *,
                 gap_ms_map=None,
                 review_outliers_cb=None,
                 show_inspector_cb=None,
                 gui_enable_inspector=False,
                 channel_idx=0,
                 prestim_ms,
                 apply_humbug,
                 humbug_harmonics=6,
                 apply_filter, apply_bandpass, apply_notch,
                 highpass, lowpass, notch_freq, notch_q,
                 filter_order,
                 filter_family="butter", cheby_ripple=1.0,
                 flexible_bandpass=False, hp_order=2, lp_order=2,
                 custom_labels=None, color_map=None, plot_included=None,
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
                 latency_map=None,
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
        latency_map=latency_map or {},
        outlier_threshold=outlier_threshold,
        enable_outlier_review=enable_outlier_review,
        custom_labels=custom_labels or {},
        color_map=color_map or {},
        plot_included=plot_included or {},
        gap_ms_map=gap_ms_map or {},
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
    _bids_prefix = meta.bids_prefix() or pathlib.Path(input_path).stem

    def _bids_path(suffix):
        return os.path.join(_deriv_base, f"{_bids_prefix}_{suffix}")

    def _write_sidecar(csv_path, extra=None):
        filter_cfg = dict(highpass=highpass, lowpass=lowpass,
                          apply_bandpass=apply_bandpass,
                          apply_notch=apply_notch, notch_freq=notch_freq,
                          notch_q=notch_q, apply_humbug=apply_humbug,
                          humbug_harmonics=humbug_harmonics,
                          filter_order=filter_order)
        sidecar = meta.to_sidecar(input_path, filter_cfg)
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
                crop_start=crop_start, crop_end=crop_end)
            stim_types = sorted(stim_times)

            log_callback(f"📂 Processing {name}  (fs={fs} Hz, {len(stim_types)} stim type(s))")

            # ── Stage 2: Filter ───────────────────────────────────────────────
            emg = pipeline_apply_filters(emg, fs, cfg)

            # ── Stage 3: Extract segments ─────────────────────────────────────
            samples_before = int(pre_ms  * fs / 1000)
            samples_after  = int(post_ms * fs / 1000)
            time_axis      = np.linspace(-pre_ms, post_ms,
                                         samples_before + samples_after,
                                         endpoint=False)
            ptp_start_idx  = samples_before + int(ptp_start * fs / 1000)
            ptp_end_idx    = samples_before + int(ptp_end   * fs / 1000)

            all_segments = pipeline_extract_segments(
                time, emg, stim_times, stim_types, fs, cfg)

            # ── Stage 4: Save "with-outliers" CSVs and figures ────────────────
            for stim_type, segs in all_segments.items():
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
                mean_tr   = emg_segs.mean(axis=0)
                mean_ptp  = float(_np_ptp(mean_tr))

                ptps, rms_vals, preptp, rms_z, ptp_z, out_idx = pipeline_detect_outliers(
                    emg_segs, pre_segs, ptp_start_idx, ptp_end_idx, cfg)

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
                    outlier_set=outlier_set)

                if rejected:
                    keep = np.ones(len(emg_segs), dtype=bool)
                    keep[rejected] = False
                    emg_segs  = emg_segs[keep]
                    pre_segs  = pre_segs[keep]
                    stats_per_type[stim_type]["segs"] = emg_segs

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
                # Build inspector segments with full prestim_ms pre-stim
                _insp_sb = int(cfg.prestim_ms * fs / 1000)
                _insp_sa = int(cfg.post_ms    * fs / 1000)
                for _t0 in [t for t in stim_times.get(stim_type, [])
                            if time.min() <= t <= time.max()]:
                    _ix  = int(np.argmin(np.abs(time - _t0)))
                    _seg = emg[max(0,_ix-_insp_sb):_ix+_insp_sa]
                    if len(_seg) == _insp_sb + _insp_sa:
                        segments_inspector[stim_type].append(_seg)
                trace_stats.append((stim_type, segs, emg_segs, mean_tr, mean_ptp))

                ptp_data.setdefault(stim_type, []).extend(ptps.tolist())
                rms_data.setdefault(stim_type, []).extend(rms_vals.tolist())
                preptp_data.setdefault(stim_type, []).extend(preptp.tolist())
                # Full-range PTP (entire post-stim window, not just analysis window).
                # Used for reference conditions like M-wave that occur before
                # the PTP analysis window (10-50ms) and would otherwise give ~0.
                _full_ptps = _np_ptp(emg_segs[:, samples_before:], axis=1)
                full_ptp_data.setdefault(stim_type, []).extend(_full_ptps.tolist())

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
                _chan_names_all = list_waveform_channels(raw_file)
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
                segments_metadata = show_inspector_cb(
                    _insp_segs, fs, cfg.prestim_ms, post_ms, unit,
                    custom_labels, color_map, prestim_ms,
                    extra_segs=_extra_segs,
                    wide_window_s=cfg.wide_window_s)

            # Parse inspector metadata
            excluded_sets = defaultdict(set)
            notes_map     = {}
            for (stype, idx), m in segments_metadata.items():
                if m.get("exclude", False):
                    excluded_sets[stype].add(idx)
                if "note" in m:
                    notes_map[(stype, idx)] = m["note"]

            # ── Stage 7: Quantify all segments ────────────────────────────────
            _ptps_per_stim = {}
            for stim_type, info in stats_per_type.items():
                auto_r, man_r, sum_r, with_r, ptps_arr = pipeline_quantify_segments(
                    stim_type,
                    info["segs_all"], info["prestim_all"],
                    info["outlier_set"], excluded_sets[stim_type],
                    segments_metadata,
                    ptp_start_idx, ptp_end_idx,
                    fs, cfg, custom_labels or {}, name)

                latency_auto.extend(auto_r)
                latency_manual.extend(man_r)
                summary_rows.append(sum_r)
                with_out_rows.append(with_r)
                _ptps_per_stim[stim_type] = ptps_arr

            # ── Stage 8: Pooled z-scores and detrending ───────────────────────
            pipeline_compute_pooled_stats(_ptps_per_stim, latency_auto, latency_manual)

            # ── Stage 9: Plots ────────────────────────────────────────────────
            combined_plot = pipeline_generate_plots(
                trace_stats, time_axis, segments_metadata,
                cfg.color_map, cfg.custom_labels, cfg.plot_included,
                figures_out, figures_all, _bids_prefix, name, unit,
                enable_individual_plots, cfg)

            log_callback(f"✔️  Finished {name}")

        except Exception as e:
            import traceback
            log_callback(f"❌ Error processing {name}: {e}")
            log_callback(traceback.format_exc())

        if progress_callback:
            progress_callback(((file_i + 1) / len(txt_files)) * 100)

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
    pipeline_write_outputs(latency_manual,
                           results_out, _bids_prefix)

    # ── Write _trials.json sidecar ────────────────────────────────────────────
    # This is the file Stage 2 scans for. It must be written alongside
    # the trials.csv in the results/ folder.
    _trials_csv = os.path.join(results_out, f"{_bids_prefix}_trials.csv")
    if os.path.isfile(_trials_csv):
        _write_sidecar(_trials_csv, extra={
            "trials_csv": f"{_bids_prefix}_trials.csv",
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

