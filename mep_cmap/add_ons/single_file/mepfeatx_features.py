"""
MEPFeatX feature extraction — built-in MEP-CMAP Analyser add-on
================================================================

A faithful Python port of the MEPFeatX MATLAB package
(https://github.com/NeuromodulationUEF/MEPFeatX), which extracts a comprehensive
set of morphological MEP features per trial.

Ported from the MEPFeatX source (extract_feature_each.m, find_latency.m,
find_duration.m, find_nPhases.m, find_zc.m, get_threshold_value.m). Method and
definitions: Nguyen et al. (2025), "MEPFeatX—automated feature extraction of
motor-evoked potentials in transcranial magnetic stimulation," Front. Neurosci.
18:1415257. See also Nguyen et al. 2019 (IEEE TNSRE 27:1521) and 2023 (Sci. Rep.
13:10604).

MEPFeatX license (retained per BSD-3-Clause):
    Copyright (c) 2023, NeuromodulationUEF. Ownership held by the University of
    Eastern Finland. Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the copyright notice, the
    conditions and the disclaimer are retained. THIS SOFTWARE IS PROVIDED "AS
    IS"; see the full BSD-3-Clause text in the MEPFeatX repository.

Output: <prefix>_mepfeatx.csv, one row per trial, columns:
    File, StimType, Trial, Amplitude(uV), Latency(ms), AUC(uV*ms), Thickness(ms),
    nTurns, nPhases, Duration(ms), T1T(ms), T1A(uV), T2T(ms), T2A(uV),
    timeDiff(ms), ampRatio, MEP_Found

Faithful to the MEPFeatX algorithm; two documented adaptations for use as a
general tool (MEPFeatX targets 3 kHz recordings in microvolts):
  • amplitudes are converted to microvolts via context.unit so the absolute
    50 uV MEP-presence floor is meaningful;
  • the -40..-5 ms background window is clamped to whatever pre-stimulus the
    bundle actually contains (MEPFeatX assumes >= 50 ms pre-stim).
The onset latency window (t_onset) is taken per condition from the tool's
latency settings (context.config['latency_map']), falling back to the analysis
window, mirroring MEPFeatX's per-muscle onset table.
"""

import os
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import CubicSpline
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

ADDON_NAME        = "mepfeatx"
ADDON_DESCRIPTION = "MEPFeatX morphological MEP features (amplitude, latency, duration, turns, phases, thickness, T1/T2)"
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "Port of MEPFeatX (Nguyen et al. 2025, Univ. of Eastern Finland, BSD-3)"
ADDON_SCOPE       = "single_file"

# Per-add-on settings surfaced as controls in the Add-ons tab. Each value is
# passed into context.config under its "key" when the add-on is run.
ADDON_SETTINGS = [
    {
        "key": "mepfeatx_noise_gate_ratio",
        "label": "Noise-gate ratio",
        "help": ("Reject a trial when 3xMAD(baseline) > ratio x p2p. MEPFeatX "
                 "default 0.2; raise (~0.35-0.5) to keep MEPs recorded during "
                 "voluntary contraction, where the pre-stim baseline is active EMG."),
        "type": "float",
        "default": 0.2,
        "min": 0.0,
        "max": 2.0,
    },
]

OUT_COLUMNS = [
    # File/StimType/Segment are the join keys used by Second Level ▸ Group
    # Analysis to merge these features back onto the core per-trial table.
    # `File` must be the SOURCE FILE NAME (as in _trials.csv), not the BIDS
    # prefix, and `Segment` is 1-based (Trial + 1) because the core table
    # numbers trials from 1. `Trial` is retained as the 0-based index into the
    # waveform stack, for backward compatibility.
    "File", "StimType", "Segment", "Trial",
    "Amplitude(uV)", "Latency(ms)", "AUC(uV*ms)", "Thickness(ms)",
    "nTurns", "nPhases", "Duration(ms)",
    "T1T(ms)", "T1A(uV)", "T2T(ms)", "T2A(uV)",
    "timeDiff(ms)", "ampRatio", "MEP_Found", "Reject_Reason",
]


# ─────────────────────────────────────────────────────────────────────────────
# Small primitives (faithful to the MATLAB helpers)
# ─────────────────────────────────────────────────────────────────────────────
def _mad(x):
    """MATLAB mad(x) default = MEAN absolute deviation about the mean."""
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x - np.mean(x))))


def _find_zc(data, threshold, updown):
    """Port of find_zc.m — indices k where data crosses +threshold (and, when
    updown == 0, also -threshold). Returns a sorted unique 0-based array."""
    data = np.asarray(data, float)
    d = data - threshold
    idx = np.where(d[:-1] * d[1:] <= 0)[0]
    zc = list(idx)
    if updown == 0:
        d2 = data + threshold
        idx2 = np.where(d2[:-1] * d2[1:] < 0)[0]
        zc.extend(list(idx2))
    return np.array(sorted(set(int(i) for i in zc)), dtype=int)


def _peaks(y, min_height=None, min_prominence=None):
    """scipy find_peaks wrapper mirroring MATLAB findpeaks(y, 'minPeakHeight',
    'minPeakProminence'); returns peak indices."""
    kw = {}
    if min_height is not None:
        kw["height"] = min_height
    if min_prominence is not None and min_prominence > 0:
        kw["prominence"] = min_prominence
    idx, _ = find_peaks(np.asarray(y, float), **kw)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Latency  (port of find_latency.m)
# ─────────────────────────────────────────────────────────────────────────────
def _find_latency(it, iy, lat_thresh, baseline_threshold, amp_threshold):
    lat_mask = (it >= lat_thresh[0]) & (it <= lat_thresh[1])
    lat_amp = np.abs(iy[lat_mask])
    lat_time = it[lat_mask]
    if lat_amp.size < 2:
        return np.nan

    ind = _peaks(lat_amp, min_height=amp_threshold)
    if ind.size:
        xx = int(np.argmax(lat_amp[ind]))
        cut_t = lat_time[ind[xx]]
        lat_mask = (it >= lat_thresh[0]) & (it < cut_t)
        lat_amp = np.abs(iy[lat_mask])
        lat_time = it[lat_mask]
        if lat_amp.size < 2:
            return np.nan

    zc_list = _find_zc(lat_amp, baseline_threshold, 0)

    amp_increment = 5.0
    n = 2000
    while zc_list.size == 0:
        n -= 1
        zc_list = _find_zc(lat_amp, baseline_threshold + amp_increment, 0)
        amp_increment += 1.0
        if (baseline_threshold + amp_increment > amp_threshold) or n == 0:
            break

    if zc_list.size == 0:
        zc_list = _find_zc(np.diff(lat_amp), 0, 0)
        if zc_list.size == 0:
            return np.nan

    onset = zc_list[-1]
    zc_amp = lat_amp[zc_list]
    if np.all(zc_amp > baseline_threshold * 3):
        onset = zc_list[int(np.argmin(zc_amp))]
    elif zc_list.size >= 3:
        a, b = zc_list[0], zc_list[-1]
        cur_time = lat_time[a:b + 1]
        cur_amp = lat_amp[a:b + 1]
        loc = _peaks(cur_amp, min_height=baseline_threshold * 3)
        if loc.size:
            lp = _find_zc(lat_time[zc_list] - cur_time[loc[0]], 0, 1)
            if lp.size:
                onset = zc_list[lp[0]]

    onset = int(np.clip(onset, 0, lat_time.size - 1))
    return float(lat_time[onset])


# ─────────────────────────────────────────────────────────────────────────────
# Duration / endpoint  (port of find_duration.m)
# ─────────────────────────────────────────────────────────────────────────────
def _find_duration(it, iy, t2t, dur_thresh, baseline_threshold, amp_threshold, n_sat):
    mask = (it >= t2t + 5) & (it <= dur_thresh[1] + 21)
    t = it[mask]
    y = np.abs(iy[mask])
    if y.size < 2:
        return None, baseline_threshold

    ind = _find_zc(y, amp_threshold, 1)
    if ind.size:
        ind = ind[t[ind] <= dur_thresh[1] - 5]
        if ind.size:
            y = y[ind[-1]:]
            t = t[ind[-1]:]
    if y.size < 2:
        return None, baseline_threshold

    zc_list = _find_zc(y, baseline_threshold, 1)
    median_list = _find_zc(y, np.median(y), 1)
    zc_list = zc_list[t[zc_list] <= dur_thresh[1]] if zc_list.size else zc_list
    median_list = median_list[t[median_list] <= dur_thresh[1]] if median_list.size else median_list

    if median_list.size == 0:
        amp_increment = 3 * _mad(y)
        n = 2000
        while median_list.size == 0 or (t[median_list] > dur_thresh[1]).all():
            n -= 1
            median_list = _find_zc(y, np.median(y) + amp_increment, 0)
            if (amp_increment > np.max(y) and median_list.size == 0) or n == 0:
                return None, baseline_threshold
            amp_increment += baseline_threshold
            if median_list.size and not (t[median_list] > dur_thresh[1]).all():
                break

    median_list = median_list[median_list + n_sat <= y.size - 1]
    if median_list.size == 0:
        return None, baseline_threshold

    end_point_index = None
    sat = np.zeros((median_list.size, 2))
    for kk, ml in enumerate(median_list):
        seg = y[ml:ml + n_sat + 1]
        sat[kk] = [np.median(seg), _mad(seg)]
    ok = np.where(sat[:, 1] <= baseline_threshold)[0]
    if ok.size:
        end_point_index = int(median_list[ok[0]])
    if end_point_index is None:
        end_point_index = int(median_list[0])

    if zc_list.size:
        closest = int(np.argmin(np.abs(end_point_index - zc_list)))
        end_point_index = int(zc_list[closest])

    end_point_index = int(np.clip(end_point_index, 0, y.size - 1))
    sat_seg = y[end_point_index:end_point_index + n_sat + 1]
    sateline_threshold = 3 * _mad(sat_seg) if sat_seg.size else baseline_threshold
    return (float(t[end_point_index]), float(y[end_point_index])), sateline_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Phases  (port of find_nPhases.m)
# ─────────────────────────────────────────────────────────────────────────────
def _find_nphases(it, iy, latency, t_end, t_turns, sateline_threshold):
    if len(t_turns) == 0:
        return 2
    mask = (it >= t_turns[0]) & (it <= t_turns[-1])
    t = it[mask]
    y = iy[mask]
    if t.size < 2:
        return 2
    zc = _find_zc(y, sateline_threshold, 0)
    if zc.size == 0:
        return 2
    zc = zc[np.concatenate(([True], np.diff(t[zc]) >= 1.5))]  # drop close crossings
    if zc.size and (t_end - t[zc[-1]] > 10):
        zc = np.append(zc, t.size - 1)
    phase_bounds = np.concatenate(([latency], t[zc], [t_end]))
    kept = []
    for k in range(len(zc)):
        lo, hi = phase_bounds[k], phase_bounds[k + 1]
        roi = (t >= lo) & (t < hi)
        if np.any(np.isin(np.round(t_turns, 6), np.round(t[roi], 6))):
            kept.append(phase_bounds[k + 1])
        elif kept:
            kept[-1] = phase_bounds[k + 1]
    phase_dur2 = [latency] + kept + [t_end]
    return int(max(len(phase_dur2) - 1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial feature extraction  (port of extract_feature_each.m)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_feature_each(t, y, fs, thr, want_detail=False):
    """Return (features13, found, detail, reason). On a missed MEP returns
    (None, False, None, reason) where reason is a short "category: details"
    string for diagnostics; on success reason is None."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # 10x cubic-spline upsampling (MATLAB interp1 'spline' = not-a-knot)
    it = np.linspace(t[0], t[-1], t.size * 10)
    iy = CubicSpline(t, y)(it)

    t_onset = list(thr["t_onset"])
    t_end_point = thr["t_end_point"]
    amp_min = thr["amp_min"]
    t_background = thr["t_background"]
    t_first_peak = thr["t_first_peak"]

    bg = iy[(it > t_background[0]) & (it < t_background[1])]
    if bg.size < 2:
        return None, False, None, "insufficient_baseline: <2 samples in -40..-5 ms"
    iy = iy - np.mean(bg)
    baseline_threshold = 3 * _mad(bg)
    baseline_p2p = 6 * _mad(bg)
    if baseline_threshold <= 0:
        baseline_threshold = 1e-9
        baseline_p2p = 2e-9

    n_sat = int(round(0.2 * fs))          # 600 samples at fs=3 kHz on the 10x grid

    # Two largest peaks in [t_onset(1), t_onset(2)+40]
    roi = (it > t_onset[0]) & (it < t_onset[1] + 40)
    if not np.any(roi):
        return None, False, None, "empty_peak_window: no samples in onset ROI"
    t_roi = it[roi]
    y_roi = iy[roi]
    t1_index = int(np.argmax(y_roi)); t1a = float(y_roi[t1_index]); t1t = float(t_roi[t1_index])
    t2_index = int(np.argmin(y_roi)); t2a = float(y_roi[t2_index]); t2t = float(t_roi[t2_index])

    posMep = False
    if t2t - t1t < 0:                     # negative peak first -> flip
        posMep = True
        iy = -iy
        y_roi = iy[roi]
        t1_index = int(np.argmax(y_roi)); t1a = float(y_roi[t1_index]); t1t = float(t_roi[t1_index])
        t2_index = int(np.argmin(y_roi)); t2a = float(y_roi[t2_index]); t2t = float(t_roi[t2_index])

    p2p = abs(t1a) + abs(t2a)

    if p2p < amp_min:
        return None, False, None, f"small_p2p: {p2p:.1f} < {amp_min:.0f}uV"
    if abs(t1a) < baseline_threshold:
        return None, False, None, f"t1a_below_baseline: |T1A| {abs(t1a):.1f} < 3xMAD {baseline_threshold:.1f}"
    ngr = thr.get("noise_gate_ratio", 0.2)
    if baseline_threshold > p2p * ngr:
        return None, False, None, f"noise_gate: 3xMAD {baseline_threshold:.1f} > {ngr:g}*p2p {p2p*ngr:.1f}"
    if max(t1t, t2t) > t_end_point[1]:
        return None, False, None, f"peaks_late: max(T1T,T2T) {max(t1t, t2t):.1f} > {t_end_point[1]:.1f}ms"
    if t1t > t2t:
        return None, False, None, f"peak_order: T1T {t1t:.1f} > T2T {t2t:.1f}"
    if t1t < t_first_peak:
        return None, False, None, f"t1_too_early: T1T {t1t:.1f} < {t_first_peak:.1f}ms"

    t_onset[1] = min(t_onset[1], t1t)
    amp_threshold = min(abs(p2p * 0.1), 50.0)

    latency = _find_latency(it, iy, t_onset, baseline_threshold, amp_threshold)
    if not np.isfinite(latency):
        return None, False, None, "no_onset: latency not found in window"

    dur_res, sateline_threshold = _find_duration(
        it, iy, t2t, t_end_point, baseline_threshold, amp_threshold, n_sat)
    if dur_res is None:
        return None, False, None, "no_endpoint: duration/endpoint not found"
    duration_end = dur_res[0]
    duration = duration_end - latency

    # Turns
    sp = (it > latency) & (it < duration_end)
    spike_y = iy[sp]
    tt = it[sp]
    if spike_y.size < 2:
        return None, False, None, "empty_region: onset >= endpoint"
    pos = _peaks(spike_y, min_height=amp_threshold, min_prominence=baseline_p2p / 2)
    neg = _peaks(-spike_y, min_height=amp_threshold, min_prominence=baseline_p2p / 2)
    spike_ind = np.array(sorted(set(list(pos) + list(neg))), dtype=int)
    if spike_ind.size == 0:
        return None, False, None, "no_turns_found: findpeaks empty"
    turns_t = tt[spike_ind]
    turns_a = spike_y[spike_ind]
    keep = np.abs(turns_a) >= baseline_p2p
    turns_t = turns_t[keep]; turns_a = turns_a[keep]
    if turns_t.size == 0:
        return None, False, None, "no_turns_above_6mad: all turns below 6xMAD"
    nTurns = int(turns_t.size)

    nPhases = _find_nphases(it, iy, latency, duration_end, turns_t, sateline_threshold)

    roi_auc = (it >= latency) & (it < duration_end)
    AUC = float(np.trapezoid(np.abs(iy[roi_auc]), it[roi_auc])) if np.any(roi_auc) else 0.0
    thickness = AUC / p2p if p2p else np.nan

    # flip peak amplitudes back for reporting if the signal was inverted
    r_t1a, r_t2a = (-t1a, -t2a) if posMep else (t1a, t2a)
    timeDiff = t2t - t1t
    ampRatio = abs(r_t1a / r_t2a) if r_t2a != 0 else np.nan

    features = [round(p2p, 4), round(latency, 4), round(AUC, 4),
                round(thickness, 4), nTurns, nPhases, round(duration, 4),
                round(t1t, 4), round(r_t1a, 4), round(t2t, 4), round(r_t2a, 4),
                round(timeDiff, 4), round(ampRatio, 4) if np.isfinite(ampRatio) else np.nan]

    detail = None
    if want_detail:
        disp = -1.0 if posMep else 1.0          # display in original polarity
        detail = {
            "it": it, "iy": iy * disp, "baseline_threshold": baseline_threshold,
            "latency": latency, "endpoint": duration_end,
            "t1t": t1t, "t1a": r_t1a, "t2t": t2t, "t2a": r_t2a,
            "turns_t": turns_t, "turns_a": turns_a * disp,
        }
    return features, True, detail, None


# ─────────────────────────────────────────────────────────────────────────────
# Add-on entry point
# ─────────────────────────────────────────────────────────────────────────────
def _unit_scale_to_uV(unit):
    """Factor to convert data in `unit` into microvolts."""
    if not unit:
        return 1.0, True                 # assume already uV; flag the assumption
    u = str(unit).strip().lower().replace("µ", "u")
    return ({"uv": 1.0, "microv": 1.0, "mv": 1000.0, "v": 1_000_000.0}.get(u, 1.0),
            u not in ("uv", "microv", "mv", "v"))


def _onset_window(cfg, stim_type):
    """t_onset [low, high] ms for this condition: prefer the tool's per-condition
    latency window, else the analysis (PTP) window, else a generic default."""
    lm = (cfg or {}).get("latency_map") or {}
    if stim_type in lm and lm[stim_type]:
        lo, hi = lm[stim_type]
        return [float(lo), float(hi)]
    lo = float((cfg or {}).get("ptp_start", 10))
    hi = float((cfg or {}).get("ptp_end", 40))
    return [lo, hi]

# ─────────────────────────────────────────────────────────────────────────────
# Figures (MEPFeatX Figure-1 style)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_trial(det, stim_type, trial_idx, feats, unit_label, out_png):
    import matplotlib.pyplot as plt
    it = det["it"]; iy = det["iy"]; bt = det["baseline_threshold"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(it, iy, lw=0.8, color="0.25", label="MEP")
    ax.axhline(bt, ls="--", lw=0.6, color="0.6")
    ax.axhline(-bt, ls="--", lw=0.6, color="0.6")
    ax.axvspan(det["latency"], det["endpoint"], color="tab:red", alpha=0.06)
    ax.axvline(0, color="0.8", lw=0.6)
    ax.axvline(det["latency"], color="tab:red", ls="--", lw=1.0, label="onset")
    ax.axvline(det["endpoint"], color="tab:blue", ls="--", lw=1.0, label="endpoint")
    ax.plot([det["t1t"], det["t2t"]], [det["t1a"], det["t2a"]], "v",
            color="tab:red", ms=8, label="T1/T2")
    if len(det["turns_t"]):
        ax.plot(det["turns_t"], det["turns_a"], "*", color="tab:orange", ms=9, label="turns")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel(f"Amplitude ({unit_label})")
    ax.set_title(f"{stim_type}  trial {trial_idx}    Amp={feats[0]:.0f} {unit_label}   "
                 f"Lat={feats[1]:.1f} ms   Dur={feats[6]:.1f} ms   "
                 f"nT={int(feats[4])} nP={int(feats[5])}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def _plot_montage(items, stim_type, unit_label, time_ms, out_png):
    import math
    import matplotlib.pyplot as plt
    n = len(items)
    if n == 0:
        return
    cols = 5; rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 1.9), squeeze=False)
    n_found = sum(1 for it in items if it[3])
    for i, (ti, det, fe, fnd, yv) in enumerate(items):
        ax = axes[i // cols][i % cols]
        if det is not None:
            ax.plot(det["it"], det["iy"], lw=0.5, color="0.25")
            ax.axvline(det["latency"], color="tab:red", lw=0.6)
            ax.axvline(det["endpoint"], color="tab:blue", lw=0.6)
            ax.plot([det["t1t"], det["t2t"]], [det["t1a"], det["t2a"]], "v",
                    color="tab:red", ms=3)
            ax.set_title(str(ti), fontsize=6)
        else:
            ax.plot(time_ms, yv, lw=0.4, color="0.7")
            ax.set_title(f"{ti}  (no MEP)", fontsize=6, color="0.5")
        ax.axvline(0, color="0.85", lw=0.4)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{stim_type} \u2014 {n_found}/{n} MEPs detected", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130); plt.close(fig)


def run(context):
    if not _HAVE_SCIPY:
        raise RuntimeError("mepfeatx add-on requires SciPy (scipy.signal, scipy.interpolate).")

    log = context.log
    fs = float(context.fs)
    _t_ms_for = getattr(context, "time_ms_for", None)
    time_ms = np.asarray(context.time_ms, float)
    scale, assumed = _unit_scale_to_uV(context.unit)
    if assumed:
        log(f"mepfeatx: amplitude unit '{context.unit}' unrecognised — assuming microvolts "
            f"(the 50 uV MEP floor may not apply correctly).")

    # The background window is clamped to the available pre-stimulus samples,
    # which is a property of each stimulus type's epoch rather than of the
    # file; computed inside the loop below.

    plot = bool((context.config or {}).get("mepfeatx_plot", True))
    debug = bool((context.config or {}).get("mepfeatx_debug", False))
    noise_gate_ratio = float((context.config or {}).get("mepfeatx_noise_gate_ratio", 0.2))
    # Source file name for the join key. context.trials is already filtered to
    # this recording, so any row carries it; fall back to the prefix when the
    # add-on is run without a per-trial table beside the bundle.
    file_name = context.bids_prefix
    _tr = getattr(context, "trials", None)
    if _tr is not None and getattr(_tr, "empty", True) is False \
            and "File" in _tr.columns and len(_tr["File"]):
        file_name = _tr["File"].iloc[0]

    rows = []
    n_found = 0
    reject_counts = {}
    details_by_stim = {}     # stim_type -> [(trial_idx, detail, feats, found, y_uV), ...]
    for stim_type, stack in context.segments.items():
        time_ms = np.asarray(_t_ms_for(stim_type) if _t_ms_for
                             else context.time_ms, float)
        pre_avail = float(time_ms.min())
        t_bg = [max(-40.0, pre_avail + 1.0 / fs), -5.0]
        if t_bg[1] <= t_bg[0]:
            raise ValueError(
                f"mepfeatx: '{stim_type}' has too little pre-stimulus baseline "
                f"for the -40..-5 ms window (its epoch starts at "
                f"{pre_avail:.1f} ms). Increase Pre (ms) for this stimulus "
                f"type on tab 1a.")
        stack = np.asarray(stack, float)
        t_onset = _onset_window(context.config, stim_type)
        thr = {
            "t_onset": t_onset,
            "t_end_point": [t_onset[0] + 8.0, t_onset[0] + 60.0],
            "t_first_peak": t_onset[0] + 2.0,
            "amp_min": 50.0,                      # microvolts
            "t_background": t_bg,
            "noise_gate_ratio": noise_gate_ratio,
        }
        bucket = details_by_stim.setdefault(stim_type, [])
        for trial_idx, trace in enumerate(stack):
            y_uV = trace * scale
            try:
                feats, found, detail, reason = _extract_feature_each(
                    time_ms, y_uV, fs, thr, want_detail=plot)
            except Exception as ex:                # never let one trial abort the run
                feats, found, detail, reason = None, False, None, f"errored: {ex}"
            if found:
                n_found += 1
                rows.append([file_name, stim_type, trial_idx + 1, trial_idx] + feats + [True, ""])
            else:
                rows.append([file_name, stim_type, trial_idx + 1, trial_idx] + [np.nan] * 13 + [False, reason])
                cat = (reason or "unknown").split(":")[0]
                reject_counts[cat] = reject_counts.get(cat, 0) + 1
                if debug:
                    log(f"mepfeatx[debug]: {stim_type} trial {trial_idx} rejected \u2014 {reason}")
            bucket.append((trial_idx, detail, feats, found, y_uV))

    df = pd.DataFrame(rows, columns=OUT_COLUMNS)
    out_path = os.path.join(context.addons_dir, f"{context.bids_prefix}_{ADDON_NAME}.csv")
    df.to_csv(out_path, index=False)
    log(f"mepfeatx: {len(rows)} trial(s), {n_found} with a detected MEP -> "
        f"{os.path.basename(out_path)}")
    if reject_counts:
        summary = ", ".join(f"{k}={v}" for k, v in
                             sorted(reject_counts.items(), key=lambda kv: -kv[1]))
        log(f"mepfeatx: rejected {sum(reject_counts.values())} trial(s) \u2014 {summary}")
        if not debug:
            log("mepfeatx: set config 'mepfeatx_debug'=True for per-trial rejection reasons.")

    # Figures: per-trial detail + per-condition montage, into the BIDS figures/ tree
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            fig_root = context.figures_dir or os.path.join(context.results_dir, "figures")
            fig_dir = os.path.join(fig_root, f"{context.bids_prefix}_{ADDON_NAME}_figures")
            os.makedirs(fig_dir, exist_ok=True)
            n_png = 0
            try:
                # Stimulus types such as ">" or "?" cannot go into a filename.
                from mep_cmap.utils import filename_token as _ftok
            except Exception:              # add-on used outside the package
                def _ftok(label):
                    return "".join(c if (c.isalnum() or c in "-_") else "_"
                                   for c in str(label))
            for stim_type, items in details_by_stim.items():
                _st_tok = _ftok(stim_type)
                for (ti, det, fe, fnd, yv) in items:
                    if det is not None:
                        _plot_trial(det, stim_type, ti, fe, "uV",
                                    os.path.join(fig_dir,
                                    f"{context.bids_prefix}_stim-{_st_tok}_trial-{ti:03d}.png"))
                        n_png += 1
                # This type's own axis, not whichever one the measuring loop
                # happened to leave behind: the figures are drawn in a second
                # pass, so the variable had already moved on.
                _mt = np.asarray(_t_ms_for(stim_type) if _t_ms_for
                                 else context.time_ms, float)
                _plot_montage(items, stim_type, "uV", _mt,
                              os.path.join(fig_dir,
                              f"{context.bids_prefix}_stim-{_st_tok}_montage.png"))
                n_png += 1
            log(f"mepfeatx: {n_png} figure(s) -> {os.path.basename(fig_dir)}/")
        except Exception as ex:
            log(f"mepfeatx: figure generation skipped ({ex})")

    return [out_path]
