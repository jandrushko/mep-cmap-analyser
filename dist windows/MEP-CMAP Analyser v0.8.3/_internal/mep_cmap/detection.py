"""
mep_cmap.detection
~~~~~~~~~~~~~~~~~~
Signal detection algorithms.

  • detect_mep_onset_peak_fraction  — MEP onset via peak-fraction + slope threshold
  • detect_csp_bootstrap            — cortical silent period via bootstrap criterion
"""

# VERSION STAMP — if you see this in the source, the new code is loaded
DETECTION_VERSION = "2025-bootstrap-ratio-v3"

import numpy as np


def detect_mep_onset_peak_fraction(signal, fs,
                                   pre_ms=100,
                                   poststim_start_ms=5,
                                   poststim_end_ms=60,
                                   peak_frac=0.15,
                                   min_consecutive=5,
                                   min_peak_amplitude=0.05,
                                   slope_threshold=0.05):
    """
    MEP onset detection using peak-fraction amplitude threshold + slope backtracking.

    Parameters
    ----------
    signal              : 1-D EMG segment (pre-stim + post-stim)
    fs                  : sampling frequency in Hz
    pre_ms              : ms of pre-stim data in the segment
    poststim_start_ms   : ms after stim to begin search
    poststim_end_ms     : ms after stim to end search
    peak_frac           : fraction of peak amplitude for threshold crossing
    min_consecutive     : min samples continuously above threshold
    min_peak_amplitude  : minimum MEP amplitude to qualify (mV)
    slope_threshold     : minimum slope (mV/ms) for onset confirmation

    Returns
    -------
    latency_ms : float, or None if no MEP detected
    """
    samples_before = int(pre_ms * fs / 1000)
    start_idx      = int((pre_ms + poststim_start_ms) * fs / 1000)
    end_idx        = int((pre_ms + poststim_end_ms)   * fs / 1000)

    rectified = np.abs(signal)
    post_stim = rectified[start_idx:end_idx]
    peak_val  = np.max(post_stim)

    if peak_val < min_peak_amplitude:
        return None

    threshold = peak_frac * peak_val
    above     = rectified[start_idx:end_idx] > threshold

    for i in range(len(above) - min_consecutive + 1):
        if np.all(above[i:i + min_consecutive]):
            for j in range(i - 1, 0, -1):
                idx           = start_idx + j
                slope_per_ms  = (rectified[idx] - rectified[idx - 1]) / (1000 / fs)
                if slope_per_ms < slope_threshold:
                    onset_sample = idx + 1
                    break
            else:
                onset_sample = start_idx
            return round((onset_sample - samples_before) * 1000 / fs, 2)

    return None


def detect_csp_bootstrap(
        emg_seg, fs, time_axis, *,
        pre_ms=100, search_start_ms=40, search_end_ms=400,
        min_silence_ms=25, min_return_ms=40,
        criterion=1.96, significance=0.99,
        n_boot=1000, rms_window_ms=10, seed=42,
        reason_out=None):
    """
    Per-trial cortical silent period detector — TMSMultiLab approach adapted
    for single trials.

    The threshold is applied to the RMS envelope rather than the raw signal.
    On single trials, voluntary EMG oscillates rapidly, making raw sample
    SD comparable to the mean; the RMS envelope averages over oscillation
    cycles and gives a stable suppression threshold.

    Algorithm
    ---------
    1.  Compute RMS envelope (rms_window_ms window).
    2.  Normalise by pre-stim RMS mean → baseline envelope ≈ 1.0.
    3.  Bootstrap minimum-duration criterion: resample pre-stim envelope,
        apply ±criterion SD threshold, record chance sequence lengths;
        criterion_samples = significance-th percentile.
    4.  Suppression threshold = max(base_mu - criterion*base_sd, 0.5)
        (floor prevents negative thresholds on flat/noisy baselines).
    5.  Find suppressed epochs meeting criterion_samples in search window.
    6.  Return first valid epoch as (start_idx, end_idx), or None.

    Parameters
    ----------
    emg_seg         : 1-D EMG segment array
    fs              : sampling frequency in Hz
    time_axis       : time axis in ms (same length as emg_seg)
    reason_out      : optional list — failure/success message appended here

    Returns
    -------
    (start_idx, end_idx) or None
    """
    MIN_FRAC = 0.5

    def _fail(msg):
        if reason_out is not None:
            reason_out.append(msg)
        return None

    rng     = np.random.default_rng(seed)
    rms_win = max(1, int(rms_window_ms * fs / 1000))
    smooth  = np.sqrt(np.convolve(emg_seg ** 2,
                                  np.ones(rms_win) / rms_win, mode='same'))

    prestim_mask = (time_axis >= -pre_ms) & (time_axis < 0.0)
    if prestim_mask.sum() < 10:
        return _fail("Too few pre-stim samples - increase pre-stim window")

    pre_env  = smooth[prestim_mask]
    pre_mean = float(pre_env.mean())
    if pre_mean < 1e-12:
        return _fail("Pre-stim signal is flat - no valid baseline")

    norm_env = smooth / pre_mean
    norm_pre = pre_env / pre_mean
    base_mu  = float(norm_pre.mean())
    base_sd  = max(float(norm_pre.std(ddof=1)) if len(norm_pre) > 1 else 1e-6, 1e-9)

    suppress_thresh = max(base_mu - criterion * base_sd, MIN_FRAC)

    n_pre = len(norm_pre)
    chance_lengths = []
    for _ in range(n_boot):
        resamp   = norm_pre[rng.integers(0, n_pre, n_pre)]
        r_mu     = float(resamp.mean())
        r_sd     = max(float(resamp.std(ddof=1)) if n_pre > 1 else 1e-6, 1e-9)
        lo       = max(r_mu - criterion * r_sd, MIN_FRAC)
        hi       = r_mu + criterion * r_sd
        sig_mask = (resamp < lo) | (resamp > hi)
        run = 0
        for v in sig_mask:
            if v:
                run += 1
            else:
                if run > 0:
                    chance_lengths.append(run)
                run = 0
        if run > 0:
            chance_lengths.append(run)

    min_sil_samp      = max(2, int(min_silence_ms * fs / 1000))
    criterion_samples = max(
        int(np.percentile(chance_lengths, significance * 100))
        if chance_lengths else min_sil_samp,
        min_sil_samp)

    si = int(np.searchsorted(time_axis, max(search_start_ms, float(time_axis[0]))))
    ei = int(np.searchsorted(time_axis, min(search_end_ms,   float(time_axis[-1]))))
    if si >= ei:
        return _fail("Search window empty - check Search start/end settings")

    search_norm = norm_env[si:ei]
    below       = search_norm < suppress_thresh
    valid_epochs, run_start = [], None
    for i, b in enumerate(below):
        if b and run_start is None:
            run_start = i
        elif not b and run_start is not None:
            if (i - run_start) >= criterion_samples:
                valid_epochs.append((run_start, i - 1))
            run_start = None
    if run_start is not None and (len(below) - run_start) >= criterion_samples:
        valid_epochs.append((run_start, len(below) - 1))

    if not valid_epochs:
        return _fail(f"No suppression >= {min_silence_ms} ms found in search window")

    csp_start_idx = si + valid_epochs[0][0]
    csp_end_idx   = si + valid_epochs[0][1]

    if (csp_end_idx - csp_start_idx) * 1000.0 / fs < min_silence_ms:
        return _fail(f"Detected epoch too short (< {min_silence_ms} ms)")

    if reason_out is not None:
        dur  = (csp_end_idx - csp_start_idx) * 1000.0 / fs
        t_on = float(time_axis[int(csp_start_idx)]) if int(csp_start_idx) < len(time_axis) else 0
        reason_out.append(f"Detected - onset ~{t_on:.0f} ms, duration ~{dur:.0f} ms")

    return int(csp_start_idx), int(min(csp_end_idx, len(smooth) - 1))


def detect_mep_onset_bootstrap(
        signal, fs, *,
        pre_ms=100,
        peak_search_start_ms=10,
        peak_search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        criterion=1.96,
        n_boot=500,
        min_duration_ms=5,
        artefact_blank_ms=2,
        seed=42):
    """
    Bootstrap-calibrated MEP onset detector — peak-anchored backward scan
    within a physiologically bounded window.

    Strategy
    --------
    1.  Confirm MEP exists via amplitude gate in PTP window.
    2.  Establish pre-stim bootstrap threshold.
    3.  Within [min_latency_ms, max_latency_ms], find the largest absolute
        peak — this anchors detection to the actual MEP rather than the
        first noise crossing.
    4.  Scan BACKWARD from that peak to find where the sliding-window mean
        first drops below threshold — that crossing is the onset.
    5.  Onset is bounded on the left by min_latency_ms (physiological floor).

    This approach is more robust than forward scanning because:
    - It ignores early artefacts and noise spikes before the MEP
    - Biphasic MEPs are handled correctly: the backward scan from the
      dominant peak crosses the true onset regardless of polarity
    - The physiological bounds prevent grossly implausible placements

    Parameters
    ----------
    min_latency_ms  : earliest plausible onset in ms post-stim (physiological
                      floor). If None, defaults to artefact_blank_ms.
    max_latency_ms  : latest plausible onset in ms post-stim (physiological
                      ceiling). If None, defaults to peak_search_end_ms.
    criterion       : z-score multiplier for onset threshold (default 1.96).
    min_duration_ms : sliding window width in ms (default 5 ms).
    artefact_blank_ms: hard minimum — search never starts before this.
    """
    samples_before    = int(pre_ms                        * fs / 1000)
    peak_search_start = int((pre_ms + peak_search_start_ms) * fs / 1000)
    peak_search_end   = int((pre_ms + peak_search_end_ms)   * fs / 1000)
    peak_search_end   = min(peak_search_end, len(signal))

    # Physiological window for onset search
    _min_lat = min_latency_ms if min_latency_ms is not None else artefact_blank_ms
    _max_lat = max_latency_ms if max_latency_ms is not None else peak_search_end_ms
    # Hard floor: never earlier than the artefact blank
    _min_lat = max(_min_lat, artefact_blank_ms)

    onset_search_start = int((pre_ms + _min_lat) * fs / 1000)
    onset_search_end   = int((pre_ms + _max_lat) * fs / 1000)
    onset_search_end   = min(onset_search_end, len(signal))
    win_samp           = max(2, int(min_duration_ms * fs / 1000))

    if peak_search_start >= peak_search_end:
        return None

    # ── Amplitude gate: confirm MEP exists in the PTP window ─────────────────
    post_abs = np.abs(signal[peak_search_start:peak_search_end])
    if len(post_abs) == 0 or post_abs.max() < min_peak_amplitude:
        return None

    # ── Pre-stim statistics on absolute signal ───────────────────────────────
    pre_abs = np.abs(signal[:samples_before])
    if len(pre_abs) < 5:
        return None

    rng      = np.random.default_rng(seed)
    n_pre    = len(pre_abs)
    boot_med = np.array([
        np.median(pre_abs[rng.integers(0, n_pre, n_pre)])
        for _ in range(n_boot)
    ])
    mu_abs    = float(np.median(boot_med))
    sigma_abs = float(np.std(pre_abs, ddof=1)) if len(pre_abs) > 1 else mu_abs
    if mu_abs < 1e-9:
        return None

    threshold = mu_abs + criterion * sigma_abs
    threshold = float(np.clip(threshold, mu_abs * 1.5, mu_abs * 5.0))

    # ── Peak-anchored backward scan ───────────────────────────────────────────
    # 1. Find largest absolute peak within the physiological latency window.
    if onset_search_start >= onset_search_end:
        return None

    search_win  = np.abs(signal[onset_search_start:onset_search_end])
    if len(search_win) == 0:
        return None
    peak_local  = int(np.argmax(search_win))
    peak_global = onset_search_start + peak_local   # index in full signal

    # 2. Scan backward from just before the peak to find MEP onset.
    #
    #    Strategy: at each candidate position i, compute the mean absolute
    #    signal in a forward window [i : i+win_samp]. Starting win_samp
    #    samples before the peak ensures the first window fully covers the
    #    peak and is above threshold.
    #
    #    To avoid placing onset on a background EMG noise spike, we require
    #    SUSTAINED below-threshold signal: the forward window must be below
    #    threshold for at least `sustain` consecutive positions before we
    #    accept it as genuine baseline. This prevents a single noise oscillation
    #    from triggering a false onset placement mid-slope.
    #
    sustain     = max(2, win_samp // 2)   # consecutive below-threshold windows needed
    scan_start  = max(onset_search_start, peak_global - win_samp)
    onset_idx   = onset_search_start      # default: physiological floor
    below_count = 0

    for i in range(scan_start, onset_search_start - 1, -1):
        i1 = min(len(signal), i + win_samp)
        win_mean = float(np.mean(np.abs(signal[i:i1]))) if i1 > i else 0.0
        if win_mean < threshold:
            below_count += 1
            if below_count >= sustain:
                # Sustained baseline found — onset is at the first above-threshold
                # position after this baseline, i.e. i + sustain
                onset_idx = min(i + sustain, peak_global)
                break
        else:
            below_count = 0   # reset — this was an above-threshold window
    # If loop completes without break, onset stays at physiological floor

    latency_ms = (onset_idx - samples_before) * 1000.0 / fs
    return round(float(latency_ms), 2)

