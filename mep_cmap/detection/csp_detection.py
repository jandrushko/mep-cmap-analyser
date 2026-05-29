"""
mep_cmap.detection.csp_detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cortical silent period detection via vectorised bootstrap criterion.

  • detect_csp_bootstrap
"""

import numpy as np


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
    pre_ms          : duration of pre-stim baseline to use (ms)
    search_start_ms : start of cSP search window (ms post-stim)
    search_end_ms   : end of cSP search window (ms post-stim)
    min_silence_ms  : minimum suppression duration to qualify (ms)
    min_return_ms   : minimum EMG return window (ms)
    criterion       : SD multiplier for suppression threshold
    significance    : bootstrap percentile for criterion_samples
    n_boot          : bootstrap iterations
    rms_window_ms   : RMS envelope smoothing window (ms)
    seed            : RNG seed for reproducibility
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

    # ── Vectorised CSP bootstrap ──────────────────────────────────────────────
    all_idx    = rng.integers(0, n_pre, size=(n_boot, n_pre))
    all_resamp = norm_pre[all_idx]

    r_mu  = all_resamp.mean(axis=1, keepdims=True)
    r_sd  = all_resamp.std(axis=1, ddof=1, keepdims=True)
    r_sd  = np.maximum(r_sd, 1e-9)
    lo    = np.maximum(r_mu - criterion * r_sd, MIN_FRAC)
    hi    = r_mu + criterion * r_sd

    sig_mask = (all_resamp < lo) | (all_resamp > hi)

    chance_lengths = []
    pad    = np.zeros((n_boot, 1), dtype=bool)
    padded = np.concatenate([pad, sig_mask, pad], axis=1)
    diffs  = np.diff(padded.view(np.uint8), axis=1)
    for b in range(n_boot):
        starts = np.where(diffs[b] == 1)[0]
        ends   = np.where(diffs[b] == 255)[0]   # uint8: -1 wraps to 255
        runs   = ends - starts
        chance_lengths.extend(runs.tolist())

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
