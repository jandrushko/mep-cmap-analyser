"""
Tests for two faults that only real recordings exposed.

Both were invisible to synthetic testing because the synthetic trials were too
clean: a burst that returns instantly to Gaussian noise, in a file containing a
single stimulus type.

1. OFFSET. The return threshold was `baseline_mean + criterion*SD`, an
   absolute level tied to a resting-quiet baseline. Real EMG does not settle
   back to that level within tens of milliseconds after a large response --
   after-activity and filter ringing keep it above. On a real resting file the
   detector found an offset on 15 of 127 trials with an onset, and the ones it
   found had a MEDIAN SNR OF 12 while the ones it missed had 38. It failed
   worse the cleaner the trial, which is the wrong way round and the signature
   of an absolute threshold rather than a tuning problem.

2. PTP WINDOW. One window for the whole file, while the latency profile is per
   stimulus type. On a real file containing both M-waves and MEPs, with the
   default 10-50 ms window, the M-wave conditions had the first 6 ms of every
   response excluded -- and an M-wave's whole biphasic deflection lasts
   5-15 ms. Amplitude, the primary outcome, was affected rather than latency.

The synthetic fixtures here deliberately include a decaying tail and a mixed
stimulus file, so both failures reproduce and stay reproduced.
"""

import numpy as np
import pytest

from mep_cmap.detection.offset_detection import detect_mep_offset
from mep_cmap.detection.quantification import compute_ptp
from mep_cmap.pipeline import (ARTEFACT_FLOOR_MS, PipelineConfig,
                               pipeline_detect_onsets,
                               ptp_window_for_stim_type)

FS = 5000.0
PRE_MS = 20
POST_MS = 400
SB = int(PRE_MS * FS / 1000)


def burst(x, onset_ms, dur_ms, amp):
    i0 = SB + int(onset_ms * FS / 1000)
    i1 = SB + int((onset_ms + dur_ms) * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    x[i0:i1] += amp * np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5
    return i1


def trial_with_tail(seed=0, amp=1.0, onset_ms=22.0, dur_ms=18.0,
                    noise=0.006, tail_frac=0.06, tau_ms=60.0):
    """A response followed by exponentially decaying after-activity.

    The tail is the part synthetic testing was missing. Without it the signal
    returns to the noise floor the instant the burst ends and any return
    criterion succeeds.
    """
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    x = rng.normal(0.0, noise, n)
    i1 = burst(x, onset_ms, dur_ms, amp)
    t = np.arange(n - i1) / FS * 1000.0
    x[i1:] += rng.normal(0, 1, n - i1) * amp * tail_frac * np.exp(-t / tau_ms)
    return x


def plain_trial(seed, onset_ms, dur_ms, amp, noise=0.006):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, int((PRE_MS + POST_MS) * FS / 1000))
    burst(x, onset_ms, dur_ms, amp)
    return x


# ── 1. Offset: the return threshold must scale with the response ──────────────

OFFSET_KW = dict(onset_ms=22.0, pre_ms=PRE_MS, search_end_ms=300, n_boot=200)


def _offset_rate(amp, peak_frac, max_duration_ms, n=15, **over):
    """`over` lets a caller widen search_end_ms, which otherwise caps the
    duration regardless of what max_duration_ms asks for."""
    kw = dict(OFFSET_KW)
    kw.update(over)
    got = [detect_mep_offset(trial_with_tail(s, amp=amp), FS,
                             peak_frac=peak_frac,
                             max_duration_ms=max_duration_ms, **kw)
           for s in range(n)]
    return [v for v in got if v is not None], n


@pytest.mark.parametrize("amp", [0.25, 1.0, 2.5, 8.0])
def test_offset_is_found_on_large_responses_with_a_decay_tail(amp):
    """The exact case that failed on real data: clean, large responses."""
    found, n = _offset_rate(amp, peak_frac=0.12, max_duration_ms=100.0)
    assert len(found) >= n - 1, (
        f"offset found on only {len(found)}/{n} trials at {amp} mV"
    )
    assert abs(float(np.mean(found)) - 40.0) < 4.0


@pytest.mark.parametrize("amp", [0.25, 1.0, 2.5, 8.0])
def test_an_absolute_threshold_alone_fails_on_those_same_responses(amp):
    """
    Guards the diagnosis, not just the fix. With the peak-relative floor
    disabled the detector must still fail here -- if this starts passing, the
    fixture has drifted back to being unrealistically clean and the test above
    no longer proves anything.
    """
    found, n = _offset_rate(amp, peak_frac=0.0, max_duration_ms=60.0)
    assert len(found) <= n // 3


def test_offset_success_no_longer_falls_away_as_snr_rises():
    """
    The real-data signature: found at median SNR 12, missed at median SNR 38.
    Detection rate must not be inversely related to amplitude.
    """
    rates = []
    for amp in (0.10, 0.25, 1.0, 2.5, 8.0):
        found, n = _offset_rate(amp, peak_frac=0.12, max_duration_ms=100.0)
        rates.append(len(found) / float(n))
    assert min(rates) >= 0.8, f"detection rate collapses with amplitude: {rates}"


def test_peak_fraction_floor_can_be_disabled():
    a = detect_mep_offset(trial_with_tail(0, amp=0.1), FS, peak_frac=0.0,
                          max_duration_ms=100.0, **OFFSET_KW)
    b = detect_mep_offset(trial_with_tail(0, amp=0.1), FS, peak_frac=0.12,
                          max_duration_ms=100.0, **OFFSET_KW)
    assert a is None or b is None or b <= a  # relative floor cannot be later


def test_offset_still_respects_min_and_max_duration():
    x = trial_with_tail(0, amp=1.0)
    assert detect_mep_offset(x, FS, peak_frac=0.12, max_duration_ms=2.0,
                             **OFFSET_KW) is None
    got = detect_mep_offset(x, FS, peak_frac=0.12, max_duration_ms=100.0,
                            min_duration_ms=5.0, **OFFSET_KW)
    assert got is None or (got - 22.0) >= 5.0


# ── 2. PTP window anchoring ───────────────────────────────────────────────────

M_WAVE = dict(onset_ms=4.0, dur_ms=8.0, amp=2.0, profile=(2.0, 12.0))
TMS_MEP = dict(onset_ms=20.0, dur_ms=18.0, amp=0.25, profile=(15.0, 35.0))

FILE_WIDE_START = SB + int(10 * FS / 1000)      # the shipped 10-50 ms default
FILE_WIDE_END = SB + int(50 * FS / 1000)


def _condition(spec, n=12):
    return np.vstack([plain_trial(s, spec["onset_ms"], spec["dur_ms"],
                                  spec["amp"]) for s in range(n)])


def _measure(spec, anchor):
    segs = _condition(spec)
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50,
                         onset_method="bigoni",
                         latency_map={"X": spec["profile"]},
                         ptp_anchor=anchor)
    onsets = pipeline_detect_onsets("X", segs, set(), FILE_WIDE_START,
                                    FILE_WIDE_END, FS, cfg,
                                    log_callback=lambda *a, **k: None)
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", onsets, FS, cfg, FILE_WIDE_START, FILE_WIDE_END, SB,
        log_callback=lambda *a, **k: None)
    measured = float(np.mean([compute_ptp(sg, s_i, e_i) for sg in segs]))
    truth = float(np.mean([compute_ptp(sg, SB, SB + int(60 * FS / 1000))
                           for sg in segs]))
    return measured, truth, win


def test_fixed_window_underestimates_an_m_wave_badly():
    """Guards the diagnosis: without anchoring the error must still be large."""
    measured, truth, win = _measure(M_WAVE, anchor=False)
    assert win is None
    assert measured < 0.75 * truth, (
        "the fixed-window M-wave error has disappeared; the fixture no longer "
        "reproduces the condition this feature exists for"
    )


def test_anchoring_recovers_the_m_wave_amplitude():
    measured, truth, win = _measure(M_WAVE, anchor=True)
    assert win is not None
    assert abs(measured - truth) / truth < 0.05


def test_anchoring_does_not_disturb_a_condition_the_window_already_suited():
    before, truth, _ = _measure(TMS_MEP, anchor=False)
    after, _, win = _measure(TMS_MEP, anchor=True)
    assert abs(before - truth) / truth < 0.05
    assert abs(after - before) / before < 0.02


def test_anchored_window_respects_the_configured_end_as_a_ceiling():
    _, _, win = _measure(TMS_MEP, anchor=True)
    assert win is not None
    assert win[1] <= 50.0


def test_anchored_window_respects_the_artefact_floor():
    _, _, win = _measure(M_WAVE, anchor=True)
    assert win is not None
    assert win[0] >= ARTEFACT_FLOOR_MS


def test_anchoring_is_on_by_default_for_new_installs():
    # On from v1.4.13. Existing preference files keep the value they have
    # been running with (see test_window_icon_and_wording).
    assert PipelineConfig().ptp_anchor is True


def test_too_few_onsets_falls_back_to_the_file_wide_window():
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50,
                         ptp_anchor=True, ptp_anchor_min_trials=8)
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", {0: 20.0, 1: 20.5}, FS, cfg,
        FILE_WIDE_START, FILE_WIDE_END, SB, log_callback=lambda *a, **k: None)
    assert win is None
    assert (s_i, e_i) == (FILE_WIDE_START, FILE_WIDE_END)


def test_no_detected_onsets_falls_back_to_the_file_wide_window():
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50, ptp_anchor=True)
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", {0: None, 1: None}, FS, cfg,
        FILE_WIDE_START, FILE_WIDE_END, SB, log_callback=lambda *a, **k: None)
    assert win is None
    assert (s_i, e_i) == (FILE_WIDE_START, FILE_WIDE_END)


def test_a_late_median_onset_cannot_invert_the_window():
    """Median onset beyond the configured end must not yield end <= start."""
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50, ptp_anchor=True)
    msgs = []
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", {i: 80.0 for i in range(10)}, FS, cfg,
        FILE_WIDE_START, FILE_WIDE_END, SB,
        log_callback=lambda *a: msgs.append(" ".join(str(x) for x in a)))
    assert win is None
    assert (s_i, e_i) == (FILE_WIDE_START, FILE_WIDE_END)
    assert any("PTP anchor" in m for m in msgs)


def test_anchoring_reports_the_window_it_chose():
    """A silently different measurement window would be worse than none."""
    segs = _condition(M_WAVE)
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50, onset_method="bigoni",
                         latency_map={"X": M_WAVE["profile"]}, ptp_anchor=True)
    onsets = pipeline_detect_onsets("X", segs, set(), FILE_WIDE_START,
                                    FILE_WIDE_END, FS, cfg,
                                    log_callback=lambda *a, **k: None)
    msgs = []
    ptp_window_for_stim_type("X", onsets, FS, cfg, FILE_WIDE_START,
                             FILE_WIDE_END, SB,
                             log_callback=lambda *a: msgs.append(
                                 " ".join(str(x) for x in a)))
    assert any("PTP anchor" in m and "->" in m for m in msgs)


# ── 3. Superseded defaults must reach existing installations ─────────────────

def test_a_stale_saved_value_is_migrated_to_the_new_default():
    """
    Raising a default reaches nobody who has pressed Apply, because the dialog
    writes every field on the tab. `mep_offset_max_duration_ms` was raised from
    60 to 100 ms after real recordings showed ~54 ms responses; on the machine
    where it mattered the stored 60 won and offset detection succeeded on 1
    trial of 81 instead of 81 of 81, with no error and no warning.
    """
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS,
                                             migrate_detection_defaults)

    data = {"mep_offset_max_duration_ms": 60.0}
    changed = migrate_detection_defaults(data)
    assert changed
    assert data["mep_offset_max_duration_ms"] == \
        DETECTION_DEFAULTS["mep_offset_max_duration_ms"]


def test_a_value_the_analyst_chose_is_never_overwritten():
    """Migration applies only where the stored value is still the old default."""
    from mep_cmap.detection.defaults import migrate_detection_defaults

    data = {"mep_offset_max_duration_ms": 75.0}
    assert migrate_detection_defaults(data) == []
    assert data["mep_offset_max_duration_ms"] == 75.0


def test_migration_is_idempotent():
    from mep_cmap.detection.defaults import migrate_detection_defaults

    data = {"mep_offset_max_duration_ms": 60.0}
    migrate_detection_defaults(data)
    first = data["mep_offset_max_duration_ms"]
    assert migrate_detection_defaults(data) == []
    assert data["mep_offset_max_duration_ms"] == first


def test_already_current_version_is_left_untouched():
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS_VERSION,
                                             migrate_detection_defaults)

    data = {"mep_offset_max_duration_ms": 60.0,
            "detection_defaults_version": DETECTION_DEFAULTS_VERSION}
    assert migrate_detection_defaults(data) == []
    assert data["mep_offset_max_duration_ms"] == 60.0


def test_superseded_table_only_names_real_settings():
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS,
                                             SUPERSEDED_DEFAULTS)

    for version, table in SUPERSEDED_DEFAULTS.items():
        for key in table:
            assert key in DETECTION_DEFAULTS, (
                f"SUPERSEDED_DEFAULTS[{version}] names '{key}', which is not a "
                f"detection setting"
            )


def test_reset_clears_every_stored_detection_setting():
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS,
                                             reset_detection_defaults)

    from mep_cmap.detection.defaults import pref_key_for

    data = {pref_key_for(k): "junk" for k in DETECTION_DEFAULTS}
    data["font_scale"] = 1.25
    removed = reset_detection_defaults(data)
    assert data["font_scale"] == 1.25, "reset must not touch unrelated settings"
    assert removed, "nothing was removed"
    for k in DETECTION_DEFAULTS:
        assert pref_key_for(k) not in data, (
            f"'{k}' survived the reset and would keep shadowing its default"
        )


# ── 4. Migration must work through the REAL load path ────────────────────────

def _prefs_with_stored(tmp_path, stored):
    """Construct a Preferences object against a temporary preferences file.

    Exercises Preferences.load() rather than calling the migration helper with
    a hand-made dict. The first version of this migration was correct as a
    helper and broken as a caller: `_data` is seeded from DEFAULTS, which
    already carries the current version, so reading the version out of `_data`
    made every install look up to date. Helper-level tests passed and the
    upgrade silently did nothing.
    """
    import json

    import mep_cmap.preferences as P

    pf = tmp_path / "preferences.json"
    pf.write_text(json.dumps(stored), encoding="utf-8")
    old_file, old_dir = P.PREFS_FILE, P.PREFS_DIR
    P.PREFS_FILE, P.PREFS_DIR = pf, tmp_path
    try:
        return type(P.prefs)(), pf
    finally:
        P.PREFS_FILE, P.PREFS_DIR = old_file, old_dir


def test_stale_value_is_migrated_when_preferences_are_actually_loaded(tmp_path):
    from mep_cmap.detection.defaults import DETECTION_DEFAULTS

    p, _ = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 60.0})
    assert p.mep_offset_max_duration_ms == \
        DETECTION_DEFAULTS["mep_offset_max_duration_ms"]
    assert p.migration_notes


def test_a_deliberate_value_survives_a_real_load(tmp_path):
    p, _ = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 75.0})
    assert p.mep_offset_max_duration_ms == 75.0
    assert p.migration_notes == []


def test_migration_stamps_the_file_so_it_runs_once(tmp_path):
    import json

    from mep_cmap.detection.defaults import DETECTION_DEFAULTS_VERSION

    p, pf = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 60.0})
    assert p.migration_notes
    written = json.loads(pf.read_text(encoding="utf-8"))
    assert written["detection_defaults_version"] == DETECTION_DEFAULTS_VERSION

    import mep_cmap.preferences as P
    old_file, old_dir = P.PREFS_FILE, P.PREFS_DIR
    P.PREFS_FILE, P.PREFS_DIR = pf, tmp_path
    try:
        second = type(P.prefs)()
    finally:
        P.PREFS_FILE, P.PREFS_DIR = old_file, old_dir
    assert second.migration_notes == []


def test_an_unstamped_file_is_treated_as_version_one(tmp_path):
    """A file with no version key predates the stamp and must be migrated."""
    p, _ = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 60.0,
                                         "onset_method": "consensus"})
    assert p.migration_notes
    assert p.onset_method == "consensus"      # unrelated settings preserved


def test_a_missing_preferences_file_does_not_break_startup(tmp_path):
    p, _ = _prefs_with_stored(tmp_path, {})
    assert p.migration_notes == []
    assert p.mep_offset_max_duration_ms > 0


# ── The refinement must not report its own search boundary ───────────────────

def test_offset_does_not_track_the_envelope_window_width():
    """
    The offset refinement used to scan a fixed neighbourhood for the first
    sustained sub-threshold run. When the fine envelope was already quiet at
    the start of that neighbourhood -- the normal case, since a centred coarse
    window smears the crossing late -- it returned the neighbourhood's own
    lower bound. The reported offset was then exactly
    ``coarse_anchor - env_window_ms`` on every trial: a function of a smoothing
    setting rather than of the signal, and visibly parked at an arbitrary point
    on the trace.

    The observable signature is that changing the envelope window shifts the
    offset by the same amount. Asserted here because it is what an analyst
    would notice, and it does not depend on the internals staying put.
    """
    offsets = {}
    for win in (3.0, 5.0, 8.0):
        vals = []
        for s in range(12):
            v = detect_mep_offset(trial_with_tail(s, amp=1.0), FS,
                                  peak_frac=0.12, env_window_ms=win,
                                  max_duration_ms=150.0, **OFFSET_KW)
            if v is not None:
                vals.append(v)
        assert vals, f"nothing detected at a {win} ms window"
        offsets[win] = float(np.mean(vals))

    spread = max(offsets.values()) - min(offsets.values())
    window_spread = 8.0 - 3.0
    assert spread < window_spread * 0.5, (
        f"offset moved {spread:.1f} ms when the envelope window moved "
        f"{window_spread:.1f} ms; it is tracking the setting, not the signal "
        f"({offsets})"
    )


def test_offset_refinement_returns_none_rather_than_a_boundary():
    """
    Where the fine envelope is quiet all the way back to the response peak,
    the coarse crossing was not late and there is nothing to refine. Returning
    the search floor in that case invents a landmark; returning None keeps the
    coarse value, which at least came from a threshold crossing.
    """
    from mep_cmap.detection.offset_detection import _refine_offset_anchor

    # The threshold is derived from the pre-stimulus baseline, so "quiet" has
    # to mean quiet RELATIVE TO IT: a noisy baseline with a genuinely silent
    # tail. Uniform low-level noise is not quiet by its own standard — around
    # 12% of its samples exceed a mean + 1 SD threshold built from itself.
    rng = np.random.default_rng(3)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)
    sig = np.empty(n)
    sig[:stim] = rng.normal(0, 0.01, stim)          # noisy baseline
    sig[stim:] = rng.normal(0, 1e-6, n - stim)      # silent afterwards

    got = _refine_offset_anchor(
        sig, FS, anchor_idx=stim + int(60 * FS / 1000),
        floor_idx=stim + int(30 * FS / 1000),
        ceiling_idx=n - 1, stim_idx=stim,
        search_radius=int(5 * FS / 1000), refine_window_ms=1.0,
        refine_sd_mult=1.0, refine_sustain_ms=1.0, floor_level=0.0)
    assert got is None


def test_offset_still_lands_where_the_response_ends():
    """The fix must not cost accuracy on the ordinary case."""
    errs = []
    for s in range(20):
        v = detect_mep_offset(trial_with_tail(s, amp=1.0), FS, peak_frac=0.12,
                              max_duration_ms=150.0, **OFFSET_KW)
        if v is not None:
            errs.append(v - 40.0)
    assert len(errs) >= 16
    assert abs(float(np.mean(errs))) < 4.0


def test_the_baseline_threshold_alone_detects_once_the_cap_is_adequate():
    """
    Records why the peak-fraction floor is small rather than large.

    The floor was introduced to fix a case where offset detection failed on 80
    of 81 real trials. The real cause was a 60 ms duration cap: once that was
    raised, the baseline threshold alone detected every trial. What the floor
    does is SHORTEN the answer, and at 0.12 it discarded genuinely time-locked
    activity — on a real M-wave it cut the offset at 42.6 ms where the trace
    plainly settled near 70.

    If this ever fails, the floor is doing detection work again and its value
    needs revisiting rather than nudging.

    The cap here is 350 ms rather than the 150 ms the real recording needed.
    This fixture's tail decays with a 60 ms time constant and is still above
    the noise floor long after the real responses had settled; the robust
    baseline estimator also puts the threshold slightly lower than the mean did
    (the median of a right-skewed envelope is below its mean), which extends it
    further. The point being tested is that the FLOOR is not required for
    detection, not that any particular cap suffices for any signal.
    """
    found, n = _offset_rate(1.0, peak_frac=0.0, max_duration_ms=350.0,
                            search_end_ms=395)
    assert len(found) >= n - 1, (
        f"baseline threshold alone detected only {len(found)}/{n}; the "
        f"peak-fraction floor may be compensating for something else"
    )


def test_the_floor_is_off_by_default():
    """
    Measured against an independent settle reference on a real M-wave
    recording, the floor truncated the offset by 45 to 97 ms depending on its
    value, because it scales with response size and so cuts hardest on the
    largest responses. At zero the same reference gave errors of -5.9 and
    +2.3 ms. On a resting MEP recording 0.04 and 0.0 are identical, so removing
    it costs nothing there.
    """
    from mep_cmap.detection.defaults import DETECTION_DEFAULTS

    assert DETECTION_DEFAULTS["mep_offset_peak_frac"] == 0.0


def test_a_smaller_floor_gives_a_later_offset():
    """The floor's only effect is to shorten; that must stay monotone."""
    high, _ = _offset_rate(1.0, peak_frac=0.12, max_duration_ms=150.0)
    low, _ = _offset_rate(1.0, peak_frac=0.04, max_duration_ms=150.0)
    assert high and low
    assert float(np.mean(low)) >= float(np.mean(high))


def test_defaults_match_what_the_real_recording_needed():
    from mep_cmap.detection.defaults import DETECTION_DEFAULTS

    assert DETECTION_DEFAULTS["mep_offset_peak_frac"] == 0.0
    assert DETECTION_DEFAULTS["mep_offset_max_duration_ms"] == 150.0


def test_anchor_fallback_uses_the_latency_profile_not_the_file_wide_window():
    """
    When a stimulus type has too few detected onsets to anchor, the fallback
    used to be the file-wide amplitude window -- which is the very thing
    anchoring exists to replace. The condition already in trouble was therefore
    also the one whose amplitude got truncated, while its neighbours anchored
    correctly and measured fine.

    Measured on a real M-wave recording: the first phase of a 3.8 mV response
    fell outside the 10-50 ms file-wide window entirely, and peak-to-peak was
    read from a 2.1 mV shoulder instead, giving 6.84 mV where the true value
    was 8.5 mV.
    """
    from mep_cmap.pipeline import PipelineConfig, ptp_window_for_stim_type

    fs, sb = 5000.0, 100
    cfg = PipelineConfig(pre_ms=20, post_ms=400, ptp_start=10, ptp_end=50,
                         ptp_anchor=True, ptp_anchor_min_trials=4,
                         ptp_anchor_pre_ms=2.0, ptp_anchor_duration_ms=40.0,
                         latency_map={"C": (1.0, 12.0)})
    start, end, anchored = ptp_window_for_stim_type(
        "C", {0: 4.5}, fs, cfg, sb + 50, sb + 250, sb,
        log_callback=lambda *a: None)

    start_ms = (start - sb) * 1000.0 / fs
    assert start_ms < 10.0, "the fallback is still the file-wide window"
    assert start_ms >= 2.0, "the fallback must respect the artefact floor"
    assert anchored is None, "this is a fallback, not an anchored window"


def test_the_fallback_never_widens_past_the_file_wide_start():
    """A profile that begins later than the configured start must not move it."""
    from mep_cmap.pipeline import PipelineConfig, ptp_window_for_stim_type

    fs, sb = 5000.0, 100
    cfg = PipelineConfig(pre_ms=20, post_ms=400, ptp_start=10, ptp_end=50,
                         ptp_anchor=True, ptp_anchor_min_trials=4,
                         ptp_anchor_pre_ms=2.0, ptp_anchor_duration_ms=40.0,
                         latency_map={"B": (20.0, 40.0)})
    start, _, _ = ptp_window_for_stim_type(
        "B", {0: 25.0}, fs, cfg, sb + 50, sb + 250, sb,
        log_callback=lambda *a: None)
    assert start == sb + 50, "a later profile must not push the window out"


def test_the_fallback_is_unaffected_when_anchoring_is_off():
    from mep_cmap.pipeline import PipelineConfig, ptp_window_for_stim_type

    fs, sb = 5000.0, 100
    cfg = PipelineConfig(pre_ms=20, post_ms=400, ptp_start=10, ptp_end=50,
                         ptp_anchor=False, latency_map={"C": (1.0, 12.0)})
    start, end, _ = ptp_window_for_stim_type(
        "C", {}, fs, cfg, sb + 50, sb + 250, sb, log_callback=lambda *a: None)
    assert (start, end) == (sb + 50, sb + 250)


# ── The baseline must survive an artefact that lands before the marker ───────

def test_envelope_baseline_is_robust_to_a_contaminated_tail():
    """
    The RMS envelope uses a CENTRED window, so a stimulus artefact smears
    backwards by half that window. The caller's guard covers an artefact at
    t=0; one landing even slightly earlier reaches past it, and a handful of
    contaminated samples then dominate the mean and SD.

    Measured on a real recording whose markers for one condition were ~2 ms
    late: the threshold went from 0.027 mV on a neighbouring condition to
    1.99 mV -- 73x -- and the offset was reported while the response was still
    at 26% of its peak, i.e. part-way down a deflection. The median and MAD are
    unmoved by a short contaminated tail.
    """
    from mep_cmap.detection.envelope_stats import compute_envelope_baseline

    rng = np.random.default_rng(0)
    quiet = np.abs(rng.normal(0.01, 0.002, 100))
    contaminated = quiet.copy()
    contaminated[-12:] = 2.4                      # smeared artefact

    clean = compute_envelope_baseline(quiet, 2.5, robust=True)
    dirty_mean = compute_envelope_baseline(contaminated, 2.5, robust=False)
    dirty_robust = compute_envelope_baseline(contaminated, 2.5, robust=True)

    # The mean-based estimate is wrecked; the robust one is barely moved.
    assert dirty_mean.threshold > 10 * clean.threshold
    assert dirty_robust.threshold < 2 * clean.threshold


def test_the_offset_detector_uses_the_robust_baseline():
    import inspect

    from mep_cmap.detection import offset_detection as od

    src = inspect.getsource(od)
    calls = [ln for ln in src.splitlines()
             if "compute_envelope_baseline(" in ln and not ln.strip().startswith("#")]
    assert calls, "no baseline call found"
    joined = "\n".join(calls) + src
    for ln in calls:
        # every call site, including its continuation line, must ask for robust
        idx = src.index(ln)
        window = src[idx:idx + 200]
        assert "robust=True" in window, f"not robust: {ln.strip()}"


def test_offset_lands_at_a_settled_point_not_mid_excursion():
    """
    The acceptance test that matters: |signal| at the reported offset, as a
    fraction of that trial's peak. Near zero means the response has settled;
    a large value means the marker is part-way down a deflection.

    On the real recording that motivated this, the failing condition measured
    0.263 before the fix and 0.002 after.
    """
    fracs = []
    for s in range(16):
        sig = trial_with_tail(s, amp=1.0)
        v = detect_mep_offset(sig, FS, peak_frac=0.0, max_duration_ms=200.0,
                              **OFFSET_KW)
        if v is None:
            continue
        stim = int(PRE_MS * FS / 1000)
        j = stim + int(v * FS / 1000)
        pk = float(np.max(np.abs(sig[stim:stim + int(60 * FS / 1000)])))
        if pk > 0:
            fracs.append(abs(sig[j]) / pk)
    assert fracs
    assert float(np.median(fracs)) < 0.10, (
        f"offset is landing mid-excursion (median |v|/peak = "
        f"{np.median(fracs):.3f})"
    )


# ── The return threshold must know where the signal actually settles ─────────

def test_the_settled_level_is_used_when_it_sits_above_the_pre_stimulus_one():
    """
    The return threshold was derived from the pre-stimulus baseline alone,
    which assumes the signal comes back to where it started. On real
    recordings it often does not: measured across every condition of one
    session, the envelope floor late in the epoch sat 1.3 to 2.0 times the
    pre-stimulus floor.

    "Return to the pre-stimulus level" is then a target the signal never
    reaches, and the offset lands wherever the envelope happens to dip -- 50 to
    90 ms after the trace has visibly flattened, dragging the area-under-curve
    window with it.
    """
    rng = np.random.default_rng(0)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)

    sig = np.empty(n)
    sig[:stim] = rng.normal(0, 0.004, stim)        # quiet before
    sig[stim:] = rng.normal(0, 0.020, n - stim)    # noisier after: 5x
    i0 = stim + int(22 * FS / 1000)
    i1 = stim + int(40 * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    sig[i0:i1] += np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5

    strict = detect_mep_offset(sig, FS, onset_ms=22.0, pre_ms=PRE_MS,
                               search_end_ms=300, max_duration_ms=150,
                               peak_frac=0.0, settle_frac=0.0, n_boot=200)
    adaptive = detect_mep_offset(sig, FS, onset_ms=22.0, pre_ms=PRE_MS,
                                 search_end_ms=300, max_duration_ms=150,
                                 peak_frac=0.0, settle_frac=1.0, n_boot=200)
    assert adaptive is not None, (
        "with a floor set where the signal settles, an offset must be found"
    )
    if strict is not None:
        assert adaptive <= strict + 1.0


def test_the_settled_floor_is_ignored_when_the_signal_returns_to_baseline():
    """
    Where the two floors agree -- an ordinary recording that really does come
    back to where it started -- the change must do nothing. Otherwise it would
    move every offset on data that never had the problem.

    This builds its own trial rather than using trial_with_tail: that fixture
    decays with a 60 ms time constant, so its "settled" region still contains
    signal and the two floors legitimately differ. Testing the no-op case needs
    a trial that is genuinely quiet by the end.
    """
    rng = np.random.default_rng(4)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sig = rng.normal(0, 0.008, n)          # same noise throughout
        i0 = stim + int(22 * FS / 1000)
        i1 = stim + int(40 * FS / 1000)
        k = np.arange(i1 - i0) / float(i1 - i0)
        sig[i0:i1] += np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5

        a = detect_mep_offset(sig, FS, onset_ms=22.0, pre_ms=PRE_MS,
                              search_end_ms=300, max_duration_ms=200,
                              peak_frac=0.0, settle_frac=0.0, n_boot=200)
        b = detect_mep_offset(sig, FS, onset_ms=22.0, pre_ms=PRE_MS,
                              search_end_ms=300, max_duration_ms=200,
                              peak_frac=0.0, settle_frac=1.0, n_boot=200)
        assert (a is None) == (b is None)
        if a is not None:
            assert abs(a - b) < 5.0, (
                f"seed {seed}: the settled floor moved a trial that returns to "
                f"baseline from {a:.1f} to {b:.1f} ms"
            )


def test_an_optional_duration_cap_does_not_crash_the_settled_estimate():
    """max_duration_ms is optional; the tail window has to cope without it."""
    sig = trial_with_tail(0, amp=1.0)
    kw = dict(OFFSET_KW)
    detect_mep_offset(sig, FS, peak_frac=0.0, settle_frac=1.0,
                      max_duration_ms=None, **kw)
