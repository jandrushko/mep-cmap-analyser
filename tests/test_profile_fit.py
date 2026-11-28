"""
A latency profile that does not fit the data is said, not silently worked
around (v1.4.16).

Single-pulse FDI MEPs starting at ~30 ms against the 18-28 ms FDI profile:
between Min and Max lat 0.05 mV, after Max lat 0.76 mV. The Inspector told
the analyst to lower the amplitude minimum; the run said nothing. On real
recordings the check fired for exactly those (19/20, 16/20, 53/95 trials)
and stayed quiet for an M-wave recruitment curve and for normal lower-limb
MEPs, whose onsets lie inside their profile and peaks after it.
"""

import pathlib

import numpy as np

from mep_cmap.pipeline import (LAT_COLS, profile_fit_warning,
                               low_agreement_warnings)

FS, PRE = 5000.0, 20.0
PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"
RNG = np.random.default_rng(7)


def _trial(onset_ms=None, amp=0.8, artefact_dip=0.0):
    t = (np.arange(int(0.1 * FS)) - int(PRE * FS / 1000)) / FS * 1000.0
    x = RNG.normal(0, 0.005, t.size)
    if artefact_dip:
        x += np.where((t > 1) & (t < 12), -artefact_dip * np.exp(-(t - 1) / 3.0), 0)
    if onset_ms is not None:
        u = np.clip(t - onset_ms, 0, None)
        x += np.where(t >= onset_ms, amp * np.sin(2 * np.pi * u / 12) * np.exp(-u / 8), 0)
    return x


def _row(lat, n=None):
    r = [None] * len(LAT_COLS)
    r[LAT_COLS.index("Latency(ms)")] = lat
    r[LAT_COLS.index("Onset_Methods_N")] = n
    r[LAT_COLS.index("StimType")] = "B"
    r[LAT_COLS.index("Condition")] = ""
    return r


def _check(segs, rows, lo=18.0, hi=28.0):
    return profile_fit_warning("B", np.array(segs), rows, FS, PRE, lo, hi, 50.0)


def test_responses_after_the_profile_with_no_onset_are_reported():
    segs = [_trial(onset_ms=31.0) for _ in range(10)]
    msg = _check(segs, [_row("Not Detected") for _ in segs])
    assert msg and "10 of 10" in msg and "18-28 ms" in msg and "tab 1a" in msg


def test_minority_onsets_on_the_noise_still_count():
    segs = [_trial(onset_ms=31.0) for _ in range(10)]
    assert _check(segs, [_row(19.4, n=2) for _ in segs])


def test_normal_mep_with_a_majority_onset_is_not_reported():
    # Onset inside the profile, peaks after it: normal (lower-limb MEPs).
    segs = [_trial(onset_ms=21.0, amp=1.0) for _ in range(10)]
    assert _check(segs, [_row(21.0, n=5) for _ in segs]) is None


def test_recruitment_curve_artefact_is_not_reported():
    # M-wave profile 3-12 ms; sub-threshold trials: an artefact dip INSIDE
    # the window and nothing after it.
    segs = [_trial(artefact_dip=0.15) for _ in range(10)]
    assert _check(segs, [_row("Not Detected") for _ in segs], 3.0, 12.0) is None


def test_a_few_odd_trials_do_not_trigger_it():
    segs = [_trial(onset_ms=31.0) for _ in range(2)] + [_trial(onset_ms=21.0) for _ in range(18)]
    rows = [_row("Not Detected")] * 2 + [_row(21.0, n=5)] * 18
    assert _check(segs, rows) is None


def test_low_agreement_ignores_trials_with_no_onset():
    # The text "Not Detected" is not an onset found by a minority.
    assert low_agreement_warnings([_row("Not Detected", n=0)] * 5) == []
    (msg,) = low_agreement_warnings([_row(19.4, n=2), _row("Not Detected", n=0)])
    assert "1 trial(s)" in msg and "#0" in msg


def test_wired_into_inspector_run_preview_and_overlay():
    insp = (PKG / "inspector.py").read_text(encoding="utf-8")
    assert "lies after \"\n" in insp or "lies after" in insp
    assert "check this stimulus type's muscle group, Max" in insp
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert "_fit = profile_fit_warning(" in pipe
    assert "latency-profile check skipped for" in pipe
    prev = (PKG / "preview.py").read_text(encoding="utf-8")
    assert "_msg = profile_fit_warning(" in prev and '"profile_note"' in prev
    ov = (PKG / "overlay.py").read_text(encoding="utf-8")
    assert 'g.get("profile_note")' in ov
