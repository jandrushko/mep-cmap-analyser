"""
The Data Inspector must review a trial the way the pipeline analysed it.

These are regression tests for a real divergence. The Inspector computed the
cSP search window itself and capped ``search_end_ms`` at
``second_peak + csp_max_mep_offset_ms``. That setting means "the cSP must
START within this many ms of the 2nd MEP peak" (see its declaration in
app.py), so capping the window truncated the quantity being measured: at the
default of 100 ms no silent period longer than ~100 ms could be found during
review, while the pipeline, which never applied the cap, reported the true
duration for the same trial.

The fix was to give both callers one entry point. These tests hold that line:
the source-level ones fail if either caller starts building detector arguments
by hand again, and the behavioural ones fail if the cap is ever reapplied to
the window rather than to the onset.
"""

import inspect
import re

import numpy as np
import pytest

from mep_cmap import inspector as inspector_mod
from mep_cmap import pipeline as pipeline_mod
from mep_cmap import preview as preview_mod
from mep_cmap.detection import CspSettings, detect_csp_for_trial
from mep_cmap.pipeline import PipelineConfig

# Every module that detects a silent period. preview.py was missed when this
# file was first written, and it was the one actually broken: it built a
# PipelineConfig without a single numeric cSP setting, so it read the dataclass
# defaults and ignored the interface entirely. Changing Min return from 40 ms
# to 2 ms gave a byte-identical preview. Any new caller belongs in this list.
CSP_CALLERS = [inspector_mod, pipeline_mod, preview_mod]
CALLER_IDS = ["inspector", "pipeline", "preview"]


FS = 2000.0
PRE, POST = 200.0, 600.0
N = int((PRE + POST) * FS / 1000)
T = np.linspace(-PRE, POST, N, endpoint=False)


def _trial(csp_on=40.0, csp_off=260.0, seed=1):
    """Background EMG with an MEP and a silent period of known duration."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.2, N)
    x[(T >= csp_on) & (T < csp_off)] *= 0.02
    mep = (T >= 25) & (T < 38)
    x[mep] += 3.0 * np.sin(np.linspace(0, 2 * np.pi, int(mep.sum())))
    return x


# ── Source-level: neither caller may hand-roll the arguments ────────────────

@pytest.mark.parametrize("module", CSP_CALLERS, ids=CALLER_IDS)
def test_callers_use_the_shared_entry_point(module):
    src = inspect.getsource(module)
    assert "detect_csp_for_trial" in src, (
        f"{module.__name__} no longer routes cSP detection through "
        f"detect_csp_for_trial")
    assert not re.search(r"\bdetect_csp_bootstrap\s*\(", src), (
        f"{module.__name__} calls detect_csp_bootstrap directly. Both callers "
        f"must go through detect_csp_for_trial so review and analysis cannot "
        f"drift apart.")


@pytest.mark.parametrize("module", CSP_CALLERS, ids=CALLER_IDS)
def test_the_max_offset_setting_is_never_applied_to_the_search_window(module):
    """
    ``csp_max_mep_offset_ms`` bounds the cSP ONSET. A caller combining it with
    a search end is reintroducing the truncation bug.
    """
    for line in inspect.getsource(module).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "csp_max_mep_offset_ms" in stripped:
            assert not re.search(r"search_end|min\s*\(", stripped), (
                f"{module.__name__} looks like it is capping the search "
                f"window with csp_max_mep_offset_ms:\n    {stripped}")


def test_every_csp_setting_reaches_the_detector():
    """
    A field on CspSettings that detect_csp_for_trial forgets to forward is a
    setting the interface offers and the detector never sees.
    """
    forwarded = inspect.getsource(detect_csp_for_trial)
    for name in CspSettings._FIELDS:
        assert f"settings.{name}" in forwarded, (
            f"CspSettings.{name} is never passed to the detector")


def test_the_preview_passes_every_numeric_csp_setting_into_its_config():
    """
    The preview builds its own PipelineConfig. It passed csp_types and nothing
    else, so every numeric cSP setting fell back to the dataclass default and
    the interface had no effect on the preview at all -- silently, because a
    default is a valid value and nothing errored.

    Asserted against the source of the config build rather than by running it,
    since that path needs a Tk app and a loaded recording.
    """
    src = inspect.getsource(preview_mod)
    start = src.index("_cfg = PipelineConfig(")
    depth, end = 0, start
    for i, ch in enumerate(src[start:], start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    build = src[start:end]

    for name in CspSettings._FIELDS:
        if name in ("seed", "min_threshold_frac", "return_duty"):
            continue
        assert f"csp_{name}=" in build, (
            f"preview.py builds a PipelineConfig without csp_{name}, so the "
            f"preview will silently use the default instead of the value set "
            f"in the interface")


def test_a_config_default_is_not_mistaken_for_a_configured_value():
    """
    The specific symptom: a configured Min return must not behave like the
    default. Guards the detector end of the wiring the preview test guards the
    top of.

    Both values are at or above the 10 ms RMS window, because anything shorter
    is raised to it (see test_a_return_shorter_than_the_envelope_is_refused).
    """
    x = _trial(csp_on=40.0, csp_off=260.0, seed=3)
    # A 30 ms burst part way through: long enough to satisfy a 20 ms return
    # criterion and end the silence there, far too short to satisfy 150 ms.
    rng = np.random.default_rng(11)
    burst = (T >= 150) & (T < 180)
    x[burst] = rng.normal(0.0, 0.2, int(burst.sum()))

    short_return = detect_csp_for_trial(
        x, FS, T, CspSettings(min_return_ms=20.0),
        second_peak_ms=38.0, pre_ms=PRE)
    long_return = detect_csp_for_trial(
        x, FS, T, CspSettings(min_return_ms=150.0),
        second_peak_ms=38.0, pre_ms=PRE)

    assert long_return is not None and short_return is not None
    assert long_return != short_return, (
        "min_return_ms of 20 ms and 150 ms gave identical results on a trial "
        "with breakthrough EMG; the setting is not reaching the detector")
    assert (long_return[1] - long_return[0]) > (short_return[1] - short_return[0])


def test_a_return_shorter_than_the_envelope_is_refused_and_said_so():
    """
    A moving-window RMS cannot rise and fall faster than its own window, so a
    return criterion shorter than that window is unenforceable and every value
    below it behaves alike.

    This is easy to hit at high sampling rates, where the two are set in ms and
    compared in samples: at 5 kHz a 10 ms window is 50 samples and a 2 ms
    criterion is 10. On a real 5 kHz recording, 2, 5 and 40 ms all produced an
    identical result and the setting looked broken. It is now raised to the
    window and the trial says so, rather than silently honoured or ignored.
    """
    fs = 5000.0
    n = int((PRE + POST) * fs / 1000)
    t = np.linspace(-PRE, POST, n, endpoint=False)
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 0.2, n)
    x[(t >= 40) & (t < 260)] *= 0.02
    mep = (t >= 25) & (t < 38)
    x[mep] += 3.0 * np.sin(np.linspace(0, 2 * np.pi, int(mep.sum())))

    results = {}
    for requested in (2.0, 5.0, 10.0):
        why = []
        results[requested] = detect_csp_for_trial(
            x, fs, t, CspSettings(min_return_ms=requested, rms_window_ms=10.0),
            second_peak_ms=38.0, pre_ms=PRE, reason_out=why)
        notes = " ".join(why)
        if requested < 10.0:
            assert "raised" in notes.lower(), (
                f"min_return_ms={requested} is shorter than the RMS window and "
                f"was clamped, but nothing said so: {notes!r}")
        else:
            assert "raised" not in notes.lower()

    assert len(set(results.values())) == 1, (
        "values below the RMS window should all behave as the window does")


def test_pipeline_config_carries_every_csp_setting():
    cfg_fields = set(PipelineConfig.__dataclass_fields__)
    for name in CspSettings._FIELDS:
        if name in ("seed", "min_threshold_frac", "return_duty"):
            continue          # not user-facing; detector defaults are canonical
        assert f"csp_{name}" in cfg_fields, (
            f"PipelineConfig has no csp_{name}, so the analysis cannot honour "
            f"it while the Inspector can")


def test_settings_are_read_off_a_pipeline_config():
    cfg = PipelineConfig(csp_min_silence_ms=33.0, csp_min_return_ms=55.0,
                         csp_max_mep_offset_ms=123.0, csp_n_boot=222)
    s = CspSettings.from_source(cfg)
    assert (s.min_silence_ms, s.min_return_ms) == (33.0, 55.0)
    assert (s.max_mep_offset_ms, s.n_boot) == (123.0, 222)


def test_settings_are_frozen():
    with pytest.raises(Exception):
        CspSettings().min_silence_ms = 999


# ── Behavioural: the cap must not shorten the measurement ───────────────────

def test_a_long_silent_period_is_not_truncated_by_the_max_offset_setting():
    """
    The silent period lasts 220 ms and starts ~40 ms in. With the cap read as
    a window bound, the reported duration collapsed to roughly the cap.
    """
    x = _trial(csp_on=40.0, csp_off=260.0)
    out = detect_csp_for_trial(x, FS, T, CspSettings(max_mep_offset_ms=100.0),
                               second_peak_ms=38.0, pre_ms=PRE)
    assert out is not None
    duration = (out[1] - out[0]) * 1000.0 / FS
    assert duration > 150.0, (
        f"cSP reported as {duration:.0f} ms; a 220 ms silent period has been "
        f"truncated, most likely by capping the search window")
    assert duration == pytest.approx(220.0, abs=25.0)


def test_the_max_offset_setting_still_rejects_a_late_suppression():
    """The cap must keep working as an onset bound, or it does nothing."""
    x = _trial(csp_on=200.0, csp_off=400.0)
    reason = []
    out = detect_csp_for_trial(x, FS, T, CspSettings(max_mep_offset_ms=40.0),
                               second_peak_ms=38.0, pre_ms=PRE,
                               reason_out=reason)
    assert out is None
    assert "later than" in reason[0]


def test_review_and_analysis_agree_on_the_same_trial():
    """
    The same settings reaching the detector by the two routes must give the
    same answer, since both now build CspSettings from the same field names.
    """
    cfg = PipelineConfig(csp_min_silence_ms=25.0, csp_min_return_ms=40.0,
                         csp_max_mep_offset_ms=100.0, csp_search_end_ms=500.0)
    from_cfg = CspSettings.from_source(cfg)
    from_gui = CspSettings.from_source({
        "csp_min_silence_ms": 25.0, "csp_min_return_ms": 40.0,
        "csp_max_mep_offset_ms": 100.0, "csp_search_end_ms": 500.0,
        "csp_criterion": 1.96, "csp_significance": 0.99,
        "csp_n_boot": 1000, "csp_rms_window_ms": 10.0,
    })
    assert from_cfg == from_gui

    x = _trial()
    a = detect_csp_for_trial(x, FS, T, from_cfg, second_peak_ms=38.0, pre_ms=PRE)
    b = detect_csp_for_trial(x, FS, T, from_gui, second_peak_ms=38.0, pre_ms=PRE)
    assert a == b is not None
