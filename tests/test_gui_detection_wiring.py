"""
Guard: every detection parameter must be reachable from the GUI and reach the
pipeline.

The Detection preferences tab used to list its methods and parameters as
literals. A detector could therefore be fully implemented, registered,
dispatched and tested, and still be unusable -- which is exactly what happened:
``rms_envelope``, ``cusum`` and ``consensus`` ran correctly everywhere except
that no radio button offered them, so the only way to select one was to hand-
edit ``preferences.json``.

The same shape of failure applies to parameters. A value can exist in
``detection.defaults``, have a preferences property, and still never reach
``PipelineConfig`` because the GUI snapshot did not copy it -- and nothing
errors, because the config falls back to the canonical default and produces a
perfectly plausible number.

These tests analyse the source statically rather than driving Tk, so they run
headless. They check wiring, not appearance: that the radio list is generated
from the registry, and that every canonical key is carried from preferences
into the params dict and out again into the config.
"""

import ast
import pathlib

import pytest

from mep_cmap.detection import ONSET_METHOD_LABELS
from mep_cmap.detection.defaults import DETECTION_DEFAULTS, pref_key_for

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFS_SRC = (ROOT / "mep_cmap" / "preferences.py").read_text(encoding="utf-8")
APP_SRC = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")


# ── The method list must be generated, not transcribed ────────────────────────

def test_detection_tab_builds_its_radios_from_the_registry():
    assert "ONSET_METHOD_LABELS" in PREFS_SRC, (
        "The Detection tab must build its method list from "
        "detection.ONSET_METHOD_LABELS. A hard-coded list is why three "
        "registered methods were unselectable."
    )
    assert "ONSET_METHOD_HINTS" in PREFS_SRC


def test_detection_tab_has_no_hardcoded_method_label_dict():
    """
    A literal mapping of method key to label is the pattern that went stale.
    The registry is the only place labels should be written down.
    """
    tree = ast.parse(PREFS_SRC)
    method_keys = set(ONSET_METHOD_LABELS)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        overlap = keys & method_keys
        # A dict mapping several method keys to string values is a label or
        # description table. Mapping them to lists (the frame table) is fine.
        if len(overlap) >= 3 and all(
                isinstance(v, ast.Constant) and isinstance(v.value, str)
                for v in node.values):
            pytest.fail(
                f"preferences.py line {node.lineno}: literal dict describing "
                f"methods {sorted(overlap)}. Use ONSET_METHOD_LABELS / "
                f"ONSET_METHOD_HINTS instead."
            )


@pytest.mark.parametrize("method", sorted(ONSET_METHOD_LABELS))
def test_every_registered_method_has_a_parameter_frame_entry(method):
    """
    Selecting a method whose key is absent from the frame table shows no
    parameters at all, which reads as "this method has none" rather than as a
    missing wiring.
    """
    assert f'"{method}"' in PREFS_SRC, (
        f"'{method}' is registered but never mentioned in the Detection tab."
    )


# ── Every canonical parameter must be exposed and plumbed ─────────────────────

@pytest.mark.parametrize("key", sorted(DETECTION_DEFAULTS))
def test_every_detection_key_has_a_preferences_property(key):
    from mep_cmap.preferences import prefs

    assert hasattr(prefs, pref_key_for(key)), (
        f"No preferences property for '{key}'. Without one the GUI cannot "
        f"read the stored value and will always show the default."
    )


@pytest.mark.parametrize("key", sorted(DETECTION_DEFAULTS))
def test_every_detection_key_is_saved_by_the_apply_handler(key):
    """
    A parameter with a widget but no entry in _apply looks editable and
    silently reverts on the next launch.
    """
    assert f'"{key}"' in PREFS_SRC, (
        f"'{key}' is never written by the Detection tab's Apply handler."
    )


def test_app_carries_detection_settings_without_transcribing_them():
    """
    The params snapshot and the PipelineConfig construction must pull detection
    keys by name from the canonical dict. Listing them by hand means a new
    parameter is silently replaced by its default until someone remembers to
    edit both sites.
    """
    assert "_detection_prefs_snapshot" in APP_SRC
    assert "_detection_config_kwargs" in APP_SRC
    # The helpers themselves live in detection/defaults.py; app.py must call
    # them rather than listing keys inline.
    assert "prefs_detection_snapshot" in APP_SRC
    assert "config_detection_kwargs" in APP_SRC


def test_app_forwards_detection_params_to_the_inspector():
    assert "detection_params" in APP_SRC, (
        "The inspector must receive the analysis's detection parameters, or "
        "re-detection during review uses different settings than the run."
    )


# ── End-to-end: preferences -> params -> PipelineConfig ───────────────────────

def test_snapshot_and_config_helpers_round_trip_every_key():
    """
    Exercises the two app.py helpers directly: everything the snapshot emits
    must be accepted by PipelineConfig under the same name.
    """
    from mep_cmap.detection import (TK_BACKED_DETECTION_KEYS,
                                    config_detection_kwargs,
                                    prefs_detection_snapshot)
    from mep_cmap.pipeline import PipelineConfig
    from mep_cmap.preferences import prefs

    snapshot = prefs_detection_snapshot(prefs)
    expected = set(DETECTION_DEFAULTS) - set(TK_BACKED_DETECTION_KEYS)
    assert set(snapshot) == expected, (
        "The snapshot must cover every non-Tk-backed detection key exactly."
    )

    kwargs = config_detection_kwargs(snapshot)
    cfg = PipelineConfig(**kwargs)
    for key, value in kwargs.items():
        assert getattr(cfg, key) == value


def test_run_pipeline_accepts_every_key_app_passes_it():
    """
    The real call path is app.py -> run_pipeline -> PipelineConfig, and
    run_pipeline has its own explicit keyword list. Testing snapshot ->
    PipelineConfig directly skips the middle link entirely: the first version
    of this plumbing spliced the detection keys into the run_pipeline call,
    every test passed, and the application died with "run_pipeline() got an
    unexpected keyword argument 'onset_env_window_ms'" on the first real file.
    """
    import inspect

    from mep_cmap.detection import (config_detection_kwargs,
                                    prefs_detection_snapshot)
    from mep_cmap.pipeline import run_pipeline
    from mep_cmap.preferences import prefs

    sig = inspect.signature(run_pipeline)
    accepts_var_kw = any(pm.kind is inspect.Parameter.VAR_KEYWORD
                         for pm in sig.parameters.values())

    payload = config_detection_kwargs(prefs_detection_snapshot(prefs))
    assert payload, "no detection keys are being forwarded at all"

    for key in payload:
        assert key in sig.parameters or accepts_var_kw or \
            "detection_params" in sig.parameters, (
                f"run_pipeline cannot accept '{key}'. Either name it in the "
                f"signature or route it through the detection_params mapping."
            )

    assert "detection_params" in sig.parameters, (
        "run_pipeline should take detection settings as one mapping rather "
        "than a keyword per parameter."
    )


def test_app_passes_detection_params_as_a_keyword_not_a_splat():
    """
    ``**_detection_config_kwargs(params)`` in the run_pipeline call is the
    specific mistake that shipped. It is valid Python and only fails at run
    time, against a real recording.
    """
    assert "**_detection_config_kwargs" not in APP_SRC, (
        "Detection settings are being splatted into a call. run_pipeline takes "
        "them as the detection_params mapping."
    )
    assert "detection_params" in APP_SRC


def test_run_pipeline_forwards_detection_params_into_the_config():
    src = (ROOT / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")
    assert "config_detection_kwargs(detection_params" in src, (
        "run_pipeline accepts detection_params but never applies it to "
        "PipelineConfig, so the settings would be silently discarded."
    )


def test_every_kwarg_app_passes_to_run_pipeline_is_accepted():
    """
    Validate the whole call, not just the detection keys.

    A keyword that run_pipeline does not accept is valid Python and raises only
    when an analysis is actually started, against a real recording -- so it
    survives the entire test suite and fails in front of the user. Checking the
    call statically catches any such mismatch, including ones introduced by
    future edits that have nothing to do with detection.
    """
    import ast
    import inspect

    from mep_cmap.pipeline import run_pipeline

    sig = inspect.signature(run_pipeline)
    accepts_var_kw = any(pm.kind is inspect.Parameter.VAR_KEYWORD
                         for pm in sig.parameters.values())

    calls = []
    for node in ast.walk(ast.parse(APP_SRC)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == "run_pipeline":
            calls.append(node)

    assert calls, "no run_pipeline call found in app.py"

    for call in calls:
        passed = [k.arg for k in call.keywords if k.arg]
        if not accepts_var_kw:
            unknown = [k for k in passed if k not in sig.parameters]
            assert not unknown, (
                f"app.py line {call.lineno} passes keyword(s) run_pipeline "
                f"does not accept: {unknown}"
            )
        required = [pm.name for pm in sig.parameters.values()
                    if pm.default is inspect.Parameter.empty
                    and pm.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    inspect.Parameter.KEYWORD_ONLY)]
        missing = [r for r in required
                   if r not in passed and len(call.args) == 0]
        assert not missing, (
            f"app.py line {call.lineno} omits required argument(s): {missing}"
        )


def test_no_undefined_names_in_the_shipped_package():
    """
    Catch names that do not exist in their scope.

    ``app.py`` cannot be imported without a working matplotlib Tk backend, so
    a large part of it is never executed by the suite. A reference to a name
    that is not in scope is valid Python, compiles cleanly, and raises only
    when that exact GUI path runs -- which is how ``_open_inspector_gui``
    shipped calling a `params` variable it never receives, crashing the
    inspector after a full analysis had completed.

    pyflakes resolves scopes statically and finds this in milliseconds. Only
    undefined names are treated as failures; unused imports and shadowing are
    style matters and are ignored.
    """
    pyflakes = pytest.importorskip(
        "pyflakes.api", reason="pyflakes not installed")
    from pyflakes import reporter as pyflakes_reporter

    import io

    pkg = ROOT / "mep_cmap"
    files = sorted(str(f) for f in pkg.rglob("*.py")
                   if "__pycache__" not in f.parts)
    out, err = io.StringIO(), io.StringIO()
    pyflakes.checkRecursive(
        files, pyflakes_reporter.Reporter(out, err))

    undefined = [ln for ln in out.getvalue().splitlines()
                 if "undefined name" in ln]
    assert not undefined, (
        "undefined name(s) found:\n  " + "\n  ".join(undefined))


def test_detection_tk_attribute_table_matches_the_canonical_key_set():
    """
    The Tk-variable table drives both the worker snapshot and the inspector.
    A key listed as Tk-backed but absent from the table would be excluded from
    the preferences snapshot AND never read from its variable, so it would
    silently fall back to the canonical default everywhere.
    """
    import ast

    from mep_cmap.detection import TK_BACKED_DETECTION_KEYS

    table = None
    for node in ast.walk(ast.parse(APP_SRC)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "_DETECTION_TK_ATTRS"
                        for t in node.targets)):
            table = node.value
    assert table is not None, "_DETECTION_TK_ATTRS not found in app.py"

    keys = {k.value for k in table.keys
            if isinstance(k, ast.Constant)}
    assert keys == set(TK_BACKED_DETECTION_KEYS), (
        f"_DETECTION_TK_ATTRS covers {sorted(keys)} but "
        f"TK_BACKED_DETECTION_KEYS is {sorted(TK_BACKED_DETECTION_KEYS)}"
    )


def test_inspector_gets_its_detection_params_from_the_gui_thread():
    """
    _open_inspector_gui runs on the Tk main loop and never receives the
    worker's params snapshot, so it must build the dict from its own state.
    """
    import ast

    tree = ast.parse(APP_SRC)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_open_inspector_gui":
            fn = node
    assert fn is not None

    body = ast.get_source_segment(APP_SRC, fn) or ""
    assert "_current_detection_params()" in body, (
        "the inspector must build detection params from Tk state"
    )

    # Resolve the name properly rather than matching text: `_det_params` and
    # `detection_params` both contain the substring and are perfectly valid.
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert "params" not in bound
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Name) and sub.id == "params" \
                and isinstance(sub.ctx, ast.Load):
            pytest.fail(
                f"app.py line {sub.lineno}: _open_inspector_gui reads "
                f"`params`, which is never in its scope -- it runs on the GUI "
                f"thread and the worker's snapshot does not reach it."
            )


def test_tk_backed_keys_are_all_real_detection_keys():
    from mep_cmap.detection import TK_BACKED_DETECTION_KEYS

    unknown = set(TK_BACKED_DETECTION_KEYS) - set(DETECTION_DEFAULTS)
    assert not unknown, (
        f"_TK_BACKED_DETECTION_KEYS names non-existent settings: "
        f"{sorted(unknown)}. Those would be excluded from the snapshot and "
        f"never reach the config."
    )


def test_a_changed_preference_reaches_pipeline_config(tmp_path, monkeypatch):
    """
    The whole point of the plumbing: change a stored preference, and the value
    the pipeline runs with changes too.
    """
    from mep_cmap.detection import (config_detection_kwargs,
                                    prefs_detection_snapshot)
    from mep_cmap.pipeline import PipelineConfig
    from mep_cmap.preferences import prefs

    original = prefs._data.get("onset_env_window_ms")
    try:
        prefs._data["onset_env_window_ms"] = 17.5
        cfg = PipelineConfig(
            **config_detection_kwargs(prefs_detection_snapshot(prefs)))
        assert cfg.onset_env_window_ms == 17.5
    finally:
        if original is None:
            prefs._data.pop("onset_env_window_ms", None)
        else:
            prefs._data["onset_env_window_ms"] = original


def test_set_detection_prefs_rejects_unknown_keys():
    from mep_cmap.preferences import prefs

    with pytest.raises(KeyError):
        prefs.set_detection_prefs(not_a_real_setting=1)


def test_the_amplitude_window_note_does_not_share_a_cell():
    """
    Tk's grid stacks widgets that share a cell rather than reflowing around
    them, so a misplaced row/column silently draws one widget on top of
    another. The explanatory note was first placed at (row 2, column 1), which
    is the Pre-stim for analysis entry, and the two rendered overlapping.

    Checks only the note against the entries it could collide with; the frame
    contains two deliberate shared cells (a label overlaying a separator, and a
    label and entry positioned with opposing sticky) that are not faults.
    """
    import re

    a = APP_SRC.index("# ── Time Window + MEP Onset Detection ─")
    b = APP_SRC.index("# ─── CSP Detection Settings ─", a)
    seg = APP_SRC[a:b]

    note = re.search(r"self\._ptp_note\.grid\(\s*row=(\d+),\s*column=(\d+)", seg)
    assert note, "the amplitude-window note is not gridded"
    note_row = int(note.group(1))

    entry_rows = set()
    for m in re.finditer(r"textvariable=self\.(ptp_start|ptp_end|prestim_ms|"
                         r"pre_time|post_time)[^)]*\)[^)]*\.?grid\(\s*row=(\d+)",
                         seg, re.S):
        entry_rows.add(int(m.group(2)))
    # Also catch the two-statement form (widget assigned, then .grid()).
    for m in re.finditer(r"self\._ptp_start_entry\.grid\(\s*row=(\d+)", seg):
        entry_rows.add(int(m.group(1)))

    assert note_row not in entry_rows, (
        f"the note is on row {note_row}, which also holds a settings entry; "
        f"grid will draw them on top of one another"
    )


def test_amplitude_start_field_is_never_disabled():
    """
    The field remains in use when PTP anchoring is on.

    ptp_window_for_stim_type falls back to the file-wide start for any event
    type with fewer than ptp_anchor_min_trials detected onsets, so disabling
    the field would remove control of a value still being applied -- and it
    would do so on the conditions with fewest trials, which are the ones most
    likely to need adjusting.
    """
    assert 'self._ptp_start_entry.config(state="disabled")' not in APP_SRC, (
        "the amplitude window start must stay editable; it is the fallback "
        "for event types that cannot be anchored"
    )
    assert 'self._ptp_start_entry.config(state="normal")' in APP_SRC


def test_anchored_start_is_not_described_as_a_1a_setting():
    """
    The anchored start is each event type's median DETECTED onset. The 1a
    latency profile bounds that search but does not determine the value, so
    labelling the field as a 1a setting would be wrong in a way that is hard
    for an analyst to catch.
    """
    a = APP_SRC.index("def _refresh_ptp_note")
    b = APP_SRC.index("self._refresh_ptp_note = _refresh_ptp_note")
    body = APP_SRC[a:b]
    assert "median onset" in body
    for wrong in ("(1a set)", "set in 1a", "from 1a"):
        assert wrong not in body


def test_the_note_refreshes_after_preferences_are_applied():
    """Anchoring is toggled in the Preferences dialog, not in 1c."""
    assert "_refresh_ptp_note()" in APP_SRC
    a = APP_SRC.index("def _on_prefs_apply")
    b = APP_SRC.index("open_preferences_dialog(self.root", a)
    assert "_refresh_ptp_note()" in APP_SRC[a:b], (
        "toggling anchoring in Preferences must update the 1c label and note"
    )


def test_every_method_has_a_short_label_for_figures():
    """
    Figures showed raw registry keys — `bigoni_walkback`, `rms_envelope`,
    `methods_median`. Those are identifiers, not names a reader should have to
    decode off an axis. The CSV keeps the key, which is what a script matches
    on; only figures use the short label.
    """
    from mep_cmap.detection import (ONSET_METHOD_LABELS,
                                    ONSET_METHOD_SHORT_LABELS)

    assert set(ONSET_METHOD_SHORT_LABELS) == set(ONSET_METHOD_LABELS)
    for key, short in ONSET_METHOD_SHORT_LABELS.items():
        assert short and short != key, f"'{key}' has no readable short label"
        assert len(short) <= 24, f"'{short}' is too long for an axis tick"


def test_figures_use_short_labels_but_tables_keep_the_key():
    import inspect

    from mep_cmap import onset_methods_report as r

    assert "_display(" in inspect.getsource(r.plot_onset_methods_on_trace)
    assert "_display(" in inspect.getsource(r.plot_method_agreement)
    assert "_display(" in inspect.getsource(r.plot_bland_altman)
    # The long-format table must still carry the machine key.
    assert '"Method": method' in inspect.getsource(r.collect_agreement_rows)


def test_display_falls_through_for_an_unknown_method():
    from mep_cmap.onset_methods_report import _display

    assert _display("some_future_method") == "some_future_method"


def test_the_two_anchor_controls_name_the_window_they_move():
    """
    There are two unrelated settings whose labels both said "anchor": one moves
    the ONSET SEARCH window (tab 1c) and one moves the AMPLITUDE measurement
    window (Preferences → Detection). Ticking the first and expecting the
    second is an easy mistake, and its symptom -- a peak-to-peak that ignores
    the largest peak because it falls before the amplitude window start -- does
    not look like a settings problem.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    app_src = (root / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    pref_src = (root / "mep_cmap" / "preferences.py").read_text(encoding="utf-8")

    # v1.4.13: the onset-search control no longer says "anchor" at all, and
    # the amplitude note names its own setting, so the two cannot be read as
    # one setting contradicting itself.
    assert "Narrow the onset search to around each event type's" in app_src
    assert "Anchor the ONSET SEARCH window" not in app_src
    assert "Amplitude window anchoring is ON (set in" in app_src
    assert "Amplitude (PTP) Window Anchoring" in pref_src
    # Neither may carry a bare "anchor" label that does not say which window.
    assert 'text="Anchor onset search window to sample-median onset"' not in app_src
    assert 'text="PTP Window Anchoring"' not in pref_src
