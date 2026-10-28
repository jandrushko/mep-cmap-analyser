"""
Regressions from Padraig's pilot data (Movement-Prep TMS, Spike2 DigMark).

Three faults surfaced together on one evening:

1. DigMark codes 32-126 decode to ASCII, so code 62 arrives as ">" and code 63
   as "?". Both went straight into output filenames and raised Errno 22 on
   Windows part way through a run.
2. The deprecated app.session_path_for alias did not accept prefix_override,
   so App.session_path() raised TypeError on every auto-save.
3. Applied conditions (a list of events with onsets) were held on the app and
   never tied to a recording, so after opening another file its EMG was cut
   at the previous file's stimulus times, and the next Apply merged the old
   events into the new file's events file.

Plus the reporting gaps that hid them: "All results saved" after a file had
failed, no warning for events past the end of a recording, and tab 1a listing
stimulus types with no trials in the selected range.
"""

import ast
import os
import pathlib
import types

import pytest

from mep_cmap.utils import filename_token
from mep_cmap import conditions as C

PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"


class _Var:
    """The one method of a tk.StringVar these paths use."""
    def __init__(self, v=""):
        self._v = v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


def _func_source(path, name):
    """Source of the function or method ``name`` in ``path``."""
    text = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} not found in {path.name}")


# ── 1. Stimulus types in filenames ───────────────────────────────────────────

ASCII_DIGMARK = [chr(c) for c in range(32, 127)]


def test_illegal_characters_are_encoded():
    assert filename_token(">") == "%3E"
    assert filename_token("?") == "%3F"
    assert filename_token("A/B") == "A%2FB"
    assert filename_token("%") == "%25"


def test_names_that_already_worked_are_unchanged():
    for label in ("A", "B", "10", "2", "=", "F", "A-pre", "A·post", "x3E"):
        assert filename_token(label) == label


def test_every_digmark_code_gets_a_distinct_name():
    # The old replace-with-underscore rule sent "=", ">" and "?" to "_", so
    # their figures overwrote one another.
    tokens = [filename_token(c) for c in ASCII_DIGMARK]
    assert len(set(tokens)) == len(tokens)


def test_every_digmark_code_can_be_written_as_a_file(tmp_path):
    # The real check, on whatever filesystem the suite runs on.
    for ch in ASCII_DIGMARK:
        p = tmp_path / f"sub-01_test000_{filename_token(ch)}.csv"
        p.write_text("x")
        assert p.parent == tmp_path and p.exists()


@pytest.mark.parametrize("mod", ["mep_cmap.onset_methods_report",
                                 "mep_cmap.detection.onset_methods_report"])
def test_onset_method_figure_names_use_the_shared_rule(mod):
    import importlib
    _safe = importlib.import_module(mod)._safe
    assert len({_safe("="), _safe(">"), _safe("?")}) == 3
    assert "/" not in _safe("A/B 120%")


_UNSAFE_PATH = (
    "{stim_type}.csv",
    "stim-{stim_type}_",
    "stim-{stim_type}\"",
)


@pytest.mark.parametrize("rel", [
    "pipeline.py",
    "add_ons/single_file/variability.py",
    "add_ons/single_file/mepfeatx_features.py",
    "add_ons/group_level/variability_group.py",
])
def test_no_raw_stim_type_reaches_a_filename(rel):
    src = (PKG / rel).read_text(encoding="utf-8")
    for pat in _UNSAFE_PATH:
        assert pat not in src, f"{rel} still writes {pat!r} into a path"


# ── 2. Auto-save: the session_path_for alias ─────────────────────────────────

def test_app_alias_forwards_prefix_override(tmp_path):
    from mep_cmap import app as A
    from mep_cmap.bids import session_path_for as real
    src = str(tmp_path / "rec.smr")
    assert A.session_path_for(src, None, "", prefix_override="sub-01_X") \
        == real(src, None, "", prefix_override="sub-01_X")
    assert A.session_path_for(src) == real(src)


def test_app_session_path_no_longer_raises(tmp_path):
    # The exact call that failed: App.session_path() passes prefix_override
    # to the module-level alias.
    from mep_cmap.app import TMSAnalysisApp
    stub = types.SimpleNamespace(
        file_path=_Var(str(tmp_path / "rec.smr")), study_metadata=None,
        derivatives_path=_Var(""), _bids_prefix_override=None)
    out = TMSAnalysisApp.session_path(stub)
    assert out and out.endswith("_session.json")


# ── 3. A failed file is not reported as a finished run ───────────────────────

def test_run_outcome_reports_failures():
    from mep_cmap.pipeline import _report_run_outcome
    log = []
    out = _report_run_outcome([("test000", "OSError: bad name")], 1, log.append)
    assert out == {"n_files": 1, "failed": [("test000", "OSError: bad name")],
                   "ok": False}
    assert "All results saved" not in log[-1]
    assert "test000" in log[-1]


def test_run_outcome_clean_run():
    from mep_cmap.pipeline import _report_run_outcome
    log = []
    out = _report_run_outcome([], 3, log.append)
    assert out["ok"] and log == ["✅ All results saved."]


def test_worker_routes_partial_failure_to_the_failed_outcome():
    # "failed" skips the auto-save and does not mark the file complete;
    # "done" does both, which is how a crashed file became "complete".
    src = (PKG / "app.py").read_text(encoding="utf-8")
    i = src.index("_run_failures.append")
    j = src.index('self.msg_q.put(("failed", _what))')
    k = src.index('self.msg_q.put(("done", None))')
    assert i < j < k


# ── 4/5. Conditions belong to one recording ──────────────────────────────────

def test_recording_key_matches_the_same_file_however_written(tmp_path):
    p = tmp_path / "Rec.smr"
    assert C.recording_key(str(p)) == C.recording_key(str(p).replace(os.sep, "/"))
    assert C.recording_key("") == ""


def test_rows_from_another_recording_are_refused(tmp_path):
    rows = [{"onset": 1.0, "duration": 0.0, "trial_type": ">", "condition": "n/a"}]
    a, b = str(tmp_path / "test000.smr"), str(tmp_path / "OCP18.smr")
    owner = C.recording_key(a)
    assert C.rows_for_recording(rows, owner, a) == rows
    assert C.rows_for_recording(rows, owner, b) == []
    # Unknown provenance passes through; the reset is what clears those.
    assert C.rows_for_recording(rows, "", b) == rows


def test_app_owned_rows(tmp_path):
    from mep_cmap.app import TMSAnalysisApp
    a, b = str(tmp_path / "test000.smr"), str(tmp_path / "sub-019.smr")
    rows = [{"onset": 5.0, "trial_type": ">", "condition": "n/a"}]
    stub = types.SimpleNamespace(file_path=_Var(b), condition_event_rows=rows,
                                 _cond_rows_path=C.recording_key(a))
    assert TMSAnalysisApp._owned_condition_rows(stub) == []
    stub.file_path.set(a)
    assert TMSAnalysisApp._owned_condition_rows(stub) == rows


def test_merge_does_not_pull_in_the_previous_recordings_events(tmp_path):
    # What wrote 152 foreign rows into OCP18's events file.
    from mep_cmap.conditions_tab import ConditionsTabMixin
    old, new = str(tmp_path / "test000.smr"), str(tmp_path / "OCP18.smr")
    stub = types.SimpleNamespace(
        file_path=_Var(new),
        condition_event_rows=[{"onset": 900.0, "duration": 0.0,
                               "trial_type": ">", "condition": "n/a"}],
        _cond_rows_path=C.recording_key(old),
        _configured_events=lambda p, channel_idx=None: ({"A": [20.0]}, []),
        _cond_selected_channel_idx=lambda: 0)
    out = ConditionsTabMixin._cond_event_rows_for_recording(stub, [])
    assert [r["onset"] for r in out] == [20.0]


def test_no_reader_bypasses_the_ownership_check():
    # The analysis snapshot and the session save both used to read the raw
    # list; each now goes through _owned_condition_rows().
    src = (PKG / "app.py").read_text(encoding="utf-8")
    assert "list(self.condition_event_rows or [])" not in src
    assert src.count("self._owned_condition_rows()") >= 3


def test_opening_a_file_clears_condition_state():
    body = _func_source(PKG / "app.py", "_reset_state_for_new_file")
    for attr in ("condition_event_rows = []", "condition_map = {}",
                 "_cond_rows_path = \"\"", "_cond_rows = []",
                 "_cond_epochs = {}", "_cond_confirmed_path = None"):
        assert f"self.{attr}" in body


def test_session_restore_records_the_owner():
    body = _func_source(PKG / "app.py", "_apply_loaded_session")
    assert "self._cond_rows_path" in body


def test_events_past_the_end_are_counted():
    from mep_cmap.pipeline import events_past_recording_end
    times = {"A": [20.0, 117.3], "G": [625.5, 2970.3]}
    assert events_past_recording_end(times, 534.1) == (2, 4)
    assert events_past_recording_end(times, 3000.0) == (0, 4)
    assert events_past_recording_end(times, 0.0) == (0, 4)


def test_events_past_the_end_are_warned_about():
    body = _func_source(PKG / "pipeline.py", "pipeline_load_file")
    assert "events_past_recording_end(" in body and "warn(" in body


# ── 6. Tab 1a lists only what the analysis will measure ──────────────────────

def test_labels_tab_drops_types_with_no_trials_in_range():
    from mep_cmap.conditions_tab import ConditionsTabMixin
    logs = []
    stub = types.SimpleNamespace(
        _snapshot_analysis_params=lambda: {},
        _analysis_channel_indices=lambda: [0],
        _cond_selected_channel_idx=lambda: 0,
        _cond_crop_for=lambda p, channel_idx=None: (None, 100.0, 500.0),
        log=logs.append)
    # test000 on FDI D: every in-range "2" excluded, "10" entirely outside.
    groups = {"10": [10.0, 20.0], "2": [30.0, 40.0], "=": [150.0, 160.0],
              ">": [170.0]}
    out = ConditionsTabMixin._cond_groups_in_range(stub, groups)
    assert sorted(out) == ["=", ">"]
    assert "10" in logs[-1] and "2" in logs[-1]


def test_labels_tab_unchanged_when_ranges_unknown():
    from mep_cmap.conditions_tab import ConditionsTabMixin

    def _boom():
        raise RuntimeError
    stub = types.SimpleNamespace(_snapshot_analysis_params=_boom, log=print)
    groups = {"A": [1.0]}
    assert ConditionsTabMixin._cond_groups_in_range(stub, groups) is groups


# ── 8. The pipeline imports what it uses ─────────────────────────────────────

def test_pipeline_draws_a_figure_with_nothing_else_imported():
    # In the app, app.py imports matplotlib.figure before any analysis runs,
    # which hid that pipeline.py never did. Called from a script, every file
    # failed at its first figure. Run in a fresh interpreter, because this
    # test process has long since loaded matplotlib.figure itself.
    import subprocess
    import sys
    code = ("import matplotlib; matplotlib.use('Agg'); "
            "import mep_cmap.pipeline as P; "
            "P.matplotlib.figure.Figure(figsize=(2, 2)); print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(PKG.parent), timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_no_module_reaches_a_submodule_it_did_not_import():
    # The same fault anywhere in the package: "import matplotlib" followed by
    # matplotlib.<submodule>.<name>, which works only if something else
    # happened to import the submodule first.
    pkgs = {"matplotlib": {"rcParams", "use", "colors", "cm"}, "scipy": set()}
    bad = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname is None:
                        parts = a.name.split(".")
                        imported.update(".".join(parts[:i])
                                        for i in range(1, len(parts) + 1))
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                parts = n.module.split(".")
                imported.update(".".join(parts[:i])
                                for i in range(1, len(parts) + 1))
                imported.update(f"{n.module}.{a.name}" for a in n.names)
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Attribute)
                    and isinstance(n.value.value, ast.Name)):
                continue
            top, sub = n.value.value.id, n.value.attr
            if top in pkgs and sub not in pkgs[top] and top in imported \
                    and f"{top}.{sub}" not in imported:
                bad.append(f"{path.relative_to(PKG)}:{n.lineno} {top}.{sub}")
    assert not bad, "submodules used without being imported: " + ", ".join(bad)
