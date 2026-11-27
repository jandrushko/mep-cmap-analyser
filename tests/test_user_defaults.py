"""
The analyst's own defaults (v1.4.12).

Settings ▸ Save current settings as my defaults stores the general settings
in the existing preferences file; they are applied at start-up and to every
recording opened without a saved session, never to one that has a session.
"""

import json
import pathlib
import types

import pytest

from mep_cmap import preferences as P
from mep_cmap.app import TMSAnalysisApp

PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")


@pytest.fixture
def prefs_file(tmp_path, monkeypatch):
    """Point preferences at a temporary file; never the analyst's own."""
    monkeypatch.setattr(P, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(P, "PREFS_FILE", tmp_path / "preferences.json")
    fresh = P.Preferences()
    monkeypatch.setattr(P, "prefs", fresh)
    return tmp_path / "preferences.json"


class Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


def _stub(**vals):
    s = types.SimpleNamespace(**{k: Var(v) for k, v in vals.items()})
    s._USER_DEFAULT_VARS = TMSAnalysisApp._USER_DEFAULT_VARS
    s.log = lambda *_: None
    for name in ("_user_default_values", "_apply_user_defaults",
                 "_init_user_defaults"):
        setattr(s, name, getattr(TMSAnalysisApp, name).__get__(s))
    return s


# ── storage ──────────────────────────────────────────────────────────────────

def test_saved_defaults_survive_a_restart(prefs_file):
    P.prefs.set_setup_defaults({"ptp_start": 2, "highpass": 10})
    again = P.Preferences()                 # what the next start-up reads
    assert again.setup_defaults == {"ptp_start": 2, "highpass": 10}
    assert json.loads(prefs_file.read_text())["setup_defaults"]["ptp_start"] == 2


def test_saving_never_alters_the_factory_defaults(prefs_file):
    P.prefs.set_setup_defaults({"ptp_start": 2})
    assert P.DEFAULTS["setup_defaults"] == {}
    P.prefs.clear_setup_defaults()
    assert P.prefs.setup_defaults == {}


# ── applying ─────────────────────────────────────────────────────────────────

def test_applies_only_listed_general_settings(prefs_file):
    s = _stub(ptp_start=10, highpass=20, label_map=None)
    P.prefs.set_setup_defaults({"ptp_start": 2, "highpass": 10,
                                "label_map": {"A": "hacked"}})
    assert s._apply_user_defaults() == 2
    assert s.ptp_start.get() == 2 and s.highpass.get() == 10
    assert s.label_map.get() is None        # not a general setting


def test_startup_remembers_the_tools_values_before_applying(prefs_file):
    s = _stub(ptp_start=10, ptp_end=50)
    P.prefs.set_setup_defaults({"ptp_start": 2})
    s._init_user_defaults()
    assert s._factory_defaults["ptp_start"] == 10      # for Restore
    assert s.ptp_start.get() == 2


def test_nothing_saved_changes_nothing(prefs_file):
    s = _stub(ptp_start=10)
    assert s._apply_user_defaults() == 0 and s.ptp_start.get() == 10


def test_recording_specific_state_is_never_a_default():
    names = set(TMSAnalysisApp._USER_DEFAULT_VARS)
    for specific in ("label_map", "color_map", "window_map", "latency_map",
                     "reference_map", "crop_ranges", "gap_ms_map",
                     "delay_ms_map", "file_path", "marker_choice",
                     "study_metadata", "event_sources"):
        assert specific not in names
    for general in ("ptp_start", "ptp_end", "prestim_ms", "highpass",
                    "apply_humbug", "outlier_threshold", "onset_anchor"):
        assert general in names


# ── wiring ───────────────────────────────────────────────────────────────────

def test_applied_at_startup_right_after_the_controls_exist():
    i = APP.index("        self.setup_gui()")
    assert "self._init_user_defaults()" in APP[i:i + 300]


def test_applied_only_to_recordings_without_a_session():
    i = APP.index("os.path.isfile(_lp_chk(fe.derivatives_json))")
    block = APP[i:i + 1600]
    restore = block.index("self._apply_loaded_session(")
    defaults = block.index("self._apply_user_defaults(announce=True)")
    assert restore < block.index("else:") < defaults


def test_settings_menu_has_the_three_items():
    for label in ("Save current settings as my defaults",
                  "Apply my defaults", "original defaults"):
        assert label in APP


def test_onset_and_epoch_saved_into_their_existing_home():
    i = APP.index("def save_settings_as_defaults")
    body = APP[i:i + 2500]
    assert "prefs.set_onset_prefs(" in body
    assert "prefs.set_default_epoch_ms(" in body


# ── the analysis record says where the settings started ──────────────────────

def test_applying_defaults_records_the_origin(prefs_file):
    s = _stub(ptp_start=10)
    P.prefs.set_setup_defaults({"ptp_start": 2})
    s._apply_user_defaults()
    assert s._settings_origin["source"] == "user defaults"
    assert s._settings_origin["values"] == {"ptp_start": 2}


def test_origin_travels_from_snapshot_to_the_sidecar():
    assert "settings_provenance = copy.deepcopy(" in APP
    assert 'settings_provenance  = params.get("settings_provenance")' in APP
    assert '{"source": "recording session"}' in APP
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert '"settings_origin": settings_provenance' in pipe
    assert "settings_provenance=None" in pipe


# ── a fixed PTP window that starts too late is reported ──────────────────────

def _cfg(**kw):
    base = dict(ptp_anchor=False, ptp_start=10,
                latency_map={"A": (2, 12), "B": (18, 28)})
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_warns_for_a_window_starting_after_an_m_wave_could_begin():
    from mep_cmap.pipeline import ptp_window_warnings
    (msg,) = ptp_window_warnings(_cfg())
    assert "A (2 ms)" in msg and "B (" not in msg
    assert "anchoring" in msg


def test_no_warning_when_anchored_or_in_time():
    from mep_cmap.pipeline import ptp_window_warnings
    assert ptp_window_warnings(_cfg(ptp_anchor=True)) == []
    assert ptp_window_warnings(_cfg(ptp_start=2)) == []
    assert ptp_window_warnings(_cfg(latency_map={})) == []


def test_warning_is_logged_once_before_the_files():
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    i = pipe.index("for _w in ptp_window_warnings(cfg):")
    assert i < pipe.index("for file_i, raw_file in enumerate(txt_files):")


# ── hum removal: one definition, off ─────────────────────────────────────────

def test_hum_removal_is_defined_once_and_off():
    defs = [l.strip() for l in APP.splitlines()
            if l.strip().startswith("self.apply_humbug") and "BooleanVar(" in l]
    assert defs == ["self.apply_humbug = tk.BooleanVar(value=False)"]
