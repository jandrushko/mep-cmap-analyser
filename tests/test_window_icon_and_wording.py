"""
Window icon and the two "anchor" settings' wording (v1.4.13).
"""

import os
import pathlib
import sys

from mep_cmap.assets import asset_path, set_window_icon

PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"
ROOT = PKG.parent
APP = (PKG / "app.py").read_text(encoding="utf-8")


def test_the_icon_is_in_the_bundled_assets():
    # mep_cmap/assets is packaged by every platform's spec; the root MEP.ico
    # only stamps the .exe and is not available to the running app.
    assert os.path.isfile(asset_path("MEP.ico"))
    for spec in ("MEP_CMAP_Windows.spec", "MEP_CMAP_Linux.spec",
                 "MEP_CMAP_Mac.spec"):
        assert "('mep_cmap/assets', 'mep_cmap/assets')" in \
            (ROOT / spec).read_text(encoding="utf-8")


def test_the_main_window_sets_it():
    i = APP.index('self.root.title(f"MEP-CMAP Analyser, Version {TOOL_VERSION}")')
    assert "set_window_icon(self.root)" in APP[i:i + 300]


class _Win:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def iconbitmap(self, **kw):
        if self.fail:
            raise RuntimeError("no icon support")
        self.calls.append(("iconbitmap", kw))

    def iconphoto(self, *a):
        self.calls.append(("iconphoto", a))


def test_windows_sets_it_for_every_later_dialog_too(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    w = _Win()
    assert set_window_icon(w) is True
    (name, kw), = w.calls
    assert name == "iconbitmap" and kw["default"].endswith("MEP.ico")


def test_never_stops_the_app_opening(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert set_window_icon(_Win(fail=True)) is False


def test_nothing_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    w = _Win()
    assert set_window_icon(w) is False and not w.calls


def test_the_two_anchor_settings_are_named_apart():
    assert "Anchoring is ON —" not in APP and "Anchoring is OFF —" not in APP
    assert "Amplitude window anchoring is ON (set in" in APP
    assert "Amplitude window anchoring is OFF (set in" in APP
    assert "Anchor the ONSET SEARCH window" not in APP
    assert "Narrow the onset search to around each event type's" in APP


# ── amplitude window anchoring: on for new installs only ─────────────────────

def _prefs_with(tmp_path, monkeypatch, stored):
    from mep_cmap import preferences as P
    monkeypatch.setattr(P, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(P, "PREFS_FILE", tmp_path / "preferences.json")
    if stored is not None:
        import json
        (tmp_path / "preferences.json").write_text(json.dumps(stored))
    return P.Preferences()


def test_a_new_install_anchors_the_amplitude_window(tmp_path, monkeypatch):
    assert _prefs_with(tmp_path, monkeypatch, None).ptp_anchor is True


def test_an_older_preferences_file_keeps_running_without_it(tmp_path, monkeypatch):
    # Written before the key existed: that analyst has been measuring with the
    # fixed window, and a re-run must not change their amplitudes silently.
    p = _prefs_with(tmp_path, monkeypatch, {"font_scale": 1.1})
    assert p.ptp_anchor is False


def test_an_explicit_choice_is_kept_either_way(tmp_path, monkeypatch):
    assert _prefs_with(tmp_path, monkeypatch, {"ptp_anchor": True}).ptp_anchor is True
    assert _prefs_with(tmp_path, monkeypatch, {"ptp_anchor": False}).ptp_anchor is False


# ── Preferences buttons ──────────────────────────────────────────────────────

def test_preferences_close_saves_then_closes():
    src = (PKG / "preferences.py").read_text(encoding="utf-8")
    body = src[src.index("def _close():"):]
    body = body[:body.index("\n\n")]
    assert body.index("_apply()") < body.index("win.destroy()")
    assert 'text="Close"' in src and 'text="Cancel"' in src


def test_reset_font_shows_only_on_the_font_tab():
    src = (PKG / "preferences.py").read_text(encoding="utf-8")
    assert 'notebook.bind("<<NotebookTabChanged>>", _show_reset_for_tab' in src
    fn = src[src.index("def _show_reset_for_tab"):]
    assert "str(font_tab)" in fn[:600] and "_reset_btn.pack_forget()" in fn[:900]


# ── onset search narrowing: off by default (on in 1.4.13 only) ──────────────

def test_onset_search_narrowing_starts_off():
    # Off again from v1.4.14: on reviewed recordings, +/-8 ms narrowing put
    # onsets on the latency profile's lower bound and lost a trial.
    assert "self.onset_anchor             = tk.BooleanVar(value=False)" in APP


def test_an_old_session_without_it_still_loads_as_off():
    # Sessions store the value; one written before the setting existed must
    # load as it ran -- off -- or a re-run would change its onsets.
    assert '_b("onset_anchor",False)' in APP
    assert '"onset_anchor":          self.onset_anchor.get(),' in APP


def test_the_record_states_the_onset_search_narrowing():
    import types
    from mep_cmap.reporting import analysis_facts
    on = analysis_facts(types.SimpleNamespace(
        onset_anchor=True, onset_anchor_halfwidth_ms=8.0))["detection"]
    off = analysis_facts(types.SimpleNamespace(
        onset_anchor=False, onset_anchor_halfwidth_ms=8.0))["detection"]
    assert on["onset_search_halfwidth_ms"] == 8.0
    assert off["onset_search_halfwidth_ms"] is None
