"""
Reading a recording's stim codes out of the file, whatever the format.

Scanning was offered for Spike2 only. Every other format fell back to the
shared stim marker label -- one code for a recording that might hold six --
with nothing saying the list was a fallback rather than the file's contents.
A LabChart .mat holding 'Start Task' and 'Trigger' showed a single 'A'.
"""

import pytest

from mep_cmap.bidsify_tab import scan_event_codes, _AUTO_EVENT_CHANNEL


class _FakeIO:
    """Stands in for mep_cmap.io. The rules under test are about dispatch and
    failure reporting, not about any one reader."""

    def __init__(self, fmt, times=None, raises=None):
        self._fmt, self._times, self._raises = fmt, times or {}, raises

    def detect_format(self, _path):
        return self._fmt

    def extract_stim_times(self, _path, _marker, **_kw):
        if self._raises is not None:
            raise self._raises
        return self._times


# ── the whole-file route ─────────────────────────────────────────────────────

def test_a_format_with_no_dedicated_reader_still_yields_its_codes():
    io = _FakeIO("labchart_mat",
                 {"Start Task": [1.0] * 6, "Trigger": [2.0] * 162})
    codes, _cfg, why = scan_event_codes("x.mat", io)
    assert why == ""
    assert codes == {_AUTO_EVENT_CHANNEL: {"Start Task": 6, "Trigger": 162}}


def test_the_counts_are_the_events_the_analysis_will_find():
    """They come from the same call the pipeline makes, so the table cannot
    offer a code the analysis would not see, nor hide one it would."""
    io = _FakeIO("signal_mat", {f"State {i}": [0.0] * i for i in range(1, 7)})
    codes, _cfg, _why = scan_event_codes("x.mat", io)
    assert codes[_AUTO_EVENT_CHANNEL] == {f"State {i}": i for i in range(1, 7)}


def test_codes_are_strings_even_when_the_reader_returns_something_else():
    io = _FakeIO("edf", {1: [0.0], 2: [0.0, 1.0]})
    codes, _cfg, _why = scan_event_codes("x.edf", io)
    assert codes[_AUTO_EVENT_CHANNEL] == {"1": 1, "2": 2}


def test_a_file_with_no_events_says_so_rather_than_showing_an_empty_table():
    codes, _cfg, why = scan_event_codes("x.csv", _FakeIO("generic_tsv", {}))
    assert codes == {}
    assert "No stim events" in why and "generic_tsv" in why


# ── failures are reported, not raised ────────────────────────────────────────

def test_a_file_needing_configuration_is_told_what_to_do():
    """
    "Scan failed: [Errno 2] No generic-TSV config found" names a condition the
    analyst cannot act on. The fix is a wizard they have not been sent to.
    """
    io = _FakeIO("generic_tsv",
                 raises=FileNotFoundError("No generic-TSV config found for x"))
    codes, _cfg, why = scan_event_codes("x.csv", io)
    assert codes == {}
    assert "configured" in why and "First Level" in why


def test_an_unreadable_file_reports_the_reason_and_the_format_assumed():
    io = _FakeIO("unsupported_binary", raises=OSError("Cannot open x"))
    codes, _cfg, why = scan_event_codes("x.mat", io)
    assert codes == {}
    assert "unsupported_binary" in why and "Cannot open" in why


def test_a_scan_never_raises_at_the_caller():
    """The dialogue shows the reason; an exception would close it instead."""
    for exc in (OSError("nope"), ValueError("bad"), KeyError("k")):
        codes, _cfg, why = scan_event_codes("x", _FakeIO("edf", raises=exc))
        assert codes == {} and why


# ── the dedicated-reader route still wins ────────────────────────────────────

def test_spike2_uses_its_own_reader_not_the_whole_file_route(tmp_path, monkeypatch):
    """It reads headers only, and it can tell several marker channels apart --
    both of which the whole-file route loses."""
    p = tmp_path / "a.txt"
    p.write_text('"Marker"\n"Chan"\t"1"\n"DigMark"\n0.1\t"A???"\n0.2\t"B???"\n',
                 encoding="latin-1")

    called = {"whole_file": False}

    class _IO(_FakeIO):
        def extract_stim_times(self, *_a, **_k):
            called["whole_file"] = True
            return {"WRONG": [0.0]}

    codes, _cfg, why = scan_event_codes(str(p), _IO("spike2"))
    assert not called["whole_file"], "the whole-file route displaced the reader"
    assert codes == {"DigMark": {"A": 1, "B": 1}}
    assert why == ""


def test_a_channel_with_no_codes_is_not_offered(tmp_path):
    """It cannot be assigned a parameter set, so offering it in the dropdown
    offers an empty table."""
    p = tmp_path / "a.txt"
    p.write_text('"Marker"\n"Chan"\t"1"\n"Empty"\n\n'
                 '"Marker"\n"Chan"\t"2"\n"Real"\n0.2\t"A???"\n',
                 encoding="latin-1")
    codes, _cfg, _why = scan_event_codes(str(p), _FakeIO("spike2"))
    assert list(codes) == ["Real"]


def test_the_pseudo_channel_is_distinguishable_from_a_real_one():
    """It is a label for the analyst, not a channel name any reader knows, and
    saving it would hand the next run a name that resolves to nothing."""
    assert _AUTO_EVENT_CHANNEL.startswith("(")
