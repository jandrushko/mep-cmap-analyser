"""
Reading stim codes out of a Spike-2 TEXT export.

Scanning was offered for .smr only, so a text export fell back to the shared
stim marker label and reported one code -- typically just 'A' -- for a
recording that carried several. Nothing said the list was a fallback rather
than the file's contents, so the missing codes were invisible.
"""

import os

import pytest

from mep_cmap.formats import spike2 as s2


_EXPORT = '''"SUMMARY"
"Chan"	"Type"	"Title"
1	"Waveform"	"EMG1"

"CHANNEL"	1
"Marker"
"Chan"	"31"
"DigMark"
0.1000	"A???"
1.2000	"B???"
2.3000	"A???"
3.4000	"C???"
4.5000	"B???"
5.6000	"B???"

"CHANNEL"	32
"Marker"
"Chan"	"32"
"Keyboard"
7.0000	"S???"
8.0000	"S???"
'''


def _write(tmp_path, text=_EXPORT):
    p = tmp_path / "sub-009_export.txt"
    p.write_text(text, encoding="latin-1")
    return str(p)


def test_every_marker_channel_is_found(tmp_path):
    assert s2.list_event_channels(_write(tmp_path)) == ["DigMark", "Keyboard"]


def test_every_code_in_a_channel_is_counted(tmp_path):
    codes = s2.list_event_codes(_write(tmp_path))
    assert codes["DigMark"] == {"A": 2, "B": 3, "C": 1}


def test_a_second_channel_does_not_absorb_the_first_ones_events(tmp_path):
    """
    The block scan stopped only at "CHANNEL". A second Marker block following
    directly would have merged its events into the first channel's counts.
    """
    codes = s2.list_event_codes(_write(tmp_path))
    assert codes["Keyboard"] == {"S": 2}
    assert "S" not in codes["DigMark"]


def test_adjacent_marker_blocks_stay_separate(tmp_path):
    text = ('"Marker"\n"Chan"\t"1"\n"First"\n0.1\t"A???"\n'
            '"Marker"\n"Chan"\t"2"\n"Second"\n0.2\t"Z???"\n')
    codes = s2.list_event_codes(_write(tmp_path, text))
    assert codes == {"First": {"A": 1}, "Second": {"Z": 1}}


def test_the_shape_matches_the_smr_reader(tmp_path):
    """One dialogue drives both, so both must answer {channel: {code: n}}."""
    codes = s2.list_event_codes(_write(tmp_path))
    assert isinstance(codes, dict)
    for chan, counts in codes.items():
        assert isinstance(chan, str)
        assert all(isinstance(c, str) and isinstance(n, int)
                   for c, n in counts.items())


def test_a_channel_with_no_parseable_events_is_not_offered(tmp_path):
    """Offering it would be offering an empty list to assign sets to."""
    text = ('"Marker"\n"Chan"\t"1"\n"Empty"\n\n'
            '"Marker"\n"Chan"\t"2"\n"Real"\n0.2\t"A???"\n')
    assert list(s2.list_event_codes(_write(tmp_path, text))) == ["Real"]


def test_a_file_with_no_markers_is_empty_not_an_error(tmp_path):
    text = '"SUMMARY"\n"Chan"\t"Type"\n1\t"Waveform"\n'
    assert s2.list_event_codes(_write(tmp_path, text)) == {}
    assert s2.list_event_channels(_write(tmp_path, text)) == []


def test_the_codes_agree_with_the_stim_times_reader(tmp_path):
    """
    The counts shown in the dialogue must be the events the analysis will
    actually find, or the table describes a recording nobody analysed.
    """
    p = _write(tmp_path)
    times = s2._extract_stim_times_py(p, "DigMark")
    counts = s2.list_event_codes(p)["DigMark"]
    assert {k: len(v) for k, v in times.items()} == counts


# ── reachable through io, which is what the dialogue calls ───────────────────

def test_io_lists_the_event_channels_for_a_text_export(tmp_path):
    from mep_cmap import io as mio
    p = _write(tmp_path)
    if mio.detect_format(p) != "spike2":
        pytest.skip("fixture is not detected as a Spike-2 text export")
    assert mio.list_event_channels(p) == ["DigMark", "Keyboard"]
