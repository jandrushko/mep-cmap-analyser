"""
Which files may enter the analysis queue, and why the others may not.

The extension set is DERIVED from the readers in one place. Three hand-written
copies of it lived in the dataset queue and its dialogue text, and nothing
failed loudly when a reader was added and they were not all updated.
"""

import os

from mep_cmap import io as mio


def test_the_globs_are_derived_from_the_supported_extensions():
    assert mio.SUPPORTED_GLOBS == tuple(f"*{e}" for e in mio.SUPPORTED_EXTENSIONS)
    assert len(mio.SUPPORTED_GLOBS) == len(mio.SUPPORTED_EXTENSIONS)


def test_every_supported_extension_appears_in_the_dialogue_text():
    """The 'no files found' message names the formats; a stale message sends
    the analyst looking for a fault in a file the tool would have read."""
    for ext in mio.SUPPORTED_EXTENSIONS:
        assert ext in mio.SUPPORTED_EXT_TEXT


def test_the_open_dialogue_filter_covers_every_reader():
    combined = mio.supported_filetypes(include_all=False)[0][1]
    for ext in mio.SUPPORTED_EXTENSIONS:
        assert f"*{ext}" in combined


def test_all_files_is_offered_only_when_asked_for():
    assert mio.supported_filetypes(include_all=True)[-1] == ("All files", "*.*")
    assert len(mio.supported_filetypes(include_all=False)) == 1


# ── the reason, not just the verdict ─────────────────────────────────────────

def test_a_readable_extension_has_no_reason_to_be_skipped():
    for ext in mio.SUPPORTED_EXTENSIONS:
        assert mio.unsupported_reason(f"/tmp/recording{ext}") is None


def test_a_known_native_format_explains_how_to_export_it():
    """
    A LabChart .adicht is the commonest thing sitting beside the data. Saying
    only "unsupported" leaves the analyst stuck; naming the export route is
    the whole reason UNREADABLE_FORMATS carries prose.
    """
    why = mio.unsupported_reason("/tmp/session.adicht")
    assert why and "export" in why.lower()
    assert mio.unsupported_reason("/tmp/x.cfs")
    assert mio.unsupported_reason("/tmp/x.s2r")


def test_an_unknown_extension_says_what_is_supported():
    why = mio.unsupported_reason("/tmp/notes.docx")
    assert why and ".docx" in why
    assert ".smr" in why


def test_a_file_with_no_extension_is_named_as_such():
    why = mio.unsupported_reason("/tmp/README")
    assert why and "no extension" in why


def test_case_does_not_decide_whether_a_file_can_be_read():
    """Windows hands back .EDF as often as .edf."""
    assert mio.unsupported_reason("/tmp/REC.EDF") is None
    assert mio.unsupported_reason("/tmp/REC.SMR") is None


def test_the_brainvision_sidecars_are_not_queued_separately():
    """
    Only the .vhdr header is globbed. Globbing .eeg and .vmrk would enter one
    recording into the queue three times, and two of those entries would be
    unopenable on their own.
    """
    assert "*.vhdr" in mio.SUPPORTED_GLOBS
    assert "*.eeg" not in mio.SUPPORTED_GLOBS
    assert "*.vmrk" not in mio.SUPPORTED_GLOBS
