"""
The release gate must not cry wolf on facts about the past.

check_release.py compares every version string against TOOL_VERSION. Some
sentences name a PAST release on purpose -- when a feature began, which
versions carried which licence -- and bumping them would make the document
false. A gate that has to be overridden at every release is one nobody reads,
so those phrasings are stripped before the comparison.

The other half matters just as much: stripping must stay narrow enough that a
genuinely stale version string is still caught.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_release", ROOT / "check_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR = _load()


def _versions(line):
    """The version strings the gate would compare, after noise stripping."""
    return CR.VERSION_RE.findall(CR.STRIP_NOISE.sub("", line))


# ── facts about the past are left alone ──────────────────────────────────────

def test_when_a_feature_began_is_not_a_stale_version():
    """
    The real line: provenance began in 1.4.5, so this sentence must keep
    saying 1.4.5 at 1.4.7 and at 2.0.
    """
    line = ("Sessions written before 1.4.5 carry no provenance, so their "
            "markers cannot be attributed and are all re-detected.")
    assert _versions(line) == []


def test_the_v_prefixed_spelling_is_also_handled():
    assert _versions("Sessions written before v1.4.5 carry no provenance") == []


def test_the_word_version_may_appear():
    assert _versions("anything recorded before version 1.4.5 lacks it") == []


def test_the_licence_history_still_works():
    """Already handled; asserted so the new patterns cannot break it."""
    line = ("Versions 1.3.3 and earlier were released under the MIT Licence "
            "and remain so.")
    assert _versions(line) == []


def test_a_historical_changelog_heading_still_works():
    assert _versions("## What's New in 1.3.2") == []


# ── a genuinely stale string is still caught ─────────────────────────────────

def test_a_plain_stale_version_is_not_stripped():
    """The whole point of the gate."""
    assert _versions("MEP-CMAP Analyser 1.4.5") == ["1.4.5"]


def test_a_citation_line_is_not_stripped():
    line = "Andrushko, J. W. (2026). MEP-CMAP Analyser (Version 1.4.5)."
    assert "1.4.5" in _versions(line)


def test_before_does_not_strip_a_version_elsewhere_on_the_line():
    """
    Only the historical clause is removed. A stale string sitting beside one
    must still be reported, which is why the file strips phrases rather than
    skipping whole lines.
    """
    line = ("Sessions written before 1.4.5 carry no provenance. "
            "This is MEP-CMAP Analyser 1.4.5 and it is current")
    assert _versions(line) == ["1.4.5"]


def test_the_word_before_alone_does_not_strip_anything():
    """It has to be 'before <version>', not merely a line containing both."""
    line = "Run the analysis before saving, in version 1.4.5 or later"
    assert _versions(line) == ["1.4.5"]


def test_a_version_at_the_end_of_a_sentence_is_invisible_to_the_gate():
    """
    Not the new patterns: VERSION_RE ends with (?![\\d.]), so a version
    followed by a full stop never matches at all. Recorded rather than fixed
    -- widening it would change what the gate sees across every file, which
    is not a thing to do on the way to a tag.
    """
    assert CR.VERSION_RE.findall("this is version 1.4.5.") == []
    assert CR.VERSION_RE.findall("this is version 1.4.5 here") == ["1.4.5"]


# ── the repo as it stands ────────────────────────────────────────────────────

def test_the_release_gate_passes_on_this_repo():
    """
    This is what release.yml runs before publishing binaries and uploading to
    PyPI, and both jobs depend on it. A failure here means a tag produces no
    release at all.
    """
    assert CR.main() == 0, (
        "check_release.py fails; a tag would publish nothing")
