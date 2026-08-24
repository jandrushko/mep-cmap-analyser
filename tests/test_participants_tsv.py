"""
participants.tsv as BIDS-ify writes it.

A file carrying only participant_id is valid BIDS and useless: nothing can
tell "this study did not record age" from "nobody added the column", and the
reporting checklist's participant items are unanswerable either way.
"""

import json
import os

from mep_cmap.bidsify import (DatasetLayout, upsert_participant,
                              PARTICIPANT_TEMPLATE_COLUMNS)


def _layout(tmp_path):
    root = tmp_path / "rawdata"
    root.mkdir()
    return DatasetLayout(rawdata_root=str(root))


def _read(tmp_path):
    p = os.path.join(str(tmp_path / "rawdata"), "participants.tsv")
    with open(p, encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    header = lines[0].split("\t")
    return header, [dict(zip(header, l.split("\t"))) for l in lines[1:]]


def test_the_standard_columns_are_present_even_when_unknown(tmp_path):
    upsert_participant(_layout(tmp_path), "sub-01")
    header, rows = _read(tmp_path)
    assert header[0] == "participant_id"
    for col in PARTICIPANT_TEMPLATE_COLUMNS:
        assert col in header
    assert rows[0]["age"] == "n/a"
    assert rows[0]["sex"] == "n/a"


def test_a_supplied_value_is_kept_not_overwritten_by_the_template(tmp_path):
    lay = _layout(tmp_path)
    upsert_participant(lay, "sub-01", {"age": 24, "sex": "F"})
    _header, rows = _read(tmp_path)
    assert rows[0]["age"] == "24"
    assert rows[0]["sex"] == "F"
    assert rows[0]["handedness"] == "n/a"


def test_a_second_participant_does_not_wipe_the_first(tmp_path):
    lay = _layout(tmp_path)
    upsert_participant(lay, "sub-01", {"age": 24})
    upsert_participant(lay, "sub-02")
    _header, rows = _read(tmp_path)
    by_id = {r["participant_id"]: r for r in rows}
    assert by_id["sub-01"]["age"] == "24"
    assert by_id["sub-02"]["age"] == "n/a"


def test_rows_are_deduplicated_by_id(tmp_path):
    lay = _layout(tmp_path)
    upsert_participant(lay, "sub-01", {"age": 24})
    upsert_participant(lay, "sub-01", {"sex": "F"})
    _header, rows = _read(tmp_path)
    assert len(rows) == 1
    assert rows[0]["age"] == "24" and rows[0]["sex"] == "F"


# ── the sidecar ──────────────────────────────────────────────────────────────

def test_the_sidecar_is_written_even_with_no_extras(tmp_path):
    """
    It was gated on `extra`, and BIDS-ify calls this with the id alone -- so
    the sidecar never appeared and the columns went out undocumented.
    """
    upsert_participant(_layout(tmp_path), "sub-01")
    p = os.path.join(str(tmp_path / "rawdata"), "participants.json")
    assert os.path.isfile(p)
    described = json.load(open(p, encoding="utf-8"))
    assert set(PARTICIPANT_TEMPLATE_COLUMNS) <= set(described)
    assert "participant_id" not in described, "BIDS already defines it"


def test_the_standard_columns_carry_their_bids_meaning(tmp_path):
    upsert_participant(_layout(tmp_path), "sub-01")
    described = json.load(open(
        os.path.join(str(tmp_path / "rawdata"), "participants.json"),
        encoding="utf-8"))
    assert described["age"]["Units"] == "years"
    assert set(described["sex"]["Levels"]) == {"M", "F", "O"}
    assert set(described["handedness"]["Levels"]) == {"L", "R", "A"}


def test_a_custom_column_is_flagged_for_description(tmp_path):
    """An undescribed custom column is a gap; a placeholder makes it visible
    in the file rather than absent from it."""
    upsert_participant(_layout(tmp_path), "sub-01", {"group": "stroke"})
    described = json.load(open(
        os.path.join(str(tmp_path / "rawdata"), "participants.json"),
        encoding="utf-8"))
    assert "TODO" in described["group"]["Description"]


def test_an_existing_sidecar_is_never_overwritten(tmp_path):
    lay = _layout(tmp_path)
    p = os.path.join(str(tmp_path / "rawdata"), "participants.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"age": {"Description": "hand written"}}, fh)
    upsert_participant(lay, "sub-01")
    assert json.load(open(p, encoding="utf-8"))["age"]["Description"] \
        == "hand written"
