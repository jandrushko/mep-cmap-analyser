"""
mep_cmap.nibs_das
~~~~~~~~~~~~~~~~~
Turning this tool's per-recording records and trial tables into the collated
layout NIBS-DAS expects, for upload to bignibsdata.com.

Pure: no disk, no Tk, no discovery. Records and frames in, columns and rows
out, so the reshape can be tested against a handful of dictionaries rather
than against a study.

Two things about the target format shape everything here.

It is WIDE ON MUSCLE. One row per participant per trial, with a column per
measure per muscle (mep_ampl_fdi, pre_rms_fdi, mep_ampl_apb, ...). This tool's
_trials.csv is one file per channel, so the reshape is a concatenate and a
pivot rather than a restructure.

It is NARROW ON EVERYTHING ELSE. Twenty columns, against fifty-six in
_trials.csv. Most of what is dropped is uninteresting, but Outlier_Decision,
Clipped and Units_Assumed have no home in the format at all, and their absence
is not visible to anyone reading the result. See ``unmapped`` in the schema.
"""

from __future__ import annotations

import os
import re
import json

from .events_model import pair_key


def load_schema():
    """The NIBS-DAS column definitions, or None if the asset is absent.

    None rather than raising, as with the TMS-RAT mapping: this drives an
    optional export and a missing asset must not stop an analysis.
    """
    path = os.path.join(os.path.dirname(__file__), "schema", "nibs_das_v1_0.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                       # noqa: BLE001 — optional asset
        return None


# ── muscle slugs ─────────────────────────────────────────────────────────────

def muscle_slug(channel) -> str:
    """A channel label as a NIBS-DAS column suffix.

    Their suffixes are short conventional abbreviations (fdi, apb). Channel is
    free text the analyst typed, so 'First Dorsal Interosseous' and 'FDI_left'
    both arrive and both have to become something valid.

    Deliberately NOT canonicalised to a muscle vocabulary. Guessing that
    'First Dorsal Interosseous' means fdi would silently rename a column, and
    a reader comparing two datasets could not tell a real fdi from a guessed
    one. The slug is mechanical; the original is recorded in the sidecar.
    """
    s = re.sub(r"[^0-9a-z]+", "_", str(channel or "").strip().lower())
    return s.strip("_") or "unknown"


# ── stimulus mode ────────────────────────────────────────────────────────────

def stim_mode(label, vocabulary) -> str:
    """'single', 'dual', or blank when the label says neither.

    Matched on the ALPHABETIC characters only and case-insensitively, so
    'sici', 'SICI_80' and 'ICF120' all resolve to the same entry. Labels carry
    intensities and suffixes in practice and an exact match would miss most of
    them.

    Blank rather than a default. A guessed 'single' on a paired-pulse trial is
    a wrong number that nothing downstream can detect, and in a shared dataset
    it is worse than a gap: a gap is visible.
    """
    letters = re.sub(r"[^a-z]", "", str(label or "").lower())
    if not letters:
        return ""
    for mode in ("single", "dual"):
        for token in (vocabulary or {}).get(mode, []):
            if letters == re.sub(r"[^a-z]", "", str(token).lower()):
                return mode
    return ""


# ── parameter-set scope ──────────────────────────────────────────────────────

def set_values_for_trial(nibs_bids, stim_type, condition="") -> dict:
    """The parameter-set values in force for ONE trial.

    Three of the exported columns -- both pulse intensities and the
    inter-stimulus interval -- are PARAMETER SET scope, not file scope. A
    recording holding a test-alone type and a SICI type has two sets, and the
    two differ in exactly those fields. Reading them once per file would stamp
    whichever set iterated first onto every row, and the collated table would
    then state a conditioning intensity for trials that never had one.

    The lookup order matters. The Conditions tab can split one stimulus code
    across conditions, and each half may carry its own set, so the split key is
    tried first and the bare code second. Going the other way would return the
    unsplit set for a trial that has a more specific one.
    """
    code_sets = (nibs_bids or {}).get("code_sets") or {}
    param_sets = (nibs_bids or {}).get("parameter_sets") or {}

    name = None
    if condition:
        name = code_sets.get(pair_key(stim_type, condition))
    if name is None:
        name = code_sets.get(stim_type)
    if name is None:
        return {}
    return dict((param_sets.get(name) or {}).get("values") or {})


# ── the collated table ───────────────────────────────────────────────────────

#: Where a fixed column's value comes from. Read from the schema at call time
#: rather than hard-coded, so a spec change is a JSON edit.
_TRIALS = "trials"
_PARTICIPANTS = "participants"
_NIBS = "nibs"
_DERIVED = "derived"


def columns(muscles, schema, *, extended=False) -> list:
    """The header, fixed columns then measures grouped BY MUSCLE.

    Grouped by muscle rather than by measure because that is the order in
    their example file: mep_ampl_fdi, pre_rms_fdi, mep_ampl_apb, pre_rms_apb.
    A reader scanning for one muscle finds its columns together.
    """
    fixed = [c["name"] for c in (schema or {}).get("columns", [])]
    measures = list(((schema or {}).get("measures") or {}).get("standard") or [])
    if extended:
        measures += list(((schema or {}).get("measures") or {}).get("extended") or [])
    return fixed + [f"{m['name']}_{slug}" for slug in muscles for m in measures]


def _value(col, *, trial, participant, fields, set_values, vocabulary):
    """One fixed column, from whichever scope the schema says owns it."""
    src, key = col.get("source"), col.get("from")
    if src == _PARTICIPANTS:
        return participant.get(key, "")
    if src == _TRIALS:
        return trial.get(key, "")
    if src == _NIBS:
        # Parameter-set scope first: those values are per stimulus type, and
        # the file-scope mapping does not hold them at all.
        if key in set_values:
            return set_values[key]
        return fields.get(key, "")
    if src == _DERIVED:
        if col["name"] == "tms_stim_mode":
            return stim_mode(trial.get(key, ""), vocabulary)
    return ""


def _normalise(name, value):
    """The small vocabulary differences between NIBS-BIDS and NIBS-DAS.

    Their example file writes 'monophasic' where NIBS-BIDS writes
    'Monophasic'. Lowercasing here rather than in the schema because it is a
    property of the target format, not of the field.
    """
    if name == "waveform" and value:
        return str(value).lower()
    return value


def collate(sessions, participants, schema, *, extended=False):
    """The collated table: (columns, rows, report).

    ``sessions`` is one entry per participant per channel:

        {"participant_id": "sub-01",
         "channel":        "FDI",
         "nibs_bids":      {...},        the record's nibs_bids block
         "trials":         [ {LAT_COLS column: value}, ... ]}

    ``participants`` maps participant_id to a participants.tsv row.

    Rows are keyed on (participant_id, trial_number) so that the channels of
    one participant MERGE into a single row rather than producing one row per
    muscle. That key is the whole reason the trial number has to be
    Segment_Overall: Segment restarts per stimulus type, so in a recording
    with two types it would collide two different trials into one row and
    silently overwrite one muscle's amplitude with another trial's.

    ``report`` carries what the caller should tell the user about: labels that
    matched no stimulus-mode token, and the muscle slugs with the channel
    names they came from.
    """
    schema = schema or {}
    vocabulary = schema.get("stim_mode_vocabulary") or {}
    measures = list((schema.get("measures") or {}).get("standard") or [])
    if extended:
        measures += list((schema.get("measures") or {}).get("extended") or [])

    muscles, slugs = [], {}
    for s in sessions:
        slug = muscle_slug(s.get("channel"))
        if slug not in muscles:
            muscles.append(slug)
        slugs.setdefault(slug, set()).add(str(s.get("channel") or ""))

    cols = columns(muscles, schema, extended=extended)
    fixed = list(schema.get("columns") or [])
    rows, order, unmatched = {}, [], set()

    for s in sessions:
        pid = s.get("participant_id") or ""
        slug = muscle_slug(s.get("channel"))
        nibs = s.get("nibs_bids") or {}
        fields = dict(nibs.get("fields") or {})
        participant = dict(participants.get(pid) or {})

        for trial in s.get("trials") or []:
            set_values = set_values_for_trial(
                nibs, trial.get("StimType", ""), trial.get("Condition", ""))
            key = (pid, str(trial.get("Segment_Overall", "")))
            if key not in rows:
                rows[key] = {c: "" for c in cols}
                order.append(key)
                for col in fixed:
                    rows[key][col["name"]] = _normalise(
                        col["name"],
                        _value(col, trial=trial, participant=participant,
                               fields=fields, set_values=set_values,
                               vocabulary=vocabulary))
                if (not rows[key].get("tms_stim_mode")
                        and trial.get("Stim_Label")):
                    unmatched.add(str(trial.get("Stim_Label")))
            for m in measures:
                rows[key][f"{m['name']}_{slug}"] = trial.get(m["from"], "")

    report = {
        "muscles": {k: sorted(v) for k, v in slugs.items()},
        "unmatched_stim_labels": sorted(unmatched),
        "extended_measures": extended,
    }
    return cols, [rows[k] for k in order], report
