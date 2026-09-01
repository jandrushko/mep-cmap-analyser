"""
mep_cmap.reprocess_report
~~~~~~~~~~~~~~~~~~~~~~~~~
What a re-run changed, per file.

A batch re-run brings old analyses up to current outputs, and some of those
outputs are corrections: the cSP/MEP ratio was inverted before v1.4, so files
carrying a cSP will legitimately produce a different number. Which is exactly
why this exists. "Some numbers are supposed to change" and "I cannot tell
which numbers changed" together mean a published figure can move without
anyone noticing.

So every re-run is diffed against the trial file it replaced, and any file
whose values moved is flagged rather than ticked off.

Compared as written, not to a tolerance
---------------------------------------
The pipeline rounds explicitly before writing (``round(x, 4)`` and friends),
so identical inputs produce identical text. Comparing the written strings
therefore catches everything an epsilon would, without anyone having to choose
an epsilon -- and choosing one means choosing a size of change that is allowed
to pass unreported, which is the property being avoided.

Pure: rows in, findings out. No file reading, so the rules can be tested on a
few dictionaries.
"""

from __future__ import annotations

#: Columns identifying a trial. Whichever of these are present in BOTH files
#: form the join key.
#:
#: Condition is included because applying conditions splits a stimulus type
#: into several rows that share a StimType: a row for A-pre reports
#: StimType=A, Condition=pre. Without it those rows collide and every one of
#: them reads as changed.
IDENTITY = ("File", "StimType", "Condition", "Segment")

#: Values that mean "nothing here". Blank and NaN are written differently by
#: different versions of the writer and must not read as a change.
_BLANKS = {"", "nan", "none", "null", "na"}

#: Columns that are reported when they move but do not, on their own, stop a
#: file being marked complete.
#:
#: These carry a stimulus timestamp read from the recording, not a measurement
#: derived from it. On real data they drift in the last digit -- 205.1477
#: against 205.1478, on one trial in twenty -- which is arithmetic noise in
#: the conversion from samples to seconds, and a file flagged for that alone
#: teaches the analyst that the flag means "look at everything", which is the
#: same as meaning nothing.
#:
#: Deliberately a NAMED LIST rather than a numeric tolerance. A tolerance
#: would apply everywhere and would mean choosing a size of change that is
#: allowed to pass unreported on amplitudes and latencies too. These columns
#: still appear in the change report; they simply do not raise the flag by
#: themselves, and any other column moving in the same file still does.
TIMING_COLUMNS = frozenset({
    "Stim_Time(s)",
    "Time_Since_Last_Stim(s)",
    "Segment_Overall",
})


def _norm(value):
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in _BLANKS else text


def identity_columns(old_rows, new_rows):
    """The join key for these two files: IDENTITY, narrowed to what both have."""
    old_cols = set(_columns(old_rows))
    new_cols = set(_columns(new_rows))
    return tuple(c for c in IDENTITY if c in old_cols and c in new_cols)


def _columns(rows):
    out = []
    for row in rows or ():
        for col in row:
            if col not in out:
                out.append(col)
    return out


def _key(row, id_cols):
    return tuple(_norm(row.get(c)) for c in id_cols)


def compare_trials(old_rows, new_rows):
    """Diff two trial tables. Returns a findings dict.

    ``columns``         -- names whose values moved on at least one trial.
    ``added_columns``   -- present only in the new file. EXPECTED: a re-run
                           under a newer version gains columns, and that is
                           the point of running it, so these are reported
                           separately and do not make a file need review.
    ``removed_columns`` -- present only in the old file. Not expected, and
                           worth a person's attention: something that used to
                           be produced no longer is.
    ``rows_only_old`` / ``rows_only_new`` -- the trial count moved, which is a
                           bigger finding than any single value.
    ``changes``         -- up to a few worked examples per column, so the
                           report says what moved rather than only that
                           something did.
    """
    old_rows = list(old_rows or ())
    new_rows = list(new_rows or ())
    id_cols = identity_columns(old_rows, new_rows)

    old_cols = _columns(old_rows)
    new_cols = _columns(new_rows)
    added = [c for c in new_cols if c not in old_cols]
    removed = [c for c in old_cols if c not in new_cols]
    shared = [c for c in old_cols if c in new_cols and c not in id_cols]

    findings = {
        "identity": list(id_cols),
        "added_columns": added,
        "removed_columns": removed,
        "columns": [],
        "changes": [],
        "rows_only_old": [],
        "rows_only_new": [],
    }

    if not id_cols:
        # Nothing to join on. Refused rather than guessed: comparing by row
        # ORDER would silently pair trial 1 of one file with trial 1 of the
        # other, which is only correct while nothing has been added or
        # removed -- precisely the case worth detecting.
        findings["unjoinable"] = True
        return findings

    old_by = {}
    for row in old_rows:
        old_by.setdefault(_key(row, id_cols), row)
    new_by = {}
    for row in new_rows:
        new_by.setdefault(_key(row, id_cols), row)

    findings["rows_only_old"] = [list(k) for k in old_by if k not in new_by]
    findings["rows_only_new"] = [list(k) for k in new_by if k not in old_by]

    moved = []
    for col in shared:
        examples = []
        for key, old_row in old_by.items():
            new_row = new_by.get(key)
            if new_row is None:
                continue
            was, now = _norm(old_row.get(col)), _norm(new_row.get(col))
            if was != now:
                examples.append({"trial": list(key), "was": was, "now": now})
        if examples:
            moved.append(col)
            findings["changes"].append({
                "column": col,
                "trials_affected": len(examples),
                "examples": examples[:3],
            })
    findings["columns"] = moved
    # Reported either way; only these raise the flag.
    findings["flagging_columns"] = [c for c in moved
                                    if c not in TIMING_COLUMNS]
    return findings


def needs_review(findings):
    """Whether these findings should stop a file being marked complete.

    Added columns alone do not. That is what a re-run is FOR: the reporting
    record, the current trial schema, the columns that did not exist when
    these files were analysed. Treating them as a change would flag all 123
    files and make the flag meaningless.

    A stimulus timestamp drifting in its last digit does not either, for the
    same reason and by the same argument. See TIMING_COLUMNS.

    Everything else does: a value that moved, a column that vanished, or a
    trial count that shifted.
    """
    if not findings or findings.get("unjoinable"):
        return True
    return bool(findings.get("flagging_columns")
                or findings.get("removed_columns")
                or findings.get("rows_only_old")
                or findings.get("rows_only_new"))


def report_rows(key, findings):
    """Flatten findings into rows for the study-level change report CSV."""
    out = []
    for change in (findings or {}).get("changes") or ():
        first = (change.get("examples") or [{}])[0]
        out.append({
            "File": key,
            "Column": change["column"],
            "TrialsAffected": change["trials_affected"],
            "ExampleWas": first.get("was", ""),
            "ExampleNow": first.get("now", ""),
        })
    for col in (findings or {}).get("removed_columns") or ():
        out.append({"File": key, "Column": col, "TrialsAffected": "",
                    "ExampleWas": "(column no longer produced)",
                    "ExampleNow": ""})
    return out
