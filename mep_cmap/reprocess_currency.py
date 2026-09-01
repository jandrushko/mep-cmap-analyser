"""
mep_cmap.reprocess_currency
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Which recordings' outputs are behind the current trial schema.

A re-run brings old analyses onto current code, and the most visible part of
that is columns which did not exist when they were analysed. This says which
files still lack them, so a batch can re-run the ones that need it instead of
starting from the top every time.

Two things it deliberately does NOT do.

It does not append the missing columns. They are derived from the signal, not
from the old numbers, so producing them IS the re-run; writing a value in
without recomputing would be inventing data.

It does not say anything about correctness. A file can carry every current
column and still hold a 386 ms cortical silent period. This is a coverage
check. Plausibility is a separate question and wants separate machinery.

Pure: headers in, verdicts out. The caller supplies the current schema, which
keeps this module free of the pipeline import and means the answer follows
LAT_COLS automatically rather than a list here that would drift out of date.
"""

from __future__ import annotations

#: A recording whose outputs carry every current column.
CURRENT = "current"
#: Outputs exist but predate one or more current columns.
BEHIND = "behind"
#: No trial file at all -- never analysed, or its output has been lost.
ABSENT = "absent"


def missing_columns(header, current_columns):
    """Current columns this file does not have, in schema order.

    Schema order rather than sorted, so a report reads the way the trial file
    does and a run of adjacent additions is recognisable as one change.
    """
    have = {str(c).strip() for c in (header or ())}
    return [c for c in (current_columns or ()) if c not in have]


def classify(header, current_columns):
    """CURRENT, BEHIND or ABSENT for one recording.

    ``header`` is None when no trial file was found. That is ABSENT rather
    than BEHIND: nothing was produced at all, which is a different situation
    from output that exists and is old, and the two want different handling.
    """
    if header is None:
        return ABSENT
    return BEHIND if missing_columns(header, current_columns) else CURRENT


def summarise(records):
    """Counts by verdict, from ``(key, verdict, missing)`` triples."""
    out = {CURRENT: 0, BEHIND: 0, ABSENT: 0}
    for _key, verdict, _missing in records or ():
        out[verdict] = out.get(verdict, 0) + 1
    return out


def most_common_missing(records, limit=10):
    """[(column, n_files)], commonest first.

    A column missing from many files is one schema addition; a column missing
    from one is more likely that file having been analysed differently. The
    distinction is what makes the report readable at a glance.
    """
    counts = {}
    for _key, verdict, missing in records or ():
        if verdict != BEHIND:
            continue
        for col in missing or ():
            counts[col] = counts.get(col, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:limit] if limit else ordered


def keys_needing_rerun(records):
    """Keys worth re-running: behind the schema, or with no output at all."""
    return [key for key, verdict, _m in records or ()
            if verdict in (BEHIND, ABSENT)]
