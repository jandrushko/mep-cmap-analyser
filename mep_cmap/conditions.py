"""
mep_cmap.conditions
~~~~~~~~~~~~~~~~~~~
Assigning a recording's stimulus events to named conditions.

Why this exists
---------------
A recording's own markers say what kind of stimulus fired, not what it was
for. Twenty pulses labelled ``A`` may be ten before an intervention and ten
after, or three intensities of a recruitment curve, and nothing in the file
distinguishes them. Only the analyst knows, and until now there was nowhere to
say so: the analysis grouped by stimulus type and a session's structure had to
be reconstructed afterwards from trial numbers, if it could be recovered at
all.

A condition is therefore a second axis alongside the stimulus type, not a
replacement for it. ``A`` determines how a response is detected -- its latency
window, its muscle, whether a silent period applies -- and those settings are
identical before and after an intervention. ``pre`` and ``post`` determine
nothing about detection and everything about what the trial means. Collapsing
the two into one name, which is what a single label field forces, loses that
distinction and leaves the analyst configuring the same latency window twice.

The output is a BIDS ``_events.tsv``: ``onset``, ``duration``, ``trial_type``
and a ``condition`` column. Not a private format, because the events file is
already read by this tool's own EDF path and by anything else that understands
BIDS, and a condition assignment is exactly the kind of thing that should
survive being read by something other than the program that wrote it.

Composition
-----------
The analysis groups by a single key, so a stimulus type and a condition are
composed into one -- ``A`` and ``pre`` become ``A·pre``. That happens HERE,
explicitly, and not in the reader: a file on disk states a trial type and a
condition in separate columns, and stays legible to a tool that has never
heard of this one. :func:`group_events` is the single place the two are joined,
so the analysis and the preview cannot compose them differently.

An event with no condition composes to its trial type alone, so a recording
whose conditions were never touched produces exactly the keys it always did.

This module holds no Tk and touches no file: it is the rules, so that they can
be tested without a display and without a recording.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

#: Joins a stimulus type to a condition in the analysis group key. Reserved:
#: a condition name containing it would not survive the round trip, so names
#: are rejected rather than quietly rewritten -- a renamed condition is a
#: different condition, and doing that silently is how a trial ends up in a
#: group the analyst never made.
SEPARATOR = "\u00b7"          # ·

#: What a BIDS events file writes where a column has no value.
NA = "n/a"


class ConditionError(ValueError):
    """A conditions table that cannot be applied, with the reason."""


# ── trial lists ──────────────────────────────────────────────────────────────

def parse_trials(text, n_available=None):
    """Parse ``"1-20, 25, 30-35"`` into a sorted tuple of 0-based indices.

    Written and displayed 1-based, because that is how the trial list, the
    inspector and the trial file number them; stored 0-based, because that is
    how the arrays are indexed. Getting that boundary wrong is an off-by-one in
    every trial of a condition, so the conversion happens in exactly these two
    functions.

    ``n_available`` bounds the range and is worth passing: a typed 1-40 against
    a recording of 20 trials is a mistake, and silently keeping the half that
    exists would leave the analyst believing they had assigned forty.
    """
    if text is None:
        return ()
    out = set()
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(\d+)\s*[-\u2013]\s*(\d+)", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            rng = range(lo, hi + 1)
        else:
            if not re.fullmatch(r"\d+", chunk):
                raise ConditionError(
                    f"{chunk!r} is not a trial number or range. Use numbers and "
                    f"ranges separated by commas, for example 1-20, 25, 30-35.")
            rng = [int(chunk)]
        for one in rng:
            if one < 1:
                raise ConditionError(
                    f"trial {one} does not exist; trials are numbered from 1.")
            if n_available is not None and one > n_available:
                raise ConditionError(
                    f"trial {one} does not exist; this stimulus type has "
                    f"{n_available}.")
            out.add(one - 1)
    return tuple(sorted(out))


def format_trials(indices):
    """Render 0-based indices as a 1-based list, collapsing runs into ranges.

    A condition holding forty consecutive trials should read ``1-40`` rather
    than forty numbers: the table is meant to be checked at a glance, and a
    wall of digits cannot be.
    """
    idx = sorted(set(int(i) for i in indices or ()))
    if not idx:
        return ""
    parts, start, prev = [], idx[0], idx[0]
    for cur in idx[1:] + [None]:
        if cur is not None and cur == prev + 1:
            prev = cur
            continue
        parts.append(f"{start + 1}" if start == prev else f"{start + 1}-{prev + 1}")
        if cur is not None:
            start = prev = cur
    return ", ".join(parts)


# ── names ────────────────────────────────────────────────────────────────────

def sanitise_name(name):
    """Normalise a condition name for a BIDS column.

    Spaces become underscores and the surrounding whitespace goes, so that a
    name survives a tab-separated file. The separator is refused rather than
    replaced, for the reason given at :data:`SEPARATOR`.
    """
    txt = str(name or "").strip()
    if SEPARATOR in txt:
        raise ConditionError(
            f"a condition name may not contain {SEPARATOR!r}, which separates "
            f"the stimulus type from the condition in the analysis.")
    if "\t" in txt or "\n" in txt:
        raise ConditionError(
            "a condition name may not contain tabs or line breaks; the events "
            "file is tab-separated.")
    return re.sub(r"\s+", "_", txt)


def deduplicate_names(pairs):
    """Make repeated (stim_type, condition) names unique by numbering repeats.

    Uniqueness is required of the GROUP KEY, not of the condition name, so the
    comparison has to include the stimulus type. Deduplicating names alone
    renamed the blank condition on the second stimulus type -- ``A`` and ``B``
    both carrying no condition are not duplicates, their keys being ``A`` and
    ``B`` -- which turned an untouched table into ``A`` and ``B·_2`` and broke
    the one property the whole design rests on: that doing nothing reproduces
    the previous analysis exactly.

    A blank condition is never renamed for the same reason. It is the absence
    of a name rather than a name that happens to be empty, and numbering it
    would invent a distinction the analyst did not make.

    Takes and returns ``(stim_type, condition)`` pairs.
    """
    seen, out = {}, []
    for stim, raw in pairs:
        name = str(raw or "")
        if not name:
            out.append((stim, name))       # no name to collide
            continue
        key = (stim, name)
        if key not in seen:
            seen[key] = 1
            out.append(key)
            continue
        seen[key] += 1
        candidate = f"{name}_{seen[key]}"
        while (stim, candidate) in seen:
            seen[key] += 1
            candidate = f"{name}_{seen[key]}"
        seen[(stim, candidate)] = 1
        out.append((stim, candidate))
    return out


# ── the table ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConditionRow:
    """One row: a stimulus type, an optional condition, and its trials.

    Frozen, and the helpers below return new rows rather than mutating: the
    table is edited by a GUI, and an undo that has to reconstruct prior state
    is harder to get right than one that kept it.
    """

    stim_type: str
    condition: str = ""
    trials: tuple = field(default_factory=tuple)
    excluded: bool = False
    # Epoch for this condition, in ms, or None to use the file-wide default.
    # Held here because the review pane is the only place the analyst can SEE
    # whether a window is right -- whether the response is truncated, whether
    # a silent period runs past the end -- and deciding it there is better
    # informed than typing numbers on a settings tab and hoping.
    pre_ms: float = None
    post_ms: float = None

    @property
    def group_key(self):
        """The key the analysis groups by."""
        return compose(self.stim_type, self.condition)

    @property
    def n(self):
        return len(self.trials)

    @property
    def window(self):
        """(pre, post) or None when this condition takes the default."""
        if self.pre_ms is None and self.post_ms is None:
            return None
        return (self.pre_ms, self.post_ms)

    def describe(self):
        where = format_trials(self.trials) or "no trials"
        if self.excluded:
            return f"{self.stim_type}: {where} — excluded"
        if not self.condition:
            return f"{self.stim_type}: {where}"
        return f"{self.stim_type} · {self.condition}: {where}"


def compose(stim_type, condition):
    """Join a stimulus type and a condition into one analysis group key.

    No condition gives the stimulus type unchanged, which is what makes an
    untouched table reproduce the previous behaviour exactly.
    """
    cond = str(condition or "").strip()
    if not cond or cond == NA:
        return str(stim_type)
    return f"{stim_type}{SEPARATOR}{cond}"


def decompose(group_key):
    """Split a group key back into (stim_type, condition).

    The inverse of :func:`compose`, used when writing the trial file, where the
    two belong in separate columns: a condition is a factor to model, not a
    substring to parse out of a name.
    """
    key = str(group_key)
    if SEPARATOR not in key:
        return key, ""
    stim, cond = key.split(SEPARATOR, 1)
    return stim, cond


# ── building the table from what a file already carries ──────────────────────

def rows_from_events(stim_times):
    """One row per stimulus type, holding all its trials.

    The state a file opens in. Doing nothing then produces exactly the analysis
    it would have produced before conditions existed, which is what makes the
    table a refinement rather than a step that must be completed.
    """
    return [ConditionRow(stim_type=st, condition="",
                         trials=tuple(range(len(times))))
            for st, times in sorted(stim_times.items())]


def split_row(rows, index, selection, new_condition, keep_condition=""):
    """Split one row's trials in two, by selection.

    The trials in ``selection`` move to a new row; the rest stay. This is the
    operation the tab exists for -- twenty ``A`` trials becoming ten ``pre``
    and ten ``post`` -- and it is expressed as a split rather than as two
    manual assignments so that no trial can be dropped or duplicated between
    them.
    """
    if not 0 <= index < len(rows):
        raise ConditionError(f"there is no row {index}.")
    row = rows[index]
    chosen = tuple(sorted(set(selection) & set(row.trials)))
    if not chosen:
        raise ConditionError(
            "none of the selected trials are in that condition.")
    remaining = tuple(t for t in row.trials if t not in set(chosen))
    if not remaining:
        raise ConditionError(
            "that would move every trial, which renames the condition rather "
            "than splitting it.")
    out = list(rows)
    out[index] = replace(row, condition=sanitise_name(keep_condition or row.condition),
                         trials=remaining)
    out.insert(index + 1, replace(row,
                                  condition=sanitise_name(new_condition),
                                  trials=chosen))
    return out


def add_blank_row(rows, stim_type):
    return list(rows) + [ConditionRow(stim_type=stim_type, condition="",
                                      trials=())]


def autofill(rows, index, per_row, n_available):
    """Deal this row's remaining trials into runs of ``per_row``.

    A recruitment curve of five intensities repeated in blocks is a real
    layout, and typing five ranges by hand invites an off-by-one that is
    invisible once applied.
    """
    if per_row < 1:
        raise ConditionError("trials per condition must be at least one.")
    if not 0 <= index < len(rows):
        raise ConditionError(f"there is no row {index}.")
    row = rows[index]
    pool = list(row.trials) or list(range(n_available))
    if not pool:
        raise ConditionError("there are no trials to divide.")
    chunks = [tuple(pool[i:i + per_row]) for i in range(0, len(pool), per_row)]
    out = list(rows)
    out[index:index + 1] = [
        replace(row, trials=chunk,
                condition=sanitise_name(f"{row.condition or 'cond'}_{i + 1}"))
        for i, chunk in enumerate(chunks)]
    return out


# ── completeness ─────────────────────────────────────────────────────────────

def unassigned(rows, stim_times):
    """{stim_type: (trial index, ...)} for every event in no row.

    Checked before Apply, because a trial that belongs to no condition simply
    vanishes between the recording and the analysis. That is the same silent
    loss as a marker filtered out of an events file, and harder to notice when
    the analyst did the removing themselves.
    """
    claimed = {}
    for row in rows:
        claimed.setdefault(row.stim_type, set()).update(row.trials)
    out = {}
    for stim, times in stim_times.items():
        missing = set(range(len(times))) - claimed.get(stim, set())
        if missing:
            out[stim] = tuple(sorted(missing))
    return out


def duplicated(rows):
    """{stim_type: (trial index, ...)} for events claimed by more than one row."""
    seen, twice = {}, {}
    for row in rows:
        for t in row.trials:
            key = (row.stim_type, t)
            if key in seen:
                twice.setdefault(row.stim_type, set()).add(t)
            seen[key] = True
    return {k: tuple(sorted(v)) for k, v in twice.items()}


def validate(rows, stim_times, allow_unassigned=False):
    """Raise unless the table can be applied. Returns the rows, names settled.

    Mixed stimulus types in one row are refused: they are different stimuli
    producing different responses, wanting different latency windows, and a
    single row could express only one set of settings for both.
    """
    if not rows:
        raise ConditionError("there are no conditions to apply.")

    for row in rows:
        sanitise_name(row.condition)

    dup = duplicated(rows)
    if dup:
        where = "; ".join(f"{k}: {format_trials(v)}" for k, v in dup.items())
        raise ConditionError(
            f"these trials are in more than one condition, so the analysis "
            f"would count them twice — {where}")

    if not allow_unassigned:
        loose = unassigned(rows, stim_times)
        if loose:
            # Name the stimulus type first. The commonest cause is a type with
            # no row at all -- a recording carrying Trigger and Start Task,
            # every Trigger assigned, and Start Task never mentioned -- and
            # "these trials are in no condition" sends the analyst looking at
            # the rows that are there rather than at the one that is missing.
            missing = [k for k, v in loose.items()
                       if len(v) == len(stim_times.get(k) or [])]
            if missing:
                head = (f"{', '.join(missing)} has no condition at all"
                        if len(missing) == 1
                        else f"{', '.join(missing)} have no conditions at all")
            else:
                head = "some trials are in no condition"
            where = "; ".join(f"{k}: {format_trials(v)}"
                              for k, v in sorted(loose.items()))
            raise ConditionError(
                f"{head} — {where}.  Give them one, or tick the option to "
                f"exclude them.")

    named = deduplicate_names([(r.stim_type, r.condition) for r in rows])
    settled = [replace(r, condition=sanitise_name(c))
               for r, (_st, c) in zip(rows, named)]

    keys = [r.group_key for r in settled if not r.excluded]
    if len(keys) != len(set(keys)):
        raise ConditionError(
            "two conditions produce the same analysis group; rename one.")
    return settled


# ── output ───────────────────────────────────────────────────────────────────

def window_map_from_rows(rows):
    """{group_key: (pre_ms, post_ms)} for conditions given their own epoch.

    Keyed by the group key rather than the stimulus type, because that is what
    the analysis groups by and what the labels tab lists: A-pre and A-post are
    separate rows there and may want different windows, which is the whole
    reason for setting one per condition.

    Conditions left on the default contribute nothing, so a table nobody
    touched produces an empty map -- which is the same as having no per-type
    windows at all.
    """
    out = {}
    for row in rows:
        if row.excluded:
            continue
        win = row.window
        if win is not None:
            out[row.group_key] = win
    return out


def to_event_rows(rows, stim_times, duration=0.0):
    """Records for a BIDS ``_events.tsv``, sorted by onset.

    Excluded trials are written with a trial type of ``n/a`` rather than left
    out, so the file still accounts for every event in the recording. A reader
    can then tell a trial that was excluded from one that was never there,
    which a file of only the kept trials cannot express.
    """
    out = []
    for row in rows:
        times = stim_times.get(row.stim_type) or []
        for t in row.trials:
            if t >= len(times):
                continue
            out.append({
                "onset": float(times[t]),
                "duration": float(duration),
                "trial_type": NA if row.excluded else row.stim_type,
                "condition": row.condition or NA,
            })
    out.sort(key=lambda r: r["onset"])
    return out


def group_events(event_rows, separator=SEPARATOR):
    """Compose events-file records into analysis groups.

    THE single place a stimulus type and a condition are joined. The reader
    returns what the file says; the analysis needs one key per group; and if
    those two were reconciled independently in the pipeline and in the preview,
    the preview would offer trials the run does not analyse. Both call this.

    Returns ``({group_key: [onset, ...]}, {group_key: (stim_type, condition)})``.
    The second is what lets the trial file write the two as separate columns
    rather than leaving a condition to be parsed out of a name.
    """
    groups, decoded = {}, {}
    for row in event_rows or []:
        stim = str(row.get("trial_type") or "").strip()
        if stim in ("", NA):
            continue                      # excluded; accounted for, not analysed
        cond = str(row.get("condition") or "").strip()
        if cond == NA:
            cond = ""
        key = (f"{stim}{separator}{cond}" if cond else stim)
        try:
            onset = float(row["onset"])
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(onset)
        decoded[key] = (stim, cond)
    return ({k: sorted(v) for k, v in groups.items()}, decoded)


def merge_rows(rows, indices, *, condition=None):
    """Combine two or more rows of the same stimulus type into one.

    The inverse of :func:`split_row`, and the reason it exists: a split made
    on a wrong boundary could only be undone while it was still the last
    action. Once the analyst had renamed something else, or reopened the file,
    the halves were permanent and the only route back was Reload from file,
    which discards every other edit too.

    Restricted to ONE stimulus type. Two rows from different types describe
    responses to different stimuli, and a row holding both would claim a
    single condition contains trials that are not comparable -- the trial
    indices are per type, so they would also collide meaninglessly.

    The union is taken, so a trial appearing in two rows appears once here.
    That cannot arise from splitting, but it can from hand-set trial lists,
    and silently duplicating a trial would double-count it in every summary
    downstream.
    """
    if len(indices) < 2:
        raise ConditionError("select two or more conditions to combine.")
    if any(not 0 <= i < len(rows) for i in indices):
        raise ConditionError("that row does not exist.")

    chosen = [rows[i] for i in sorted(indices)]
    stim_types = {r.stim_type for r in chosen}
    if len(stim_types) > 1:
        raise ConditionError(
            "those conditions belong to different stimulus types "
            f"({', '.join(sorted(stim_types))}), and their trials are "
            "numbered separately, so they cannot be combined.")

    trials = tuple(sorted({t for r in chosen for t in r.trials}))
    if condition is None:
        # The first row's name, because a split leaves the original there and
        # keeping it is the outcome someone undoing a split expects. Falls
        # back to the next name that is set, so combining two halves of an
        # unnamed row does not silently discard the only label present.
        condition = next((r.condition for r in chosen if r.condition), "")

    first = min(indices)
    out = [r for i, r in enumerate(rows) if i not in set(indices)]
    out.insert(min(first, len(out)),
               replace(chosen[0], condition=sanitise_name(condition),
                       trials=trials))
    return out


# ── which recording the applied rows belong to ───────────────────────────────

def recording_key(path) -> str:
    """A comparable identity for a recording path. Empty for no path.

    Normalised so that the same file reached as C:/a/b.smr and C:\\a\\B.smr,
    or through a relative path, is recognised as the same recording.
    """
    if not path:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))
    except Exception:                       # noqa: BLE001 - odd input
        return str(path)


def rows_for_recording(rows, owner_key: str, path) -> list:
    """The applied event rows, if they belong to the recording at ``path``.

    Applied conditions are held on the app as one flat list of events with
    onsets. Nothing tied that list to the recording it came from, so after
    opening a different file the previous file's events were still there:
    the analysis cut the new file's EMG at the old file's stimulus times, and
    applying conditions merged the old events into the new file's events
    file, where they survived a restart.

    ``owner_key`` is :func:`recording_key` of the file the rows were applied
    to. Rows owned by another recording come back empty. An empty owner means
    the rows predate this record and their origin is unknown; they are
    returned unchanged, because the reset on opening a file is what removes
    those.
    """
    rows = list(rows or [])
    if not rows or not owner_key:
        return rows
    return rows if owner_key == recording_key(path) else []
