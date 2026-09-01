"""
mep_cmap.reprocess
~~~~~~~~~~~~~~~~~~
Deciding whether a previously analysed recording can be re-run from its saved
session without asking the analyst anything.

Why this exists
---------------
A study analysed under an earlier version carries all of its manual work in
per-file session JSONs -- crop ranges, conditions, marker placements,
per-channel settings -- but not the newer outputs: no reporting record, so no
TMS-RAT and no NIBS-DAS export, and trial columns that have since been renamed
or corrected. Re-running each file by hand restores nothing that is not
already saved, so the only real cost of doing it interactively is the
analyst's time, one dialogue at a time, across a hundred and thirty files.

What it deliberately does NOT do
--------------------------------
Guess. A file whose session cannot be resolved is reported and left alone,
because the failure mode of guessing here is silent: an analysis that runs to
completion against the wrong channel produces numbers that look entirely
reasonable.

Pure: no Tk, no disk, no pipeline. A session dict and a channel list in, a
decision out, so the rules can be tested against a handful of dictionaries
rather than against a study.
"""

from __future__ import annotations

#: The file can be re-run exactly as saved.
OK = "ok"
#: The file can be re-run, but something about WHERE it lives had to be
#: corrected -- the recording moved. Nothing about the analysis changes.
REPAIRED = "repaired"
#: The session disagrees with itself about which channel was analysed. The
#: file is NOT run and is marked needs_review in the queue, because neither
#: reading can be shown to be the right one from the session alone.
CONFLICT = "conflict"
#: The file cannot be re-run without asking the analyst something.
SKIP = "skip"

#: Key under which a conflict is recorded in FileEntry.review_flags, so the
#: reason survives in the queue rather than only in a run log.
REVIEW_FLAG = "reprocess_channel_conflict"


def resolve_channels(session, chan_list):
    """Work out which channels a saved session meant, in today's file.

    Returns ``(status, resolved, note)``. ``resolved`` is None unless the
    session can be trusted.

    A session records the channel twice: ``channel_choice`` as a name and
    ``channel_idx`` as a position. Neither is authoritative on its own.

    The obvious rule -- trust the name, because a position means nothing once
    a file has been re-exported -- is wrong on real data. In the study this
    was built against, 77 of 123 resolvable sessions name
    'Ch 1 - Handgrip 1' while their index points at 'Ch 3 - MEP', the channel
    list is identical across every file, and the analyses plainly ran on MEP.
    The NAME was the unreliable field there: it recorded the dropdown's
    default when the dropdown was never touched. Trusting it would have
    re-run 77 files against a handgrip force trace and produced a complete,
    plausible analysis of the wrong signal.

    Trusting the index instead is the same mistake facing the other way: it
    is right in that study and would be wrong in one that had been
    re-exported with a channel inserted.

    So a disagreement is not something to resolve. It is the finding, and it
    is reported as CONFLICT for the analyst to settle -- once for a group of
    identical files, not once per file.
    """
    chan_list = list(chan_list or [])
    name = str(session.get("channel_choice") or "").strip()
    saved_idx = session.get("channel_idx")
    idx_valid = isinstance(saved_idx, int) and 0 <= saved_idx < len(chan_list)

    if not chan_list:
        return SKIP, None, "the recording reports no channels."

    if not name:
        if idx_valid:
            return CONFLICT, None, (
                f"the session records no channel name, only index "
                f"{saved_idx} ({chan_list[saved_idx]}), which cannot be "
                f"confirmed against anything.")
        return SKIP, None, "the session records no channel."

    if name not in chan_list:
        if idx_valid:
            return CONFLICT, None, (
                f"the saved channel name '{name}' is not in this recording, "
                f"but the saved index {saved_idx} points at "
                f"'{chan_list[saved_idx]}'.")
        return SKIP, None, (
            f"the saved channel '{name}' is not in this recording "
            f"(it has: {', '.join(chan_list)}).")

    idx = chan_list.index(name)

    if idx_valid and saved_idx != idx:
        return CONFLICT, None, (
            f"the session names '{name}' (index {idx}) but records index "
            f"{saved_idx}, which is '{chan_list[saved_idx]}'. The analysis "
            f"ran on one of them and the session does not say which.")

    saved_set = session.get("analyse_channels") or []
    # An empty analyse_channels is NOT a gap. It is what a single-channel
    # analysis looks like: the set is only populated when the assignment
    # dialogue picks more than one, so 130 of 135 sessions in the study this
    # was built against have nothing there.
    analyse = {i for i in saved_set if isinstance(i, int)
               and 0 <= i < len(chan_list)} or set()
    analyse.add(idx)

    return OK, {"channel_choice": name,
                "channel_idx": idx,
                "analyse_channels": sorted(analyse)}, ""


#: Fields the pipeline needs that a session must actually carry. Deliberately
#: short: everything else either has a defensible default or is restored
#: wholesale by _apply_loaded_session.
_REQUIRED = ("marker_choice",)


def plan_file(session, chan_list, *, needs_unit=False, path=None,
              path_note=""):
    """Whether one file can be re-run unattended, and what it would use.

    ``path`` is the recording as located by :func:`locate_recording`, which
    may differ from the one the session stored; None means it could not be
    found, and ``path_note`` says why.

    ``needs_unit`` says the format asks the analyst to confirm an amplitude
    unit at load time -- the epoched MAT path. A session that recorded one can
    reuse it; a session that did not must NOT fall back to the suggestion the
    dialogue would have offered. The trial files carry a Units_Assumed flag
    precisely because an assumed unit is worth marking, and accepting a
    suggestion silently across a whole study is the opposite of that.
    """
    if path is None and path_note:
        return {"status": SKIP, "resolved": None, "notes": [path_note]}

    status, resolved, note = resolve_channels(session, chan_list)
    if status in (SKIP, CONFLICT):
        return {"status": status, "resolved": None,
                "notes": [n for n in (path_note, note) if n]}

    notes = [n for n in (path_note, note) if n]
    if path_note and status == OK:
        # A moved recording is a change worth reporting even when nothing
        # about the session had to be corrected.
        status = REPAIRED

    missing = [f for f in _REQUIRED
               if not str(session.get(f) or "").strip()]
    if missing:
        return {"status": SKIP, "resolved": None,
                "notes": notes + [f"the session records no "
                                  f"{', '.join(missing)}."]}

    if needs_unit and not str(session.get("emg_unit") or "").strip():
        return {"status": SKIP, "resolved": None,
                "notes": notes + [
                    "this format asks for the amplitude unit and the session "
                    "did not record one; guessing it would change every "
                    "amplitude in the file."]}

    if outlier_review_was_on(session):
        return {"status": SKIP, "resolved": None,
                "notes": notes + [
                    "outlier review was enabled for this analysis, and which "
                    "flagged trials were kept is not recorded in the session; "
                    "re-running would silently re-include every trial that "
                    "was removed."]}

    return {"status": status, "resolved": resolved, "notes": notes}


def outlier_review_was_on(session):
    """Whether this session's analysis put flagged trials to the analyst.

    Absent means ON. The setting defaults to True both where the variable is
    constructed and where a session restores it, so a session that does not
    record it will open the dialogue on the re-run -- and an unattended run
    then hangs on a window nobody is looking at.

    This matters because the decision it produced is not saved anywhere the
    pipeline reads back. ``pipeline_review_outliers`` returns no rejections at
    all when review is off, so re-running a file whose trials were rejected by
    hand would put every one of them back into the mean, and the trial file
    would record them as "Not flagged" -- a changed published number with no
    trace of what changed it.

    Inspector exclusions are a different mechanism and are NOT affected: they
    live in segments_metadata, which the session does carry, and they survive
    a re-run with the Inspector closed.

    In the study this was built against the setting is False in all 135
    sessions and no trial is marked Removed anywhere, so this refuses nothing
    there. It is here for the study where that is not true.
    """
    settings = session.get("settings")
    if not isinstance(settings, dict):
        # No settings block at all: an old session format, which restores by
        # leaving every variable at its constructed default. That default is
        # True.
        return True
    return bool(settings.get("outlier_review", True))


def summarise(plans):
    """Counts by status, for a dry run the analyst reads before committing."""
    out = {OK: 0, REPAIRED: 0, CONFLICT: 0, SKIP: 0}
    for p in plans:
        out[p["status"]] = out.get(p["status"], 0) + 1
    return out


def queue_status_for(plan, statuses):
    """What a planned file's queue entry should become, or None to leave it.

    ``statuses`` supplies the application's status constants, so this module
    stays free of that import and testable on its own.

    A conflict becomes needs_review rather than being logged and forgotten.
    The queue is where the analyst looks; a run report is read once and
    closed, and a file that was quietly not reprocessed is indistinguishable
    from one that was, six months later.

    A skip does NOT change the status. The file could not be planned, which
    is a statement about the session rather than about the analysis already on
    disk, and marking a complete file as needing review because its recording
    has been moved would be misleading.
    """
    if plan.get("status") == CONFLICT:
        return statuses["needs_review"]
    return None


def review_flag_for(plan):
    """The review_flags entry that explains a conflict in the queue."""
    if plan.get("status") != CONFLICT:
        return None
    return {REVIEW_FLAG: " ".join(plan.get("notes") or []) or
            "the saved session disagrees with itself about the channel."}


# ── finding a recording that has moved ───────────────────────────────────────

def index_recordings(paths):
    """basename -> [paths], the lookup a moved-file search needs.

    Built once for a study and passed in, rather than walked per file: a
    hundred and thirty lookups against a network or cloud-synced folder is
    slow enough to matter, and the answer does not change between files.
    """
    out = {}
    for p in paths or ():
        out.setdefault(_basename(p).lower(), []).append(p)
    return out


def _basename(path):
    # Sessions are written on whichever platform ran the analysis, so a path
    # from Windows can be read on macOS and the reverse. os.path.basename only
    # splits on the separator of the CURRENT platform, which silently returns
    # the whole path when the two disagree.
    return str(path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def locate_recording(stored_path, study_root, recordings, *, isfile=None):
    """Find the recording a session refers to, allowing for a moved study.

    Returns ``(path, note)``, with path None when it cannot be resolved.

    Stored paths are relative to the study root, so a study that has been
    reorganised leaves every session pointing somewhere that no longer exists
    -- in the study this was built against, 74 of 135. Those files are not
    missing and their sessions are not stale; only the folder above them
    changed, which is not a reason to refuse to reprocess.

    Matched on basename, and ONLY when exactly one recording has that name.
    Two candidates is not a near miss to break with a heuristic: attaching one
    participant's manual marker edits to another participant's recording
    produces a complete, plausible analysis of the wrong data.
    """
    import os
    isfile = isfile or os.path.isfile

    stored = str(stored_path or "")
    if not stored:
        return None, "the session records no recording path."

    direct = stored if os.path.isabs(stored) else os.path.join(study_root,
                                                               stored)
    if isfile(direct):
        return direct, ""

    # The same path, with the separators of whichever platform saved it.
    #
    # Sessions store a path relative to the study root, written with the
    # separator of the machine that ran the analysis. os.path.join does not
    # translate: on POSIX a backslash is an ordinary filename character, so
    # "rawdata\sub-y001\...\x.txt" becomes a single absurd filename that never
    # exists, and the lookup falls through to the basename search below.
    #
    # That fallback usually finds the file anyway, which is what hid this: in
    # the study this was built against it reported 104 of 106 recordings as
    # having MOVED when almost none had. Two costs. The count misleads, and
    # more seriously the match then rests on the basename being unique rather
    # than on the path the session actually recorded -- so sub-y001, which has
    # a first-session and a second-session holding identically named files,
    # was refused as ambiguous even though its sessions name the right one.
    #
    # Tried only AFTER the literal path, never instead of it, because a POSIX
    # filename may legitimately contain a backslash and the recorded path is
    # the better evidence where it resolves.
    if not os.path.isabs(stored) and ("\\" in stored or "/" in stored):
        parts = [p for p in stored.replace("\\", "/").split("/") if p]
        native = os.path.join(study_root, *parts)
        if native != direct and isfile(native):
            return native, ""

    hits = (recordings or {}).get(_basename(stored).lower(), [])
    if len(hits) == 1:
        return hits[0], (f"the recording has moved since this session was "
                         f"saved; found it at {hits[0]}.")
    if len(hits) > 1:
        return None, (f"{len(hits)} recordings are named "
                      f"'{_basename(stored)}' and the saved path no longer "
                      f"exists, so which one this session belongs to cannot "
                      f"be told apart.")
    return None, (f"the recording '{_basename(stored)}' is not in this study "
                  f"(the session expected it at {stored}).")


# ── grouping conflicts so they are answered once ─────────────────────────────

def conflict_signature(session, chan_list):
    """What makes two conflicts the same question.

    A study that was analysed in one sitting produces the same disagreement in
    every file: the same saved name, the same saved index, the same channel
    layout. Asking about each one separately would be 77 identical questions
    in the study this was built against, and an interface that asks 77
    questions gets answered by clicking through it.

    The channel list is part of the signature, not just the two saved values.
    Two files that disagree in the same way but have different channels are
    different questions, because the index means something different in each.
    """
    return (str(session.get("channel_choice") or "").strip(),
            session.get("channel_idx"),
            tuple(chan_list or ()))


def describe_conflict(signature):
    """The question to put to the analyst, for one signature.

    Returns a dict carrying both readings and what each points at, so the
    caller can offer them as choices rather than as free text. None when the
    signature does not describe a resolvable pair -- there is nothing to
    choose between.
    """
    name, idx, chans = signature
    chans = list(chans or ())
    by_name = chans.index(name) if name in chans else None
    by_index = chans[idx] if isinstance(idx, int) and 0 <= idx < len(chans) \
        else None
    if by_name is None and by_index is None:
        return None
    return {
        "saved_name": name,
        "saved_index": idx,
        "name_points_at": (name, by_name),
        "index_points_at": (by_index, idx),
        "channels": chans,
    }


#: Key under which a shared-recording refusal is recorded in review_flags.
SHARED_RECORDING_FLAG = "reprocess_shared_recording"


def sessions_sharing_a_recording(items):
    """Recordings claimed by more than one session -> the keys claiming them.

    ``items`` is ``(key, recording_path)`` pairs. Matched on basename, for the
    same cross-platform reason :func:`_basename` exists.

    A session records the recording it analysed and the metadata it was
    analysed under, and the two can disagree. In the study this was built
    against, 135 sessions name only 122 distinct recordings: one rest
    recording is claimed by SIX sessions carrying four different task labels
    between them. At most one of those can be right.

    Nothing in a single session reveals this -- each one resolves perfectly on
    its own, which is why plan_file cannot catch it and this is a separate
    pass over the whole study. Run unattended, the six would analyse the same
    rest recording six times and file the results under task-30, task-60,
    task-30BFR and task-REST, every one of them a complete and plausible
    analysis of a recording that is not what the filename says.

    Which half of the pair is wrong -- the path or the metadata -- cannot be
    told from here, so nothing is repaired. They are reported and left.
    """
    by_recording = {}
    for key, path in items or ():
        name = _basename(path).lower()
        if not name:
            continue
        by_recording.setdefault(name, []).append(key)
    return {name: keys for name, keys in by_recording.items()
            if len(keys) > 1}


def shared_recording_note(path, keys):
    """The refusal note for one over-claimed recording."""
    return (f"{len(keys)} saved sessions all name '{_basename(path)}' as the "
            f"recording they analysed, under different output names, so which "
            f"of them describes this recording cannot be told from the "
            f"sessions alone.")


def group_conflicts(items):
    """signature -> [keys], from ``(key, session, chan_list)`` triples.

    ``key`` is whatever identifies a file to the caller; this module does not
    care what it is.
    """
    out = {}
    for key, session, chan_list in items:
        out.setdefault(conflict_signature(session, chan_list), []).append(key)
    return out


#: How a conflict group was answered.
BY_INDEX = "index"
BY_NAME = "name"
UNRESOLVED = "unresolved"


def apply_conflict_answer(session, chan_list, answer):
    """Resolve a conflicted session using the analyst's answer for its group.

    The answer is recorded rather than inferred, and it is applied to the
    whole group at once, so the decision is made once and auditable
    afterwards. UNRESOLVED leaves the file alone: an analyst who does not know
    which reading is right must be able to say so, and the file then stays
    marked for review rather than being run on a coin toss.
    """
    chans = list(chan_list or [])
    if answer == BY_INDEX:
        idx = session.get("channel_idx")
        if not (isinstance(idx, int) and 0 <= idx < len(chans)):
            return None
        return {"channel_choice": chans[idx], "channel_idx": idx,
                "analyse_channels": [idx]}
    if answer == BY_NAME:
        name = str(session.get("channel_choice") or "").strip()
        if name not in chans:
            return None
        idx = chans.index(name)
        return {"channel_choice": name, "channel_idx": idx,
                "analyse_channels": [idx]}
    return None
