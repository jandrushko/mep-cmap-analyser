"""
mep_cmap.results_layout
~~~~~~~~~~~~~~~~~~~~~~~
Where a result file goes inside ``results/``.

One recording writes nine files per channel. A two-channel study is eighteen
loose files in one folder; a five-channel study is forty-five, and every add-on
adds one more per channel. The folder stops being readable long before the
study stops being ordinary.

Grouped by OUTPUT FAMILY rather than by channel: an analyst comparing a
measurement across channels has them side by side, and the number of folders is
fixed by what the tool produces rather than growing with the recording.

FILENAMES DO NOT CHANGE. A file has to be identifiable from its name wherever
it ends up -- moved, copied into a manuscript folder, or attached to an email --
so the full BIDS prefix stays even though it is redundant inside a folder that
already implies part of it. It also means nothing that already reads these
files by name has to change.

Nothing is ever moved. Existing studies keep the flat layout they were written
with, and :func:`search_roots` reads both, so a folder half in each state still
loads completely.
"""

from __future__ import annotations

import os

#: Suffix (after the BIDS prefix) -> subfolder. Longest match wins, so
#: ``_summary_with_outliers.csv`` cannot be captured by ``_summary.csv``.
#:
#: Folder names say GRANULARITY where two folders differ only in that:
#: ``trial-level`` is one row per trial, ``summary`` one row per condition.
#: Not ``trials``, which collided with the session-level ``trials/`` folder of
#: per-stimulus exports, leaving one session tree with two different things
#: under the same name.
FAMILIES = {
    "trials.csv": "trial-level",
    "trials.json": "trial-level",
    # Column-narrowed copy of trials.csv, same one row per trial. Beside the
    # file it is drawn from, because it is the same table viewed through fewer
    # columns rather than a different result. The longest-suffix rule in
    # family_for and sibling picks this over "trials.csv", which it also ends
    # with.
    "trials_selected.csv": "trial-level",
    "summary.csv": "summary",
    "summary_with_outliers.csv": "summary",
    "averaged.csv": "summary",
    "onset_methods.csv": "onset-methods",
    "onset_method_summary.csv": "onset-methods",
    "segments.npz": "segments",
    "report.csv": "report",
    # The per-recording record of HOW the analysis was configured. Filed with
    # the report family because it is written FOR a report rather than being
    # one, and because the report family is already excluded from the search
    # for analysis input -- which this must be, since a stacked JSON beside
    # the trial files is exactly the kind of thing a loose glob picks up.
    "reporting.json": "report",
}

#: Anything an add-on writes that is not named above. Add-ons are the reason
#: this folder grows without bound, and they are the files an analyst is least
#: likely to be hunting for, so they get their own place rather than a guess.
ADDON_DIR = "add-ons"

#: Never searched for analysis input. The report is a narrow, stacked VIEW of
#: trials.csv, and a stacked file with a Channel column is close enough in
#: shape to Stage 2's own input to be picked up by a loose glob. Excluded by
#: folder as well as by name, so neither alone has to be right.
EXCLUDED_FROM_SEARCH = ("report",)


def family_for(filename: str) -> str:
    """The subfolder a result file belongs in.

    Matched on the end of the name, because everything carries a long BIDS
    prefix and only the tail says what the file is. Longest suffix first:
    ``_summary_with_outliers.csv`` ends with ``summary.csv`` too.
    """
    name = os.path.basename(filename)
    for suffix in sorted(FAMILIES, key=len, reverse=True):
        if name.endswith(suffix):
            return FAMILIES[suffix]
    return ADDON_DIR


def result_path(results_root: str, filename: str, create: bool = True) -> str:
    """Full path for a result file, in its family's subfolder.

    ``filename`` is the complete name including the BIDS prefix, so callers
    keep building names exactly as they did and only change where they put
    them.
    """
    folder = os.path.join(results_root, family_for(filename))
    if create:
        os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, os.path.basename(filename))


def search_roots(results_root: str) -> list:
    """Folders to scan when looking for result files, flat layout included.

    The flat folder first, then one level of subfolders. One level only: these
    are the tool's own folders, and recursing without limit would wander into
    ``figures/`` and anything an analyst has left in there.

    ``report`` is skipped -- see EXCLUDED_FROM_SEARCH.
    """
    if not results_root or not os.path.isdir(results_root):
        return []
    roots = [results_root]
    try:
        for entry in sorted(os.listdir(results_root)):
            if entry in EXCLUDED_FROM_SEARCH:
                continue
            path = os.path.join(results_root, entry)
            if os.path.isdir(path):
                roots.append(path)
    except OSError:
        pass
    return roots


def find_results(results_root: str, suffix: str) -> list:
    """Every file under ``results/`` whose name ends with ``suffix``.

    The one way anything should look for results, so a caller cannot
    accidentally search only the flat folder and silently miss a study written
    with the newer layout -- which would look like a study with no results
    rather than like a bug.
    """
    out = []
    for root in search_roots(results_root):
        try:
            for name in sorted(os.listdir(root)):
                if name.endswith(suffix) and os.path.isfile(
                        os.path.join(root, name)):
                    out.append(os.path.join(root, name))
        except OSError:
            continue
    # A file present in both layouts is one result, not two.
    seen, unique = set(), []
    for p in out:
        key = os.path.basename(p).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def results_root_for(path: str) -> str:
    """The ``results/`` folder a result file lives under, flat or foldered.

    A file in the flat layout sits directly in it; one in the new layout sits a
    single level down. Both answer the same question, and nothing else should
    have to know which layout it is looking at.
    """
    d = os.path.dirname(os.path.abspath(path))
    if os.path.basename(d) in set(FAMILIES.values()) | {ADDON_DIR}:
        return os.path.dirname(d)
    return d


def sibling(path: str, suffix: str) -> str:
    """The matching result file for another family, given any one of them.

    Several places used to reach a related output by string surgery on the
    path -- ``main_csv.replace("_trials.csv", "_summary.csv")`` -- which is
    exact while every file shares one folder and wrong the moment they do not.
    The prefix is taken from the file that is known to exist and the
    destination resolved through the layout, so both arrangements work.

    An existing flat file wins over a foldered path that does not exist yet, so
    a study written before the folders is read where it actually is.
    """
    base = os.path.basename(path)
    stem = base
    for known in sorted(FAMILIES, key=len, reverse=True):
        if base.endswith("_" + known):
            stem = base[: -len(known) - 1]
            break
    else:
        stem = os.path.splitext(base)[0]
    name = f"{stem}_{suffix}"

    root = results_root_for(path)
    flat = os.path.join(root, name)
    if os.path.isfile(flat):
        return flat
    return result_path(root, name, create=False)
