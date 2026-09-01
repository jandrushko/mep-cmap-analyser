"""
NIBS-DAS export — built-in second-level (group-level) MEP-CMAP add-on.

Reshapes the study's per-channel ``_trials.csv`` files into the collated
layout that bignibsdata.com accepts, and builds the folder structure NIBS-DAS
specifies around it:

    collated_participant_data/
        <study>data_<YYYY-MM-DD>.csv    the collated table
        <study>data_<YYYY-MM-DD>.json   units, provenance, what was dropped
        usedata/      the read-only copy statistical software reads, + old/
        code/         data management and analysis scripts, + old/
        outputfiles/  figures and analysis outputs, + old/

Why NOT under derivatives/: NIBS-DAS is not a BIDS derivative dataset. Its
template puts the collated folder BESIDE the raw data
(``participant_data/`` and ``collated_participant_data/`` as siblings), and
that pair is what gets uploaded. Filing it under derivatives/ would break the
structure the platform expects.

Why the date is in the filename and old/ exists: that is NIBS-DAS's version
control. Re-exporting moves the previous file aside rather than overwriting
it, so a collated table someone has already analysed against is never
silently replaced.

The reshape itself lives in ``mep_cmap.nibs_das``; this file is the I/O around
it, so the mapping can be tested without a study on disk.
"""

import csv
import datetime
import glob
import json
import os
import shutil

# Absolute, not relative. Add-ons are loaded with spec_from_file_location under
# a synthetic module name and therefore have NO parent package, so a relative
# import raises at import time and the add-on is silently skipped as broken.
try:
    from mep_cmap import nibs_das as ND
    from mep_cmap.reporting import SIDECAR_SUFFIX
    _IMPORT_ERROR = None
except Exception as _ex:                  # pragma: no cover - packaging failure
    ND = None
    SIDECAR_SUFFIX = "reporting.json"
    _IMPORT_ERROR = _ex

ADDON_NAME        = "nibs_das_export"
ADDON_DESCRIPTION = ("Export the study as a NIBS-DAS collated dataset, ready "
                     "to upload to Big NIBS data")
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "MEP-CMAP Analyser (built-in)"
ADDON_SCOPE       = "group_level"

ADDON_SETTINGS = [
    {"key": "nibs_das_study", "label": "Study name",
     "help": ("Used for the collated filename, e.g. 'mep_example' gives "
              "mep_exampledata_2026-02-04.csv."),
     "type": "str", "default": ""},
    {"key": "nibs_das_root", "label": "Project folder (optional)",
     "help": ("Where collated_participant_data/ is created. Left blank, the "
              "parent of the derivatives folder is used, so the collated "
              "folder sits beside rawdata and derivatives as NIBS-DAS "
              "expects."),
     "type": "str", "default": ""},
    {"key": "nibs_das_extended", "label": "Include extended measures",
     "help": ("Adds cSP duration, latency, area and MEP duration as "
              "csp_dur_<muscle> and so on. OFF by default: NIBS-DAS v1.0-beta "
              "defines mep_ampl and pre_rms only, and these names are this "
              "tool's proposal until its authors adopt them. Turning this on "
              "produces a file that is valid to read but not yet standard."),
     "type": "bool", "default": False},
]


# ── discovery ────────────────────────────────────────────────────────────────

def participant_id(record, trials_path):
    """The BIDS participant label for a recording.

    From the sub- entity of the filename, which every BIDS-ified derivative
    carries, rather than from anything the analyst typed. A study that was
    never BIDS-ified has no label to find, and an empty one is reported rather
    than replaced with the filename: a collated file whose participant_id is
    a recording stem is not something anyone can join to a participants.tsv.
    """
    name = os.path.basename(str(record.get("recording", {}).get("file")
                                or trials_path or ""))
    for part in name.split("_"):
        if part.startswith("sub-") and len(part) > 4:
            return part
    return ""


def trials_for(record):
    """The _trials.csv beside a record, and its rows.

    The record lives in ``results/report/`` and the trials table in the
    results family beside it, both under the same session prefix. Matched on
    that prefix rather than searched for, so two recordings that happen to
    share a folder cannot have their trials swapped.
    """
    path = str(record.get("_path") or "")
    stem = os.path.basename(path)
    if not stem.endswith("_" + SIDECAR_SUFFIX):
        return None, []
    stem = stem[:-(len(SIDECAR_SUFFIX) + 1)]

    report_dir = os.path.dirname(path)
    for folder in (report_dir, os.path.dirname(report_dir),
                   os.path.join(os.path.dirname(report_dir), "data")):
        candidate = os.path.join(folder, f"{stem}_trials.csv")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8", newline="") as fh:
                return candidate, list(csv.DictReader(fh))

    # Last resort: anywhere under the recording's own tree.
    hits = glob.glob(os.path.join(os.path.dirname(report_dir), "**",
                                  f"{stem}_trials.csv"), recursive=True)
    if hits:
        with open(hits[0], encoding="utf-8", newline="") as fh:
            return hits[0], list(csv.DictReader(fh))
    return None, []


def _channel_of(record, trials):
    """Which muscle a trials table describes.

    The record names it, but a single-channel analysis leaves it null because
    the filenames were never channel-tagged. The Channel column is written
    either way, so the table itself is the fallback.
    """
    channel = record.get("recording", {}).get("channel")
    if channel:
        return str(channel)
    for row in trials:
        if row.get("Channel"):
            return str(row["Channel"])
    return ""


# ── the NIBS-DAS folder ──────────────────────────────────────────────────────

#: Every folder their template defines, each with the old/ subfolder that is
#: how NIBS-DAS does version control. Created empty rather than on demand, so
#: the exported folder is upload-ready as it stands and an analyst has an
#: obvious place to put a script instead of inventing one.
_SUBFOLDERS = ("usedata", "code", "outputfiles")


def build_scaffold(project_root):
    """Create collated_participant_data/ and its subfolders. Returns the path."""
    base = os.path.join(project_root, "collated_participant_data")
    os.makedirs(os.path.join(base, "old"), exist_ok=True)
    for sub in _SUBFOLDERS:
        os.makedirs(os.path.join(base, sub, "old"), exist_ok=True)
    return base


def retire(path):
    """Move an existing file into the old/ beside it, dated if it must be.

    NIBS-DAS keeps superseded files rather than deleting them, and this is the
    reason: a collated table someone has already run an analysis against must
    not be silently replaced by a re-export. Overwriting it would invalidate
    their results with no record that anything changed.
    """
    if not os.path.isfile(path):
        return None
    old_dir = os.path.join(os.path.dirname(path), "old")
    os.makedirs(old_dir, exist_ok=True)
    target = os.path.join(old_dir, os.path.basename(path))
    if os.path.exists(target):
        stem, ext = os.path.splitext(os.path.basename(path))
        stamp = datetime.datetime.now().strftime("%H%M%S")
        target = os.path.join(old_dir, f"{stem}_{stamp}{ext}")
    shutil.move(path, target)
    return target


def write_table(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sidecar(schema, report, *, records, columns, rows, study):
    """What the collated table cannot say about itself.

    Three things go in here that the CSV has no column for.

    The units, because the collated format states mV by convention and never
    writes it down. The muscle slugs against the channel names they came from,
    because 'fdi' in the header might have been 'FDI' or 'First Dorsal
    Interosseous' and a reader cannot tell. And what was DROPPED: the trial
    status, the clipping flag and the assumed-units flag have no home in
    NIBS-DAS, so a clipped trial with an underestimated amplitude sits in the
    table looking like an ordinary, lower, number. Naming that here is the
    only place it can be named.
    """
    measures = (schema or {}).get("measures") or {}
    used = list(measures.get("standard") or [])
    if report.get("extended_measures"):
        used += list(measures.get("extended") or [])
    return {
        "Name": study or "collated NIBS-DAS dataset",
        "NIBSDASVersion": (schema or {}).get("schema_version", ""),
        "GeneratedBy": [{
            "Name": "MEP-CMAP Analyser",
            "Version": (records[0].get("tool_version") if records else "") or "",
            "DetectionVersion": (records[0].get("detection_version")
                                 if records else "") or "",
        }],
        "GeneratedOn": datetime.date.today().isoformat(),
        "Counts": {"participants": len({r.get("participant_id") for r in rows}),
                   "rows": len(rows), "columns": len(columns)},
        "Measures": {f"{m['name']}_<muscle>": {"Units": m.get("units", ""),
                                               "SourceColumn": m.get("from", ""),
                                               "Description": m.get("description", "")}
                     for m in used},
        "Muscles": report.get("muscles", {}),
        "TrialNumbering": next(
            (c.get("description") for c in (schema or {}).get("columns", [])
             if c.get("name") == "trial_number"), ""),
        "UnmatchedStimulusLabels": report.get("unmatched_stim_labels", []),
        "NotRepresented": (schema or {}).get("unmapped", []),
        "ExtendedMeasuresAreProposals": bool(report.get("extended_measures")),
    }


# ── the add-on entry point ───────────────────────────────────────────────────

def run(context):
    log = context.log
    root = context.results_dir

    if ND is None:
        log(f"{ADDON_NAME}: could not import mep_cmap.nibs_das "
            f"({_IMPORT_ERROR}), so nothing can be exported.")
        return []

    schema = ND.load_schema()
    if schema is None:
        log(f"{ADDON_NAME}: the NIBS-DAS column definitions are missing from "
            f"the install, so nothing can be exported.")
        return []

    # Reusing the TMS-RAT add-on's discovery rather than writing a second one:
    # two functions that find the same records would drift, and the one that
    # drifted would export a different study from the one being assessed.
    from mep_cmap.add_ons.group_level import tms_rat_report as TRR
    records = TRR.find_records(root, log=log)
    if not records:
        log(f"{ADDON_NAME}: no analysis records found under {root}. "
            f"Re-run First Level to write them.")
        return []

    participant_rows, ppath = TRR.find_participants(root)
    participants = {str(p.get("participant_id") or ""): p
                    for p in participant_rows}
    if ppath is None:
        log(f"{ADDON_NAME}: no participants.tsv found, so age, sex and "
            f"handedness will be blank. NIBS-DAS expects them; this tool does "
            f"not collect participant data, so they have to come from there.")

    sessions, unlabelled = [], []
    for rec in records:
        tpath, trials = trials_for(rec)
        if not trials:
            log(f"{ADDON_NAME}: no trials table for "
                f"{os.path.basename(rec.get('_path', ''))}; skipped.")
            continue
        pid = participant_id(rec, tpath)
        if not pid:
            unlabelled.append(os.path.basename(tpath or rec.get("_path", "")))
            continue
        sessions.append({"participant_id": pid,
                         "channel": _channel_of(rec, trials),
                         "nibs_bids": rec.get("nibs_bids") or {},
                         "trials": trials})

    if unlabelled:
        # Not substituted with the filename. participant_id is the join key to
        # participants.tsv and to every other dataset on the platform, and a
        # recording stem in that column is not a participant label.
        shown = ", ".join(unlabelled[:5])
        more = f" (+{len(unlabelled) - 5} more)" if len(unlabelled) > 5 else ""
        log(f"{ADDON_NAME}: ⚠️  {len(unlabelled)} recording(s) carry no sub- "
            f"entity and were left out — {shown}{more}. BIDS-ify them first, "
            f"or their trials cannot be attributed to a participant.")

    if not sessions:
        log(f"{ADDON_NAME}: nothing to export.")
        return []
    return _write(context, schema, records, sessions, participants)


def project_root_for(results_dir, override="", log=None):
    """The folder collated_participant_data/ belongs beside. ALWAYS absolute.

    NIBS-DAS uploads a project folder holding the raw and the collated data as
    siblings, so the target is the folder that CONTAINS derivatives/, not the
    derivatives tree and not the analysis output folder.

    That has to be found by walking up to the derivatives folder rather than
    taking the parent of results_dir, because for a group-level add-on
    results_dir is the folder holding whichever group_level_LME_ready.csv was
    selected in Stage 2. Its parent is wherever the analyst happened to save
    that file, which is not a property of the study at all.

    A RELATIVE override is resolved against that derived root, not against the
    working directory. Typing "test" into a folder setting means "beside my
    study", not "wherever this application happens to have been launched
    from" -- and the latter puts an export somewhere the analyst has no way to
    guess, which is exactly the confusion this argument was meant to prevent.
    """
    node = os.path.abspath(results_dir)
    auto = os.path.dirname(node)
    while True:
        parent, name = os.path.split(node)
        if not parent or parent == node:
            break
        if name.lower() == "derivatives":
            auto = parent
            break
        node = parent

    override = str(override or "").strip()
    if not override:
        return auto
    if os.path.isabs(override):
        return override
    resolved = os.path.join(auto, override)
    if log:
        log(f"{ADDON_NAME}: the project folder setting '{override}' is a "
            f"relative path, taken as {resolved}. Use a full path, or clear "
            f"the setting to export beside the derivatives folder.")
    return resolved


def _write(context, schema, records, sessions, participants):
    log = context.log
    cfg = context.config or {}
    root = context.results_dir

    extended = bool(cfg.get("nibs_das_extended"))
    study = str(cfg.get("nibs_das_study") or "").strip()

    columns, rows, report = ND.collate(sessions, participants, schema,
                                       extended=extended)
    if not rows:
        log(f"{ADDON_NAME}: the collated table came out empty; nothing written.")
        return []

    if report["unmatched_stim_labels"]:
        log(f"{ADDON_NAME}: tms_stim_mode is blank for these stimulus labels, "
            f"which match neither the single- nor the dual-pulse vocabulary: "
            f"{', '.join(report['unmatched_stim_labels'])}. Add them to the "
            f"schema rather than leaving the column half filled.")

    # Beside rawdata and derivatives, not inside them. NIBS-DAS uploads the
    # project folder with the raw and collated data as siblings.
    project_root = project_root_for(root, cfg.get("nibs_das_root"), log=log)
    base = build_scaffold(project_root)

    stem = f"{study or 'study'}data_{datetime.date.today().isoformat()}"
    csv_path = os.path.join(base, f"{stem}.csv")
    json_path = os.path.join(base, f"{stem}.json")

    for path in (csv_path, json_path):
        moved = retire(path)
        if moved:
            log(f"{ADDON_NAME}: an export of the same date already existed; "
                f"moved to {os.path.relpath(moved, base)} rather than "
                f"overwritten.")

    write_table(csv_path, columns, rows)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar(schema, report, records=records, columns=columns,
                          rows=rows, study=study), fh, indent=2)
        fh.write("\n")

    # The copy statistical software reads. Their template keeps the original
    # untouched and works from this one, so an analysis that mangles its input
    # has not mangled the dataset.
    use_path = os.path.join(base, "usedata", f"{stem}_copy.csv")
    retire(use_path)
    shutil.copy2(csv_path, use_path)

    # The ABSOLUTE path. A relative one here always printed the literal string
    # "collated_participant_data", which says what the folder is called and
    # nothing about where it is -- and where it is, is the question, because
    # results_dir depends on which group CSV was selected.
    log(f"{ADDON_NAME}: {len(rows)} trial rows for "
        f"{len({r['participant_id'] for r in rows})} participant(s) "
        f"across {len(report['muscles'])} muscle(s)")
    log(f"{ADDON_NAME}: written to {base}")
    if extended:
        log(f"{ADDON_NAME}: extended measures are included. Their names are "
            f"this tool's proposal, not NIBS-DAS v1.0-beta, so the file is "
            f"readable but not yet standard.")
    return [csv_path, json_path, use_path]
