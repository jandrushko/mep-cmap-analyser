"""
mep_cmap.bids
~~~~~~~~~~~~~
BIDS-style metadata handling.

  • StudyMetadata  dataclass for participant / session / task / timepoint
  • _sanitise_bids_label  strips illegal characters from BIDS labels
"""

import os
import re
import datetime
from dataclasses import dataclass, field, asdict

TOOL_VERSION = "1.4.8"


@dataclass
class StudyMetadata:
    """Holds BIDS-style metadata for a single stage-1 processing run."""
    participant_id: str = ""       # e.g. "sub-JD001"
    session:        str = "ses-01"
    task:           str = ""       # e.g. "fatigue"  (optional)
    timepoint:      str = ""       # e.g. "pre" / "post"  (optional)
    limb:           str = ""       # e.g. "left" / "right"  (optional)
    measure:        str = ""       # e.g. "CSE" / "SICI" / "ICF"  (optional)
    acq:            str = ""       # e.g. "cse-cond30" — acquisition/condition label
    run:            str = ""       # e.g. "01" — index within a multi-file session

    def bids_prefix(self) -> str:
        """
        Build the filename prefix from active fields.
        e.g.  sub-JD001_ses-01_task-fatigue_tp-pre_run-01
        Fields that are blank are omitted.

        ``run`` is the BIDS entity for several acquisitions that belong to one
        session — e.g. a 600-pulse protocol saved as six files of 100 trials.
        It is what keeps their derivatives distinct on disk and their rows
        distinct in the Stage 2 table; without it they collapse onto a single
        (participant_id, session) key and all but one are lost.
        """
        parts = [self.participant_id]
        if self.session:
            parts.append(self.session)
        if self.limb:
            parts.append(f"limb-{self.limb}")
        if self.task:
            parts.append(f"task-{self.task}")
        if self.timepoint:
            parts.append(f"tp-{self.timepoint}")
        if self.measure:
            parts.append(f"measure-{self.measure}")
        if self.acq:
            parts.append(f"acq-{self.acq}")
        if self.run:
            parts.append(f"run-{self.run}")
        return "_".join(p for p in parts if p)

    def sub_ses_path(self) -> str:
        """
        Return the relative sub-XX/ses-XX subfolder path for derivatives.
        e.g.  sub-JD001/ses-01
        """
        sub = self.participant_id or "sub-unknown"
        ses = self.session        or "ses-01"
        return os.path.join(sub, ses)

    def to_sidecar(self, source_file: str, filter_settings: dict,
                   event_delay_ms: dict = None,
                   event_delay_source: dict = None,
                   event_sources: list = None) -> dict:
        """Return a dict ready to be serialised as a JSON sidecar.

        ``event_delay_ms`` records any correction applied between the file's
        event markers and the actual stimulus, per stimulus type, and
        ``event_delay_source`` says whether each was measured from the stimulus
        artefact or typed in.

        A delay shifts every latency in the file, so it has to be recorded --
        and the distinction between a measured and a typed value matters when
        someone else reads the derivative and asks where the number came from.
        Both are written even when empty, so their absence in a sidecar means
        "this version did not support delays" rather than "no delay was set".

        ``event_sources`` records how the stimuli were identified. A run whose
        events came from a 2.5 V crossing on channel 5 produces different
        trials from one that read the file's comments, and nothing in the
        outputs shows which unless it is written here -- so the derivative
        would not be reproducible from itself, and a methods section could not
        be written from it. An empty list is the ordinary answer and means the
        file's own markers were used.
        """
        d = asdict(self)
        d["source_file"]     = os.path.basename(source_file)
        d["date_processed"]  = datetime.date.today().isoformat()
        d["tool_version"]    = TOOL_VERSION
        d["filter_settings"] = filter_settings
        d["event_delay_ms"]     = dict(event_delay_ms or {})
        d["event_delay_source"] = dict(event_delay_source or {})
        d["event_sources"]      = list(event_sources or [])
        return d


def _sanitise_bids_label(text: str) -> str:
    """
    Strip characters that are illegal in BIDS labels / filenames.
    Keeps alphanumerics, hyphens and underscores; collapses spaces to nothing.
    """
    text = text.strip()
    text = re.sub(r"[^\w\-]", "", text)
    return text or "unknown"


# ── Where a recording's derivatives live ─────────────────────────────────────
#
# These live here rather than in app.py because things that are NOT the GUI
# need them: the converter has to find a recording's session, and so do the
# tests. app.py imports pywt, matplotlib and Tk, so importing it to compute a
# filename drags the whole application in -- which is exactly what broke CI,
# where the optional analysis dependencies are not installed. A path rule
# should not require a working display.

def make_bids_prefix(meta_prefix: str, file_stem: str) -> str:
    """Build a unique, clean BIDS prefix from metadata and source file stem.

    Strategy
    --------
    1. No metadata → use file stem as-is.
    2. File stem is a substring of the metadata prefix → use prefix as-is.
    3. Strip universally redundant tokens from the stem:
         - sub-XX / ses-XX  (always encoded in the directory path)
         - bare noise words: "session", "data", "raw"
         - "emg" when the stem is BIDS-originated (starts with "sub-")
    4. Keep only tokens not already present in the metadata prefix
       (token-level exact match — avoids false positives like "01" matching
       inside "ses-01").
    5. No novel tokens → return prefix as-is.
       Novel tokens exist → append them.
    """
    if not meta_prefix:
        return file_stem
    if file_stem in meta_prefix:
        return meta_prefix

    _is_bids_stem = bool(re.match(r"^sub-", file_stem, re.I))
    _NOISE = {"session", "data", "raw"}
    if _is_bids_stem:
        _NOISE.add("emg")

    meta_tokens = set(meta_prefix.split("_"))
    stem_tokens = [t for t in file_stem.split("_")
                   if not re.match(r"^(sub|ses)-", t, re.I)
                   and t.lower() not in _NOISE]

    novel = [t for t in stem_tokens if t not in meta_tokens]

    if not novel:
        return meta_prefix
    return f"{meta_prefix}_{'_'.join(novel)}"


def session_path_for(source_path: str, metadata=None,
                     derivatives_root: str = "",
                     prefix_override: str = "",
                     legacy: bool = False) -> str:
    """Where a recording's session JSON lives.

        <study>/derivatives/mep-cmap-analyser/<sub>/<ses>/emg/<bids_prefix>_session.json

    or, with ``legacy=True``, where versions before 1.4.8 put it:

        <study>/derivatives/<sub>/<ses>/<bids_prefix>_session.json

    The study folder is the configured derivatives root, falling back to the
    source file's own folder when none is configured.

    Derivatives rather than beside the recording, because a session is
    something the tool produced; raw data is what the amplifier and the
    stimulator wrote and is better left as they wrote it. The autosave and Save
    Session used to disagree about this -- one wrote a BIDS-named file under
    derivatives, the other opened a dialogue beside the raw data -- so a
    recording could carry two sessions that knew nothing of each other, and
    whichever the analyst happened to pick on the way back in was the one that
    won.

    A FUNCTION OF ITS ARGUMENTS, not of the app. The app method reads the open
    file's metadata off self, which answers only for the file currently loaded;
    anything working over a list of recordings has none of them loaded. Two
    builders would drift, and the one that drifted would delete or fail to find
    files silently.
    """
    if not source_path:
        return ""
    bids_prefix = make_bids_prefix(
        metadata.bids_prefix() if metadata else "",
        os.path.splitext(os.path.basename(source_path))[0])
    # A caller that already knows where the session lives says so.
    #
    # Only the batch re-runner passes this, and it passes the name of the
    # session file it restored. The naming rule above has changed between
    # versions, so recomputing sends a re-run's autosave to a NEW file beside
    # the one it read: the study it was built against went from 135 sessions
    # to 161 in half a run. The queue still points at the originals, so a
    # second pass would restore stale state and silently undo the first.
    if prefix_override:
        bids_prefix = str(prefix_override)
    source_dir = os.path.dirname(source_path)
    deriv_root = derivatives_root or source_dir
    save_dir = (legacy_session_output_dir(deriv_root, metadata) if legacy
                else session_output_dir(deriv_root, metadata))
    return os.path.join(save_dir, f"{bids_prefix}_session.json")


# ── The derivatives layout ───────────────────────────────────────────────────
#
#   <study>/derivatives/mep-cmap-analyser/dataset_description.json
#   <study>/derivatives/mep-cmap-analyser/sub-XX/ses-YY/emg/
#       <prefix>_session.json, trials/, results/, figures/, ...
#
# BIDS derivatives mirror the raw layout -- sub, ses, then the datatype folder
# -- and sit in a folder named for the pipeline that made them, described by a
# dataset_description.json of DatasetType "derivative". EMG has been a BIDS
# datatype since 1.11.0.
#
# Before v1.4.8 everything went straight into <study>/derivatives/sub-XX/ses-YY/.
# That LEGACY layout is still read -- sessions restore from it and group-level
# scans find it -- and is moved aside, not deleted, when a recording is re-run
# or a study is migrated (mep_cmap.layout_migration).

PIPELINE_NAME = "mep-cmap-analyser"
DATATYPE = "emg"
BIDS_VERSION = "1.11.0"
CODE_URL = "https://github.com/jandrushko/mep-cmap-analyser"


def derivatives_dir(root: str) -> str:
    """The study's ``derivatives`` folder, from whatever root was configured.

    The analyst may point the tool at the study folder, at its
    ``derivatives`` folder, or at this pipeline's folder inside it. All three
    mean the same place; appending blindly is what produced
    ``derivatives/derivatives/`` in the past.
    """
    root = os.path.normpath(root or ".")
    base = os.path.basename(root).lower()
    if base == PIPELINE_NAME:
        return os.path.dirname(root)
    if base == "derivatives":
        return root
    return os.path.join(root, "derivatives")


def pipeline_dir(root: str) -> str:
    """``<study>/derivatives/mep-cmap-analyser``."""
    return os.path.join(derivatives_dir(root), PIPELINE_NAME)


def _sub_ses(metadata) -> str:
    return (metadata.sub_ses_path() if metadata
            else os.path.join("sub-unknown", "ses-01"))


def session_output_dir(root: str, metadata=None) -> str:
    """Where a recording's outputs and session JSON go: ``.../sub/ses/emg``."""
    return os.path.join(pipeline_dir(root), _sub_ses(metadata), DATATYPE)


def legacy_session_output_dir(root: str, metadata=None) -> str:
    """Where versions before 1.4.8 wrote them: ``derivatives/sub/ses``."""
    return os.path.join(derivatives_dir(root), _sub_ses(metadata))


def is_in_pipeline_dir(path: str) -> bool:
    """True if ``path`` sits inside a ``derivatives/mep-cmap-analyser`` tree."""
    parts = [p.lower() for p in os.path.normpath(os.path.abspath(path)).split(os.sep)]
    return any(a == "derivatives" and b == PIPELINE_NAME
               for a, b in zip(parts, parts[1:]))


def study_root_for(path: str) -> str:
    """The study folder a derivatives file belongs to.

    The parent of the nearest enclosing ``derivatives`` folder. Session JSONs
    store their recording's path relative to this so a study can move between
    computers; both the save and the load used to count three folders up from
    the session, which was right for the legacy layout only and would have
    resolved every recording against the wrong folder once sessions moved
    into ``mep-cmap-analyser/.../emg``. Falls back to three folders up when
    no ``derivatives`` folder encloses the path.
    """
    d = os.path.abspath(path)
    # A file (existing, or named with an extension) answers for its folder; a
    # folder may not exist yet when a session is about to be saved into it.
    if os.path.isfile(d) or (not os.path.isdir(d) and os.path.splitext(d)[1]):
        d = os.path.dirname(d)
    probe = d
    while True:
        if os.path.basename(probe).lower() == "derivatives":
            return os.path.dirname(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return os.path.dirname(os.path.dirname(os.path.dirname(d)))


def write_derivatives_description(root: str, version: str = "") -> str:
    """Create ``dataset_description.json`` for this pipeline's derivatives.

    Required of every BIDS derivative dataset. Written once and then left
    alone: an analyst who has added a name, authors or a licence has made it
    theirs. Returns the path; never raises, because a missing description is
    a validation warning and not a reason to lose an analysis.
    """
    import json
    try:
        pdir = pipeline_dir(root)
        os.makedirs(pdir, exist_ok=True)
        path = os.path.join(pdir, "dataset_description.json")
        if os.path.exists(path):
            return path
        desc = {
            "Name": "MEP-CMAP Analyser derivatives",
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": "MEP-CMAP Analyser",
                "Version": version or TOOL_VERSION,
                "CodeURL": CODE_URL,
            }],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(desc, fh, indent=2)
        return path
    except Exception:                       # noqa: BLE001 - never block a run
        return ""
