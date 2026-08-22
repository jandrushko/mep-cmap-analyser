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

TOOL_VERSION = "1.4.5"


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
                     derivatives_root: str = "") -> str:
    """Where a recording's session JSON lives.

        <derivatives_root>/derivatives/<sub>/<ses>/<bids_prefix>_session.json

    falling back to the source file's own folder when no derivatives root is
    configured.

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
    source_dir = os.path.dirname(source_path)
    deriv_root = derivatives_root or source_dir
    sub_ses = (metadata.sub_ses_path() if metadata
               else os.path.join("sub-unknown", "ses-01"))
    # Avoid derivatives/derivatives/ — same fix as in pipeline.py
    if os.path.basename(os.path.normpath(deriv_root)).lower() == "derivatives":
        save_dir = os.path.join(deriv_root, sub_ses)
    else:
        save_dir = os.path.join(deriv_root, "derivatives", sub_ses)
    return os.path.join(save_dir, f"{bids_prefix}_session.json")
