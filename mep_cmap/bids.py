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

TOOL_VERSION = "0.8.8"


@dataclass
class StudyMetadata:
    """Holds BIDS-style metadata for a single stage-1 processing run."""
    participant_id: str = ""       # e.g. "sub-JD001"
    session:        str = "ses-01"
    task:           str = ""       # e.g. "fatigue"  (optional)
    timepoint:      str = ""       # e.g. "pre" / "post"  (optional)
    limb:           str = ""       # e.g. "left" / "right"  (optional)
    measure:        str = ""       # e.g. "CSE" / "SICI" / "ICF"  (optional)

    def bids_prefix(self) -> str:
        """
        Build the filename prefix from active fields.
        e.g.  sub-JD001_ses-01_task-fatigue_tp-pre
        Fields that are blank are omitted.
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
        return "_".join(p for p in parts if p)

    def sub_ses_path(self) -> str:
        """
        Return the relative sub-XX/ses-XX subfolder path for derivatives.
        e.g.  sub-JD001/ses-01
        """
        sub = self.participant_id or "sub-unknown"
        ses = self.session        or "ses-01"
        return os.path.join(sub, ses)

    def to_sidecar(self, source_file: str, filter_settings: dict) -> dict:
        """Return a dict ready to be serialised as a JSON sidecar."""
        d = asdict(self)
        d["source_file"]     = os.path.basename(source_file)
        d["date_processed"]  = datetime.date.today().isoformat()
        d["tool_version"]    = TOOL_VERSION
        d["filter_settings"] = filter_settings
        return d


def _sanitise_bids_label(text: str) -> str:
    """
    Strip characters that are illegal in BIDS labels / filenames.
    Keeps alphanumerics, hyphens and underscores; collapses spaces to nothing.
    """
    text = text.strip()
    text = re.sub(r"[^\w\-]", "", text)
    return text or "unknown"
