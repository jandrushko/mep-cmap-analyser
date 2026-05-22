"""
app.py integration patch — Format Wizard for generic TSV files
==============================================================

Three changes needed.  Line numbers are approximate; match by context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGE 1 — Add needs_wizard import  (~line 59–60)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND:
    from .io import (list_waveform_channels, extract_emg_waveform_and_fs,
                     extract_stim_times, detect_format)

REPLACE WITH:
    from .io import (list_waveform_channels, extract_emg_waveform_and_fs,
                     extract_stim_times, detect_format, needs_wizard)
    from .format_wizard import FormatWizard


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGE 2 — Intercept generic_tsv before scanning stim events
           Insert immediately after `_fmt = detect_format(fpath)`
           (~line 3014)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND:
        # ── Detect file format and scan accordingly ───────────────────────────
        _fmt = detect_format(fpath)

        if _fmt == 'labchart':

REPLACE WITH:
        # ── Detect file format and scan accordingly ───────────────────────────
        _fmt = detect_format(fpath)

        # ── Generic TSV: launch Format Wizard if no sidecar config yet ────────
        if _fmt == 'generic_tsv' and needs_wizard(fpath):
            self.log("🔧 Generic TSV detected — launching Format Wizard…")

            def _on_wizard_complete(cfg, _fpath=fpath, _auto=auto_run):
                if cfg is None:
                    self.log("⚠️  Format Wizard cancelled — file not loaded.")
                    return
                self.log(
                    f"✅ Format Wizard complete — "
                    f"{len([c for c in cfg['channels'] if c['role'] != 'ignore'])} "
                    f"signal(s) defined, fs={cfg['fs']} Hz"
                )
                # Re-enter _browse_file_path now that the sidecar exists
                self._browse_file_path(_fpath, auto_run=_auto)

            FormatWizard(self.root, fpath, on_complete=_on_wizard_complete)
            return   # wizard is modal; _browse_file_path re-enters on close

        if _fmt == 'labchart':


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGE 3 — Handle generic_tsv stim events in the format scan block
           In the same `if _fmt == 'labchart': ... else:` block,
           extend the labchart branch into a three-way branch.
           (~line 3016–3047)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND:
        if _fmt == 'labchart':
            # LabChart: stim times come from the analogue stim channel.
            # No DigMark channels — marker_choice is unused for LabChart.
            # Set to 'A' so extract_stim_times gets label 'A' (not 'L').
            self.marker_choice.set('A')
            self.log("📋 LabChart format detected — stim times from analogue trigger channel")
            # stim_events populated later via extract_stim_times in pipeline
        else:
            # Spike2: scan for DigMark channels and stim type timestamps

REPLACE WITH:
        if _fmt == 'labchart':
            # LabChart: stim times come from the analogue stim channel.
            # No DigMark channels — marker_choice is unused for LabChart.
            # Set to 'A' so extract_stim_times gets label 'A' (not 'L').
            self.marker_choice.set('A')
            self.log("📋 LabChart format detected — stim times from analogue trigger channel")
            # stim_events populated later via extract_stim_times in pipeline

        elif _fmt == 'generic_tsv':
            # Generic TSV: wizard has already run (we returned above if it hadn't).
            # Stim times are extracted from the designated Stim/Trigger column.
            self.marker_choice.set('A')
            self.log("📋 Generic TSV format — stim times from Stim/Trigger channel")
            # stim_events populated later via extract_stim_times in pipeline

        else:
            # Spike2: scan for DigMark channels and stim type timestamps
"""
