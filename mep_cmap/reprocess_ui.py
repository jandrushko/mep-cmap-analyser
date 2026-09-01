"""
mep_cmap.reprocess_ui
~~~~~~~~~~~~~~~~~~~~~
The batch re-run, as the analyst drives it.

Everything here is Tk and study-level bookkeeping. The decisions live
elsewhere and are tested there: mep_cmap.reprocess decides whether a file can
be re-run, mep_cmap.reprocess_runner sequences the run, and
mep_cmap.reprocess_report says what changed. This module only asks the
questions, drives the loop, and writes the report out.

Why a re-run is not just "Run All again"
----------------------------------------
Run All walks the queue with a person watching, one dialogue at a time. This
walks it with nobody watching, which means every question has to have been
answered in advance and every answer has to come from the saved session rather
than from whatever the previous file happened to leave behind. It also has to
write its results back over the ones it read, which is not where today's
naming rules would put them for most files.
"""

from __future__ import annotations

import csv
import json
import os
import time
import tkinter as tk
from tkinter import messagebox

from . import archive_superseded as AR
from . import reprocess as R
from . import reprocess_currency as CU
from . import reprocess_report as RP
from . import reprocess_runner as RR
from .dataset_session import STATUS_COMPLETE, STATUS_NEEDS_REVIEW

#: Passed to the runner so it stays free of the dataset import.
_STATUSES = {"complete": STATUS_COMPLETE, "needs_review": STATUS_NEEDS_REVIEW}

#: How long one file may take before the run gives up on it and moves on. A
#: hung file must not cost the other hundred and twenty.
_FILE_TIMEOUT_S = 3600.0


class ReprocessMixin:
    """Batch re-run of already-analysed recordings, from their saved sessions."""

    # ── entry point ──────────────────────────────────────────────────────────

    def reprocess_study(self):
        """Plan, ask, confirm, run. The one entry point for a batch re-run.

        Scope is chosen inside the confirmation rather than by picking between
        two menu items, because the two are the same job and the difference
        between them is a reason, not a name. "Everything" and "only what is
        behind" both re-run files from their saved sessions; which you want
        depends on whether the thing that changed shows up as a column.
        """
        self._reprocess(None)

    def _reprocess_currency_records(self):
        """(key, verdict, missing) for every queue entry with a session."""
        from .pipeline import LAT_COLS      # the live schema, not a copy here
        out = []
        for fe in self._dataset.files:
            if not (getattr(fe, "derivatives_json", "") or ""):
                continue
            header = self._reprocess_trials_header(fe)
            verdict = CU.classify(header, LAT_COLS)
            out.append((fe.id, verdict,
                        CU.missing_columns(header or [], LAT_COLS)
                        if verdict == CU.BEHIND else []))
        return out

    def _reprocess_trials_header(self, fe):
        """The column names of this entry's current trial file, or None."""
        path = self._reprocess_trials_path(fe)
        if not path:
            return None
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                return next(csv.reader(fh))
        except Exception:
            return None

    def reprocess_selected(self):
        """The same, for the files selected in the queue.

        Exists so a re-run can be proved on ONE recording before it is turned
        loose on a hundred and twenty. The two old runs already sitting in the
        study this was built against differ by far more than a version bump --
        different labels, different onsets, a 50% difference in amplitude --
        and nothing recorded which settings produced which. Running one file
        and reading its change report is the only way to tell a genuine
        version difference from a setup that was never restored, and it costs
        a minute rather than an afternoon.
        """
        ids = set(self._queue_selected_ids() or ())
        if not ids:
            messagebox.showinfo(
                "Nothing selected",
                "Select one or more files in the queue first.",
                parent=self.root)
            return
        self._reprocess(ids)

    def _reprocess(self, only_ids):
        if self._dataset is None:
            messagebox.showwarning(
                "No study open",
                "Open a study folder first. The re-run works from the saved "
                "session beside each recording.", parent=self.root)
            return
        if getattr(self, "_analysis_running", False):
            messagebox.showwarning(
                "Analysis in progress",
                "Wait for the current analysis to finish.", parent=self.root)
            return

        plans = self._reprocess_plan_all(only_ids)
        if not plans:
            messagebox.showinfo(
                "Nothing to re-run",
                "No saved sessions were found for the files in this study.",
                parent=self.root)
            return

        plans = self._reprocess_answer_conflicts(plans)
        runnable = [(k, p) for k, p in plans.items()
                    if p["status"] in (R.OK, R.REPAIRED)]
        runnable, archive_after = self._reprocess_confirm(plans, runnable)
        if not runnable:
            # Archiving on its own is a valid answer to the dialogue: a study
            # already re-run needs tidying before a merge, and that is not a
            # reason to re-run it a second time.
            if archive_after:
                self._reprocess_archive_outputs()
            return
        self._reprocess_archive_after = archive_after
        self._reprocess_execute(runnable)

    # ── planning ─────────────────────────────────────────────────────────────

    def _reprocess_plan_all(self, only_ids=None):
        """{file_id: plan} for every queue entry that has a saved session.

        ``only_ids`` narrows the RESULT, not the analysis. Every session in
        the study is still planned and the shared-recording check still runs
        over all of them, because that check is only meaningful across the
        whole study: selecting one of six sessions that claim the same
        recording would otherwise hide the very thing that makes it unsafe.
        """
        study_root = self._reprocess_study_root()
        recordings = R.index_recordings(self._reprocess_all_recordings(study_root))

        plans, claims = {}, []
        for fe in self._dataset.files:
            sess = self._reprocess_read_session(fe)
            if sess is None:
                continue
            stored = sess.get("file_path") or fe.path
            path, note = R.locate_recording(stored, study_root, recordings)
            chans = self._reprocess_channels_of(path) if path else []
            # FileEntry's identifier is `id`. Keyed by it throughout, because
            # _dataset.get_file() matches on it and the paths move.
            plans[fe.id] = R.plan_file(
                sess, chans, path=path, path_note=note,
                needs_unit=self._reprocess_needs_unit(path))
            plans[fe.id]["path"] = path
            plans[fe.id]["session"] = sess
            plans[fe.id]["channels"] = chans
            if path:
                claims.append((fe.id, path))

        # A pass over the whole study, not over one session.
        #
        # Each of these sessions resolves perfectly on its own, which is why
        # plan_file cannot catch them: the disagreement is only visible when
        # several are seen together. In the study this was built against six
        # sessions carrying four different task labels all name the same rest
        # recording, and running them would file that one recording under
        # three names it is not.
        for _name, keys in R.sessions_sharing_a_recording(claims).items():
            for key in keys:
                plans[key]["status"] = R.SKIP
                plans[key].setdefault("notes", []).append(
                    R.shared_recording_note(plans[key]["path"], keys))

        if only_ids is not None:
            plans = {k: v for k, v in plans.items() if k in only_ids}
        return plans

    def _reprocess_study_root(self):
        raw = (self._rawdata_path.get()
               if hasattr(self, "_rawdata_path") else "")
        if raw:
            return os.path.dirname(os.path.normpath(raw)) or raw
        return os.path.dirname(self._dataset.json_path or "") or "."

    def _reprocess_all_recordings(self, study_root):
        """Every readable recording under the study, for the moved-file search."""
        out = []
        raw = os.path.join(study_root, "rawdata")
        for base in (raw if os.path.isdir(raw) else study_root,):
            for dirpath, _dirs, files in os.walk(base):
                for fn in files:
                    if not fn.lower().endswith((".json", ".tsv", ".md")):
                        out.append(os.path.join(dirpath, fn))
        return out

    def _reprocess_read_session(self, fe):
        path = getattr(fe, "derivatives_json", "") or ""
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not read the session for "
                     f"{fe.basename}: {exc}")
            return None

    def _reprocess_channels_of(self, path):
        try:
            from .io import list_waveform_channels
            return list(list_waveform_channels(path) or [])
        except Exception:
            # Reported as an empty channel list, which the planner turns into
            # a refusal with its own note. Guessing a channel layout here
            # would be the one thing the planner exists to prevent.
            return []

    @staticmethod
    def _reprocess_needs_unit(path):
        """Whether this format asks the analyst to confirm an amplitude unit."""
        try:
            from .io import detect_format
            return detect_format(path) == "epoched_mat"
        except Exception:
            return False

    # ── the conflict questions ───────────────────────────────────────────────

    def _reprocess_answer_conflicts(self, plans):
        """Put each DISTINCT channel disagreement once, and apply the answer.

        Grouped deliberately. A study analysed in one sitting produces the
        same disagreement in every file -- 77 of them here -- and an interface
        that asks 77 identical questions gets answered by clicking through it.
        """
        conflicted = [(k, p) for k, p in plans.items()
                      if p["status"] == R.CONFLICT]
        if not conflicted:
            return plans

        groups = R.group_conflicts(
            [(k, p["session"], p["channels"]) for k, p in conflicted])
        for signature, keys in groups.items():
            described = R.describe_conflict(signature)
            if described is None:
                continue
            answer = self._reprocess_ask_conflict(described, len(keys))
            if answer == R.UNRESOLVED:
                continue
            for key in keys:
                plan = plans[key]
                resolved = R.apply_conflict_answer(
                    plan["session"], plan["channels"], answer)
                if resolved is None:
                    continue
                plan["resolved"] = resolved
                plan["status"] = R.REPAIRED
                plan.setdefault("notes", []).append(
                    f"channel taken from the saved "
                    f"{'index' if answer == R.BY_INDEX else 'name'} "
                    f"({resolved['channel_choice']}), as answered for this "
                    f"group of {len(keys)} file(s).")
        return plans

    def _reprocess_ask_conflict(self, described, n_files):
        """One disagreement, as a choice between the two readings."""
        win = tk.Toplevel(self.root)
        win.title("Which channel was analysed?")
        win.transient(self.root)
        win.resizable(False, False)

        name, by_name = described["name_points_at"]
        chan_by_index, idx = described["index_points_at"]

        tk.Label(win, justify="left", wraplength=560, text=(
            f"{n_files} saved session(s) disagree with themselves about which "
            f"channel was analysed. The name they record and the position "
            f"they record point at different channels, and the session does "
            f"not say which one the analysis actually used.")).pack(
            anchor="w", padx=14, pady=(12, 8))

        choice = tk.StringVar(value=R.UNRESOLVED)
        box = tk.LabelFrame(win, text="The analysis ran on")
        box.pack(fill="x", padx=14, pady=(0, 10))
        tk.Radiobutton(
            box, variable=choice, value=R.BY_INDEX, justify="left",
            text=(f"{chan_by_index}\n"
                  f"    the channel at the saved position ({idx})")).pack(
            anchor="w", padx=8, pady=3)
        tk.Radiobutton(
            box, variable=choice, value=R.BY_NAME, justify="left",
            text=(f"{name}\n"
                  f"    the channel with the saved name"
                  + (f" (position {by_name})" if by_name is not None else ""))
        ).pack(anchor="w", padx=8, pady=3)
        tk.Radiobutton(
            box, variable=choice, value=R.UNRESOLVED, justify="left",
            text=("I do not know\n"
                  "    leave these files alone and mark them for review")
        ).pack(anchor="w", padx=8, pady=3)

        tk.Label(win, fg="#8a6d3b", justify="left", wraplength=560, text=(
            "If this is wrong, the re-run produces a complete and plausible "
            "analysis of the wrong signal. A saved figure from the original "
            "analysis will usually settle it.")).pack(
            anchor="w", padx=14, pady=(0, 10))

        # Anchored to the bottom for the same reason as the dry run's buttons.
        tk.Button(win, text=f"Apply to all {n_files} file(s)", width=22,
                  command=win.destroy).pack(side="bottom", pady=(0, 12))
        win.update_idletasks()
        max_h = int(self.root.winfo_screenheight() * 0.8)
        if win.winfo_reqheight() > max_h:
            win.geometry(f"{win.winfo_reqwidth()}x{max_h}")
        win.grab_set()
        self.root.wait_window(win)
        return choice.get()

    # ── the dry run ──────────────────────────────────────────────────────────

    def _reprocess_confirm(self, plans, runnable):
        """Show what would happen and let the analyst choose the scope.

        Returns ``(list to run, archive afterwards)``. The list is empty if
        cancelled, and also when the analyst chose to archive without
        re-running -- so the flag has to be read either way. Nothing has been
        written at this point.

        A fixed-size window rather than a messagebox. A messagebox grows to
        fit its text, and the refusal list grows with the study -- on a real
        study it grew until the buttons were off the bottom of the screen and
        the dialogue could not be answered at all. The list is the part that
        varies, so the list is the part that scrolls; the summary, the scope
        choice and the buttons stay put.
        """
        counts = R.summarise(list(plans.values()))
        refused = [(k, p) for k, p in plans.items() if p["status"] == R.SKIP]

        # Every refusal goes to the log in full, whatever the window shows.
        for key, plan in refused:
            self.log(f"   ⏭  {self._reprocess_name(key)}: "
                     + " ".join(plan.get("notes") or []))

        # Which of the runnable files are behind the current outputs.
        #
        # Shown even when the analyst goes on to re-run everything: knowing
        # that 131 of 262 outputs predate the current schema is worth having
        # in front of you either way, and it was previously only visible if
        # you happened to pick the narrower of two menu items.
        try:
            records = self._reprocess_currency_records()
        except Exception:
            records = []
        behind = set(CU.keys_needing_rerun(records))
        cur = CU.summarise(records)
        outdated = [(k, p) for k, p in runnable if k in behind]

        win = tk.Toplevel(self.root)
        win.title("Re-run this study")
        win.transient(self.root)
        result = {"run": [], "archive": False}

        scope = tk.StringVar(value="outdated" if outdated else "all")
        archive_var = tk.BooleanVar(value=False)

        # Counted now so the option can say how much it would move, but
        # recomputed after the run: the run creates the current copies, and a
        # file that is the only one there is before the run is superseded
        # after it.
        arc_plan = self._reprocess_archive_plan()
        arc_n = arc_plan.n_items if arc_plan else 0

        # Buttons first and anchored to the bottom, so they keep their space
        # no matter how long the list above is. This is the bug: packed after
        # an expanding widget, they are the first thing squeezed off-screen.
        bar = tk.Frame(win)
        bar.pack(side="bottom", fill="x", padx=14, pady=(6, 12))

        def _go():
            if scope.get() == "archive":
                result["run"], result["archive"] = [], True
            else:
                result["run"] = (outdated if scope.get() == "outdated"
                                 else runnable)
                result["archive"] = bool(archive_var.get())
            win.destroy()

        tk.Button(bar, text="Cancel", width=12,
                  command=win.destroy).pack(side="right")
        # Offered when there is anything to do at all, which includes a study
        # that needs no re-run but still carries superseded outputs.
        if runnable or arc_n:
            tk.Button(bar, text="Go", width=16,
                      command=_go).pack(side="right", padx=(0, 8))

        head = (f"{len(runnable)} file(s) can be re-run from their saved "
                f"sessions.\n\n"
                f"    ready, nothing to correct        "
                f"{counts.get(R.OK, 0)}\n"
                f"    ready, recording located again   "
                f"{counts.get(R.REPAIRED, 0)}\n"
                f"    left alone, unresolved conflict  "
                f"{counts.get(R.CONFLICT, 0)}\n"
                f"    left alone, cannot be planned    "
                f"{counts.get(R.SKIP, 0)}\n\n"
                f"Each file keeps its crop, conditions, channel setup and "
                f"manual marker edits, and writes back over its existing "
                f"results rather than beside them.\n\n"
                f"Numbers that move are expected: the cSP/MEP ratio was "
                f"corrected in v1.4. Any file whose values change is marked "
                f"for review rather than complete, and the change report "
                f"lists what moved.")
        tk.Label(win, justify="left", wraplength=560, text=head).pack(
            anchor="w", padx=14, pady=(12, 8))

        if runnable or arc_n:
            box = tk.LabelFrame(win, text="How much to re-run")
            box.pack(fill="x", padx=14, pady=(0, 8))
            tk.Radiobutton(
                box, variable=scope, value="outdated", justify="left",
                state=("normal" if outdated else "disabled"),
                text=(f"Only what is out of date  —  {len(outdated)} file(s)\n"
                      f"    Their outputs predate columns the current version "
                      f"produces.\n"
                      f"    Resumes a run that stopped part way without "
                      f"redoing finished work.")).pack(anchor="w", padx=8,
                                                       pady=3)
            tk.Radiobutton(
                box, variable=scope, value="all", justify="left",
                state=("normal" if runnable else "disabled"),
                text=(f"Everything  —  {len(runnable)} file(s)\n"
                      f"    Use after a change to how a value is COMPUTED. "
                      f"Such a change adds\n"
                      f"    no column, so every file still looks up to date "
                      f"while none of them is.")).pack(anchor="w", padx=8,
                                                       pady=3)
            tk.Radiobutton(
                box, variable=scope, value="archive", justify="left",
                state=("normal" if arc_n else "disabled"),
                text=(f"Archive superseded outputs only  —  {arc_n} file(s), "
                      f"no re-run\n"
                      f"    For a study already re-run. Group merges read "
                      f"every result file under\n"
                      f"    the study, so a session left with two copies "
                      f"counts twice.")).pack(anchor="w", padx=8, pady=3)
            tk.Label(box, fg="#666", justify="left", wraplength=540,
                     text=(f"Up to date: {cur.get(CU.CURRENT, 0)}   "
                           f"behind: {cur.get(CU.BEHIND, 0)}   "
                           f"no trial file: {cur.get(CU.ABSENT, 0)}. "
                           f"This counts columns, not correctness: a file can "
                           f"carry every column and still hold a bad value.")
                     ).pack(anchor="w", padx=8, pady=(0, 6))
            # Off by default. It moves files, and the summary it writes is
            # worth reading once before it becomes a side effect of pressing
            # Go.
            tk.Checkbutton(
                box, variable=archive_var, justify="left",
                state=("normal" if runnable else "disabled"),
                text=("Archive superseded outputs when the run finishes\n"
                      "    Counted again afterwards, since the run itself "
                      "creates the current copies.")).pack(
                anchor="w", padx=8, pady=(0, 6))

        if refused:
            tk.Label(win, justify="left", anchor="w",
                     text=f"Left alone ({len(refused)}), also in the log:"
                     ).pack(anchor="w", padx=14)
            frame = tk.Frame(win)
            frame.pack(fill="both", expand=True, padx=14, pady=(2, 0))
            scroll = tk.Scrollbar(frame)
            scroll.pack(side="right", fill="y")
            text = tk.Text(frame, height=8, width=70, wrap="word",
                           yscrollcommand=scroll.set)
            text.pack(side="left", fill="both", expand=True)
            scroll.config(command=text.yview)
            for key, plan in refused:
                text.insert("end", f"{self._reprocess_name(key)}\n"
                                   + "    "
                                   + " ".join(plan.get("notes") or [])
                                   + "\n\n")
            text.config(state="disabled")

        # Never taller than the screen allows, however long the list is.
        win.update_idletasks()
        max_h = int(self.root.winfo_screenheight() * 0.8)
        win.geometry(f"620x{min(win.winfo_reqheight(), max_h)}")
        win.grab_set()
        self.root.wait_window(win)
        return result["run"], result["archive"]

    def _reprocess_name(self, file_id):
        fe = self._dataset.get_file(file_id)
        return fe.basename if fe is not None else str(file_id)

    # ── archiving what the run replaced ──────────────────────────────────────

    def _reprocess_derivatives_root(self):
        """The folder holding the study's per-session outputs.

        The same one the change report is written to, so the archive and the
        report agree about which study they describe.
        """
        return (self.derivatives_path.get().strip()
                or os.path.dirname(self._dataset.json_path or "") or ".")

    def _reprocess_archive_plan(self):
        """What is superseded right now, or None if it cannot be worked out.

        Never raises: a study whose folders are arranged unexpectedly should
        cost the dialogue its archive count, not stop a re-run from being
        offered at all.
        """
        try:
            return AR.plan_archive(self._reprocess_derivatives_root())
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not check for superseded outputs: {exc}")
            return None

    def _reprocess_archive_outputs(self):
        """Move superseded outputs aside and say what happened.

        Planned again here rather than reusing the count from the dialogue,
        because the run that has just finished is what made the current copies
        current. A file that was the only one there is beforehand -- and so
        deliberately left alone -- is superseded once its session has been
        re-run.
        """
        plan = self._reprocess_archive_plan()
        if plan is None:
            return
        if not plan.items:
            self.log("   Nothing superseded to archive.")
            return

        root = AR.default_archive_root(self._reprocess_derivatives_root())
        self.log("\n📦 Archiving superseded outputs.")
        try:
            res = AR.apply_archive(plan, root, log_callback=self.log)
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not archive: {exc}")
            messagebox.showwarning(
                "Archive failed",
                f"The superseded outputs were left where they are.\n\n{exc}",
                parent=self.root)
            return

        by = plan.by_reason()
        summary = (
            f"Archived {res['moved']} superseded file(s).\n\n"
            f"    replaced by the current layout  "
            f"{by.get(AR.REASON_FLAT, 0)}\n"
            f"    written under an old name       "
            f"{by.get(AR.REASON_UNCLAIMED, 0)}\n"
            f"    kept in place, still current    {len(plan.current)}\n"
            f"\nMoved to: {res['archive_root']}"
            f"\nManifest: {res['manifest']}")
        if res["failed"]:
            summary += (f"\n\n{len(res['failed'])} file(s) could not be moved "
                        f"and are still in place. See the log.")
        if plan.stale:
            # Named, not just counted: these are the sessions a group merge
            # would read months-old numbers from, and nothing else in the
            # tool will say so.
            summary += (f"\n\n{len(plan.stale)} session(s) have no current "
                        f"output and were left alone. They still need "
                        f"re-running; they are named in the log.")
            for st in plan.stale:
                self.log(f"   ⚠️  No current output: {st.prefix}")
        self.log("\n" + summary)
        messagebox.showinfo("Archive finished", summary, parent=self.root)

    # ── the run ──────────────────────────────────────────────────────────────

    def _reprocess_execute(self, runnable):
        self._reprocess_stop = False
        self._reprocess_before = {}
        self._reprocess_findings = {}
        self._reprocess_done = 0
        self._reprocess_total = len(runnable)
        self.log(f"\n🔁 Re-running {len(runnable)} file(s) from their saved "
                 f"sessions.")

        # A way out. Without one the only exit from a run of a hundred files
        # is force-quitting the application, which is how the first real run
        # ended: four files had already been analysed and the report that
        # would have said what changed in them was never written, because it
        # was only written at the end.
        win = self._reprocess_progress_window(len(runnable))
        try:
            results = RR.run_batch(
                runnable,
                load=self._reprocess_load,
                analyse=self._reprocess_analyse,
                compare=self._reprocess_compare,
                statuses=_STATUSES,
                log=lambda msg: self.log(f"   {msg}"),
                should_stop=lambda: self._reprocess_stop
                                    or getattr(self, "_closing", False),
            )
        finally:
            try:
                win.destroy()
            except Exception:
                pass
        self._reprocess_finish(results)

    def _reprocess_progress_window(self, total):
        """A small always-visible window carrying the Stop button."""
        win = tk.Toplevel(self.root)
        win.title("Re-running")
        win.transient(self.root)
        win.resizable(False, False)
        self._reprocess_progress_var = tk.StringVar(
            value=f"Starting… 0 of {total}")
        tk.Label(win, textvariable=self._reprocess_progress_var,
                 padx=18, pady=12, width=46, anchor="w",
                 justify="left").pack()
        tk.Button(win, text="Stop after this file", width=22,
                  command=self.stop_reprocess).pack(pady=(0, 12))
        # Closing the window means the same thing as pressing Stop. A window
        # whose close button did nothing would be tried first and trusted.
        win.protocol("WM_DELETE_WINDOW", self.stop_reprocess)
        win.bind("<Escape>", lambda _e: self.stop_reprocess())
        win.update_idletasks()
        return win

    def _reprocess_load(self, file_id, plan):
        """Put one file and its saved session into the application."""
        fe = self._dataset.get_file(file_id)
        if fe is None:
            return False
        self._reprocess_done = getattr(self, "_reprocess_done", 0) + 1
        try:
            self._reprocess_progress_var.set(
                f"{self._reprocess_done} of {self._reprocess_total}\n"
                f"{fe.basename}")
            self.root.update_idletasks()
        except Exception:
            pass
        # Read the trial file BEFORE it is overwritten. There is no other copy
        # once the run has written over it, and without it the comparison has
        # nothing to compare against.
        self._reprocess_before[file_id] = self._reprocess_read_trials(fe)

        # The two settings that make the load unattended. Both are cleared in
        # _reprocess_finish, so an interactive run afterwards is unaffected.
        self._unattended = dict(plan.get("resolved") or {})
        self._unattended.setdefault(
            "emg_unit", (plan.get("session") or {}).get("emg_unit", ""))
        self._bids_prefix_override = self._reprocess_saved_prefix(fe)

        if plan.get("path") and plan["path"] != fe.path:
            fe.path = plan["path"]

        # Nothing may carry over from the file before this one.
        #
        # _reset_state_for_new_file deliberately KEEPS the flat 1a maps, so an
        # analyst configuring a second file inherits the first as a starting
        # point. That is right interactively and wrong here: a session that
        # does not carry window_map would silently analyse this recording with
        # the previous recording's analysis window, and the run would look
        # completely clean. Only 6 of the 135 sessions in this study carry a
        # window_map at all.
        #
        # Cleared BEFORE the load, so the session restores onto empty and
        # whatever it does not carry is genuinely absent rather than
        # inherited.
        for _key in getattr(self, "_chan_settings_keys", ()):
            _cur = getattr(self, _key, None)
            if isinstance(_cur, dict):
                setattr(self, _key, {})
            elif isinstance(_cur, set):
                setattr(self, _key, set())
        self.crop_ranges = None
        self.crop_start = None
        self.crop_end = None

        try:
            self._load_file_entry(fe)
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not load {fe.basename}: {exc}")
            return False
        if not self.file_path.get():
            return False

        self._reprocess_adopt_flat_setup()
        return True

    def _reprocess_adopt_flat_setup(self):
        """Register the restored setup against the channel being analysed.

        131 of the 135 sessions here predate per-channel setup: they store the
        1a table in the flat maps (label_map, latency_map, latency_stim_map,
        colour_map), and _chan_settings is empty. _validate_analysis_setup
        checks _chan_settings, so every one of those files was refused with
        "these channels are selected for analysis but have no setup of their
        own" and the run could not start at all.

        The setup is not missing, only stored in the older shape. This snaps
        the restored flat maps into the per-channel entry, which is what
        _snapshot_chan_settings does and exactly what tab 1a would write if
        the analyst opened it and pressed Confirm without changing anything.

        Safe only because the caller cleared those maps first: without that,
        a map this session does not carry would be the PREVIOUS file's, and
        this would bake one recording's setup into another's results.

        Does nothing when the session already carries a per-channel entry,
        which the four newer sessions do.
        """
        idx = getattr(self, "channel_idx", 0)
        settings = getattr(self, "_chan_settings", None)
        if settings is None:
            return
        if idx in settings:
            # Setup already present, but confirmation is a SEPARATE record and
            # a session can carry one without the other. One file in the study
            # did: chan_settings ['2'], chan_confirmed [], written by a run
            # that was interrupted after the setup was entered and before it
            # was confirmed. The early return skipped the confirming line
            # below, _validate_analysis_setup refused at its second guard, and
            # that file alone failed out of a hundred and ten.
            #
            # Confirming here is not inventing a decision. The setup came from
            # the session, this is a re-run of that session, and the analysis
            # it is reproducing ran with exactly these settings.
            self._reprocess_confirm_channel(idx)
            return
        if not getattr(self, "label_map", None):
            # No setup in either shape. Left alone so the existing guard
            # refuses it rather than running on defaults.
            self.log("   ⚠️  The session carries no stimulus setup.")
            return
        try:
            self._snapshot_chan_settings(idx)
            self._reprocess_confirm_channel(idx)
            self.log("   Setup adopted from the saved session "
                     "(older per-file format).")
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not adopt the saved setup: {exc}")

    def _reprocess_confirm_channel(self, idx):
        """Mark this channel's setup confirmed for the run.

        Both flags, because _validate_analysis_setup reads them at different
        guards and a channel that satisfies one but not the other is refused.
        """
        try:
            self._chan_confirmed.add(idx)
            self._labels_tab_confirmed = True
        except Exception:
            pass

    def _reprocess_analyse(self):
        """Start the analysis and wait for the GUI thread to finish with it.

        The wait pumps the event loop rather than blocking on the event.
        Blocking would deadlock: the latch is set by _poll_queue, which is an
        `after` callback on this very thread, so a thread waiting on it stops
        the thing it is waiting for. _load_file_entry_inner waits the same way
        for the same reason.
        """
        # The Inspector is on in 132 of the 135 sessions this was built
        # against, and it opens per channel per file. Forced off for the run
        # and restored in _reprocess_finish.
        #
        # Nothing is lost by it: the saved marker edits reach the pipeline
        # through existing_segments_metadata whether the window opens or not,
        # and exclusions live in the same place. Outlier review is NOT forced
        # -- the planner refuses any session that had it on, because what that
        # dialogue decided was never recorded.
        self._reprocess_inspector_was = self.enable_inspector.get()
        self.enable_inspector.set(False)

        self.run_analysis_start()
        if not getattr(self, "_analysis_running", False) \
                and not self._run_finished.is_set():
            # A guard in _validate_analysis_setup refused, and it has already
            # said why in a dialogue. Reported as a failure rather than a hang.
            return ("error", "the analysis would not start; see the log.")

        deadline = time.time() + _FILE_TIMEOUT_S
        while not self._run_finished.is_set():
            if getattr(self, "_closing", False):
                return ("error", "the application was closing.")
            if self._reprocess_stop:
                return ("error", "stopped by the analyst.")
            if time.time() > deadline:
                return ("error", f"gave up after {_FILE_TIMEOUT_S:g} s.")
            self.root.update()
            time.sleep(0.02)
        return self._run_outcome or ("error", "the run reported no outcome.")

    def _reprocess_compare(self, file_id):
        """What moved between the old trial file and the one just written."""
        fe = self._dataset.get_file(file_id)
        before = self._reprocess_before.get(file_id)
        after = self._reprocess_read_trials(fe) if fe is not None else None
        if before is None or after is None:
            # No basis for a comparison. Treated as a change so the file is
            # reviewed: "I could not check" must not read the same as
            # "I checked and nothing moved".
            return {"columns": [], "unchecked": True}
        findings = RP.compare_trials(before, after)
        self._reprocess_findings[file_id] = findings
        return findings if RP.needs_review(findings) else None

    def _reprocess_read_trials(self, fe):
        """The current trial file for this entry, as a list of dicts."""
        path = self._reprocess_trials_path(fe)
        if not path or not os.path.isfile(path):
            return None
        try:
            # Long paths are ordinary in a BIDS study under OneDrive: 265
            # files in the study this was built against are already past 260
            # characters, the longest at 369.
            with open("\\\\?\\" + os.path.abspath(path)
                      if os.name == "nt" else path,
                      newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not read the previous trial file "
                     f"({os.path.basename(path)}): {exc}")
            return None

    def _reprocess_trials_path(self, fe):
        """Where this entry's trial file is, old layout or new."""
        prefix = self._reprocess_saved_prefix(fe)
        base = os.path.dirname(getattr(fe, "derivatives_json", "") or "")
        if not prefix or not base:
            return None
        name = f"{prefix}_trials.csv"
        for candidate in (os.path.join(base, "results", "trial-level", name),
                          os.path.join(base, "results", name),
                          os.path.join(base, name)):
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _reprocess_saved_prefix(fe):
        """The prefix this file's outputs were written under.

        Taken from the session JSON's own name, which is the only record of it
        that does not depend on naming rules that have since changed.
        """
        name = os.path.basename(getattr(fe, "derivatives_json", "") or "")
        return name[:-len("_session.json")] \
            if name.endswith("_session.json") else ""

    # ── finishing ────────────────────────────────────────────────────────────

    def _reprocess_finish(self, results):
        """Apply each outcome to the queue, write the report, say what happened."""
        # Cleared first, and unconditionally. Left set, the next file the
        # analyst opens by hand would load without asking anything and write
        # its results under the previous study's prefix.
        self._unattended = None
        self._bids_prefix_override = None
        try:
            self.enable_inspector.set(
                getattr(self, "_reprocess_inspector_was", True))
        except Exception:
            pass

        for result in results:
            fe = self._dataset.get_file(result["key"])
            if fe is None or result["status"] is None:
                continue
            fe.status = result["status"]
            flag = RR.review_flag_for(result)
            if flag and hasattr(fe, "review_flags"):
                try:
                    fe.review_flags.update(flag)
                except Exception:
                    pass
        self._dataset.save()
        self._queue_refresh()

        report_path = self._reprocess_write_report(results)
        counts = RR.summarise(results)
        summary = (
            f"Re-run finished.\n\n"
            f"    unchanged, marked complete   {counts[RR.UNCHANGED]}\n"
            f"    values changed, for review   {counts[RR.CHANGED]}\n"
            f"    failed, for review           {counts[RR.FAILED]}\n"
            f"    could not be loaded          {counts[RR.LOAD_FAILED]}\n"
            f"    not reached                  {counts[RR.NOT_REACHED]}\n")
        if report_path:
            summary += f"\nChange report: {report_path}"
        if counts[RR.CHANGED]:
            summary += ("\n\nThe files marked for review produced different "
                        "numbers from the ones already on disk. Some of that "
                        "is the corrected cSP/MEP ratio. Check the report "
                        "before using these results.")
        self.log("\n" + summary)
        messagebox.showinfo("Re-run finished", summary, parent=self.root)

        # After the report, not before: the archive is a consequence of the
        # run, and its own summary should not arrive on top of the run's.
        if getattr(self, "_reprocess_archive_after", False):
            self._reprocess_archive_after = False
            self._reprocess_archive_outputs()

    def _reprocess_write_report(self, results):
        """One row per (file, column) that moved. Returns the path, or None."""
        rows = []
        for result in results:
            findings = self._reprocess_findings.get(result["key"])
            if not findings:
                continue
            rows.extend(RP.report_rows(self._reprocess_name(result["key"]),
                                       findings))
        if not rows:
            return None
        root = (self.derivatives_path.get().strip()
                or os.path.dirname(self._dataset.json_path or "") or ".")
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path = os.path.join(root, f"reprocess_changes_{stamp}.csv")
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["File", "Column", "TrialsAffected",
                                    "ExampleWas", "ExampleNow"])
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not write the change report: {exc}")
            return None
        return path

    def stop_reprocess(self):
        """Ask the run to stop after the file it is on."""
        self._reprocess_stop = True
        self.log("   Stopping after the current file…")
