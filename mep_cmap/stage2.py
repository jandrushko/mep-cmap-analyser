"""
mep_cmap.stage2
~~~~~~~~~~~~~~~
Stage 2 — Group Analysis mixin.

Contains all _s2_* methods and _build_stage2 / _on_tab_changed that implement
the group-level analysis tab.  Mixed into TMSAnalysisApp via Stage2Mixin.
"""

import os
import json
import pathlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import pandas as pd

from .bids import StudyMetadata


class Stage2Mixin:
    """
    Mixin providing the Stage 2 (Group Analysis) tab functionality.
    All methods are intended to be used as part of TMSAnalysisApp.
    """

    def _on_tab_changed(self, event):
        """Build Stage 2 UI lazily on first visit."""
        if self.notebook.index(self.notebook.select()) == 1:
            if not self._stage2_built:
                self._build_stage2()
                self._stage2_built = True

    def _build_stage2(self):
        """Construct the entire Stage 2 panel inside self.tab2_frame."""
        f = self.tab2_frame

        # ── Top toolbar ───────────────────────────────────────────────────────
        toolbar = tk.Frame(f)
        toolbar.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(toolbar, text="Derivatives folder:").pack(side="left")
        self._s2_deriv_var = tk.StringVar(value=self.derivatives_path.get())
        tk.Entry(toolbar, textvariable=self._s2_deriv_var, width=45)            .pack(side="left", padx=(4, 2))
        tk.Button(toolbar, text="Browse",
                  command=self._s2_browse_deriv).pack(side="left")
        tk.Button(toolbar, text="Scan folder",
                  command=self._s2_scan).pack(side="left", padx=(12, 0))
        tk.Button(toolbar, text="Save design",
                  command=self._s2_save_design).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Load design",
                  command=self._s2_load_design).pack(side="left", padx=(2, 0))

        # ── Group column manager ──────────────────────────────────────────────
        col_bar = tk.Frame(f)
        col_bar.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(col_bar, text="Group columns:").pack(side="left")
        tk.Button(col_bar, text="+ Add column",
                  command=self._s2_add_column).pack(side="left", padx=(6, 0))
        self._s2_col_buttons_frame = tk.Frame(col_bar)
        self._s2_col_buttons_frame.pack(side="left", padx=(8, 0))

        # ── Assignment table (Treeview + scrollbars) ──────────────────────────
        tbl_frame = tk.Frame(f)
        tbl_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        hscroll = ttk.Scrollbar(tbl_frame, orient="horizontal")
        hscroll.pack(side="bottom", fill="x")
        vscroll2 = ttk.Scrollbar(tbl_frame, orient="vertical")
        vscroll2.pack(side="right", fill="y")

        self._s2_tree = ttk.Treeview(
            tbl_frame,
            show="headings",
            selectmode="browse",
            yscrollcommand=vscroll2.set,
            xscrollcommand=hscroll.set,
        )
        self._s2_tree.pack(fill="both", expand=True)
        vscroll2.config(command=self._s2_tree.yview)
        hscroll.config(command=self._s2_tree.xview)

        # Bind double-click to edit a cell
        self._s2_tree.bind("<Double-1>", self._s2_on_double_click)
        # Bind right-click on heading for column context menu
        self._s2_tree.bind("<Button-3>", self._s2_on_right_click)

        # ── Status bar + quick-select ─────────────────────────────────────────
        bot = tk.Frame(f)
        bot.pack(fill="x", padx=10, pady=(0, 6))
        tk.Button(bot, text="Select all",
                  command=lambda: self._s2_set_all_include(True))            .pack(side="left", padx=(0, 4))
        tk.Button(bot, text="Deselect all",
                  command=lambda: self._s2_set_all_include(False))            .pack(side="left")
        self._s2_status = tk.Label(bot, text="", anchor="e")
        self._s2_status.pack(side="right")

        # ── Internal state ────────────────────────────────────────────────────
        # group_columns: list of {"name": str, "type": "between"|"within"}
        self._s2_group_cols  = []
        # group_values: {col_name: [val1, val2, ...]}
        self._s2_group_vals  = {}
        # row data: list of dicts (one per session row)
        self._s2_rows        = []

        self._s2_rebuild_tree_columns()
        self._s2_update_status()

    # ── Browse / scan ─────────────────────────────────────────────────────────

    def _s2_browse_deriv(self):
        folder = filedialog.askdirectory(
            title="Select derivatives folder", mustexist=True)
        if folder:
            self._s2_deriv_var.set(folder)
            self.derivatives_path.set(folder)

    def _s2_scan(self):
        """Scan the derivatives folder for sidecar JSONs and populate the table."""
        root_dir = self._s2_deriv_var.get().strip()
        if not root_dir or not os.path.isdir(root_dir):
            messagebox.showerror("No folder",
                "Please enter or browse to a valid derivatives folder.",
                parent=self.root)
            return

        deriv_dir = os.path.join(root_dir, "derivatives")
        if not os.path.isdir(deriv_dir):
            # Try the folder itself as the derivatives root
            deriv_dir = root_dir

        # Walk and find all *_trials_manual.json sidecars (one per session)
        found = []
        for dirpath, dirnames, filenames in os.walk(deriv_dir):
            for fn in filenames:
                if fn.endswith("_trials_manual.json"):
                    jpath = os.path.join(dirpath, fn)
                    try:
                        with open(jpath, encoding="utf-8") as jf:
                            meta = json.load(jf)
                        found.append({
                            "include":        True,
                            "participant_id": meta.get("participant_id", ""),
                            "session":        meta.get("session", ""),
                            "task":           meta.get("task", ""),
                            "timepoint":      meta.get("timepoint", ""),
                            "_json_path":     jpath,
                            "_trials_csv":    jpath.replace(".json", ".csv"),
                        })
                    except Exception:
                        pass

        if not found:
            messagebox.showinfo("Nothing found",
                "No Stage 1 outputs found in that folder. "
                "Make sure you have run Stage 1 with a derivatives folder set.",
                parent=self.root)
            return

        # Merge with existing rows: preserve group assignments for known sessions
        existing = {(r["participant_id"], r["session"]): r
                    for r in self._s2_rows}
        merged = []
        for row in found:
            key = (row["participant_id"], row["session"])
            if key in existing:
                # Keep existing group assignments, update metadata
                old = existing[key].copy()
                old.update({k: v for k, v in row.items()
                            if k not in self._s2_group_cols})
                merged.append(old)
            else:
                # New row — add empty group columns
                for gc in self._s2_group_cols:
                    row[gc["name"]] = ""
                merged.append(row)

        # Sort by participant then session
        merged.sort(key=lambda r: (r["participant_id"], r["session"]))
        self._s2_rows = merged
        self._s2_refresh_tree()
        self._s2_update_status()

    # ── Tree column management ────────────────────────────────────────────────

    def _s2_rebuild_tree_columns(self):
        """Rebuild Treeview columns from current state."""
        fixed = ["include", "participant_id", "session", "task", "timepoint", "configure"]
        group_names = [gc["name"] for gc in self._s2_group_cols]
        all_cols = fixed + group_names

        self._s2_tree["columns"] = all_cols
        col_widths = {
            "include":        55,
            "participant_id": 110,
            "session":        80,
            "task":           90,
            "timepoint":      80,
            "configure":      80,
        }
        col_labels = {
            "include":        "Include",
            "participant_id": "Participant",
            "session":        "Session",
            "task":           "Task",
            "timepoint":      "Timepoint",
            "configure":      "Setup",
        }
        for col in all_cols:
            w = col_widths.get(col, 100)
            lbl = col_labels.get(col, col)
            # Mark between-subjects columns with a tilde prefix in header
            gc_meta = next((gc for gc in self._s2_group_cols
                            if gc["name"] == col), None)
            if gc_meta and gc_meta["type"] == "between":
                lbl = f"~ {col}"
            self._s2_tree.heading(col, text=lbl,
                command=lambda c=col: self._s2_sort_by(c))
            self._s2_tree.column(col, width=w, minwidth=50, anchor="center")

        self._s2_rebuild_col_buttons()

    def _s2_rebuild_col_buttons(self):
        """Refresh the row of column-management buttons."""
        for w in self._s2_col_buttons_frame.winfo_children():
            w.destroy()
        for gc in self._s2_group_cols:
            name = gc["name"]
            btn = tk.Button(
                self._s2_col_buttons_frame,
                text=f"⚙ {name}",
                relief="groove",
                padx=4,
                command=lambda n=name: self._s2_manage_column(n),
            )
            btn.pack(side="left", padx=2)

    def _s2_refresh_tree(self):
        """Clear and repopulate the Treeview from self._s2_rows."""
        for item in self._s2_tree.get_children():
            self._s2_tree.delete(item)
        fixed = ["participant_id", "session", "task", "timepoint"]
        for i, row in enumerate(self._s2_rows):
            vals = []
            vals.append("☑" if row.get("include", True) else "☐")
            for col in ["participant_id", "session", "task", "timepoint"]:
                vals.append(row.get(col, ""))
            # Configure column: show tick if already configured
            cfg = row.get("_config", {})
            vals.append("⚙ configured" if cfg.get("_done") else "⚙ setup")
            for gc in self._s2_group_cols:
                vals.append(row.get(gc["name"], ""))
            tag = "even" if i % 2 == 0 else "odd"
            self._s2_tree.insert("", "end", iid=str(i),
                                 values=vals, tags=(tag,))
        self._s2_tree.tag_configure("even", background="#f8f8f8")
        self._s2_tree.tag_configure("odd",  background="#ffffff")

    # ── Cell editing ──────────────────────────────────────────────────────────

    def _s2_on_double_click(self, event):
        """Handle double-click: toggle Include or open cell editor."""
        region = self._s2_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col_id = self._s2_tree.identify_column(event.x)
        row_id = self._s2_tree.identify_row(event.y)
        if not row_id:
            return

        col_idx  = int(col_id.lstrip("#")) - 1
        all_cols = list(self._s2_tree["columns"])
        col_name = all_cols[col_idx]
        row_idx  = int(row_id)

        if col_name == "include":
            self._s2_rows[row_idx]["include"] = \
                not self._s2_rows[row_idx].get("include", True)
            self._s2_refresh_tree()
            self._s2_update_status()
            return

        if col_name == "configure":
            self._s2_open_configure(row_idx)
            return

        # Only group columns are editable
        gc_meta = next((gc for gc in self._s2_group_cols
                        if gc["name"] == col_name), None)
        if gc_meta is None:
            return

        self._s2_edit_cell(row_idx, col_name, gc_meta, event.x_root, event.y_root)

    def _s2_open_configure(self, row_idx):
        """
        Open the per-session Configure dialog.
        Reads stim types from the session's trials_manual.csv, lets the user
        assign roles (Reference, Conditioned, M-wave, None), configure M-wave
        source and trial selection, and define paired-pulse pairings.
        """
        row = self._s2_rows[row_idx]
        csv_path = row.get("_trials_csv", "")

        # ── Load the trial CSV ────────────────────────────────────────────────
        if not csv_path or not os.path.isfile(csv_path):
            # Try to find it by scanning the session folder
            json_path = row.get("_json_path", "")
            if json_path:
                csv_path = json_path.replace(".json", ".csv")
            if not csv_path or not os.path.isfile(csv_path):
                messagebox.showerror(
                    "Could not locate the trials_manual.csv for this session. "
                    "Please re-scan the derivatives folder.",
                    "Please re-scan the derivatives folder.",
                    parent=self.root)
                return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            messagebox.showerror("CSV error", str(e), parent=self.root)
            return

        # Stim types present in this session
        stim_types = sorted(df["StimType"].unique()) if "StimType" in df.columns else []
        if not stim_types:
            messagebox.showinfo("No stim types",
                "No stim types found in this session's trial CSV.",
                parent=self.root)
            return

        # Restore previous config if any
        cfg = row.setdefault("_config", {})

        # ── Build dialog ──────────────────────────────────────────────────────
        title = " – ".join(filter(None, [
            row.get("participant_id",""), row.get("session",""),
            row.get("task",""), row.get("timepoint","")]))
        win = tk.Toplevel(self.root)
        win.title(f"Configure – {title}")
        win.transient(self.root)
        win.resizable(True, True)

        ROLES = ["None", "Reference (single pulse)", "Conditioned", "M-wave"]
        ROLE_COLOURS = {
            "Reference (single pulse)": "#d4edda",
            "Conditioned":              "#cce5ff",
            "M-wave":                   "#fff3cd",
            "None":                     "#ffffff",
        }

        # ════════════════════════ Section 1: Stim roles ═══════════════════════
        sec1 = tk.LabelFrame(win, text="Section 1 — Stim type roles", padx=8, pady=6)
        sec1.pack(fill="x", padx=10, pady=(10,4))

        tk.Label(sec1, text="Stim", width=8, anchor="w", font=("TkDefaultFont",9,"bold"))            .grid(row=0, column=0, sticky="w")
        tk.Label(sec1, text="Label", width=14, anchor="w", font=("TkDefaultFont",9,"bold"))            .grid(row=0, column=1, sticky="w")
        tk.Label(sec1, text="Role", width=28, anchor="w", font=("TkDefaultFont",9,"bold"))            .grid(row=0, column=2, sticky="w")
        tk.Label(sec1, text="N trials", width=8, anchor="w", font=("TkDefaultFont",9,"bold"))            .grid(row=0, column=3, sticky="w")

        role_vars = {}
        for r, st in enumerate(stim_types, start=1):
            n_trials = int((df["StimType"] == st).sum())
            lbl = df.loc[df["StimType"]==st, "Stim_Label"].iloc[0]                   if "Stim_Label" in df.columns else st
            tk.Label(sec1, text=st, width=8, anchor="w").grid(row=r, column=0, sticky="w")
            tk.Label(sec1, text=str(lbl), width=14, anchor="w").grid(row=r, column=1, sticky="w")
            v = tk.StringVar(value=cfg.get(f"role_{st}", "None"))
            role_vars[st] = v
            cb = ttk.Combobox(sec1, textvariable=v, values=ROLES,
                               state="readonly", width=26)
            cb.grid(row=r, column=2, sticky="w", padx=4)
            tk.Label(sec1, text=str(n_trials), width=8, anchor="w")                .grid(row=r, column=3, sticky="w")

        # ════════════════════════ Section 2: M-wave ═══════════════════════════
        sec2 = tk.LabelFrame(win, text="Section 2 — M-wave / Mmax", padx=8, pady=6)
        sec2.pack(fill="x", padx=10, pady=4)

        mw_source = tk.StringVar(value=cfg.get("mwave_source", "stim_type"))
        mw_manual_val = tk.DoubleVar(value=cfg.get("mwave_manual_value", 0.0))
        mw_ext_path   = tk.StringVar(value=cfg.get("mwave_ext_path", ""))
        mw_ext_stim   = tk.StringVar(value=cfg.get("mwave_ext_stim", ""))

        tk.Radiobutton(sec2, text="Use stim type assigned above",
                       variable=mw_source, value="stim_type",
                       command=lambda: _refresh_mwave()).grid(
                       row=0, column=0, sticky="w", columnspan=2)
        tk.Radiobutton(sec2, text="Manual Mmax value (mV):",
                       variable=mw_source, value="manual",
                       command=lambda: _refresh_mwave()).grid(
                       row=1, column=0, sticky="w")
        mw_manual_entry = tk.Entry(sec2, textvariable=mw_manual_val, width=10)
        mw_manual_entry.grid(row=1, column=1, sticky="w", padx=4)
        tk.Radiobutton(sec2, text="Separate file:",
                       variable=mw_source, value="ext_file",
                       command=lambda: _refresh_mwave()).grid(
                       row=2, column=0, sticky="w")
        mw_ext_entry = tk.Entry(sec2, textvariable=mw_ext_path, width=34)
        mw_ext_entry.grid(row=2, column=1, sticky="w", padx=4)
        tk.Button(sec2, text="Browse", command=lambda: _browse_ext_mwave())            .grid(row=2, column=2, sticky="w")

        # Ext file stim type selector
        tk.Label(sec2, text="Stim type in file:").grid(row=3, column=0, sticky="e", padx=4)
        mw_ext_stim_cb = ttk.Combobox(sec2, textvariable=mw_ext_stim,
                                       state="readonly", width=12)
        mw_ext_stim_cb.grid(row=3, column=1, sticky="w", padx=4)

        # Trial PTP table for M-wave selection
        mw_tbl_frame = tk.LabelFrame(sec2, text="Select plateau trials for Mmax mean", padx=4, pady=4)
        mw_tbl_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6,0))

        mw_tree = ttk.Treeview(mw_tbl_frame, columns=["trial","ptp","include"],
                                show="headings", height=5)
        mw_tree.heading("trial", text="Trial"); mw_tree.column("trial", width=50)
        mw_tree.heading("ptp",   text="PTP (mV)"); mw_tree.column("ptp", width=80)
        mw_tree.heading("include", text="Include"); mw_tree.column("include", width=60)
        mw_tree.pack(fill="x")
        mw_tree.bind("<Double-1>", lambda e: _toggle_mwave_trial(e))

        mw_mean_lbl = tk.Label(sec2, text="Mmax mean: —")
        mw_mean_lbl.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4,0))

        # Storage for mwave trial include state
        _mw_trials = {}   # {trial_idx: {"ptp": float, "include": bool}}

        def _load_mwave_trials(source_df, stim):
            _mw_trials.clear()
            for item in mw_tree.get_children():
                mw_tree.delete(item)
            if stim not in source_df["StimType"].values:
                mw_mean_lbl.config(text="Mmax mean: — (stim type not found)")
                return
            trials = source_df[source_df["StimType"] == stim].reset_index(drop=True)
            prev = cfg.get("mwave_trials", {})
            for i, row_t in trials.iterrows():
                ptp = row_t.get("PeakToPeak(mV)", float("nan"))
                inc = prev.get(str(i), {}).get("include", True)
                _mw_trials[i] = {"ptp": ptp, "include": inc}
                mw_tree.insert("", "end", iid=str(i),
                               values=[i+1, f"{ptp:.4f}", "☑" if inc else "☐"])
            _update_mmax()

        def _toggle_mwave_trial(event):
            row_id = mw_tree.identify_row(event.y)
            if row_id:
                i = int(row_id)
                _mw_trials[i]["include"] = not _mw_trials[i]["include"]
                inc = _mw_trials[i]["include"]
                ptp = _mw_trials[i]["ptp"]
                mw_tree.item(row_id, values=[i+1, f"{ptp:.4f}", "☑" if inc else "☐"])
                _update_mmax()

        def _update_mmax():
            vals = [v["ptp"] for v in _mw_trials.values()
                    if v["include"] and not np.isnan(v["ptp"])]
            if vals:
                mean_v = float(np.mean(vals))
                mw_mean_lbl.config(text=f"Mmax mean: {mean_v:.4f} mV  ({len(vals)} trials)")
            else:
                mw_mean_lbl.config(text="Mmax mean: — (no trials selected)")

        def _browse_ext_mwave():
            p = filedialog.askopenfilename(
                title="Select external M-wave trials_manual.csv",
                filetypes=[("CSV", "*.csv")])
            if p:
                mw_ext_path.set(p)
                try:
                    ext_df = pd.read_csv(p)
                    stims = sorted(ext_df["StimType"].unique())                             if "StimType" in ext_df.columns else []
                    mw_ext_stim_cb["values"] = stims
                    if stims:
                        mw_ext_stim.set(stims[0])
                        _load_mwave_trials(ext_df, stims[0])
                except Exception as ex:
                    messagebox.showerror("Error", str(ex), parent=win)

        def _refresh_mwave(*_):
            src = mw_source.get()
            mw_manual_entry.config(state="normal" if src == "manual" else "disabled")
            mw_ext_entry.config(   state="normal" if src == "ext_file" else "disabled")
            mw_ext_stim_cb.config( state="readonly" if src == "ext_file" else "disabled")
            if src == "stim_type":
                # Find the stim assigned M-wave role
                mw_st = next((s for s,v in role_vars.items()
                              if v.get() == "M-wave"), None)
                if mw_st:
                    _load_mwave_trials(df, mw_st)
                else:
                    for item in mw_tree.get_children():
                        mw_tree.delete(item)
                    mw_mean_lbl.config(text="Mmax mean: — (assign M-wave role above)")
            elif src == "ext_file" and mw_ext_path.get() and os.path.isfile(mw_ext_path.get()):
                try:
                    ext_df = pd.read_csv(mw_ext_path.get())
                    stims = sorted(ext_df["StimType"].unique())                             if "StimType" in ext_df.columns else []
                    mw_ext_stim_cb["values"] = stims
                    st = mw_ext_stim.get() or (stims[0] if stims else "")
                    mw_ext_stim.set(st)
                    _load_mwave_trials(ext_df, st)
                except Exception:
                    pass

        # Bind role changes to refresh m-wave panel
        for v in role_vars.values():
            v.trace_add("write", _refresh_mwave)
        mw_ext_stim.trace_add("write",
            lambda *_: _refresh_mwave() if mw_source.get()=="ext_file" else None)

        # ════════════════════════ Section 3: Paired pulse ═════════════════════
        sec3 = tk.LabelFrame(win, text="Section 3 — Paired pulse pairings", padx=8, pady=6)
        sec3.pack(fill="x", padx=10, pady=4)

        tk.Label(sec3, text="Conditioned stim", width=18, anchor="w",
                 font=("TkDefaultFont",9,"bold")).grid(row=0, column=0, sticky="w")
        tk.Label(sec3, text="Reference stim", width=18, anchor="w",
                 font=("TkDefaultFont",9,"bold")).grid(row=0, column=1, sticky="w")
        tk.Label(sec3, text="Ratio preview", width=18, anchor="w",
                 font=("TkDefaultFont",9,"bold")).grid(row=0, column=2, sticky="w")

        pair_ref_vars = {}   # {cond_stim: StringVar for ref}
        pair_ratio_lbls = {} # {cond_stim: Label}
        pairs_frame = tk.Frame(sec3)
        pairs_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        def _rebuild_pairs(*_):
            """Refresh Section 3 based on which stims are assigned Conditioned."""
            for w in pairs_frame.winfo_children():
                w.destroy()
            pair_ref_vars.clear()
            pair_ratio_lbls.clear()
            cond_stims = [s for s,v in role_vars.items() if v.get()=="Conditioned"]
            ref_stims  = [s for s,v in role_vars.items()
                          if v.get()=="Reference (single pulse)"]
            if not cond_stims:
                tk.Label(pairs_frame, text="No conditioned stim types assigned yet.",
                         fg="grey").grid(row=0, column=0, columnspan=3, sticky="w")
                return
            ref_choices = ref_stims or stim_types
            for r, cs in enumerate(cond_stims):
                tk.Label(pairs_frame, text=cs, width=18, anchor="w")                    .grid(row=r, column=0, sticky="w")
                prev_ref = cfg.get(f"pair_ref_{cs}", ref_choices[0] if ref_choices else "")
                v_ref = tk.StringVar(value=prev_ref)
                pair_ref_vars[cs] = v_ref
                ref_cb = ttk.Combobox(pairs_frame, textvariable=v_ref,
                                      values=ref_choices, state="readonly", width=16)
                ref_cb.grid(row=r, column=1, sticky="w", padx=4)
                lbl = tk.Label(pairs_frame, text="—", width=20, anchor="w")
                lbl.grid(row=r, column=2, sticky="w")
                pair_ratio_lbls[cs] = lbl
                v_ref.trace_add("write", lambda *_, c=cs: _update_ratio(c))
                _update_ratio(cs)

        def _update_ratio(cond_st):
            lbl = pair_ratio_lbls.get(cond_st)
            if lbl is None:
                return
            ref_st = pair_ref_vars[cond_st].get()
            try:
                cond_mean = df[df["StimType"]==cond_st]["PeakToPeak(mV)"].mean()
                ref_mean  = df[df["StimType"]==ref_st ]["PeakToPeak(mV)"].mean()
                ratio = (cond_mean / ref_mean) * 100 if ref_mean else float("nan")
                lbl.config(text=f"{ratio:.1f}% (mean)")
            except Exception:
                lbl.config(text="—")

        for v in role_vars.values():
            v.trace_add("write", _rebuild_pairs)
        _rebuild_pairs()

        # ════════════════════════ Save / Cancel ═══════════════════════════════
        btn_row = tk.Frame(win)
        btn_row.pack(pady=10)

        def _save_config():
            new_cfg = {"_done": True}
            # Roles
            for st, v in role_vars.items():
                new_cfg[f"role_{st}"] = v.get()
            # M-wave
            new_cfg["mwave_source"] = mw_source.get()
            new_cfg["mwave_manual_value"] = mw_manual_val.get()
            new_cfg["mwave_ext_path"]     = mw_ext_path.get()
            new_cfg["mwave_ext_stim"]     = mw_ext_stim.get()
            new_cfg["mwave_trials"] = {
                str(i): {"ptp": d["ptp"], "include": d["include"]}
                for i, d in _mw_trials.items()
            }
            # Compute Mmax
            if mw_source.get() == "manual":
                new_cfg["mmax_ptp"] = mw_manual_val.get()
            else:
                included = [d["ptp"] for d in _mw_trials.values()
                            if d["include"] and not np.isnan(d["ptp"])]
                new_cfg["mmax_ptp"] = float(np.mean(included)) if included else None
            # Paired pulse refs
            for cs, v_ref in pair_ref_vars.items():
                new_cfg[f"pair_ref_{cs}"] = v_ref.get()
            # Store
            self._s2_rows[row_idx]["_config"] = new_cfg
            self._s2_refresh_tree()
            win.destroy()

        tk.Button(btn_row, text="Save & close", width=14,
                  command=_save_config).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side="left", padx=6)

        # Initial state
        _refresh_mwave()
        win.grab_set()

    def _s2_edit_cell(self, row_idx, col_name, gc_meta, x_root, y_root):
        """Pop a small Combobox editor for a group cell."""
        current = self._s2_rows[row_idx].get(col_name, "")
        vals    = self._s2_group_vals.get(col_name, [])

        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.geometry(f"+{x_root}+{y_root}")

        var = tk.StringVar(value=current)
        cb  = ttk.Combobox(popup, textvariable=var,
                           values=vals, width=18)
        cb.pack(padx=2, pady=2)
        cb.focus_set()
        cb.event_generate("<Button-1>")

        def _commit(_e=None):
            new_val = var.get().strip()
            if new_val and new_val not in self._s2_group_vals.get(col_name, []):
                self._s2_group_vals.setdefault(col_name, []).append(new_val)
            self._s2_rows[row_idx][col_name] = new_val
            # Auto-fill between-subjects: propagate to same participant
            if gc_meta["type"] == "between" and new_val:
                pid = self._s2_rows[row_idx]["participant_id"]
                for r in self._s2_rows:
                    if r["participant_id"] == pid and r.get(col_name, "") == "":
                        r[col_name] = new_val
            popup.destroy()
            self._s2_refresh_tree()

        cb.bind("<Return>",    _commit)
        cb.bind("<FocusOut>",  _commit)
        cb.bind("<<ComboboxSelected>>", _commit)

    # ── Include toggles ───────────────────────────────────────────────────────

    def _s2_set_all_include(self, state: bool):
        for row in self._s2_rows:
            row["include"] = state
        self._s2_refresh_tree()
        self._s2_update_status()

    def _s2_sort_by(self, col):
        """Sort table rows by the clicked column header."""
        self._s2_rows.sort(key=lambda r: str(r.get(col, "")))
        self._s2_refresh_tree()

    def _s2_update_status(self):
        n_total   = len(self._s2_rows)
        n_include = sum(1 for r in self._s2_rows if r.get("include", True))
        self._s2_status.config(
            text=f"{n_include} / {n_total} sessions included")

    # ── Right-click column context menu ───────────────────────────────────────

    def _s2_on_right_click(self, event):
        """Show context menu when right-clicking a column heading."""
        region = self._s2_tree.identify_region(event.x, event.y)
        if region != "heading":
            return
        col_id   = self._s2_tree.identify_column(event.x)
        col_idx  = int(col_id.lstrip("#")) - 1
        all_cols = list(self._s2_tree["columns"])
        col_name = all_cols[col_idx]
        gc_meta  = next((gc for gc in self._s2_group_cols
                         if gc["name"] == col_name), None)
        if gc_meta is None:
            return   # fixed column — no context menu

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Rename column…",
                         command=lambda: self._s2_rename_column(col_name))
        menu.add_command(label="Edit allowed values…",
                         command=lambda: self._s2_manage_column(col_name))
        menu.add_separator()
        if gc_meta["type"] == "between":
            menu.add_command(label="Change to Within-subjects",
                             command=lambda: self._s2_set_col_type(col_name, "within"))
        else:
            menu.add_command(label="Change to Between-subjects",
                             command=lambda: self._s2_set_col_type(col_name, "between"))
        menu.add_separator()
        menu.add_command(label="Delete column",
                         command=lambda: self._s2_delete_column(col_name))
        menu.tk_popup(event.x_root, event.y_root)

    # ── Add / rename / delete columns ─────────────────────────────────────────

    def _s2_add_column(self):
        """Dialog to add a new group column."""
        win = tk.Toplevel(self.root)
        win.title("Add group column")
        win.resizable(False, False)
        win.transient(self.root)

        tk.Label(win, text="Column name:").grid(
            row=0, column=0, sticky="e", padx=8, pady=6)
        v_name = tk.StringVar()
        tk.Entry(win, textvariable=v_name, width=20).grid(
            row=0, column=1, sticky="w", padx=4)

        tk.Label(win, text="Column type:").grid(
            row=1, column=0, sticky="e", padx=8, pady=4)
        v_type = tk.StringVar(value="between")
        type_frame = tk.Frame(win)
        type_frame.grid(row=1, column=1, sticky="w")
        tk.Radiobutton(type_frame, text="Between-subjects  (auto-fills per participant)",
                       variable=v_type, value="between").pack(anchor="w")
        tk.Radiobutton(type_frame, text="Within-subjects / crossover  (fill each session independently)",
                       variable=v_type, value="within").pack(anchor="w")

        err = tk.Label(win, text="", fg="red")
        err.grid(row=2, column=0, columnspan=2, padx=8)

        def _ok(_e=None):
            name = v_name.get().strip()
            if not name:
                err.config(text="Name required.")
                return
            if any(gc["name"] == name for gc in self._s2_group_cols):
                err.config(text="Column already exists.")
                return
            self._s2_group_cols.append({"name": name, "type": v_type.get()})
            self._s2_group_vals[name] = []
            for row in self._s2_rows:
                row[name] = ""
            self._s2_rebuild_tree_columns()
            self._s2_refresh_tree()
            win.destroy()

        btn_row = tk.Frame(win)
        btn_row.grid(row=3, column=0, columnspan=2, pady=8)
        tk.Button(btn_row, text="Add", width=9, command=_ok).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", width=9,
                  command=win.destroy).pack(side="left", padx=4)
        win.bind("<Return>", _ok)
        win.grab_set()

    def _s2_manage_column(self, col_name):
        """Dialog to view/edit/delete the allowed values for a group column."""
        win = tk.Toplevel(self.root)
        win.title(f"Manage values – {col_name}")
        win.resizable(False, False)
        win.transient(self.root)

        tk.Label(win, text=f"Allowed values for  '{col_name}'  "
                           f"(double-click to rename, Delete key to remove):").pack(
                           padx=10, pady=(8, 2), anchor="w")

        lb_frame = tk.Frame(win)
        lb_frame.pack(fill="both", expand=True, padx=10)
        lb = tk.Listbox(lb_frame, height=8, width=28, selectmode="single")
        lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lb_frame, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        def _repopulate():
            lb.delete(0, "end")
            for v in self._s2_group_vals.get(col_name, []):
                lb.insert("end", v)

        _repopulate()

        def _delete_val(_e=None):
            sel = lb.curselection()
            if not sel:
                return
            val = lb.get(sel[0])
            if messagebox.askyesno("Delete value",
                    f"Remove '{val}' from allowed values? "
                    "Cells currently set to this value will be cleared.",
                    parent=win):
                self._s2_group_vals[col_name].remove(val)
                for row in self._s2_rows:
                    if row.get(col_name) == val:
                        row[col_name] = ""
                _repopulate()
                self._s2_refresh_tree()

        def _rename_val(_e=None):
            sel = lb.curselection()
            if not sel:
                return
            old_val = lb.get(sel[0])
            new_val = simpledialog.askstring(
                "Rename value", f"New name for '{old_val}':",
                initialvalue=old_val, parent=win)
            if new_val and new_val.strip() and new_val.strip() != old_val:
                new_val = new_val.strip()
                idx = self._s2_group_vals[col_name].index(old_val)
                self._s2_group_vals[col_name][idx] = new_val
                for row in self._s2_rows:
                    if row.get(col_name) == old_val:
                        row[col_name] = new_val
                _repopulate()
                self._s2_refresh_tree()

        lb.bind("<Double-1>", _rename_val)
        lb.bind("<Delete>",   _delete_val)

        btn_row = tk.Frame(win)
        btn_row.pack(pady=6)
        tk.Button(btn_row, text="Rename selected",
                  command=_rename_val).pack(side="left", padx=4)
        tk.Button(btn_row, text="Delete selected",
                  command=_delete_val).pack(side="left", padx=4)
        tk.Button(btn_row, text="Close",
                  command=win.destroy).pack(side="left", padx=4)
        win.grab_set()

    def _s2_rename_column(self, old_name):
        new_name = simpledialog.askstring(
            "Rename column", f"New name for '{old_name}':",
            initialvalue=old_name, parent=self.root)
        if not new_name or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        if any(gc["name"] == new_name for gc in self._s2_group_cols):
            messagebox.showerror("Duplicate", f"A column named '{new_name}' already exists.")
            return
        for gc in self._s2_group_cols:
            if gc["name"] == old_name:
                gc["name"] = new_name
        if old_name in self._s2_group_vals:
            self._s2_group_vals[new_name] = self._s2_group_vals.pop(old_name)
        for row in self._s2_rows:
            if old_name in row:
                row[new_name] = row.pop(old_name)
        self._s2_rebuild_tree_columns()
        self._s2_refresh_tree()

    def _s2_delete_column(self, col_name):
        if not messagebox.askyesno("Delete column",
                f"Delete column '{col_name}' and all its assignments?",
                parent=self.root):
            return
        self._s2_group_cols = [gc for gc in self._s2_group_cols
                               if gc["name"] != col_name]
        self._s2_group_vals.pop(col_name, None)
        for row in self._s2_rows:
            row.pop(col_name, None)
        self._s2_rebuild_tree_columns()
        self._s2_refresh_tree()

    def _s2_set_col_type(self, col_name, new_type):
        for gc in self._s2_group_cols:
            if gc["name"] == col_name:
                gc["type"] = new_type
        self._s2_rebuild_tree_columns()
        self._s2_refresh_tree()

    # ── Save / load study design ──────────────────────────────────────────────

    def _s2_save_design(self):
        root_dir = self._s2_deriv_var.get().strip()
        if not root_dir:
            messagebox.showerror("No folder",
                "Please set the derivatives folder first.", parent=self.root)
            return
        deriv_dir = os.path.join(root_dir, "derivatives")
        os.makedirs(deriv_dir, exist_ok=True)
        design = {
            "group_columns": self._s2_group_cols,
            "group_values":  self._s2_group_vals,
            "assignments": [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in self._s2_rows
            ],
        }
        path = os.path.join(deriv_dir, "study_design.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(design, f, indent=2)
        self._s2_status.config(text=f"Design saved → {path}")

    def _s2_load_design(self):
        root_dir = self._s2_deriv_var.get().strip()
        deriv_dir = os.path.join(root_dir, "derivatives")             if root_dir else ""
        init = deriv_dir if os.path.isdir(deriv_dir) else root_dir
        path = filedialog.askopenfilename(
            title="Load study design",
            initialdir=init,
            filetypes=[("Study design", "study_design.json"),
                       ("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                design = json.load(f)
            self._s2_group_cols = design.get("group_columns", [])
            self._s2_group_vals = design.get("group_values", {})
            rows = design.get("assignments", [])
            # Re-attach private keys (csv paths) by re-scanning if needed
            self._s2_rows = rows
            self._s2_rebuild_tree_columns()
            self._s2_refresh_tree()
            self._s2_update_status()
        except Exception as e:
            messagebox.showerror("Load error", str(e), parent=self.root)

    # ─── End Stage 2 ──────────────────────────────────────────────────────────

    # ─── Input File Selection ────────────────────────────────────────────
