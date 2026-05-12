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

import numpy as np
import pandas as pd

from .bids import StudyMetadata


class Stage2Mixin:
    """
    Mixin providing the Stage 2 (Group Analysis) tab functionality.
    All methods are intended to be used as part of TMSAnalysisApp.
    """

    def _on_tab_changed(self, event):
        """Build Stage 2 UI lazily on first visit."""
        if self.notebook.index(self.notebook.select()) == 2:
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
        tk.Button(toolbar, text="▶  Build group analysis file",
                  command=self._s2_run,
                  bg="#5cb85c", fg="white",
                  font=("TkDefaultFont", 9, "bold")).pack(side="right", padx=(0, 4))

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

        # If derivatives path already set from Stage 1, populate and auto-scan
        _existing_deriv = self.derivatives_path.get() \
            if hasattr(self, 'derivatives_path') else ""
        if _existing_deriv:
            self._s2_deriv_var.set(_existing_deriv)
            self.root.after(100, self._s2_scan)
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

        # Walk and find all *_All_stims_trial_summary.json sidecars (one per session)
        found = []
        for dirpath, dirnames, filenames in os.walk(deriv_dir):
            for fn in filenames:
                if fn.endswith("_All_stims_trial_summary.json"):
                    jpath = os.path.join(dirpath, fn)
                    try:
                        with open(jpath, encoding="utf-8") as jf:
                            meta = json.load(jf)
                        # Fall back to parsing BIDS folder structure if metadata
                        # fields are blank (e.g. files processed before this fix)
                        _parts = pathlib.Path(dirpath).parts
                        _sub = next((p for p in _parts if p.startswith("sub-")), "")
                        _ses = next((p for p in _parts if p.startswith("ses-")), "")
                        found.append({
                            "include":        True,
                            "participant_id": meta.get("participant_id") or _sub,
                            "session":        meta.get("session")        or _ses,
                            "task":           meta.get("task",           ""),
                            "timepoint":      meta.get("timepoint",      ""),
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
        Per-session Configure dialog — Section 1 only: stim type role assignment.
        Normalisation is already handled by Stage 1; role labels are the only
        additional metadata needed at the group level.
        """
        row = self._s2_rows[row_idx]
        csv_path = row.get("_trials_csv", "")

        if not csv_path or not os.path.isfile(csv_path):
            json_path = row.get("_json_path", "")
            if json_path:
                csv_path = json_path.replace(".json", ".csv")
            if not csv_path or not os.path.isfile(csv_path):
                messagebox.showerror("File not found",
                    "Could not locate the All_stims_trial_summary.csv for this session. "
                    "Please re-scan the derivatives folder.",
                    parent=self.root)
                return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            messagebox.showerror("CSV error", str(e), parent=self.root)
            return

        stim_types = sorted(df["StimType"].unique()) if "StimType" in df.columns else []
        if not stim_types:
            messagebox.showinfo("No stim types",
                "No stim types found in this session's trial CSV.",
                parent=self.root)
            return

        cfg = row.setdefault("_config", {})

        title = " – ".join(filter(None, [
            row.get("participant_id",""), row.get("session",""),
            row.get("task",""), row.get("timepoint","")]))
        win = tk.Toplevel(self.root)
        win.title(f"Configure – {title}")
        win.transient(self.root)
        win.resizable(False, False)

        ROLES = ["None", "Reference (single pulse)", "Conditioned", "M-wave", "Other"]

        sec1 = tk.LabelFrame(win, text="Stim type roles", padx=8, pady=6)
        sec1.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(sec1, text="Stim",     width=8,  anchor="w",
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(sec1, text="Label",    width=16, anchor="w",
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, sticky="w")
        tk.Label(sec1, text="Role",     width=28, anchor="w",
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=2, sticky="w")
        tk.Label(sec1, text="N trials", width=8,  anchor="w",
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=3, sticky="w")

        role_vars = {}
        for r, st in enumerate(stim_types, start=1):
            n_trials = int((df["StimType"] == st).sum())
            lbl = (df.loc[df["StimType"] == st, "Stim_Label"].iloc[0]
                   if "Stim_Label" in df.columns else st)
            tk.Label(sec1, text=st,       width=8,  anchor="w").grid(row=r, column=0, sticky="w")
            tk.Label(sec1, text=str(lbl), width=16, anchor="w").grid(row=r, column=1, sticky="w")
            v = tk.StringVar(value=cfg.get(f"role_{st}", "None"))
            role_vars[st] = v
            ttk.Combobox(sec1, textvariable=v, values=ROLES,
                         state="readonly", width=26).grid(row=r, column=2, sticky="w", padx=4)
            tk.Label(sec1, text=str(n_trials), width=8, anchor="w").grid(row=r, column=3, sticky="w")

        tk.Label(win,
            text="Stage 1 already handles normalisation. Roles are appended as\n"
                 "metadata to help identify stim type function in the merged file.",
            fg="grey", justify="left").pack(padx=10, pady=(4, 0), anchor="w")

        btn_row = tk.Frame(win)
        btn_row.pack(pady=10)

        def _save_config():
            new_cfg = {"_done": True}
            for st, v in role_vars.items():
                new_cfg[f"role_{st}"] = v.get()
            self._s2_rows[row_idx]["_config"] = new_cfg
            self._s2_refresh_tree()
            win.destroy()

        tk.Button(btn_row, text="Save & close", width=14,
                  command=_save_config).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side="left", padx=6)
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

    def _s2_run(self):
        """
        Merge all included sessions' trial-level CSVs into a single
        group-level LME-ready file, appending study design columns.

        Output: derivatives/group_level_LME_ready.csv
        """
        # ── Validate ──────────────────────────────────────────────────────────
        included = [r for r in self._s2_rows if r.get("include", True)]
        if not included:
            messagebox.showwarning("Nothing included",
                "No sessions are included. Use the checkboxes to include sessions.",
                parent=self.root)
            return

        root_dir  = self._s2_deriv_var.get().strip()
        deriv_dir = os.path.join(root_dir, "derivatives")
        if not os.path.isdir(deriv_dir):
            deriv_dir = root_dir
        if not os.path.isdir(deriv_dir):
            messagebox.showerror("No folder",
                "Could not locate the derivatives folder.", parent=self.root)
            return

        # ── Identify design columns ───────────────────────────────────────────
        group_cols  = [gc["name"] for gc in self._s2_group_cols]

        # ── Load and annotate each session ────────────────────────────────────
        all_frames = []
        skipped    = []

        for row in included:
            csv_path = row.get("_trials_csv", "")
            if not csv_path or not os.path.isfile(csv_path):
                skipped.append(row.get("participant_id", "?") + "/" +
                               row.get("session", "?"))
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                skipped.append(f"{row.get('participant_id','?')}: {e}")
                continue

            if df.empty:
                skipped.append(row.get("participant_id", "?") + " (empty CSV)")
                continue

            # ── Append Stim_Role from Configure dialog ─────────────────────────
            cfg = row.get("_config", {})
            df["Stim_Role"] = df["StimType"].map(
                lambda st: cfg.get(f"role_{st}", "None") if cfg else "None")

            # ── Prepend design columns in correct order ────────────────────────
            # Target order: File, participant_id, [group cols], session,
            #               task, timepoint, StimType, Stim_Label, Segment ...
            # Insert right-to-left so index 0 ends up as File
            for gc_name in reversed(group_cols):
                df.insert(1, gc_name, row.get(gc_name, ""))
            df.insert(1, "participant_id", row.get("participant_id", ""))

            # Move session/task/timepoint to just after participant/group cols
            # They already exist in the CSV from Stage 1 if BIDS was set,
            # otherwise add them from the study design
            n_design = 2 + len(group_cols)  # File + participant_id + group cols
            for i, col in enumerate(["session", "task", "timepoint"]):
                if col in df.columns:
                    # Move existing column to correct position
                    s = df.pop(col)
                    df.insert(n_design + i, col, row.get(col, "") or s)
                else:
                    df.insert(n_design + i, col, row.get(col, ""))

            # ── Reorder columns ────────────────────────────────────────────────
            # Final order: File, participant_id, [group cols], session, task,
            # timepoint, Limb, StimType, Stim_Label, Segment, [metrics...]
            if "Limb" in df.columns:
                limb = df.pop("Limb")
                # Insert after timepoint (n_design + 3 cols: session/task/timepoint)
                df.insert(n_design + 3, "Limb", limb)

            all_frames.append(df)

        # ── Bail if nothing loaded ─────────────────────────────────────────────
        if not all_frames:
            messagebox.showerror("No data",
                "Could not load any session CSVs. Check that Stage 1 has been "
                "run and the derivatives folder is correct.", parent=self.root)
            return

        # ── Stack all sessions ─────────────────────────────────────────────────
        # Use outer join so sessions with different columns don't crash —
        # missing columns are filled with NaN.
        merged = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)

        # Sort: participant → session → stim type → segment
        sort_cols = [c for c in ["participant_id", "session", "StimType", "Segment"]
                     if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)

        # ── Write output ──────────────────────────────────────────────────────
        out_path = os.path.join(deriv_dir, "group_level_LME_ready.csv")
        try:
            merged.to_csv(out_path, index=False)
        except Exception as e:
            messagebox.showerror("Write error", str(e), parent=self.root)
            return

        # ── Report ────────────────────────────────────────────────────────────
        n_sessions = len(all_frames)
        n_trials   = len(merged)
        n_cols     = len(merged.columns)
        msg = (f"Group analysis complete.\n\n"
               f"Sessions merged:  {n_sessions}\n"
               f"Total trials:     {n_trials}\n"
               f"Columns:          {n_cols}\n\n"
               f"Saved to:\n{out_path}")
        if skipped:
            msg += f"\n\nSkipped ({len(skipped)}):\n" + "\n".join(skipped)
        messagebox.showinfo("Done", msg, parent=self.root)
        self._s2_status.config(
            text=f"✔  Exported {n_trials} trials from {n_sessions} sessions → "
                 f"{os.path.basename(out_path)}")

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
