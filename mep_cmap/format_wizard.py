"""
mep_cmap.format_wizard
~~~~~~~~~~~~~~~~~~~~~~~
FormatWizard — a multi-page Tkinter Toplevel that guides the user through
defining the layout and channel roles for a headerless generic-TSV data file.

Pages
-----
  0 — Data Preview & Layout
        • Shows the first few rows/columns of raw data.
        • User chooses column-wise vs row-wise orientation.
        • User confirms or changes the delimiter.

  1 — Time Column
        • Auto-detects a monotonically increasing, evenly-spaced column and
          pre-selects it as the time axis.
        • User can override or set "none".
        • Sampling rate is inferred from the time axis (or entered manually).

  2 — Channel Definition
        • One row per data column (column-wise) or data row (row-wise),
          excluding the time column.
        • Each row shows:  mini waveform plot | signal name entry | role dropdown
        • Roles: EMG  /  Stim/Trigger  /  Ignore
        • Unit entry (mV / V / µV / custom)

  3 — Summary & Save
        • Shows a compact summary of choices.
        • "Save" writes the sidecar JSON and calls the on_complete callback.

Usage
-----
    wizard = FormatWizard(parent_root, file_path, on_complete=callback)
    # parent_root.wait_window(wizard.top)   ← call from main thread only

on_complete is called with the saved config dict once the user clicks Save,
or with None if they cancel.

Thread safety
-------------
FormatWizard must be instantiated and driven from the Tk main thread.
app.py posts a request to the main thread via root.after() before calling
this; never instantiate from a background worker thread.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Callable, Optional

import numpy as np
import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ── Role options ──────────────────────────────────────────────────────────────
_ROLES = ["EMG", "Stim/Trigger", "Ignore"]
_ROLE_KEYS = {"EMG": "emg", "Stim/Trigger": "stim", "Ignore": "ignore"}

def _delimiter_char_for(name: str) -> str:
    """Convert delimiter name string to actual character."""
    return {'tab': '\t', 'space': ' ', 'comma': ','}.get(name, '\t')

# ── Colour palette (matches app.py style) ─────────────────────────────────────
_ACCENT  = "#2196F3"
_BG      = "#f5f5f5"
_FG      = "#212121"
_BTN_FG  = "white"

_PREVIEW_ROWS = 6   # rows shown in the data-preview table
_PREVIEW_COLS = 8   # max columns shown


class FormatWizard:
    """
    Multi-page wizard for defining a generic-TSV file's layout.

    Parameters
    ----------
    parent : tk.Tk or tk.Toplevel
    file_path : str
        Absolute path to the data file.
    on_complete : callable(cfg | None)
        Called on the main thread once the user saves or cancels.
        Receives the config dict on success, or None on cancel.
    """

    def __init__(
        self,
        parent: tk.Misc,
        file_path: str,
        on_complete: Optional[Callable] = None,
    ):
        self.parent       = parent
        self.file_path    = file_path
        self.on_complete  = on_complete
        self._cfg: dict   = {}            # built up across pages
        self._raw: Optional[np.ndarray] = None  # loaded data array

        # ── State vars (set after page 0 choices) ─────────────────────────────
        self._delimiter_var  = tk.StringVar(value='tab')
        self._layout_var     = tk.StringVar(value='column_wise')
        self._time_col_var   = tk.StringVar(value='None')
        self._fs_var         = tk.StringVar(value='')

        # Per-channel widgets (page 2) — filled in _build_channels_page
        self._ch_name_vars:  list[tk.StringVar]  = []
        self._ch_role_vars:  list[tk.StringVar]  = []
        self._ch_unit_vars:  list[tk.StringVar]  = []

        # ── Create window ─────────────────────────────────────────────────────
        self.top = tk.Toplevel(parent)
        self.top.title("Format Wizard — Define Signal Layout")
        self.top.resizable(True, True)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._page_index = 0
        self._pages: list[tk.Frame] = []

        # Content area (pages swapped here) + navigation bar
        self._content = tk.Frame(self.top, bg=_BG)
        self._content.pack(fill='both', expand=True, padx=0, pady=0)

        self._nav = tk.Frame(self.top, bg='#e0e0e0', pady=6)
        self._nav.pack(fill='x', side='bottom')

        self._btn_back = tk.Button(
            self._nav, text='← Back', width=10,
            command=self._go_back, state='disabled',
            bg='#9e9e9e', fg=_BTN_FG, relief='flat', bd=0,
        )
        self._btn_back.pack(side='left', padx=12)

        self._btn_next = tk.Button(
            self._nav, text='Next →', width=10,
            command=self._go_next,
            bg=_ACCENT, fg=_BTN_FG, relief='flat', bd=0,
        )
        self._btn_next.pack(side='right', padx=12)

        self._btn_cancel = tk.Button(
            self._nav, text='Cancel', width=8,
            command=self._on_cancel,
            bg='#e53935', fg=_BTN_FG, relief='flat', bd=0,
        )
        self._btn_cancel.pack(side='right', padx=4)

        # Progress label
        self._progress_lbl = tk.Label(
            self._nav, text='Step 1 of 4', bg='#e0e0e0', fg='#555',
        )
        self._progress_lbl.pack(side='left', padx=8)

        # Load data immediately so all pages can reference it
        self._load_data()

        # Build all pages (hidden until shown)
        self._build_preview_page()     # page 0
        self._build_time_page()        # page 1
        self._build_channels_page()    # page 2
        self._build_summary_page()     # page 3

        self._show_page(0)
        self._centre()

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self, delimiter: str = '\t') -> None:
        """Load (or reload) the raw array with the given delimiter."""
        try:
            # Load enough rows to span at least 2 trials for trials_stacked detection.
            # 10000 rows covers most sampling rates (e.g. 4 kHz × 2.5 s).
            self._raw = np.loadtxt(self.file_path, delimiter=delimiter, max_rows=10000)
        except Exception:
            try:
                self._raw = np.loadtxt(self.file_path, max_rows=4000)
            except Exception:
                self._raw = np.zeros((10, 2))

    # ─────────────────────────────────────────────────────────────────────────
    # Page builders
    # ─────────────────────────────────────────────────────────────────────────

    # ── Page 0: Data Preview & Layout ────────────────────────────────────────

    def _build_preview_page(self) -> None:
        page = tk.Frame(self._content, bg=_BG, padx=16, pady=12)
        self._pages.append(page)

        _section(page, "Data Preview")
        tk.Label(
            page,
            text=(f"File: {os.path.basename(self.file_path)}\n"
                  "No header was detected.  This wizard will help you define\n"
                  "the signal layout so the file can be analysed."),
            bg=_BG, fg=_FG, justify='left', wraplength=560,
        ).pack(anchor='w', pady=(0, 8))

        # Raw data table
        frame_tbl = tk.Frame(page, bg=_BG)
        frame_tbl.pack(fill='x', pady=(0, 10))
        self._preview_table = frame_tbl
        self._refresh_preview_table()

        ttk.Separator(page, orient='horizontal').pack(fill='x', pady=8)

        # Delimiter
        _section(page, "Delimiter")
        d_frame = tk.Frame(page, bg=_BG)
        d_frame.pack(anchor='w', pady=(0, 8))
        for label, val in [('Tab', 'tab'), ('Space', 'space'), ('Comma', 'comma')]:
            tk.Radiobutton(
                d_frame, text=label, variable=self._delimiter_var,
                value=val, bg=_BG, fg=_FG,
                command=self._on_delimiter_change,
            ).pack(side='left', padx=8)

        ttk.Separator(page, orient='horizontal').pack(fill='x', pady=8)

        # Layout
        _section(page, "Data Orientation")
        l_frame = tk.Frame(page, bg=_BG)
        l_frame.pack(anchor='w')
        tk.Radiobutton(
            l_frame, text='Column-wise  (each column = one signal, rows = time samples)',
            variable=self._layout_var, value='column_wise',
            bg=_BG, fg=_FG,
        ).pack(anchor='w', pady=2)
        tk.Radiobutton(
            l_frame, text='Row-wise  (each row = one trial/sweep, columns = time samples)',
            variable=self._layout_var, value='row_wise',
            bg=_BG, fg=_FG,
        ).pack(anchor='w', pady=2)

    def _refresh_preview_table(self) -> None:
        for w in self._preview_table.winfo_children():
            w.destroy()
        if self._raw is None:
            return
        n_rows = min(_PREVIEW_ROWS, self._raw.shape[0])
        n_cols = min(_PREVIEW_COLS, self._raw.shape[1])
        # Header row
        for c in range(n_cols):
            tk.Label(
                self._preview_table, text=f'Col {c}',
                bg='#bbdefb', fg=_FG, width=11,
                relief='groove', padx=2, pady=2,
            ).grid(row=0, column=c, sticky='ew', padx=1, pady=1)
        # Data rows
        for r in range(n_rows):
            for c in range(n_cols):
                val = self._raw[r, c]
                tk.Label(
                    self._preview_table,
                    text=f'{val:.6g}',
                    bg='white', fg=_FG, width=11,
                    relief='groove', padx=2, pady=1,
                ).grid(row=r + 1, column=c, sticky='ew', padx=1, pady=1)
        if self._raw.shape[1] > _PREVIEW_COLS:
            tk.Label(
                self._preview_table,
                text=f'… {self._raw.shape[1] - _PREVIEW_COLS} more cols',
                bg=_BG, fg='#888',
            ).grid(row=0, column=_PREVIEW_COLS, padx=4)

    def _on_delimiter_change(self) -> None:
        d = {'tab': '\t', 'space': ' ', 'comma': ','}.get(
            self._delimiter_var.get(), '\t')
        self._load_data(delimiter=d)
        self._refresh_preview_table()

    # ── Page 1: Time Column & Sampling Rate ───────────────────────────────────

    def _build_time_page(self) -> None:
        page = tk.Frame(self._content, bg=_BG, padx=16, pady=12)
        self._pages.append(page)

        _section(page, "Time Column")
        tk.Label(
            page,
            text=("Select the column that contains the time axis (in seconds),\n"
                  "or choose 'None' if there is no time column."),
            bg=_BG, fg=_FG, justify='left',
        ).pack(anchor='w', pady=(0, 6))

        time_opts = ['None'] + [f'Column {i}' for i in range(
            min(16, self._raw.shape[1]) if self._raw is not None else 8
        )]
        auto_col = self._detect_time_col()
        default  = f'Column {auto_col}' if auto_col is not None else 'None'
        self._time_col_var.set(default)

        self._time_cb = ttk.Combobox(
            page, textvariable=self._time_col_var,
            values=time_opts, state='readonly', width=16,
        )
        self._time_cb.pack(anchor='w', pady=(0, 4))
        self._time_cb.bind('<<ComboboxSelected>>', self._on_time_col_change)

        if auto_col is not None:
            tk.Label(
                page,
                text=f'✓  Column {auto_col} auto-detected as a uniform time axis.',
                bg=_BG, fg='#388e3c',
            ).pack(anchor='w', pady=(0, 8))
        else:
            tk.Label(
                page,
                text='No uniform time axis detected — please select one if available.',
                bg=_BG, fg='#e65100',
            ).pack(anchor='w', pady=(0, 8))

        ttk.Separator(page, orient='horizontal').pack(fill='x', pady=8)

        _section(page, "Sampling Rate")
        self._fs_info_lbl = tk.Label(page, text='', bg=_BG, fg='#388e3c')
        self._fs_info_lbl.pack(anchor='w')

        fs_frame = tk.Frame(page, bg=_BG)
        fs_frame.pack(anchor='w', pady=4)
        tk.Label(fs_frame, text='Sampling rate (Hz):', bg=_BG, fg=_FG).pack(
            side='left', padx=(0, 6))
        self._fs_entry = tk.Entry(
            fs_frame, textvariable=self._fs_var, width=10)
        self._fs_entry.pack(side='left')

        # Trigger initial update
        self._on_time_col_change()

    def _detect_time_col(self) -> Optional[int]:
        """Return the index of the most likely time column, or None."""
        if self._raw is None or self._raw.ndim < 2:
            return None
        n_rows, n_cols = self._raw.shape
        best = None
        for c in range(min(n_cols, 4)):   # only inspect left columns
            col = self._raw[:, c]
            if n_rows < 4:
                continue
            diffs = np.diff(col[:min(500, n_rows)])
            if diffs.min() <= 0:
                continue  # not monotonically increasing
            rel_std = diffs.std() / (abs(diffs.mean()) + 1e-12)
            if rel_std < 1e-4:           # very uniform spacing
                best = c
                break
        return best

    def _on_time_col_change(self, *_) -> None:
        val = self._time_col_var.get()
        if val == 'None' or self._raw is None:
            self._fs_info_lbl.config(text='')
            return
        col = int(val.split()[-1])
        col_data = self._raw[:500, col]
        diffs = np.diff(col_data)
        if len(diffs) < 2 or abs(diffs.mean()) < 1e-9:
            self._fs_info_lbl.config(text='')
            return
        inferred = round(1.0 / abs(diffs.mean()))
        self._fs_var.set(str(inferred))
        self._fs_info_lbl.config(
            text=f'✓  Sampling rate inferred from time axis: {inferred} Hz',
            fg='#388e3c',
        )

    # ── Page 2: Channel Definition ────────────────────────────────────────────

    def _build_channels_page(self) -> None:
        page = tk.Frame(self._content, bg=_BG)
        self._pages.append(page)

        _section_frame = tk.Frame(page, bg=_BG, padx=16, pady=8)
        _section_frame.pack(fill='x')
        _section(_section_frame, "Define Signals")
        tk.Label(
            _section_frame,
            text=("For each column give a name, assign its role, and optionally\n"
                  "specify the unit.  At least one channel must be 'Stim/Trigger'."),
            bg=_BG, fg=_FG, justify='left',
        ).pack(anchor='w', pady=(0, 4))

        # Scrollable channel list
        outer = tk.Frame(page, bg=_BG)
        outer.pack(fill='both', expand=True, padx=8, pady=4)

        canvas = tk.Canvas(outer, bg=_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        self._ch_frame = tk.Frame(canvas, bg=_BG)

        self._ch_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')),
        )
        canvas.create_window((0, 0), window=self._ch_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        # Header row
        hdr = tk.Frame(self._ch_frame, bg='#bbdefb')
        hdr.pack(fill='x', pady=(0, 2))
        for text, width in [
            ('Col', 4), ('Preview', 14), ('Signal Name', 18),
            ('Role', 14), ('Unit', 8),
        ]:
            tk.Label(
                hdr, text=text, bg='#bbdefb', fg=_FG,
                width=width, relief='groove', pady=3,
            ).pack(side='left', padx=1)

        # Will be populated in _populate_channels_page()
        self._ch_canvas = canvas

    def _auto_detect_roles(self, cols: list[int]) -> dict[int, tuple[str, str]]:
        """
        Heuristically suggest role and unit for each column.

        Returns dict mapping col_index -> (role_label, unit_string).

        Stim/Trigger heuristic (all three must hold):
          1. Unipolar: min >= -1% of max  (stim pulses don't go negative)
          2. High duty-cycle near zero: >97% of samples within 5% of max
          3. At least one rising edge above 50% threshold

        Unit heuristic:
          max amplitude > 1.0  →  'V'   (stim / nerve channels)
          max amplitude <= 1.0 →  'mV'  (typical EMG)
        """
        if self._raw is None:
            return {}

        # Count trials to help rank stim candidates
        time_col = self._resolved_time_col()
        n_trials = 1
        if time_col is not None:
            t = self._raw[:, time_col].astype(float)
            resets = np.where(np.diff(t) < -1e-6)[0]
            n_trials = len(resets) + 1

        stim_candidates: list[tuple[int, int]] = []  # (col, edge_count)
        result: dict[int, tuple[str, str]] = {}

        for col in cols:
            data = self._raw[:, col].astype(float)
            mx   = data.max()
            mn   = data.min()

            # Unit guess from amplitude scale
            unit = 'V' if abs(mx) > 1.0 or abs(mn) > 1.0 else 'mV'

            # Stim heuristic
            unipolar    = mn >= -0.01 * max(abs(mx), 1e-9)
            pct_nz      = float(np.mean(np.abs(data) < 0.05 * max(abs(mx), 1e-9)))
            thr         = mx * 0.5
            above       = (data >= thr).astype(int)
            edges       = int(np.sum(np.diff(above) == 1))
            is_stim     = unipolar and pct_nz > 0.97 and edges >= 1

            if is_stim:
                stim_candidates.append((col, edges))
                result[col] = ('Stim/Trigger', unit)
            else:
                result[col] = ('EMG', unit)

        # If multiple stim candidates, keep only the best one
        # (prefer the one whose edge count is closest to n_trials)
        if len(stim_candidates) > 1:
            best = min(stim_candidates,
                       key=lambda x: abs(x[1] - n_trials))
            for col, _ in stim_candidates:
                if col != best[0]:
                    # Downgrade to EMG — keep unit guess
                    result[col] = ('EMG', result[col][1])

        return result

    def _populate_channels_page(self) -> None:
        """
        Build one row per data column (excluding the time column).
        Called just before page 2 is shown, so layout/time choices are final.
        """
        # Clear existing rows
        for w in self._ch_frame.winfo_children():
            if isinstance(w, tk.Frame) and w.cget('bg') != '#bbdefb':
                w.destroy()
        self._ch_name_vars.clear()
        self._ch_role_vars.clear()
        self._ch_unit_vars.clear()

        if self._raw is None:
            return

        time_col = self._resolved_time_col()
        n_cols   = self._raw.shape[1]
        layout   = self._layout_var.get()

        cols = [i for i in range(n_cols) if i != time_col]

        # Auto-detect roles and units
        suggestions = self._auto_detect_roles(cols)
        stim_count  = sum(1 for r, _ in suggestions.values() if r == 'Stim/Trigger')

        for idx, col in enumerate(cols):
            suggested_role, suggested_unit = suggestions.get(col, ('EMG', 'mV'))
            name_var = tk.StringVar(value=f'Channel {idx + 1}')
            role_var = tk.StringVar(value=suggested_role)
            unit_var = tk.StringVar(value=suggested_unit)

            self._ch_name_vars.append(name_var)
            self._ch_role_vars.append(role_var)
            self._ch_unit_vars.append(unit_var)

            row_frame = tk.Frame(self._ch_frame, bg=_BG, pady=2)
            row_frame.pack(fill='x', padx=2)

            # Column index label
            tk.Label(
                row_frame, text=str(col), bg=_BG, fg=_FG, width=4,
            ).pack(side='left', padx=1)

            # Mini waveform canvas
            mini = tk.Canvas(
                row_frame, width=110, height=45, bg='white',
                highlightthickness=1, highlightbackground='#bbb',
            )
            mini.pack(side='left', padx=4)
            self._draw_mini_wave(mini, col, layout)

            # Name entry
            tk.Entry(row_frame, textvariable=name_var, width=18).pack(
                side='left', padx=4)

            # Role combobox
            cb = ttk.Combobox(
                row_frame, textvariable=role_var,
                values=_ROLES, state='readonly', width=13,
            )
            cb.pack(side='left', padx=4)

            # Unit entry
            tk.Entry(row_frame, textvariable=unit_var, width=7).pack(
                side='left', padx=4)

        # Force canvas to resize
        self._ch_frame.update_idletasks()
        self._ch_canvas.configure(scrollregion=self._ch_canvas.bbox('all'))

    def _draw_mini_wave(
        self,
        canvas: tk.Canvas,
        col: int,
        layout: str,
    ) -> None:
        """Draw a thumbnail waveform on a tk.Canvas (no matplotlib)."""
        W, H = 110, 45
        try:
            if layout == 'column_wise':
                signal = self._raw[:, col].astype(float)
                # Use first trial only if stacked (first 4000 samples max)
                signal = signal[:min(4000, len(signal))]
            else:
                signal = self._raw[col, :].astype(float)

            # Downsample to canvas width
            n = len(signal)
            if n > W:
                step   = n // W
                signal = signal[::step]
            if len(signal) < 2:
                return

            mn, mx = signal.min(), signal.max()
            rng = mx - mn
            if rng < 1e-12:
                rng = 1.0

            def _y(v):
                return int(H - 3 - (v - mn) / rng * (H - 6))

            pts: list[float] = []
            for xi, v in enumerate(signal):
                pts.extend([xi * W / len(signal), _y(v)])

            if len(pts) >= 4:
                canvas.create_line(pts, fill='#1565c0', width=1, smooth=False)
        except Exception:
            canvas.create_text(55, 22, text='preview\nunavailable',
                               fill='#aaa', justify='center', font=('Arial', 7))

    # ── Page 3: Summary & Save ────────────────────────────────────────────────

    def _build_summary_page(self) -> None:
        page = tk.Frame(self._content, bg=_BG, padx=16, pady=12)
        self._pages.append(page)

        _section(page, "Summary")

        self._summary_text = tk.Text(
            page, height=14, width=60,
            bg='white', fg=_FG, relief='groove',
            font=('Courier', 10), wrap='word',
            state='disabled',
        )
        self._summary_text.pack(fill='both', expand=True, pady=(4, 8))

        tk.Label(
            page,
            text='Click Save to write the configuration alongside the data file.',
            bg=_BG, fg='#555',
        ).pack(anchor='w')

    def _populate_summary_page(self) -> None:
        cfg   = self._build_config()
        lines = [
            f"File       : {os.path.basename(self.file_path)}",
            f"Layout     : {cfg['layout']}",
            f"Delimiter  : {cfg['delimiter']}",
            f"Fs         : {cfg['fs']} Hz",
            f"Time col   : {cfg['time_col']}",
            "",
            f"{'Col':<5}  {'Name':<22}  {'Role':<14}  {'Unit'}",
            "─" * 52,
        ]
        for ch in cfg['channels']:
            lines.append(
                f"{ch['col']:<5}  {ch['name']:<22}  {ch['role']:<14}  {ch.get('unit','')}"
            )
        self._summary_text.config(state='normal')
        self._summary_text.delete('1.0', 'end')
        self._summary_text.insert('end', '\n'.join(lines))
        self._summary_text.config(state='disabled')

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────────────────────────

    def _show_page(self, idx: int) -> None:
        for p in self._pages:
            p.pack_forget()

        # Pre-populate dynamic pages just before showing them
        if idx == 2:
            self._populate_channels_page()
        elif idx == 3:
            self._populate_summary_page()
            self._btn_next.config(text='Save', bg='#43a047')

        self._pages[idx].pack(fill='both', expand=True)
        self._page_index = idx
        n = len(self._pages)
        self._progress_lbl.config(text=f'Step {idx + 1} of {n}')
        self._btn_back.config(state='normal' if idx > 0 else 'disabled')
        if idx < n - 1:
            self._btn_next.config(text='Next →', bg=_ACCENT)

    def _go_next(self) -> None:
        if self._page_index == 0:
            if not self._validate_preview():
                return
        elif self._page_index == 1:
            if not self._validate_time():
                return
        elif self._page_index == 2:
            if not self._validate_channels():
                return
        elif self._page_index == 3:
            self._save_and_finish()
            return
        self._show_page(self._page_index + 1)

    def _go_back(self) -> None:
        if self._page_index > 0:
            self._show_page(self._page_index - 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_preview(self) -> bool:
        if self._raw is None or self._raw.size == 0:
            messagebox.showerror(
                "Cannot read file",
                "The file could not be parsed with the selected delimiter.\n"
                "Please choose a different delimiter.",
                parent=self.top,
            )
            return False
        return True

    def _validate_time(self) -> bool:
        try:
            fs = int(float(self._fs_var.get()))
            if fs < 1:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror(
                "Sampling rate required",
                "Please enter a valid sampling rate in Hz (e.g. 4000).",
                parent=self.top,
            )
            return False
        return True

    def _validate_channels(self) -> bool:
        if not self._ch_role_vars:
            messagebox.showerror(
                "No channels",
                "No data columns were found.  Check the delimiter setting.",
                parent=self.top,
            )
            return False
        roles = [v.get() for v in self._ch_role_vars]
        if 'Stim/Trigger' not in roles:
            return messagebox.askyesno(
                "No Stim/Trigger channel",
                "No channel has been assigned the 'Stim/Trigger' role.\n\n"
                "Stimulation timing cannot be auto-detected without a trigger channel.\n"
                "Continue anyway?",
                parent=self.top,
            )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Config assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _resolved_time_col(self) -> Optional[int]:
        val = self._time_col_var.get()
        if val == 'None':
            return None
        try:
            return int(val.split()[-1])
        except (ValueError, IndexError):
            return None

    def _build_config(self) -> dict:
        time_col = self._resolved_time_col()
        n_cols   = self._raw.shape[1] if self._raw is not None else 0
        cols     = [i for i in range(n_cols) if i != time_col]

        channels = []
        for idx, col in enumerate(cols):
            if idx >= len(self._ch_name_vars):
                break
            role_label = self._ch_role_vars[idx].get()
            channels.append({
                'col':  col,
                'name': self._ch_name_vars[idx].get() or f'Channel {idx + 1}',
                'role': _ROLE_KEYS.get(role_label, 'emg'),
                'unit': self._ch_unit_vars[idx].get() or '',
            })

        # Detect whether the time axis resets (stacked trials).
        # Read the first 20000 rows from disk to ensure we span at least 2 trials
        # regardless of how many rows were loaded for the preview.
        trials_stacked = False
        if time_col is not None:
            try:
                sep = _delimiter_char_for(self._delimiter_var.get())
                _probe = np.loadtxt(self.file_path, delimiter=sep, max_rows=20000)
                t_probe = _probe[:, time_col].astype(float)
                n_resets = int(np.sum(np.diff(t_probe) < -1e-6))
                trials_stacked = n_resets > 0
            except Exception:
                # Fall back to preview array if file read fails
                if self._raw is not None:
                    t = self._raw[:, time_col]
                    trials_stacked = int(np.sum(np.diff(t) < -1e-6)) > 0

        try:
            fs = int(float(self._fs_var.get()))
        except (ValueError, TypeError):
            fs = 1000

        return {
            'layout':         self._layout_var.get(),
            'delimiter':      self._delimiter_var.get(),
            'fs':             fs,
            'time_col':       time_col,
            'trials_stacked': trials_stacked,
            'channels':       channels,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Save & finish
    # ─────────────────────────────────────────────────────────────────────────

    def _save_and_finish(self) -> None:
        from .formats.generic_tsv import save_config
        cfg = self._build_config()
        try:
            save_config(self.file_path, cfg)
        except Exception as exc:
            messagebox.showerror(
                "Save failed",
                f"Could not write config file:\n{exc}",
                parent=self.top,
            )
            return
        self._cfg = cfg
        self.top.destroy()
        if self.on_complete:
            self.on_complete(cfg)

    def _on_cancel(self) -> None:
        self.top.destroy()
        if self.on_complete:
            self.on_complete(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def _centre(self) -> None:
        self.top.update_idletasks()
        self.top.minsize(620, 480)
        w, h = 660, 560
        px = self.parent.winfo_x() + (self.parent.winfo_width()  - w) // 2
        py = self.parent.winfo_y() + (self.parent.winfo_height() - h) // 2
        self.top.geometry(f'{w}x{h}+{px}+{py}')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(parent: tk.Widget, text: str) -> tk.Label:
    lbl = tk.Label(
        parent, text=text,
        bg=_BG, fg=_ACCENT,
        font=('TkDefaultFont', 10, 'bold'),
    )
    lbl.pack(anchor='w', pady=(6, 2))
    return lbl
