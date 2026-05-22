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
        • "Skip header rows" spinbox: ignore N non-numeric rows at the top
          of the file (e.g. metadata, channel-name headers).

  1 — Time Column / Row
        • Column-wise: auto-detects a monotonically increasing, evenly-spaced
          column and pre-selects it as the time axis.
        • Row-wise: user selects which row is the time axis (or "None").
        • Sampling rate is inferred from the time axis (or entered manually).

  2 — Channel Definition
        • Column-wise: one row per data column, excluding the time column.
        • Row-wise: one row per data ROW, excluding the time row.
        • Each row shows:  index label | mini waveform plot | signal name |
                           role dropdown | unit entry.
        • Roles: EMG  /  Stim/Trigger  /  Ignore

  3 — Summary & Save
        • Shows a compact summary of choices.
        • "Save" writes the sidecar JSON and calls the on_complete callback.

Usage
-----
    wizard = FormatWizard(parent_root, file_path, on_complete=callback)

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

import re
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



def _parse_header(filepath: str, delimiter: str) -> dict:
    """
    Scan the top of a file for non-numeric header content.

    Returns a dict with:
      skip_rows    : int       — number of lines before the first all-numeric row
      channel_names: list[str] — names parsed from a channel-identifier header row,
                                 or None if not found
      fs_detected  : float     — sampling rate parsed from a metadata line
                                 (e.g. "Sample Clock Rate,2000.00"), or None

    This handles formats such as KinEMG / NI-DAQ CSV exports which have:
      Line 0: Author,KinEMG
      Line 1: TimeStamp,1/31/2017 ...
      Line 2: Sample Clock Rate,2000.00
      Line 3: (blank)
      Line 4: Dev1/ai0,Dev1/ai1,...      ← channel names
      Line 5: 0.006,-0.008,...           ← data begins

    For files with no header (e.g. Delsys row-wise .txt) skip_rows=0 is
    returned and the data loads correctly without any lines being skipped.
    """
    _FS_RE  = re.compile(
        r'(?:sample\s*(?:clock\s*)?rate|sampling\s*(?:frequency|rate)|\bfs\b|\bhz\b)'
        r'[,:\s=]+([0-9]+(?:\.[0-9]*)?)',
        re.IGNORECASE,
    )
    _ID_RE  = re.compile(r'^[A-Za-z][A-Za-z0-9/._\-]*$')

    channel_names: Optional[list[str]] = None
    fs_detected:   Optional[float]     = None
    n_skip = 0

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            for i, raw_line in enumerate(fh):
                if i > 50:          # never scan more than 50 lines
                    break
                line = raw_line.strip()

                if not line:        # blank line → skip it
                    n_skip = i + 1
                    continue

                parts = [p.strip() for p in line.split(delimiter)]
                parts = [p for p in parts if p]

                # Try parsing as all-numeric data
                try:
                    [float(p) for p in parts]
                    break           # first numeric row → stop scanning
                except ValueError:
                    pass

                # Scan for embedded sampling rate
                m = _FS_RE.search(line)
                if m:
                    try:
                        fs_detected = float(m.group(1))
                    except ValueError:
                        pass

                # Detect a channel-name row: all parts match an identifier
                # pattern AND there is more than one part (not a key/value pair
                # like "Author,KinEMG" where the value contains spaces/digits).
                if len(parts) > 1 and all(_ID_RE.match(p) for p in parts):
                    channel_names = parts

                n_skip = i + 1
    except Exception:
        pass

    return {
        'skip_rows':     n_skip,
        'channel_names': channel_names,
        'fs_detected':   fs_detected,
    }


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
        self._raw: Optional[np.ndarray] = None  # loaded data array (after skip)

        # ── State vars (set after page 0 choices) ─────────────────────────────
        self._delimiter_var  = tk.StringVar(value='tab')
        self._layout_var     = tk.StringVar(value='column_wise')
        self._skip_rows_var  = tk.StringVar(value='0')   # ← new
        self._time_col_var   = tk.StringVar(value='None')
        self._fs_var         = tk.StringVar(value='')

        # Per-channel widgets (page 2) — filled in _populate_channels_page
        self._ch_name_vars:  list[tk.StringVar]  = []
        self._ch_role_vars:  list[tk.StringVar]  = []
        self._ch_unit_vars:  list[tk.StringVar]  = []

        # Auto-detected header information (populated by _parse_header)
        self._header_info: dict = {}   # keys: skip_rows, channel_names, fs_detected

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

        # Auto-detect header structure before loading data.
        # Try comma first (most structured formats use CSV); only adopt the
        # result if meaningful info was found (fs or channel names).  This
        # avoids incorrectly switching a pure-tab file to comma delimiter.
        _comma_info = _parse_header(self.file_path, ',')
        if _comma_info.get('fs_detected') or _comma_info.get('channel_names'):
            self._header_info = _comma_info
            self._delimiter_var.set('comma')
            self._skip_rows_var.set(str(self._header_info.get('skip_rows', 0)))
            if self._header_info.get('fs_detected'):
                self._fs_var.set(str(int(self._header_info['fs_detected'])))
        else:
            # Fall back to tab (or whichever default was set)
            self._header_info = _parse_header(
                self.file_path, _delimiter_char_for(self._delimiter_var.get())
            )
            if self._header_info.get('skip_rows', 0) > 0:
                self._skip_rows_var.set(str(self._header_info['skip_rows']))
            if self._header_info.get('fs_detected'):
                self._fs_var.set(str(int(self._header_info['fs_detected'])))

        # Load data immediately so all pages can reference it
        self._load_data(
            delimiter=_delimiter_char_for(self._delimiter_var.get()),
            skip=int(self._skip_rows_var.get()),
        )

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

    def _load_data(self, delimiter: str = '\t', skip: int = 0) -> None:
        """Load (or reload) the raw array with the given delimiter and skip count."""
        try:
            self._raw = np.loadtxt(
                self.file_path, delimiter=delimiter,
                skiprows=skip, max_rows=10000,
            )
        except Exception:
            try:
                self._raw = np.loadtxt(
                    self.file_path, skiprows=skip, max_rows=4000,
                )
            except Exception:
                self._raw = np.zeros((10, 2))
        # Guarantee 2-D
        if self._raw is not None and self._raw.ndim == 1:
            self._raw = self._raw.reshape(1, -1)

    # ─────────────────────────────────────────────────────────────────────────
    # Page builders
    # ─────────────────────────────────────────────────────────────────────────

    # ── Page 0: Data Preview & Layout ────────────────────────────────────────

    def _build_preview_page(self) -> None:
        page = tk.Frame(self._content, bg=_BG, padx=16, pady=12)
        self._pages.append(page)

        _section(page, "Data Preview")
        # Build description text, noting any auto-detected header info
        _desc_parts = [f"File: {os.path.basename(self.file_path)}"]
        _hi = self._header_info
        if _hi.get('skip_rows', 0) > 0:
            _desc_parts.append(
                f"✓  {_hi['skip_rows']} header line(s) detected and will be skipped automatically."
            )
        if _hi.get('fs_detected'):
            _desc_parts.append(
                f"✓  Sampling rate {int(_hi['fs_detected'])} Hz read from file header."
            )
        if _hi.get('channel_names'):
            _desc_parts.append(
                f"✓  {len(_hi['channel_names'])} channel name(s) found: "
                + ', '.join(_hi['channel_names'][:6])
                + (f' … (+{len(_hi["channel_names"])-6} more)' if len(_hi['channel_names']) > 6 else '')
            )
        if _hi.get('skip_rows', 0) == 0:
            _desc_parts.append(
                "No header was detected.  This wizard will help you define\n"
                "the signal layout so the file can be analysed."
            )
        tk.Label(
            page,
            text='\n'.join(_desc_parts),
            bg=_BG, fg=_FG, justify='left', wraplength=580,
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

        # Skip header rows ← new
        _section(page, "Skip Header Rows")
        skip_frame = tk.Frame(page, bg=_BG)
        skip_frame.pack(anchor='w', pady=(0, 8))
        tk.Label(
            skip_frame,
            text="Rows to ignore at the top of the file\n"
                 "(non-numeric metadata or channel-name headers):",
            bg=_BG, fg=_FG, justify='left',
        ).pack(side='left', padx=(0, 8))
        tk.Spinbox(
            skip_frame, textvariable=self._skip_rows_var,
            from_=0, to=100, width=5,
            command=self._on_skip_change,
        ).pack(side='left')

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
            l_frame, text='Row-wise  (each row = one channel/trial, columns = time samples)',
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
        d    = _delimiter_char_for(self._delimiter_var.get())
        # Re-run header detection with the new delimiter
        self._header_info = _parse_header(self.file_path, d)
        # Update skip_rows and fs if the new detection found something
        if self._header_info.get('skip_rows', 0) > 0:
            self._skip_rows_var.set(str(self._header_info['skip_rows']))
        if self._header_info.get('fs_detected'):
            self._fs_var.set(str(int(self._header_info['fs_detected'])))
        skip = self._get_skip()
        self._load_data(delimiter=d, skip=skip)
        self._refresh_preview_table()

    def _on_skip_change(self, *_) -> None:
        d    = _delimiter_char_for(self._delimiter_var.get())
        skip = self._get_skip()
        self._load_data(delimiter=d, skip=skip)
        self._refresh_preview_table()

    def _get_skip(self) -> int:
        try:
            return max(0, int(self._skip_rows_var.get()))
        except (ValueError, TypeError):
            return 0

    # ── Page 1: Time Column/Row & Sampling Rate ───────────────────────────────

    def _build_time_page(self) -> None:
        page = tk.Frame(self._content, bg=_BG, padx=16, pady=12)
        self._pages.append(page)

        # Determine orientation at build time; we'll also refresh on show.
        # We use a container frame and rebuild its contents when the page is shown.
        self._time_page_frame = page
        self._time_page_content: Optional[tk.Frame] = None

    def _populate_time_page(self) -> None:
        """Rebuild the time-page contents based on current layout choice."""
        page = self._time_page_frame

        # Remove previous content
        if self._time_page_content is not None:
            self._time_page_content.destroy()
        self._time_page_content = tk.Frame(page, bg=_BG)
        self._time_page_content.pack(fill='both', expand=True)
        frame = self._time_page_content

        layout = self._layout_var.get()

        if layout == 'column_wise':
            _section(frame, "Time Column")
            tk.Label(
                frame,
                text=("Select the column that contains the time axis (in seconds),\n"
                      "or choose 'None' if there is no time column."),
                bg=_BG, fg=_FG, justify='left',
            ).pack(anchor='w', pady=(0, 6))

            n_cols    = self._raw.shape[1] if self._raw is not None else 8
            time_opts = ['None'] + [f'Column {i}' for i in range(min(16, n_cols))]
            auto_col  = self._detect_time_col()
            default   = f'Column {auto_col}' if auto_col is not None else 'None'
            self._time_col_var.set(default)

        else:  # row_wise
            _section(frame, "Time Row")
            tk.Label(
                frame,
                text=("Select the row that contains the time axis (in seconds),\n"
                      "or choose 'None' if there is no time row."),
                bg=_BG, fg=_FG, justify='left',
            ).pack(anchor='w', pady=(0, 6))

            n_rows    = self._raw.shape[0] if self._raw is not None else 8
            time_opts = ['None'] + [f'Row {i}' for i in range(n_rows)]
            auto_col  = self._detect_time_row()
            default   = f'Row {auto_col}' if auto_col is not None else 'None'
            self._time_col_var.set(default)

        self._time_cb = ttk.Combobox(
            frame, textvariable=self._time_col_var,
            values=time_opts, state='readonly', width=16,
        )
        self._time_cb.pack(anchor='w', pady=(0, 4))
        self._time_cb.bind('<<ComboboxSelected>>', self._on_time_col_change)

        if layout == 'column_wise':
            auto_col = self._detect_time_col()
            if auto_col is not None:
                tk.Label(frame,
                         text=f'✓  Column {auto_col} auto-detected as a uniform time axis.',
                         bg=_BG, fg='#388e3c').pack(anchor='w', pady=(0, 8))
            else:
                tk.Label(frame,
                         text='No uniform time axis detected — please select one if available.',
                         bg=_BG, fg='#e65100').pack(anchor='w', pady=(0, 8))
        else:
            auto_col = self._detect_time_row()
            if auto_col is not None:
                tk.Label(frame,
                         text=f'✓  Row {auto_col} auto-detected as a uniform time axis.',
                         bg=_BG, fg='#388e3c').pack(anchor='w', pady=(0, 8))
            else:
                tk.Label(frame,
                         text='No uniform time row detected — please select one if available.',
                         bg=_BG, fg='#e65100').pack(anchor='w', pady=(0, 8))

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=8)

        _section(frame, "Sampling Rate")
        self._fs_info_lbl = tk.Label(frame, text='', bg=_BG, fg='#388e3c')
        self._fs_info_lbl.pack(anchor='w')

        fs_frame = tk.Frame(frame, bg=_BG)
        fs_frame.pack(anchor='w', pady=4)
        tk.Label(fs_frame, text='Sampling rate (Hz):', bg=_BG, fg=_FG).pack(
            side='left', padx=(0, 6))
        self._fs_entry = tk.Entry(
            fs_frame, textvariable=self._fs_var, width=10)
        self._fs_entry.pack(side='left')

        # Trigger initial update
        self._on_time_col_change()

    def _detect_time_col(self) -> Optional[int]:
        """Return the index of the most likely time column (column-wise), or None."""
        if self._raw is None or self._raw.ndim < 2:
            return None
        n_rows, n_cols = self._raw.shape
        for c in range(min(n_cols, 4)):
            col = self._raw[:, c]
            if n_rows < 4:
                continue
            diffs = np.diff(col[:min(500, n_rows)])
            if diffs.min() <= 0:
                continue
            rel_std = diffs.std() / (abs(diffs.mean()) + 1e-12)
            if rel_std < 1e-4:
                return c
        return None

    def _detect_time_row(self) -> Optional[int]:
        """Return the index of the most likely time row (row-wise), or None."""
        if self._raw is None or self._raw.ndim < 2:
            return None
        n_rows, n_cols = self._raw.shape
        for r in range(min(n_rows, 4)):
            row = self._raw[r, :]
            if n_cols < 4:
                continue
            diffs = np.diff(row[:min(500, n_cols)])
            if diffs.min() <= 0:
                continue
            rel_std = diffs.std() / (abs(diffs.mean()) + 1e-12)
            if rel_std < 1e-4:
                return r
        return None

    def _on_time_col_change(self, *_) -> None:
        val    = self._time_col_var.get()
        layout = self._layout_var.get()

        if not hasattr(self, '_fs_info_lbl'):
            return

        if val == 'None' or self._raw is None:
            self._fs_info_lbl.config(text='')
            return

        try:
            idx = int(val.split()[-1])
        except (ValueError, IndexError):
            self._fs_info_lbl.config(text='')
            return

        if layout == 'column_wise':
            axis_data = self._raw[:500, idx]
        else:
            axis_data = self._raw[idx, :500]

        diffs = np.diff(axis_data)
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
        self._ch_page_desc_lbl = tk.Label(
            _section_frame,
            text="",
            bg=_BG, fg=_FG, justify='left',
        )
        self._ch_page_desc_lbl.pack(anchor='w', pady=(0, 4))

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

        # Header row (populated in _populate_channels_page to reflect orientation)
        self._ch_header_frame: Optional[tk.Frame] = None
        self._ch_canvas = canvas

    def _auto_detect_roles(
        self, indices: list[int], layout: str
    ) -> dict[int, tuple[str, str]]:
        """
        Heuristically suggest role and unit for each column (column-wise) or
        row (row-wise).

        Returns dict mapping index -> (role_label, unit_string).
        """
        if self._raw is None:
            return {}

        result: dict[int, tuple[str, str]] = {}
        stim_candidates: list[tuple[int, int]] = []

        for idx in indices:
            if layout == 'column_wise':
                data = self._raw[:, idx].astype(float)
            else:
                data = self._raw[idx, :].astype(float)

            mx = data.max()
            mn = data.min()
            unit = 'V' if abs(mx) > 1.0 or abs(mn) > 1.0 else 'mV'

            # Use a percentile-robust lower bound for the unipolar check so that
            # a single-sample startup artifact (e.g. the -0.75 V transient at
            # sample 0 from Delsys Trigno) does not disqualify the trigger channel.
            # The threshold itself still uses the global max: trigger pulses are
            # so sparse (~0.004 % of samples) that percentile-based peaks land
            # in the EMG noise floor.
            p001    = float(np.percentile(data, 0.1))
            unipolar = p001 >= -0.01 * max(abs(mx), 1e-9)
            pct_nz   = float(np.mean(np.abs(data) < 0.05 * max(abs(mx), 1e-9)))
            thr      = mx * 0.5
            above    = (data >= thr).astype(int)
            edges    = int(np.sum(np.diff(above) == 1))
            is_stim  = unipolar and pct_nz > 0.97 and edges >= 1

            if is_stim:
                stim_candidates.append((idx, edges))
                result[idx] = ('Stim/Trigger', unit)
            else:
                result[idx] = ('EMG', unit)

        # Keep only the best stim candidate
        if len(stim_candidates) > 1:
            best = max(stim_candidates, key=lambda x: x[1])
            for idx, _ in stim_candidates:
                if idx != best[0]:
                    result[idx] = ('EMG', result[idx][1])

        return result

    def _populate_channels_page(self) -> None:
        """
        Build one row per data column (column-wise) or data row (row-wise),
        excluding the time column/row.
        Called just before page 2 is shown, so layout/time choices are final.
        """
        # Clear existing rows
        for w in self._ch_frame.winfo_children():
            w.destroy()
        self._ch_name_vars.clear()
        self._ch_role_vars.clear()
        self._ch_unit_vars.clear()

        if self._raw is None:
            return

        layout       = self._layout_var.get()
        time_idx     = self._resolved_time_col()
        is_row_wise  = layout == 'row_wise'

        # Update description label
        if is_row_wise:
            self._ch_page_desc_lbl.config(
                text="For each row give a name, assign its role, and optionally\n"
                     "specify the unit.  Use 'Ignore' to exclude a row from analysis."
            )
        else:
            self._ch_page_desc_lbl.config(
                text="For each column give a name, assign its role, and optionally\n"
                     "specify the unit.  Use 'Ignore' to exclude a column from analysis."
            )

        # Build the list of indices to show
        if is_row_wise:
            n_items = self._raw.shape[0]
            axis_label = 'Row'
        else:
            n_items = self._raw.shape[1]
            axis_label = 'Col'

        indices = [i for i in range(n_items) if i != time_idx]

        # Auto-detect roles
        suggestions = self._auto_detect_roles(indices, layout)

        # Rebuild header row
        if self._ch_header_frame is not None:
            self._ch_header_frame.destroy()
        self._ch_header_frame = tk.Frame(self._ch_frame, bg='#bbdefb')
        self._ch_header_frame.pack(fill='x', pady=(0, 2))
        for text, width in [
            (axis_label, 4), ('Preview', 14), ('Signal Name', 18),
            ('Role', 14), ('Unit', 8),
        ]:
            tk.Label(
                self._ch_header_frame, text=text, bg='#bbdefb', fg=_FG,
                width=width, relief='groove', pady=3,
            ).pack(side='left', padx=1)

        for list_idx, data_idx in enumerate(indices):
            suggested_role, suggested_unit = suggestions.get(data_idx, ('EMG', 'mV'))
            # Use auto-detected channel name if available
            _detected_names = self._header_info.get('channel_names') or []
            _default_name = (
                _detected_names[data_idx]
                if data_idx < len(_detected_names)
                else f'Channel {list_idx + 1}'
            )
            name_var = tk.StringVar(value=_default_name)
            role_var = tk.StringVar(value=suggested_role)
            unit_var = tk.StringVar(value=suggested_unit)

            self._ch_name_vars.append(name_var)
            self._ch_role_vars.append(role_var)
            self._ch_unit_vars.append(unit_var)

            row_frame = tk.Frame(self._ch_frame, bg=_BG, pady=2)
            row_frame.pack(fill='x', padx=2)

            # Index label
            tk.Label(
                row_frame, text=str(data_idx), bg=_BG, fg=_FG, width=4,
            ).pack(side='left', padx=1)

            # Mini waveform canvas
            mini = tk.Canvas(
                row_frame, width=110, height=45, bg='white',
                highlightthickness=1, highlightbackground='#bbb',
            )
            mini.pack(side='left', padx=4)
            self._draw_mini_wave(mini, data_idx, layout)

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
        idx: int,
        layout: str,
    ) -> None:
        """Draw a thumbnail waveform on a tk.Canvas (no matplotlib)."""
        W, H = 110, 45
        try:
            if layout == 'column_wise':
                signal = self._raw[:, idx].astype(float)
                signal = signal[:min(4000, len(signal))]
            else:  # row_wise: idx is a row index
                signal = self._raw[idx, :].astype(float)

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
        skip  = cfg.get('skip_rows', 0)
        lines = [
            f"File       : {os.path.basename(self.file_path)}",
            f"Layout     : {cfg['layout']}",
            f"Delimiter  : {cfg['delimiter']}",
            f"Skip rows  : {skip}",
            f"Fs         : {cfg['fs']} Hz",
            f"Time col   : {cfg['time_col']}",
            "",
            f"{'Idx':<5}  {'Name':<22}  {'Role':<14}  {'Unit'}",
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
        if idx == 1:
            self._populate_time_page()
        elif idx == 2:
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
        layout       = self._layout_var.get()
        time_idx     = self._resolved_time_col()
        is_row_wise  = layout == 'row_wise'
        skip         = self._get_skip()

        if self._raw is not None:
            n_items = self._raw.shape[0] if is_row_wise else self._raw.shape[1]
        else:
            n_items = 0

        indices = [i for i in range(n_items) if i != time_idx]

        channels = []
        for list_idx, data_idx in enumerate(indices):
            if list_idx >= len(self._ch_name_vars):
                break
            role_label = self._ch_role_vars[list_idx].get()
            channels.append({
                'col':  data_idx,          # row index for row-wise, col for column-wise
                'name': self._ch_name_vars[list_idx].get() or f'Channel {list_idx + 1}',
                'role': _ROLE_KEYS.get(role_label, 'emg'),
                'unit': self._ch_unit_vars[list_idx].get() or '',
            })

        # Detect stacked trials (column-wise only)
        trials_stacked = False
        if layout == 'column_wise' and time_idx is not None:
            try:
                sep    = _delimiter_char_for(self._delimiter_var.get())
                _probe = np.loadtxt(
                    self.file_path, delimiter=sep,
                    skiprows=skip, max_rows=20000,
                )
                if _probe.ndim == 1:
                    _probe = _probe.reshape(1, -1)
                t_probe   = _probe[:, time_idx].astype(float)
                n_resets  = int(np.sum(np.diff(t_probe) < -1e-6))
                trials_stacked = n_resets > 0
            except Exception:
                if self._raw is not None:
                    t = self._raw[:, time_idx]
                    trials_stacked = int(np.sum(np.diff(t) < -1e-6)) > 0

        try:
            fs = int(float(self._fs_var.get()))
        except (ValueError, TypeError):
            fs = 1000

        return {
            'layout':         layout,
            'delimiter':      self._delimiter_var.get(),
            'skip_rows':      skip,
            'fs':             fs,
            'time_col':       time_idx,
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
        w, h = 660, 580
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
