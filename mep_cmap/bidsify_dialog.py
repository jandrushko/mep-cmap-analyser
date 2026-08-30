"""
mep_cmap.bidsify_dialog
~~~~~~~~~~~~~~~~~~~~~~~~
BidsifyDialog — a Tkinter Toplevel that collects the shared settings for a
BIDS-ify run: the NIBS modality, the BEP037 metadata fields (rendered from the
schema), and the operation settings (EDF/BDF container, power-line frequency,
stim marker label).

It deliberately does NOT pick files or resolve BIDS entities — that stays in
app.py, which already owns the file tree and `_parse_bids_from_filename`. This
dialog returns a :class:`BidsifyDialogResult`; app.py combines it with the
per-file entities to build the BidsifyItems and the Plan.

Fields are generated from mep_cmap.bids_schema: only the selected modality's
fields are shown, grouped by block (device / parameters / targeting), with
advanced fields hidden behind a toggle. Enum fields become read-only combo
boxes; everything else is an entry. Values for shared keys (e.g. Manufacturer)
are preserved when the modality is switched.

Usage (main thread only, mirrors FormatWizard)
----------------------------------------------
    BidsifyDialog(parent_root, on_complete=callback)

on_complete is called with a BidsifyDialogResult on OK, or None on cancel.

Thread safety: instantiate and drive from the Tk main thread only. The actual
BIDS-ify work (execute_plan) runs on a worker thread in app.py, not here.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── Palette (matches app.py / format_wizard) ──────────────────────────────────
_ACCENT = "#2196F3"
_BG     = "#f5f5f5"
_FG     = "#212121"
_BTN_FG = "white"

#: Fallback only. The list actually offered comes from the schema's own
#: `modalities`, because a second hardcoded copy is a list that silently goes
#: stale: PNS was added to the schema for v6.3 (M-waves, H-reflexes and CMAPs
#: are peripheral) and this dropdown carried on offering three.
_MODALITIES = ["TMS", "tES", "TUS", "PNS"]


def _modalities_for(schema):
    """Modalities the schema declares, falling back to the constant above."""
    try:
        vals = [m for m in (getattr(schema, "modalities", None) or [])
                if m and m != "common"]
    except Exception:                       # noqa: BLE001 — never block the dialogue
        vals = []
    return vals or list(_MODALITIES)

_GROUP_TITLES = {"device": "Device", "parameters": "Stimulation parameters",
                 "targeting": "Targeting", "event": "Events"}


# ── Result ────────────────────────────────────────────────────────────────────
@dataclass
class BidsifyDialogResult:
    modality:       str
    sidecar_values: dict
    container:      str          # 'EDF' or 'BDF'
    powerline_hz:   int
    marker_name:    str


# ── Pure helpers (unit-testable without a display) ────────────────────────────
def field_widget_kind(fld) -> str:
    """Which widget a schema field needs: 'combo' (enum) or 'entry'."""
    if getattr(fld, "enum", None):
        return "combo"
    return "entry"


def assemble_result(modality: str, raw_values: dict, container: str,
                    powerline_hz, marker_name: str) -> BidsifyDialogResult:
    """
    Build a BidsifyDialogResult from raw (string) widget values. Blank values are
    dropped; StimulationModality is forced to the selected modality; powerline is
    coerced to int with a 50 Hz fallback; container is validated.
    """
    vals = {k: v for k, v in raw_values.items()
            if v not in (None, "") and str(v).strip() != ""}
    vals["StimulationModality"] = modality
    try:
        pl = int(float(powerline_hz))
    except (TypeError, ValueError):
        pl = 50
    return BidsifyDialogResult(
        modality=modality,
        sidecar_values=vals,
        container=container if container in ("EDF", "BDF") else "BDF",
        powerline_hz=pl,
        marker_name=(marker_name or "A"),
    )


# ── Dialog ────────────────────────────────────────────────────────────────────
class BidsifyDialog:
    def __init__(self, parent: tk.Misc,
                 schema=None,
                 on_complete: Optional[Callable] = None,
                 defaults: Optional[dict] = None):
        if schema is None:
            from .bids_schema import load_schema
            schema = load_schema()
        self.parent = parent
        self.schema = schema
        self.on_complete = on_complete
        self.defaults = defaults or {}

        # persistent value vars, keyed by schema field key (kept across modality switches)
        self._field_vars: dict = {}

        self.modality_var  = tk.StringVar(value=self.defaults.get("modality", "TMS"))
        self.container_var  = tk.StringVar(value=self.defaults.get("container", "BDF"))
        self.powerline_var  = tk.StringVar(value=str(self.defaults.get("powerline_hz", 50)))
        self.marker_var     = tk.StringVar(value=self.defaults.get("marker_name", "A"))
        self.show_adv_var   = tk.BooleanVar(value=False)

        # seed any provided default field values
        for k, v in (self.defaults.get("sidecar_values") or {}).items():
            self._var_for(k).set("" if v is None else str(v))

        # ── window ────────────────────────────────────────────────────────────
        self.top = tk.Toplevel(parent)
        self.top.title("BIDS-ify — Stimulation metadata")
        self.top.configure(bg=_BG)
        self.top.resizable(True, True)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_header()
        self._build_scroll_area()
        self._build_footer()
        self._rebuild_fields()
        self._centre()

    # ---- var management -------------------------------------------------------
    def _var_for(self, key: str) -> tk.StringVar:
        if key not in self._field_vars:
            self._field_vars[key] = tk.StringVar(value="")
        return self._field_vars[key]

    # ---- layout ---------------------------------------------------------------
    def _build_header(self) -> None:
        hdr = tk.Frame(self.top, bg=_BG)
        hdr.pack(fill="x", padx=14, pady=(12, 6))

        tk.Label(hdr, text="Stimulation modality", bg=_BG, fg=_FG,
                 font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
        mod = ttk.Combobox(hdr, textvariable=self.modality_var,
                           values=_modalities_for(self.schema),
                           state="readonly", width=10)
        mod.grid(row=0, column=1, sticky="w", padx=(8, 0))
        mod.bind("<<ComboboxSelected>>", lambda _e: self._rebuild_fields())

        tk.Checkbutton(hdr, text="Show advanced fields", variable=self.show_adv_var,
                       bg=_BG, fg=_FG, activebackground=_BG,
                       command=self._rebuild_fields).grid(row=0, column=2,
                                                          sticky="e", padx=(20, 0))
        hdr.columnconfigure(2, weight=1)

    def _build_scroll_area(self) -> None:
        wrap = tk.Frame(self.top, bg=_BG)
        wrap.pack(fill="both", expand=True, padx=14, pady=4)

        self._canvas = tk.Canvas(wrap, bg=_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._fields_frame = tk.Frame(self._canvas, bg=_BG)
        self._win = self._canvas.create_window((0, 0), window=self._fields_frame,
                                               anchor="nw")
        self._fields_frame.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfigure(self._win, width=e.width))
        # mousewheel
        #
        # bind_all is application-wide, so this binding OUTLIVES the dialogue:
        # once the window closes, the next wheel event anywhere still calls
        # back into a destroyed canvas and raises "invalid command name". The
        # handler therefore checks the widget is still alive and removes itself
        # when it is not, rather than assuming its own lifetime.
        def _wheel(e):
            try:
                if not self._canvas.winfo_exists():
                    raise tk.TclError("canvas gone")
                self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except tk.TclError:
                try:
                    self.top.unbind_all("<MouseWheel>")
                except Exception:       # noqa: BLE001 — already torn down
                    pass

        self._canvas.bind_all("<MouseWheel>", _wheel)

    def _build_footer(self) -> None:
        opt = tk.Frame(self.top, bg="#ececec")
        opt.pack(fill="x", side="bottom")

        row = tk.Frame(opt, bg="#ececec")
        row.pack(fill="x", padx=14, pady=8)

        tk.Label(row, text="Container:", bg="#ececec", fg=_FG).pack(side="left")
        # BDF first, matching the default. EDF's 16-bit step is coarse enough
        # to be visible in a small MEP at EMG gains, so BDF is the safer
        # starting point; EDF stays available for anyone who needs it.
        for c in ("BDF", "EDF"):
            tk.Radiobutton(row, text=c + "+", value=c, variable=self.container_var,
                           bg="#ececec", activebackground="#ececec").pack(side="left")

        tk.Label(row, text="   Power line (Hz):", bg="#ececec",
                 fg=_FG).pack(side="left")
        tk.Entry(row, textvariable=self.powerline_var, width=6).pack(side="left")

        tk.Label(row, text="   Stim marker label(s):", bg="#ececec",
                 fg=_FG).pack(side="left")
        tk.Entry(row, textvariable=self.marker_var, width=16).pack(side="left")
        tk.Label(row, text="(one or more codes, comma-separated, e.g. A,B,C,D; "
                 "or the stim channel name for all codes)", bg="#ececec",
                 fg="#666", font="TkSmallCaptionFont").pack(side="left", padx=(6, 0))

        btns = tk.Frame(opt, bg="#ececec")
        btns.pack(fill="x", padx=14, pady=(0, 10))
        tk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right")
        tk.Button(btns, text="OK", bg=_ACCENT, fg=_BTN_FG,
                  command=self._on_ok).pack(side="right", padx=(0, 8))
        tk.Button(btns, text="Validate", command=self._on_validate).pack(side="right",
                                                                         padx=(0, 8))

    # ---- field rendering ------------------------------------------------------
    def _rebuild_fields(self) -> None:
        for w in self._fields_frame.winfo_children():
            w.destroy()

        modality = self.modality_var.get()
        include_adv = self.show_adv_var.get()
        row = 0
        for group in ("device", "parameters", "targeting"):
            flds = self.schema.fields_for(modality, group=group,
                                          include_advanced=include_adv)
            if not flds:
                continue
            tk.Label(self._fields_frame, text=_GROUP_TITLES.get(group, group),
                     bg=_BG, fg=_ACCENT,
                     font=("TkDefaultFont", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(10, 2))
            row += 1
            for fld in flds:
                self._add_field_row(fld, row)
                row += 1

        self._fields_frame.columnconfigure(1, weight=1)

    def _add_field_row(self, fld, row: int) -> None:
        req = {"required": " *", "recommended": "", "optional": ""}[fld.level]
        unit = f"  [{fld.units}]" if fld.units else ""
        label = f"{fld.key}{req}{unit}"
        lbl = tk.Label(self._fields_frame, text=label, bg=_BG, fg=_FG, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=(4, 8), pady=1)
        if fld.level == "required":
            lbl.configure(font=("TkDefaultFont", 9, "bold"))

        var = self._var_for(fld.key)
        if field_widget_kind(fld) == "combo":
            values = [""] + list(fld.enum)
            w = ttk.Combobox(self._fields_frame, textvariable=var, values=values,
                             state="readonly")
        else:
            w = tk.Entry(self._fields_frame, textvariable=var)
        w.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=1)
        if fld.description:
            self._tooltip(lbl, fld.description)

    def _tooltip(self, widget, text: str) -> None:
        # lightweight hover tooltip
        tip = {"win": None}

        def show(_e):
            if tip["win"] or not text:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=text, bg="#ffffe0", fg=_FG, relief="solid",
                     borderwidth=1, wraplength=320, justify="left",
                     font=("TkDefaultFont", 8)).pack()
            tip["win"] = tw

        def hide(_e):
            if tip["win"]:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ---- collection / validation ----------------------------------------------
    def _collect_raw(self) -> dict:
        modality = self.modality_var.get()
        applicable = {f.key for f in self.schema.fields_for(modality)}
        return {k: v.get() for k, v in self._field_vars.items()
                if k in applicable and v.get().strip() != ""}

    def _current_result(self) -> BidsifyDialogResult:
        return assemble_result(self.modality_var.get(), self._collect_raw(),
                               self.container_var.get(), self.powerline_var.get(),
                               self.marker_var.get())

    def _on_validate(self) -> None:
        res = self._current_result()
        vr = self.schema.validate(res.sidecar_values, modality=res.modality)
        messagebox.showinfo("Validation", vr.summary(), parent=self.top)

    def _on_ok(self) -> None:
        res = self._current_result()
        vr = self.schema.validate(res.sidecar_values, modality=res.modality)
        if not vr.ok:
            messagebox.showerror("Cannot continue",
                                 "Please fix these before continuing:\n\n"
                                 + "\n".join(f"• {e}" for e in vr.errors),
                                 parent=self.top)
            return
        if vr.warnings:
            proceed = messagebox.askyesno(
                "Recommended fields missing",
                f"{len(vr.warnings)} recommended field(s) are blank. "
                "BIDS-ify anyway?", parent=self.top)
            if not proceed:
                return
        self.top.destroy()
        if self.on_complete:
            self.on_complete(res)

    def _on_cancel(self) -> None:
        self.top.destroy()
        if self.on_complete:
            self.on_complete(None)

    # ---- geometry -------------------------------------------------------------
    def _centre(self) -> None:
        self.top.update_idletasks()
        self.top.minsize(560, 520)
        w, h = 640, 680
        try:
            px = self.parent.winfo_x() + (self.parent.winfo_width() - w) // 2
            py = self.parent.winfo_y() + (self.parent.winfo_height() - h) // 2
        except Exception:
            px, py = 200, 100
        self.top.geometry(f"{w}x{h}+{max(0, px)}+{max(0, py)}")
