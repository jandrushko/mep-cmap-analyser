"""
mep_cmap.tk_compat
~~~~~~~~~~~~~~~~~~
The handful of places where Tk behaves differently depending on which
windowing system it is running under.

Kept in one module because these differences are invisible on the machine you
develop on. A binding that is simply never delivered produces no error, no log
line and no failing test: the feature just does nothing, on someone else's
computer, and the report comes back weeks later as "the right-click menu
doesn't work on Mac".
"""

from __future__ import annotations

from tkinter import ttk


def is_aqua(widget) -> bool:
    """True when Tk is drawing through macOS Aqua.

    Asked of Tk rather than read from ``sys.platform`` because what matters is
    which Tk is running, not which operating system: a Mac using an X11 Tk
    reports "x11" and behaves the X11 way throughout.
    """
    try:
        return widget.tk.call("tk", "windowingsystem") == "aqua"
    except Exception:                   # noqa: BLE001 — a widget without a Tk
        return False


def bind_context_menu(widget, handler) -> None:
    """Bind ``handler`` to whatever counts as a right-click here.

    Aqua numbers the mouse buttons differently from X11 and Windows: the RIGHT
    button arrives as Button-2, and Button-3 is the MIDDLE button. So the
    obvious ``bind("<Button-3>")`` listens, on a Mac, for a button most mice do
    not have, and the menu never opens.

    Control-click is not a separate button either. It arrives as Button-1 with
    a modifier, so it needs its own binding.

    Bound per platform rather than binding all of them everywhere, so that
    middle-clicking a row on Windows or Linux does not start opening context
    menus that were never meant to appear there.
    """
    if is_aqua(widget):
        widget.bind("<Button-2>", handler)
        widget.bind("<Control-Button-1>", handler)
    else:
        widget.bind("<Button-3>", handler)


def maximise(root) -> None:
    """Open the window filling the screen, on every platform.

    The naive version of this is ``root.state("zoomed")`` everywhere, and it
    is wrong on exactly one platform in a way that does not raise.

    On Windows, zoomed means maximise: fill the display. On Aqua, "zoom" is
    what the green traffic-light button does, which is size the window TO FIT
    ITS CONTENT. Given a scrollable container whose requested width exceeds
    the display, Tk obligingly makes the window wider than the screen, and
    because nothing has constrained it, later geometry propagation grows it
    further. The call succeeds, so a try/except fallback never fires.

    On Aqua the window is therefore sized explicitly to the usable area. That
    area comes from ``wm maxsize`` rather than ``winfo_screenheight`` because
    Tk already sets it to the work area, with the menu bar AND the Dock
    excluded. maxsize is then pinned, which is what stops the window growing
    past the screen when the content asks for more room.

    Positioned at +0+0 and left alone: Aqua refuses to put a title bar under
    the menu bar and slides the window down by exactly the right amount, which
    is better than any arithmetic here could do. Subtracting the work area
    from the screen height would charge the window for the Dock as well and
    push it down twice as far as it should go.
    """
    if is_aqua(root):
        root.update_idletasks()
        try:
            mw, mh = root.wm_maxsize()
        except Exception:               # noqa: BLE001 — fall back to raw screen
            mw, mh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.maxsize(mw, mh)
        root.geometry(f"{mw}x{mh}+0+0")
        return

    try:
        root.state("zoomed")            # Windows
    except Exception:                   # noqa: BLE001 — X11 has no zoomed state
        root.attributes("-zoomed", True)


def fit_to_screen(root, *, floor_w: int = 680, floor_h: int = 600) -> None:
    """Clamp a shrink-wrapped window to the usable screen area.

    ``root.geometry("")`` asks Tk to size the window to whatever its content
    requests, which is right on a large display and catastrophic on a small
    one: the request is honoured literally, so the window opens wider than the
    screen with its right-hand edge unreachable.

    Pinning ``minsize`` to that size, as the obvious next line does, is the
    worse half. It makes the oversized dimensions the FLOOR, so dragging the
    window edge cannot bring it back, and the window appears to grow whenever
    it is touched. The size is not growing; it cannot shrink.

    So both are clamped to the work area. On a display large enough for the
    content this changes nothing at all, which is why the fault only ever
    appeared on laptops.
    """
    root.update_idletasks()
    try:
        mw, mh = root.wm_maxsize()
    except Exception:                   # noqa: BLE001 — fall back to raw screen
        mw, mh = root.winfo_screenwidth(), root.winfo_screenheight()

    w = min(max(root.winfo_width(), floor_w), mw)
    h = min(max(root.winfo_height(), floor_h), mh)

    root.maxsize(mw, mh)
    # The floor is the content size, or the screen where the content does not
    # fit. Never larger than the screen, or the window cannot be resized at all.
    root.minsize(min(w, mw), min(h, mh))
    root.geometry(f"{w}x{h}")


def attach_hscroll(container, canvas, body, *, side="bottom"):
    """Give a scrolling canvas a horizontal scrollbar that appears when needed.

    Returns a ``sync()`` to call whenever the geometry changes. Kept here, and
    shared, because the two scrolling surfaces in this application were built
    separately and only one of them could scroll sideways: the setup table on
    tab 1a simply ran off the edge with no way to reach it. Whether content
    fits is a property of the font and the platform, not of the layout, so
    every scrolling body needs this rather than the ones that happened to
    overflow on the machine it was written on.

    The scrollbar is packed and unpacked rather than left in place, because a
    permanent horizontal scrollbar on a panel whose content fits reads as a
    layout that is broken when it is not.
    """
    hscroll = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
    canvas.configure(xscrollcommand=hscroll.set)

    def sync(_event=None):
        need = body.winfo_reqwidth()
        have = canvas.winfo_width()
        if need > have > 1:
            if not hscroll.winfo_ismapped():
                hscroll.pack(side=side, fill="x", before=canvas)
        elif hscroll.winfo_ismapped():
            hscroll.pack_forget()
            canvas.xview_moveto(0)

    def _wheel(event):
        delta = event.delta
        if delta:
            canvas.xview_scroll(
                int(-delta / 120) or (-1 if delta > 0 else 1), "units")
        return "break"

    # Shift+wheel scrolls sideways, which is what anyone reaches for before
    # dragging a scrollbar.
    canvas.bind("<Shift-MouseWheel>", _wheel)
    body.bind("<Shift-MouseWheel>", _wheel)
    return sync
