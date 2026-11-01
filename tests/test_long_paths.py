"""
Windows' 260-character path limit (v1.4.9).

A lab PC with the study a few folders deep in OneDrive failed a run at its
first figure: the BIDS layout, a full BIDS prefix and a condition label made
a 268-character path, and Windows reported it as "[Errno 2] No such file or
directory". Outputs are now written through the extended-length form
(utils.long_path), which no Windows setting is needed for.
"""

import glob
import json
import os
import pathlib

import pytest

from mep_cmap.utils import (long_path, display_path, path_too_long_hint,
                            WINDOWS_MAX_PATH)

PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"
WINDOWS = os.name == "nt"
only_windows = pytest.mark.skipif(not WINDOWS, reason="Windows path limit")


def _deep(tmp_path, over=300):
    """A folder under tmp_path whose path is longer than ``over``."""
    p = str(tmp_path)
    i = 0
    while len(p) <= over:
        p = os.path.join(p, f"Movement Prep TMS pilot folder {i}")
        i += 1
    return p


# ── the helpers ──────────────────────────────────────────────────────────────

@only_windows
def test_long_path_round_trips(tmp_path):
    lp = long_path(str(tmp_path))
    assert lp.startswith("\\\\?\\")
    assert long_path(lp) == lp                      # idempotent
    assert display_path(lp) == os.path.abspath(str(tmp_path))


@only_windows
def test_network_shares_use_the_unc_form():
    lp = long_path(r"\\server\share\study\derivatives")
    assert lp == r"\\?\UNC\server\share\study\derivatives"
    assert display_path(lp) == r"\\server\share\study\derivatives"


def test_unchanged_off_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert long_path("/home/user/study") == "/home/user/study"
    assert long_path("") == ""


@only_windows
def test_hint_explains_a_long_path_failure():
    p = "C:\\" + "x" * 300 + "\\fig.png"
    hint = path_too_long_hint(FileNotFoundError(2, "No such file", p))
    assert str(len(p)) in hint and "shorter folder" in hint
    assert path_too_long_hint(FileNotFoundError(2, "No such file", "C:\\a.png")) == ""
    assert path_too_long_hint(ValueError("x")) == ""


# ── every kind of output, at a path Windows would otherwise refuse ───────────

@only_windows
def test_every_writer_the_pipeline_uses_works_past_the_limit(tmp_path):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.backends.backend_agg as agg

    base = long_path(_deep(tmp_path))
    figs = os.path.join(base, "figures_with_outliers")
    os.makedirs(figs, exist_ok=True)
    name = "sub-test-cooper_ses-01_limb-right_task-finger_test002"
    assert len(display_path(figs)) > WINDOWS_MAX_PATH

    png = os.path.join(figs, f"{name}_stim-=·Simple_+35%25_traces_all.png")
    fig = matplotlib.figure.Figure(figsize=(1, 1))
    agg.FigureCanvasAgg(fig).print_figure(png, dpi=20)
    csv = os.path.join(figs, f"{name}_trials.csv")
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    npz = os.path.join(figs, f"{name}_segments.npz")
    np.savez(npz, x=np.zeros(3))
    js = os.path.join(figs, f"{name}_session.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"ok": True}, fh)

    for p in (png, csv, npz, js):
        assert os.path.isfile(p)
    # The stale-figure clean-up globs these folders; escaping keeps the "?"
    # of the prefix from being read as a wildcard.
    found = glob.glob(os.path.join(glob.escape(figs), "*_traces_all.png"))
    assert found == [png]


@only_windows
def test_plain_path_still_fails_so_the_test_means_something(tmp_path):
    # Without the prefix the same path must fail, or the tests above prove
    # nothing. A machine with long paths switched on has no limit to test.
    figs = _deep(tmp_path)
    try:
        os.makedirs(figs)
        with open(os.path.join(figs, "x.txt"), "w") as fh:
            fh.write("x")
    except OSError:
        return
    pytest.skip("long paths are enabled on this machine")


@only_windows
def test_archive_moves_past_the_limit(tmp_path):
    from mep_cmap import archive_superseded as AR
    deriv = os.path.join(_deep(tmp_path, over=200), "derivatives")
    src = os.path.join(deriv, "sub-001", "ses-01", "results",
                       "sub-001_ses-01_" + "x" * 60 + "_trials.csv")
    os.makedirs(long_path(os.path.dirname(src)))
    with open(long_path(src), "w") as fh:
        fh.write("x")
    plan = AR.ArchivePlan(derivatives_root=deriv, items=[AR.ArchiveItem(
        path=src, reason="legacy-layout",
        session_dir=os.path.dirname(os.path.dirname(src)),
        prefix="sub-001_ses-01", mtime=0.0)])
    out = AR.apply_archive(plan, AR.default_archive_root(deriv))
    assert out["moved"] == 1 and not out["failed"]


# ── the paths that matter go through it ──────────────────────────────────────

def test_wired_in_where_it_must_be():
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert "_deriv_base = long_path(session_output_dir(" in pipe
    assert pipe.count("glob.glob(os.path.join(glob.escape(") == 2
    assert "webbrowser.open(display_path(" in pipe
    assert "path_too_long_hint(e)" in pipe
    addons = (PKG / "addons.py").read_text(encoding="utf-8")
    assert "segments_npz_path = long_path(segments_npz_path)" in addons
    # ...and what add-ons report back is shown to people, so it is plain.
    assert "display_path(str(p)) for p in paths" in addons
    app = (PKG / "app.py").read_text(encoding="utf-8")
    assert 'open(_lp(save_path), "w"' in app
    assert "os.path.isfile(_lp_chk(fe.derivatives_json))" in app
    arch = (PKG / "archive_superseded.py").read_text(encoding="utf-8")
    assert "shutil.move(_lp(item.path), _lp(target))" in arch
    mig = (PKG / "layout_migration.py").read_text(encoding="utf-8")
    assert "shutil.move(long_path(src), long_path(dst))" in mig
