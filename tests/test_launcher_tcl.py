"""
The built app's Tcl/Tk start-up (v1.4.11).

The 1.4.10 Windows build stopped at its first window with "Can't find a
usable init.tcl": TCL_LIBRARY was not set when Tcl looked for it, although the
files were bundled and PyInstaller's runtime hook was the same as 1.4.9's.
Started with TCL_LIBRARY already in its environment, the same build ran.

launcher.py now (1) sets TCL_LIBRARY / TK_LIBRARY itself before anything
imports tkinter, and (2) if the first window still fails on init.tcl,
relaunches once with the variables set from the start.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "launcher.py").read_text(encoding="utf-8")


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("_launcher_under_test",
                                                  ROOT / "launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # __main__ block does not run
    return mod


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    (tmp_path / "_tcl_data").mkdir()
    (tmp_path / "_tk_data").mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    for var in ("TCL_LIBRARY", "TK_LIBRARY", "MEPCMAP_TCL_RELAUNCHED"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# ── order ────────────────────────────────────────────────────────────────────

def test_folders_are_set_before_anything_imports_tkinter():
    call = SRC.index("\n_point_tcl_at_bundled_files()")
    assert call < SRC.index("from splash_screen import show_splash")
    assert call < SRC.index("import tkinter")


def test_the_first_window_is_guarded_by_the_relaunch():
    i = SRC.index("splash = show_splash()")
    block = SRC[i:i + 200]
    assert "_relaunch_with_tcl_env(" in block and "sys.exit(0)" in block


# ── part 1: set in-process ───────────────────────────────────────────────────

def test_sets_both_folders_in_a_built_app(launcher, frozen):
    launcher._point_tcl_at_bundled_files()
    assert os.environ["TCL_LIBRARY"] == str(frozen / "_tcl_data")
    assert os.environ["TK_LIBRARY"] == str(frozen / "_tk_data")


def test_keeps_a_valid_existing_value(launcher, frozen, monkeypatch, tmp_path_factory):
    mine = tmp_path_factory.mktemp("my_tcl")
    monkeypatch.setenv("TCL_LIBRARY", str(mine))
    launcher._point_tcl_at_bundled_files()
    assert os.environ["TCL_LIBRARY"] == str(mine)


def test_replaces_a_value_pointing_nowhere(launcher, frozen, monkeypatch):
    monkeypatch.setenv("TCL_LIBRARY", r"C:\nowhere\tcl8.6")
    launcher._point_tcl_at_bundled_files()
    assert os.environ["TCL_LIBRARY"] == str(frozen / "_tcl_data")


def test_does_nothing_when_run_from_source(launcher, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("TCL_LIBRARY", raising=False)
    launcher._point_tcl_at_bundled_files()
    assert "TCL_LIBRARY" not in os.environ


# ── part 2: relaunch once ────────────────────────────────────────────────────

ERR = "Can't find a usable init.tcl in the following directories: ..."


def _capture(monkeypatch):
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen",
                        lambda args, **kw: calls.append((args, kw)))
    return calls


def test_relaunches_once_with_the_folders_in_its_environment(launcher, frozen, monkeypatch):
    calls = _capture(monkeypatch)
    launcher._point_tcl_at_bundled_files()
    assert launcher._relaunch_with_tcl_env(Exception(ERR)) is True
    (args, kw), = calls
    assert args[0] == sys.executable
    env = kw["env"]
    assert env["TCL_LIBRARY"] == str(frozen / "_tcl_data")
    assert env["MEPCMAP_TCL_RELAUNCHED"] == "1"
    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"


def test_never_relaunches_twice(launcher, frozen, monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setenv("MEPCMAP_TCL_RELAUNCHED", "1")
    assert launcher._relaunch_with_tcl_env(Exception(ERR)) is False
    assert not calls


def test_other_errors_are_not_swallowed(launcher, frozen, monkeypatch):
    calls = _capture(monkeypatch)
    assert launcher._relaunch_with_tcl_env(Exception("no display")) is False
    assert not calls


def test_not_from_source(launcher, monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert launcher._relaunch_with_tcl_env(Exception(ERR)) is False
    assert not calls
