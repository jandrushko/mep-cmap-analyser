"""
Guard: every shipped module must be importable on the oldest supported Python.

`pyproject.toml` declares `requires-python = ">=3.9"`. The PEP 604 union
shorthand (`dict | None`) is only a valid runtime type expression from 3.10, and
annotations are evaluated at function-definition time unless a module opts into
PEP 563 with `from __future__ import annotations`. A module using the shorthand
without that import therefore raises TypeError the moment it is imported on 3.9
— the app cannot start at all.

This went unnoticed in `mep_cmap/app.py` because no test imports it, so the 3.9
CI leg exercised everything except the one module that was broken. A static
check catches it regardless of what the suite happens to import.

If this test fails, either add `from __future__ import annotations` as the first
statement after the module docstring, or rewrite the annotation as
`typing.Optional[...]`.
"""

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"


def _module_files():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _has_future_annotations(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                return True
    return False


def _pep604_annotations(tree):
    """Annotations using `X | Y`, which 3.9 cannot evaluate at runtime.

    Only annotations are collected. A `|` in ordinary expressions is fine — it
    is set/int union and has nothing to do with typing.
    """
    hits = []

    def _is_union(node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)

    for node in ast.walk(tree):
        anns = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            anns.append(node.returns)
            for grp in (a.args, a.posonlyargs, a.kwonlyargs):
                anns.extend(x.annotation for x in grp)
            for extra in (a.vararg, a.kwarg):
                if extra is not None:
                    anns.append(extra.annotation)
        elif isinstance(node, ast.AnnAssign):
            anns.append(node.annotation)

        for ann in anns:
            if ann is None:
                continue
            for sub in ast.walk(ann):
                if _is_union(sub):
                    hits.append(getattr(node, "lineno", ann.lineno))
                    break
    return hits


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_is_importable_on_python39(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if _has_future_annotations(tree):
        return                                  # annotations never evaluated
    hits = _pep604_annotations(tree)
    assert not hits, (
        f"{path.relative_to(PKG.parent)} uses the PEP 604 `X | Y` annotation "
        f"shorthand at line(s) {sorted(set(hits))} without "
        f"`from __future__ import annotations`. This raises TypeError on import "
        f"under Python 3.9. pyproject.toml now requires 3.10, but this "
        f"guard is kept while development environments (Apple's bundled "
        f"Python on macOS) are still 3.9."
    )


def test_guard_detects_a_known_bad_module(tmp_path):
    """The guard must actually fire — a check that silently passes is worthless."""
    bad = tmp_path / "bad.py"
    bad.write_text("def f(x: dict | None = None) -> str | None:\n    return None\n")
    tree = ast.parse(bad.read_text())
    assert not _has_future_annotations(tree)
    assert _pep604_annotations(tree)

    good = tmp_path / "good.py"
    good.write_text("from __future__ import annotations\n"
                    "def f(x: dict | None = None) -> str | None:\n    return None\n")
    assert _has_future_annotations(ast.parse(good.read_text()))


def test_guard_ignores_non_annotation_bitwise_or(tmp_path):
    """`|` outside an annotation is ordinary set/int union, not a typing union."""
    ok = tmp_path / "ok.py"
    ok.write_text("def f(a, b):\n    mask = a | b\n    return {1} | {2}, mask\n")
    assert not _pep604_annotations(ast.parse(ok.read_text()))
