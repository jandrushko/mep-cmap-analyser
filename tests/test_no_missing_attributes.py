"""
Catch reads of attributes and methods that do not exist.

Python resolves ``self.foo`` at runtime, so a typo or an invented name is
invisible until that line executes -- and in a GUI, that can mean a handler
that only fires when someone clicks a particular control. pyflakes does not
look at attribute access, so nothing in the existing checks catches it.

This has now bitten three times in one working session, each time in a
callback:

  * ``self.stim_types_found``     -- invented; the real name is
                                    ``_current_stim_types``
  * ``self._harvest_labels_tab``  -- called before it was written
  * ``self._set_confirm_state``   -- likewise

Every one imported cleanly, passed the whole suite, and failed in front of the
analyst. The check below is crude -- it cannot see attributes created
dynamically -- but it reads the whole class in a second and would have caught
all three.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Modules whose main class is worth checking. Mixin bases are resolved so a
# method defined in a mixin does not read as missing.
TARGETS = [
    ("app.py", "TMSAnalysisApp",
     ["stage2.py", "filter_preview.py", "bidsify_tab.py", "preview.py",
      "conditions_tab.py", "reprocess_ui.py"]),
]


def _class_names(path, class_name):
    """Attributes assigned and methods defined on a class, including bases."""
    src = (ROOT / "mep_cmap" / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if class_name and cls.name != class_name:
            continue
        # class-level constants
        for st in cls.body:
            if isinstance(st, ast.Assign):
                names |= {t.id for t in st.targets if isinstance(t, ast.Name)}
        for n in ast.walk(cls):
            if isinstance(n, ast.FunctionDef):
                names.add(n.name)
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "self"
                    and isinstance(n.ctx, (ast.Store, ast.Del))):
                names.add(n.attr)
            # setattr(self, "x", ...) is an assignment too
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "setattr" and len(n.args) >= 2
                    and isinstance(n.args[0], ast.Name)
                    and n.args[0].id == "self"
                    and isinstance(n.args[1], ast.Constant)):
                names.add(n.args[1].value)
    return names, src


@pytest.mark.parametrize("path,class_name,mixins", TARGETS)
def test_every_attribute_read_is_defined_somewhere(path, class_name, mixins):
    known, src = _class_names(path, class_name)
    for m in mixins:
        extra, _ = _class_names(m, None)
        known |= extra

    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)

    missing = {}
    for n in ast.walk(cls):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self" and isinstance(n.ctx, ast.Load)
                and n.attr not in known):
            missing.setdefault(n.attr, n.lineno)

    assert not missing, (
        "read but never assigned or defined:\n" +
        "\n".join(f"    line {ln}: self.{name}"
                  for name, ln in sorted(missing.items(), key=lambda kv: kv[1]))
    )


def test_the_check_would_catch_an_invented_name():
    """Guards the guard: a check that cannot fail is worse than none."""
    src = '''
class Thing:
    def a(self):
        self.real = 1
    def b(self):
        return self.invented
'''
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    known = {n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)}
    known |= {n.attr for n in ast.walk(cls)
              if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    reads = {n.attr for n in ast.walk(cls)
             if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)}
    assert reads - known == {"invented"}
