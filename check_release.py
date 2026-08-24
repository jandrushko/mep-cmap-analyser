#!/usr/bin/env python
"""
Pre-release check: every version string in the repo must agree with
mep_cmap/bids.py TOOL_VERSION (the single source of truth).

    python check_release.py

Exits non-zero if anything disagrees, so it can gate a release.
Not packaged — a developer utility that lives at the repo root.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# Files that carry a release version, and patterns that are NOT the tool
# version (schema versions, DOIs, Python requirements) and must be ignored.
FILES = [
    "pyproject.toml",
    "CITATION.cff",
    "zenodo.json",
    "README.md",
    "MEP_CMAP_Windows.spec",
    "MEP_CMAP_Mac.spec",
    "MEP_CMAP_Linux.spec",
]

# Only the CFF schema declaration is skipped outright.
IGNORE_LINE = re.compile(r"cff-version", re.IGNORECASE)

# Everything else is *stripped from the line* rather than skipping the line, so
# one incidental match cannot mask a real one. zenodo.json holds its whole
# changelog on a single line that also says "Python 3.9 or later"; skipping the
# line on that basis hid a stale "Changes in version 1.2.6" sitting beside it.
STRIP_NOISE = re.compile(
    r"https?://\S+"                 # URLs (carry journal DOIs)
    r"|\b10\.\d{4,}/\S+"           # bare DOIs: 10.1093/brain/113.6.1843
    r"|[Pp]ython[_ -]?requires\s*=?\s*[\"\']?[><=~!\d. ,]+"
    r"|[Pp]ython \d[\d.]*"           # "Python 3.9 or later"
    r"|setuptools[><=~!\d. ]*"
    r"|[><=~!]=?\s*\d[\d.]*"        # dependency pins: pyedflib>=0.1.30
    # Historical changelog headings. "## What's New in 1.3.2" sitting under a
    # 1.3.3 release is correct -- it describes what 1.3.2 changed and must not
    # be rewritten. Without this the check fails on every release that keeps a
    # previous section, and a check that cries wolf every time stops being read.
    # The authoritative version strings (the header line and the citation) are
    # still checked, so nothing real is masked.
    r"|What's New in \d[\d.]*"
    # Statements about which versions carried which licence. "Versions 1.3.3
    # and earlier were released under the MIT Licence" is a permanent fact
    # about the past, not a version string that needs bumping -- and rewriting
    # it to say 1.4.0 would make it false. Narrow on purpose: it matches the
    # phrasing, not any mention of a number near the word licence.
    #
    # Both spellings, and both cases. The abbreviated form ("from v1.4.0",
    # "v1.3.3 and earlier") and a sentence-initial "From version" were missed,
    # so 1.4.0 -> 1.4.1 reported the licence history in pyproject.toml and the
    # README as out of sync and failed the release gate on facts that were
    # correct.
    r"|[Vv]ersions? \d[\d.]* and earlier"
    r"|v\d[\d.]* and earlier"
    r"|[Ff]rom [Vv]ersions? \d[\d.]*"
    r"|[Ff]rom v\d[\d.]*"
    # Statements about when a feature BEGAN. "Sessions written before 1.4.5
    # carry no provenance" names the release that introduced landmark
    # provenance; bumping it would assert something false and strand every
    # session written under 1.4.5. Like the licence history above, this is a
    # permanent fact about the past, and it will still have to say 1.4.5 at
    # 1.4.7 and at 2.0 -- so without this the gate fails at EVERY future
    # release on a line that is correct, and a gate that has to be overridden
    # routinely is one nobody reads.
    #
    # "before" is what makes it safe to strip: a sentence about versions
    # before some number is describing what used to be true, never what this
    # release is. Both the bare and the v-prefixed spellings, since the README
    # uses one and the changelog the other.
    r"|before [Vv]ersions? \d[\d.]*"
    r"|before v\d[\d.]*"
    r"|before \d+\.\d+\.\d+"
)

# Matches 1.2.7 and v1.2.7 alike — \b fails after 'v', since both are word
# characters, which is how MEP_CMAP_Windows.spec was missed on the first pass.
VERSION_RE = re.compile(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])")

# A backslash before a markdown-escapable character is corruption from a
# rich-text round-trip, never intentional in this repo (verified: no fenced
# code block here contains a legitimate backslash). The damage compounds on
# each round-trip, 1 -> 3 -> 7 -> 15 backslashes (n -> 2n+1), silently
# breaking tables, headings, horizontal rules and code blocks.
ESCAPE_RE = re.compile(r"\\+([\]\[()#*_&|<>~`.+\-!{}\"\'])")


def tool_version():
    text = (ROOT / "mep_cmap" / "bids.py").read_text(encoding="utf-8")
    m = re.search(r'^TOOL_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        sys.exit("could not read TOOL_VERSION from mep_cmap/bids.py")
    return m.group(1)


CONFLICT_RE = re.compile(r"^(<{7}|={7}|>{7})(\s|$)", re.M)


def check_conflict_markers():
    """
    Flag unresolved git conflict markers in tracked text files.

    An aborted merge or rebase leaves <<<<<<< / ======= / >>>>>>> in the file,
    which means BOTH sides of the conflict are present: content is silently
    duplicated. Never auto-repaired here, because choosing a side is a
    judgement call.
    """
    bad = 0
    for name in FILES + ["mep_cmap/bids.py"]:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits = [(n, line) for n, line in enumerate(text.splitlines(), 1)
                if CONFLICT_RE.match(line)]
        if not hits:
            continue
        bad += len(hits)
        print(f"  CONFLICT {name}  {len(hits)} unresolved conflict marker(s)")
        for n, line in hits[:3]:
            print(f"            line {n}: {line.strip()[:70]}")
        if len(hits) > 3:
            print(f"            ... and {len(hits) - 3} more")
        print("            resolve the merge/rebase, or restore a clean copy")
    if not bad:
        print("  ok       no unresolved conflict markers")
    return bad


def check_licence_agreement():
    """Every declaration of the licence must say the same thing.

    A licence is stated in five places that nothing keeps in step, and the
    consequences of their disagreeing are worse than for a version string: a
    PyPI page and a repository claiming different terms is a question about
    what a user is actually permitted to do, and it is the metadata rather
    than the LICENSE file that most tooling reads.
    """
    checks = [
        ("LICENSE",       "GNU GENERAL PUBLIC LICENSE"),
        ("LICENSE",       "Version 3, 29 June 2007"),
        ("pyproject.toml", "GNU General Public License v3 or later (GPLv3+)"),
        ("CITATION.cff",  "license: GPL-3.0-or-later"),
        ("zenodo.json",   '"license": "GPL-3.0-or-later"'),
        ("README.md",     "License-GPLv3"),
        ("NOTICE",        "GNU General Public License"),
    ]
    bad = 0
    for name, needle in checks:
        path = ROOT / name
        if not path.exists():
            print(f"  MISSING  {name}")
            bad += 1
            continue
        if needle not in path.read_text(encoding="utf-8", errors="replace"):
            print(f"  MISMATCH {name}  (expected to contain {needle!r})")
            bad += 1
    # A stale MIT claim is the specific failure worth naming, since the project
    # was MIT through 1.3.3 and the strings are easy to leave behind.
    for name in ("pyproject.toml", "CITATION.cff", "zenodo.json"):
        path = ROOT / name
        if path.exists() and '"MIT"' in path.read_text(encoding="utf-8",
                                                       errors="replace"):
            print(f"  MISMATCH {name}  (still declares MIT)")
            bad += 1
    if not bad:
        print("  ok       licence stated consistently (GPL-3.0-or-later)")
    return bad


def check_markdown_escapes(fix=False):
    """Report (and optionally repair) backslash-escaped markdown in README.md."""
    path = ROOT / "README.md"
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8")
    hits = ESCAPE_RE.findall(text)
    if not hits:
        print("  ok       README.md  (markdown not escaped)")
        return 0

    depth = max(len(m.group(0)) - 1 for m in ESCAPE_RE.finditer(text))
    print(f"  MISMATCH README.md  {len(hits)} escaped character(s), "
          f"up to {depth} backslashes deep")
    for n, line in enumerate(text.splitlines(), 1):
        if ESCAPE_RE.search(line):
            print(f"            first at line {n}: {line.strip()[:70]}")
            break

    if fix:
        path.write_text(ESCAPE_RE.sub(r"\1", text), encoding="utf-8")
        print(f"            FIXED: removed {len(hits)} escape sequence(s)")
        return 0

    print("            run  python check_release.py --fix  to repair")
    return len(hits)


def main():
    fix = "--fix" in sys.argv
    want = tool_version()
    print(f"TOOL_VERSION (source of truth): {want}\n")

    problems = []
    for name in FILES:
        path = ROOT / name
        if not path.is_file():
            print(f"  --  {name}  (absent)")
            continue
        hits, bad = [], []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if IGNORE_LINE.search(line):
                continue
            found = VERSION_RE.findall(STRIP_NOISE.sub(" ", line))
            if not found:
                continue
            hits.extend(found)
            # A line is fine if it mentions the current version anywhere: the
            # README's "Point releases (1.2.1-1.2.7)" range is legitimate.
            if want not in found:
                bad.append((n, ", ".join(sorted(set(found))), line.strip()[:88]))
        if not hits:
            print(f"  --  {name}  (no version string)")
            continue
        mark = "ok" if not bad else "MISMATCH"
        print(f"  {mark:8} {name}  ({len(hits)} version string(s))")
        for n, found, line in bad:
            print(f"            line {n}: found {found!r}, expected {want!r}")
            print(f"                    {line}")
            problems.append((name, n, found))

    print()
    conflicts = check_conflict_markers()
    licence = check_licence_agreement()
    escapes = check_markdown_escapes(fix=fix)
    print()
    if problems or escapes or conflicts or licence:
        if problems:
            print(f"{len(problems)} version string(s) out of sync.")
        if conflicts:
            print(f"{conflicts} unresolved git conflict marker(s).")
        if escapes:
            print(f"{escapes} escaped markdown character(s) in README.md.")
        if licence:
            print(f"{licence} licence declaration(s) inconsistent.")
        print("Fix before releasing.")
        return 1
    print("All version strings agree; licence is consistent; no conflict "
          "markers; markdown is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
