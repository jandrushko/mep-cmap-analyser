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
)

# Matches 1.2.7 and v1.2.7 alike — \b fails after 'v', since both are word
# characters, which is how MEP_CMAP_Windows.spec was missed on the first pass.
VERSION_RE = re.compile(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])")


def tool_version():
    text = (ROOT / "mep_cmap" / "bids.py").read_text(encoding="utf-8")
    m = re.search(r'^TOOL_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        sys.exit("could not read TOOL_VERSION from mep_cmap/bids.py")
    return m.group(1)


def main():
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
    if problems:
        print(f"{len(problems)} version string(s) out of sync — fix before releasing.")
        return 1
    print("All version strings agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
