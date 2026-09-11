"""Build paper.pdf through the one sequence that is known to converge.

Hand-rolled loops have twice left the build in a state where the log said
one thing and the PDF another: once because a PDF viewer held paper.pdf
open, which makes pdflatex emergency-stop and truncate paper.aux, and once
because an interleaved run left the aux out of step with the bbl, so the
final pass reported every reference undefined against a 47 kB aux that
looked healthy to a size check.

The sequence is: drop the derived files, one pass to write the aux, bibtex,
then two passes to settle the references. Refuses to start if anything holds
the PDF open, because that failure is silent and its symptom is a stale log.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
DERIVED = ("paper.aux", "paper.bbl", "paper.blg", "paper.out", "paper.log")


def locked(p: pathlib.Path) -> bool:
    if not p.exists():
        return False
    try:
        with p.open("ab"):
            return False
    except OSError:
        return True


def run(*cmd):
    return subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name in ("paper.pdf",):
        for d in (PAPER, ROOT / "submission"):
            if locked(d / name):
                raise SystemExit("%s is open in another program; close it "
                                 "and re-run" % (d / name))

    for f in DERIVED:
        (PAPER / f).unlink(missing_ok=True)

    run("pdflatex", "-interaction=nonstopmode", "paper.tex")
    aux = (PAPER / "paper.aux").stat().st_size
    if aux < 10_000:
        raise SystemExit("first pass wrote a %d B aux; the build is broken" % aux)
    run("bibtex", "paper")
    bbl = (PAPER / "paper.bbl").stat().st_size
    if bbl < 5_000:
        raise SystemExit("bibtex wrote a %d B bbl; check refs.bib" % bbl)
    for _ in range(2):
        run("pdflatex", "-interaction=nonstopmode", "paper.tex")

    log = (PAPER / "paper.log").read_text(encoding="utf-8", errors="ignore")
    pages = re.search(r"Output written.*?\((\d+) pages", log)
    undef_r = len(re.findall(r"Reference .*? undefined", log))
    undef_c = len(re.findall(r"Citation .*? undefined", log))
    print("  %s pages, aux %d B, bbl %d B" % (pages.group(1) if pages else "?",
                                              aux, bbl))
    print("  undefined: %d references, %d citations" % (undef_r, undef_c))
    if undef_r or undef_c:
        raise SystemExit("references did not settle")


if __name__ == "__main__":
    main()
