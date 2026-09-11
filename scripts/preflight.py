"""Pre-submission preflight over the PDF a reviewer downloads and the
tree that ships with it.

Run before freezing. Everything here is a check that has actually caught
something in this paper at least once: a local path in released code, a
page-limit overrun read from the wrong signal, an undefined citation from
a build that silently failed against a locked PDF.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

# the purge_credentials test case inscribes a fake key on purpose
KEY_FIXTURES = {"sk-abc-XYZ-secret-1234567890"}

fail: list[str] = []


def check(name, ok, detail=""):
    print("  %-46s %s%s" % (name, "ok" if ok else "FAIL",
                            ("  " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def pdftotext(*args):
    return subprocess.run(["pdftotext", *args, str(PAPER / "paper.pdf"), "-"],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    txt = pdftotext()

    IDENT = ["35574", "private_projects", "C:/Users", "C:\\Users",
             "WaylandYang", "wayland", "gmail.com"]
    hits = [t for t in IDENT if t.lower() in txt.lower()]
    check("PDF carries no local path or account name", not hits, str(hits))
    check("PDF marked under double-blind review",
          "Under review as a conference paper at ICLR 2027" in txt)
    check("author block is anonymous", "Anonymous authors" in txt)

    # case-sensitive: "Anonymized mirror for review" is prose, ANONYMIZED
    # would be a leftover marker
    ph = [w for w in ("TODO", "FIXME", "XXX", "ANONYMIZED", "PLACEHOLDER")
          if re.search(r"\b" + w + r"\b", txt)]
    check("no placeholder markers in the PDF", not ph, str(ph))

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()

    keypat = re.compile(r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
                        r"ghp_[A-Za-z0-9]{30,}")
    # Substrings, matched case-insensitively. Bare "reviewer" is not
    # here: the adversarial fixtures contain a memory record about one.
    VENUE = ("emnlp", "acl rolling", "arr ", "addresses reviewer",
             "reviewer 1", "reviewer 2", "reviewer 3",
             "reviewer r1", "reviewer r2", "reviewer r3",
             "reviewer r4", "reviewer asked", "reviewer ask:",
             "reviewer q", "rebuttal", "camera-ready",
             "resubmission")
    leaks, paths, venue = [], [], []
    for rel in tracked:
        p = ROOT / rel
        if not p.is_file() or p.suffix.lower() in (".pdf", ".png"):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in keypat.findall(t):
            if m not in KEY_FIXTURES:
                leaks.append("%s: %s" % (rel, m[:12]))
        # this file names the patterns it looks for, so it matches itself
        if rel == "scripts/preflight.py":
            continue
        if rel.endswith(".py") and ("C:/Users" in t or "C:\\Users" in t):
            paths.append(rel)
        # A supplement that says "Addresses reviewer 2 2.4", or names the
        # venue the paper was previously prepared for, tells a
        # double-blind reader this is a resubmission and where it came
        # from. The body was cleared of these; the released scripts
        # carried eight anyway, because nothing looked outside the PDF.
        if rel.endswith((".py", ".md", ".toml", ".cff")):
            low = t.lower()
            venue += ["%s: %s" % (rel, m) for m in VENUE if m in low]
    check("no live API keys in tracked files", not leaks, str(leaks[:3]))
    check("no hardcoded local paths in tracked code", not paths,
          str(paths[:3]))
    check("no venue or review-round markers in released code",
          not venue, str(venue[:4]))

    # A PDF viewer holding paper.pdf open makes pdflatex emergency-stop,
    # which truncates paper.aux to a stub and leaves paper.bbl empty. The
    # next build then reports every citation undefined, and the log that
    # says otherwise is the stale one from before the lock.
    aux = (PAPER / "paper.aux").stat().st_size if (PAPER / "paper.aux").exists() else 0
    bbl = (PAPER / "paper.bbl").stat().st_size if (PAPER / "paper.bbl").exists() else 0
    check("build state intact (aux and bbl non-stub)",
          aux > 10_000 and bbl > 5_000,
          "aux %d B, bbl %d B" % (aux, bbl))

    # paper.tex resolves graphicspath to paper/figures/. The generators
    # used to write to a figures/ at the repository root, so a
    # regenerated figure never reached the paper and the
    # regenerate-and-diff check could not see it: it compared the file
    # the generator wrote, not the one LaTeX read.
    import re as _re
    tex = (PAPER / "paper.tex").read_text(encoding="utf-8")
    want = _re.findall(r"includegraphics\[[^]]*\]\{([^}]+)\}", tex)
    absent = [g for g in want if not (PAPER / "figures" / g).exists()]
    check("every figure the paper includes is present", not absent,
          str(absent))

    log = (PAPER / "paper.log").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Output written.*?\((\d+) pages", log)
    check("PDF built", bool(m), (m.group(1) + " pages") if m else "")
    check("no undefined citations",
          not re.findall(r"Citation .*? undefined", log))
    check("no undefined references",
          not re.findall(r"Reference .*? undefined", log))
    check("no overfull boxes", log.count("Overfull") == 0)
    check("no LaTeX errors",
          not [l for l in log.split("\n") if l.startswith("!")])

    # the page limit is on the main text; the statements and references
    # do not count, so the question is whether section 5 spills past 9
    p10 = pdftotext("-f", "10", "-l", "10")
    body = 0
    for line in p10.split("\n"):
        if "REPRODUCIBILITY STATEMENT" in line:
            break
        # the running header is on every page and is not body text;
        # counting it reported a one-line overflow that was not there
        if "Under review as a conference paper" in line:
            continue
        if re.search(r"[a-z]{3}", line):
            body += 1
    p9 = pdftotext("-f", "9", "-l", "9")
    ok = "REPRODUCIBILITY STATEMENT" in p9 or body == 0
    check("main text ends by page 9", ok,
          "" if ok else "%d body lines on page 10" % body)

    print()
    print("preflight: %s"
          % ("ALL CLEAR" if not fail else "%d FAILED: %s" % (len(fail), fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
