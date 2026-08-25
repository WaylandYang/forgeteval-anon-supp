"""Build the frozen submission package.

Two artifacts:

  submission/paper.pdf              what goes in the OpenReview PDF slot
  submission/supplementary.zip      the anonymized code and data

The archive is built from git-tracked files only. That is deliberate:
it excludes the local virtualenvs, the scratch files at the top level,
and -- most importantly -- .git itself, whose commit metadata carries
the author's name and email. Copying the working directory would ship
all three.

Everything in it has been through scripts/preflight.py, which fails on a
live API key or a hardcoded local path in any tracked file.
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"

# tracked but not part of what a reviewer needs
EXCLUDE_PREFIX = ("logs/", "iaa_results/")
EXCLUDE_EXACT = {"paper_for_review.txt", "canon_sample.txt",
                 "prefix_sample.txt", "idobf.txt", "canon_cases.json"}
# the built PDF ships on its own, not inside the archive
EXCLUDE_SUFFIX = (".aux", ".log", ".out", ".blg", ".synctex.gz")


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    r = subprocess.run([sys.executable, "scripts/preflight.py"], cwd=ROOT)
    if r.returncode:
        print("\npreflight failed -- not freezing")
        return 1

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("\nworking tree is dirty; commit before freezing:")
        print(dirty[:400])
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    pdf = ROOT / "paper" / "paper.pdf"
    shutil.copy2(pdf, OUT / "paper.pdf")

    members = []
    for rel in tracked:
        if rel.startswith(EXCLUDE_PREFIX) or rel in EXCLUDE_EXACT:
            continue
        if rel.endswith(EXCLUDE_SUFFIX):
            continue
        p = ROOT / rel
        if p.is_file():
            members.append(rel)

    zpath = OUT / "supplementary.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for rel in sorted(members):
            z.write(ROOT / rel, arcname=rel)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    pages = subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    npages = next((l.split()[-1] for l in pages.split("\n")
                   if l.startswith("Pages")), "?")

    manifest = OUT / "MANIFEST.txt"
    manifest.write_text(
        "ForgetEval submission package\n"
        "commit        %s\n"
        "paper.pdf     %d bytes, %s pages, sha256 %s\n"
        "supplementary %d bytes, %d files, sha256 %s\n"
        % (commit, (OUT / "paper.pdf").stat().st_size, npages,
           sha256(OUT / "paper.pdf"), zpath.stat().st_size, len(members),
           sha256(zpath)),
        encoding="utf-8")

    print()
    print("froze at %s" % commit[:12])
    print("  submission/paper.pdf          %8.1f KB  %s pages"
          % ((OUT / "paper.pdf").stat().st_size / 1024, npages))
    print("  submission/supplementary.zip  %8.1f KB  %d files"
          % (zpath.stat().st_size / 1024, len(members)))
    print("  submission/MANIFEST.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
