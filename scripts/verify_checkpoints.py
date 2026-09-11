"""Check that every emitted result matches its checkpoint, and that no
checkpoint was written by two processes at once.

Runs are checkpointed per case and resumable, which makes them cheap to
restart -- and makes it possible for a failed launch and its replacement
to append to the same file. When that happens the checkpoint holds two
verdicts for some cases and each process counts only what it ran, so the
emitted JSON can disagree with the file it was supposedly derived from.
Nothing errors; the number is just quietly wrong.

This recomputes each score from the deduplicated checkpoint (last write
wins, matching the resume logic) and compares. A conflict count above
zero means two runs disagreed on a case, which is expected at the ~2%
level from model nondeterminism (App. M) and suspicious well above it.

  python scripts/verify_checkpoints.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(path):
    rows, conflicts = {}, 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "id" not in r or not ({"ok", "passed", "nli_pass"} & set(r)):
            continue  # blob / per-case files use a different shape
        cid = r["id"]
        ok = bool(r.get("ok", r.get("passed", r.get("nli_pass"))))
        if cid in rows and rows[cid] != ok:
            conflicts += 1
        rows[cid] = ok
    return rows, conflicts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    print(f"{'run':<58}{'json':>7}{'ckpt':>7}{'dup':>6}{'conf':>6}")
    for ck in sorted(DATA.glob("*_ckpt.jsonl")):
        result = DATA / (ck.name[:-len("_ckpt.jsonl")] + ".json")
        if not result.exists():
            continue
        rows, conflicts = load(ck)
        n_lines = sum(1 for l in ck.read_text(encoding="utf-8-sig").splitlines()
                      if l.strip())
        try:
            claimed = json.loads(result.read_text(encoding="utf-8-sig"))
            claimed = claimed.get("overall_pass")
        except Exception:
            continue
        recomputed = sum(rows.values())
        dup = n_lines - len(rows)
        flag = "" if claimed == recomputed else "   <-- MISMATCH"
        if flag or conflicts:
            bad += 1
        print(f"{ck.name[:-len('_ckpt.jsonl')][:56]:<58}"
              f"{claimed:>7}{recomputed:>7}{dup:>6}{conflicts:>6}{flag}")
    print(f"\nruns needing attention: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
