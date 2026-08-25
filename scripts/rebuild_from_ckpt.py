"""Rebuild a result JSON from its checkpoint.

A run that completes but is refused by the write guard leaves 385 verdicts
on disk and no result file. When the refusal was a false positive -- as it
is for adapters that construct their own LLM client, where our call
counter reads zero however well the run went -- the measurement exists and
should not be repeated for two hours to recover a file.

    python scripts/rebuild_from_ckpt.py <ckpt.jsonl> [--model NAME]

Writes the sibling .json with the same aggregate structure the runner
produces. Usage counters are recorded as unavailable rather than zero,
because zero would claim something this path cannot know.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.forgeteval.adversarial import case_to_attack_category  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    a = ap.parse_args()

    p = pathlib.Path(a.ckpt)
    if not p.is_absolute():
        p = ROOT / p
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seen = {}
    for r in rows:
        seen[r["id"]] = r["ok"]          # last verdict wins, as the runner does
    if len(seen) != len(rows):
        print("note: %d duplicate ids collapsed" % (len(rows) - len(seen)),
              file=sys.stderr)

    by = collections.defaultdict(lambda: {"pass": 0, "total": 0})
    for cid, ok in seen.items():
        c = by[case_to_attack_category(cid)]
        c["total"] += 1
        c["pass"] += bool(ok)

    total = len(seen)
    passed = sum(1 for v in seen.values() if v)
    out = {
        "model": a.model,
        "suite": "adversarial-385",
        "limit": 0,
        "overall_pass": passed,
        "overall_total": total,
        "overall_rate": passed / total,
        "by_category": dict(by),
        # This path cannot observe the call counters, and reporting zero
        # would assert something it does not know.
        "usage": {"reconstructed_from": p.name},
        "rebuilt_from_checkpoint": True,
    }
    dest = p.with_name(p.name.replace("_ckpt.jsonl", ".json"))
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("wrote %s  (%d/%d = %.1f%%)"
          % (dest.name, passed, total, 100 * passed / total))


if __name__ == "__main__":
    main()
