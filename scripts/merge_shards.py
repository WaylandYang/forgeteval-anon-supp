"""Merge sharded ForgetEval runs into one system-level result.

Cognee has to be sharded across processes rather than threads (its
dataset scoping does not isolate retrieval, so reset() prunes globally),
which leaves one checkpoint per shard.  This unions them, verifies the
shards partition the suite exactly, and writes the aggregate the paper
cites.

  python scripts/merge_shards.py cognee_chunks --shards 4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="e.g. cognee_chunks")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    verdicts: dict[str, tuple[bool, bool]] = {}
    dupes = []
    for i in range(args.shards):
        p = DATA / f"{args.slug}_s{i}_ckpt.jsonl"
        if not p.exists():
            print(f"  missing shard {i}: {p.name}")
            continue
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in verdicts:
                dupes.append(r["id"])
            verdicts[r["id"]] = (bool(r["ok"]), bool(r.get("na", False)))
            n += 1
        print(f"  shard {i}: {n} verdicts")

    expected = {c.id for c in ADVERSARIAL_TESTS}
    missing = expected - set(verdicts)
    extra = set(verdicts) - expected
    print(f"\ncoverage: {len(verdicts)}/{len(expected)}")
    if dupes:
        print(f"  WARNING {len(dupes)} case(s) appear in more than one shard")
    if missing:
        print(f"  WARNING {len(missing)} case(s) missing, e.g. "
              f"{sorted(missing)[:5]}")
    if extra:
        print(f"  WARNING {len(extra)} unexpected id(s)")
    if missing:
        print("\nrefusing to write an aggregate over an incomplete suite")
        return

    by_cat = defaultdict(lambda: {"pass": 0, "total": 0, "na": 0})
    passed = na = 0
    for cid, (ok, is_na) in verdicts.items():
        cat = case_to_attack_category(cid)
        by_cat[cat]["total"] += 1
        if is_na:
            by_cat[cat]["na"] += 1
            na += 1
        if ok:
            by_cat[cat]["pass"] += 1
            passed += 1

    total = len(verdicts)
    evaluable = total - na
    label = args.label or args.slug
    print(f"\n=== {label} ({total} cases, merged from {args.shards} shards) ===")
    print(f"OVERALL (strict, N/A=fail)  {passed}/{total} = {passed/total:.1%}")
    if evaluable:
        print(f"OVERALL (evaluable only)    {passed}/{evaluable} = "
              f"{passed/evaluable:.1%}   [{na} N/A]")
    print(f"\n{'category':<28}{'pass/tot':>10}  na  rate")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"{cat:<28}{d['pass']:>4}/{d['total']:<4} {d['na']:>4}  "
              f"{d['pass']/max(d['total'],1):.0%}")

    out = {"system": label, "suite": "adversarial-385",
           "shards": args.shards,
           "overall_pass": passed, "overall_total": total,
           "na": na, "evaluable": evaluable,
           "overall_rate_strict": passed / total,
           "overall_rate_evaluable": passed / evaluable if evaluable else None,
           "by_category": dict(by_cat)}
    dest = DATA / f"{args.slug}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote data/{dest.name}")


if __name__ == "__main__":
    main()
