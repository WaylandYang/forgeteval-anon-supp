"""Run ForgetEval-Adv against MemOS / MemoryOS v2.

MemOS is included because it has a native partial-edit primitive
(``update``), which no comparator except the reference implementation
has.  That makes it the control for the objection that
``compound_fact`` was written around one system's feature set.

Retrieval is embedding-only, so MemOS sits in the deterministic /
vec-only regime; the configured LLM is MemOS's extractor and never runs
on the recall path.

  LLM_API_KEY=... python scripts/run_memos_bench.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.adapter import MemOSAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cases = ADVERSARIAL_TESTS[: args.limit] if args.limit else ADVERSARIAL_TESTS
    slug = f"memos{args.tag}"
    ckpt = OUT / f"{slug}_ckpt.jsonl"

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r["ok"]
        print(f"resume: {len(done)} done")

    _local = threading.local()

    def get_adapter():
        if not hasattr(_local, "a"):
            _local.a = MemOSAdapter()
        return _local.a

    by_cat = defaultdict(lambda: {"pass": 0, "total": 0, "na": 0})
    counters = {"passed": 0, "errors": 0, "na": 0, "finished": 0}
    lock = threading.Lock()
    fout = ckpt.open("a", encoding="utf-8")
    t0 = time.perf_counter()

    def evaluate(c):
        if c.id in done:
            return c.id, done[c.id], False, False
        na = False
        try:
            ok = c.run(get_adapter())
        except NotImplementedError:
            # honest N/A: the store has no such primitive, which is a
            # different fact about it than "implemented and failed"
            ok, na = False, True
        except Exception as e:
            ok = False
            with lock:
                counters["errors"] += 1
                print(f"  [case error] {c.id}: {type(e).__name__}: {str(e)[:90]}")
        return c.id, ok, True, na

    def record(case_id, ok, fresh, na):
        with lock:
            if fresh:
                fout.write(json.dumps({"id": case_id, "ok": ok, "na": na}) + "\n")
                fout.flush()
            cat = case_to_attack_category(case_id)
            by_cat[cat]["total"] += 1
            if na:
                by_cat[cat]["na"] += 1
                counters["na"] += 1
            if ok:
                by_cat[cat]["pass"] += 1
                counters["passed"] += 1
            counters["finished"] += 1
            if counters["finished"] % 25 == 0:
                print(f"  {counters['finished']}/{len(cases)} "
                      f"pass={counters['passed']} na={counters['na']} "
                      f"err={counters['errors']}", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(evaluate, cases):
                record(*r)
    else:
        for c in cases:
            record(*evaluate(c))
    fout.close()

    wall = time.perf_counter() - t0
    total, passed, na = len(cases), counters["passed"], counters["na"]
    evaluable = total - na
    print(f"\n=== MemOS v2 on {total} cases ===")
    print(f"OVERALL (strict, N/A=fail)  {passed}/{total} = {passed/total:.1%}")
    if evaluable:
        print(f"OVERALL (evaluable only)    {passed}/{evaluable} = "
              f"{passed/evaluable:.1%}   [{na} N/A: no release primitive]")
    print(f"case_errors={counters['errors']}  wall={wall:.0f}s")
    print(f"\n{'category':<28}{'pass/tot':>10}  na  rate")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"{cat:<28}{d['pass']:>4}/{d['total']:<4} {d['na']:>4}  "
              f"{d['pass']/max(d['total'],1):.0%}")

    out = {"system": "memos-2.0.30", "suite": "adversarial-385",
           "overall_pass": passed, "overall_total": total,
           "na": na, "evaluable": evaluable,
           "overall_rate_strict": passed / total,
           "overall_rate_evaluable": passed / evaluable if evaluable else None,
           "by_category": dict(by_cat), "case_errors": counters["errors"],
           "wall_seconds": wall}
    (OUT / f"{slug}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote data/{slug}.json")


if __name__ == "__main__":
    main()
