"""Run ForgetEval-Adv against an extended-ecosystem store.

One runner for the systems that need no per-system harness: the adapter
is constructed by name and everything else -- N/A accounting, resumable
checkpoints, per-category reporting -- is shared, so a new system joins
the comparison by writing an adapter, not a script.

  LLM_API_KEY=... python scripts/run_ecosystem_bench.py --system memos
  python scripts/run_ecosystem_bench.py --system tencentdb --workers 4

N/A is recorded separately from failure: a store without a soft-delete
primitive has not *failed* the release cases, and collapsing the two
would flatter systems that implement a primitive badly at the expense of
systems that honestly do not have it.
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
from bench.forgeteval import adapter as A  # noqa: E402

SYSTEMS = {
    "memos": ("memos-2.0.30", lambda: A.MemOSAdapter()),
    "tencentdb": ("tencentdb-agent-memory-2.0.0", lambda: A.TencentDBAdapter()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    label, factory = SYSTEMS[args.system]
    cases = ADVERSARIAL_TESTS[: args.limit] if args.limit else ADVERSARIAL_TESTS
    slug = f"{args.system}{args.tag}"
    ckpt = OUT / f"{slug}_ckpt.jsonl"

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = (r["ok"], r.get("na", False))
        print(f"resume: {len(done)} done")

    _local = threading.local()

    def get_adapter():
        if not hasattr(_local, "a"):
            _local.a = factory()
        return _local.a

    by_cat = defaultdict(lambda: {"pass": 0, "total": 0, "na": 0})
    counters = {"passed": 0, "errors": 0, "na": 0, "finished": 0}
    lock = threading.Lock()
    fout = ckpt.open("a", encoding="utf-8")
    t0 = time.perf_counter()

    def evaluate(c):
        if c.id in done:
            ok, na = done[c.id]
            return c.id, ok, False, na
        na = False
        try:
            ok = c.run(get_adapter())
        except NotImplementedError:
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
    print(f"\n=== {label} on {total} cases ===")
    print(f"OVERALL (strict, N/A=fail)  {passed}/{total} = {passed/total:.1%}")
    if evaluable:
        print(f"OVERALL (evaluable only)    {passed}/{evaluable} = "
              f"{passed/evaluable:.1%}   [{na} N/A]")
    print(f"case_errors={counters['errors']}  wall={wall:.0f}s")
    print(f"\n{'category':<28}{'pass/tot':>10}  na  rate")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"{cat:<28}{d['pass']:>4}/{d['total']:<4} {d['na']:>4}  "
              f"{d['pass']/max(d['total'],1):.0%}")

    out = {"system": label, "suite": "adversarial-385",
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
