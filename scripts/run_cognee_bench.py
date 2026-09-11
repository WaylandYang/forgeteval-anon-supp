"""Run ForgetEval-Adv against Cognee v1.5.

Cognee is the only system in the study with a native soft delete, a
native partial edit, and a native hard purge, so it tests whether having
the primitives is sufficient.  Two retrieval configurations are scored
separately:

    --retrieval chunks   record-level recall of stored surface forms;
                         the apples-to-apples comparison with every other
                         store in the study
    --retrieval graph    GRAPH_COMPLETION, what a real Cognee deployment
                         hands an agent -- a synthesised answer over the
                         derived graph

The gap between the two is the quantity of interest: a derived layer can
answer from artifacts that outlive the record they were derived from.

  LLM_API_KEY=... python scripts/run_cognee_bench.py --retrieval chunks

Requires an OpenAI-compatible endpoint (cognify calls the LLM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("CACHING", "false")

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.adapter import CogneeAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", choices=["chunks", "graph"],
                    default="chunks")
    ap.add_argument("--limit", type=int, default=0, help="0 = full 385")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1,
                    help="Cognee cannot be thread-parallel: reset() has to "
                         "prune globally because dataset scoping does not "
                         "isolate retrieval. Shard across PROCESSES instead, "
                         "each with its own cognee storage root.")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.shards > 1:
        import cognee
        root = ROOT / ".cognee_shards" / f"s{args.shard}"
        cognee.config.system_root_directory(str(root / "system"))
        cognee.config.data_root_directory(str(root / "data"))
        print(f"shard {args.shard}/{args.shards}  storage={root}")

    cases = ADVERSARIAL_TESTS[: args.limit] if args.limit else ADVERSARIAL_TESTS
    if args.shards > 1:
        cases = cases[args.shard::args.shards]
    shard_sfx = f"_s{args.shard}" if args.shards > 1 else ""
    slug = f"cognee_{args.retrieval}{args.tag}{shard_sfx}"
    ckpt = OUT / f"{slug}_ckpt.jsonl"

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r["ok"]
        print(f"resume: {len(done)} cases already done")

    # one adapter (and one private cognee dataset) per worker thread
    _local = threading.local()

    def get_adapter():
        if not hasattr(_local, "a"):
            _local.a = CogneeAdapter(retrieval=args.retrieval)
        return _local.a

    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    counters = {"passed": 0, "errors": 0, "finished": 0}
    lock = threading.Lock()
    fout = ckpt.open("a", encoding="utf-8")
    t0 = time.perf_counter()

    def evaluate(c):
        if c.id in done:
            return c.id, done[c.id], False
        try:
            ok = c.run(get_adapter())
        except Exception as e:
            ok = False
            with lock:
                counters["errors"] += 1
                print(f"  [case error] {c.id}: {type(e).__name__}: {str(e)[:90]}")
        return c.id, ok, True

    def record(case_id, ok, fresh):
        with lock:
            if fresh:
                fout.write(json.dumps({"id": case_id, "ok": ok}) + "\n")
                fout.flush()
            cat = case_to_attack_category(case_id)
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["pass"] += 1
                counters["passed"] += 1
            counters["finished"] += 1
            if counters["finished"] % 10 == 0:
                print(f"  {counters['finished']}/{len(cases)} "
                      f"pass={counters['passed']} err={counters['errors']}",
                      flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(evaluate, cases):
                record(*r)
    else:
        for c in cases:
            record(*evaluate(c))
    fout.close()

    wall = time.perf_counter() - t0
    total = len(cases)
    passed = counters["passed"]
    print(f"\n=== Cognee v1.5 ({args.retrieval}) on {total} cases ===")
    print(f"OVERALL {passed}/{total} = {passed/total:.1%}  "
          f"case_errors={counters['errors']}  wall={wall:.0f}s")
    print(f"\n{'category':<28}{'pass/tot':>10} rate")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"{cat:<28}{d['pass']:>4}/{d['total']:<4} "
              f"{d['pass']/max(d['total'],1):.0%}")

    out = {"system": "cognee-1.5", "retrieval": args.retrieval,
           "suite": "adversarial-385", "overall_pass": passed,
           "overall_total": total, "overall_rate": passed / total,
           "by_category": dict(by_cat), "case_errors": counters["errors"],
           "wall_seconds": wall}
    (OUT / f"{slug}.json").write_text(json.dumps(out, indent=2),
                                      encoding="utf-8")
    print(f"\nwrote data/{slug}.json")


if __name__ == "__main__":
    main()
