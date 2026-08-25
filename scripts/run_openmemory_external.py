"""Run OpenMemory on the 77-case external-authored subset.

Reuses OpenMemoryAdapter from run_openmemory_bench.py and the
ExtCase wrapper from run_external_subset.py.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))
sys.path.insert(0, str(PAPER_ROOT / "scripts"))

DATA = PAPER_ROOT / "data"

from run_openmemory_bench import OpenMemoryAdapter  # noqa: E402
from run_external_subset import ExtCase  # noqa: E402


def main():
    cases_path = DATA / "external_subset_cases.json"
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    admitted = data["admitted_cases"]
    cases = [ExtCase(c) for c in admitted]
    print(f"loaded {len(cases)} admitted external cases", flush=True)

    adapter = OpenMemoryAdapter()
    n_pass = n_na = 0
    per_case = []
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"pass": 0, "total": 0, "na": 0})
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            passed = c.run(adapter)
            err = None
        except NotImplementedError:
            passed = None
            err = "N/A"
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        per_case.append({"id": c.id, "category": c.category,
                         "passed": passed, "error": err})
        d = by_cat[c.category]
        if passed is True:
            d["pass"] += 1; d["total"] += 1; n_pass += 1
        elif passed is False:
            d["total"] += 1
        else:
            d["na"] += 1; n_na += 1
        if i % 10 == 0 or i == len(cases):
            tot_e = sum(b["total"] for b in by_cat.values())
            print(f"  OpenMem [{i:3}/{len(cases)}] "
                  f"so-far {n_pass}/{tot_e} (N/A {n_na})",
                  flush=True)
    wall = time.time() - t0
    n_eval = sum(d["total"] for d in by_cat.values())
    rate = n_pass / n_eval * 100 if n_eval else 0

    print(f"\n=== OpenMemory (external) summary ===")
    print(f"  pass: {n_pass}/{n_eval} ({rate:.1f}%)  N/A: {n_na}  wall: {wall:.1f}s")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        r = d["pass"] / d["total"] * 100 if d["total"] else 0
        print(f"    {cat:30s} {d['pass']:2}/{d['total']:<2} ({r:5.1f}%)  N/A: {d['na']}")

    # Append to external_subset_results.json
    path = DATA / "external_subset_results.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"systems": {}}
    out.setdefault("systems", {})["OpenMemory"] = {
        "n_pass": n_pass, "n_eval": n_eval, "n_na": n_na, "rate": rate,
        "by_category": dict(by_cat), "per_case": per_case, "wall_s": wall,
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
