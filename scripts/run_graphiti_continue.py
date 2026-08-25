"""Continue Graphiti bench from existing results — runs all
ADVERSARIAL_TESTS cases NOT already in adversarial_results_graphiti.json.

After current 100-case stratified run finishes, this picks up the
remaining 285 cases so we end with full 385-case coverage matching
the primary systems.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault(
    "OPENAI_API_KEY",
    "")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)

from run_graphiti_bench import GraphitiAdapter  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

# Load existing results (whichever files exist).
done_ids: set[str] = set()
existing_results: list = []
existing_by_cat: dict[str, dict[str, int]] = {}

for path in [DATA / "adversarial_results_graphiti.json",
             DATA / "adversarial_results_graphiti_partial.json"]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for r in d.get("results", []):
            cid = r["case_id"]
            if cid not in done_ids:
                done_ids.add(cid)
                existing_results.append(r)
                cat = r.get("category", case_to_attack_category(cid))
                existing_by_cat.setdefault(cat,
                    {"pass": 0, "fail": 0, "na": 0})
                if r.get("passed") is True:
                    existing_by_cat[cat]["pass"] += 1
                elif r.get("passed") is False:
                    existing_by_cat[cat]["fail"] += 1
                else:
                    existing_by_cat[cat]["na"] += 1

print(f"Loaded {len(done_ids)} previously-done cases", flush=True)

remaining = [c for c in ADVERSARIAL_TESTS if c.id not in done_ids]
print(f"{len(remaining)} cases remaining to reach full {len(ADVERSARIAL_TESTS)}",
      flush=True)


def main():
    adapter = GraphitiAdapter()
    results = list(existing_results)
    by_cat = {k: dict(v) for k, v in existing_by_cat.items()}
    t_start = time.time()

    for i, case in enumerate(remaining, 1):
        cat = case_to_attack_category(case.id)
        try:
            try:
                passed = case.run(adapter)
                applied = True
            except NotImplementedError:
                passed = None
                applied = False

            by_cat.setdefault(cat, {"pass": 0, "fail": 0, "na": 0})
            if passed is True:
                by_cat[cat]["pass"] += 1
            elif passed is False:
                by_cat[cat]["fail"] += 1
            else:
                by_cat[cat]["na"] += 1

            results.append({
                "case_id": case.id, "category": cat,
                "passed": passed, "applied": applied,
            })

            elapsed = time.time() - t_start
            tot_p = sum(c["pass"] for c in by_cat.values())
            tot_e = sum(c["pass"] + c["fail"] for c in by_cat.values())
            rate = tot_p / tot_e * 100 if tot_e else 0
            verdict = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
            total_n = len(remaining)
            print(f"  {i:3d}/{total_n:3d} {case.id[:40]:40s} {cat:25s} {verdict}  "
                  f"agg={tot_p}/{tot_e} ({rate:.1f}%)  t={elapsed:.0f}s",
                  flush=True)

            if i % 25 == 0:
                with open(DATA / "adversarial_results_graphiti_partial.json",
                          "w", encoding="utf-8") as f:
                    json.dump({"n": len(results), "results": results,
                               "by_cat": by_cat}, f, indent=2)
        except KeyboardInterrupt:
            print("Interrupted")
            break
        except Exception as e:
            print(f"  [case {case.id}] error: {type(e).__name__}: {e}",
                  flush=True)
            by_cat.setdefault(cat, {"pass": 0, "fail": 0, "na": 0})
            by_cat[cat]["fail"] += 1
            results.append({
                "case_id": case.id, "category": cat,
                "passed": False, "error": f"{type(e).__name__}: {e}",
            })

    # Final save (overwrites the 100-case file)
    with open(DATA / "adversarial_summary_graphiti.json", "w",
              encoding="utf-8") as f:
        json.dump({"system": "graphiti", "by_category": by_cat,
                   "model": "deepseek-ai/DeepSeek-V3.1-Terminus",
                   "n_total": len(results)}, f, indent=2)
    with open(DATA / "adversarial_results_graphiti.json", "w",
              encoding="utf-8") as f:
        json.dump({"system": "graphiti",
                   "model": "deepseek-ai/DeepSeek-V3.1-Terminus",
                   "results": results}, f, indent=2)

    print("\n=== Graphiti FULL aggregate ===", flush=True)
    tot_p = tot_f = tot_na = 0
    for cat, d in sorted(by_cat.items()):
        n_eval = d["pass"] + d["fail"]
        rate = d["pass"] / n_eval * 100 if n_eval else float("nan")
        na_note = f" (N/A {d['na']})" if d["na"] else ""
        print(f"  {cat:30s}  {d['pass']:3d}/{n_eval:<3d} ({rate:5.1f}%){na_note}",
              flush=True)
        tot_p += d["pass"]
        tot_f += d["fail"]
        tot_na += d["na"]
    print(f"  OVERALL  {tot_p}/{tot_p + tot_f}  N/A {tot_na}", flush=True)


if __name__ == "__main__":
    main()

