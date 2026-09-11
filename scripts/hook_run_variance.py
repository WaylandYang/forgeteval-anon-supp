"""Run-to-run variance of the mutation-time hook.

Every cross-LLM number in App. K is a single run at temperature 0, which
does not make an API-served model deterministic: retries, batching, and
kernel nondeterminism all leak in. This script aggregates N independent
full-suite repeats of the same model and reports (a) the spread of the
aggregate pass rate and (b) the case-level flip rate --- how often a single
case changes verdict between two runs --- which is the quantity that
actually bounds how much of a reported difference could be noise.

  python scripts/hook_run_variance.py deepseek/deepseek-v4-pro

Reads data/openrouter_hook_<slug>{,_r2,_r3,...}_ckpt.jsonl.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_ckpt(path):
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = bool(r["ok"])
    return out


def category_of(case_id):
    # adv_<category>_<nn> / ext2_<initials>_<category>_<nn>
    parts = case_id.split("_")
    return "_".join(parts[1:-1]) if len(parts) > 2 else "unknown"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-v4-pro"
    slug = model.replace("/", "_").replace(".", "")

    runs = []
    for path in sorted(DATA.glob(f"openrouter_hook_{slug}*_ckpt.jsonl")):
        if "smoke" in path.name:
            continue
        v = load_ckpt(path)
        if len(v) < 385:
            print(f"  (skipping incomplete {path.name}: {len(v)}/385)")
            continue
        runs.append((path.name, v))

    if len(runs) < 2:
        sys.exit(f"need >=2 complete runs for {model}; found {len(runs)}")

    common = sorted(set.intersection(*(set(v) for _, v in runs)))
    n = len(common)
    rates = [sum(v[c] for c in common) / n for _, v in runs]

    print(f"=== {model}: {len(runs)} independent full-suite runs, n={n} ===")
    for (name, v), r in zip(runs, rates):
        print(f"  {name:<62} {sum(v[c] for c in common):>3}/{n} = {r:.1%}")
    print(f"\n  mean   {statistics.mean(rates):.2%}")
    if len(rates) > 1:
        print(f"  stdev  {statistics.stdev(rates):.2%}  "
              f"(range {max(rates)-min(rates):.2%})")

    # case-level instability: cases that are not unanimous across runs
    unstable, flips_by_cat = [], defaultdict(int)
    for c in common:
        vals = [v[c] for _, v in runs]
        if len(set(vals)) > 1:
            unstable.append((c, sum(vals), len(vals)))
            flips_by_cat[category_of(c)] += 1
    print(f"\n  cases with non-unanimous verdict: {len(unstable)}/{n} "
          f"({len(unstable)/n:.1%})")
    for cat in sorted(flips_by_cat, key=lambda k: -flips_by_cat[k]):
        print(f"    {cat:<28} {flips_by_cat[cat]}")

    out = {
        "model": model, "n_runs": len(runs), "n_cases": n,
        "run_files": [name for name, _ in runs],
        "rates": rates,
        "mean": statistics.mean(rates),
        "stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "unstable_cases": [{"id": c, "passes": p, "runs": t}
                           for c, p, t in unstable],
        "unstable_by_category": dict(flips_by_cat),
    }
    dest = DATA / f"hook_run_variance_{slug}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
