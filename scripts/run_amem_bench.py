"""Run ForgetEval-Adv 365 cases on A-MEM with DeepSeek-V3 via SiliconFlow.

A-MEM (Xu et al., NeurIPS 2025) is a Zettelkasten-style memory with explicit
delete/update primitives at the memory_id level.  This run is the
"5th system" entry to address reviewer concerns about empirical breadth.

Output:
  data/adversarial_results_amem.json   (per-case verdicts)
  data/adversarial_summary_amem.json   (aggregate by category)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Configure OpenAI SDK to use SiliconFlow's DeepSeek-V3 endpoint
# (must be set BEFORE importing agentic_memory)
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"

# Add lethe repo to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adapter import AMemAdapter  # noqa: E402
from bench.forgeteval.adversarial import ADVERSARIAL_TESTS, case_to_attack_category  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)


class AMemSF(AMemAdapter):
    """A-MEM wired to SiliconFlow's DeepSeek-V3."""

    def __init__(self):
        super().__init__(llm_backend="openai",
                         embedder_model="all-MiniLM-L6-v2")

    def reset(self) -> None:
        # llm_model = DeepSeek-V3 via SiliconFlow (env var OPENAI_BASE_URL).
        # evo_threshold high so evolution never fires; evolution would call the
        # LLM again per note and DeepSeek returns non-JSON for those prompts.
        self.ms = self._System(
            model_name=self.embedder_model,
            llm_backend="openai",
            llm_model="deepseek-ai/DeepSeek-V3",
            evo_threshold=100000,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        try:
            for mid in list(self.ms.memories.keys()):
                self.ms.delete(mid)
        except Exception:
            pass


def stratified_sample(cases, n_per_cat: int = 10):
    """Take up to n_per_cat cases per category, preserving order."""
    by_cat: dict[str, list] = {}
    for c in cases:
        cat = case_to_attack_category(c.id)
        by_cat.setdefault(cat, []).append(c)
    out = []
    for cat, lst in by_cat.items():
        out.extend(lst[:n_per_cat])
    return out


def main():
    cases = ADVERSARIAL_TESTS
    sample = int(os.environ.get("SAMPLE_PER_CAT", "0"))
    if sample > 0:
        cases = stratified_sample(cases, n_per_cat=sample)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        cases = cases[:limit]
    print(f"Running {len(cases)} adversarial cases", flush=True)

    adapter = AMemSF()
    results = []
    by_cat = {}
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        try:
            cat = case_to_attack_category(case.id)
            # Use the case's built-in run() helper but catch N/A from primitive misses.
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
                "case_id": case.id,
                "category": cat,
                "passed": passed,
                "applied": applied,
            })

            elapsed = time.time() - t_start
            tot_pass = sum(c["pass"] for c in by_cat.values())
            tot_evaluable = sum(c["pass"] + c["fail"] for c in by_cat.values())
            rate = tot_pass / tot_evaluable if tot_evaluable else 0
            verdict = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
            print(f"  {i:3d}/{len(cases):3d} {case.id[:40]:40s} {cat:25s} {verdict}  "
                  f"agg={tot_pass}/{tot_evaluable} ({rate*100:.1f}%)  "
                  f"t={elapsed:.0f}s",
                  flush=True)
            # Save partial results every 25 cases for crash safety
            if i % 25 == 0:
                with open(DATA / "adversarial_results_amem_partial.json", "w",
                          encoding="utf-8") as f:
                    json.dump({"system": "amem", "n_completed": i,
                               "results": results}, f, indent=2)
        except KeyboardInterrupt:
            print("Interrupted")
            break
        except Exception as e:
            print(f"  [case {case.id}] error: {type(e).__name__}: {e}")
            cat = case_to_attack_category(case.id)
            by_cat.setdefault(cat, {"pass": 0, "fail": 0, "na": 0})
            by_cat[cat]["fail"] += 1
            results.append({
                "case_id": case.id,
                "category": cat,
                "passed": False,
                "error": f"{type(e).__name__}: {e}",
            })

    # Save
    with open(DATA / "adversarial_results_amem.json", "w", encoding="utf-8") as f:
        json.dump({"system": "amem", "results": results}, f, indent=2)

    with open(DATA / "adversarial_summary_amem.json", "w", encoding="utf-8") as f:
        json.dump({"system": "amem", "by_category": by_cat}, f, indent=2)

    print("\n=== A-MEM aggregate ===")
    tot_pass = tot_fail = tot_na = 0
    for cat, d in sorted(by_cat.items()):
        n = d["pass"] + d["fail"] + d["na"]
        evaluable = d["pass"] + d["fail"]
        rate = (d["pass"] / evaluable * 100) if evaluable else float("nan")
        na_note = f" (N/A {d['na']})" if d["na"] else ""
        print(f"  {cat:30s}  {d['pass']:3d}/{evaluable:<3d}  ({rate:5.1f}%){na_note}")
        tot_pass += d["pass"]
        tot_fail += d["fail"]
        tot_na += d["na"]
    tot_evaluable = tot_pass + tot_fail
    print(f"  {'OVERALL':30s}  {tot_pass}/{tot_evaluable}  "
          f"({tot_pass/tot_evaluable*100:.1f}%)  N/A {tot_na}")


if __name__ == "__main__":
    main()

