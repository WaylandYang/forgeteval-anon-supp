"""Re-run all 5 primary systems on the new 20 identifier_obfuscation
cases (ids 19-38) so the v0.5.1 balance is reflected in Table 2.

Outputs:
  data/identifier_obfuscation_v051_results.json
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault(
    "OPENAI_API_KEY",
    "")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))
sys.path.insert(0, str(REPO_ROOT / "lethe-paper" / "scripts"))

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.adapter import (  # noqa: E402
    LetheAdapter, Mem0Adapter, LangGraphAdapter, MemPalaceAdapter,
)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

# Filter to the new 20 hand-crafted identifier_obfuscation cases.
NEW_IDS = {f"adv_identifier_obfuscation_{i:02d}" for i in range(19, 39)}
new_cases = [c for c in ADVERSARIAL_TESTS if c.id in NEW_IDS]
assert len(new_cases) == 20, f"expected 20 new cases, got {len(new_cases)}"


def run_one(name: str, adapter):
    print(f"--- {name} ---", flush=True)
    results = {"system": name, "results": []}
    t0 = time.time()
    pass_count = 0
    for c in new_cases:
        try:
            ok = c.run(adapter)
            results["results"].append({"case_id": c.id, "pass": ok})
            pass_count += int(bool(ok))
        except NotImplementedError:
            results["results"].append({"case_id": c.id, "pass": None})
    wall = time.time() - t0
    evaluable = sum(1 for r in results["results"] if r["pass"] is not None)
    na = sum(1 for r in results["results"] if r["pass"] is None)
    print(f"  {name}: {pass_count}/{evaluable} pass + {na} N/A  ({wall:.1f}s)",
          flush=True)
    results["pass"] = pass_count
    results["evaluable"] = evaluable
    results["na"] = na
    results["wall_s"] = wall
    return results


def main():
    print(f"Running {len(new_cases)} new identifier_obfuscation cases on "
          "5 systems", flush=True)
    out = {"new_case_count": len(new_cases), "by_system": {}}

    # Load embedder once
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embedder(t):
        return list(next(iter(model.embed([t]))))

    for name, factory in [
        ("Lethe",     lambda: LetheAdapter(embedder=embedder, vector_dim=384)),
        ("LangGraph", lambda: LangGraphAdapter(embedder=embedder, vector_dim=384)),
        ("MemPalace", lambda: MemPalaceAdapter()),
        ("Mem0",      lambda: Mem0Adapter()),
    ]:
        try:
            out["by_system"][name] = run_one(name, factory())
        except Exception as e:
            print(f"  [{name}] failed: {type(e).__name__}: {e}", flush=True)
            out["by_system"][name] = {"error": str(e)}

    # Lethe+LLM
    try:
        from lethe import Lethe  # noqa
        SF_KEY = os.environ["OPENAI_API_KEY"]
        SF_BASE = os.environ["OPENAI_BASE_URL"]

        def sf_llm(prompt: str) -> str:
            import requests
            r = requests.post(
                f"{SF_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {SF_KEY}"},
                json={
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        lethe_llm = LetheAdapter(embedder=embedder, vector_dim=384, llm=sf_llm)
        out["by_system"]["Lethe+LLM"] = run_one("Lethe+LLM", lethe_llm)
    except Exception as e:
        print(f"  [Lethe+LLM] failed: {type(e).__name__}: {e}", flush=True)

    # LangGraph+LLM
    try:
        from bench.forgeteval.adapter import LangGraphLLMAdapter  # noqa
        lg_llm = LangGraphLLMAdapter(embedder=embedder, vector_dim=384,
                                      llm=sf_llm)
        out["by_system"]["LangGraph+LLM"] = run_one("LangGraph+LLM", lg_llm)
    except Exception as e:
        print(f"  [LangGraph+LLM] failed: {type(e).__name__}: {e}", flush=True)

    out_path = DATA / "identifier_obfuscation_v051_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    print("\n=== Summary on v0.5.1 additions (n=20) ===")
    print(f"{'System':15s}  {'pass':>5s}/{'evl':<3s}  {'N/A':>3s}  {'wall':>6s}")
    for name, d in out["by_system"].items():
        if "pass" in d:
            print(f"{name:15s}  {d['pass']:3d}/{d['evaluable']:<3d}  "
                  f"{d['na']:>3d}  {d['wall_s']:>5.1f}s")


if __name__ == "__main__":
    main()

