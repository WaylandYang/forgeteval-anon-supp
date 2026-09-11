"""Run 5 ecosystem systems on the 77-case external-authored subset:
Letta, A-MEM, Graphiti, Letta+LLM, Mem0+v3 (infer=True + json-repair).

Imports adapter classes from each system's existing bench script and
runs admitted external cases through them.  Each system writes to its
own JSON; aggregator at the end folds them into external_subset_results.json.
"""
from __future__ import annotations
import io
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
DATA = PAPER_ROOT / "data"

sys.path.insert(0, str(REPO_ROOT / "lethe"))
sys.path.insert(0, str(SCRIPTS_DIR))

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"
LETTA_URL = os.environ.get("LETTA_URL", "http://127.0.0.1:8283")


class ExtCase:
    """GeneratedCase-compatible wrapper for external cases."""
    def __init__(self, d: dict):
        self.id = d["id"]
        self.family = "ext"
        self.category = d["category"]
        self.attack_category = d["category"]
        self.setup_facts = d["setup_facts"]
        self.mutations = [tuple(m) for m in d["mutations"]]
        self.final_query = d["final_query"]
        self.must_contain = d["must_contain"]
        self.must_not_contain = d["must_not_contain"]

    def run(self, adapter) -> bool:
        adapter.reset()
        for f in self.setup_facts:
            adapter.inscribe(f)
        for mut in self.mutations:
            op = mut[0]
            try:
                if op == "supersede":
                    _, oldq, newt = mut
                    adapter.supersede(oldq, newt)
                elif op == "purge":
                    _, q = mut
                    adapter.purge(q)
                elif op == "release":
                    _, q = mut
                    adapter.release(q)
            except NotImplementedError:
                raise
        retrieved = adapter.recall_texts(self.final_query, k=10)
        blob = " ".join(retrieved).lower()
        must_ok = all(s.lower() in blob for s in self.must_contain)
        not_ok = all(s.lower() not in blob for s in self.must_not_contain)
        return must_ok and not_ok


def load_admitted():
    path = DATA / "external_subset_cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ExtCase(c) for c in data["admitted_cases"]]


def evaluate(name: str, adapter, cases) -> dict:
    n_pass = n_na = 0
    per_case = []
    by_cat: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0, "na": 0})
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
            print(f"  {name} [{i:3}/{len(cases)}] "
                  f"so-far {n_pass}/{i-n_na if i>n_na else 0} (N/A {n_na})",
                  flush=True)
    wall = time.time() - t0
    n_eval = sum(d["total"] for d in by_cat.values())
    rate = n_pass / n_eval * 100 if n_eval else 0
    return {"n_pass": n_pass, "n_eval": n_eval, "n_na": n_na, "rate": rate,
            "by_category": dict(by_cat), "per_case": per_case,
            "wall_s": wall}


def print_breakdown(name: str, result: dict):
    print(f"\n=== {name} summary ===")
    r = result
    print(f"  pass: {r['n_pass']}/{r['n_eval']} ({r['rate']:.1f}%)  N/A: {r['n_na']}  wall: {r['wall_s']:.1f}s")
    for cat in sorted(r["by_category"]):
        d = r["by_category"][cat]
        rate = d["pass"] / d["total"] * 100 if d["total"] else 0
        print(f"    {cat:30s} {d['pass']:2}/{d['total']:<2} ({rate:5.1f}%)  N/A: {d['na']}")


def merge_into_global(name: str, result: dict):
    """Add this system's result into data/external_subset_results.json."""
    path = DATA / "external_subset_results.json"
    if path.exists():
        global_data = json.loads(path.read_text(encoding="utf-8"))
    else:
        global_data = {"systems": {}}
    global_data.setdefault("systems", {})[name] = result
    path.write_text(json.dumps(global_data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def make_sf_llm(model: str = "deepseek-ai/DeepSeek-V3"):
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)
    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, max_tokens=2048, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    return llm


# ───── A-MEM ─────────────────────────────────────────────────────────────
def run_amem(cases):
    print("\n========== A-MEM ==========", flush=True)
    os.environ["OPENAI_API_KEY"] = SF_KEY
    os.environ["OPENAI_BASE_URL"] = SF_BASE
    from run_amem_bench import AMemSF
    adapter = AMemSF()
    r = evaluate("A-MEM", adapter, cases)
    print_breakdown("A-MEM", r)
    merge_into_global("A-MEM", r)


# ───── Letta (no LLM hook) ──────────────────────────────────────────────
def run_letta(cases):
    print("\n========== Letta ==========", flush=True)
    from run_letta_bench import LettaAdapter  # noqa: E402
    adapter = LettaAdapter(LETTA_URL)
    adapter.reset()  # creates agent
    try:
        r = evaluate("Letta", adapter, cases)
        print_breakdown("Letta", r)
        merge_into_global("Letta", r)
    finally:
        try:
            adapter._agent_delete()
        except Exception:
            pass


# ───── Letta+LLM ────────────────────────────────────────────────────────
def run_letta_llm(cases):
    print("\n========== Letta+LLM ==========", flush=True)
    from run_letta_llm_bench import LettaLLMAdapter
    adapter = LettaLLMAdapter(LETTA_URL)
    adapter.reset()
    try:
        r = evaluate("Letta+LLM", adapter, cases)
        print_breakdown("Letta+LLM", r)
        merge_into_global("Letta+LLM", r)
    finally:
        try:
            adapter._agent_delete()
        except Exception:
            pass


# ───── Graphiti ─────────────────────────────────────────────────────────
def run_graphiti(cases):
    print("\n========== Graphiti ==========", flush=True)
    from run_graphiti_bench import GraphitiAdapter
    adapter = GraphitiAdapter()
    r = evaluate("Graphiti", adapter, cases)
    print_breakdown("Graphiti", r)
    merge_into_global("Graphiti", r)


# ───── Mem0+v3 (infer=True + json-repair) ──────────────────────────────
def run_mem0_v3(cases):
    print("\n========== Mem0+v3 ==========", flush=True)
    # Patch OpenAILLM with json-repair retry
    from run_mem0_v3_robust import (
        patch_openai_llm_with_retry, build_mem0_v3, run_case, RETRY_STATS,
    )
    patch_openai_llm_with_retry()
    m = build_mem0_v3()

    n_pass = n_na = 0
    per_case = []
    by_cat: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0, "na": 0})
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            passed = run_case(m, c)
            err = None
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        per_case.append({"id": c.id, "category": c.category,
                         "passed": passed, "error": err})
        d = by_cat[c.category]
        if passed is True:
            d["pass"] += 1; d["total"] += 1; n_pass += 1
        else:
            d["total"] += 1
        if i % 10 == 0 or i == len(cases):
            print(f"  Mem0+v3 [{i:3}/{len(cases)}] "
                  f"so-far {n_pass}/{i} ({n_pass/i*100:.1f}%) "
                  f"repaired={RETRY_STATS.get('repaired',0)}",
                  flush=True)
    wall = time.time() - t0
    n_eval = sum(d["total"] for d in by_cat.values())
    rate = n_pass / n_eval * 100 if n_eval else 0
    r = {"n_pass": n_pass, "n_eval": n_eval, "n_na": n_na, "rate": rate,
         "by_category": dict(by_cat), "per_case": per_case, "wall_s": wall,
         "retry_stats": dict(RETRY_STATS)}
    print_breakdown("Mem0+v3", r)
    merge_into_global("Mem0+v3", r)


SYSTEM_NAME_MAP = {
    "amem": "A-MEM", "mem0_v3": "Mem0+v3", "letta": "Letta",
    "letta_llm": "Letta+LLM", "graphiti": "Graphiti",
}


def already_done(system_name: str) -> bool:
    """Skip if results JSON already contains this system (from a parallel run)."""
    path = DATA / "external_subset_results.json"
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        sys_data = d.get("systems", {}).get(system_name)
        if sys_data and sys_data.get("n_eval", 0) + sys_data.get("n_na", 0) >= 70:
            return True
    except Exception:
        return False
    return False


def main():
    cases = load_admitted()
    print(f"loaded {len(cases)} admitted external cases")

    target = os.environ.get("ONLY_SYSTEM", "all").lower()
    target_set = (set(t.strip() for t in target.split(",")) if target != "all"
                  else None)
    force = os.environ.get("FORCE", "").lower() in ("1", "true", "yes")
    runners = [
        ("amem",      run_amem),
        ("mem0_v3",   run_mem0_v3),
        ("letta",     run_letta),
        ("letta_llm", run_letta_llm),
        ("graphiti",  run_graphiti),
    ]
    for key, fn in runners:
        if target_set is not None and key not in target_set:
            continue
        name = SYSTEM_NAME_MAP[key]
        if not force and already_done(name):
            print(f"\n=== {name} ALREADY DONE (skip; FORCE=1 to re-run) ===",
                  flush=True)
            continue
        try:
            fn(cases)
        except Exception as e:
            print(f"\n!!! {key} FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()

