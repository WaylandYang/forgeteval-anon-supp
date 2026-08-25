"""Run Mem0 v2.0.2 with infer=True (the token-efficient
ADD-only-with-entity-extraction algorithm) on ForgetEval-Adv.

The OSS v2.0.2 release ships
ADDITIVE_EXTRACTION_PROMPT, extract_entities*, and BM25+entity
multi-signal scoring (mem0/memory/main.py imports).  Our headline
Mem0 number used infer=False (pure ADD without extraction); this
variant runs infer=True with a real LLM (DeepSeek-V3 via
SiliconFlow) which exercises the published token-efficient
algorithm path.

Output: data/adversarial_results_v05_mem0_v3.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

OUT = Path(__file__).resolve().parent.parent / "data"

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""


def build_mem0_v3():
    """Mem0 v2.0.2 configured with infer=True + a real OpenAI-compatible
    LLM (DeepSeek-V3 via SiliconFlow), exercising the token-efficient
    ADDITIVE_EXTRACTION_PROMPT path."""
    # Mem0 reads OPENAI_API_KEY env var unconditionally; route to SiliconFlow.
    os.environ["OPENAI_API_KEY"] = SF_KEY
    os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"

    from mem0 import Memory
    import tempfile
    qpath = tempfile.mkdtemp(prefix="mem0v3_fe_")
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "forget_eval_v3",
                "path": qpath,
                "embedding_model_dims": 384,
                "on_disk": True,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "deepseek-ai/DeepSeek-V3",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        },
    }
    return Memory.from_config(config)


def run_case(m, case, infer: bool = True):
    """Run one ForgetEval-Adv case through Mem0 with infer=infer."""
    user_id = "fe_v3"
    m.delete_all(user_id=user_id)
    for fact in case.setup_facts:
        m.add(fact, user_id=user_id, infer=infer)
    for mut in case.mutations:
        op = mut[0]
        if op == "supersede":
            # Mem0 v3 expects supersede via add with infer=True (router
            # will choose UPDATE/DELETE).  We add the new text; the
            # token-efficient algorithm should decide what to do.
            m.add(mut[2], user_id=user_id, infer=infer)
        elif op == "purge":
            # Search for matches and delete them.
            out = m.search(query=mut[1],
                           filters={"user_id": user_id}, top_k=20)
            items = (out.get("results") if isinstance(out, dict)
                     else out) or []
            for it in items:
                if isinstance(it, dict) and it.get("id"):
                    try:
                        m.delete(memory_id=it["id"])
                    except Exception:
                        pass
        elif op == "release":
            out = m.search(query=mut[1],
                           filters={"user_id": user_id}, top_k=20)
            items = (out.get("results") if isinstance(out, dict)
                     else out) or []
            # Use the same gap-threshold policy as LangGraph release for
            # fairness.
            scores = [it.get("score", 0.0) for it in items
                      if isinstance(it, dict)]
            if scores:
                from bench.forgeteval.adapter import LetheAdapter
                thr = LetheAdapter._gap_threshold(scores)
                for it in items:
                    if isinstance(it, dict) and it.get("score", 0.0) >= thr:
                        try:
                            m.delete(memory_id=it["id"])
                        except Exception:
                            pass

    # Recall
    out = m.search(query=case.final_query,
                   filters={"user_id": user_id}, top_k=10)
    items = (out.get("results") if isinstance(out, dict) else out) or []
    texts = [it.get("memory") or it.get("text") or ""
             for it in items if isinstance(it, dict)]
    blob = " ".join(texts).lower()
    must_ok = all(s.lower() in blob for s in case.must_contain)
    not_ok = all(s.lower() not in blob for s in case.must_not_contain)
    return must_ok and not_ok


def main():
    from bench.forgeteval.adversarial import (
        ADVERSARIAL_TESTS, ATTACK_CATEGORIES, case_to_attack_category,
    )

    # At minimum prefix_collision + cross_lingual.
    # We run full bench for completeness.
    target_cats: set[str] = set(os.environ.get(
        "TARGET_CATS", "all"
    ).split(",")) if os.environ.get("TARGET_CATS") else set()

    print(f"Mem0 v3 (v2.0.2 + infer=True + DeepSeek-V3 LLM)")
    print(f"Target cats: {'all' if not target_cats else target_cats}")

    print(f"\nbuilding Mem0 with token-efficient algorithm...", flush=True)
    m = build_mem0_v3()
    print("ok\n", flush=True)

    results = []
    n_pass = 0
    t0 = time.perf_counter()
    cases_to_run = [
        c for c in ADVERSARIAL_TESTS
        if not target_cats or case_to_attack_category(c.id) in target_cats
    ]
    print(f"running on {len(cases_to_run)} cases\n", flush=True)
    for i, case in enumerate(cases_to_run, 1):
        cat = case_to_attack_category(case.id)
        try:
            passed = run_case(m, case, infer=True)
            err = None
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        results.append({
            "id": case.id, "family": case.family,
            "attack_category": cat, "passed": passed, "error": err,
        })
        if passed:
            n_pass += 1
        if i % 5 == 0 or i == len(cases_to_run):
            try:
                print(f"  [{i:3}/{len(cases_to_run)}] "
                      f"{'pass' if passed else 'fail'} {case.id}  "
                      f"so-far {n_pass}/{i} = {n_pass/i*100:.1f}%",
                      flush=True)
            except UnicodeEncodeError:
                pass

    wall = time.perf_counter() - t0
    print(f"\noverall = {n_pass/len(cases_to_run):.4f} "
          f"({n_pass}/{len(cases_to_run)})  wall={wall:.1f}s")

    by_cat: dict[str, dict] = {}
    for cat in ATTACK_CATEGORIES:
        rows = [r for r in results if r["attack_category"] == cat]
        if not rows:
            continue
        passed = sum(1 for r in rows if r["passed"])
        by_cat[cat] = {"pass": passed, "total": len(rows),
                        "rate": passed / max(len(rows), 1)}
        print(f"  {cat:<28} {passed:>2}/{len(rows):<3} "
              f"({passed/len(rows)*100:>3.0f}%)")

    by_family: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        by_family[r["family"]]["total"] += 1
        if r["passed"]:
            by_family[r["family"]]["pass"] += 1
    for fam, d in by_family.items():
        d["rate"] = d["pass"] / max(d["total"], 1)

    out_data = [{
        "adapter": "mem0_v3_infer_true",
        "config": "v2.0.2 + infer=True + DeepSeek-V3 LLM (token-efficient path)",
        "suite": "adversarial",
        "case_count": len(cases_to_run),
        "overall_pass": n_pass,
        "overall_total": len(cases_to_run),
        "overall_rate": n_pass / max(len(cases_to_run), 1),
        "by_family": dict(by_family),
        "by_attack_category": by_cat,
        "per_case": results,
        "wall_seconds": wall,
    }]
    out_path = OUT / "adversarial_results_v05_mem0_v3.json"
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

