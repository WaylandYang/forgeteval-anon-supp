"""Mem0 v2.0.2 + infer=True + DeepSeek-V3 hook on the full 385-case
ForgetEval-Adv, with a JSON-validation retry wrapper around Mem0's
LLM to address the DeepSeek-V3 vs. gpt-4o-mini prompt-tuning gap.

Fairness to Mem0: infer=False reports a stripped-down Mem0; this
script reports infer=True with retries so the headline reflects
Mem0's published token-efficient algorithm path.

Output: data/adversarial_results_v05_mem0_v3.json
"""
from __future__ import annotations

import io
import json
import os
import re
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
HOOK_MODEL = os.environ.get("HOOK_MODEL", "deepseek-ai/DeepSeek-V3")
MAX_RETRIES = int(os.environ.get("JSON_RETRIES", "3"))

RETRY_STATS = {"calls": 0, "repaired": 0, "api_retries": 0, "final_failures": 0}


def _strip_markdown(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                     flags=re.MULTILINE)
    return cleaned.strip()


def _try_parse(text: str):
    """Returns parsed JSON or raises."""
    return json.loads(_strip_markdown(text), strict=False)


def _try_repair(text: str) -> str | None:
    """Use json-repair to fix common malformed-JSON outputs from LLMs.
    Returns repaired JSON string, or None if repair did not yield valid JSON."""
    import json_repair
    cleaned = _strip_markdown(text)
    try:
        repaired = json_repair.repair_json(cleaned)
        # repair_json always returns SOME string; verify it parses
        json.loads(repaired, strict=False)
        return repaired
    except Exception:
        return None


def patch_openai_llm_with_retry():
    """Monkey-patch Mem0's OpenAILLM.generate_response to (a) locally repair
    malformed JSON via json-repair and (b) re-call the API with a stricter
    prompt if repair fails, when response_format requests a JSON object."""
    from mem0.llms.openai import OpenAILLM
    orig = OpenAILLM.generate_response

    def patched(self, messages, response_format=None, tools=None,
                tool_choice="auto", **kwargs):
        RETRY_STATS["calls"] += 1
        wants_json = (
            response_format is not None
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )

        response = orig(self, messages, response_format=response_format,
                        tools=tools, tool_choice=tool_choice, **kwargs)
        if not wants_json or not isinstance(response, str):
            return response

        # Path 1 — strict parse OK, return as-is.
        try:
            _try_parse(response)
            return response
        except Exception:
            pass

        # Path 2 — local json-repair (handles bad commas, missing braces, ...)
        repaired = _try_repair(response)
        if repaired is not None:
            RETRY_STATS["repaired"] += 1
            return repaired

        # Path 3 — API retry with stricter follow-up prompt.
        last_response = response
        last_error = "previous output was malformed"
        for attempt in range(MAX_RETRIES):
            RETRY_STATS["api_retries"] += 1
            try:
                _try_parse(last_response)
                last_error = "(impossible: should have returned earlier)"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
            cur_messages = list(messages) + [
                {"role": "assistant", "content": last_response},
                {"role": "user",
                 "content": (
                     "Your previous response was not valid JSON "
                     f"({last_error}).  Reply with a VALID JSON object only "
                     "— no markdown, no prose, no commas inside string "
                     "values, no trailing text."
                 )},
            ]
            last_response = orig(self, cur_messages,
                                 response_format=response_format,
                                 tools=tools, tool_choice=tool_choice,
                                 **kwargs)
            try:
                _try_parse(last_response)
                return last_response
            except Exception:
                pass
            # Try repair again on retry output.
            r2 = _try_repair(last_response)
            if r2 is not None:
                RETRY_STATS["repaired"] += 1
                return r2

        RETRY_STATS["final_failures"] += 1
        return last_response

    OpenAILLM.generate_response = patched
    print(f"[robust-llm] patched OpenAILLM.generate_response: "
          f"json-repair + up to {MAX_RETRIES} API retries", flush=True)


def build_mem0_v3():
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
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": HOOK_MODEL,
                "temperature": 0.0,
                "max_tokens": 4096,  # bump from 1024 to avoid truncation
            },
        },
    }
    return Memory.from_config(config)


def run_case(m, case):
    user_id = "fe_v3"
    m.delete_all(user_id=user_id)
    for fact in case.setup_facts:
        m.add(fact, user_id=user_id, infer=True)
    for mut in case.mutations:
        op = mut[0]
        if op == "supersede":
            m.add(mut[2], user_id=user_id, infer=True)
        elif op == "purge":
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
    patch_openai_llm_with_retry()

    from bench.forgeteval.adversarial import (
        ADVERSARIAL_TESTS, ATTACK_CATEGORIES, case_to_attack_category,
    )

    print(f"Mem0 v3 (v2.0.2 + infer=True + {HOOK_MODEL} + JSON-retry)")
    print(f"building Mem0...", flush=True)
    m = build_mem0_v3()
    print("ok\n", flush=True)

    results = []
    n_pass = 0
    t0 = time.perf_counter()
    cases_to_run = list(ADVERSARIAL_TESTS)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        cases_to_run = cases_to_run[:limit]
    print(f"running on {len(cases_to_run)} cases\n", flush=True)
    for i, case in enumerate(cases_to_run, 1):
        cat = case_to_attack_category(case.id)
        try:
            passed = run_case(m, case)
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
        if i % 10 == 0 or i == len(cases_to_run):
            try:
                print(f"  [{i:3}/{len(cases_to_run)}] "
                      f"{'pass' if passed else 'fail'} {case.id[:42]}  "
                      f"so-far {n_pass}/{i} = {n_pass/i*100:.1f}%  "
                      f"repaired={RETRY_STATS['repaired']} "
                      f"api_retries={RETRY_STATS['api_retries']} "
                      f"fail={RETRY_STATS['final_failures']}",
                      flush=True)
            except UnicodeEncodeError:
                pass

        # Incremental save every 25 cases for resume safety.
        if i % 25 == 0:
            partial = OUT / "adversarial_results_v05_mem0_v3_partial.json"
            partial.write_text(json.dumps({
                "n": i, "n_pass": n_pass, "results": results,
                "retry_stats": dict(RETRY_STATS),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    wall = time.perf_counter() - t0
    print(f"\noverall = {n_pass/len(cases_to_run):.4f} "
          f"({n_pass}/{len(cases_to_run)})  wall={wall:.1f}s")
    print(f"LLM calls={RETRY_STATS['calls']}, "
          f"local repairs={RETRY_STATS['repaired']}, "
          f"API retries={RETRY_STATS['api_retries']}, "
          f"final failures={RETRY_STATS['final_failures']}")

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
        "adapter": "mem0_v3_infer_true_robust",
        "config": (f"v2.0.2 + infer=True + {HOOK_MODEL} + "
                   f"JSON-retry (up to {MAX_RETRIES})"),
        "suite": "adversarial",
        "case_count": len(cases_to_run),
        "overall_pass": n_pass,
        "overall_total": len(cases_to_run),
        "overall_rate": n_pass / max(len(cases_to_run), 1),
        "by_family": dict(by_family),
        "by_attack_category": by_cat,
        "per_case": results,
        "wall_seconds": wall,
        "retry_stats": dict(RETRY_STATS),
    }]
    out_path = OUT / "adversarial_results_v05_mem0_v3.json"
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

