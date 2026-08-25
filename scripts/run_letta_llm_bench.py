"""Run ForgetEval-Adv on Letta + mutation-time LLM hook.

This validates the §5.7 "partly complementary" prediction: a system
combining inscribe-time embedding (Letta's pgvector) with the
mutation-time LLM hook (our pattern) should recover BOTH
canonicalization AND intent-aware deletion categories simultaneously.

Adapter design:
  inscribe: same as LettaAdapter (POST archival-memory)
  recall:   same as LettaAdapter (GET archival-memory, no LLM)
  supersede(old_q, new_text):
       1. retrieve top-5 candidates by old_q
       2. LLM plans which candidate(s) to delete and what to insert
       3. apply plan
  purge(query):
       1. retrieve top-5 candidates by query
       2. LLM picks which to delete
       3. apply

Recall hot path stays LLM-free (apples-to-apples).
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")

LETTA_URL = os.environ.get("LETTA_URL", "http://127.0.0.1:8283")
SF_KEY = (os.environ.get("LLM_API_KEY")
          or os.environ.get("OPENROUTER_API_KEY", ""))
SF_BASE = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
HOOK_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash")
MAX_TOK = int(os.environ.get("LLM_MAX_TOKENS", "3000"))

# Archival memory needs an embedding provider; use the benchmark's
# own, served locally, as the deterministic Letta runner does.
EMB_BASE = os.environ.get("LETTA_EMBED_BASE",
                          "http://host.docker.internal:8399/v1")
EMB_MODEL = os.environ.get("LETTA_EMBED_MODEL",
                           "sentence-transformers/all-MiniLM-L6-v2")
EMB_DIM = int(os.environ.get("LETTA_EMBED_DIM", "384"))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

LLM_CONFIG = {
    "model": "deepseek-ai/DeepSeek-V3.1-Terminus",
    "model_endpoint_type": "openai",
    "model_endpoint": SF_BASE,
    "context_window": 30000,
}
EMB_CONFIG = {
    "embedding_model": EMB_MODEL,
    "embedding_endpoint_type": "openai",
    "embedding_endpoint": EMB_BASE,
    "embedding_dim": EMB_DIM,
    "embedding_chunk_size": 300,
}


SUPERSEDE_PROMPT = """You are a memory-planning assistant. Given an
old query, a new replacement text, and the current memory
candidates that match the old query, decide which candidates
should be DELETED (because they conflict with the new fact) and
what NEW text should be INSERTED.

old_query: {old_q}
new_text: {new_text}

candidates (id : text):
{candidates}

Return strict JSON with EXACTLY these keys:
  "to_delete": [list of candidate ids to delete]
  "to_insert": "the text to insert (typically the new_text)"

Rules:
- Delete only candidates whose factual claim conflicts with new_text.
- Keep candidates that mention the same entity but a different
  attribute (e.g. delete "Alice email = X" but keep "Alice phone = Y").
- Consider canonicalization: if a candidate is the same identifier
  in different surface form (case, separator, alias), delete it.
- Output ONLY the JSON object, nothing else."""

PURGE_PROMPT = """You are a memory-planning assistant. Given a
purge query, decide which of the matching memory candidates
should be DELETED. The user wants to forget everything matching
the query (including surface variants).

purge_query: {query}

candidates (id : text):
{candidates}

Return strict JSON with EXACTLY this key:
  "to_delete": [list of candidate ids to delete]

Rules:
- Delete every candidate that refers to the same entity as the
  query (including surface variants: case, separators, aliases,
  prefix/suffix differences, cross-language transliterations).
- Do NOT delete candidates that share only an attribute (e.g.
  same domain or same company); only delete same-entity matches.
- Output ONLY the JSON object, nothing else."""




def _suite_suffix(tag):
    """"" for the default suite, "_external" and so on for the others.

    Without this an external run overwrites the main-suite result for the
    same system, which is a 77-case file replacing a 385-case one.
    """
    return "_external" if "external" in (tag or "") else ""
def _cat(case_id):
    """Attack category, falling back to the external subset's own field."""
    c = case_to_attack_category(case_id)
    if c == "unknown":
        try:
            from bench.forgeteval.external import external_category
            return external_category(case_id)
        except Exception:
            return c
    return c
def sf_chat(prompt: str) -> dict:
    """Call SiliconFlow chat-completions; return parsed JSON dict."""
    r = httpx.post(
        f"{SF_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {SF_KEY}"},
        json={
            "model": HOOK_MODEL,
            "max_tokens": MAX_TOK,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        },
        timeout=60.0,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    # Robust JSON extraction
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"to_delete": [], "to_insert": ""}


class LettaLLMAdapter:
    name = "letta+llm"

    def __init__(self, base_url: str = LETTA_URL):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)
        self.agent_id: str | None = None

    def _agent_create(self):
        r = self.client.post(
            f"{self.base}/v1/agents/",
            json={
                "name": "fe_llm_" + uuid.uuid4().hex[:8],
                "system": "You are a memory assistant.",
                "llm_config": LLM_CONFIG,
                "embedding_config": EMB_CONFIG,
            },
        )
        r.raise_for_status()
        self.agent_id = r.json()["id"]

    def _agent_delete(self):
        if self.agent_id:
            try:
                self.client.delete(f"{self.base}/v1/agents/{self.agent_id}")
            except Exception:
                pass
            self.agent_id = None

    def reset(self):
        self._agent_delete()
        self._agent_create()

    def inscribe(self, text: str) -> str:
        r = self.client.post(
            f"{self.base}/v1/agents/{self.agent_id}/archival-memory",
            json={"text": text},
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]["id"]
        return ""

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        r = self.client.get(
            f"{self.base}/v1/agents/{self.agent_id}/archival-memory",
            params={"query": query, "limit": k},
        )
        r.raise_for_status()
        return [p["text"] for p in r.json()
                if isinstance(p, dict) and "text" in p]

    def _search_raw(self, query: str, k: int = 5) -> list[dict]:
        r = self.client.get(
            f"{self.base}/v1/agents/{self.agent_id}/archival-memory",
            params={"query": query, "limit": k},
        )
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []

    def _delete_passage(self, pid: str):
        try:
            self.client.delete(
                f"{self.base}/v1/agents/{self.agent_id}/archival-memory/{pid}"
            )
        except Exception:
            pass

    def supersede(self, old_query: str, new_text: str) -> None:
        # Get top-5 candidates
        hits = self._search_raw(old_query, k=5)
        if not hits:
            self.inscribe(new_text)
            return
        candidates_str = "\n".join(
            f"  {h['id']} : {h.get('text', '')}" for h in hits[:5]
            if isinstance(h, dict)
        )
        plan = sf_chat(SUPERSEDE_PROMPT.format(
            old_q=old_query, new_text=new_text, candidates=candidates_str
        ))
        for pid in plan.get("to_delete", []) or []:
            if pid:
                self._delete_passage(str(pid))
        ins_text = plan.get("to_insert") or new_text
        if ins_text:
            self.inscribe(ins_text)

    def release(self, query: str) -> int:
        # Letta has no soft-delete; under LLM hook, we *could* simulate
        # release by purging, but that would conflate semantics.  Stay N/A.
        raise NotImplementedError("Letta+LLM: no release primitive")

    def purge(self, query: str) -> int:
        hits = self._search_raw(query, k=10)
        if not hits:
            return 0
        candidates_str = "\n".join(
            f"  {h['id']} : {h.get('text', '')}" for h in hits[:10]
            if isinstance(h, dict)
        )
        plan = sf_chat(PURGE_PROMPT.format(
            query=query, candidates=candidates_str
        ))
        n = 0
        for pid in plan.get("to_delete", []) or []:
            if pid:
                self._delete_passage(str(pid))
                n += 1
        return n


def _forgeteval_suite():
    """(cases, tag, probed) for the suite named on the command line."""
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--suite", choices=["v051", "v07", "external"],
                   default="v051")
    p.add_argument("--probed", action="store_true")
    a, _ = p.parse_known_args()
    if a.suite == "external":
        from bench.forgeteval.external import load_external_cases
        cases = load_external_cases()
    elif a.suite == "v07":
        from scripts.repair_cross_lingual_queries import build_suite
        cases, _n = build_suite()
    else:
        from bench.forgeteval.adversarial import ADVERSARIAL_TESTS as cases
    tag = ("" if a.suite == "v051" else "_" + a.suite) + \
          ("_probed" if a.probed else "")
    return list(cases), tag, a.probed


def _forgeteval_run(case, adapter, probed):
    if probed:
        from bench.forgeteval.scoring import run_scored
        return run_scored(case, adapter, probed=True)
    return case.run(adapter)


def stratified_sample(cases, n_per_cat: int = 10):
    by_cat: dict[str, list] = {}
    for c in cases:
        cat = _cat(c.id)
        by_cat.setdefault(cat, []).append(c)
    out = []
    for cat, lst in by_cat.items():
        out.extend(lst[:n_per_cat])
    return out


def main():
    cases, _fe_tag, _fe_probed = _forgeteval_suite()
    sample = int(os.environ.get("SAMPLE_PER_CAT", "0"))
    if sample > 0:
        cases = stratified_sample(cases, n_per_cat=sample)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        cases = cases[:limit]
    print(f"Running {len(cases)} cases on Letta+LLM @ {LETTA_URL}",
          flush=True)

    adapter = LettaLLMAdapter()
    results = []
    by_cat: dict[str, dict[str, int]] = {}
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        cat = _cat(case.id)
        try:
            try:
                passed = _forgeteval_run(case, adapter, _fe_probed)
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

            results.append({"case_id": case.id, "category": cat,
                            "passed": passed, "applied": applied})

            elapsed = time.time() - t_start
            tot_p = sum(c["pass"] for c in by_cat.values())
            tot_e = sum(c["pass"] + c["fail"] for c in by_cat.values())
            rate = tot_p / tot_e * 100 if tot_e else 0
            verdict = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
            print(f"  {i:3d}/{len(cases):3d} {case.id[:40]:40s} {cat:25s} {verdict}  "
                  f"agg={tot_p}/{tot_e} ({rate:.1f}%)  t={elapsed:.0f}s",
                  flush=True)

            if i % 25 == 0:
                with open(DATA / ("adversarial_results_letta_llm_partial" + _suite_suffix(_fe_tag) + ".json"),
                          "w", encoding="utf-8") as f:
                    json.dump({"n": i, "results": results, "by_cat": by_cat},
                              f, indent=2)
        except KeyboardInterrupt:
            print("Interrupted")
            break
        except Exception as e:
            print(f"  [case {case.id}] error: {type(e).__name__}: {e}",
                  flush=True)
            by_cat.setdefault(cat, {"pass": 0, "fail": 0, "na": 0})
            by_cat[cat]["fail"] += 1
            results.append({"case_id": case.id, "category": cat,
                            "passed": False,
                            "error": f"{type(e).__name__}: {e}"})

    adapter._agent_delete()

    with open(DATA / ("adversarial_summary_letta_llm" + _suite_suffix(_fe_tag) + ".json"), "w",
              encoding="utf-8") as f:
        json.dump({"system": "letta+llm", "by_category": by_cat,
                   "hook_model": HOOK_MODEL,
                   "letta_llm": LLM_CONFIG["model"],
                   "n_total": len(results)}, f, indent=2)
    with open(DATA / ("adversarial_results_letta_llm" + _suite_suffix(_fe_tag) + ".json"), "w",
              encoding="utf-8") as f:
        json.dump({"system": "letta+llm", "hook_model": HOOK_MODEL,
                   "results": results}, f, indent=2)

    print("\n=== Letta+LLM aggregate ===", flush=True)
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

