"""Run ForgetEval-Adv on Letta (self-hosted v0.16.7 via Docker on
127.0.0.1).  Each test case gets its own Letta agent (per-case
isolation).  We use Letta's archival-memory REST endpoints DIRECTLY
(not via agent.send_message), keeping the LLM out of the recall hot
path so the comparison is apples-to-apples with our four primary
systems.

Mapping (6-method Protocol):
  inscribe(text)  -> POST /v1/agents/{aid}/archival-memory body={"text": text}
  recall(query,k) -> GET  /v1/agents/{aid}/archival-memory?query=...&limit=k
                     returns list of passage dicts with raw `text`
                     (no LLM synthesis)
  supersede(old_q, new_text) -> search top-1 by old_q,
                                DELETE that passage_id, insert new_text
  release(query)             -> NotImplementedError (Letta has no
                                soft-delete / hidden primitive)
  purge(query)               -> search top-k by query, DELETE each id

Inscribe triggers an embedding call (~150ms via SiliconFlow bge-m3);
search uses pgvector cosine similarity (server-side, no LLM call).

Output: data/adversarial_summary_letta.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")

LETTA_URL = os.environ.get("LETTA_URL", "http://127.0.0.1:8283")
SF_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
# Archival memory needs an embedding provider. The hosted one this was
# written against needs a key that is not part of the release, so the
# default is the benchmark's own embedder served over the OpenAI wire
# format by scripts/local_embeddings_server.py. Same model as every other
# adapter here, which also removes an axis on which this row differed.
EMB_BASE = os.environ.get("LETTA_EMBED_BASE",
                          "http://host.docker.internal:8399/v1")
EMB_MODEL = os.environ.get("LETTA_EMBED_MODEL",
                           "sentence-transformers/all-MiniLM-L6-v2")
EMB_DIM = int(os.environ.get("LETTA_EMBED_DIM", "384"))
SF_BASE = "https://api.siliconflow.cn/v1"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)




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
    return _forgeteval_run(case, adapter, _fe_probed)


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


class LettaAdapter:
    name = "letta"

    def __init__(self, base_url: str = LETTA_URL):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)
        self.agent_id: str | None = None

    def _agent_create(self):
        r = self.client.post(
            f"{self.base}/v1/agents/",
            json={
                "name": "fe_" + uuid.uuid4().hex[:8],
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
        # The endpoint returns a list of passages (chunked); we use the
        # first id as the canonical handle.
        if isinstance(data, list) and data:
            return data[0]["id"]
        return ""

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        r = self.client.get(
            f"{self.base}/v1/agents/{self.agent_id}/archival-memory",
            params={"query": query, "limit": k},
        )
        r.raise_for_status()
        passages = r.json()
        return [p["text"] for p in passages if isinstance(p, dict) and "text" in p]

    def _delete_passage(self, pid: str):
        try:
            self.client.delete(
                f"{self.base}/v1/agents/{self.agent_id}/archival-memory/{pid}"
            )
        except Exception:
            pass

    def _search_raw(self, query: str, k: int = 5) -> list[dict]:
        r = self.client.get(
            f"{self.base}/v1/agents/{self.agent_id}/archival-memory",
            params={"query": query, "limit": k},
        )
        r.raise_for_status()
        return r.json()

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self._search_raw(old_query, k=1)
        if hits and isinstance(hits[0], dict) and hits[0].get("id"):
            self._delete_passage(hits[0]["id"])
        self.inscribe(new_text)

    def release(self, query: str) -> int:
        # Letta has no soft-delete / hidden semantic.
        raise NotImplementedError("Letta: no release primitive")

    def purge(self, query: str) -> int:
        hits = self._search_raw(query, k=10)
        n = 0
        for h in hits:
            if isinstance(h, dict) and h.get("id"):
                self._delete_passage(h["id"])
                n += 1
        return n


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
    sample = int(os.environ.get("SAMPLE_PER_CAT", "10"))
    if sample > 0:
        cases = stratified_sample(cases, n_per_cat=sample)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        cases = cases[:limit]
    print(f"Running {len(cases)} cases on Letta @ {LETTA_URL}", flush=True)

    adapter = LettaAdapter()
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
                with open(DATA / ("adversarial_results_letta_partial" + _suite_suffix(_fe_tag) + ".json"),
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

    with open(DATA / ("adversarial_summary_letta" + _suite_suffix(_fe_tag) + ".json"), "w",
              encoding="utf-8") as f:
        json.dump({"system": "letta", "by_category": by_cat,
                   "model": LLM_CONFIG["model"],
                   "embedding_model": EMB_CONFIG["embedding_model"],
                   "n_total": len(results)}, f, indent=2)
    with open(DATA / ("adversarial_results_letta" + _suite_suffix(_fe_tag) + ".json"), "w",
              encoding="utf-8") as f:
        json.dump({"system": "letta",
                   "model": LLM_CONFIG["model"],
                   "results": results}, f, indent=2)

    print("\n=== Letta aggregate ===", flush=True)
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

