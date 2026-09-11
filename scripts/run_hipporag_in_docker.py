"""Run HippoRAG bench inside the `hipporag:cpu2` docker container.

Wraps HippoRAG (NeurIPS 2024, KG + PPR) into the 6-method Protocol.
Designed to be invoked INSIDE the docker container with bench cases
mounted at /cases.json and writing results to /out.json.

Mapping (6-method Protocol):
  inscribe(text)  -> hipporag.index([text])
  recall_texts(q) -> hipporag.retrieve([q], num_to_retrieve=k)
  supersede(old_q, new)
                  -> retrieve top-1 by old_q, hipporag.delete([that doc]),
                     index([new])
  release(q)      -> NotImplementedError (no soft-delete primitive)
  purge(q)        -> retrieve top-k, hipporag.delete(retrieved docs)

Each case re-creates the HippoRAG instance into a fresh save_dir
(per-case isolation; HippoRAG persists the KG to disk).
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import httpx  # for keeping interface consistent

sys.stdout.reconfigure(encoding="utf-8")

# These come from environment (no hardcoded keys)
SF_KEY = os.environ["SILICONFLOW_API_KEY"]
SF_BASE = os.environ.get("SF_BASE", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.environ.get("HIPPO_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
# HippoRAG's _get_embedding_model_class hardcodes a whitelist (GritLM /
# NV-Embed-v2 / contriever / text-embedding-*).  To use SiliconFlow's
# BAAI/bge-m3 we pass a "text-embedding-..." string to pass the whitelist,
# then monkey-patch the OpenAIEmbeddingModel instance's
# .embedding_model_name to the real SiliconFlow model name after init
# (the actual API call reads .embedding_model_name at encode time).
EMB_MODEL_REAL = os.environ.get("HIPPO_EMB_MODEL", "BAAI/bge-m3")
EMB_MODEL_STUB = "text-embedding-siliconflow"  # passes whitelist
EMB_BASE = os.environ.get("HIPPO_EMB_BASE", SF_BASE)

# Pipe SiliconFlow as the OpenAI-compatible endpoint
os.environ["OPENAI_API_KEY"] = SF_KEY
os.environ["OPENAI_BASE_URL"] = SF_BASE

from hipporag import HippoRAG  # noqa: E402


class HippoRAGAdapter:
    name = "hipporag"

    def __init__(self):
        self._save_dir = None
        self._hrag = None
        # Track facts we inscribed so we can delete by content
        self._inscribed: list[str] = []

    def reset(self) -> None:
        if self._save_dir and os.path.exists(self._save_dir):
            shutil.rmtree(self._save_dir, ignore_errors=True)
        self._save_dir = tempfile.mkdtemp(prefix="hippo_fe_")
        # Pass stub name to satisfy HippoRAG's whitelist check; the OpenAI
        # client uses base_url=SiliconFlow.
        # OpenAI Python client requires api_key to be set even for compat
        # endpoints; SiliconFlow expects the same key as OPENAI_API_KEY.
        os.environ["OPENAI_API_KEY"] = SF_KEY
        self._hrag = HippoRAG(
            save_dir=self._save_dir,
            llm_model_name=LLM_MODEL,
            llm_base_url=SF_BASE,
            embedding_model_name=EMB_MODEL_STUB,
            embedding_base_url=EMB_BASE,
        )
        # Monkey-patch the real model name (read at encode time).
        try:
            emb = self._hrag.embedding_model
            emb.embedding_model_name = EMB_MODEL_REAL
            emb.embedding_config.embedding_model_name = EMB_MODEL_REAL
        except AttributeError:
            pass
        self._inscribed = []

    def inscribe(self, text: str) -> str:
        self._hrag.index([text])
        self._inscribed.append(text)
        return text

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        results = self._hrag.retrieve([query], num_to_retrieve=k)
        if isinstance(results, tuple):
            results = results[0]
        out = []
        for r in results:
            # QuerySolution has docs / passages attribute
            passages = getattr(r, "passages", None) or getattr(r, "docs", None) or []
            out.extend(p for p in passages if isinstance(p, str))
        return out[:k]

    def supersede(self, old_query: str, new_text: str) -> None:
        # Retrieve top-1 candidate matching old_q from inscribed docs.
        hits = self.recall_texts(old_query, k=1)
        if hits:
            try:
                self._hrag.delete(hits[:1])
            except Exception:
                pass
            for h in hits[:1]:
                if h in self._inscribed:
                    self._inscribed.remove(h)
        self.inscribe(new_text)

    def release(self, query: str) -> int:
        raise NotImplementedError("HippoRAG: no soft-delete primitive")

    def purge(self, query: str) -> int:
        hits = self.recall_texts(query, k=10)
        if not hits:
            return 0
        try:
            self._hrag.delete(hits)
        except Exception:
            return 0
        for h in hits:
            if h in self._inscribed:
                self._inscribed.remove(h)
        return len(hits)


def run_case(adapter, case: dict) -> tuple[object, str | None]:
    adapter.reset()
    for f in case["setup_facts"]:
        adapter.inscribe(f)
    for mut in case["mutations"]:
        op = mut[0]
        if op == "supersede":
            _, oldq, newt = mut
            adapter.supersede(oldq, newt)
        elif op == "purge":
            _, q = mut
            try:
                adapter.purge(q)
            except NotImplementedError:
                return None, "N/A"
        elif op == "release":
            return None, "N/A"
    retrieved = adapter.recall_texts(case["final_query"], k=10)
    blob = " ".join(retrieved).lower()
    must_ok = all(s.lower() in blob for s in case["must_contain"])
    not_ok = all(s.lower() not in blob for s in case["must_not_contain"])
    return (must_ok and not_ok), None


def main():
    cases_path = os.environ.get("CASES_JSON", "/cases.json")
    out_path = os.environ.get("OUT_JSON", "/out.json")
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    print(f"loaded {len(cases)} cases", flush=True)

    adapter = HippoRAGAdapter()
    results = []
    by_cat = defaultdict(lambda: {"pass": 0, "fail": 0, "na": 0})
    t_start = time.time()
    for i, case in enumerate(cases, 1):
        cat = case["category"]
        try:
            passed, err = run_case(adapter, case)
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        results.append({"id": case["id"], "category": cat,
                        "passed": passed, "error": err})
        if passed is True:
            by_cat[cat]["pass"] += 1
        elif passed is False:
            by_cat[cat]["fail"] += 1
        else:
            by_cat[cat]["na"] += 1
        if i % 5 == 0 or i == len(cases):
            tot_p = sum(c["pass"] for c in by_cat.values())
            tot_e = sum(c["pass"] + c["fail"] for c in by_cat.values())
            rate = tot_p / tot_e * 100 if tot_e else 0
            elapsed = time.time() - t_start
            print(f"  [{i:3}/{len(cases)}] {case['id'][:35]:35s} "
                  f"{cat:25s} {'P' if passed is True else ('F' if passed is False else 'N/A')}  "
                  f"agg={tot_p}/{tot_e} ({rate:.1f}%)  t={elapsed:.0f}s",
                  flush=True)
        # Incremental save for resume safety
        if i % 25 == 0:
            Path(out_path).write_text(json.dumps({
                "n": i, "results": results, "by_cat": dict(by_cat),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(out_path).write_text(json.dumps({
        "system": "hipporag", "n_total": len(results),
        "results": results, "by_cat": dict(by_cat),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tot_p = sum(c["pass"] for c in by_cat.values())
    tot_e = sum(c["pass"] + c["fail"] for c in by_cat.values())
    tot_na = sum(c["na"] for c in by_cat.values())
    print(f"\nDONE  pass={tot_p}/{tot_e}  N/A={tot_na}", flush=True)


if __name__ == "__main__":
    main()
