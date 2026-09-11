"""Run ForgetEval-Adv stratified 100-case sample on Graphiti.

Graphiti is the open-source successor to the deprecated Zep CE
(graphiti-core, Neo4j-backed knowledge graph with temporal edge
invalidation).  This run is the "6th system" entry to deepen
ecosystem coverage beyond A-MEM.

Setup:
  - Neo4j 5.x on bolt://127.0.0.1:7687
  - LLM: DeepSeek-V3.1-Terminus via SiliconFlow (tolerant client
    fixes JSON-schema-shape mismatches with non-OpenAI proxy)
  - Embeddings: BAAI/bge-m3 via SiliconFlow

Output: data/adversarial_summary_graphiti.json
"""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault(
    "OPENAI_API_KEY",
    "")
os.environ.setdefault(
    "OPENAI_BASE_URL",
    os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.embedder.openai import (  # noqa: E402
    OpenAIEmbedder, OpenAIEmbedderConfig,
)
from graphiti_core.nodes import EpisodeType  # noqa: E402

from graphiti_tolerant_client import (  # noqa: E402
    TolerantOpenAIGenericClient,
    CohereRerankerClient,
)

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

SF_KEY = os.environ["OPENAI_API_KEY"]
SF_BASE = os.environ["OPENAI_BASE_URL"]
MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash")


def stratified_sample(cases, n_per_cat: int = 10):
    by_cat: dict[str, list] = {}
    for c in cases:
        cat = _cat(c.id)
        by_cat.setdefault(cat, []).append(c)
    out = []
    for cat, lst in by_cat.items():
        out.extend(lst[:n_per_cat])
    return out


_INDICES_BUILT = False


class GraphitiAdapter:
    """Wraps Graphiti as a ForgetEval Adapter.

    Each test case is its own group_id so cases are isolated.
    Mutations map:
      supersede(old_q, new) → add_episode(new) with same group_id;
                              relies on Graphiti's temporal edge
                              invalidation to make old fact stale.
      release / purge       → not exposed → N/A
    Recall:
      Graphiti search returns synthesised edge facts; we concatenate
      them and apply substring scoring like other adapters.
    """

    name = "graphiti"

    def __init__(self):
        self.llm = TolerantOpenAIGenericClient(
            config=LLMConfig(api_key=SF_KEY, base_url=SF_BASE, model=MODEL))
        # Same embedder as every other system in the comparison, served
        # locally, so this row does not need a credential that is not in
        # the release and does not differ on the embedding axis either.
        self.emb = OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key=os.environ.get("EMB_API_KEY", "local"),
            base_url=os.environ.get(
                "EMB_BASE_URL", "http://127.0.0.1:8399/v1"),
            embedding_model=os.environ.get(
                "EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dim=int(os.environ.get("EMB_DIM", "384"))))
        self.rer = (CohereRerankerClient(api_key=SF_KEY, base_url=SF_BASE)
                    if SF_KEY else None)
        self.g: Graphiti | None = None
        self.gid: str = ""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def reset(self) -> None:
        if self.g is not None:
            try:
                self._run(self.g.close())
            except Exception:
                pass
        self.g = Graphiti(
            uri=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
            user="neo4j",
            password=os.environ.get("NEO4J_PASSWORD", "forgeteval2027"),
            llm_client=self.llm, embedder=self.emb, cross_encoder=self.rer,
        )
        # Graphiti's search runs against fulltext indices that have to be
        # created before first use. Without them every query raises
        # "no such fulltext schema index: edge_name_and_fact", nothing is
        # ever retrieved, and every case fails -- which reads as a real
        # score, because this system's published score is near zero
        # anyway. Build once per process.
        global _INDICES_BUILT
        if not _INDICES_BUILT:
            self._run(self.g.build_indices_and_constraints())
            _INDICES_BUILT = True
        self.gid = "fe_" + uuid.uuid4().hex[:10]

    def inscribe(self, text: str) -> str:
        ep_name = "f" + uuid.uuid4().hex[:6]
        self._run(self.g.add_episode(
            name=ep_name, episode_body=text, source=EpisodeType.text,
            source_description="forgeteval",
            reference_time=datetime.now(timezone.utc),
            group_id=self.gid,
        ))
        return ep_name

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        results = self._run(self.g.search(
            query=query, group_ids=[self.gid], num_results=k))
        out: list[str] = []
        for r in results:
            fact = getattr(r, "fact", None) or getattr(r, "name", None) or str(r)
            out.append(fact)
        return out

    def supersede(self, old_query: str, new_text: str) -> None:
        # Add new fact; Graphiti's temporal edge invalidation should
        # mark conflicting prior edges as expired automatically.
        self.inscribe(new_text)

    def release(self, query: str) -> int:
        raise NotImplementedError("Graphiti: no release primitive")

    def purge(self, query: str) -> int:
        # Graphiti's remove_episode requires episode UUID, but we don't
        # track per-fact UUID by query.  Mark as N/A for honesty.
        raise NotImplementedError(
            "Graphiti: per-query purge is not exposed; "
            "remove_episode requires a known episode UUID")


def main():
    cases, _fe_tag, _fe_probed = _forgeteval_suite()
    sample = int(os.environ.get("SAMPLE_PER_CAT", "10"))
    if sample > 0:
        cases = stratified_sample(cases, n_per_cat=sample)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        cases = cases[:limit]
    print(f"Running {len(cases)} adversarial cases on Graphiti", flush=True)

    # One adapter per worker thread: it owns an event loop and a Graphiti
    # client, neither of which is shareable. The graph itself is shared,
    # which is fine because every case scopes its writes to its own
    # group_id.
    _local = threading.local()

    def get_adapter():
        if not hasattr(_local, "adapter"):
            _local.adapter = GraphitiAdapter()
        return _local.adapter

    results = []
    by_cat: dict[str, dict[str, int]] = {}
    t_start = time.time()
    io_lock = threading.Lock()
    workers = int(os.environ.get("GRAPHITI_WORKERS", "6"))

    ckpt_path = DATA / "graphiti_v07_probed_ckpt.jsonl"
    done: dict[str, object] = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["case_id"]] = r["passed"]
        print(f"resume: {len(done)} cases already scored", flush=True)
    ckpt = ckpt_path.open("a", encoding="utf-8")
    counter = {"i": 0}

    def record(case, cat, passed, applied, fresh, err=None):
        with io_lock:
            counter["i"] += 1
            i = counter["i"]
            if fresh and err is None:
                ckpt.write(json.dumps(
                    {"case_id": case.id, "passed": passed}) + "\n")
                ckpt.flush()
            by_cat.setdefault(cat, {"pass": 0, "fail": 0, "na": 0})
            if passed is True:
                by_cat[cat]["pass"] += 1
            elif passed is False:
                by_cat[cat]["fail"] += 1
            else:
                by_cat[cat]["na"] += 1
            row = {"case_id": case.id, "category": cat,
                   "passed": passed, "applied": applied}
            if err:
                row["error"] = err
            results.append(row)
            tot_p = sum(c["pass"] for c in by_cat.values())
            tot_e = sum(c["pass"] + c["fail"] for c in by_cat.values())
            rate = tot_p / tot_e * 100 if tot_e else 0
            verdict = ("PASS" if passed is True
                       else ("FAIL" if passed is False else "N/A"))
            print(f"  {i:3d}/{len(cases):3d} {case.id[:40]:40s} {cat:25s} "
                  f"{verdict}  agg={tot_p}/{tot_e} ({rate:.1f}%)  "
                  f"t={time.time() - t_start:.0f}s", flush=True)
            if i % 25 == 0:
                with open(DATA / ("adversarial_results_graphiti_partial" + _suite_suffix(_fe_tag) + ".json"),
                          "w", encoding="utf-8") as f:
                    json.dump({"n": i, "results": results, "by_cat": by_cat},
                              f, indent=2)

    def work(case):
        cat = _cat(case.id)
        if case.id in done:
            record(case, cat, done[case.id], True, False)
            return
        try:
            try:
                passed = _forgeteval_run(case, get_adapter(), _fe_probed)
                applied = True
            except NotImplementedError:
                passed, applied = None, False
        except Exception as e:
            record(case, cat, False, False, False,
                   err=f"{type(e).__name__}: {e}")
            return
        record(case, cat, passed, applied, True)

    print(f"  ({workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, cases))
    ckpt.close()

    # Final save
    with open(DATA / ("adversarial_summary_graphiti" + _suite_suffix(_fe_tag) + ".json"), "w",
              encoding="utf-8") as f:
        json.dump({"system": "graphiti", "by_category": by_cat,
                   "model": MODEL}, f, indent=2)
    with open(DATA / ("adversarial_results_graphiti" + _suite_suffix(_fe_tag) + ".json"), "w",
              encoding="utf-8") as f:
        json.dump({"system": "graphiti", "model": MODEL,
                   "results": results}, f, indent=2)

    print("\n=== Graphiti aggregate ===", flush=True)
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

