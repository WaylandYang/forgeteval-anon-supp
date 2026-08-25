"""Run ForgetEval-Adv on OpenMemory (CaviraOSS, self-hosted via Docker
on 127.0.0.1:8284).  OpenMemory has explicit add / query / delete-by-id
primitives, so supersede / purge / release are composable.

Mapping (6-method Adapter Protocol):
  inscribe(text)  -> POST /memory/add body={"content": text}
                     returns {"id": "<uuid>"}
  recall(query,k) -> POST /memory/query body={"q": query, "limit": k}
                     returns [{id, content, score, ...}]
  supersede(old_q, new_text) -> query top-1 by old_q, DELETE that id,
                                 add new_text
  purge(query)               -> query top-k by query, DELETE each id
  release(query)             -> NotImplementedError (OpenMemory has no
                                soft-delete; we expose this as N/A
                                rather than conflate semantics)

Output: data/adversarial_results_openmemory.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")

OM_URL = os.environ.get("OPENMEMORY_URL", "http://127.0.0.1:8284")
OM_API_KEY = os.environ.get("OPENMEMORY_API_KEY", "fe-secret-key")

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


class OpenMemoryAdapter:
    name = "openmemory"

    def __init__(self, base_url: str = OM_URL, api_key: str = OM_API_KEY):
        self.base = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(timeout=60.0, headers=headers)
        self._ids: set[str] = set()

    def _post(self, path: str, body: dict) -> dict:
        r = self.client.post(f"{self.base}{path}", json=body)
        r.raise_for_status()
        return r.json() if r.text else {}

    def _delete(self, path: str) -> None:
        try:
            self.client.delete(f"{self.base}{path}")
        except Exception:
            pass

    def reset(self) -> None:
        # Wipe our tracked IDs; OpenMemory has /memory/all GET but no
        # bulk-delete, so we delete each tracked id.  In practice the
        # 385-case loop only retains the ids it created on this adapter
        # instance, so this is sufficient for per-case isolation.
        for mid in list(self._ids):
            self._delete(f"/memory/{mid}")
        self._ids.clear()

    def inscribe(self, text: str) -> str:
        data = self._post("/memory/add", {"content": text})
        mid = (data.get("id") or data.get("memory_id")
               or data.get("data", {}).get("id") or "")
        if mid:
            self._ids.add(mid)
        return mid

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        data = self._post("/memory/query", {"query": query, "limit": k})
        results = data.get("matches", []) if isinstance(data, dict) else []
        texts = []
        for r in results:
            if isinstance(r, dict):
                t = (r.get("content") or r.get("text")
                     or r.get("memory") or r.get("chunk") or "")
                if t:
                    texts.append(t)
        return texts

    def _search_raw(self, query: str, k: int = 5) -> list[dict]:
        data = self._post("/memory/query", {"query": query, "limit": k})
        return data.get("matches", []) if isinstance(data, dict) else []

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self._search_raw(old_query, k=1)
        if hits and isinstance(hits[0], dict):
            mid = hits[0].get("id")
            if mid:
                self._delete(f"/memory/{mid}")
                self._ids.discard(mid)
        if new_text:
            self.inscribe(new_text)

    def release(self, query: str) -> int:
        # OpenMemory: no soft-delete primitive; N/A
        raise NotImplementedError("OpenMemory: no release primitive")

    def purge(self, query: str) -> int:
        hits = self._search_raw(query, k=10)
        n = 0
        for h in hits:
            if isinstance(h, dict):
                mid = h.get("id")
                if mid:
                    self._delete(f"/memory/{mid}")
                    self._ids.discard(mid)
                    n += 1
        return n


def main():
    cases, _fe_tag, _fe_probed = _forgeteval_suite()
    sample = int(os.environ.get("SAMPLE_PER_CAT", "0"))
    if sample > 0:
        by_cat: dict[str, list] = {}
        for c in cases:
            by_cat.setdefault(_cat(c.id), []).append(c)
        cases = [c for lst in by_cat.values() for c in lst[:sample]]
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        cases = cases[:limit]
    print(f"Running {len(cases)} adversarial cases on OpenMemory @ {OM_URL}",
          flush=True)

    # Smoke-test connectivity
    try:
        r = httpx.get(f"{OM_URL}/health", timeout=5)
        print(f"  /health → {r.status_code}", flush=True)
    except Exception as e:
        print(f"  /health probe failed: {type(e).__name__}: {e}", flush=True)
        # Try root anyway

    adapter = OpenMemoryAdapter()
    results = []
    by_cat: dict[str, dict[str, int]] = {}
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        cat = _cat(case.id)
        by_cat.setdefault(cat, {"pass": 0, "fail": 0, "na": 0})
        try:
            try:
                passed = _forgeteval_run(case, adapter, _fe_probed)
                applied = True
            except NotImplementedError:
                passed = None
                applied = False

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
            verdict = ("PASS" if passed is True
                       else ("FAIL" if passed is False else "N/A"))
            if i % 10 == 0 or i == len(cases):
                print(f"  {i:3d}/{len(cases):3d} {case.id[:36]:36s} "
                      f"{cat:23s} {verdict}  agg={tot_p}/{tot_e} "
                      f"({rate:.1f}%)  t={elapsed:.0f}s", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  [case {case.id}] error: {type(e).__name__}: {e}",
                  flush=True)
            by_cat[cat]["fail"] += 1
            results.append({"case_id": case.id, "category": cat,
                            "passed": False,
                            "error": f"{type(e).__name__}: {e}"})

    summary_path = DATA / ("adversarial_summary_openmemory" + _suite_suffix(_fe_tag) + ".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"system": "openmemory", "by_category": by_cat,
                   "n_total": len(results)}, f, indent=2)
    results_path = DATA / ("adversarial_results_openmemory" + _suite_suffix(_fe_tag) + ".json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"system": "openmemory", "results": results}, f, indent=2)

    print("\n=== OpenMemory aggregate ===", flush=True)
    tot_p = tot_f = tot_na = 0
    for cat, d in sorted(by_cat.items()):
        n_eval = d["pass"] + d["fail"]
        rate = d["pass"] / n_eval * 100 if n_eval else float("nan")
        na_note = f" (N/A {d['na']})" if d["na"] else ""
        print(f"  {cat:30s}  {d['pass']:3d}/{n_eval:<3d} "
              f"({rate:5.1f}%){na_note}", flush=True)
        tot_p += d["pass"]
        tot_f += d["fail"]
        tot_na += d["na"]
    print(f"  OVERALL  {tot_p}/{tot_p + tot_f}  N/A {tot_na}", flush=True)


if __name__ == "__main__":
    main()

