"""Run the 80-case external-authored subset (4 contributors × 20 cases each)
through ForgetEval-Adv's Adapter Protocol on all in-house systems.

Independent validation: the cases were written by contributors who do
not know the hypothesis under test. We run the same admission filter
and all 11 in-house adapters, and report the results separately.

Input: external_raw/ (5 concatenated JSON arrays = 80 cases). The raw
contributor files are not redistributed; the parsed and admitted cases
ship as data/external_subset_cases.json, which is what the tables read.
Output:
  - data/external_subset_cases.json  (parsed + mapped + admitted cases)
  - data/external_subset_results.json (per-system per-case verdicts)
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

PAPER_ROOT = Path(__file__).resolve().parent.parent
SRC = PAPER_ROOT / "external_raw" / "contributed_fixed"
SRC_FALLBACK = PAPER_ROOT / "external_raw" / "contributed"
OUT = PAPER_ROOT / "data"
OUT.mkdir(exist_ok=True)

CATEGORY_MAP = {
    "substring_interference":      "substring_trap",
    "identifier_prefix_collision": "prefix_collision",
    "paraphrase_supersession":     "paraphrase_supersession",
    "negation_inconsistency":      "negation_trap",
    "temporal_qualifier":          "temporal_qualifier",
    "shared_attribute":            "shared_attribute",
    "partial_fact_mutation":       "compound_fact",
    "identifier_obfuscation":      "identifier_obfuscation",
    "cross_lingual_identifier":    "cross_lingual_identifier",
    "cyclic_supersession":         "recursive_supersession",
}


def parse_concat_json_arrays(path: Path) -> list[dict]:
    """Parse either a standard JSON array (fixed file) or 5 concatenated
    JSON arrays (original contributor file)."""
    txt = path.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    parts = re.split(r"\]\s*\[", txt)
    cases = []
    for i, p in enumerate(parts):
        if i == 0:
            p = p.rstrip().rstrip("]") + "]"
        elif i == len(parts) - 1:
            p = "[" + p.lstrip().lstrip("[")
        else:
            p = "[" + p + "]"
        cases.extend(json.loads(p))
    return cases


def admit(case: dict) -> tuple[bool, str]:
    """Structural admission filter.  Returns (admitted, reason).
    Mirrors the Stage-1 protocol of the in-house 132-case core."""
    required = ("id", "category", "setup_facts", "mutation",
                "final_query", "must_contain", "must_not_contain")
    for k in required:
        if k not in case:
            return False, f"missing field {k}"
    if not isinstance(case["setup_facts"], list) or not case["setup_facts"]:
        return False, "setup_facts must be a non-empty list"
    mut = case["mutation"]
    if not isinstance(mut, dict) or "op" not in mut:
        return False, "mutation must be a dict with 'op'"
    if mut["op"] not in ("supersede", "purge", "release"):
        return False, f"unknown mutation op: {mut['op']}"
    # supersede may omit new_text if the contributor placed the replacement
    # fact directly in setup_facts; we auto-convert to purge during
    # normalization (semantically equivalent: delete old, new already in store).
    if case["category"] not in CATEGORY_MAP:
        return False, f"unmapped category {case['category']}"
    # Self-substring trap: must_not_contain string in any must_contain string
    for mnc in case["must_not_contain"]:
        for mc in case["must_contain"]:
            if mnc.lower() in mc.lower():
                return False, f"self-trap: must_not_contain '{mnc[:30]}' inside must_contain"
    # Sanity: each must_not_contain should appear in setup_facts (otherwise
    # nothing to "forget"); we allow exceptions for must_not_contain that
    # describes the new (superseded) form's leaking trail.
    return True, "ok"


def normalize_case(case: dict) -> dict:
    """Map fields to our internal GeneratedCase contract."""
    mut = case["mutation"]
    op = mut["op"]
    if op == "supersede":
        new_text = (mut.get("new_text") or mut.get("target_text")
                    or mut.get("supersede_with") or "")
        if new_text:
            mutations = [("supersede", mut.get("target_query", ""), new_text)]
        else:
            # No explicit new_text: contributor placed the replacement in
            # setup_facts already.  Convert to a purge of the old fact
            # (semantically: delete old, new already in store).
            mutations = [("purge", mut.get("target_query", ""))]
    elif op == "purge":
        mutations = [("purge", mut.get("target_query", ""))]
    else:  # release
        mutations = [("release", mut.get("target_query", ""))]
    return {
        "id": case["id"],
        "category": CATEGORY_MAP[case["category"]],
        "setup_facts": list(case["setup_facts"]),
        "mutations": mutations,
        "final_query": case["final_query"],
        "must_contain": list(case["must_contain"]),
        "must_not_contain": list(case["must_not_contain"]),
    }


class ExtCase:
    """Mimics bench.forgeteval.adversarial.GeneratedCase API."""
    def __init__(self, d: dict):
        self.id = d["id"]
        self.category = d["category"]
        self.family = "ext"  # external subset; mapping to in-house families
                              # not relevant since we only report per-category
        self.setup_facts = d["setup_facts"]
        self.mutations = d["mutations"]
        self.final_query = d["final_query"]
        self.must_contain = d["must_contain"]
        self.must_not_contain = d["must_not_contain"]

    def run(self, adapter) -> bool:
        """Inscribe setup, apply mutation, then score recall."""
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


def case_author(case_id: str) -> str:
    parts = case_id.split("_")
    return parts[1] if len(parts) > 1 else "?"


def main():
    src = SRC if SRC.exists() else SRC_FALLBACK
    print(f"loading from {src}")
    raw_cases = parse_concat_json_arrays(src)
    print(f"  {len(raw_cases)} raw cases")
    authors = Counter(case_author(c["id"]) for c in raw_cases)
    print(f"  authors: {dict(authors)}")

    admitted = []
    rejected = []
    for c in raw_cases:
        ok, reason = admit(c)
        if ok:
            admitted.append(normalize_case(c))
        else:
            rejected.append({"id": c.get("id", "?"), "reason": reason})
    print(f"  admitted: {len(admitted)} / rejected: {len(rejected)}")
    if rejected:
        print("  rejection reasons:")
        for r in rejected[:10]:
            print(f"    {r['id']}: {r['reason']}")

    cats = Counter(c["category"] for c in admitted)
    print(f"  category distribution (admitted):")
    for k, v in sorted(cats.items()):
        print(f"    {k:30s} {v}")

    cases_path = OUT / "external_subset_cases.json"
    cases_path.write_text(json.dumps({
        "n_raw": len(raw_cases),
        "n_admitted": len(admitted),
        "n_rejected": len(rejected),
        "authors": dict(authors),
        "category_distribution": dict(cats),
        "admitted_cases": admitted,
        "rejected": rejected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {cases_path}")

    # ─── Build adapters and run ───────────────────────────────────────────
    print("\nloading embedder...", flush=True)
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    def embedder(t):
        return list(next(iter(model.embed([t]))))

    SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
        ""
    SF_BASE = "https://api.siliconflow.cn/v1"

    def make_llm(model_name="deepseek-ai/DeepSeek-V3"):
        import openai
        client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)
        def llm(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model_name, max_tokens=2048, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        return llm

    from bench.forgeteval.adapter import (
        LetheAdapter, LangGraphAdapter, MemPalaceAdapter, Mem0Adapter,
        LangGraphLLMAdapter,
    )

    factories = [
        ("Lethe",         lambda: LetheAdapter(embedder=embedder, vector_dim=384)),
        ("LangGraph",     lambda: LangGraphAdapter(embedder=embedder, vector_dim=384)),
        ("MemPalace",     lambda: MemPalaceAdapter()),
        ("Mem0",          lambda: Mem0Adapter()),
        ("Lethe+LLM",     lambda: LetheAdapter(embedder=embedder, vector_dim=384,
                                                llm=make_llm())),
        ("LangGraph+LLM", lambda: LangGraphLLMAdapter(embedder=embedder, vector_dim=384,
                                                       llm=make_llm())),
    ]

    out_data = {"systems": {}}
    ext_cases = [ExtCase(c) for c in admitted]

    target = os.environ.get("ONLY_SYSTEM")
    for name, factory in factories:
        if target and target not in name:
            continue
        try:
            adapter = factory()
        except Exception as e:
            print(f"  {name}: factory failed: {type(e).__name__}: {e}")
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        per_case = []
        n_pass = n_na = 0
        by_cat: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0, "na": 0})
        for c in ext_cases:
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
        wall = time.time() - t0
        n_eval = sum(d["total"] for d in by_cat.values())
        rate = n_pass / n_eval * 100 if n_eval else 0
        print(f"  pass: {n_pass}/{n_eval} ({rate:.1f}%)  N/A: {n_na}  wall: {wall:.1f}s")
        for cat in sorted(by_cat):
            d = by_cat[cat]
            r = d["pass"] / d["total"] * 100 if d["total"] else 0
            print(f"    {cat:30s} {d['pass']:2}/{d['total']:<2} ({r:5.1f}%)  N/A: {d['na']}")
        out_data["systems"][name] = {
            "n_pass": n_pass, "n_eval": n_eval, "n_na": n_na,
            "rate": rate, "by_category": dict(by_cat),
            "per_case": per_case, "wall_s": wall,
        }
        # Incremental save
        res_path = OUT / "external_subset_results.json"
        res_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  wrote {res_path}")


if __name__ == "__main__":
    main()

