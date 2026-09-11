"""Head-to-head evaluation: run our 4 ForgetEval adapters on the
Memora benchmark (ACL 2026 Findings, arXiv:2604.20006).

Translates Memora's conversational sessions into our 6-method
Adapter Protocol:
  - session.operation = "add"     -> inscribe(share_memory turn)
  - session.operation = "update"  -> supersede(prior item, new item)
  - session.operation = "delete"  -> purge(item identifier)
  - session.operation = None      -> inscribe(share_memory turn)

For each evaluation_question, runs recall(question, k=10) and scores
via deterministic substring match against:
  - memory_evidence values     (must_contain)
  - forgetting_evidence.forgotten_items values (must_not_contain)

Output:
    data/memora_results.json   per-system per-question results

Usage:
    py scripts/eval_on_memora.py [--persona academic_researcher]
                                 [--timescale weekly]
                                 [--systems lethe mem0 langmem mempalace]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
MEMORA_DIR = Path(os.environ.get("MEMORA_DIR", Path(__file__).resolve().parent.parent.parent / "memora-bench"))
sys.path.insert(0, str(LETHE_REPO))

OUT = Path(__file__).resolve().parent.parent / "data" / "memora_results.json"


def load_persona_data(persona: str, timescale: str) -> tuple[list, dict]:
    """Load all sessions + evaluation questions for one persona/timescale.
    Returns (sessions, questions_doc)."""
    base = MEMORA_DIR / "data" / timescale / persona
    conv_dir = base / "conversations"
    sessions = []
    for f in sorted(os.listdir(conv_dir)):
        sessions.append(json.loads((conv_dir / f).read_text(encoding="utf-8")))
    qfile = base / f"evaluation_questions_{persona}.json"
    questions_doc = json.loads(qfile.read_text(encoding="utf-8"))
    return sessions, questions_doc


def share_memory_text(session: dict) -> str:
    """Concatenate all share_memory turns from a session into one fact-row."""
    parts = []
    for t in session["conversation"]:
        if t.get("share_memory"):
            parts.append(f"[{t.get('speaker','?')}] {t['message']}")
    return " ".join(parts).strip()


def operation_identifier(session: dict) -> str:
    """Extract a string identifier from operation_details for
    supersede/purge mutations."""
    od = session.get("operation_details") or {}
    item = od.get("item")
    if not isinstance(item, dict):
        item = {}
    category = od.get("category", "")
    pieces = [category]
    for key in ("expense_type", "item", "name", "title", "description"):
        if key in item and item[key]:
            pieces.append(str(item[key]))
    return " ".join(p for p in pieces if p).strip()


def apply_session_to_adapter(adapter, session: dict) -> None:
    """Translate one Memora session into Adapter Protocol calls."""
    text = share_memory_text(session)
    if not text:
        return
    op = session.get("operation") or "None"
    op_id = operation_identifier(session)
    if op == "add" or op == "None":
        adapter.inscribe(text)
    elif op == "update":
        try:
            adapter.supersede(op_id, text)
        except (NotImplementedError, AttributeError):
            adapter.inscribe(text)
    elif op == "delete":
        try:
            adapter.purge(op_id)
        except (NotImplementedError, AttributeError):
            pass  # N/A scoring
    else:
        adapter.inscribe(text)


def collect_memory_strings(question: dict) -> list[str]:
    """Pull all 'memory_evidence' literal values out of a question."""
    out = []
    me = question.get("memory_evidence") or {}
    def _walk(x):
        if isinstance(x, dict):
            if "value" in x and isinstance(x["value"], (str, int, float)):
                out.append(str(x["value"]))
            elif "item" in x and isinstance(x["item"], str):
                out.append(x["item"])
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
    _walk(me)
    return [s for s in out if isinstance(s, str) and len(s) > 1]


def collect_forgotten_strings(question: dict) -> list[str]:
    """Pull all 'forgetting_evidence.forgotten_items' literal values."""
    fe = question.get("forgetting_evidence") or {}
    items = fe.get("forgotten_items") or []
    out = []
    for it in items:
        if isinstance(it, dict) and "value" in it:
            out.append(str(it["value"]))
        elif isinstance(it, str):
            out.append(it)
    return out


def score_question(adapter, question: dict, k: int = 10) -> dict:
    """Run recall and score on a single evaluation question."""
    q = question["question"]
    must = collect_memory_strings(question)
    must_not = collect_forgotten_strings(question)
    try:
        results = adapter.recall_texts(q, k=k)
    except Exception as e:
        return {
            "question_id": question.get("question_id", "?"),
            "error": f"{type(e).__name__}: {e}",
            "passed": False,
        }
    blob = " ".join(results).lower()

    must_hit = sum(1 for s in must if s.lower() in blob)
    must_miss = [s for s in must if s.lower() not in blob]
    must_not_hit = [s for s in must_not if s.lower() in blob]

    # Pass = all must_contain present AND no must_not_contain present.
    passed = (not must_miss) and (not must_not_hit)
    return {
        "question_id": question.get("question_id", "?"),
        "n_must": len(must),
        "n_must_hit": must_hit,
        "n_must_miss": len(must_miss),
        "n_must_not": len(must_not),
        "n_must_not_hit": len(must_not_hit),
        "passed": passed,
        "memory_recall_rate": must_hit / max(len(must), 1),
        "forgetting_rate": 1 - (len(must_not_hit) / max(len(must_not), 1)),
    }


def build_adapter(name: str):
    from bench.forgeteval.adapter import (
        LetheAdapter, MemPalaceAdapter, Mem0Adapter, LangGraphAdapter,
        AMemAdapter,
    )
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    def embedder(t):
        return list(next(iter(model.embed([t]))))
    if name == "lethe":
        return LetheAdapter(embedder=embedder, vector_dim=384, llm=None)
    if name == "mempalace":
        return MemPalaceAdapter()  # MemPalace takes no constructor args
    if name == "mem0":
        return Mem0Adapter()
    if name in ("langmem", "langgraph"):
        return LangGraphAdapter(embedder=embedder, vector_dim=384)
    if name == "amem":
        # A-MEM via SiliconFlow DeepSeek-V3; evolution disabled.
        import os
        os.environ.setdefault(
            "OPENAI_API_KEY",
            "")
        os.environ.setdefault(
            "OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

        class AMemSF(AMemAdapter):
            def reset(self):
                self.ms = self._System(
                    model_name="all-MiniLM-L6-v2",
                    llm_backend="openai",
                    llm_model="deepseek-ai/DeepSeek-V3",
                    evo_threshold=100000,
                    api_key=os.environ["OPENAI_API_KEY"],
                )
                try:
                    for mid in list(self.ms.memories.keys()):
                        self.ms.delete(mid)
                except Exception:
                    pass
        return AMemSF(llm_backend="openai",
                      embedder_model="all-MiniLM-L6-v2")
    raise ValueError(f"Unknown system: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="academic_researcher")
    ap.add_argument("--timescale", default="weekly",
                    choices=["weekly", "monthly", "quarterly"])
    ap.add_argument("--systems", nargs="+",
                    default=["lethe", "mempalace", "langmem"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading Memora data for persona={args.persona}, "
          f"timescale={args.timescale}")
    sessions, qdoc = load_persona_data(args.persona, args.timescale)
    print(f"  {len(sessions)} sessions")
    n_q = sum(len(qdoc["questions"].get(t, []))
              for t in ("remembering", "reasoning", "recommending"))
    print(f"  {n_q} evaluation questions "
          f"(remembering/reasoning/recommending)")

    all_runs = []
    for sys_name in args.systems:
        print(f"\n### {sys_name} ###")
        try:
            adapter = build_adapter(sys_name)
        except Exception as e:
            print(f"  FAILED to build adapter: {type(e).__name__}: {e}")
            continue
        adapter.reset()
        # Ingest all sessions in chronological order.
        t0 = time.perf_counter()
        for s in sessions:
            try:
                apply_session_to_adapter(adapter, s)
            except Exception as e:
                print(f"  ingest error on session {s.get('session_id')}: "
                      f"{type(e).__name__}: {e}")
        ingest_t = time.perf_counter() - t0
        print(f"  ingested {len(sessions)} sessions in {ingest_t:.1f}s")

        # Score each evaluation question.
        per_q = []
        t1 = time.perf_counter()
        for task in ("remembering", "reasoning", "recommending"):
            for q in qdoc["questions"].get(task, []):
                r = score_question(adapter, q, k=args.k)
                r["task"] = task
                per_q.append(r)
        eval_t = time.perf_counter() - t1
        n_pass = sum(1 for r in per_q if r.get("passed"))
        print(f"  evaluated {len(per_q)} questions in {eval_t:.1f}s")
        print(f"  PASS: {n_pass}/{len(per_q)} "
              f"({n_pass/max(len(per_q),1)*100:.1f}%)")
        # Aggregate by task.
        by_task = {}
        for r in per_q:
            t = r["task"]
            by_task.setdefault(t, {"total": 0, "pass": 0})
            by_task[t]["total"] += 1
            if r.get("passed"):
                by_task[t]["pass"] += 1
        for t, d in by_task.items():
            print(f"    {t:>13}: {d['pass']}/{d['total']} "
                  f"({d['pass']/max(d['total'],1)*100:.1f}%)")
        all_runs.append({
            "system": sys_name,
            "persona": args.persona, "timescale": args.timescale,
            "n_sessions": len(sessions),
            "ingest_seconds": ingest_t,
            "eval_seconds": eval_t,
            "by_task": by_task,
            "overall_pass": n_pass,
            "overall_total": len(per_q),
            "per_question": per_q,
        })

    OUT.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

