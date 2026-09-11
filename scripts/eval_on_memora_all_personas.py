"""Run eval_on_memora across all 10 personas (weekly) for a larger
sample size — supports the axis-flip claim in §6.5."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from eval_on_memora import (
    load_persona_data, apply_session_to_adapter,
    score_question, build_adapter,
)

PERSONAS = [
    "academic_researcher", "business_executive", "content_writer",
    "creative_designer", "financial_analyst", "management_consultant",
    "marketing_manager", "sales_manager", "software_engineer",
    "startup_founder",
]
TIMESCALE = "weekly"
SYSTEMS = ["lethe", "langgraph", "mempalace"]

OUT = Path(__file__).resolve().parent.parent / "data" / \
    "memora_xeval_all_personas.json"


def main():
    all_rows = []
    for sys_name in SYSTEMS:
        print(f"\n========== {sys_name} ==========")
        try:
            adapter = build_adapter(sys_name)
        except Exception as e:
            print(f"  build failed: {e}")
            continue
        sys_rows = []
        for persona in PERSONAS:
            adapter.reset()
            sessions, qdoc = load_persona_data(persona, TIMESCALE)
            t0 = time.perf_counter()
            for s in sessions:
                try:
                    apply_session_to_adapter(adapter, s)
                except Exception:
                    pass
            t_ingest = time.perf_counter() - t0
            per_q = []
            t1 = time.perf_counter()
            for task in ("remembering", "reasoning", "recommending"):
                for q in qdoc["questions"].get(task, []):
                    r = score_question(adapter, q, k=10)
                    r["task"] = task
                    r["persona"] = persona
                    per_q.append(r)
            t_eval = time.perf_counter() - t1
            n_pass = sum(1 for r in per_q if r.get("passed"))
            print(f"  {persona:<24} ingest={t_ingest:5.1f}s "
                  f"eval={t_eval:5.1f}s "
                  f"pass={n_pass}/{len(per_q)} "
                  f"({n_pass/max(len(per_q),1)*100:.0f}%)",
                  flush=True)
            sys_rows.extend([
                {"system": sys_name, **r} for r in per_q
            ])
        all_rows.extend(sys_rows)
        # Aggregate per system across all personas.
        n_pass = sum(1 for r in sys_rows if r.get("passed"))
        n_total = len(sys_rows)
        print(f"  {sys_name} TOTAL: {n_pass}/{n_total} "
              f"({n_pass/max(n_total,1)*100:.1f}%)")
        # By task
        by_task = {}
        for r in sys_rows:
            t = r["task"]
            by_task.setdefault(t, {"total": 0, "pass": 0})
            by_task[t]["total"] += 1
            if r.get("passed"):
                by_task[t]["pass"] += 1
        for t, d in by_task.items():
            print(f"    {t:>13}: {d['pass']}/{d['total']} "
                  f"({d['pass']/max(d['total'],1)*100:.1f}%)")

    OUT.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
