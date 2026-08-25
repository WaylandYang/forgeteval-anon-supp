"""Cross-LLM hook ablation via OpenRouter (frontier-model extension of App. K).

Wires the LetheAdapter mutation-time hook to an OpenRouter model (OpenAI-
compatible) using the SAME JSON prompts as the SiliconFlow runs, so the
comparison is apples-to-apples: only the LLM behind the supersede/purge
planner changes.  Scored by the same deterministic substring scorer on the
full 385-case ForgetEval-Adv suite.

Reads OPENROUTER_API_KEY + OPENROUTER_MODEL from the environment ONLY
(never hard-coded). Tracks real token usage to report actual $ cost.

  OPENROUTER_MODEL=openai/gpt-5.5 python scripts/run_openrouter_hook.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import threading
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
OUT = LETHE_REPO / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.adapter import LetheAdapter  # noqa: E402

# provider-generic: defaults to OpenRouter, override via LLM_* env vars
KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("LLM_MODEL") or os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.5")
BASE = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
# Completion budget.  512 was enough for the mutation-time prompts
# (205-231 tokens observed) but truncated the write-time canonicalisation
# prompt on non-Latin identifiers, which returned empty or half a JSON
# object -- silently, since an empty response falls through to the
# deterministic path.  That biased the placement ablation in favour of
# its own conclusion, so the budget is now wide enough for every prompt
# in either arm and is reported with the run.
MAX_TOK = int(os.environ.get("LLM_MAX_TOKENS", "3000"))
# per-1M-token pricing (USD); auto-filled from OpenRouter, else 0 (tokens still reported)
PRICE = {"in": None, "out": None}



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
def fetch_price(model):
    if "openrouter" not in BASE:
        return  # non-OpenRouter providers: report tokens, skip $ estimate
    import urllib.request
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20) as r:
            data = json.load(r)
        for m in data.get("data", []):
            if m.get("id") == model:
                p = m.get("pricing", {})
                PRICE["in"] = float(p.get("prompt", 0)) * 1_000_000
                PRICE["out"] = float(p.get("completion", 0)) * 1_000_000
    except Exception as e:
        print(f"(price fetch failed: {e})")


def make_llm():
    import openai
    # An un-timed request is worse than a failed one: the SDK default is
    # 600 s, so one hung connection removes a worker for ten minutes and
    # the run just looks slow.  Observed for real -- four workers stalled
    # together and the checkpoint sat still for nine minutes.  Bound the
    # attempt, let the SDK retry, and count what still fails: a call that
    # returns "" falls through to the deterministic path, which quietly
    # understates whatever arm it happens to hit.
    client = openai.OpenAI(api_key=KEY, base_url=BASE,
                           timeout=120.0, max_retries=3)
    cache: dict[str, str] = {}
    usage = {"calls": 0, "cache_hits": 0, "in_tok": 0, "out_tok": 0, "errors": 0}

    def llm(prompt: str) -> str:
        if prompt in cache:
            usage["cache_hits"] += 1
            return cache[prompt]
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=MAX_TOK, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            if resp.usage:
                usage["in_tok"] += resp.usage.prompt_tokens
                usage["out_tok"] += resp.usage.completion_tokens
            usage["calls"] += 1
            cache[prompt] = text
            return text
        except Exception as e:
            usage["errors"] += 1
            print(f"  [llm error] {type(e).__name__}: {str(e)[:120]}")
            return ""  # fall through to deterministic
    return llm, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full 385")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel cases; each worker gets its own adapter "
                         "and embedder. Default 1 keeps runs bit-comparable "
                         "with the sequential results already reported.")
    ap.add_argument("--tag", default="", help="suffix for repeat runs; keeps "
                    "each repetition's checkpoint and output separate")
    ap.add_argument("--suite", choices=["v051", "v06", "v07", "external"],
                    default="v051",
                    help="v051 = as originally shipped; v06 = with the "
                         "canonicalization repair, where passing also "
                         "requires sparing a sibling entity; v07 = v06 plus "
                         "cross-lingual purge queries reduced to one surface "
                         "form")
    ap.add_argument("--probed", action="store_true",
                    help="also score each case under the probe-based scorer "
                         "(must_not_contain checked against direct probes, "
                         "not only the final query). Costs no extra LLM "
                         "calls -- retrieval is off the hook path -- so both "
                         "verdicts come out of one run.")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic baseline: same adapter, no hook")
    ap.add_argument("--adapter", choices=["lethe", "langgraph", "routed", "routed-langgraph",
                             "mem0", "mem0-infer", "mempalace", "amem",
                             "inscribe", "inscribe+mutation",
                             "inscribe-aware",
                             "merge-inscribe", "merge-inscribe+mutation"],
                    default="lethe",
                    help="backend the hook is wired into. langgraph runs "
                         "LangGraph's InMemoryStore under the identical "
                         "contract, which is what makes the "
                         "architecture-agnosticism claim testable on the "
                         "repaired suite rather than only the shipped one")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    # --no-llm is the deterministic path: no hook is constructed and no
    # request leaves the machine, which is what lets the four primary
    # adapters run on a CPU with no key. Demanding one here anyway made
    # that claim false as shipped -- the check ran before args.no_llm was
    # ever consulted, so the deterministic run exited on a missing key it
    # would never have used.
    if not KEY and not args.no_llm:
        sys.exit("set OPENROUTER_API_KEY (or pass --no-llm for the "
                 "deterministic adapters, which need no key)")

    from fastembed import TextEmbedding
    if args.no_llm:
        # Same shape as make_llm()'s counters: the summary block below
        # reads usage[...] unconditionally, and a deterministic run
        # legitimately has zeros rather than no counters at all.
        llm = None
        usage = {"calls": 0, "cache_hits": 0, "in_tok": 0,
                 "out_tok": 0, "errors": 0}
    else:
        fetch_price(MODEL)
        print(f"model: {MODEL}   price in/out per 1M: "
              f"${PRICE['in']}/${PRICE['out']}")
        llm, usage = make_llm()

    # Cases are independent -- GeneratedCase.run() calls adapter.reset()
    # first -- but only if each worker owns its adapter; a shared one would
    # have its store reset out from under a concurrent case.  onnxruntime
    # sessions are likewise not safely shared, so the embedder is
    # per-thread too.
    _local = threading.local()
    from bench.forgeteval.router import EscalationRouter
    ROUTER = EscalationRouter()   # shared: aggregate escalation stats
    PLACEMENT_STATS = {'inscribe_calls': 0, 'annotated': 0, 'failed': 0}

    def get_adapter():
        if not hasattr(_local, "adapter"):
            emb = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            embedder = lambda t: list(next(iter(emb.embed([t]))))
            hook = None if args.no_llm else llm
            if args.adapter.startswith(("inscribe", "merge-inscribe")):
                from bench.forgeteval import placement as _pl
                if args.adapter == "inscribe-aware":
                    from bench.forgeteval.placement_aware import (
                        AnnotationAwareLetheAdapter as Cls,
                    )
                else:
                    Cls = (_pl.MergeInscribeLetheAdapter
                           if args.adapter.startswith("merge-") else
                           _pl.InscribeLLMLetheAdapter)
                _local.adapter = Cls(
                    embedder=embedder, vector_dim=384, llm=llm,
                    mutation_llm=(llm if args.adapter.endswith("+mutation")
                                  else None),
                    stats=PLACEMENT_STATS)
            elif args.adapter == "amem":
                from bench.forgeteval.adapter import AMemAdapter
                _local.adapter = AMemAdapter(
                    llm_backend="openai", llm_model=MODEL)
            elif args.adapter in ("mem0", "mem0-infer"):
                from bench.forgeteval.adapter import Mem0Adapter
                infer = args.adapter == "mem0-infer"
                _local.adapter = Mem0Adapter(
                    infer=infer,
                    llm_model=MODEL if infer else None,
                    llm_base_url=BASE if infer else None,
                    llm_api_key=KEY if infer else None)
            elif args.adapter == "mempalace":
                from bench.forgeteval.adapter import MemPalaceAdapter
                _local.adapter = MemPalaceAdapter()
            elif args.adapter == "routed-langgraph":
                from bench.forgeteval.router import RoutedLangGraphAdapter
                _local.adapter = RoutedLangGraphAdapter(
                    embedder=embedder, vector_dim=384, llm=hook,
                    router=ROUTER)
            elif args.adapter == "routed":
                from bench.forgeteval.router import RoutedLetheAdapter
                _local.adapter = RoutedLetheAdapter(
                    embedder=embedder, vector_dim=384, llm=hook,
                    router=ROUTER)
            elif args.adapter == "langgraph":
                from bench.forgeteval.adapter import (
                    LangGraphAdapter, LangGraphLLMAdapter,
                )
                _local.adapter = (
                    LangGraphAdapter(embedder=embedder, vector_dim=384)
                    if hook is None else
                    LangGraphLLMAdapter(embedder=embedder, vector_dim=384,
                                        llm=hook))
            else:
                _local.adapter = LetheAdapter(embedder=embedder,
                                              vector_dim=384, llm=hook)
        return _local.adapter

    if args.suite == "external":
        from bench.forgeteval.external import load_external_cases
        SUITE = load_external_cases()
    elif args.suite == "v06":
        from bench.forgeteval.repaired import REPAIRED_TESTS as SUITE
    elif args.suite == "v07":
        from scripts.repair_cross_lingual_queries import build_suite
        SUITE, _n = build_suite()
    else:
        SUITE = ADVERSARIAL_TESTS
    cases = SUITE[: args.limit] if args.limit else SUITE
    slug0 = (MODEL.replace("/", "_").replace(".", "")
             + ("" if args.adapter == "lethe" else "_" + args.adapter)
             + ("_nollm" if args.no_llm else "")
             + ("" if args.suite == "v051" else "_" + args.suite)
             + ("_probed" if args.probed else "") + args.tag)
    ckpt = OUT / f"openrouter_hook_{slug0}_ckpt.jsonl"
    # resume: load already-done case verdicts
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r["ok"]
        print(f"resume: {len(done)} cases already done in checkpoint")
    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    t0 = time.perf_counter()
    counters = {"passed": 0, "case_errors": 0, "finished": 0}
    io_lock = threading.Lock()
    fout = ckpt.open("a", encoding="utf-8")

    def evaluate(c):
        if c.id in done:
            return c.id, done[c.id], False
        try:
            if args.probed:
                from bench.forgeteval.scoring import run_scored
                ok = run_scored(c, get_adapter(), probed=True)
            else:
                ok = c.run(get_adapter())
        except Exception as e:
            ok = False
            with io_lock:
                counters["case_errors"] += 1
                print(f"  [case error] {c.id}: {type(e).__name__}: {str(e)[:90]}")
        return c.id, ok, True

    def record(case_id, ok, fresh):
        with io_lock:
            if fresh:
                fout.write(json.dumps({"id": case_id, "ok": ok}) + "\n")
                fout.flush()
            cat = _cat(case_id)
            by_cat[cat]["total"] += 1
            if ok is None:
                # Primitive absent: counted in the strict denominator,
                # excluded from the evaluable one.
                by_cat[cat]["na"] = by_cat[cat].get("na", 0) + 1
            elif ok:
                by_cat[cat]["pass"] += 1
                counters["passed"] += 1
            counters["finished"] += 1
            if counters["finished"] % 25 == 0:
                print(f"  {counters['finished']}/{len(cases)}  "
                      f"pass={counters['passed']}  calls={usage['calls']} "
                      f"err={usage['errors']} "
                      f"case_err={counters['case_errors']}", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for case_id, ok, fresh in pool.map(evaluate, cases):
                record(case_id, ok, fresh)
    else:
        for c in cases:
            record(*evaluate(c))

    fout.close()
    passed_n = counters["passed"]
    case_errors = counters["case_errors"]
    wall = time.perf_counter() - t0

    cost = (usage["in_tok"] * (PRICE["in"] or 0)
            + usage["out_tok"] * (PRICE["out"] or 0)) / 1_000_000
    total = len(cases)
    # A --no-llm run reaches no model, so naming one in the banner would
    # label a deterministic result with a system it never used.
    who = "deterministic (no model)" if args.no_llm else f"{MODEL} hook"
    print("")
    print(f"=== {who} on {total} cases ===")
    print(f"OVERALL {passed_n}/{total} = {passed_n/total:.1%}")
    print(f"calls={usage['calls']} cache_hits={usage['cache_hits']} "
          f"errors={usage['errors']}")
    if PLACEMENT_STATS["inscribe_calls"]:
        p = PLACEMENT_STATS
        print(f"inscribe-time: {p['annotated']}/{p['inscribe_calls']} "
              f"annotated, {p['failed']} unparseable")
    print(f"tokens in={usage['in_tok']} out={usage['out_tok']}  "
          f"COST=${cost:.3f}  wall={wall:.0f}s")
    print(f"\n{'category':<26}{'pass/tot':>10} rate")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"{cat:<26}{d['pass']:>4}/{d['total']:<4} {d['pass']/max(d['total'],1):.0%}")

    # Refuse to write a result the LLM did not actually produce.
    #
    # Three cross-model runs came back at exactly 63.6% -- the
    # deterministic baseline -- from three different vendors, because the
    # model id was wrong and all 447 calls 400'd. A failed call returns ""
    # and falls through to the deterministic path, so the run completes,
    # exits zero, and reports a number that looks like "this model gives
    # no lift". A Mem0 run in the same batch reported 0/385 with calls=0
    # after every case hit a Qdrant lock. Neither is a measurement, and
    # neither is distinguishable from one by looking at the score.
    # Adapters that construct their own client do not go through the
    # wrapper these counters watch, so zero calls is expected for them.
    OWNS_ITS_LLM = {"amem", "mem0-infer"}
    if not args.no_llm and args.adapter not in OWNS_ITS_LLM:
        if usage["calls"] == 0:
            raise SystemExit(
                f"REFUSING TO WRITE: LLM was requested but never called "
                f"({usage['errors']} errors, {case_errors} case errors). "
                f"This is a broken run, not a result.")
        bad = usage["errors"] / max(usage["errors"] + usage["calls"], 1)
        if bad > 0.02:
            raise SystemExit(
                f"REFUSING TO WRITE: {usage['errors']} of "
                f"{usage['errors'] + usage['calls']} LLM calls failed "
                f"({bad:.0%}). Failed calls fall through to the "
                f"deterministic path, so this score understates the "
                f"system by an unknown amount.")
    if case_errors > 0.02 * total:
        raise SystemExit(
            f"REFUSING TO WRITE: {case_errors}/{total} cases raised. "
            f"A raising case scores as a failure.")

    slug = slug0
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = {"model": MODEL, "suite": "adversarial-385", "limit": args.limit,
           "overall_pass": passed_n, "overall_total": total,
           "overall_rate": passed_n / total, "by_category": dict(by_cat),
           "usage": usage, "cost_usd": cost, "wall_seconds": wall,
           # How many write-time calls actually produced an
           # annotation.  A silently truncated response falls
           # through to the deterministic path, so an arm can look
           # like "placement buys nothing" when it is really "the
           # annotation was never written".  Recorded per run.
           "placement_stats": dict(PLACEMENT_STATS)}
    (OUT / f"openrouter_hook_{slug}{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote data/openrouter_hook_{slug}{suffix}.json")


if __name__ == "__main__":
    main()
