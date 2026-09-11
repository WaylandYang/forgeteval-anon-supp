"""Generate additional ForgetEval-Adv cases via DeepSeek-V3 on SiliconFlow.

For each attack category, this script:
  1. Loads existing cases as 3 exemplars
  2. Asks DeepSeek-V3 to generate N more cases following the same format
     and the same attack-class semantics, with DIFFERENT entities so
     the bench gains diversity, not duplication
  3. Parses returned JSON, validates each case structurally
  4. Writes the new cases to an output Python module appended onto
     adversarial.py

The model is non-thinking (DeepSeek-V3 standard, not R1) so prompts
elicit one JSON object per call with no <think> wrapper.

Usage:
    py scripts/generate_adversarial_cases.py --target 100 [--dry-run]

By default targets 100 cases per category total (so for a category
currently at 8, generates 92 new cases in batches of 10).

Cost estimate: ~$0.30 for ~1000 cases at DeepSeek-V3 pricing.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from textwrap import dedent

# Force UTF-8 stdout so multilingual content (CJK / Arabic / Hebrew /
# Cyrillic) in judge / generator replies doesn't choke Windows GBK.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"
MODEL = os.environ.get("LETHE_LLM_MODEL", "deepseek-ai/DeepSeek-V3")

# Independent admission oracle: a model from a DIFFERENT family than
# the one wired into Lethe+LLM, so the well-formedness check is
# architecturally decoupled from the systems under evaluation.
JUDGE_MODEL = os.environ.get(
    "LETHE_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct"
)

OUT_FILE = LETHE_REPO / "bench" / "forgeteval" / "adversarial_generated.py"


CATEGORY_PROMPT_HINTS = {
    "substring_trap": dedent("""\
        ATTACK CLASS: substring_trap
        FAMILY: any (supersession / decay / amnesia / purge / drift)
        WHAT IT PROBES: must_not_contain substring of the target row
          accidentally appears inside a distractor or related fact.
          Pass = the system correctly removes the target row and the
          surviving distractor doesn't falsely satisfy must_not_contain.
        TIPS: pick a meaningful entity (city, person, brand, code) whose
          name happens to be a common English word or phrase fragment.
          The distractor should contain that fragment IN A DIFFERENT
          SENSE (e.g. "Vienna sausage" the food vs. "Vienna" the city)."""),
    "prefix_collision": dedent("""\
        ATTACK CLASS: prefix_collision
        FAMILY: purge
        WHAT IT PROBES: two identifiers share a long common prefix.
          Purging one must NOT also delete the other.
        TIPS: pick a domain where prefix-substring confusion is real:
          emails (alice@x / alice.smith@x), api keys (sk-abc-prod-XX /
          sk-abc-dev-XX), case ids (12345 / 123456), domains (x.com /
          x.com.au), phone numbers (+1- / no country code), UUIDs.
          Always include the surviving identifier in must_contain."""),
    "paraphrase_supersession": dedent("""\
        ATTACK CLASS: paraphrase_supersession
        FAMILY: supersession, drift
        WHAT IT PROBES: the new fact is lexically distant from the
          old (low surface overlap); only semantic alignment can match.
        TIPS: pick a "before / after" where the new wording avoids the
          old's key tokens. Example: "User works at Stripe" supersedes
          to "User quit payments to join an AI safety lab" — overlap is
          just the subject "User". The query may be vague like "user
          current employer", not exact."""),
    "negation_trap": dedent("""\
        ATTACK CLASS: negation_trap
        FAMILY: supersession, decay
        WHAT IT PROBES: a NEGATED fact ("does NOT work at X") must
          not be confused with the AFFIRMATIVE form.
        TIPS: store a negated fact in setup; supersede with the
          affirmative.  After supersede, the negated form must not
          remain in recall.  Use clear negation tokens: "does not",
          "doesn't", "never", "no longer", "refuses to"."""),
    "temporal_qualifier": dedent("""\
        ATTACK CLASS: temporal_qualifier
        FAMILY: supersession, drift
        WHAT IT PROBES: facts with embedded dates; supersession must
          respect the temporal window (latest wins).
        TIPS: include explicit dates in the facts. Set up multi-step
          chains spanning several years. The latest dated state
          should win recall, even if an EARLIER dated mention of the
          same entity appears in a distractor."""),
    "shared_attribute": dedent("""\
        ATTACK CLASS: shared_attribute
        FAMILY: amnesia
        WHAT IT PROBES: two entities share an attribute; forgetting
          one entity must not collapse the link of the other entity
          to that same attribute.
        TIPS: e.g. "Dana lives in Berlin" + "Eve lives in Berlin" +
          release "everything about Dana".  After release, Eve must
          still be retrievable, and Eve's Berlin link must survive."""),
    "compound_fact": dedent("""\
        ATTACK CLASS: compound_fact
        FAMILY: supersession
        WHAT IT PROBES: one row carries TWO DISTINCT-TOPIC facts
          joined by " and " (e.g. location AND employer).
          Superseding the query for one topic must preserve the
          other topic.
        TIPS: setup row should have form "User lives in X and works at
          Y." Supersede with a new fact about ONLY the city.  The
          employer (Y) must survive in recall.  Both clauses must be
          GENUINELY DIFFERENT TOPICS — not restatements of one
          (city + employer = good; "does not work at X and has never
          interviewed there" = co-dependent, NOT a compound)."""),
    "identifier_obfuscation": dedent("""\
        ATTACK CLASS: identifier_obfuscation
        FAMILY: purge
        WHAT IT PROBES: same identifier stored in different SURFACE
          FORMS (case, whitespace, separators, quoting, prefix '@',
          format variants like phone country codes / UUID hyphens).
          Purge by one form should reach the others.
        TIPS: case (ALICE@X.com vs alice@x.com), trailing whitespace,
          quoted local part ("user"@x.com), separator stripping
          (123-45-6789 vs 123456789), prefix @ (@handle vs handle),
          UUID with/without dashes."""),
    "cross_lingual_identifier": dedent("""\
        ATTACK CLASS: cross_lingual_identifier
        FAMILY: purge
        WHAT IT PROBES: same entity stored under different SCRIPTS
          or romanizations.  GDPR-relevant for multilingual deployments.
        TIPS: 张伟 vs Zhang Wei, 田中太郎 vs Tanaka Taro, الزائر vs the
          visitor, José vs Jose, Großmann vs Grossmann.  Both forms
          should appear in setup; the purge should target the unified
          identity ("Zhang Wei aka 张伟").  Cover a variety of script
          families: Chinese, Japanese, Korean, Arabic, Cyrillic,
          Greek, Hebrew, accent-stripping, ß/ss, emoji-containing."""),
    "recursive_supersession": dedent("""\
        ATTACK CLASS: recursive_supersession
        FAMILY: drift
        WHAT IT PROBES: a supersede chain where the LATEST state
          matches an earlier-superseded state ("back to X" after
          having moved to Y).
        TIPS: 3-step chains.  Step 1: initial state (e.g. Chrome user).
          Step 2: supersede to alternative (e.g. Brave).
          Step 3: supersede to the ORIGINAL state again (e.g. back to
          Chrome).  Final query should yield the latest state; the
          intermediate alternative must NOT appear."""),
}


PROMPT_TEMPLATE = dedent("""\
You generate test cases for ForgetEval-Adv, a benchmark for memory-system
forgetting evaluation.  Each case probes a specific attack class against
the memory system's purge / release / supersede primitives.

{hint}

EXISTING EXEMPLARS (study format and intent, do not duplicate):

{exemplars}

Your task: produce {n} NEW cases following the EXACT JSON shape below.
Each case must:
  - Use DIFFERENT entities (people, companies, identifiers) from the
    exemplars.  Diversity is the point.
  - Be a clean test of the attack class as described above.
  - Have unambiguous must_contain / must_not_contain checks.
  - Be self-consistent: the setup_facts inscribed must make the
    must_contain / must_not_contain check decidable after the
    mutations are applied.

Reply with EXACTLY ONE JSON ARRAY of {n} case objects and NO other
text:

[
  {{
    "id":               "adv_{cat}_<integer>",
    "family":           "<one of: supersession | decay | amnesia | purge | drift>",
    "setup_facts":      ["<fact 1>", "<fact 2>", ...],
    "mutations":        [["<op>", "<arg1>", "<arg2 if any>"], ...],
    "final_query":      "<the question>",
    "must_contain":     ["<substring required in top-k recall>", ...],
    "must_not_contain": ["<substring forbidden in top-k recall>", ...]
  }},
  ...
]

Notes on the mutations field:
  - ["supersede", "old_query", "new_text"]  — 3-tuple
  - ["release", "query"]                    — 2-tuple
  - ["purge", "query"]                      — 2-tuple

Start id numbering from {start_id}.
""")


def load_existing_cases() -> dict:
    """Load existing cases grouped by category."""
    from bench.forgeteval.adversarial import (
        ADVERSARIAL_TESTS, ATTACK_CATEGORIES, case_to_attack_category,
    )
    by_cat = {cat: [] for cat in ATTACK_CATEGORIES}
    for case in ADVERSARIAL_TESTS:
        cat = case_to_attack_category(case.id)
        if cat in by_cat:
            by_cat[cat].append(case)
    return by_cat


def case_to_json(case) -> dict:
    """Convert a GeneratedCase to JSON-serializable dict."""
    return {
        "id": case.id,
        "family": case.family,
        "setup_facts": list(case.setup_facts),
        "mutations": [list(m) for m in case.mutations],
        "final_query": case.final_query,
        "must_contain": list(case.must_contain),
        "must_not_contain": list(case.must_not_contain),
    }


def make_llm():
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)

    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return llm


def make_judge_llm():
    """Judge LLM for case admission.  A model from a different family
    than the one Lethe+LLM uses, so the well-formedness check is
    architecturally decoupled from any system under evaluation."""
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)

    def judge(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=0.0,          # deterministic judge
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return judge


JUDGE_PROMPT = dedent("""\
You are an independent judge for a memory-system benchmark.  Given a
single test case, decide whether it is WELL-FORMED.

WHAT WELL-FORMED MEANS

Trace through the case as if you were the agent's memory:

1. START STATE: the setup_facts are written verbatim into memory,
   one row per fact.

2. APPLY MUTATIONS in order:
   - ("supersede", q, new):  the fact best-matching `q` is REPLACED
                             by the row `new`.  All other facts stay.
   - ("release",   q):       facts best-matching `q` are MARKED
                             unreachable (soft-deleted).  Other facts
                             stay reachable.
   - ("purge",     q):       facts matching identifier `q` are
                             HARD-DELETED (any surface form).

3. END STATE: the rows in memory still reachable for retrieval.

A case is WELL-FORMED iff, reasoning over the END STATE:

  (a) For each string s in must_contain: there EXISTS a reachable
      row whose text literally contains s.  (Not "could be inferred
      from"; literally contains the substring s.)

  (b) For each string t in must_not_contain: NO reachable row's
      text contains t as a substring.

  (c) must_contain and must_not_contain do not contradict each other
      and the final_query is a sensible question to ask of the end
      state.

You are NOT judging whether any particular memory system can solve
the case.  You are judging whether the END STATE, computed by hand,
satisfies the must_contain / must_not_contain assertions.

COMMON FAILURE MODES (REJECT these)

  - A must_contain string only ever appeared in a fact that was
    SUPERSEDED away (e.g. setup says "User speaks Spanish and
    Mandarin", supersede on "user languages" replaces this whole
    row, but must_contain still asks for "Spanish").
  - A must_not_contain string is a substring of a fact that was NOT
    targeted by any mutation (the self-substring-trap, partially
    caught by an earlier structural filter).
  - SUBSTRING-COLLISION BUG (very common in prefix_collision /
    identifier categories): a must_not_contain string is a
    SUBSTRING of any must_contain string OR of any surviving fact's
    text.  Example: must_not_contain "TXN-12345" with surviving row
    "Transaction TXN-123456 ..." — purge could be perfect, but the
    surviving row literally contains the 14 characters "TXN-12345".
    Under literal-substring scoring this case can never pass; REJECT.
  - The final_query is so vague it can't be answered from the end
    state, or so leading it pre-supposes the answer.

CASE:
{case_json}

Reason carefully, then output EXACTLY ONE JSON object and no other
text.  Show your trace in `reason`.

{{"well_formed": true | false, "reason": "<one or two sentences tracing the end state>"}}
""")


def extract_json_array(text: str) -> list:
    """Extract the first top-level JSON array from a model response."""
    # Find the first [ and the matching ]
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError("no JSON array in response")
    return json.loads(m.group(0))


def validate_case(c: dict, expected_cat: str) -> tuple[bool, str]:
    """Structural check on a generated case."""
    if not isinstance(c, dict):
        return False, "not a dict"
    required = {"id", "family", "setup_facts", "mutations", "final_query",
                "must_contain", "must_not_contain"}
    if not required.issubset(c.keys()):
        return False, f"missing keys: {required - set(c.keys())}"
    if not c["id"].startswith(f"adv_{expected_cat}_"):
        return False, f"id doesn't start with adv_{expected_cat}_"
    if c["family"] not in {"supersession", "decay", "amnesia", "purge", "drift"}:
        return False, f"bad family: {c['family']}"
    if not c["setup_facts"] or not isinstance(c["setup_facts"], list):
        return False, "setup_facts must be non-empty list"
    if not c["mutations"] or not isinstance(c["mutations"], list):
        return False, "mutations must be non-empty list"
    for m in c["mutations"]:
        if not isinstance(m, list) or len(m) < 2 or m[0] not in {"supersede", "release", "purge"}:
            return False, f"bad mutation: {m}"
        if m[0] == "supersede" and len(m) != 3:
            return False, "supersede needs 3 args"
        if m[0] in ("release", "purge") and len(m) != 2:
            return False, f"{m[0]} needs 2 args"
    if not isinstance(c["must_contain"], list):
        return False, "must_contain not list"
    if not isinstance(c["must_not_contain"], list):
        return False, "must_not_contain not list"
    if not c["must_contain"] and not c["must_not_contain"]:
        return False, "both must lists empty"
    # Self-substring-trap check: if any must_not_contain substring
    # appears in a setup_fact that isn't being targeted by the
    # mutation, the case is self-bugged (the LLM generated something
    # like "must_not 'TKT-100'" but kept a fact containing 'TKT-1001'
    # which contains 'TKT-100' as a substring).
    targeted_facts: set[int] = set()
    for m in c["mutations"]:
        target_text = m[1].lower() if len(m) > 1 else ""
        for i, fact in enumerate(c["setup_facts"]):
            # Heuristic: a fact is "targeted" if it shares >=3-char
            # contiguous token with the target_text.
            if any(tok in fact.lower() for tok in target_text.split()
                   if len(tok) >= 4):
                targeted_facts.add(i)
    for nott in c["must_not_contain"]:
        nott_lo = nott.lower()
        for i, fact in enumerate(c["setup_facts"]):
            if i in targeted_facts:
                continue        # this fact will be removed by mutation
            if nott_lo in fact.lower():
                return False, (f"self-substring-trap: must_not "
                               f"{nott!r} appears in a non-targeted "
                               f"setup_fact: {fact!r}")
    return True, ""


_lethe_ref_adapter = None       # cached
_lethe_llm_ref_adapter = None   # cached


def _make_lethe_adapter(use_llm: bool):
    """Build a LetheAdapter as the reference 'oracle' for validation.
    When use_llm=True, wires DeepSeek-V3 via SiliconFlow."""
    global _lethe_ref_adapter, _lethe_llm_ref_adapter
    if use_llm and _lethe_llm_ref_adapter is not None:
        return _lethe_llm_ref_adapter
    if not use_llm and _lethe_ref_adapter is not None:
        return _lethe_ref_adapter
    from bench.forgeteval.adapter import LetheAdapter
    from fastembed import TextEmbedding
    embed_model_name = os.environ.get(
        "LETHE_EMBEDDER",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    model = TextEmbedding(embed_model_name)
    def embedder(t):
        return list(next(iter(model.embed([t]))))
    if use_llm:
        llm = make_llm()
        adapter = LetheAdapter(embedder=embedder, vector_dim=384, llm=llm)
        _lethe_llm_ref_adapter = adapter
    else:
        adapter = LetheAdapter(embedder=embedder, vector_dim=384, llm=None)
        _lethe_ref_adapter = adapter
    return adapter


_judge_llm_cached = None


def _get_judge():
    global _judge_llm_cached
    if _judge_llm_cached is None:
        _judge_llm_cached = make_judge_llm()
    return _judge_llm_cached


def extract_json_object(text: str) -> dict:
    """Extract the LAST valid top-level JSON object from a model
    response, scanning bracket-balanced spans.  The judge may emit
    reasoning before the JSON; we want the final answer."""
    depth = 0
    start = -1
    candidates: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1
    for span in reversed(candidates):
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON object in response")


def judge_admission(case_dict: dict) -> tuple[bool, str]:
    """Independent LLM judge: is this case well-formed?  Returns
    (admitted, reason).  Uses a model from a different family than
    Lethe+LLM to keep the oracle architecturally decoupled."""
    judge = _get_judge()
    case_for_judge = {k: case_dict[k] for k in (
        "setup_facts", "mutations", "final_query",
        "must_contain", "must_not_contain",
    )}
    prompt = JUDGE_PROMPT.format(
        case_json=json.dumps(case_for_judge, ensure_ascii=False, indent=2)
    )
    try:
        raw = judge(prompt)
        verdict = extract_json_object(raw)
    except Exception as e:
        return False, f"judge parse error: {type(e).__name__}: {e}"
    if not isinstance(verdict.get("well_formed"), bool):
        return False, "judge did not return well_formed bool"
    return bool(verdict["well_formed"]), str(verdict.get("reason", ""))


def label_case(case_dict: dict) -> tuple[str, dict]:
    """Run the admitted case on Lethe and Lethe+LLM, return a label
    capturing the analytic role of the case:

      easy           : both Lethe and Lethe+LLM pass
      llm_lift       : Lethe-base fails, Lethe+LLM passes
      llm_regression : Lethe-base passes, Lethe+LLM fails (suspect)
      unsolvable     : both fail (well-formed per judge, but no system
                       in our comparison can solve it)
    """
    from bench.forgeteval.generate import GeneratedCase
    gc = GeneratedCase(
        id=case_dict["id"],
        family=case_dict["family"],
        setup_facts=list(case_dict["setup_facts"]),
        mutations=[tuple(m) for m in case_dict["mutations"]],
        final_query=case_dict["final_query"],
        must_contain=list(case_dict["must_contain"]),
        must_not_contain=list(case_dict["must_not_contain"]),
    )
    try:
        base_pass = gc.run(_make_lethe_adapter(use_llm=False))
    except Exception as e:
        return "runtime_error", {"stage": "base", "error": str(e)}
    try:
        llm_pass = gc.run(_make_lethe_adapter(use_llm=True))
    except Exception as e:
        return "runtime_error", {"stage": "llm", "error": str(e)}
    if base_pass and llm_pass:
        return "easy", {"base": True, "llm": True}
    if not base_pass and llm_pass:
        return "llm_lift", {"base": False, "llm": True}
    if base_pass and not llm_pass:
        return "llm_regression", {"base": True, "llm": False}
    return "unsolvable", {"base": False, "llm": False}


# Labels we keep in the bench (well-formed and analytically useful).
# "llm_regression" cases are admitted by judge but the LLM hook
# regressed — keep but flag for spot-review.
KEEP_LABELS = {"easy", "llm_lift", "unsolvable", "llm_regression"}


def generate_for_category(cat: str, n_new: int, existing: list,
                           batch_size: int = 10, *, dry_run: bool) -> list:
    """Generate `n_new` cases for `cat` in batches.

    Pipeline per case:
      Stage 1 — structural check (validate_case).
      Stage 2 — independent LLM-judge admission (judge_admission).
      Stage 3 — dual-system labeling on Lethe / Lethe+LLM
                (label_case): easy / llm_lift / llm_regression / unsolvable.

    Cases that pass stages 1+2 are admitted to the bench with their
    stage-3 label attached, regardless of whether deterministic Lethe
    can solve them.  This decouples oracle (well-formedness) from
    system performance (the thing the bench measures).
    """
    llm = make_llm() if not dry_run else None
    accepted: list[dict] = []
    start_id = len(existing) + 1
    remaining = n_new
    seen_ids: set[str] = {c.id for c in existing}
    label_tally: dict[str, int] = {}

    exemplars_json = "\n".join(
        json.dumps(case_to_json(c), ensure_ascii=False, indent=2)
        for c in existing[:3]
    )
    hint = CATEGORY_PROMPT_HINTS[cat]

    batch_no = 0
    while remaining > 0:
        batch_no += 1
        batch_n = min(batch_size, remaining)
        prompt = PROMPT_TEMPLATE.format(
            hint=hint, exemplars=exemplars_json, n=batch_n,
            cat=cat, start_id=start_id + len(accepted),
        )
        if dry_run:
            print(f"  [{cat} batch {batch_no}] DRY RUN — would request {batch_n}")
            remaining -= batch_n
            continue
        try:
            raw = llm(prompt)
            cases = extract_json_array(raw)
        except Exception as e:
            print(f"  [{cat} batch {batch_no}] LLM/parse error: {e}", flush=True)
            continue
        kept = 0
        for c in cases:
            # Stage 1 — structural check.
            ok, why = validate_case(c, cat)
            if not ok:
                print(f"  [{cat} batch {batch_no}] reject (structural): {why}",
                      flush=True)
                continue
            if c["id"] in seen_ids:
                continue
            # Stage 2 — independent LLM-judge admission.
            admitted, why = judge_admission(c)
            if not admitted:
                print(f"  [{cat} batch {batch_no}] reject (judge): {why}",
                      flush=True)
                continue
            # Stage 3 — dual-system labeling (analytic, not gating).
            label, detail = label_case(c)
            if label == "runtime_error":
                print(f"  [{cat} batch {batch_no}] reject (runtime): "
                      f"{detail}", flush=True)
                continue
            if label not in KEEP_LABELS:
                print(f"  [{cat} batch {batch_no}] reject (label): {label}",
                      flush=True)
                continue
            c["_label"] = label
            seen_ids.add(c["id"])
            accepted.append(c)
            label_tally[label] = label_tally.get(label, 0) + 1
            kept += 1
        remaining -= kept
        print(f"  [{cat}] batch {batch_no}: kept {kept}/{len(cases)} "
              f"(total so far: {len(accepted)}/{n_new}; labels: {label_tally})",
              flush=True)
        if kept == 0:
            print(f"  [{cat}] batch yielded 0 — stopping early",
                  flush=True)
            break

    print(f"  [{cat}] label distribution: {label_tally}", flush=True)
    return accepted


def write_module(by_cat_new: dict, out_path: Path) -> None:
    """Write the generated cases as a Python module that the runner
    can import alongside the hand-crafted v0.4 cases."""
    lines = [
        "\"\"\"Generated by scripts/generate_adversarial_cases.py.",
        "Do not hand-edit; regenerate from the script if cases need changes.",
        "\"\"\"",
        "from bench.forgeteval.generate import GeneratedCase",
        "",
    ]
    for cat, cases in by_cat_new.items():
        if not cases:
            continue
        var = f"ADV_{cat.upper()}_GENERATED"
        lines.append(f"{var} = [")
        for c in cases:
            lines.append("    GeneratedCase(")
            lines.append(f"        id={c['id']!r},")
            lines.append(f"        family={c['family']!r},")
            lines.append(f"        setup_facts={c['setup_facts']!r},")
            lines.append(f"        mutations={[tuple(m) for m in c['mutations']]!r},")
            lines.append(f"        final_query={c['final_query']!r},")
            lines.append(f"        must_contain={c['must_contain']!r},")
            lines.append(f"        must_not_contain={c['must_not_contain']!r},")
            lines.append("    ),")
        lines.append("]")
        lines.append("")

    lines.extend([
        "ADVERSARIAL_GENERATED: dict[str, list[GeneratedCase]] = {",
    ])
    for cat in by_cat_new:
        if by_cat_new[cat]:
            lines.append(f"    {cat!r}: ADV_{cat.upper()}_GENERATED,")
    lines.append("}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Sidecar JSON: maps case_id -> label (easy / llm_lift /
    # llm_regression / unsolvable).  Loaded by stats scripts to
    # partition results without touching the bench dataclass.
    labels: dict[str, str] = {}
    for cat, cases in by_cat_new.items():
        for c in cases:
            label = c.get("_label", "unknown")
            labels[c["id"]] = label
    labels_path = out_path.with_name("adversarial_generated_labels.json")
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(labels)} labels to {labels_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100,
                    help="Target total cases per category (default 100).")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Only generate for these categories.")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing = load_existing_cases()
    cats = args.categories or list(existing.keys())

    print(f"Target: {args.target} cases per category")
    print(f"Generator model: {MODEL}")
    print(f"Judge model:     {JUDGE_MODEL}")
    print(f"Categories: {cats}\n")

    by_cat_new: dict[str, list] = {}
    for cat in cats:
        cur = existing[cat]
        n_new = max(0, args.target - len(cur))
        if n_new == 0:
            print(f"  [{cat}] already at {len(cur)}; skip")
            by_cat_new[cat] = []
            continue
        print(f"\n=== {cat}: have {len(cur)}, need {n_new} more ===",
              flush=True)
        t0 = time.perf_counter()
        new_cases = generate_for_category(
            cat, n_new, cur, batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        wall = time.perf_counter() - t0
        print(f"  [{cat}] DONE: generated {len(new_cases)} new cases "
              f"({wall:.1f}s)", flush=True)
        by_cat_new[cat] = new_cases
        # Incremental save after each category so a downstream crash
        # doesn't lose hours of LLM-generated content.
        if not args.dry_run:
            write_module(by_cat_new, OUT_FILE)
            print(f"  [{cat}] incremental save: "
                  f"{sum(len(cs) for cs in by_cat_new.values())} cases "
                  f"persisted to {OUT_FILE.name}", flush=True)

    if args.dry_run:
        print("\nDRY RUN — nothing written")
        return

    write_module(by_cat_new, OUT_FILE)
    total_new = sum(len(cs) for cs in by_cat_new.values())
    print(f"\nWrote {total_new} new cases to {OUT_FILE}")


if __name__ == "__main__":
    main()

