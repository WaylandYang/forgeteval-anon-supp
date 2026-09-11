"""Probe-based scoring: forgetting means unreachable, not merely unranked.

ForgetEval-Adv scores ``must_not_contain`` against the top-k returned for a
single ``final_query``. That makes the criterion satisfiable by a store that
never deleted anything, as long as its retriever happens not to rank the
offending row for that one query. Measured on v0.6 with a five-line
Unicode-normalising store, 34 of its 242 passes (14%) are of exactly this
kind -- the forbidden text is still in the store. They concentrate in
identifier_obfuscation (19) and cross_lingual_identifier (9); six of the ten
categories have none, because their case design forces the target into the
answer set.

The repair is at the scoring layer rather than in the cases. A forbidden
string is checked against the union of the answer sets for

    [final_query] + [one probe derived from each forbidden string]

so a system is credited with forgetting only if the content cannot be
retrieved *when asked for directly*. This keeps the scorer deterministic,
black-box, and inside the six-method Adapter Protocol -- no store
introspection, no LLM judge -- while changing what passing means from "did
not surface it here" to "cannot surface it".

The distinction is the same one \\citet{memleak} draw for multimodal stores:
deletion that survives only the queries you thought to ask is not deletion.

Both scorers are exposed, because every number that moves under the change
has to be reportable under each:

    run_scored(case, adapter, probed=False)   # as shipped through v0.6
    run_scored(case, adapter, probed=True)    # probe-based
"""
from __future__ import annotations

TOP_K = 5


def _probe_queries(case) -> list[str]:
    """The final query plus one direct probe per forbidden string.

    Using the forbidden string itself is the strongest black-box probe
    available and needs no authoring: if a store still holds the text, a
    query *made of that text* is the query most likely to surface it.
    """
    seen, out = set(), []
    for q in [case.final_query, *case.must_not_contain]:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def run_scored(case, adapter, probed: bool = True, k: int = TOP_K) -> bool:
    """Replay a case and score it. Mirrors GeneratedCase.run when probed=False.

    Returns True on pass, False on failure, and None when the store
    cannot express one of the case's mutations at all -- the N/A of
    the Adapter Protocol. A caller that ignores the third value and
    treats it as False is computing the strict score, which is a
    legitimate reading but should be a deliberate one.
    """
    adapter.reset()
    for fact in case.setup_facts:
        adapter.inscribe(fact)

    for m in case.mutations:
        op = m[0]
        try:
            if op == "supersede":
                adapter.supersede(m[1], m[2])
            elif op == "release":
                adapter.release(m[1])
            elif op == "purge":
                adapter.purge(m[1])
            else:
                raise ValueError(f"unknown mutation {op!r}")
        except NotImplementedError:
            # The store cannot express this operation through any
            # composition of its API. That is a coverage fact, not a
            # forgetting failure, so it is reported as neither: callers
            # that want a strict score count None as a failure, and
            # callers that want an evaluable score drop it.
            return None

    answer = " ".join(adapter.recall_texts(case.final_query, k))
    for need in case.must_contain:
        if need.casefold() not in answer.casefold():
            return False

    if probed:
        reachable = " ".join(
            " ".join(adapter.recall_texts(q, k)) for q in _probe_queries(case))
    else:
        reachable = answer

    for forbidden in case.must_not_contain:
        if forbidden.casefold() in reachable.casefold():
            return False
    return True


__all__ = ["run_scored", "TOP_K"]
