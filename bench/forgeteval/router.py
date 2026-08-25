"""Escalation routing: pay for the LLM only where the store cannot cope.

The placement result says a mutation-time LLM recovers capability the
deterministic control plane lacks. It does not say every mutation needs
one. On the repaired suite the hook changes the verdict on 86 of 385
cases and five of the ten attack categories never need it at all, so an
oracle that knew which mutations to escalate would buy the whole lift --
331/385, one case *better* than always calling, because it also skips the
one case the hook breaks -- at 22.3% of the calls.

This module is the deterministic approximation of that oracle. Every
signal is computed from the mutation request and the candidate rows the
store already retrieved, so routing costs one extra pass over at most 20
short strings and never an extra model call.

Three triggers, each tied to a failure the deterministic plane is known
to have:

  cross_script      the request or a candidate carries a non-ASCII
                    letter, so NFKC-equality cannot decide identity
                    (the cross_lingual_identifier failure)

  variant_family    the rows NFKC-equality would act on are a strict
                    subset of the rows sharing the request's rarest
                    content token -- surface variants of one entity that
                    the deterministic path is about to half-delete
                    (the identifier_obfuscation failure)

  compound_row      the row that would be deleted carries two clauses
                    and the request names one, so whole-row deletion
                    would take a fact the user did not ask to forget
                    (the compound_fact failure)

Anything else runs the deterministic path untouched. The router is
deliberately allowed to over-trigger: a false escalation costs a fraction
of a cent, a missed one costs a case.
"""
from __future__ import annotations

import re
import unicodedata

from .adapter import LetheAdapter

CLAUSE = re.compile(r";|,\s+and\s+|\s+and\s+|\s+&\s+")
STOP = frozenset("""
the a an of in on at to for from by with and or is are was were be been
this that these those it its as into over under about user customer
please remove delete forget purge drop erase account record entry
""".split())


def has_nonascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s)


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).lower().split())


def _content_tokens(s: str) -> list[str]:
    toks = re.findall(r"[\w@.\-/]+", _norm(s))
    return [t for t in toks if len(t) > 2 and t not in STOP]


def _norm_hard(s: str) -> str:
    """Alphanumerics only. Collapses the punctuation and spacing variants
    that carry most of identifier_obfuscation -- ``j.doe@corp.com`` and
    ``J DOE (corp.com)`` land on the same string, while NFKC-equality
    still sees two different rows."""
    return re.sub(r"[^0-9a-z]+", "", _norm(s))


def _identifiers(s: str) -> list[str]:
    """Hard-normalised forms of the identifier-shaped terms in a request.

    Identifier-shaped means: carries a digit, an ``@``, or an internal
    separator between alphanumerics -- e-mails, phone numbers, card and
    account numbers, handles, paths. Plain prose words are excluded, so
    the trigger does not fire on every sentence that shares a noun with a
    stored row. Short forms are dropped: a 3-character normalised token
    matches too much to be evidence of anything.
    """
    out = []
    for tok in re.findall(r"[\w@.+\-/:]+", s):
        if not (any(ch.isdigit() for ch in tok) or "@" in tok
                or re.search(r"[a-zA-Z][._\-/][a-zA-Z]", tok)):
            continue
        h = _norm_hard(tok)
        if len(h) >= 5:
            out.append(h)
    return out


def _trigrams(s: str) -> set[str]:
    s = _norm_hard(s)
    return {s[i:i + 3] for i in range(max(len(s) - 2, 0))} or {s}


def _similar(a: str, b: str, thr: float = 0.4) -> bool:
    """Character-trigram Jaccard. Catches the surface variants that even
    hard normalisation misses (spelled-out separators, reordered parts)
    without reaching for an embedding or a model."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= thr


class EscalationRouter:
    """Decides, per mutation, whether the LLM is worth calling.

    Kept separate from the adapter so the same policy can be dropped into
    any backend implementing the Protocol, and so the trigger counts can
    be audited independently of the verdicts.
    """

    def __init__(self):
        self.stats = {"seen": 0, "escalated": 0,
                      "cross_script": 0, "variant_family": 0,
                      "compound_row": 0}

    def _fire(self, name: str) -> bool:
        self.stats[name] += 1
        self.stats["escalated"] += 1
        return True

    def should_escalate(self, query: str, hit_texts: list[str],
                        op: str = "purge") -> bool:
        self.stats["seen"] += 1
        if not hit_texts:
            return False

        if has_nonascii(query) or any(has_nonascii(t) for t in hit_texts):
            return self._fire("cross_script")

        # What the deterministic plane will actually delete: rows whose
        # WHOLE TEXT is NFKC-equal to the top hit. Two rows naming the
        # same entity in different sentences are never equal, so it
        # deletes one and leaves the other -- the failure to detect.
        head = hit_texts[0]
        equal = sum(1 for t in hit_texts if _norm(t) == _norm(head))

        # How many rows carry the entity the request names, compared at
        # identifier level rather than row level. Hard normalisation puts
        # ALICE@ACME.IO, alice@acme.io and +1-415-555-0123 on the same
        # footing as their variants without touching a neighbouring
        # identifier that differs in its digits.
        for tok in _identifiers(query):
            carrying = sum(1 for t in hit_texts if tok in _norm_hard(t))
            if carrying > equal:
                return self._fire("variant_family")

        # A supersede replaces the matched row wholesale. If that row
        # carries more than one clause, the replacement necessarily drops
        # whichever clause the request did not restate -- and the request
        # is often a description ("user city of residence") that shares no
        # words with the row at all, so a lexical overlap test would look
        # for evidence that cannot exist. Multi-clause target plus a
        # replace-whole-row primitive is sufficient reason to escalate.
        if op == "supersede":
            clauses = [c for c in CLAUSE.split(head) if c and c.strip()]
            if len(clauses) > 1:
                return self._fire("compound_row")
        return False


class RoutedLetheAdapter(LetheAdapter):
    """LetheAdapter that consults an EscalationRouter before each mutation.

    Implemented by hiding ``self.llm`` for the duration of a mutation the
    router declines, so the deterministic path taken is byte-for-byte the
    one the no-LLM configuration takes -- the comparison stays honest.
    """

    name = "lethe_routed"

    def __init__(self, embedder, vector_dim: int = 384, *, llm=None,
                 router: EscalationRouter | None = None):
        super().__init__(embedder=embedder, vector_dim=vector_dim, llm=llm)
        self.router = router or EscalationRouter()
        self._full_llm = llm

    def _decide(self, query: str, hit_texts: list[str], op: str) -> None:
        self.llm = self._full_llm if self.router.should_escalate(
            query, hit_texts, op) else None

    def _texts(self, query: str, k: int, **kw) -> list[str]:
        return [h.memory.text for h in self.lethe.recall(query, k=k, **kw)]

    def supersede(self, old_query: str, new_text: str) -> None:
        self._decide(old_query, self._texts(old_query, 1, hybrid=False),
                     "supersede")
        try:
            return super().supersede(old_query, new_text)
        finally:
            self.llm = self._full_llm

    def release(self, query: str) -> int:
        self._decide(query, self._texts(query, 20, hybrid=True), "release")
        try:
            return super().release(query)
        finally:
            self.llm = self._full_llm

    def purge(self, query: str) -> int:
        self._decide(query, self._texts(query, 20, lexical=True), "purge")
        try:
            return super().purge(query)
        finally:
            self.llm = self._full_llm


class RoutedLangGraphAdapter:
    """The same escalation policy over LangGraph's ``InMemoryStore``.

    Written as a wrapper rather than a subclass because the routing
    decision only needs the Protocol surface: the candidate rows come from
    ``recall_texts``, which every backend already implements. That the
    policy transfers unchanged is the point -- it is a property of the
    control plane, not of one store's internals.
    """

    name = "langmem_routed"

    def __init__(self, embedder, vector_dim: int = 384, *, llm=None,
                 router: EscalationRouter | None = None):
        from .adapter import LangGraphLLMAdapter
        self.inner = LangGraphLLMAdapter(embedder=embedder,
                                         vector_dim=vector_dim, llm=llm)
        self.router = router or EscalationRouter()
        self._full_llm = llm

    def reset(self):
        return self.inner.reset()

    def inscribe(self, text):
        return self.inner.inscribe(text)

    def recall_texts(self, query, k=5):
        return self.inner.recall_texts(query, k)

    def _decide(self, query: str, op: str) -> None:
        hits = self.inner.recall_texts(query, 20)
        self.inner.llm = self._full_llm if self.router.should_escalate(
            query, hits, op) else None

    def supersede(self, old_query: str, new_text: str) -> None:
        self._decide(old_query, "supersede")
        try:
            return self.inner.supersede(old_query, new_text)
        finally:
            self.inner.llm = self._full_llm

    def release(self, query: str) -> int:
        self._decide(query, "release")
        try:
            return self.inner.release(query)
        finally:
            self.inner.llm = self._full_llm

    def purge(self, query: str) -> int:
        self._decide(query, "purge")
        try:
            return self.inner.purge(query)
        finally:
            self.inner.llm = self._full_llm


__all__ = ["EscalationRouter", "RoutedLetheAdapter",
           "RoutedLangGraphAdapter", "has_nonascii"]
