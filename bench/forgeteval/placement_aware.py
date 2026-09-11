"""The fourth inscribe-time cell: annotation that the control plane can read.

`InscribeLLMLetheAdapter` writes canonical forms alongside each row and
leaves the deterministic control plane untouched.  That arm scores at the
deterministic level, and the obvious reading is that write-time knowledge
cannot reach the control plane.  The reading is not safe, because the
deterministic plane never consults the annotation as a *matching key*:

  * `purge` retrieves lexically -- the annotation helps here, it is part
    of the row text -- and then groups by whole-row NFKC equivalence.
    Annotating a row changes its whole-row text, so the annotation can
    only ever *hurt* that grouping.
  * `release` thresholds on embedding similarity.
  * `supersede` takes the top-1 hit.

So the arm confounds "write-time placement is powerless" with "our purge
ignores the annotation column".  This module separates them: the same
write-time annotation, and a control plane that matches against it.

Deviation from the deterministic path happens *only* when a canonical
form is implicated by the request.  When no annotation fires, every
operation falls through to the parent's code path unchanged, so the one
moving part between this adapter and the plain inscribe arm is whether
the stored canonical forms are visible to the matcher.

What this arm still does not get is the deletion request at write time.
If capability tracks *when the model runs*, this configuration should
close the canonicalization gap -- the surface-form bridge is both built
and readable -- while leaving `compound_fact` where it was, because no
amount of identifier knowledge tells the store which clause of a
multi-clause row the user asked to drop.
"""
from __future__ import annotations

import re

from .placement import InscribeLLMLetheAdapter

_CANON_RE = re.compile(r"\[canonical:\s*([^\]]*)\]\s*$")


def canonical_forms(text: str) -> list[str]:
    """The canonical forms written onto a row at inscribe time."""
    m = _CANON_RE.search(text or "")
    if not m:
        return []
    return [f.strip() for f in m.group(1).split(";") if f.strip()]


class AnnotationAwareLetheAdapter(InscribeLLMLetheAdapter):
    """Inscribe-time annotation, plus a control plane that reads it."""

    name = "lethe_inscribe_aware"

    # ── matching ────────────────────────────────────────────────────

    def _implicated(self, query_norm: str, text: str) -> str | None:
        """The canonical form on this row that the request names, if any.

        Containment in either direction: a request carries a surface form
        ("@alice_smith", "remove Tanaka's record") and the stored form is
        the bare identifier, so neither string is reliably a prefix of the
        other.  The bridge between surface forms is the LLM's job and has
        already happened at write time; this is plain lexical containment
        over what it produced.
        """
        for f in canonical_forms(text):
            fn = self._norm_lexical(f)
            if fn and (fn in query_norm or query_norm in fn):
                return fn
        return None

    def _annotation_group(self, query: str, hits: list) -> list[int]:
        """Rows sharing a canonical form with the one the request names."""
        qn = self._norm_lexical(query)
        keys = {k for k in (self._implicated(qn, h.memory.text) for h in hits)
                if k}
        if not keys:
            return []
        return [h.memory.id for h in hits
                if any(self._norm_lexical(f) in keys
                       for f in canonical_forms(h.memory.text))]

    # ── control plane ───────────────────────────────────────────────

    def purge(self, query: str) -> int:
        hits = self.lethe.recall(query, k=20, lexical=True)
        if not hits:
            return 0
        if self.llm is not None:
            matched = self._llm_match_for_purge(query, hits)
            if matched:
                self.lethe.surrender(matched, mode="purge")
                return len(matched)
        ids = self._annotation_group(query, hits)
        if ids:
            self.lethe.surrender(ids, mode="purge")
            return len(ids)
        target = self._norm_lexical(hits[0].memory.text)
        ids = [h.memory.id for h in hits
               if self._norm_lexical(h.memory.text) == target]
        self.lethe.surrender(ids, mode="purge")
        return len(ids)

    def release(self, query: str) -> int:
        hits = self.lethe.recall(query, k=20, hybrid=True)
        if not hits:
            return 0
        if self.llm is not None:
            matched = self._llm_match_for_release(query, hits)
            if matched:
                self.lethe.surrender(matched, mode="release")
                return len(matched)
        ids = self._annotation_group(query, hits)
        if ids:
            self.lethe.surrender(ids, mode="release")
            return len(ids)
        thr = self._gap_threshold([h.similarity for h in hits])
        ids = [h.memory.id for h in hits if h.similarity >= thr]
        if not ids:
            return 0
        self.lethe.surrender(ids, mode="release")
        return len(ids)

    def supersede(self, old_query: str, new_text: str) -> None:
        # The parent takes top-1 of a k=1 recall.  Widen the recall so an
        # annotation can be seen, but keep top-1 as the answer whenever no
        # annotation fires -- otherwise widening k would itself change the
        # deterministic behaviour and confound the arm.
        hits = self.lethe.recall(old_query, k=20, hybrid=False)
        if not hits:
            self.lethe.inscribe(new_text)
            return
        qn = self._norm_lexical(old_query)
        target = next((h for h in hits if self._implicated(qn, h.memory.text)),
                      hits[0])

        if self.llm is not None:
            plan = self._llm_plan_supersede(target.memory.text,
                                            old_query, new_text)
            if plan.get("mode") == "partial":
                merged = plan.get("merged_text") or ""
                if merged.strip():
                    self.lethe.surrender(target.memory.id, mode="edit",
                                         new_text=merged)
                    return

        self.lethe.surrender(
            {"old": target.memory.id, "new": new_text},
            mode="supersede",
        )


__all__ = ["AnnotationAwareLetheAdapter", "canonical_forms"]
