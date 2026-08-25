"""The controlled placement ablation: one backend, one model, four cells.

The placement claim was previously supported by comparing *systems* --
A-MEM and Mem0's router for the inscribe-time regime, our hook for the
mutation-time one. Those systems differ in storage, retrieval, embedder
and extraction prompt as well as in where the LLM sits, so the comparison
identifies "these systems fail differently", not "placement causes it".

This module supplies the missing cells. The same store, the same model,
the same prompt family; the only thing that moves is *when* the model is
called:

                    mutation: none        mutation: hook
  inscribe: none    deterministic         mutation-time only
  inscribe: LLM     inscribe-time only    both

Inscribe-time canonicalisation is modelled the way the systems in that
regime actually work -- A-MEM extracts tags and links at write time, Mem0's
router rewrites and merges on ADD -- by asking the model, once per written
fact, for the canonical form of any identifier it contains and storing
those forms alongside the original text. The stored row is a superset of
what the deterministic path stores, so nothing is lost; what is gained is
exactly the surface-form bridge that later lexical matching needs.

What it deliberately does NOT get is the deletion request. That is the
asymmetry under test: an inscribe-time model sees what things *are* and
never sees what the user asked to *remove*.
"""
from __future__ import annotations

import json

from .adapter import LetheAdapter, _parse_json_response

LLM_PROMPT_INSCRIBE_CANON = """\
You are indexing a memory item at write time.  List the canonical
forms of every identifier the text contains, so that later lookups
using a different surface form still find this item.

Surface-form variations (case, whitespace, quoting, leading @,
optional separators in phone numbers / UUIDs / SSNs / credit cards)
are the SAME identifier and should collapse to one canonical form.
For a name written in a non-Latin script, also give the usual
romanised form, and vice versa.

TEXT:  {text}

Return exactly one JSON object:

  {{"canonical": ["<form>", ...]}}

Return an empty list if the text contains no identifier.  Do NOT
invent identifiers that are not present.
"""


class InscribeLLMLetheAdapter(LetheAdapter):
    """LLM at write time only: canonical forms are indexed, mutations are
    handled by the deterministic control plane."""

    name = "lethe_inscribe_llm"

    def __init__(self, embedder, vector_dim: int = 384, *, llm=None,
                 mutation_llm=None, stats: dict | None = None):
        # `llm` drives inscribe; `mutation_llm` drives the control plane.
        # Passing both gives the fourth cell of the table.
        super().__init__(embedder=embedder, vector_dim=vector_dim,
                         llm=mutation_llm)
        self.inscribe_llm = llm
        self.stats = stats if stats is not None else {
            "inscribe_calls": 0, "annotated": 0, "failed": 0}

    def inscribe(self, text: str) -> int:
        if self.inscribe_llm is None:
            return super().inscribe(text)
        self.stats["inscribe_calls"] += 1
        try:
            plan = _parse_json_response(
                self.inscribe_llm(LLM_PROMPT_INSCRIBE_CANON.format(text=text)))
            forms = [f for f in (plan.get("canonical") or [])
                     if isinstance(f, str) and f.strip()]
        except Exception:
            forms = []
            self.stats["failed"] += 1
        if not forms:
            return super().inscribe(text)
        # Store the canonical forms with the row rather than replacing it:
        # the deterministic scorer reads stored text, so rewriting would
        # change what must_not_contain sees and confound the ablation.
        self.stats["annotated"] += 1
        annotated = f"{text} [canonical: {'; '.join(dict.fromkeys(forms))}]"
        return super().inscribe(annotated)


__all__ = ["InscribeLLMLetheAdapter", "LLM_PROMPT_INSCRIBE_CANON"]


LLM_PROMPT_INSCRIBE_MERGE = """\
You are deciding, at write time, whether a new memory item describes an
entity already in the store under a different surface form.

Surface-form variations (case, whitespace, quoting, leading @, optional
separators in phone numbers / UUIDs / SSNs / credit cards) are the SAME
entity, as are a name in a non-Latin script and its romanisation.
Distinct entities that merely share a prefix (alice@acme.io vs
alice.smith@acme.io, 12345 vs 123456) are NOT the same.

NEW_ITEM:  {text}

EXISTING (one per line, indexed from 0):
{candidates}

Return exactly one JSON object.  If the new item describes the same
entity as some existing rows, list their indices and give a single
merged text that preserves every distinct fact from the new item and
those rows, written with one canonical form of the identifier:

  {{"same_entity_indices": [<int>, ...], "merged_text": "<text>"}}

If it is a new entity, return {{"same_entity_indices": [], "merged_text": ""}}.
"""


class MergeInscribeLetheAdapter(InscribeLLMLetheAdapter):
    """Inscribe-time LLM with *write authority*: it may rewrite and collapse
    existing rows, the way Mem0's ADD/UPDATE router and A-MEM's note
    evolution do, rather than only annotating what it sees.

    This is the strong form of the inscribe-time regime, and it sits
    deliberately on the boundary the ablation is probing: merging at write
    time IS a control-plane mutation, just one triggered by an arriving
    fact rather than by a user request. If capability tracks *when* the
    model runs, this configuration should stay at the deterministic level;
    if it tracks *whether the mutating operation is model-mediated*, this
    one should move.
    """

    name = "lethe_merge_inscribe"

    def inscribe(self, text: str) -> int:
        if self.inscribe_llm is None:
            return LetheAdapter.inscribe(self, text)
        self.stats["inscribe_calls"] += 1
        hits = self.lethe.recall(text, k=10, hybrid=True)
        if not hits:
            return LetheAdapter.inscribe(self, text)
        cands = "\n".join(f"{i}: {h.memory.text}" for i, h in enumerate(hits))
        try:
            plan = _parse_json_response(self.inscribe_llm(
                LLM_PROMPT_INSCRIBE_MERGE.format(text=text, candidates=cands)))
            idx = [i for i in (plan.get("same_entity_indices") or [])
                   if isinstance(i, int) and 0 <= i < len(hits)]
            merged = (plan.get("merged_text") or "").strip()
        except Exception:
            idx, merged = [], ""
            self.stats["failed"] += 1
        if not idx or not merged:
            return LetheAdapter.inscribe(self, text)
        self.stats["annotated"] += 1
        self.lethe.surrender([hits[i].memory.id for i in idx], mode="purge")
        return LetheAdapter.inscribe(self, merged)


__all__.append("MergeInscribeLetheAdapter")
