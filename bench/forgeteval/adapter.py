"""Minimal adapter interface for ForgetEval.

Any memory system that wants to be evaluated implements these methods.
ForgetEval tests use only this surface — they're system-agnostic.

Capabilities are explicit. A system that can't supersede (e.g. ChromaDB
or Mem0 in ADD-only mode) returns NotImplementedError; tests that need
that operation are scored as failures for that system, which is fair.
"""
from __future__ import annotations

import os

from typing import Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    name: str

    def reset(self) -> None: ...
    def inscribe(self, text: str) -> int | str: ...
    def recall_texts(self, query: str, k: int = 5) -> list[str]: ...

    # Optional operations. Systems that don't support them raise
    # NotImplementedError; ForgetEval marks those tests as N/A for the system.
    def supersede(self, old_query: str, new_text: str) -> None:
        raise NotImplementedError

    def release(self, query: str) -> int:
        """Soft-evict any memory matching the query.  Returns count released."""
        raise NotImplementedError

    def purge(self, query: str) -> int:
        """Hard-delete any memory matching the query."""
        raise NotImplementedError


# ─── Lethe adapter ────────────────────────────────────────────────────

# LLM-hook prompts.  Kept as module-level constants so they're easy to
# audit and to swap.  Both prompts ask the model for a small, well-
# structured JSON response — no free-form prose, no embedded reasoning.
# This keeps the parsing tiny and the LLM call narrow.

LLM_PROMPT_SUPERSEDE = """\
You are deciding how to apply a memory supersession.

EXISTING_MEMORY:  {old_text}
SUPERSEDE_QUERY:  {query}
NEW_FACT:         {new_text}

**Default is ATOMIC** — replace the whole memory with NEW_FACT.
Choose ATOMIC when EXISTING_MEMORY's topic matches SUPERSEDE_QUERY
(even if paraphrased, dated, or recursive).

Choose PARTIAL when EXISTING_MEMORY combines two **distinct-topic**
facts (different attributes about the same subject, like
location-vs-employer or marital-status-vs-employer), and
SUPERSEDE_QUERY targets only one attribute.  In partial mode,
preserve the unaffected attribute and splice NEW_FACT into the
addressed one.

Examples
--------

EXISTING_MEMORY:  User lives in Berlin and works at Stripe as a backend engineer.
SUPERSEDE_QUERY:  user city of residence
NEW_FACT:         User relocated to Madrid and continues working remotely.
→ {{"mode": "partial", "merged_text": "User relocated to Madrid and works at Stripe as a backend engineer."}}

EXISTING_MEMORY:  User is married to Jamie and works at Google.
SUPERSEDE_QUERY:  user employer Google
NEW_FACT:         User joined Anthropic as a research engineer.
→ {{"mode": "partial", "merged_text": "User is married to Jamie and joined Anthropic as a research engineer."}}

EXISTING_MEMORY:  User does NOT work at Anthropic and has never interviewed there.
SUPERSEDE_QUERY:  user Anthropic employment status
NEW_FACT:         User actually joined Anthropic last quarter.
→ {{"mode": "atomic"}}      (both clauses are about Anthropic-employment — atomic supersedes them together)

EXISTING_MEMORY:  User joined Google in 2020.
SUPERSEDE_QUERY:  user current employer
NEW_FACT:         User joined Meta in 2022.
→ {{"mode": "atomic"}}      (single-topic supersession)

Format
------
Reply with exactly one JSON object and nothing else:
  {{"mode": "atomic"}}
or
  {{"mode": "partial", "merged_text": "<the merged sentence>"}}

Do not add facts not present in either source.
"""

LLM_PROMPT_PURGE_MATCH = """\
You are grouping memory items that describe the same underlying
identifier.  Surface-form variations (case, whitespace, quoting,
leading @, optional separators in phone numbers / UUIDs / SSNs /
credit cards) all count as the same identifier.

TARGET_IDENTIFIER:  {target}

CANDIDATES (one per line, indexed from 0):
{candidates}

Return exactly one JSON object listing the candidate indices whose
text describes the SAME identifier as the target.  Distinct
identifiers that happen to share a prefix (e.g. alice@acme.io vs
alice.smith@acme.io, 12345 vs 123456) MUST NOT be grouped.

  {{"matching_indices": [<int>, ...]}}
"""


LLM_PROMPT_RELEASE_MATCH = """\
You are deciding which memory items should be released (soft-deleted)
based on a natural-language release request.

RELEASE_REQUEST:  {request}

CANDIDATES (one per line, indexed from 0):
{candidates}

Return exactly one JSON object listing the candidate indices whose
content should be released according to the request.  Include a
candidate if and only if it (a) describes the entity or topic the
request targets, OR (b) mentions that entity even as part of a
compound statement.  Do NOT include candidates that only share an
attribute (e.g. same city, same job) with the target — those are
sibling facts, not target facts.

  {{"matching_indices": [<int>, ...]}}
"""


def _parse_json_response(s: str) -> dict:
    """Extract the first JSON object from a model response, tolerating
    fenced code blocks or surrounding prose.  Raises ValueError if no
    JSON object is found."""
    import json
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError(f"no JSON object in model response: {s!r}")
    return json.loads(m.group(0))


class LetheAdapter:
    """Default ForgetEval adapter for Lethe.

    With ``llm=None`` (the default), the adapter exposes only Lethe's
    deterministic primitives and ships no semantic heuristics:

      - supersede  →  atomic: wipe the best BM25 match, inscribe new
      - release    →  adaptive-gap (a documented numerical procedure,
                      not a string heuristic)
      - purge      →  case-insensitive equality of the BM25-top-1 text,
                      plus exact-text duplicates

    With ``llm`` set to a ``Callable[[str], str]`` that takes a prompt
    and returns a model response, the adapter routes semantic decisions
    through the LLM — clause-aware supersede and identifier-equivalent
    purge.  The recall hot path remains LLM-free in both modes; only
    the explicit mutation operations consult the model, and only once
    per call.
    """

    name = "lethe"

    def __init__(self, embedder, vector_dim: int = 384, *,
                 llm=None, lethe_llm=None):
        from lethe import Lethe
        self._Lethe = Lethe
        self.embedder = embedder
        self.vector_dim = vector_dim
        self.llm = llm                 # for supersede/purge planning
        self.lethe_llm = lethe_llm     # passed through to Lethe(...)
        self.lethe = None
        if llm is not None:
            self.name = "lethe+llm"

    def reset(self) -> None:
        if self.lethe is not None:
            try:
                self.lethe.close()
            except Exception:
                pass
        self.lethe = self._Lethe(":memory:",
                                 vector_dim=self.vector_dim,
                                 embedder=self.embedder,
                                 llm=self.lethe_llm)

    def inscribe(self, text: str) -> int:
        return self.lethe.inscribe(text)

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        results = self.lethe.recall(query, k=k, hybrid=False)
        return [r.memory.text for r in results]

    # ─── supersede ──────────────────────────────────────────────────

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self.lethe.recall(old_query, k=1, hybrid=False)
        if not hits:
            self.lethe.inscribe(new_text)
            return
        target = hits[0]

        if self.llm is not None:
            plan = self._llm_plan_supersede(target.memory.text,
                                            old_query, new_text)
            if plan.get("mode") == "partial":
                merged = plan.get("merged_text") or ""
                if merged.strip():
                    self.lethe.surrender(target.memory.id, mode="edit",
                                         new_text=merged)
                    return
            # mode "atomic" or unrecognized → fall through to atomic

        self.lethe.surrender(
            {"old": target.memory.id, "new": new_text},
            mode="supersede",
        )

    def _llm_plan_supersede(self, old_text: str, old_query: str,
                             new_text: str) -> dict:
        prompt = LLM_PROMPT_SUPERSEDE.format(
            old_text=old_text, query=old_query, new_text=new_text,
        )
        try:
            return _parse_json_response(self.llm(prompt))
        except Exception:
            return {"mode": "atomic"}    # any failure → safe default

    # ─── release ────────────────────────────────────────────────────

    @staticmethod
    def _gap_threshold(sims: list[float], min_gap: float = 0.05) -> float:
        """Find the natural cutoff in a sorted-descending similarity list:
        the midpoint of the largest gap.  Falls back to top * 0.95 when
        there is no significant gap (only one tight cluster of hits)."""
        if not sims:
            return float("inf")
        if len(sims) == 1:
            return sims[0] * 0.95
        s = sorted(sims, reverse=True)
        best_gap = 0.0
        best_mid = s[0] * 0.95
        for i in range(len(s) - 1):
            gap = s[i] - s[i + 1]
            if gap > best_gap:
                best_gap = gap
                best_mid = (s[i] + s[i + 1]) / 2.0
        return best_mid if best_gap >= min_gap else s[0] * 0.95

    def release(self, query: str) -> int:
        # Hybrid recall (vec + BM25 via RRF) for release.  For
        # identifier-shaped queries the BM25 leg sharpens the
        # ranking; for natural-language queries the vec leg carries
        # the semantic load.  RRF weights both — no detection
        # heuristic needed.
        hits = self.lethe.recall(query, k=20, hybrid=True)
        if not hits:
            return 0

        if self.llm is not None:
            matched = self._llm_match_for_release(query, hits)
            if matched:
                self.lethe.surrender(matched, mode="release")
                return len(matched)
            # empty LLM result → fall through to adaptive-gap below

        thr = self._gap_threshold([h.similarity for h in hits])
        ids = [h.memory.id for h in hits if h.similarity >= thr]
        if not ids:
            return 0
        self.lethe.surrender(ids, mode="release")
        return len(ids)

    def _llm_match_for_release(self, query: str, hits: list) -> list[int]:
        candidates = "\n".join(f"{i}: {h.memory.text}"
                               for i, h in enumerate(hits))
        prompt = LLM_PROMPT_RELEASE_MATCH.format(
            request=query, candidates=candidates,
        )
        try:
            plan = _parse_json_response(self.llm(prompt))
            picks = plan.get("matching_indices") or []
            return [hits[i].memory.id for i in picks
                    if isinstance(i, int) and 0 <= i < len(hits)]
        except Exception:
            return []

    # ─── purge ──────────────────────────────────────────────────────

    @staticmethod
    def _norm_lexical(s: str) -> str:
        """Minimal text normalization: NFKC + lowercase + collapsed
        whitespace.  This is plain Unicode hygiene — not an identifier-
        aware canonicalizer.  Two strings that differ only in case or
        whitespace will compare equal under this norm; anything else
        (separators, quoting, format variations) is left to the LLM."""
        import unicodedata
        return " ".join(unicodedata.normalize("NFKC", s).lower().split())

    def purge(self, query: str) -> int:
        hits = self.lethe.recall(query, k=20, lexical=True)
        if not hits:
            return 0

        if self.llm is not None:
            matched = self._llm_match_for_purge(query, hits)
            if matched:
                self.lethe.surrender(matched, mode="purge")
                return len(matched)
            # Empty LLM result → fall through to default below.

        # Default: group by NFKC-lowercase-whitespace equivalence.
        target = self._norm_lexical(hits[0].memory.text)
        ids = [h.memory.id for h in hits
               if self._norm_lexical(h.memory.text) == target]
        self.lethe.surrender(ids, mode="purge")
        return len(ids)

    def _llm_match_for_purge(self, query: str, hits: list) -> list[int]:
        candidates = "\n".join(f"{i}: {h.memory.text}"
                               for i, h in enumerate(hits))
        prompt = LLM_PROMPT_PURGE_MATCH.format(
            target=query, candidates=candidates,
        )
        try:
            plan = _parse_json_response(self.llm(prompt))
            picks = plan.get("matching_indices") or []
            return [hits[i].memory.id for i in picks
                    if isinstance(i, int) and 0 <= i < len(hits)]
        except Exception:
            return []


# ─── Mem0 adapter ─────────────────────────────────────────────────────

class Mem0Adapter:
    """Mem0 has add/search/update/delete, but no native supersession.
    For amnesia/purge we delete-by-query via list-and-match. Mem0's
    A.U.D.N. mode (infer=True) tries to auto-supersede via LLM but is
    flaky in production (their own team recently shipped ADD-only as
    default); we don't enable it here, to keep apples-to-apples."""

    name = "mem0"

    def __init__(self, embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dims: int = 384, *, infer: bool = False,
                 llm_model: str | None = None, llm_base_url: str | None = None,
                 llm_api_key: str | None = None):
        # infer=True engages Mem0's LLM-driven ADD/UPDATE/DELETE router --
        # the design distinctive of Mem0 relative to a plain vector store,
        # and the configuration a real deployment runs.
        self.infer = infer
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        # Mem0 v2 unconditionally instantiates an OpenAI client at init time,
        # even though we run with infer=False and never actually call the LLM.
        # Set a no-op key so the constructor doesn't raise.
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-noop-forgeteval")
        try:
            from mem0 import Memory
        except ImportError as e:
            raise ImportError("pip install mem0ai") from e
        self._Memory = Memory
        self.embedder_model = embedder_model
        self.embedding_dims = embedding_dims
        self.user_id = "forget_eval"
        self.m = None

    def _config(self) -> dict:
        import tempfile
        qpath = tempfile.mkdtemp(prefix="mem0_fe_")
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "forget_eval",
                    "path": qpath,
                    "embedding_model_dims": self.embedding_dims,
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": self.embedder_model},
            },
            # With infer=False the LLM is never called, but Mem0 v2 still
            # builds an OpenAI client at construct time, hence the no-op key
            # shim in __init__.  With infer=True it is the router's model.
            **({"llm": {"provider": "openai", "config": {
                "model": self.llm_model,
                "temperature": 0.0,
                # The router emits rewritten row text, so it is the
                # prompt that overruns a small budget; 1024 truncated it.
                "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "3000")),
                # Without this Mem0 builds its client from the ambient
                # OPENAI_API_KEY and 401s wherever that is unset, storing
                # nothing and raising nothing.
                **({"api_key": self.llm_api_key} if self.llm_api_key else {}),
                **({"openai_base_url": self.llm_base_url}
                   if self.llm_base_url else {}),
            }}} if self.infer and self.llm_model else {}),
        }

    def reset(self) -> None:
        # First call: instantiate (which loads the HF embedder once).
        # Subsequent calls: just wipe the user's memories so we don't reload
        # the model on every test — that was costing ~1s per test.
        if self.m is None:
            self.m = self._Memory.from_config(self._config())
        else:
            try:
                self.m.delete_all(user_id=self.user_id)
            except Exception:
                # Fall back to full re-init if delete_all isn't available
                self.m = self._Memory.from_config(self._config())

    def inscribe(self, text: str) -> str:
        # infer=False: pure ADD, no A.U.D.N., no LLM call.  This is the
        # mode Mem0 itself defaulted to after their UPDATE/DELETE
        # reliability issues.
        result = self.m.add(text, user_id=self.user_id,
                            infer=self.infer)
        return str(result)

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        out = self.m.search(query=query, filters={"user_id": self.user_id}, top_k=k)
        items = (out.get("results") if isinstance(out, dict) else out) or []
        texts: list[str] = []
        for it in items:
            if isinstance(it, dict):
                texts.append(it.get("memory") or it.get("text") or "")
            else:
                texts.append(str(it))
        return texts

    def supersede(self, old_query: str, new_text: str) -> None:
        # Mem0 in infer=False is ADD-only — the old fact persists.  We
        # emulate by deleting the best match for `old_query` and adding
        # the new text, which is the most charitable interpretation.
        out = self.m.search(query=old_query, filters={"user_id": self.user_id}, top_k=1)
        items = (out.get("results") if isinstance(out, dict) else out) or []
        if items:
            mid = (items[0].get("id") if isinstance(items[0], dict) else None)
            if mid:
                try:
                    self.m.delete(memory_id=mid)
                except Exception:
                    pass
        self.m.add(new_text, user_id=self.user_id, infer=self.infer)

    def _delete_matching(self, query: str, top_k: int = 20) -> int:
        out = self.m.search(query=query, filters={"user_id": self.user_id}, top_k=top_k)
        items = (out.get("results") if isinstance(out, dict) else out) or []
        # Mem0's search doesn't expose cosine similarity uniformly, so use
        # its own "score" if present; otherwise delete the top hit only.
        scores: list[float] = []
        ids: list[str] = []
        for it in items:
            if isinstance(it, dict):
                ids.append(it.get("id") or "")
                scores.append(float(it.get("score") or 0.0))
        if not ids:
            return 0
        # Same adaptive-gap idea as LetheAdapter when scores are available
        if scores and max(scores) > 0:
            from bench.forgeteval.adapter import LetheAdapter
            thr = LetheAdapter._gap_threshold(scores)
            chosen = [i for i, s in zip(ids, scores) if s >= thr and i]
        else:
            chosen = ids[:1]
        for mid in chosen:
            try:
                self.m.delete(memory_id=mid)
            except Exception:
                pass
        return len(chosen)

    def release(self, query: str) -> int:
        return self._delete_matching(query)

    def purge(self, query: str) -> int:
        return self._delete_matching(query)


# ─── LangGraph / LangMem adapter ──────────────────────────────────────

class LangGraphAdapter:
    """LangChain's default agent-memory primitive: LangGraph's
    ``InMemoryStore`` with vector indexing.  This is the bare BaseStore
    that the LangMem package layers on top of — we benchmark the
    underlying store directly so the comparison is between *storage*
    primitives, not between LLM-driven memory managers.

    LangMem's higher-level manager calls an LLM at every write to
    extract/dedup/delete memories, which (a) breaks the LLM-free
    invariant we hold for ForgetEval and (b) makes supersession
    implicit rather than programmatic.  The InMemoryStore baseline
    here is what an engineer gets when they use LangChain "out of the
    box" without wiring an LLM into the write path.

    Primitive coverage:

        inscribe  ↦  store.put((ns,), key=uuid, value={"text": text})
        recall    ↦  store.search((ns,), query=q, limit=k)
        supersede ↦  delete(old_key) + put(new uuid, new_text)
        release   ↦  delete (no soft-delete in BaseStore)
        purge     ↦  delete top-1 BM25-equivalent

    No external service, no API key, pure CPU.
    """

    name = "langmem"

    def __init__(self, embedder, vector_dim: int = 384):
        try:
            from langgraph.store.memory import InMemoryStore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install langmem  (pulls langgraph)") from e
        self._Store = InMemoryStore
        self.embedder = embedder
        self.vector_dim = vector_dim
        self.store = None
        self.ns = ("forget_eval",)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self.embedder(t)) for t in texts]

    def reset(self) -> None:
        # InMemoryStore is reset by re-instantiating; embed callback
        # is invoked by .put/.search internally on the text field.
        self.store = self._Store(
            index={"dims": self.vector_dim,
                   "embed": self._embed_batch,
                   "fields": ["text"]},
        )

    def inscribe(self, text: str) -> str:
        import uuid
        key = uuid.uuid4().hex
        self.store.put(self.ns, key, {"text": text})
        return key

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        hits = self.store.search(self.ns, query=query, limit=k)
        return [h.value.get("text", "") for h in hits]

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self.store.search(self.ns, query=old_query, limit=1)
        if hits:
            self.store.delete(self.ns, hits[0].key)
        import uuid
        self.store.put(self.ns, uuid.uuid4().hex, {"text": new_text})

    def release(self, query: str) -> int:
        hits = self.store.search(self.ns, query=query, limit=20)
        if not hits:
            return 0
        # Use the same adaptive-gap policy as LetheAdapter for fairness.
        scores = [h.score or 0.0 for h in hits]
        thr = LetheAdapter._gap_threshold(scores)
        evicted = 0
        for h in hits:
            if (h.score or 0.0) >= thr:
                self.store.delete(self.ns, h.key)
                evicted += 1
        return evicted

    def purge(self, query: str) -> int:
        hits = self.store.search(self.ns, query=query, limit=20)
        if not hits:
            return 0
        target_text = hits[0].value.get("text", "")
        purged = 0
        for h in hits:
            if h.value.get("text", "") == target_text:
                self.store.delete(self.ns, h.key)
                purged += 1
        return purged


# ─── LangGraph + LLM hook adapter ─────────────────────────────────────

class LangGraphLLMAdapter(LangGraphAdapter):
    """LangGraph InMemoryStore augmented with the same LLM-hook contract
    as LetheAdapter (supersede planner, release-match, purge-match).
    Used to disentangle the LLM-hook lift from the edit-primitive lift
    in the ablation: LangGraph has no in-place edit, so for partial
    supersede the LLM hook returns a merged text which we add as a
    fresh row replacing the old (functionally equivalent to Lethe's
    edit primitive under substring scoring)."""

    name = "langmem_llm"

    def __init__(self, embedder, vector_dim: int = 384, *,
                 llm=None):
        super().__init__(embedder=embedder, vector_dim=vector_dim)
        self.llm = llm

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self.store.search(self.ns, query=old_query, limit=1)
        if not hits:
            import uuid
            self.store.put(self.ns, uuid.uuid4().hex, {"text": new_text})
            return
        target = hits[0]
        target_text = target.value.get("text", "")

        # LLM-planned supersede: atomic (replace whole row) vs partial
        # (merge one clause into a fresh row).
        if self.llm is not None:
            try:
                prompt = LLM_PROMPT_SUPERSEDE.format(
                    old_text=target_text, query=old_query, new_text=new_text,
                )
                plan = _parse_json_response(self.llm(prompt))
            except Exception:
                plan = {"mode": "atomic"}
            if plan.get("mode") == "partial":
                merged = plan.get("merged_text") or ""
                if merged.strip():
                    # No native edit primitive -> delete old + add merged.
                    self.store.delete(self.ns, target.key)
                    import uuid
                    self.store.put(self.ns, uuid.uuid4().hex,
                                   {"text": merged})
                    return
            # atomic / fallthrough: delete old + add new
        self.store.delete(self.ns, target.key)
        import uuid
        self.store.put(self.ns, uuid.uuid4().hex, {"text": new_text})

    def release(self, query: str) -> int:
        hits = self.store.search(self.ns, query=query, limit=20)
        if not hits:
            return 0

        if self.llm is not None:
            try:
                candidates = "\n".join(
                    f"{i}: {h.value.get('text', '')}"
                    for i, h in enumerate(hits)
                )
                prompt = LLM_PROMPT_RELEASE_MATCH.format(
                    request=query, candidates=candidates,
                )
                plan = _parse_json_response(self.llm(prompt))
                picks = plan.get("matching_indices") or []
                keys = [hits[i].key for i in picks
                        if isinstance(i, int) and 0 <= i < len(hits)]
                if keys:
                    for k in keys:
                        self.store.delete(self.ns, k)
                    return len(keys)
            except Exception:
                pass  # fall through to gap-threshold

        scores = [h.score or 0.0 for h in hits]
        thr = LetheAdapter._gap_threshold(scores)
        evicted = 0
        for h in hits:
            if (h.score or 0.0) >= thr:
                self.store.delete(self.ns, h.key)
                evicted += 1
        return evicted

    def purge(self, query: str) -> int:
        hits = self.store.search(self.ns, query=query, limit=20)
        if not hits:
            return 0

        if self.llm is not None:
            try:
                candidates = "\n".join(
                    f"{i}: {h.value.get('text', '')}"
                    for i, h in enumerate(hits)
                )
                prompt = LLM_PROMPT_PURGE_MATCH.format(
                    target=query, candidates=candidates,
                )
                plan = _parse_json_response(self.llm(prompt))
                picks = plan.get("matching_indices") or []
                keys = [hits[i].key for i in picks
                        if isinstance(i, int) and 0 <= i < len(hits)]
                if keys:
                    for k in keys:
                        self.store.delete(self.ns, k)
                    return len(keys)
            except Exception:
                pass  # fall through to default

        # Default: NFKC-lowercase-whitespace equivalence on text.
        target_text = LetheAdapter._norm_lexical(hits[0].value.get("text", ""))
        purged = 0
        for h in hits:
            if LetheAdapter._norm_lexical(h.value.get("text", "")) == target_text:
                self.store.delete(self.ns, h.key)
                purged += 1
        return purged


# ─── Cognee adapter (requires LLM API key) ────────────────────────────

class CogneeAdapter:
    """Cognee v1.5 (``topoteretes/cognee``).

    Cognee is the only system in this study that exposes all six Protocol
    methods *natively*: a soft delete (``delete(..., mode="soft")``), a
    partial edit (``update``), and a hard purge (``forget``).  It is
    therefore the sharpest available test of whether primitive
    completeness is sufficient for forgetting, or whether LLM placement
    still dominates.

    Cognee's retriever exposes the abstraction cascade directly, and the
    choice of ``SearchType`` decides what "recall" means:

        CHUNKS            verbatim stored text  (``text`` + ``document_id``)
        SUMMARIES         LLM-derived summary of a chunk, stored as its
                          own vector entry with ``source_chunk_id``
        GRAPH_COMPLETION  a synthesised answer over the knowledge graph

    We evaluate two configurations.  ``retrieval="chunks"`` is the
    apples-to-apples comparison with every other store in the study
    (record-level recall of stored surface forms).  ``retrieval="graph"``
    is what a real Cognee deployment returns to an agent.  Reporting both
    is the point: a derived layer can answer from artifacts that outlive
    the record their source was purged from.

    Purge/release/supersede target selection uses *Cognee's own*
    retriever -- search for the query, take ``document_id`` off the hits,
    act on those ids -- exactly as the Mem0 and LangGraph adapters do.
    No privileged oracle is used to locate the row to delete.

    *Requires:* ``pip install cognee fastembed`` and an OpenAI-compatible
    endpoint (``LLM_PROVIDER`` / ``LLM_ENDPOINT`` / ``LLM_API_KEY``);
    ``cognify`` calls the LLM unconditionally, so this adapter cannot run
    fully offline.  Each instance owns a private dataset so that parallel
    workers do not clear each other's state.

    Primitive coverage:

        inscribe  |->  cognee.add(text, dataset_name=<private>)
        recall    |->  cognee.search(q, query_type=CHUNKS|GRAPH_COMPLETION)
        supersede |->  forget(data_id of best hit) + add(new_text)
        release   |->  delete(data_id, dataset_id, mode="soft")
        purge     |->  forget(data_id=..., dataset_id=...)
    """

    name = "cognee"

    def __init__(self, dataset: str | None = None, *,
                 retrieval: str = "chunks", top_k: int = 10):
        try:
            import cognee
            from cognee.modules.search.types import SearchType
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install cognee fastembed") from e
        import asyncio
        import uuid

        self._cognee = cognee
        self._SearchType = SearchType
        self.retrieval = retrieval
        self.top_k = top_k
        # private dataset per instance: reset() must not wipe a sibling
        # worker's store, and cognee's backing DB is process-global.
        self.dataset = dataset or f"forgeteval_{uuid.uuid4().hex[:12]}"
        self._loop = asyncio.new_event_loop()
        self._dataset_id = None
        self._dirty = False          # facts added but not yet cognified
        # A swallowed exception here would look exactly like a store that
        # refuses to forget, so failures are recorded, not ignored.
        self.errors: list[str] = []

    # ── plumbing ─────────────────────────────────────────────────────

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _cognify_if_needed(self) -> None:
        """cognify is what makes added text searchable, and it is an LLM
        call, so we batch it: mark dirty on inscribe and materialise once,
        lazily, before the first read or mutation of each case."""
        if not self._dirty:
            return
        try:
            self._run(self._cognee.cognify(datasets=[self.dataset]))
        except Exception:
            pass
        self._dirty = False

    def _resolve_dataset_id(self):
        if self._dataset_id is not None:
            return self._dataset_id
        from cognee.modules.data.methods import get_datasets
        from cognee.modules.users.methods import get_default_user

        async def _find():
            user = await get_default_user()
            for d in await get_datasets(user.id):
                if d.name == self.dataset:
                    return d.id
            return None

        self._dataset_id = self._run(_find())
        return self._dataset_id

    def _hits(self, query: str, k: int) -> list[dict]:
        """Cognee's own chunk-level retrieval, carrying document_id."""
        self._cognify_if_needed()
        try:
            res = self._run(self._cognee.search(
                query, query_type=self._SearchType.CHUNKS,
                datasets=[self.dataset], top_k=k))
        except Exception:
            return []
        import uuid as _uuid
        out = []
        for r in (res or []):
            if not isinstance(r, dict):
                continue
            did = r.get("document_id")
            if isinstance(did, str):
                try:
                    r = {**r, "document_id": _uuid.UUID(did)}
                except ValueError:
                    continue
            out.append(r)
        return out

    def _forget_ids(self, doc_ids) -> int:
        ds_id = self._resolve_dataset_id()
        n = 0
        for did in doc_ids:
            try:
                self._run(self._cognee.forget(data_id=did, dataset_id=ds_id))
                n += 1
            except Exception as e:
                self.errors.append(f"forget({did}): {type(e).__name__}: {e}")
        return n

    # ── Protocol ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Full prune, not a dataset-scoped forget.

        Cognee's ``datasets=[...]`` filter does not isolate CHUNKS
        retrieval: with a dataset filter in place, search still returns
        chunks written under other dataset names.  A dataset-scoped
        forget therefore leaves earlier cases visible and silently
        corrupts every case after the first.  We verified this and fall
        back to a global prune, which does isolate correctly.

        Consequence: a Cognee run is single-worker.  Two workers sharing
        the process-global store would prune each other mid-case.
        """
        try:
            self._run(self._cognee.prune.prune_data())
            self._run(self._cognee.prune.prune_system(metadata=True))
        except Exception:
            pass
        self._dataset_id = None
        self._dirty = False

    def inscribe(self, text: str) -> str:
        self._run(self._cognee.add(text, dataset_name=self.dataset))
        self._dirty = True
        return ""

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        self._cognify_if_needed()
        if self.retrieval == "graph":
            try:
                res = self._run(self._cognee.search(
                    query, query_type=self._SearchType.GRAPH_COMPLETION,
                    datasets=[self.dataset], top_k=k))
            except Exception:
                return []
            return [str(r) for r in (res or [])][:k]
        return [h.get("text", "") for h in self._hits(query, k)][:k]

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self._hits(old_query, self.top_k)
        if hits:
            self._forget_ids({hits[0]["document_id"]}
                             if hits[0].get("document_id") else set())
        self.inscribe(new_text)

    def release(self, query: str) -> int:
        """Cognee's native soft delete.  Unlike every other store in the
        extended set, this is a first-class primitive rather than N/A."""
        ds_id = self._resolve_dataset_id()
        n = 0
        for h in self._hits(query, self.top_k):
            did = h.get("document_id")
            if not did:
                continue
            try:
                self._run(self._cognee.delete(data_id=did, dataset_id=ds_id,
                                              mode="soft"))
                n += 1
            except Exception:
                pass
        return n

    def purge(self, query: str) -> int:
        ids = {h["document_id"] for h in self._hits(query, self.top_k)
               if h.get("document_id")}
        return self._forget_ids(ids)



# ─── A-MEM adapter (requires Ollama or OpenAI) ────────────────────────

class AMemAdapter:
    """A-MEM (Xu et al., NeurIPS 2025, arXiv 2502.12110): Zettelkasten-
    style agentic memory.  Each ``add_note`` triggers an LLM call to
    generate tags / context / inter-note links, so the system cannot
    run without either an Ollama daemon or an OpenAI API key.

    *Install:* not on PyPI as of May 2026 — clone the repo and pip-install.
    ::

        git clone https://github.com/agiresearch/A-mem
        cd A-mem && pip install -e .

    Primitive coverage:

        inscribe  ↦  ms.add_note(text)
        recall    ↦  ms.search_agentic(query, k=k)
        supersede ↦  ms.update(memory_id, content=new_text)
        release   ↦  raises NotImplementedError (no soft-delete)
        purge     ↦  ms.delete(memory_id)
    """

    name = "amem"

    def __init__(self, llm_backend: str = "ollama",
                 embedder_model: str = "all-MiniLM-L6-v2",
                 llm_model: str | None = None):
        # A-MEM's tag/link extraction runs at inscribe time -- this is the
        # inscribe-time-LLM regime, so which model does the extraction is a
        # reportable configuration choice, not an implementation detail.
        self.llm_model = llm_model
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "A-MEM is not on PyPI — clone "
                "https://github.com/agiresearch/A-mem and `pip install -e .`"
            ) from e
        self._System = AgenticMemorySystem
        self.llm_backend = llm_backend
        self.embedder_model = embedder_model
        self.ms = None

    def reset(self) -> None:
        # Recreate fresh in-process system.  ChromaDB is embedded; the
        # constructor instantiates the LLM controller, which is where
        # the Ollama/OpenAI dependency materializes.
        self.ms = self._System(
            model_name=self.embedder_model,
            llm_backend=self.llm_backend,
            **({"llm_model": self.llm_model} if self.llm_model else {}),
        )
        # ChromaDB collection lives across instances by default; wipe
        # the namespace explicitly so cases don't leak into each other.
        try:
            for mid in list(self.ms.memories.keys()):
                self.ms.delete(mid)
        except Exception:
            pass

    def inscribe(self, text: str) -> str:
        return self.ms.add_note(text)

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        results = self.ms.search_agentic(query, k=k)
        out: list[str] = []
        for r in results or []:
            if isinstance(r, dict):
                out.append(r.get("content") or r.get("text") or str(r))
            elif hasattr(r, "content"):
                out.append(r.content)
            else:
                out.append(str(r))
        return out

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self.ms.search_agentic(old_query, k=1)
        if not hits:
            self.ms.add_note(new_text)
            return
        target = hits[0]
        mid = target.get("id") if isinstance(target, dict) else getattr(target, "id", None)
        if mid is None:
            self.ms.add_note(new_text)
            return
        try:
            self.ms.update(mid, content=new_text)
        except Exception:
            self.ms.delete(mid)
            self.ms.add_note(new_text)

    def release(self, query: str) -> int:
        raise NotImplementedError(
            "A-MEM has no documented soft-delete primitive."
        )

    def purge(self, query: str) -> int:
        hits = self.ms.search_agentic(query, k=5)
        purged = 0
        for h in hits or []:
            mid = h.get("id") if isinstance(h, dict) else getattr(h, "id", None)
            if mid is None:
                continue
            try:
                self.ms.delete(mid)
                purged += 1
            except Exception:
                pass
        return purged


# ─── MemPalace adapter ────────────────────────────────────────────────

class MemPalaceAdapter:
    """MemPalace is verbatim-everything: it does NOT support delete,
    update, or supersede.  All forgetting operations raise — the
    benchmark scores them as N/A, which is the honest reflection of
    the library's design."""

    name = "mempalace"

    def __init__(self):
        try:
            from mempalace.diary_ingest import get_collection
            from mempalace.layers import MemoryStack
            from mempalace.miner import add_drawer
        except ImportError as e:
            raise ImportError("pip install mempalace") from e
        self._get_collection = get_collection
        self._MemoryStack = MemoryStack
        self._add_drawer = add_drawer
        self.palace = None
        self.col = None
        self.stack = None
        self.chunk = 0

    def reset(self) -> None:
        import tempfile
        from pathlib import Path
        self.palace = Path(tempfile.mkdtemp(prefix="mp_fe_"))
        self.col = self._get_collection(str(self.palace), create=True)
        self.stack = self._MemoryStack(palace_path=str(self.palace))
        self.chunk = 0

    def inscribe(self, text: str) -> int:
        self._add_drawer(
            collection=self.col,
            wing="bench",
            room="conv",
            content=text,
            source_file="forgeteval.txt",
            chunk_index=self.chunk,
            agent="bench",
        )
        self.chunk += 1
        return self.chunk

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        out = self.stack.search(query, wing="bench", room="conv", n_results=k)
        # MemPalace returns a string blob; treat the whole thing as one "text"
        return [str(out)] if out else []

    # MemPalace has no delete / update / supersede primitives — these
    # raise so the runner records them as N/A (honest, not a failure).
    def supersede(self, old_query: str, new_text: str) -> None:
        raise NotImplementedError("MemPalace has no supersede primitive")

    def release(self, query: str) -> int:
        raise NotImplementedError("MemPalace has no release primitive")

    def purge(self, query: str) -> int:
        raise NotImplementedError("MemPalace has no purge primitive")


# ─── MemOS adapter ────────────────────────────────────────────────────

class MemOSAdapter:
    """MemOS / MemoryOS v2 (``MemTensor/MemOS``, arXiv 2507.03724).

    MemOS matters here for one structural reason: its unified memory API
    exposes ``add`` / ``search`` / ``update`` / ``delete``, so it is the
    only system besides the reference implementation with a **native
    partial-edit primitive**.  \\famname{compound\\_fact} is otherwise
    unwinnable by construction for every comparator, which invites the
    objection that the category was written around one system's feature.
    MemOS lets that objection be tested.

    The recall path is embedding-only (sentence-transformers +
    a local Qdrant); the LLM is configured as MemOS's *extractor* and is
    not consulted on retrieval, so this row sits in the deterministic /
    vec-only regime rather than the hook regime.

    *Install:* ``pip install MemoryOS sentence_transformers qdrant_client``
    (the distribution is ``MemoryOS``; the import name is ``memos``).

    Primitive coverage:

        inscribe  |->  add([{"memory": text}])
        recall    |->  search(query, top_k=k)
        supersede |->  update(id_of_top_hit, {"memory": new_text})
                       -- native in-place edit, no delete+add round trip
        release   |->  N/A: MemOS has no soft-delete / TTL primitive
        purge     |->  delete([ids whose text matches the top hit])

    ``reset`` deletes by id rather than calling ``delete_all()``: on
    v2.0.30 ``delete_all()`` returns without clearing the collection
    (verified -- ``get_all()`` is unchanged across the call), which would
    leak every case into the next one.
    """

    name = "memos"

    def __init__(self, *, llm_backend: str = "deepseek",
                 llm_model: str = "deepseek-v4-flash",
                 api_key: str | None = None,
                 api_base: str = "https://api.deepseek.com/v1",
                 embedder_model: str = "all-MiniLM-L6-v2",
                 vector_dim: int = 384):
        try:
            from memos.configs.memory import GeneralTextMemoryConfig
            from memos.memories.textual.general import GeneralTextMemory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install MemoryOS sentence_transformers qdrant_client"
            ) from e
        import os
        import tempfile
        import uuid

        key = api_key or os.environ.get("LLM_API_KEY") or ""
        # private on-disk Qdrant per instance so parallel workers cannot
        # see or clear each other's collection
        self._dir = tempfile.mkdtemp(prefix="memos_forgeteval_")
        cfg = GeneralTextMemoryConfig(
            extractor_llm={
                "backend": llm_backend,
                "config": {"model_name_or_path": llm_model,
                           "api_key": key, "api_base": api_base,
                           "temperature": 0.0},
            },
            vector_db={
                "backend": "qdrant",
                "config": {"collection_name": f"fe_{uuid.uuid4().hex[:10]}",
                           "vector_dimension": vector_dim,
                           "distance_metric": "cosine",
                           "path": self._dir},
            },
            embedder={
                "backend": "sentence_transformer",
                "config": {"model_name_or_path": embedder_model,
                           "embedding_dims": vector_dim},
            },
        )
        self.m = GeneralTextMemory(cfg)
        self.top_k = 20

    # ── Protocol ─────────────────────────────────────────────────────

    def reset(self) -> None:
        try:
            ids = [it.id for it in self.m.get_all() if it.id]
        except Exception:
            ids = []
        if ids:
            try:
                self.m.delete(ids)
            except Exception:
                pass

    def inscribe(self, text: str) -> str:
        self.m.add([{"memory": text}])
        return ""

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        return [h.memory for h in self.m.search(query, top_k=k)][:k]

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self.m.search(old_query, top_k=1)
        if hits:
            self.m.update(hits[0].id, {"memory": new_text})
        else:
            self.inscribe(new_text)

    def release(self, query: str) -> int:
        raise NotImplementedError(
            "MemOS exposes no soft-delete / TTL primitive."
        )

    def purge(self, query: str) -> int:
        hits = self.m.search(query, top_k=self.top_k)
        if not hits:
            return 0
        # Same rule as the LangGraph comparator: MemOS does not surface a
        # uniform similarity score, so we take the top hit's text and
        # remove its exact duplicates rather than guessing a threshold.
        target = hits[0].memory
        ids = [h.id for h in hits if h.memory == target and h.id]
        if ids:
            self.m.delete(ids)
        return len(ids)


# ─── TencentDB Agent Memory adapter ───────────────────────────────────

class TencentDBAdapter:
    """TencentDB Agent Memory v2 (``TencentCloud/TencentDB-Agent-Memory``).

    Structurally the most distinctive system in the study: memory is a
    four-tier distillation pipeline (L0 raw conversation -> L1 atomic
    memory -> L2 scenario -> L3 persona), where each tier is produced
    from the one below it by an LLM.  That makes it an *abstraction
    cascade* built on summarisation rather than on a knowledge graph, and
    therefore an independent test of the mechanism we attribute the
    KG-regime failures to: if surface forms are shed by distillation
    generally, and not by graph construction specifically, this system
    should fail purge the same way Graphiti and HippoRAG do despite
    sharing none of their machinery.

    Two properties matter for the harness.  **(1) Extraction is
    asynchronous.**  A write to L0 returns before the L1 atom exists, so
    every read is preceded by a poll on ``/v2/pipeline/status`` until the
    L1 worker is idle; without that the store looks empty rather than
    wrong.  **(2) L1 atoms are rewritten, not copied** -- inscribing
    "User currently lives in Vienna." yields the atom "The user currently
    lives in Vienna." plus a separate ``background`` gloss.  Surface form
    is already not preserved at the tier the delete primitives address.

    Isolation is per-case via a fresh ``team_id`` rather than a global
    wipe, so cases cannot see each other and workers can share one
    container.

    *Deployment:* the official Docker image (``node:22-slim`` base),
    standalone mode, LLM pointed at an OpenAI-compatible endpoint::

        docker build -t tencentdb-agent-memory:latest MemoryCore/
        docker run -d -p 8420:8420 -e TDAI_GATEWAY_API_KEY=... \\
          -e TDAI_LLM_API_KEY=... -e TDAI_LLM_BASE_URL=... \\
          tencentdb-agent-memory:latest

    Note the shipped standalone config sets ``embedding.provider: none``
    and ``bm25.language: zh``; we leave retrieval at its BM25 default but
    set the language to ``en`` to match the suite, so the row measures
    the memory system rather than a Chinese tokenizer on English text.

    Primitive coverage:

        inscribe  |->  POST /v2/conversation/add        (L0)
        recall    |->  POST /v2/atomic/search           (L1)
        supersede |->  POST /v2/atomic/update           (native edit)
        release   |->  N/A: no soft-delete / TTL primitive
        purge     |->  POST /v2/atomic/delete
    """

    name = "tencentdb"

    def __init__(self, base_url: str = "http://127.0.0.1:8420",
                 api_key: str = "fe-bench-key",
                 service_id: str = "default",
                 pipeline_timeout: float = 90.0):
        import uuid
        self.base = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "x-tdai-service-id": service_id,
            "Content-Type": "application/json",
        }
        self.pipeline_timeout = pipeline_timeout
        self.user_id = "u1"
        self.agent_id = "a1"
        self.session_id = "s1"
        self.team_id = f"fe_{uuid.uuid4().hex[:12]}"
        self._pending = False

    # ── plumbing ─────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        import json
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return json.load(e)
            except Exception:
                return {"code": e.code, "message": str(e)}
        except Exception as e:
            return {"code": -1, "message": f"{type(e).__name__}: {e}"}

    def _ids(self) -> dict:
        return {"team_id": self.team_id, "user_id": self.user_id,
                "agent_id": self.agent_id}

    def _await_extraction(self) -> None:
        """L0 writes return before the L1 atom exists; read-after-write
        without this poll measures pipeline latency, not memory."""
        if not self._pending:
            return
        import time
        deadline = time.time() + self.pipeline_timeout
        while time.time() < deadline:
            st = self._post("/v2/pipeline/status",
                            {"team_id": self.team_id}).get("data") or {}
            l1 = st.get("l1") or {}
            if l1.get("idle", False) and not l1.get("running", 0) \
               and not l1.get("queued", 0):
                break
            time.sleep(1.0)
        # one extra beat: the atom is indexed just after the worker
        # reports idle
        time.sleep(1.0)
        self._pending = False

    def _search(self, query: str, k: int) -> list[dict]:
        self._await_extraction()
        out = self._post("/v2/atomic/search",
                         {**self._ids(), "query": query, "top_k": k})
        return ((out.get("data") or {}).get("items")) or []

    # ── Protocol ─────────────────────────────────────────────────────

    def reset(self) -> None:
        # Fresh namespace beats deletion: cheaper, exact, and it lets
        # several workers share one gateway.
        import uuid
        self.team_id = f"fe_{uuid.uuid4().hex[:12]}"
        self._pending = False

    def inscribe(self, text: str) -> str:
        self._post("/v2/conversation/add",
                   {**self._ids(), "session_id": self.session_id,
                    "messages": [{"role": "user", "content": text}]})
        self._pending = True
        return ""

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        return [it.get("content", "") for it in self._search(query, k)][:k]

    def supersede(self, old_query: str, new_text: str) -> None:
        hits = self._search(old_query, 1)
        if hits and hits[0].get("id"):
            self._post("/v2/atomic/update",
                       {**self._ids(), "id": hits[0]["id"],
                        "content": new_text})
        else:
            self.inscribe(new_text)

    def release(self, query: str) -> int:
        raise NotImplementedError(
            "TencentDB Agent Memory exposes no soft-delete / TTL primitive."
        )

    def purge(self, query: str) -> int:
        hits = self._search(query, 20)
        if not hits:
            return 0
        target = hits[0].get("content", "")
        ids = [h["id"] for h in hits
               if h.get("content", "") == target and h.get("id")]
        if ids:
            self._post("/v2/atomic/delete", {**self._ids(), "ids": ids})
        return len(ids)
