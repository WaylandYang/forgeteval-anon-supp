"""Tolerant wrappers for Graphiti against SiliconFlow.

- TolerantOpenAIGenericClient: fixes JSON-shape mismatches from
  non-OpenAI proxied LLMs (DeepSeek, Kimi, etc).
- CohereRerankerClient: uses SiliconFlow's Cohere-format /v1/rerank
  endpoint with BAAI/bge-reranker-v2-m3 instead of trying to coerce
  an LLM into reranker behaviour via chat completions.

Common issues:
  - Pydantic expects {'entity_resolutions': [obj]} but model returns
    {<obj fields>} (single object, not list-wrapped).
  - Pydantic expects {'extracted_entities': [...]} but model returns
    {'entities': [...]}.
  - Pydantic expects {'edges': [...]} but model returns single edge dict.

The fixer detects "the response would have validated had we
wrapped/aliased it" and applies a heuristic correction before raising.
"""
from __future__ import annotations

import json
import typing
from typing import Any

from pydantic import BaseModel, ValidationError

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message


# Common single-list field aliases we observe in practice.
LIST_ALIASES = {
    "extracted_entities": ["entities"],
    "extracted_edges": ["edges"],
    "entity_resolutions": ["resolutions", "duplicates"],
    "edges": ["facts", "extracted_edges"],
}


def fix_response_shape(
    raw: dict[str, Any], response_model: type[BaseModel] | None
) -> dict[str, Any]:
    """If `raw` doesn't validate against `response_model`, try a few
    heuristic transformations and return the first that validates.
    If nothing works, return `raw` unchanged (caller will raise the
    original error).
    """
    if response_model is None:
        return raw
    # If already valid, no fix needed.
    try:
        response_model(**raw)
        return raw
    except ValidationError:
        pass

    # Discover which list field the model expects.
    expected_list_fields: list[str] = []
    for fname, finfo in response_model.model_fields.items():
        ann = finfo.annotation
        # Cheap structural test: annotation contains 'list['
        if "list[" in str(ann).lower() or "List[" in str(ann):
            expected_list_fields.append(fname)

    candidates: list[dict[str, Any]] = []

    # Heuristic 1: wrap the entire response in {field: [raw]} for each
    # expected list field. Useful when the model returned a single
    # NodeDuplicate-shaped dict instead of {'entity_resolutions': [d]}.
    for fname in expected_list_fields:
        candidates.append({fname: [raw]})
        # If raw already has the list under a wrong alias, rename.
        for alias in LIST_ALIASES.get(fname, []):
            if alias in raw and isinstance(raw[alias], list):
                fixed = dict(raw)
                fixed[fname] = fixed.pop(alias)
                candidates.append(fixed)

    # Heuristic 2: empty-list initialisation for the expected list field
    # (last resort — accepts the response but produces no extracted data).
    for fname in expected_list_fields:
        candidates.append({fname: []})

    for cand in candidates:
        try:
            response_model(**cand)
            return cand
        except ValidationError:
            continue

    return raw  # let caller raise the original error


def first_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from `text`.

    Used to recover from LLM outputs that emit multiple top-level
    objects ('{...}\\n{...}'), trailing prose, or both. Returns the
    parsed dict; raises if no valid JSON object can be found.
    """
    text = text.strip()
    # Strip a leading ```json fence if present.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    # Walk to find the first {...} balanced span.
    start = text.find("{")
    if start < 0:
        return json.loads(text)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    # If we get here, no balanced object found — let json.loads raise.
    return json.loads(text)


class TolerantOpenAIGenericClient(OpenAIGenericClient):
    """OpenAIGenericClient with two post-processing layers:
    1. tolerant JSON decoder (extract first balanced object);
    2. response-shape fixer (wrap/alias when Pydantic validation fails).
    """

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # Re-do the API call here so we can intercept the raw text before
        # the parent's strict json.loads.
        from openai.types.chat import ChatCompletionMessageParam  # noqa: F401
        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})

        response_format: dict[str, Any] = {"type": "json_object"}
        if response_model is not None:
            schema_name = getattr(response_model, "__name__", "structured_response")
            json_schema = response_model.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            }

        from graphiti_core.llm_client.openai_generic_client import DEFAULT_MODEL  # noqa
        completion = await self.client.chat.completions.create(
            model=self.model or DEFAULT_MODEL,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,  # type: ignore[arg-type]
        )
        result_text = completion.choices[0].message.content or ""
        try:
            raw = first_json_object(result_text)
        except json.JSONDecodeError:
            # Last-ditch: empty dict, shape-fixer will then return {} or {"field": []}
            raw = {}
        return fix_response_shape(raw, response_model)


# ─── Cohere-format reranker over SiliconFlow ─────────────────────────

import httpx  # noqa: E402

from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402


class CohereRerankerClient(CrossEncoderClient):
    """SiliconFlow exposes a Cohere-format /v1/rerank endpoint that hosts
    BAAI/bge-reranker-v2-m3 (and other rerankers).  This is faster and
    cheaper than the OpenAIRerankerClient (which abuses an LLM chat
    completion to rank passages).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def rank(
        self, query: str, passages: list[str]
    ) -> list[tuple[str, float]]:
        if not passages:
            return []
        # SiliconFlow accepts Cohere's {model, query, documents, top_n}.
        async with httpx.AsyncClient(timeout=30.0) as cx:
            r = await cx.post(
                f"{self.base_url}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": passages,
                    "top_n": len(passages),
                    "return_documents": False,
                },
            )
            r.raise_for_status()
            payload = r.json()
        out: list[tuple[str, float]] = []
        for item in payload.get("results", []):
            idx = item["index"]
            score = float(item.get("relevance_score", 0.0))
            if 0 <= idx < len(passages):
                out.append((passages[idx], score))
        return out
