"""An OpenAI-compatible /v1/embeddings endpoint backed by fastembed.

Letta stores archival memory through an embedding provider and the repo's
runner was configured against a hosted one whose key was scrubbed for
release. Rather than require a credential to reproduce the Letta row,
this serves the same model the rest of the benchmark already uses
(all-MiniLM-L6-v2, 384-dim) over the OpenAI wire format.

Using the benchmark's own embedder for Letta is also the fairer choice:
it removes one more axis on which the Letta row differed from the others.

    python scripts/local_embeddings_server.py --port 8399

From inside a container, reach it at http://host.docker.internal:8399/v1.
"""
from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastembed import TextEmbedding
from pydantic import BaseModel
import uvicorn

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

app = FastAPI()
_embedder = TextEmbedding(MODEL)


class Req(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.post("/v1/embeddings")
@app.post("/embeddings")
def embeddings(req: Req):
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vecs = [list(map(float, v)) for v in _embedder.embed(texts)]
    return {
        "object": "list",
        "model": req.model or MODEL,
        "data": [{"object": "embedding", "index": i, "embedding": v}
                 for i, v in enumerate(vecs)],
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts),
                  "total_tokens": sum(len(t.split()) for t in texts)},
    }


@app.get("/v1/models")
@app.get("/models")
def models():
    return {"object": "list",
            "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "dim": DIM}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8399)
    a = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
