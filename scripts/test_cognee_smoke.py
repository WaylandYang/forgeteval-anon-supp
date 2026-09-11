"""Smoke test Cognee with DeepSeek-V3 via SiliconFlow.

If this passes (inscribe a fact, query, recall it back), we can wire up
the adversarial run on a 365-case Forget-Eval adversarial layer.
"""
import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["CACHING"] = "false"
# LLM via SiliconFlow OpenAI-compatible endpoint
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "openai/deepseek-ai/DeepSeek-V3"
os.environ["LLM_ENDPOINT"] = "https://api.siliconflow.cn/v1"
os.environ["LLM_API_KEY"] = ""
# Embeddings: use fastembed locally (no tokenizer mismatch)
os.environ["EMBEDDING_PROVIDER"] = "fastembed"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["EMBEDDING_DIMENSIONS"] = "384"

import cognee  # noqa: E402


async def main():
    print("=== Cognee 1.0.9 smoke test ===")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Pruned.")
    r = await cognee.remember("Alice has email alice@example.com.")
    print("Remember 1 ok:", type(r).__name__)
    r = await cognee.remember("Bob plays the piano.")
    print("Remember 2 ok:", type(r).__name__)
    res = await cognee.recall("What is Alice's email?")
    print("Recall result type:", type(res).__name__)
    print("Recall returned:", res)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

