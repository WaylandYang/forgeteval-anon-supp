"""Integration shim for A-MEM's LLM controller.

A-MEM requests structured output with ``response_format={"type":
"json_schema", ...}``. Against DeepSeek-V4 via OpenRouter that call
intermittently returns a message with ``content=None`` -- most often on
the memory-evolution prompt, whose schema is the more complex of the two
A-MEM uses. A-MEM catches the resulting exception and continues, so the
run completes while its distinctive mechanism (note linking and
evolution) is silently disabled: 12 of 15 cases hit it in a probe run.
A number produced that way is not a measurement of A-MEM.

The shim retries once with ``{"type": "json_object"}``, which the same
model handles reliably, and reports how often it had to. It changes the
request envelope, never the prompt or the parsing, so what is measured is
still A-MEM's algorithm. This is the same class of intervention as the
json-repair pass documented for Mem0's extraction parser.

Apply after `pip install -e` of the A-MEM clone:

  python scripts/patch_amem_json.py /path/to/A-mem
"""
from __future__ import annotations

import sys
from pathlib import Path

SHIM = '''
    # --- ForgetEval integration shim (see scripts/patch_amem_json.py) ---
    _fe_stats = {"calls": 0, "fallback": 0, "empty": 0}

    def get_completion(self, prompt: str, response_format: dict,
                       temperature: float = 0.7) -> str:
        schema = ((response_format or {}).get("json_schema") or {}).get("schema") or {}
        required = schema.get("required") or []
        attempts = [(response_format, prompt)]
        if (response_format or {}).get("type") == "json_schema":
            # json_object mode does not enforce the schema, so the required
            # keys have to travel in the prompt or the caller gets a
            # KeyError instead of a parse error -- same failure, later.
            # Two attempts: a hint, then the schema itself spelled out.
            hint = ("\\n\\nRespond with a single JSON object containing "
                    "exactly these keys: " + ", ".join(required) + ".")
            attempts.append(({"type": "json_object"}, prompt + hint))
            strict = (hint + "\\nEvery listed key must be present. "
                      "The JSON schema is:\\n"
                      + __import__("json").dumps(schema))
            attempts.append(({"type": "json_object"}, prompt + strict))
        for fmt, text in attempts:
            OpenAIController._fe_stats["calls"] += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You must respond with a JSON object."},
                        {"role": "user", "content": text},
                    ],
                    response_format=fmt,
                    temperature=temperature,
                    # A-MEM's own 1000 truncates the evolution response,
                    # whose schema nests two arrays of arrays.
                    max_tokens=3000,
                )
                content = response.choices[0].message.content
            except Exception:
                content = None
            if content and (not required
                            or all(('"%s"' % k) in content for k in required)):
                if fmt is not attempts[0][0]:
                    OpenAIController._fe_stats["fallback"] += 1
                return content
        OpenAIController._fe_stats["empty"] += 1
        return "{}"
'''


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    target = root / "agentic_memory" / "llm_controller.py"
    src = target.read_text(encoding="utf-8")

    if "_fe_stats" in src:
        print("already patched")
        return

    start = src.index("    def get_completion", src.index("class OpenAIController"))
    end = src.index("class OllamaController")
    patched = src[:start] + SHIM.lstrip("\n") + "\n" + src[end:]
    target.write_text(patched, encoding="utf-8")
    print(f"patched {target}")


if __name__ == "__main__":
    main()
