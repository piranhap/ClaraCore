#!/usr/bin/env python3
"""
Hermes orchestrator loop for ClaraCore (raw OpenAI-compatible API, stdlib only).

Hermes runs on the Ada (GPU1). When it decides a sub-task suits a worker, it
emits a strict JSON delegation directive; this script parses it, calls that
worker model id on ClaraCore (which lazy-loads it on the 3060 / GPU0 and runs
it *concurrently* with Hermes), then feeds the result back for synthesis.

Why a text protocol instead of OpenAI `tools`: the Hermes-3 GGUF's chat
template doesn't render the `tools` array, so llama-server never emits
structured `tool_calls`. A fenced-JSON directive is reliable and needs no
fragile per-model --chat-template override. (jinja is enabled, which is what
makes the chat templates / this behavior work.)

Usage:  python3 hermes_orchestrate.py "your request here"
"""
import json, re, sys, urllib.request

BASE = "http://localhost:5890/v1/chat/completions"
ORCHESTRATOR = "hermes-3-llama-31-8bq8-0-8b"

# Worker roster. PREFER the GPU0 workers — they run alongside Hermes (Ada/GPU1).
# The "heavy" tensor-split models use BOTH GPUs and therefore PAUSE Hermes while
# they run, so only delegate to them when their extra capability is worth it.
SYSTEM_PROMPT = """You are Hermes, an orchestrator on a local multi-GPU server (ClaraCore).
You run on GPU1 (RTX 2000 Ada). You can DELEGATE a sub-task to a specialist
worker model that runs CONCURRENTLY on GPU0 (RTX 3060).

To delegate, reply with ONLY a fenced json block and nothing else:
```json
{"action":"delegate","model_id":"<worker id>","task":"<full self-contained instruction>"}
```
The "task" must be self-contained — the worker sees only that string, not this
conversation.

Concurrent workers (GPU0 — run alongside you, prefer these):
- qwen35-9b-q6-k-9b            Qwen3.5 9B (Q6_K), 196K ctx — coding, reasoning, general. DEFAULT worker.
- qwen35-9b-ud-q6-k-xl-9b      Qwen3.5 9B (higher-quality quant), 196K ctx.
- qwen35-9b-9b                 Qwen3.5 9B (Q8_0), 196K ctx — best quality, most VRAM.
- meta-llama-31-8b-instruct-q6-k-l-8b   Llama 3.1 8B Instruct, 131K ctx — general.
- gemma-4-e2b-it-bf16-2b       Gemma 4 E2B, 40K ctx — tiny/fast for cheap tasks.

Heavy specialists (tensor-split across BOTH GPUs — these PAUSE you while running):
- qwen36-27b-q6-k-7b, gemma-4-26b-a4b-it-ud-q6-k-4b, lfm2-24b-a2bq8-0-2b, deepseek-v2-liteq8-0

Embeddings: bge-large-en-v15q8-0 (use the /v1/embeddings endpoint, not chat).

If you can answer directly yourself, just answer normally (no json block).
After you receive a worker's result, give the user a clear final answer."""


def chat(model, messages, max_tokens=512, temperature=0.3):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(BASE, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_delegation(text):
    """Return the delegation dict if Hermes asked to delegate, else None."""
    m = re.search(r'\{[^{}]*"action"\s*:\s*"delegate"[^{}]*\}', text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if d.get("model_id") and d.get("task") else None
    except json.JSONDecodeError:
        return None


def run(user_request, max_hops=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_request}]
    for hop in range(max_hops):
        reply = chat(ORCHESTRATOR, messages)
        d = parse_delegation(reply)
        if not d:
            return reply  # Hermes answered directly / gave the final synthesis
        print(f"  ↳ delegating to {d['model_id']}: {d['task'][:80]}...", file=sys.stderr)
        worker_out = chat(d["model_id"], [{"role": "user", "content": d["task"]}])
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user",
             "content": f"Worker {d['model_id']} returned:\n{worker_out}\n\n"
                        f"Use this to give the user their final answer."},
        ]
    return chat(ORCHESTRATOR, messages)  # final synthesis after last hop


if __name__ == "__main__":
    request = " ".join(sys.argv[1:]) or "Get the coder to write a python function that reverses a string, then tell me what it returns for 'hello'."
    print(run(request))
