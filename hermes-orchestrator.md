# Hermes orchestrator on ClaraCore

Hermes (`hermes-3-llama-31-8bq8-0-8b`) runs on **GPU1 (RTX 2000 Ada, 16 GB)** and
acts as an orchestrator: it delegates sub-tasks to worker models that ClaraCore
lazy-loads on **GPU0 (RTX 3060, 12 GB)** and runs **concurrently** with Hermes.

ClaraCore is just an OpenAI-compatible router — the orchestration loop (decide →
call worker → feed back) lives in *your* client. A runnable reference is
[`hermes_orchestrate.py`](hermes_orchestrate.py); this doc explains the protocol
and shows the raw `curl` calls.

## How delegation works

Hermes's GGUF chat template doesn't render OpenAI `tools`, so it won't emit
structured `tool_calls`. Instead we use a **text protocol**: the system prompt
tells Hermes to reply with *only* a fenced JSON block when it wants to delegate:

```json
{"action":"delegate","model_id":"qwen35-9b-q6-k-9b","task":"<self-contained instruction>"}
```

Your client parses that, calls the worker's model id, then feeds the result back
to Hermes for a final answer. The full system prompt is in `hermes_orchestrate.py`
(`SYSTEM_PROMPT`).

> Requires `enableJinja: true` (now set). Without jinja the chat templates don't
> apply and behavior degrades.

## Worker roster

**Concurrent workers — GPU0, run alongside Hermes (prefer these):**

| model id | what | ctx |
|---|---|---|
| `qwen35-9b-q6-k-9b` | Qwen3.5 9B Q6_K — coding/reasoning/general (**default**) | 196K |
| `qwen35-9b-ud-q6-k-xl-9b` | Qwen3.5 9B, higher-quality quant | 196K |
| `qwen35-9b-9b` | Qwen3.5 9B Q8_0 — best quality, most VRAM | 196K |
| `meta-llama-31-8b-instruct-q6-k-l-8b` | Llama 3.1 8B Instruct — general | 131K |
| `gemma-4-e2b-it-bf16-2b` | Gemma 4 E2B — tiny/fast | 40K |
| `bge-large-en-v15q8-0` | embeddings — use `/v1/embeddings` | — |

**Heavy specialists — tensor-split across BOTH GPUs, so they PAUSE Hermes while
they run:** `qwen36-27b-q6-k-7b`, `gemma-4-26b-a4b-it-ud-q6-k-4b`,
`lfm2-24b-a2bq8-0-2b`, `deepseek-v2-liteq8-0`.

**On Hermes's own GPU (Ada):** `gemma-3n-e4b-it-ud-q6-k-xl-4b` — won't fit
alongside Hermes (12.6 GB) on the 16 GB card, so delegating to it evicts Hermes.

## Raw curl walkthrough

**1. Ask Hermes (it returns a delegation directive):**

```bash
curl -s http://localhost:5890/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "hermes-3-llama-31-8bq8-0-8b",
  "messages": [
    {"role":"system","content":"<SYSTEM_PROMPT from hermes_orchestrate.py>"},
    {"role":"user","content":"Get the coder to write a python function that reverses a string."}
  ],
  "max_tokens": 256, "temperature": 0
}'
# -> assistant content: {"action":"delegate","model_id":"qwen35-9b-q6-k-9b","task":"..."}
```

**2. Run the worker** (lazy-loads on GPU0, ~20–40 s cold start, then concurrent):

```bash
curl -s http://localhost:5890/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen35-9b-q6-k-9b",
  "messages": [{"role":"user","content":"<the task string from step 1>"}],
  "max_tokens": 512
}'
```

**3. Feed the worker's output back to Hermes** as a new user turn for synthesis
(append the prior assistant message + the worker result, then call Hermes again).

Verified live: Hermes (GPU1, ~12.6 GB) and a Qwen worker (GPU0, ~8.8 GB) were
resident **simultaneously**, and Hermes synthesized the worker's result.

## Notes

- **Cold start** ~20–40 s per model; idle models unload after `ttl: 300 s`.
- Pins are managed in the UI: **Configuration → GPU Pinning** (multi-GPU only).
- The roster IDs are this deployment's models (`/v1/models` to list). Update the
  `SYSTEM_PROMPT` roster if you add/remove models.
