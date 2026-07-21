# GenAI-Stack-Coder

Fine-tuning **Qwen2.5-Coder-7B** on a single RTX A2000 12GB (native Windows, no admin rights) to do one thing better than models 100× its size: **write correct, current generative-AI application code** — modern OpenAI / Anthropic / Ollama SDK usage, streaming, tool calling, structured output.

**Why:** every base model hallucinates 2023-era SDK calls (`openai.ChatCompletion.create`, anyone?). This model is QLoRA-trained exclusively on current, human-written, permissively-licensed example code so it doesn't.

**Finale:** the tuned model, driven by the minimal agent harness in this repo, vibe-codes a streaming chat app *powered by itself* via Ollama.

## Pipeline

```
data harvest → instruct pairs → filter/dedupe → QLoRA train → eval (SDK-currency benchmark)
→ merge → GGUF Q4_K_M → Ollama → agent harness → demo app (written by the model)
```

## Status

- [x] Plan + repo scaffold
- [x] Phase 0 — environment bootstrap (torch 2.10 cu128, Unsloth 2026.7.2, Ollama — all verified on GPU)
- [x] Phases 1–3 — data pipeline (3,190 snippets → 14,195 raw pairs → 6,665 clean: 5,973 train / 348 val / 344 test)
- [x] Phase 4 — QLoRA training (748 steps / 2 epochs, 4h57m on the A2000, final train loss 0.59)
- [x] Phase 5 — eval: base vs tuned SDK-currency pass rates (**12% → 72%**, table below)
- [x] Phase 6 — package: GGUF Q4_K_M → `ollama run genai-coder`
- [x] Phase 7 — agent harness (ReAct/JSON protocol with fenced file writes; `agent/`)
- [x] Phase 8 — finale: the model built `demo/` itself in 7 agent iterations (below)

## Results

50-prompt SDK-currency benchmark; both models served by Ollama at Q4_K_M, temp 0.2, identical system prompt (`eval/run_eval.py`).

| SDK | base qwen2.5-coder:7b | genai-coder (tuned) |
|---|---|---|
| openai | 1/20 (5%) | **14/20 (70%)** |
| anthropic | 1/20 (5%) | **15/20 (75%)** |
| ollama | 4/10 (40%) | **7/10 (70%)** |
| **total** | **6/50 (12%)** | **36/50 (72%)** |

Canonical "before" example — the base model answers a streaming question with `openai.ChatCompletion.create(stream=True)`, removed from the SDK in Nov 2023; the tuned model instantiates the current `OpenAI()` client and streams with the `client.chat.completions.stream(...)` helper.

## Finale

Driven by `agent/agent.py`, **genai-coder wrote `demo/main.py` and `demo/index.html` itself** (7 iterations: two fenced file writes, two verify commands, done) — a FastAPI backend that proxies Ollama's `/api/generate` as a `StreamingResponse`, and a vanilla-JS page that renders the stream via `getReader()`. Verified end-to-end: the reply arrives as ~per-token chunks, served by the very model that wrote the code.

```
cd demo && python main.py   # then open http://127.0.0.1:8000
```

**Chat mode** streams answers with proper code blocks (copy button on each) and has a Stop button mid-generation. Messages are intent-routed: a 1-token classifier call (plus a deterministic SDK-keyword override) decides whether you're asking for code — code questions run under the same system prompt the benchmark used; conversational messages run under a plain-English prompt, so asking "what's my favorite number" gets a sentence back instead of a FastAPI app.

**Build mode:** flip the selector next to the input from *Chat* to *Build* and describe something small (template skeletons are in the dropdown). Instead of replying with code to copy-paste, the model runs the agent harness server-side — you watch its write/run/fix loop stream into the chat, and the finished files land in `workspace/<run>/` (listed at the end of each run). Web apps are verified for real: the harness's `check_http` tool starts the server, probes the URL, reports the response, and shuts it down — the model uses it instead of just grepping its own output. Every Build task gets discipline rules appended automatically (no interactive `input()` programs, verify before done, use port 8123 — others are blocked on this box). Phrase tasks like a spec, not a wish: name the file, the exact command shape and expected output, and the verify commands to run — and prefer CLI-args programs over interactive `input()` apps, which the harness's non-interactive runner cannot verify. ("Can you make a calculator app" yields a stub; "calc.py where `python calc.py 2 + 3` prints 5, `5 / 0` prints an error and exits 1 — verify each" yields working, edge-case-tested software.) (The chat UI stays model-written per above; the Build-mode wiring — the `/agent` endpoint and mode toggle — is human-written infrastructure.)

It took 6 attempts to get there; the failures (and the harness fixes they forced) are catalogued in `docs/NOTES.md` — all of them agent-skill issues (JSON-escaping of file bodies, misread tracebacks), none of them SDK-currency issues. The app's FastAPI/requests/Ollama surfaces were current and correct on the first try.

**v2** (current): the model then upgraded its own app — full-viewport dark chat UI with message bubbles, and real conversation memory (the full history flows through Ollama's `/api/chat`, so follow-ups have context). Four more harness rounds, catalogued in `docs/NOTES.md`; the model's final output was ~95% right and shipped with a disclosed **~15-line human patch** (sync generator + `stream=True`, one growing reply bubble instead of bubble-per-chunk, viewport flex). Ask it a follow-up question and it remembers — and being a code model, it usually answers in Python: asked for a remembered number, it replies `print(42)`.

## Layout

| Path | What |
|---|---|
| `scripts/` | data harvest → pairs → filter → split |
| `train/` | QLoRA configs + training scripts (Unsloth primary, HF/TRL fallback) |
| `eval/` | SDK-currency benchmark + scoring |
| `packaging/` | LoRA merge → GGUF → Ollama Modelfile |
| `agent/` | minimal coding-agent harness (ReAct/JSON protocol) |
| `demo/` | the app the tuned model builds itself |
| `docs/` | engineering notes, licenses manifest, model card |

## Hardware

RTX A2000 12GB · 32GB RAM · Windows 11 (per-user installs only — no admin) · Python 3.12
