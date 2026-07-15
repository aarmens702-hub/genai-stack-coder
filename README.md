# GenAI-Stack-Coder

Fine-tuning **Qwen2.5-Coder-7B** on a single RTX A2000 12GB (native Windows, no admin rights) to do one thing better than models 100× its size: **write correct, current generative-AI application code** — modern OpenAI / Anthropic / Ollama SDK usage, streaming, tool calling, RAG.

**Why:** every base model hallucinates 2023-era SDK calls (`openai.ChatCompletion.create`, anyone?). This model is QLoRA-trained exclusively on current, human-written, permissively-licensed example code so it doesn't.

**Finale:** the tuned model, driven by the minimal agent harness in this repo, vibe-codes a streaming chat app *powered by itself* via Ollama.

## Pipeline

```
data harvest → instruct pairs → filter/dedupe → QLoRA train → eval (SDK-currency benchmark)
→ merge → GGUF Q4_K_M → Ollama → agent harness → demo app (written by the model)
```

## Status

- [x] Plan + repo scaffold
- [ ] Phase 0 — environment bootstrap (torch cu128, Unsloth, Ollama)
- [ ] Phases 1–3 — data pipeline (~8–12k pairs from MIT/Apache sources)
- [ ] Phase 4 — QLoRA training (overnight run)
- [ ] Phase 5 — eval: base vs tuned SDK-currency pass rates
- [ ] Phase 6 — package: GGUF → `ollama run genai-coder`
- [ ] Phase 7 — agent harness
- [ ] Phase 8 — finale: model builds its own chat app
- [ ] Results table here

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
