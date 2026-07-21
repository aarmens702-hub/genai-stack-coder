# Model card: genai-coder

A QLoRA fine-tune of **Qwen2.5-Coder-7B-Instruct** that writes *current* generative-AI
application code — modern OpenAI / Anthropic / Ollama SDK usage — instead of the
2023-era APIs that base models hallucinate. Trained on one consumer GPU in one
overnight run.

## Summary

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` (Qwen2.5-Coder-7B-Instruct, Apache-2.0) |
| Method | QLoRA — rank 16, alpha 32, dropout 0, all 7 attention/MLP target modules |
| Data | 6,665 question→code pairs; **answers are 100% human-written** current example code from MIT/Apache repos; questions backtranslated by a local model |
| Training | 748 steps (2 epochs), lr 2e-4 cosine, effective batch 16, bf16, seq len 2048 |
| Hardware | 1× RTX A2000 12 GB, native Windows, ~5 h wall, peak 9.79 GB VRAM |
| Loss | train 0.59 final; val 1.038 → 0.955 (best, epoch 0.8) → 0.985 (epoch 2.0) |
| Distribution | LoRA adapter (155 MB safetensors) in `models/adapter/final/` via git-lfs; GGUF Q4_K_M (4.7 GB) rebuildable — see "Rebuilding the GGUF" |
| License | Adapter: Apache-2.0 (matches base). Repo code: MIT. Data licenses: `docs/LICENSES.md` |

## Intended use

Answering "write me code for X" where X is generative-AI application glue: chat
completions, streaming, tool calling, structured output, embeddings, vision
inputs — against the current OpenAI (`OpenAI()` client / Responses API),
Anthropic (`messages.create` / `messages.stream`), and Ollama
(`chat` / `generate` / `embed`) Python SDKs. It is a **code generator, not a
general assistant** — it was trained on nothing but code answers and will answer
almost anything with code unless steered (the demo app routes conversational
messages to a plain-prose prompt for this reason).

## Evaluation

50-prompt SDK-currency benchmark (`eval/`): each prompt asks for code against
one SDK; automated scoring requires a syntactically valid answer using at least
one current API surface and zero deprecated ones. Both models serve at Q4_K_M
via Ollama, temperature 0.2, identical system prompt.

| model | openai | anthropic | ollama | total |
|---|---|---|---|---|
| qwen2.5-coder:7b (base) | 1/20 | 1/20 | 4/10 | **6/50 (12%)** |
| genai-coder | 14/20 | 15/20 | 7/10 | **36/50 (72%)** |

A re-run four days later reproduced 35/50 (70%) — stable within temp-0.2
sampling variance. Failure analysis of the re-run: **zero answers used a
deprecated API** (the trained-against behavior is gone); most misses are
valid modern code that the pattern-based scorer's positive lists are too
narrow to credit — a scorer audit found the honest score is ~76%
(see `docs/NOTES.md`, 2026-07-21).

## Known limitations

- **Ollama is the weakest SDK** (3% of training data): occasionally grafts
  Anthropic idioms (`.text_stream`) or invents surfaces (`Embedding` class).
- Occasionally hallucinates plausible-but-nonexistent beta surfaces
  (`conversations.thread`, agents-SDK grafts) on tool-heavy prompts.
- 7B at Q4: small-model syntax slips happen (~2/50 on the benchmark).
- Knowledge is frozen at harvest time (July 2026); as SDKs evolve, re-harvest
  and retrain — or pair with retrieval (planned).

## Training data & licensing rules

Sources (exact refs in `docs/LICENSES.md`): openai-cookbook, anthropic-cookbook,
openai-python, anthropic-sdk-python, ollama-python — examples/tests only.
Rules enforced by the pipeline (`scripts/01–04`): answers must be human-written
code from MIT/Apache repos; a local open model (qwen2.5-coder:7b via Ollama)
generated *questions only*; no GPT/Claude outputs anywhere in the dataset;
deprecated-API answers filtered out (347 dropped); secrets scanned (6 dropped);
deduped, openai share capped at 60%. Final split 5,973 train / 348 val / 344
test, grouped by source snippet.

## Rebuilding the GGUF from the adapter

The repo ships the adapter only (the 4.7 GB GGUF and 15 GB intermediates are
derivable):

1. `python packaging/merge_lora.py` — merges `models/adapter/final` into the
   base model (downloads base from Hugging Face on first run) → `models/merged`
2. `python packaging/make_ollama.py` — converts to f16 GGUF (llama.cpp
   `convert_hf_to_gguf.py`) and creates the Ollama model with Q4_K_M
   quantization done inside the Ollama server (`packaging/create_via_api.py`
   is the kill-resilient fallback)
3. `ollama run genai-coder`

## Provenance

Built 2026-07-14 → 2026-07-16 on a single Windows 11 machine without admin
rights. Full engineering log with every decision, failure, and fix:
`docs/NOTES.md`. Benchmark methodology and results files: `eval/`.
