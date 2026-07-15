# Engineering notes

Running log of decisions and findings. Newest first.

## 2026-07-15 — Dataset built: 6,665 pairs

- Pair generation: 14,192 raw pairs from 3,190 snippets in ~3.7h (0.24 snippets/s on the A2000). Bonus: Ollama's schema enforcement doesn't cap array length, so the teacher often returned >2 questions per snippet — free augmentation, kept.
- Filter results: 6,665 kept. Drops: 5,420 openai-share rebalance, 1,455 dup questions, **347 deprecated-API answers** (the filter catching legacy cookbook code — exactly why it exists), 297 non-parsing, 6 secrets, 5 length.
- Final mix: 60% openai / 37% anthropic / 3% ollama; 26% template / 74% teacher questions. Split grouped by snippet: 5,973 train / 348 val / 344 test.
- 20-pair audit: PASS. Answers dense in current surfaces (responses API, vector_stores.search, messages.stream). Known blemishes: some template questions grammatically clunky; a few teacher questions generic-Python. Acceptable noise.
- Derived data (data/processed) now gitignored — rebuilt deterministically by scripts 01–04 from pinned refs.

## 2026-07-14 — Phase 0 smoke test PASSED + baseline evidence captured

- 4-bit Qwen2.5-Coder-7B loads and generates on the A2000: **peak VRAM 5.74 GB** of 12 — ample training headroom. Unsloth banner confirms full wiring: CUDA 8.6, bf16 TRUE, xformers 0.0.34, Triton 3.6.
- **Golden baseline sample:** asked to "stream a chat response from the OpenAI API", the *base* model produced `openai.api_key = ...` + `openai.ChatCompletion.create(stream=True)` — the exact API removed in openai-python 1.0 (Nov 2023). The project's premise, demonstrated on prompt #1. Keep this as the canonical "before" example in the README.
- Confirms the eval gate: base model measurably fails SDK-currency prompts → real signal to improve.

## 2026-07-14 — Windows dependency trap (hit + resolved)

- Installing per the original recipe (torch 2.8 → triton → xformers → unsloth) broke the env twice in one pass: latest xformers (0.0.35) silently **upgraded torch to 2.11+cu128**, then unsloth (which caps `torch<2.11`) **downgraded torch to 2.10.0 from PyPI — a CPU-only build on Windows**. Net result: no CUDA.
- Fix: pin the coherent set from unsloth's own extras table (`cu128onlytorch2100`): torch **2.10.0+cu128** + xformers **0.0.34** (installed `--no-deps` so it can't touch torch) + triton-windows **3.6.0.post26** (3.6.x pairs with torch 2.10). torchaudio removed (never needed).
- Rule: on Windows, never let pip resolve torch transitively — PyPI torch wheels are CPU-only. Full pinned set lives in `requirements-train.txt`.

## 2026-07-14 — Project start

- Plan approved: QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct into a "GenAI-stack coder" (correct, current OpenAI/Anthropic/Ollama SDK code).
- Verified machine: RTX A2000 12GB (Ampere, 70W), 32GB RAM, Python 3.12, CUDA driver 595.95, nvcc 12.3. **No admin rights** → native Windows, per-user installs only; WSL2 off the table.
- Verified toolchain (July 2026): Unsloth works natively on Windows now — triton-windows is an official triton-lang project, bitsandbytes ≥0.47 ships Windows wheels, xformers cu128 wheels on the PyTorch index. Pin: torch 2.8/cu128 + triton-windows <3.5.
- Fallback if Triton misbehaves: HF Transformers + PEFT + TRL + bitsandbytes (no Triton/xformers), ~2× slower.
- Base model check: Qwen3-Coder line is MoE-only (30B-A3B+) — untrainable on 12GB. Qwen2.5-Coder-7B remains the right base.
- Licensing rule for training data: answers must be human-written code from MIT/Apache repos (cookbooks, SDK examples). Local open models may generate *questions only* (backtranslation). No Claude/GPT outputs anywhere in the dataset.
- Agent harness will use a ReAct/JSON text protocol instead of native tool-calling — fine-tuning + Q4 quantization can degrade tool-call token templates; plain text is robust.
