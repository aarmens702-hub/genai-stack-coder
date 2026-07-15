# Engineering notes

Running log of decisions and findings. Newest first.

## 2026-07-14 — Project start

- Plan approved: QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct into a "GenAI-stack coder" (correct, current OpenAI/Anthropic/Ollama SDK code).
- Verified machine: RTX A2000 12GB (Ampere, 70W), 32GB RAM, Python 3.12, CUDA driver 595.95, nvcc 12.3. **No admin rights** → native Windows, per-user installs only; WSL2 off the table.
- Verified toolchain (July 2026): Unsloth works natively on Windows now — triton-windows is an official triton-lang project, bitsandbytes ≥0.47 ships Windows wheels, xformers cu128 wheels on the PyTorch index. Pin: torch 2.8/cu128 + triton-windows <3.5.
- Fallback if Triton misbehaves: HF Transformers + PEFT + TRL + bitsandbytes (no Triton/xformers), ~2× slower.
- Base model check: Qwen3-Coder line is MoE-only (30B-A3B+) — untrainable on 12GB. Qwen2.5-Coder-7B remains the right base.
- Licensing rule for training data: answers must be human-written code from MIT/Apache repos (cookbooks, SDK examples). Local open models may generate *questions only* (backtranslation). No Claude/GPT outputs anywhere in the dataset.
- Agent harness will use a ReAct/JSON text protocol instead of native tool-calling — fine-tuning + Q4 quantization can degrade tool-call token templates; plain text is robust.
