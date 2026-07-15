"""Phase 0 smoke test: load Qwen2.5-Coder-7B in 4-bit on the GPU and generate.

Exercises the full Windows QLoRA stack (torch cu128 + triton-windows + xformers
+ bitsandbytes) end-to-end before any real training run, and prints peak VRAM
so we know the headroom on the 12GB card.

Run:  .venv\\Scripts\\python.exe scripts\\00_smoke_gpu.py
"""

from unsloth import FastLanguageModel  # must import before torch/transformers
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{
    "role": "user",
    "content": "Write a Python function that streams a chat response from the OpenAI API.",
}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=120)

print("=" * 60)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
print("=" * 60)
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
