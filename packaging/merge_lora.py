"""Phase 6a: merge the trained LoRA adapter into a 16-bit model for GGUF export.

Loads the 4-bit base + adapter from models/adapter/final, dequantizes and
merges to fp16 at models/merged/. Needs ~16 GB free RAM and ~16 GB disk.

Run:  .venv\\Scripts\\python.exe packaging\\merge_lora.py
"""

from unsloth import FastLanguageModel  # must import before transformers/torch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models" / "adapter" / "final"
OUT = ROOT / "models" / "merged"

model, tokenizer = FastLanguageModel.from_pretrained(
    str(ADAPTER), max_seq_length=2048, load_in_4bit=True
)
model.save_pretrained_merged(str(OUT), tokenizer, save_method="merged_16bit")
print(f"merged 16-bit model -> {OUT}")
