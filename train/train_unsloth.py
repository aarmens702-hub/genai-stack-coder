"""Phase 4: QLoRA fine-tune of Qwen2.5-Coder-7B on the GenAI-stack dataset.

Config lives right here as constants — one model, one machine, no yaml.
Two modes:
  --sanity   100 examples, 60 steps, no eval: proves the loss goes to ~0 and
             the stack doesn't OOM before we spend a night on the full run
  (default)  full 2-epoch run with periodic eval on the val split

Run:  .venv\\Scripts\\python.exe train\\train_unsloth.py [--sanity]
"""

from unsloth import FastLanguageModel  # must import before transformers/torch
from unsloth.chat_templates import train_on_responses_only

import argparse
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "adapter"

BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch 16
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="100-example overfit sanity run")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    data_files = {"train": str(DATA / "train.jsonl"), "val": str(DATA / "val.jsonl")}
    ds = load_dataset("json", data_files=data_files)

    def render(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = ds["train"].map(render, remove_columns=ds["train"].column_names)
    val_ds = ds["val"].map(render, remove_columns=ds["val"].column_names)
    if args.sanity:
        train_ds = train_ds.select(range(100))

    config = SFTConfig(
        output_dir=str(OUT),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        max_steps=60 if args.sanity else -1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.07,
        bf16=True,
        optim="adamw_8bit",
        seed=SEED,
        logging_steps=5 if args.sanity else 10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="no" if args.sanity else "steps",
        eval_steps=100,
        per_device_eval_batch_size=2,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=None if args.sanity else val_ds.select(range(min(200, len(val_ds)))),
        args=config,
    )
    # Mask everything except assistant responses so loss is on answers only.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Unattended-run insurance: resume from the newest checkpoint after a
    # crash/forced logoff. Delete models/adapter/checkpoint-* for a fresh run.
    ckpts = [] if args.sanity else sorted(OUT.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if ckpts else None)
    print(f"final loss: {result.training_loss:.4f}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if not args.sanity:
        model.save_pretrained(str(OUT / "final"))
        tokenizer.save_pretrained(str(OUT / "final"))
        print(f"adapter saved -> {OUT / 'final'}")


if __name__ == "__main__":
    main()
