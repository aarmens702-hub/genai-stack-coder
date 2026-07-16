"""Run the SDK benchmark against a model served by local Ollama.

Usage: python eval/run_eval.py --model qwen2.5-coder:7b [--limit N] [--out results.jsonl]
"""

import argparse
import json
import os
import re
import sys

import requests

import assertions

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "benchmark_prompts.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"

# Same system prompt as training (scripts/system_prompt.txt) so base and
# tuned models are scored under identical conditions.
with open(os.path.join(HERE, "..", "scripts", "system_prompt.txt"), encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()


def query(model, prompt):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 800},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def print_summary(results):
    def rate(rows):
        n = len(rows)
        p = sum(r["passed"] for r in rows)
        return "%d/%d (%.0f%%)" % (p, n, 100.0 * p / n) if n else "-"

    print("\n%-11s %-11s %s" % ("sdk", "task", "pass rate"))
    print("-" * 36)
    for sdk in ("openai", "anthropic", "ollama"):
        sdk_rows = [r for r in results if r["sdk"] == sdk]
        if not sdk_rows:
            continue
        for task in sorted({r["task"] for r in sdk_rows}):
            print("%-11s %-11s %s" % (sdk, task, rate([r for r in sdk_rows if r["task"] == task])))
        print("%-11s %-11s %s" % (sdk, "ALL", rate(sdk_rows)))
        print("-" * 36)
    print("%-11s %-11s %s" % ("TOTAL", "", rate(results)))


def main():
    ap = argparse.ArgumentParser(description="Score a model on the current-vs-deprecated SDK benchmark.")
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. qwen2.5-coder:7b")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N prompts")
    ap.add_argument("--out", default=None, help="results jsonl path (default: eval/results/<model>.jsonl)")
    args = ap.parse_args()

    with open(PROMPTS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    out_path = args.out or os.path.join(
        HERE, "results", re.sub(r"[^A-Za-z0-9._-]", "_", args.model) + ".jsonl"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Resume support: keep rows already scored so a killed run continues
    # where it stopped instead of starting over.
    results = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        done = {r["id"] for r in results}
        records = [r for r in records if r["id"] not in done]
        if done:
            print("resuming: %d already scored, %d to go" % (len(done), len(records)))

    with open(out_path, "a", encoding="utf-8") as out:
        for i, rec in enumerate(records, 1):
            try:
                answer = query(args.model, rec["prompt"])
            except requests.RequestException as e:
                sys.exit("ollama request failed on %s: %s" % (rec["id"], e))
            s = assertions.score(answer, rec)
            row = {"id": rec["id"], "sdk": rec["sdk"], "task": rec["task"], **s, "answer": answer}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            results.append(row)
            print("[%d/%d] %s: %s" % (i, len(records), rec["id"], "PASS" if s["passed"] else "FAIL"))

    print_summary(results)
    print("\nresults written to %s" % out_path)


if __name__ == "__main__":
    main()
