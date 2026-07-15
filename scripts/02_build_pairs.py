"""Phase 2: build instruct pairs from harvested snippets.

For every snippet, produce up to 3 training pairs that share the same answer
(the real, human-written code):
  1. a deterministic template question from the snippet's heading/filename
  2. up to 2 natural developer questions written by a LOCAL open model
     (qwen2.5-coder via Ollama) — backtranslation. The teacher writes only
     the QUESTION; the answer is always the harvested code, never generated.

Licensing: no closed-model outputs anywhere. Teacher is Apache-2.0 Qwen.

Progressive + resumable: appends to the output as it goes and skips snippets
already processed, so an interrupted run continues where it left off.

Run:  .venv\\Scripts\\python.exe scripts\\02_build_pairs.py
"""

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = ROOT / "data" / "processed" / "snippets.raw.jsonl"
OUT_JSONL = ROOT / "data" / "processed" / "pairs.raw.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
TEACHER = "qwen2.5-coder:7b"
N_TEACHER_QUESTIONS = 2

SDK_NAME = {
    "openai": "OpenAI Python SDK",
    "anthropic": "Anthropic Python SDK",
    "ollama": "Ollama Python library",
}

TEACHER_SYSTEM = (
    "You write realistic developer questions. Given a code snippet, write "
    "questions that a developer would ask an AI coding assistant, where this "
    "exact code is a correct and complete answer. Mention the SDK/library "
    "when it matters. Vary phrasing and specificity. Output JSON only."
)

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": N_TEACHER_QUESTIONS,
        }
    },
    "required": ["questions"],
}


def humanize(snippet):
    """Turn heading or filename into a rough task description."""
    text = snippet["heading"] or Path(snippet["path"]).stem
    text = re.sub(r"^[\d.\s]+", "", text)          # strip "3.2 " numbering
    text = re.sub(r"[_\-]+", " ", text).strip()
    return text[0].lower() + text[1:] if text else "do this"


def template_question(snippet):
    sdk = SDK_NAME.get(snippet["sdk"], "Python")
    return f"Show me how to {humanize(snippet)} using the {sdk}."


def teacher_questions(snippet, retries=2):
    prompt = (
        f"SDK: {SDK_NAME.get(snippet['sdk'], 'Python')}\n"
        f"Context: {snippet['heading'] or snippet['path']}\n\n"
        f"Code:\n```python\n{snippet['code'][:3000]}\n```\n\n"
        f"Write {N_TEACHER_QUESTIONS} distinct developer questions this code answers."
    )
    body = {
        "model": TEACHER,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "format": QUESTIONS_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.8, "num_ctx": 8192, "num_predict": 300},
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=body, timeout=300)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            qs = json.loads(content)["questions"]
            return [q.strip() for q in qs if isinstance(q, str) and len(q.strip()) > 15]
        except Exception as e:  # noqa: BLE001 — log and retry, never crash the run
            if attempt == retries:
                print(f"    teacher failed for {snippet['id']}: {e}", flush=True)
                return []
            time.sleep(2)
    return []


def make_pair(snippet, question, origin):
    return {
        "snippet_id": snippet["id"],
        "origin": origin,
        "sdk": snippet["sdk"],
        "source": snippet["source"],
        "license": snippet["license"],
        "question": question,
        "code": snippet["code"],
    }


def main():
    snippets = [json.loads(l) for l in IN_JSONL.open(encoding="utf-8")]
    done = set()
    if OUT_JSONL.exists():
        done = {json.loads(l)["snippet_id"] for l in OUT_JSONL.open(encoding="utf-8")}
        print(f"resuming: {len(done)} snippets already processed")

    todo = [s for s in snippets if s["id"] not in done]
    start = time.time()
    with OUT_JSONL.open("a", encoding="utf-8") as out:
        for i, snippet in enumerate(todo):
            pairs = [make_pair(snippet, template_question(snippet), "template")]
            for q in teacher_questions(snippet):
                pairs.append(make_pair(snippet, q, "teacher"))
            for p in pairs:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - start)
                eta_h = (len(todo) - i - 1) / rate / 3600 if rate else 0
                print(f"{i + 1}/{len(todo)} snippets | {rate:.2f}/s | ETA {eta_h:.1f}h", flush=True)

    total = sum(1 for _ in OUT_JSONL.open(encoding="utf-8"))
    print(f"DONE: {total} pairs -> {OUT_JSONL.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
