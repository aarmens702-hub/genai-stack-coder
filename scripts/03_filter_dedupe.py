"""Phase 3a: filter raw pairs into a clean training set.

Gates, in order (drop reasons counted and printed):
  1. question sanity (length bounds)
  2. answer code must parse (ast.parse; notebook cells with top-level await
     get one retry wrapped in an async def)
  3. secret scan
  4. deprecated-API denylist — legacy code from old cookbook corners must
     never become a training answer in a project whose whole point is
     API currency
  5. near-duplicate questions for the same snippet
  6. SDK balance: cap the openai share at 60% (seeded downsample)

Output records are chat-format: {"messages": [system, user, assistant],
"meta": {...}} with the assistant answer as a fenced code block.

Run:  .venv\\Scripts\\python.exe scripts\\03_filter_dedupe.py
"""

import ast
import json
import random
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = ROOT / "data" / "processed" / "pairs.raw.jsonl"
OUT_JSONL = ROOT / "data" / "processed" / "dataset.jsonl"
SYSTEM_PROMPT = (ROOT / "scripts" / "system_prompt.txt").read_text(encoding="utf-8").strip()

OPENAI_SHARE_CAP = 0.60
SEED = 42

SECRET_RES = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
]

# Deprecated / dead API surface per SDK. A pair whose ANSWER matches any of
# these is dropped — we never train on the disease we're curing.
DEPRECATED = {
    "openai": [
        re.compile(r"openai\.ChatCompletion"),
        re.compile(r"openai\.Completion\b"),
        re.compile(r"openai\.Embedding\b"),
        re.compile(r"openai\.api_key\s*="),
        re.compile(r"\bengine\s*="),
        re.compile(r"text-davinci|code-davinci|text-curie|text-babbage|text-ada-001"),
    ],
    "anthropic": [
        re.compile(r"HUMAN_PROMPT|AI_PROMPT"),
        re.compile(r"anthropic\.Client\("),
        re.compile(r"client\.completions?\.create"),
        re.compile(r"claude-instant|claude-v?1\b|claude-2\b"),
    ],
    "ollama": [],
}


def code_parses(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        pass
    try:  # notebook cells may use top-level await
        ast.parse("async def _cell():\n" + textwrap.indent(code, "    "))
        return True
    except SyntaxError:
        return False


def norm_question(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def main():
    pairs = [json.loads(l) for l in IN_JSONL.open(encoding="utf-8")]
    drops = Counter()
    kept, seen_questions = [], set()
    parse_cache = {}

    for p in pairs:
        q = p["question"].strip()
        if not 20 <= len(q) <= 400:
            drops["question_length"] += 1
            continue
        sid = p["snippet_id"]
        if sid not in parse_cache:
            parse_cache[sid] = code_parses(p["code"])
        if not parse_cache[sid]:
            drops["code_no_parse"] += 1
            continue
        if any(r.search(p["code"]) for r in SECRET_RES):
            drops["secret"] += 1
            continue
        if any(r.search(p["code"]) for r in DEPRECATED.get(p["sdk"], [])):
            drops["deprecated_api"] += 1
            continue
        qkey = (sid, norm_question(q))
        if qkey in seen_questions:
            drops["dup_question"] += 1
            continue
        seen_questions.add(qkey)
        kept.append(p)

    # SDK balance: cap openai share
    rng = random.Random(SEED)
    openai_pairs = [p for p in kept if p["sdk"] == "openai"]
    others = [p for p in kept if p["sdk"] != "openai"]
    max_openai = int(len(others) / (1 - OPENAI_SHARE_CAP) * OPENAI_SHARE_CAP)
    if len(openai_pairs) > max_openai:
        drops["balance_downsample"] += len(openai_pairs) - max_openai
        openai_pairs = rng.sample(openai_pairs, max_openai)
    final = openai_pairs + others
    rng.shuffle(final)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for p in final:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["question"].strip()},
                    {"role": "assistant", "content": "```python\n" + p["code"] + "\n```"},
                ],
                "meta": {k: p[k] for k in ("snippet_id", "sdk", "source", "origin", "license")},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"in: {len(pairs)}  kept: {len(final)}")
    print("drops:", dict(drops))
    print("by sdk:", dict(Counter(p["sdk"] for p in final)))
    print("by origin:", dict(Counter(p["origin"] for p in final)))
    print("by source:", dict(Counter(p["source"] for p in final)))
    print(f"-> {OUT_JSONL.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
