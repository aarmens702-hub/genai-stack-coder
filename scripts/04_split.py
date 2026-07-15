"""Phase 3b: split the dataset 90/5/5 — grouped by snippet.

All pairs sharing a snippet (same answer code, different questions) land in
the same split, so the val/test answers are never seen in training.

Run:  .venv\\Scripts\\python.exe scripts\\04_split.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = ROOT / "data" / "processed" / "dataset.jsonl"
OUT_DIR = ROOT / "data" / "processed"
SEED = 42
FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}


def main():
    by_snippet = defaultdict(list)
    for line in IN_JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        by_snippet[r["meta"]["snippet_id"]].append(r)

    groups = sorted(by_snippet)
    random.Random(SEED).shuffle(groups)

    n = len(groups)
    n_train = int(n * FRACTIONS["train"])
    n_val = int(n * FRACTIONS["val"])
    splits = {
        "train": groups[:n_train],
        "val": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }

    for name, split_groups in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for g in split_groups:
                for r in by_snippet[g]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    count += 1
        print(f"{name}: {len(split_groups)} snippets, {count} pairs -> {path.name}")


if __name__ == "__main__":
    sys.exit(main())
