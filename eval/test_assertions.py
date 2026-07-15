"""Sanity checks: benchmark file is valid, and score() behaves on 3 fake answers."""

import json
import os

import assertions

HERE = os.path.dirname(os.path.abspath(__file__))


def test_prompts_file():
    with open(os.path.join(HERE, "benchmark_prompts.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 50, "expected 50 prompts, got %d" % len(records)
    assert len({r["id"] for r in records}) == 50, "duplicate ids"
    counts = {}
    for r in records:
        assert set(r) == {"id", "sdk", "task", "prompt"}, r["id"]
        assert r["sdk"] in ("openai", "anthropic", "ollama"), r["id"]
        assert (r["sdk"], r["task"]) in assertions.POSITIVE, "no POSITIVE entry for %s/%s" % (r["sdk"], r["task"])
        assert r["prompt"].strip(), r["id"]
        counts[r["sdk"]] = counts.get(r["sdk"], 0) + 1
    print("prompts ok: 50 lines, sdk counts =", counts)


CURRENT = """Here is how to do it:
```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain list comprehensions"}],
)
print(completion.choices[0].message.content)
```
"""

DEPRECATED = """```python
import openai

openai.api_key = "sk-..."
response = openai.ChatCompletion.create(
    engine="gpt-4",
    messages=[{"role": "user", "content": "Explain list comprehensions"}],
)
print(response["choices"][0]["message"]["content"])
```
"""

BROKEN = """```python
from openai import OpenAI
client = OpenAI(
completion = client.chat.completions.create(model="gpt-4o", messages=[)
print(completion
```
"""


def test_score():
    rec = {"id": "openai-01", "sdk": "openai", "task": "chat"}

    good = assertions.score(CURRENT, rec)
    print("current-API answer:   ", good)
    assert good == {"passed": True, "syntax_ok": True, "positive_hit": True, "negative_hits": []}

    dep = assertions.score(DEPRECATED, rec)
    print("deprecated-API answer:", dep)
    assert dep["syntax_ok"] is True
    assert dep["positive_hit"] is False
    assert dep["negative_hits"] and dep["passed"] is False

    broken = assertions.score(BROKEN, rec)
    print("syntax-error answer:  ", broken)
    assert broken["syntax_ok"] is False and broken["passed"] is False


if __name__ == "__main__":
    test_prompts_file()
    test_score()
    print("all checks passed")
