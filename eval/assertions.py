"""Static scoring: does an answer use the current SDK API surface or dead/legacy calls?

Ground truth derived from the pinned clones in data/raw/:
openai-python v2.45.0, anthropic-sdk-python v0.116.0, ollama-python v0.6.2.
Answers are never executed; scoring is ast.parse + regex only.
"""

import ast
import re
import textwrap

# --- per-SDK denylist, applied to every answer for that SDK -----------------
DENYLIST = {
    "openai": [
        # module-level resource classes removed in openai-python 1.0
        # (openai/lib/_old_api.py raises APIRemovedInV1 for each of these)
        r"openai\.ChatCompletion",
        r"openai\.Completion\b",
        r"openai\.Embedding\b",
        r"openai\.Edit\b",
        r"openai\.Image\b",
        r"openai\.Audio\b",
        r"openai\.Moderation\b",
        r"openai\.FineTune",
        # same dead classes reached via `from openai import ChatCompletion`;
        # case-sensitive, so current `client.chat.completions.create` is unaffected
        r"ChatCompletion\.a?create",
        r"\bCompletion\.a?create",
        r"\bEmbedding\.a?create",
        r"openai\.error\b",           # exceptions moved to top level in 1.0
        r"openai\.api_base",          # renamed to base_url in 1.0
        r"\bengine\s*=\s*[\"']",      # engine= kwarg removed in 1.0; use model=
    ],
    "anthropic": [
        # Text Completions era (Claude 2). Still shipped in v0.116 but legacy;
        # never the right answer for the Messages-API tasks in this benchmark.
        r"\bHUMAN_PROMPT",
        r"\bAI_PROMPT",
        r"max_tokens_to_sample",
        r"\.completions\.create\s*\(",
        r"\.completion\s*\(",         # pre-0.3 client.completion() method
    ],
    "ollama": [
        r"\bembeddings\s*\(",         # deprecated in favor of embed()
    ],
}

# --- current-API patterns per (sdk, task): must match at least one ----------
POSITIVE = {
    ("openai", "chat"): [r"\.chat\.completions\.create\s*\(", r"\.responses\.create\s*\("],
    ("openai", "streaming"): [
        r"\.delta\.content",
        r"\.responses\.stream\s*\(",
        r"(?s)\.create\s*\(.*stream\s*=\s*True",
    ],
    ("openai", "responses"): [r"\.responses\.(create|parse|stream)\s*\("],
    ("openai", "tools"): [r"tools\s*=", r"pydantic_function_tool", r"\.tool_calls"],
    ("openai", "structured"): [r"\.parse\s*\(", r"response_format\s*=", r"text_format\s*=", r"json_schema"],
    ("openai", "embeddings"): [r"\.embeddings\.create\s*\("],
    ("openai", "async"): [r"AsyncOpenAI"],
    ("openai", "vision"): [r"image_url", r"input_image"],

    ("anthropic", "chat"): [r"\.messages\.(create|stream)\s*\("],
    ("anthropic", "streaming"): [
        r"\.messages\.stream\s*\(",
        r"text_stream",
        r"(?s)\.messages\.create\s*\(.*stream\s*=\s*True",
    ],
    ("anthropic", "tools"): [r"input_schema", r"tool_use"],
    ("anthropic", "structured"): [r"\.messages\.(create|parse)\s*\(", r"output_format\s*="],
    ("anthropic", "vision"): [r"[\"']type[\"']\s*:\s*[\"']image[\"']", r"media_type"],
    ("anthropic", "async"): [r"AsyncAnthropic"],
    ("anthropic", "thinking"): [r"thinking\s*=", r"budget_tokens"],
    ("anthropic", "tokens"): [r"\.count_tokens\s*\("],

    ("ollama", "chat"): [r"ollama\.chat\s*\(", r"from ollama import[^\n]*\bchat\b", r"\.chat\s*\("],
    ("ollama", "streaming"): [r"(?s)\bchat\s*\(.*stream\s*=\s*True", r"(?s)\bgenerate\s*\(.*stream\s*=\s*True"],
    ("ollama", "generate"): [r"\bgenerate\s*\("],
    ("ollama", "structured"): [r"format\s*=", r"model_json_schema"],
    ("ollama", "embeddings"): [r"\bembed\s*\(", r"ollama\.embed\b"],
    ("ollama", "tools"): [r"tools\s*=", r"tool_calls"],
    ("ollama", "async"): [r"AsyncClient"],
}

# --- extra negatives for specific tasks --------------------------------------
TASK_NEGATIVE = {
    # 2023-era function calling; repo docs: "Deprecated in favor of tools/tool_choice"
    ("openai", "tools"): [r"functions\s*=", r"function_call\s*="],
    # old client.count_tokens("text"); current is client.messages.count_tokens(model=..., messages=...)
    ("anthropic", "tokens"): [r"\.count_tokens\s*\(\s*[\"']"],
}

# per-prompt-id overrides: {"<id>": {"positive": [...], "negative": [...]}}
BY_ID = {}  # intentionally empty extension point: per-prompt {"positive": [...], "negative": [...]} overrides, merged by score()

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)(?:```|\Z)", re.S)


def extract_code(answer_text):
    """Return python code from fenced blocks (``` / ```python, unterminated ok),
    skipping blocks tagged with another language. No fences -> whole text."""
    blocks = []
    for lang, body in _FENCE.findall(answer_text):
        if lang.strip().lower() in ("", "python", "py"):
            body = textwrap.dedent(body).strip()
            if body:
                blocks.append(body)
    if blocks:
        return "\n\n".join(blocks)
    return answer_text.strip()


_compiled = {}


def _search(pattern, text):
    if pattern not in _compiled:
        _compiled[pattern] = re.compile(pattern)
    return _compiled[pattern].search(text)


# compile everything at import time so a bad pattern fails immediately
for _plist in list(DENYLIST.values()) + list(POSITIVE.values()) + list(TASK_NEGATIVE.values()):
    for _p in _plist:
        _search(_p, "")


def score(answer_text, prompt_record):
    """Score one model answer against its prompt record (needs id/sdk/task keys)."""
    code = extract_code(answer_text)
    sdk, task = prompt_record["sdk"], prompt_record["task"]
    try:
        ast.parse(code)
        syntax_ok = True
    except (SyntaxError, ValueError):
        syntax_ok = False

    extra = BY_ID.get(prompt_record["id"], {})
    positives = POSITIVE[(sdk, task)] + extra.get("positive", [])
    negatives = DENYLIST[sdk] + TASK_NEGATIVE.get((sdk, task), []) + extra.get("negative", [])

    positive_hit = any(_search(p, code) for p in positives)
    negative_hits = [p for p in negatives if _search(p, code)]
    return {
        "passed": syntax_ok and positive_hit and not negative_hits,
        "syntax_ok": syntax_ok,
        "positive_hit": positive_hit,
        "negative_hits": negative_hits,
    }
