# SDK Benchmark Eval

50 prompts (20 openai / 20 anthropic / 10 ollama) measuring whether a model emits current GenAI SDK code
or deprecated calls. Ground truth: the pinned clones in `data\raw\` (openai v2.45.0, anthropic v0.116.0, ollama v0.6.2).

**Run** (needs Ollama serving on localhost:11434):
`.venv\Scripts\python.exe eval\run_eval.py --model qwen2.5-coder:7b [--limit 5] [--out path.jsonl]`
Per-prompt results go to `eval\results\<model>.jsonl`; a per-SDK/per-task pass-rate table is printed.

**Scoring** (`assertions.py`, static only -- answers never executed): a prompt passes iff its extracted
code parses (`ast.parse`), matches >=1 current-API regex in `POSITIVE[(sdk, task)]`, and matches none
of `DENYLIST[sdk]` + `TASK_NEGATIVE[(sdk, task)]`.

**Add a prompt**: append a JSON line (id, sdk, task, prompt) to `benchmark_prompts.jsonl`; make sure
`POSITIVE[(sdk, task)]` exists (or add a `BY_ID` override), then run `eval\test_assertions.py`.
