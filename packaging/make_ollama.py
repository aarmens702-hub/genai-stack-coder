"""Phase 6b: merged fp16 -> GGUF f16 -> Q4_K_M inside Ollama -> local model.

Uses the vendored llama.cpp convert script (llama.cpp-src) for the f16 GGUF,
then `ollama create --quantize q4_K_M` so quantization runs inside the Ollama
service — standalone llama-quantize.exe gets killed mid-run on this
locked-down box. Reuses the chat template of the stock qwen2.5-coder:7b
Ollama model so the tuned model is served under the exact conditions the
baseline was scored in.

The convert step is skipped if its output already exists, so the script is
safe to re-run after a partial failure (delete models/gguf/* for a rebuild).

Run:  .venv\\Scripts\\python.exe packaging\\make_ollama.py
"""

import os
import re
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "models" / "merged"
GGUF_DIR = ROOT / "models" / "gguf"
F16 = GGUF_DIR / "genai-coder-f16.gguf"
CONVERT = ROOT / "llama.cpp-src" / "convert_hf_to_gguf.py"
MODEL_NAME = "genai-coder"


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


GGUF_DIR.mkdir(parents=True, exist_ok=True)

if not F16.exists():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "llama.cpp-src" / "gguf-py"))
    run([sys.executable, CONVERT, MERGED, "--outfile", F16, "--outtype", "f16"], env=env)
else:
    print(f"skip convert: {F16} exists")

# Modelfile: stock qwen2.5-coder template/params, our GGUF weights.
show = subprocess.run(
    ["ollama", "show", "--modelfile", "qwen2.5-coder:7b"],
    check=True, capture_output=True, text=True, encoding="utf-8",
).stdout
# repl as a function so backslashes in the Windows path aren't parsed as escapes
modelfile = re.sub(r"(?m)^FROM .*$", lambda _: f"FROM {F16}", show, count=1)
mf_path = ROOT / "packaging" / "Modelfile"
mf_path.write_text(modelfile, encoding="utf-8")
run(["ollama", "create", MODEL_NAME, "--quantize", "q4_K_M", "-f", mf_path])
print(f"done: ollama run {MODEL_NAME}")
