"""Phase 6b: merged fp16 -> GGUF f16 -> Q4_K_M -> local Ollama model.

Uses the vendored llama.cpp convert script (llama.cpp-src) and the prebuilt
llama-quantize.exe (llama.cpp-bin). Reuses the chat template of the stock
qwen2.5-coder:7b Ollama model so the tuned model is served under the exact
conditions the baseline was scored in.

Steps are skipped if their output already exists, so the script is safe to
re-run after a partial failure (delete models/gguf/* for a full rebuild).

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
Q4 = GGUF_DIR / "genai-coder-Q4_K_M.gguf"
CONVERT = ROOT / "llama.cpp-src" / "convert_hf_to_gguf.py"
QUANTIZE = ROOT / "llama.cpp-bin" / "llama-quantize.exe"
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

if not Q4.exists():
    run([QUANTIZE, F16, Q4, "Q4_K_M"])
else:
    print(f"skip quantize: {Q4} exists")

# Modelfile: stock qwen2.5-coder template/params, our GGUF weights.
show = subprocess.run(
    ["ollama", "show", "--modelfile", "qwen2.5-coder:7b"],
    check=True, capture_output=True, text=True, encoding="utf-8",
).stdout
modelfile = re.sub(r"(?m)^FROM .*$", f"FROM {Q4}", show, count=1)
mf_path = ROOT / "packaging" / "Modelfile"
mf_path.write_text(modelfile, encoding="utf-8")
run(["ollama", "create", MODEL_NAME, "-f", mf_path])
print(f"done: ollama run {MODEL_NAME}")
