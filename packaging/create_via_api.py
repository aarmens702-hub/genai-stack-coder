"""Create genai-coder via Ollama's /api/create using the already-uploaded blob.

Client does no bulk I/O: the 15 GB f16 blob is referenced by digest and all
parsing/quantization happens inside the Ollama server process.
"""

import json
import re
import sys

import requests

MODELFILE = r"C:\Users\asa618\documents\genai-stack-coder\packaging\Modelfile"
DIGEST = "sha256:0a202ae7e992e63b397bcde12383ee498632dc99e6f8c3960d717309cae84893"

text = open(MODELFILE, encoding="utf-8").read()
template = re.search(r'TEMPLATE """(.*?)"""', text, re.S).group(1)

params = {}
for key, val in re.findall(r'(?m)^PARAMETER (\S+) (.+)$', text):
    val = val.strip().strip('"')
    if key == "stop":
        params.setdefault("stop", []).append(val)
    else:
        try:
            params[key] = json.loads(val)
        except ValueError:
            params[key] = val

body = {
    "model": "genai-coder",
    "files": {"genai-coder-f16.gguf": DIGEST},
    "template": template,
    "quantize": "q4_K_M",
}
if params:
    body["parameters"] = params

print("POST /api/create:", json.dumps({k: (v if k != "template" else "<%d chars>" % len(v)) for k, v in body.items()}))
resp = requests.post("http://localhost:11434/api/create", json=body, stream=True, timeout=3600)
resp.raise_for_status()
last = ""
for line in resp.iter_lines():
    if not line:
        continue
    msg = json.loads(line)
    status = msg.get("status") or msg.get("error") or str(msg)
    if status != last:
        print(status, flush=True)
        last = status
    if msg.get("error"):
        sys.exit("create failed: " + msg["error"])
    if msg.get("status") == "success":
        print("CREATE OK")
        sys.exit(0)
sys.exit("stream ended without success")
