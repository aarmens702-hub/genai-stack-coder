import json
import os
import re
import subprocess
import sys
import time

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same system prompt as training and the benchmark, so chat behaves like the
# configuration the 12%->72% numbers were measured under.
with open(os.path.join(REPO, "scripts", "system_prompt.txt"), encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()

# For messages that are conversation, not code requests. Absolute wording on
# purpose: softer variants ("avoid code unless...") still yield code.
CHAT_PLAIN_PROMPT = (
    "You are a helpful assistant chatting with a user. Reply with short plain"
    " English sentences. Never output code, code blocks, or programming examples."
)

# Deterministic override: mentions of the product's domain always take the
# code path, because the 1-token classifier misses terse imperatives like
# "stream tokens from ollama".
SDK_HINT = re.compile(r"\b(openai|anthropic|claude|ollama|gpt|sdk|api)\b", re.IGNORECASE)


def wants_code(text):
    """True if the message asks for code: 1-token model classify + SDK-keyword
    override. Defaults to True (the product's specialty) on any failure."""
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "genai-coder",
                "messages": [
                    {"role": "system", "content": "You are a classifier. Answer with exactly one word: yes or no."},
                    {"role": "user", "content": "Does the following message ask for programming code, a script, API/SDK usage, or a technical implementation?\n\nMESSAGE: " + text},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 3},
            },
            timeout=30,
        )
        if r.json()["message"]["content"].strip().lower().startswith("yes"):
            return True
    except Exception:
        return True
    return bool(SDK_HINT.search(text))


# Appended to every Build task: the discipline rules learned from live runs.
TASK_EPILOGUE = """

RULES (always apply):
- Do not build interactive programs that wait for keyboard input; build CLIs
  that take command-line arguments instead.
- Before using done, run every verify command the task lists and confirm its
  expected output. If the task lists none, invent at least one command that
  proves the program works, and run it.
- To verify a web server, use ONLY the check_http tool. Starting a server with
  run_command hangs for 60 seconds and then fails -- never do it.
- Use port 8123 for any server you build (other ports may be blocked on this
  machine)."""


app = FastAPI()

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(REPO, "demo", "index.html"))

@app.post("/chat")
async def post_chat(request: Request):
    body = await request.json()
    last_user = next((m["content"] for m in reversed(body["messages"]) if m.get("role") == "user"), "")
    system = SYSTEM_PROMPT if wants_code(last_user) else CHAT_PLAIN_PROMPT
    messages = [{"role": "system", "content": system}] + body["messages"]
    
    response = requests.post("http://localhost:11434/api/chat", json={"model": "genai-coder", "messages": messages, "stream": True, "options": {"temperature": 0.2}}, stream=True, timeout=(5, 300))

    def stream_response(resp):
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                yield chunk["message"]["content"]
    
    return StreamingResponse(stream_response(response), media_type="text/plain")

@app.post("/agent")
async def agent(request: Request):
    """Build mode: run the coding-agent harness on a task, streaming its
    progress log; files land in a fresh workspace/<run>/ folder."""
    body = await request.json()
    task = body["task"] + TASK_EPILOGUE

    slug = "".join(c if c.isalnum() else "-" for c in body["task"][:30]).strip("-").lower() or "task"
    workdir = os.path.join(REPO, "workspace", time.strftime("%Y%m%d-%H%M%S") + "-" + slug)

    def run_agent_stream():
        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(REPO, "agent", "agent.py"),
             "--task", task, "--workdir", workdir, "--max-iters", "15"],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        try:
            yield "building in " + workdir + "\n\n"
            for line in proc.stdout:
                yield line
            proc.wait()
            status = "done" if proc.returncode == 0 else "stopped (exit %d)" % proc.returncode
            yield "\n[" + status + "] files are in " + workdir + "\n"
            listing = []
            for root, _dirs, files in os.walk(workdir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, workdir)
                    listing.append("  %s (%d bytes)" % (rel, os.path.getsize(full)))
            if listing:
                yield "files created:\n" + "\n".join(sorted(listing)) + "\n"
        finally:
            if proc.poll() is None:
                proc.kill()

    return StreamingResponse(run_agent_stream(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
