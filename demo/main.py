import json
import os
import subprocess
import sys
import time

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI()

@app.get("/")
async def get_index(request: Request):
    return FileResponse("index.html")

@app.post("/chat")
async def post_chat(request: Request):
    body = await request.json()
    messages = body["messages"]
    
    response = requests.post("http://localhost:11434/api/chat", json={"model": "genai-coder", "messages": messages, "stream": True}, stream=True)

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
    task = body["task"]

    slug = "".join(c if c.isalnum() else "-" for c in task[:30]).strip("-").lower() or "task"
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
        finally:
            if proc.poll() is None:
                proc.kill()

    return StreamingResponse(run_agent_stream(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
