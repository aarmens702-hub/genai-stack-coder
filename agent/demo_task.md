# TASK: minimal streaming chat web app

Build a streaming chat web app in the working directory. Create EXACTLY two
files: `main.py` and `index.html`. No other files, no frameworks beyond
fastapi + uvicorn, no build tools.

## Step 1: write main.py (FastAPI backend)

- Imports — use exactly these and no others:
  `from fastapi import FastAPI, Request`,
  `from fastapi.responses import FileResponse, StreamingResponse`,
  plus `import requests`, `import json`, `import uvicorn`.
- `GET /` — copy these two lines exactly:
  ```python
  @app.get("/")
  async def serve_index():
  ```
  and return `FileResponse("index.html")` from it.
- `POST /chat` — copy these two lines exactly:
  ```python
  @app.post("/chat")
  async def chat(request: Request):
  ```
  (it MUST be `async def` because it awaits the body). First line of the body:
  `body = await request.json()`; the user message is `body["message"]`.
  Stream plain text back:
  1. POST to `http://localhost:11434/api/generate` with
     `json={"model": "genai-coder", "prompt": message, "stream": True}` using
     `requests.post(..., stream=True)`.
  2. Ollama replies with one JSON object per line, like
     `{"response": "next chunk", "done": false}`.
  3. Return `StreamingResponse(generator, media_type="text/plain")` where the
     generator is a plain inner `def` (NOT async) that loops over
     `resp.iter_lines()`, parses each line with `json.loads`, and yields the
     `"response"` value.
- At the bottom add:
  `if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=8000)`

## Step 2: write index.html (vanilla JavaScript, no libraries)

- A chat log `<div id="log">`, a text `<input id="msg">`, and a Send `<button>`.
- On Send:
  1. Append the user's message to the log.
  2. `fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({message: text})})`
  3. Create an empty reply element in the log, then read the response body with
     `response.body.getReader()` and a `TextDecoder`, appending each decoded
     chunk to the reply element as it arrives (so tokens appear live).
- Pressing Enter in the input also sends. Minimal inline CSS is fine.

## Step 3: verify (do NOT start the server — it would hang the 60s timeout)

1. Run: `python -c "import main"` — must exit with code 0. If it fails, read
   the error, fix main.py, and verify again.
2. Run: `Select-String -Path index.html -Pattern "getReader"` — must print a match.

## Step 4: finish

Use the `done` tool. The message must say: start the server with
`python main.py`, then open `http://127.0.0.1:8000` in a browser.
