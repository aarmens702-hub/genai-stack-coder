# TASK: upgrade the streaming chat app — real UI + conversation memory

The working directory already contains `main.py` and `index.html` (you wrote
them). Upgrade both IN PLACE. Still exactly two files, no frameworks beyond
fastapi + uvicorn + requests, vanilla JavaScript only, no external libraries
or fonts.

## Step 1: rewrite main.py (add conversation memory)

Keep the same structure you have now (`GET /` returning
`FileResponse("index.html")`, the `if __name__ == "__main__":` uvicorn block
with host `127.0.0.1` port `8000`, and `async def chat(request: Request)` with
`body = await request.json()`). Rules:

- Use EXACTLY these imports and no others:
  `import json`, `import requests`, `import uvicorn`,
  `from fastapi import FastAPI, Request`,
  `from fastapi.responses import FileResponse, StreamingResponse`.
- Do NOT use httpx, aiohttp, async HTTP clients, StaticFiles, API keys, or
  environment variables. Ollama needs no authentication. Talk to it with
  `requests` exactly like the current main.py does.
- The request body is now `{"messages": [{"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}, ...]}` — the FULL conversation so
  far. Get it with `messages = body["messages"]`.
- POST to `http://localhost:11434/api/chat` (NOT /api/generate) with
  `json={"model": "genai-coder", "messages": messages, "stream": True}` using
  `requests.post(..., stream=True)`.
- Ollama replies with one JSON object per line like
  `{"message": {"role": "assistant", "content": "next chunk"}, "done": false}`.
  In a plain (NOT async) inner `def` generator, loop over `resp.iter_lines()`,
  skip empty lines, `chunk = json.loads(line.decode("utf-8"))`, and yield
  `chunk["message"]["content"]` — yield ONLY that string, never the raw line.
- Return `StreamingResponse(generator, media_type="text/plain")` as before.

## Step 2: rewrite index.html (make it look like a real chat app)

Layout — full viewport, flex column, inline CSS in one `<style>` block:

- Header bar at top: text `genai-coder` (bold) and a smaller subtitle
  `local 7B fine-tune, streaming via Ollama`.
- Middle: a scrollable messages area that fills all remaining height.
- Bottom: a fixed input row — copy this HTML exactly:
  `<div id="input-row"><input type="text" id="user-input" placeholder="Type a message..."><button id="send-button" onclick="sendMessage()">Send</button></div>`
  Your JavaScript must look elements up ONLY by these ids: `user-input`,
  `send-button`, and `messages` (the messages area div). Look each up ONCE at
  the top of the script into consts and reuse them — never call
  getElementById with any other id string.

Dark theme: page background `#0f1115`, default text `#e6e6e6`, header
background `#161a22`. Message bubbles with `border-radius: 12px`, padding
`8px 12px`, margin `6px 0`, `max-width: 75%`, `white-space: pre-wrap`:

- user messages: right-aligned (`margin-left: auto`), background `#2b5cd9`,
  white text.
- assistant messages: left-aligned (`margin-right: auto`), background
  `#1e2128`.

JavaScript:

- Keep a `messages` array of `{role, content}` objects — the whole
  conversation.
- On send: push `{role: "user", content: text}`, render the user bubble,
  clear the input, then `fetch("/chat", {method: "POST", headers:
  {"Content-Type": "application/json"}, body: JSON.stringify({messages})})`.
- Create an empty assistant bubble, read `response.body.getReader()` with a
  `TextDecoder`, append each decoded chunk to the bubble as it arrives, and
  scroll the messages area to the bottom after every chunk.
- When the stream ends, push `{role: "assistant", content: fullReply}` onto
  `messages`.
- Disable the input and button while streaming (`userInput.disabled = true;`
  `sendButton.disabled = true;` on the consts) and re-enable + focus the input
  in a `finally` block. Pressing Enter in the input sends. Add NO other event
  listeners.

## Step 3: verify (do NOT start the server — it would hang the 60s timeout)

1. Run: `python -c "import main"` — must exit 0. If it fails, read the error,
   fix, verify again.
2. Run: `if (Select-String -Path main.py -Pattern "api/chat" -Quiet) { "main.py OK" } else { throw "main.py is missing api/chat" }`
   — must print `main.py OK` with exit code 0.
3. Run: `if (Select-String -Path main.py -Pattern "requests.post" -Quiet) { "requests OK" } else { throw "main.py must call requests.post" }`
   — must print `requests OK` with exit code 0.
4. Run: `if (Select-String -Path main.py -Pattern "httpx" -Quiet) { throw "main.py must not use httpx" } else { "no httpx OK" }`
   — must print `no httpx OK` with exit code 0.
5. Run: `if (Select-String -Path index.html -Pattern "getReader" -Quiet) { "index.html OK" } else { throw "index.html is missing getReader" }`
   — must print `index.html OK` with exit code 0.
6. Run: `if (Select-String -Path index.html -Pattern 'id="send-button"' -Quiet) { "button id OK" } else { throw "index.html is missing the send-button id" }`
   — must print `button id OK` with exit code 0.

## Step 4: finish

Use the `done` tool. The message must say: start the server with
`python main.py`, then open `http://127.0.0.1:8000` in a browser.
