from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
import requests
import json
import uvicorn

app = FastAPI()

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = body["message"]

    def generate_stream(message):
        headers = {
            "Content-Type": "application/json"
        }
        resp = requests.post("http://localhost:11434/api/generate", json={"model": "genai-coder", "prompt": message, "stream": True}, headers=headers, stream=True)
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                yield chunk["response"]

    return StreamingResponse(generate_stream(message), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
