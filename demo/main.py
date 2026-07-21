import json
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse


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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
