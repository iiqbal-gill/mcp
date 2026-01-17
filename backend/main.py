from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import process_query
import uvicorn

app = FastAPI(title="UET AI Agent API")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    API Endpoint as per Step 5 requirements.
    """
    response_text = process_query(request.message)
    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)