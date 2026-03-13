from fastapi import FastAPI

app = FastAPI(name="langgraph-ai-agent")

@app.get("/health")
async def health_check():
    return {"status" : "ok"}