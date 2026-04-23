"""
Browse AI — FastAPI server.
browse-ai-backend calls this instead of BrowseBy API.

Run:
    uvicorn api.main:app --reload --port 8001
"""
from fastapi import FastAPI
from api.routes.search import router as search_router

app = FastAPI(title="Browse AI", version="1.0.0")

app.include_router(search_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
