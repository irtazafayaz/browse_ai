import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.dependencies import get_embedder, get_qdrant
from api.routes.search import router as search_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_embedder()
    get_qdrant()
    yield


app = FastAPI(title="Browse AI", version="1.0.0", lifespan=lifespan)

app.include_router(search_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
