import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_embedder, get_clip_embedder, get_qdrant, get_collection
from api.routes.search import router as search_router
from api.routes.status import router as status_router

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# The httpx logger prints every Qdrant HTTP call — noise. Keep our own
# search diagnostics readable by raising its threshold.
logging.getLogger("httpx").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm heavy singletons + ensure Mongo indexes once, before serving traffic.
    get_embedder()
    get_clip_embedder()
    get_collection()

    # Validate/create the Qdrant collection at startup so a schema mismatch
    # (e.g. a legacy single-vector collection) surfaces here, not mid-request.
    # Logged rather than fatal so a transient Qdrant blip can't block boot —
    # /api/v1/status will report it as degraded.
    try:
        get_qdrant().ensure_collection()
    except Exception as e:
        logger.error("Qdrant collection check failed at startup: %s", e)

    yield


tags_metadata = [
    {"name": "search", "description": "Text, cross-modal (text→image) and image→image product search."},
    {"name": "system", "description": "Health, readiness and status dashboard."},
]

app = FastAPI(
    title="Browse AI",
    version="1.0.0",
    description="Semantic + image search over Pakistani fashion products.",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

# CORS — the dashboard and any browser frontend call these endpoints directly.
_origins = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/api/v1")
app.include_router(status_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
def health():
    """Liveness — the process is up. Does not touch DBs (use /api/v1/status for readiness)."""
    return {"status": "ok"}
