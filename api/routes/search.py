import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, File, UploadFile, HTTPException
from pydantic import BaseModel, ConfigDict
from pymongo.collection import Collection

from api.dependencies import get_embedder, get_clip_embedder, get_reranker, get_qdrant, get_collection
from vectorizer.embedder import Embedder
from vectorizer.clip_embedder import ClipEmbedder
from vectorizer.reranker import Reranker
from db.qdrant import Qdrant
from vectorizer.search import search, image_search, text_to_image_search

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


class ProductHit(BaseModel):
    # Payloads carry extra fields (mongo_id, tags, category, ...) — allow them.
    model_config = ConfigDict(extra="allow")
    brand: Optional[str] = None
    name: Optional[str] = None
    price: Optional[float] = None
    score: Optional[float] = None
    imageUrl: Optional[str] = None
    productUrl: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[ProductHit]
    total: int
    debug: Optional[dict] = None  # populated only when ?debug=true


def _clean_query(q: str) -> str:
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return q


def _guard(fn, what: str):
    """Run a search call, converting backend failures into a clean 503 instead
    of an opaque 500 stack trace."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("%s failed", what)
        raise HTTPException(status_code=503, detail=f"{what} temporarily unavailable.") from e


@router.get("/search", response_model=SearchResponse, tags=["search"])
def search_products(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(20, ge=1, le=100),
    brand: Optional[str] = Query(None),
    gender: Optional[str] = Query(None, description="Filter/boost by gender: women | men | kids"),
    debug: bool = Query(False, description="Include retrieval diagnostics (timings, parsed attrs, rerank scores)"),
    embedder: Embedder = Depends(get_embedder),
    qdrant: Qdrant = Depends(get_qdrant),
    col: Collection = Depends(get_collection),
    reranker: Reranker = Depends(get_reranker),
):
    """Hybrid search: bge semantic + keyword, RRF-merged, cross-encoder re-ranked
    with color/gender attribute boosting."""
    q = _clean_query(q)
    dbg: Optional[dict] = {} if debug else None
    results = _guard(
        lambda: search(query=q, embedder=embedder, qdrant=qdrant, col=col, top_k=top_k,
                       brand=brand, debug=dbg, reranker=reranker, gender=gender),
        "Text search",
    )
    return {"results": results, "total": len(results), "debug": dbg}


@router.get("/search/visual", response_model=SearchResponse, tags=["search"])
def search_visual(
    q: str = Query(..., description="Text query matched against product images (cross-modal)"),
    top_k: int = Query(20, ge=1, le=100),
    brand: Optional[str] = Query(None),
    clip: ClipEmbedder = Depends(get_clip_embedder),
    qdrant: Qdrant = Depends(get_qdrant),
):
    """Text → image (cross-modal) search via CLIP's shared text/image space."""
    q = _clean_query(q)
    results = _guard(
        lambda: text_to_image_search(query=q, clip_embedder=clip, qdrant=qdrant, top_k=top_k, brand=brand),
        "Visual search",
    )
    return {"results": results, "total": len(results)}


@router.post("/search/image", response_model=SearchResponse, tags=["search"])
async def search_by_image(
    file: UploadFile = File(..., description="Product image to find visual matches for"),
    top_k: int = Query(20, ge=1, le=100),
    brand: Optional[str] = Query(None),
    clip: ClipEmbedder = Depends(get_clip_embedder),
    qdrant: Qdrant = Depends(get_qdrant),
):
    """Image → image search. Upload a photo, get visually similar products."""
    if file.content_type and not file.content_type.lower().startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Expected an image, got '{file.content_type}'.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")

    image = clip.load_image_bytes(data)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG/PNG.")

    vector = clip.embed_image_one(image)
    results = _guard(
        lambda: image_search(image_vector=vector, qdrant=qdrant, top_k=top_k, brand=brand),
        "Image search",
    )
    return {"results": results, "total": len(results)}
