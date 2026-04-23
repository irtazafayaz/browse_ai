from fastapi import APIRouter, Depends, Query
from pymongo.collection import Collection

from api.dependencies import get_embedder, get_qdrant, get_collection
from vectorizer.embedder import Embedder
from db.qdrant import Qdrant
from vectorizer.search import search

router = APIRouter()


@router.get("/search")
def search_products(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(20, ge=1, le=100),
    brand: str = Query(None),
    embedder: Embedder = Depends(get_embedder),
    qdrant: Qdrant = Depends(get_qdrant),
    col: Collection = Depends(get_collection),
):
    results = search(query=q, embedder=embedder, qdrant=qdrant, col=col, top_k=top_k, brand=brand)
    return {"results": results, "total": len(results)}
