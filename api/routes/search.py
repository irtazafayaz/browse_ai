from fastapi import APIRouter, Query
from vectorizer.search import search

router = APIRouter()


@router.get("/search")
def search_products(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(20, ge=1, le=100),
    brand: str = Query(None),
):
    results = search(query=q, top_k=top_k, brand=brand)
    return {"results": results, "total": len(results)}
