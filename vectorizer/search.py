import logging
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue
from pymongo.collection import Collection

from vectorizer.embedder import Embedder
from vectorizer.clip_embedder import ClipEmbedder
from db.qdrant import Qdrant, TEXT_VECTOR, IMAGE_VECTOR
from db.mongo import keyword_search

logger = logging.getLogger(__name__)

RRF_K = 60


def _brand_filter(brand: Optional[str]) -> Optional[Filter]:
    if not brand:
        return None
    return Filter(must=[FieldCondition(key="brand", match=MatchValue(value=brand))])


def _query_vector(
    qdrant: Qdrant,
    vector: list[float],
    using: str,
    top_k: int,
    brand: Optional[str],
) -> list[dict]:
    response = qdrant._client.query_points(
        collection_name=qdrant.collection_name,
        query=vector,
        using=using,
        limit=top_k,
        query_filter=_brand_filter(brand),
        with_payload=True,
    )
    return [
        {
            **(hit.payload or {}),
            "mongo_id": (hit.payload or {}).get("mongo_id", ""),
            "score": round(hit.score, 6),
        }
        for hit in response.points
    ]


def _semantic_search(
    query: str,
    embedder: Embedder,
    qdrant: Qdrant,
    top_k: int,
    brand: Optional[str],
) -> list[dict]:
    vector = embedder.embed_one(query)
    return _query_vector(qdrant, vector, TEXT_VECTOR, top_k, brand)


def image_search(
    image_vector: list[float],
    qdrant: Qdrant,
    top_k: int = 20,
    brand: Optional[str] = None,
) -> list[dict]:
    """Image -> image search. Encode the query image with CLIP, search the
    'image' named vector for visually similar products."""
    logger.info("Image search: top_k=%d brand=%s", top_k, brand)
    return _query_vector(qdrant, image_vector, IMAGE_VECTOR, top_k, brand)


def text_to_image_search(
    query: str,
    clip_embedder: ClipEmbedder,
    qdrant: Qdrant,
    top_k: int = 20,
    brand: Optional[str] = None,
) -> list[dict]:
    """Cross-modal text -> image search. CLIP encodes the text query into the
    shared space and matches it against product image vectors."""
    logger.info("Text-to-image search: '%s' top_k=%d brand=%s", query, top_k, brand)
    vector = clip_embedder.embed_text_one(query)
    return _query_vector(qdrant, vector, IMAGE_VECTOR, top_k, brand)


def _rrf_merge(semantic: list[dict], keyword: list[dict], top_k: int) -> list[dict]:
    scores: dict[str, float] = {}
    data: dict[str, dict] = {}

    for rank, doc in enumerate(semantic):
        key = doc.get("mongo_id") or doc.get("productUrl", "")
        scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        data[key] = doc

    for rank, doc in enumerate(keyword):
        key = str(doc.get("_id", "")) or doc.get("productUrl", "")
        scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        if key not in data:
            doc.pop("_id", None)
            doc.pop("score", None)
            data[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for key, score in ranked:
        doc = data[key].copy()
        doc["score"] = round(score, 6)
        results.append(doc)

    return results


def search(
    query: str,
    embedder: Embedder,
    qdrant: Qdrant,
    col: Collection,
    top_k: int = 20,
    brand: Optional[str] = None,
) -> list[dict]:
    logger.info("Hybrid search: '%s' top_k=%d brand=%s", query, top_k, brand)

    semantic = _semantic_search(query, embedder, qdrant, top_k * 2, brand)
    keyword = keyword_search(col, query, top_k * 2, brand)

    logger.info("Semantic hits: %d | Keyword hits: %d", len(semantic), len(keyword))

    results = _rrf_merge(semantic, keyword, top_k)
    logger.info("Merged results: %d", len(results))

    return results
