import logging
import time
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
    t0 = time.perf_counter()
    logger.info("Image search: top_k=%d brand=%s", top_k, brand)
    results = _query_vector(qdrant, image_vector, IMAGE_VECTOR, top_k, brand)
    logger.info("Image search done: %d results (%s) %.0fms",
                len(results), _score_span(results), (time.perf_counter() - t0) * 1000)
    _log_results("image", "<uploaded image>", results)
    return results


def text_to_image_search(
    query: str,
    clip_embedder: ClipEmbedder,
    qdrant: Qdrant,
    top_k: int = 20,
    brand: Optional[str] = None,
) -> list[dict]:
    """Cross-modal text -> image search. CLIP encodes the text query into the
    shared space and matches it against product image vectors."""
    t0 = time.perf_counter()
    logger.info("Text-to-image search: '%s' top_k=%d brand=%s", query, top_k, brand)
    vector = clip_embedder.embed_text_one(query)
    results = _query_vector(qdrant, vector, IMAGE_VECTOR, top_k, brand)
    logger.info("Text-to-image done: %d results (%s) %.0fms",
                len(results), _score_span(results), (time.perf_counter() - t0) * 1000)
    _log_results("visual", query, results)
    return results


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


def _score_span(docs: list[dict]) -> str:
    """Compact 'max→min' of a result set's scores — a flat span signals the
    ranker is barely differentiating (e.g. weak BM25 keyword scores)."""
    scores = [d["score"] for d in docs if isinstance(d.get("score"), (int, float))]
    if not scores:
        return "n/a"
    return f"{max(scores):.4f}->{min(scores):.4f}"


def _log_results(mode: str, query: str, results: list[dict]) -> None:
    """DEBUG: dump the ranked results so relevance can be eyeballed."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("[%s] top %d for '%s':", mode, len(results), query)
    for i, r in enumerate(results, 1):
        logger.debug("   %2d. score=%-9s %-16s %s",
                     i, r.get("score"), (r.get("brand") or "")[:16], (r.get("name") or "")[:60])


def search(
    query: str,
    embedder: Embedder,
    qdrant: Qdrant,
    col: Collection,
    top_k: int = 20,
    brand: Optional[str] = None,
) -> list[dict]:
    t0 = time.perf_counter()
    logger.info("Hybrid search: '%s' top_k=%d brand=%s", query, top_k, brand)

    t = time.perf_counter()
    semantic = _semantic_search(query, embedder, qdrant, top_k * 2, brand)
    sem_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    keyword = keyword_search(col, query, top_k * 2, brand)
    kw_ms = (time.perf_counter() - t) * 1000

    # Capture score spans + provenance BEFORE merge — _rrf_merge pops "score"
    # and "_id" off keyword docs, so reading them afterwards would show nothing.
    # Attribute by productUrl: it's present on both semantic and keyword docs and
    # survives the merge, unlike _id/mongo_id.
    sem_span, kw_span = _score_span(semantic), _score_span(keyword)
    sem_rank = {d["productUrl"]: i for i, d in enumerate(semantic) if d.get("productUrl")}
    kw_rank = {d["productUrl"]: i for i, d in enumerate(keyword) if d.get("productUrl")}
    overlap = len(set(sem_rank) & set(kw_rank))

    results = _rrf_merge(semantic, keyword, top_k)
    total_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Hybrid done: sem=%d(%s %.0fms) kw=%d(%s %.0fms) overlap=%d merged=%d total=%.0fms",
        len(semantic), sem_span, sem_ms,
        len(keyword), kw_span, kw_ms,
        overlap, len(results), total_ms,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[hybrid] ranked results for '%s' (S=semantic rank, K=keyword rank):", query)
        for i, r in enumerate(results, 1):
            key = r.get("productUrl", "")
            s, k = sem_rank.get(key), kw_rank.get(key)
            src = (f"S#{s + 1}" if s is not None else "S  -") + (f"/K#{k + 1}" if k is not None else "/K  -")
            logger.debug("   %2d. rrf=%-9s [%-9s] %-16s %s",
                         i, r.get("score"), src, (r.get("brand") or "")[:16], (r.get("name") or "")[:55])

    return results
