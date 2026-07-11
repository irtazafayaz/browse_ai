import logging
import time
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue
from pymongo.collection import Collection

from vectorizer.embedder import Embedder
from vectorizer.clip_embedder import ClipEmbedder
from vectorizer.attributes import parse_query, structured_attributes
from vectorizer.reranker import Reranker
from db.qdrant import Qdrant, TEXT_VECTOR, IMAGE_VECTOR
from db.mongo import keyword_search

logger = logging.getLogger(__name__)

RRF_K = 60
RERANK_POOL = 50  # max candidates handed to the cross-encoder


def _build_filter(brand: Optional[str] = None, apparel_only: bool = False) -> Optional[Filter]:
    must, must_not = [], []
    if brand:
        must.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    if apparel_only:
        # A garment/color query should never surface home/fragrance items.
        must_not.append(FieldCondition(key="product_type", match=MatchValue(value="non_apparel")))
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _query_vector(qdrant: Qdrant, vector, using: str, top_k: int, query_filter=None) -> list[dict]:
    response = qdrant._client.query_points(
        collection_name=qdrant.collection_name,
        query=vector, using=using, limit=top_k,
        query_filter=query_filter, with_payload=True,
    )
    return [
        {**(hit.payload or {}), "mongo_id": (hit.payload or {}).get("mongo_id", ""),
         "score": round(hit.score, 6)}
        for hit in response.points
    ]


def _semantic_search(query, embedder, qdrant, top_k, query_filter=None) -> list[dict]:
    vector = embedder.embed_query(query)  # bge query instruction applied inside
    return _query_vector(qdrant, vector, TEXT_VECTOR, top_k, query_filter)


def _score_span(docs: list[dict]) -> str:
    scores = [d["score"] for d in docs if isinstance(d.get("score"), (int, float))]
    return f"{max(scores):.4f}->{min(scores):.4f}" if scores else "n/a"


def _log_results(mode: str, query: str, results: list[dict]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("[%s] top %d for '%s':", mode, len(results), query)
    for i, r in enumerate(results, 1):
        logger.debug("   %2d. score=%-9s %-16s %s",
                     i, r.get("score"), (r.get("brand") or "")[:16], (r.get("name") or "")[:60])


def image_search(image_vector, qdrant, top_k: int = 20, brand: Optional[str] = None) -> list[dict]:
    """Image -> image search."""
    t0 = time.perf_counter()
    logger.info("Image search: top_k=%d brand=%s", top_k, brand)
    results = _query_vector(qdrant, image_vector, IMAGE_VECTOR, top_k, _build_filter(brand))
    logger.info("Image search done: %d results (%s) %.0fms",
                len(results), _score_span(results), (time.perf_counter() - t0) * 1000)
    _log_results("image", "<uploaded image>", results)
    return results


def text_to_image_search(query, clip_embedder, qdrant, top_k: int = 20, brand: Optional[str] = None) -> list[dict]:
    """Cross-modal text -> image search."""
    t0 = time.perf_counter()
    logger.info("Text-to-image search: '%s' top_k=%d brand=%s", query, top_k, brand)
    vector = clip_embedder.embed_text_one(query)
    results = _query_vector(qdrant, vector, IMAGE_VECTOR, top_k, _build_filter(brand))
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


def _drop_non_apparel(docs: list[dict]) -> list[dict]:
    # Keyword (Mongo) docs lack structured attrs. Compute once here, attach them
    # for the reranker to reuse, and drop non-apparel in the same pass.
    kept = []
    for d in docs:
        attrs = structured_attributes(d)
        if attrs["product_type"] == "non_apparel":
            continue
        d["colors"] = attrs["colors"]
        d["product_type"] = attrs["product_type"]
        if attrs["gender"]:
            d["gender"] = attrs["gender"]
        kept.append(d)
    return kept


def search(
    query: str,
    embedder: Embedder,
    qdrant: Qdrant,
    col: Collection,
    top_k: int = 20,
    brand: Optional[str] = None,
    debug: Optional[dict] = None,
    reranker: Optional[Reranker] = None,
    gender: Optional[str] = None,
) -> list[dict]:
    t0 = time.perf_counter()
    parsed = parse_query(query)
    q_colors = parsed["colors"]
    q_gender = gender or parsed["gender"]          # explicit gender (toggle) wins
    apparel_only = bool(q_colors or q_gender)      # treat as a garment query
    logger.info("Hybrid search: '%s' top_k=%d brand=%s colors=%s gender=%s apparel_only=%s",
                query, top_k, brand, q_colors, q_gender, apparel_only)

    cand_k = max(top_k * 2, 30)
    filt = _build_filter(brand, apparel_only)

    t = time.perf_counter()
    semantic = _semantic_search(query, embedder, qdrant, cand_k, filt)
    sem_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    keyword = keyword_search(col, query, cand_k, brand)
    if apparel_only:
        keyword = _drop_non_apparel(keyword)
    kw_ms = (time.perf_counter() - t) * 1000

    sem_span, kw_span = _score_span(semantic), _score_span(keyword)
    overlap = len({d.get("productUrl") for d in semantic if d.get("productUrl")}
                  & {d.get("productUrl") for d in keyword if d.get("productUrl")})

    pool_size = min(RERANK_POOL, max(top_k * 3, 20))
    pool = _rrf_merge(semantic, keyword, pool_size)

    if reranker is not None and pool:
        t = time.perf_counter()
        results = reranker.rerank(query, pool, top_k, q_colors, q_gender)
        rr_ms = (time.perf_counter() - t) * 1000
    else:
        if reranker is None:
            logger.warning("No reranker — returning RRF order without cross-encoder rerank")
        results, rr_ms = pool[:top_k], 0.0

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Hybrid done: sem=%d(%s %.0fms) kw=%d(%s %.0fms) overlap=%d pool=%d rerank=%.0fms top=%d total=%.0fms",
        len(semantic), sem_span, sem_ms, len(keyword), kw_span, kw_ms,
        overlap, len(pool), rr_ms, len(results), total_ms,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[hybrid] '%s' colors=%s gender=%s:", query, q_colors, q_gender)
        for i, r in enumerate(results, 1):
            logger.debug("   %2d. rerank=%-8s ce=%-8s boost=%-5s [%s/%s] %-14s %s",
                         i, r.get("rerank_score"), r.get("ce_score"), r.get("boost"),
                         r.get("colors"), r.get("gender"),
                         (r.get("brand") or "")[:14], (r.get("name") or "")[:48])

    if debug is not None:
        debug.update({
            "parsed": {"colors": q_colors, "gender": q_gender, "apparel_only": apparel_only},
            "timings_ms": {"semantic": round(sem_ms, 1), "keyword": round(kw_ms, 1),
                           "rerank": round(rr_ms, 1), "total": round(total_ms, 1)},
            "semantic": {"count": len(semantic), "score_span": sem_span},
            "keyword": {"count": len(keyword), "score_span": kw_span},
            "overlap": overlap, "pool": len(pool), "rrf_k": RRF_K,
            "results": [
                {"rank": i + 1, "rerank_score": r.get("rerank_score"), "ce_score": r.get("ce_score"),
                 "boost": r.get("boost"), "colors": r.get("colors"), "gender": r.get("gender"),
                 "brand": r.get("brand"), "name": r.get("name")}
                for i, r in enumerate(results)
            ],
        })

    return results
