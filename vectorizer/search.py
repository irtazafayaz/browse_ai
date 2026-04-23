import logging
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from vectorizer.embedder import Embedder
from db.qdrant import Qdrant

logger = logging.getLogger(__name__)


def search(
    query: str,
    embedder: Embedder,
    qdrant: Qdrant,
    top_k: int = 20,
    brand: Optional[str] = None,
) -> list[dict]:
    logger.info("Search query: '%s' top_k=%d brand=%s", query, top_k, brand)

    vector = embedder.embed_one(query)

    query_filter = None
    if brand:
        query_filter = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=brand))]
        )

    response = qdrant._client.query_points(
        collection_name=qdrant.collection_name,
        query=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    results = []
    for hit in response.points:
        result = hit.payload or {}
        result["score"] = round(hit.score, 4)
        results.append(result)

    logger.info("Returned %d results", len(results))
    return results
