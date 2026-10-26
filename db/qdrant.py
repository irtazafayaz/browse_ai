import logging
import os
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

logger = logging.getLogger(__name__)

PAYLOAD_SCHEMA = {
    "mongo_id",
    "name",
    "brand",
    "category",
    "tags",
    "price",
    "imageUrl",
    "productUrl",
    "available",
    "description",  # <=1200 chars (scraper cap); read by the reranker at query time
}

# Named vectors. "text" = bge text embeddings, "image" = CLIP/SigLIP image
# embeddings. Both cosine. Dimensions are NOT fixed here — they come from the
# loaded models (Embedder.vector_dimension / ClipEmbedder.vector_dimension) so
# swapping a model cannot silently write wrong-sized vectors.
TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"

FILTER_FIELDS = ["brand", "category", "available", "colors", "gender", "product_type"]


class Qdrant:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        text_dimension: Optional[int] = None,
        image_dimension: Optional[int] = None,
    ):
        # Fall back to env so every call site (API, indexer, CLIs) shares one name.
        self.collection_name = collection_name or os.environ.get("QDRANT_COLLECTION_NAME", "products")
        self.text_dimension = text_dimension
        self.image_dimension = image_dimension
        # check_compatibility=False: silences the client/server minor-version warning.
        self._client = QdrantClient(url=url, api_key=api_key, timeout=60, check_compatibility=False)

    def _vectors_config(self) -> dict:
        if not self.text_dimension or not self.image_dimension:
            raise ValueError("text_dimension and image_dimension are required to create a collection")
        return {
            TEXT_VECTOR: VectorParams(size=self.text_dimension, distance=Distance.COSINE),
            IMAGE_VECTOR: VectorParams(size=self.image_dimension, distance=Distance.COSINE),
        }

    def _existing_vectors(self) -> Optional[dict]:
        """{name: size} for the live collection, or None if it doesn't exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            return None
        vectors = self._client.get_collection(self.collection_name).config.params.vectors
        if not isinstance(vectors, dict):
            return {"(unnamed)": getattr(vectors, "size", None)}
        return {name: params.size for name, params in vectors.items()}

    def ensure_collection(self, recreate: bool = False):
        live = self._existing_vectors()
        logger.info("Qdrant collection '%s': %s", self.collection_name, live or "absent")

        if live is not None and recreate:
            logger.warning("Dropping collection '%s' (recreate=True)", self.collection_name)
            self.delete_collection()
            live = None

        if live is not None:
            # Same model family but a different size (e.g. bge-small -> bge-large)
            # is the classic silent-breakage case — fail loudly with the fix.
            want = {TEXT_VECTOR: self.text_dimension, IMAGE_VECTOR: self.image_dimension}
            mismatch = {
                name: (live.get(name), size) for name, size in want.items()
                if size is not None and live.get(name) != size
            }
            if TEXT_VECTOR not in live or IMAGE_VECTOR not in live or mismatch:
                raise ValueError(
                    f"Collection '{self.collection_name}' has vectors {live}, but the loaded models need "
                    f"{want}. Re-index with `python vectorize.py --recreate` (drops and rebuilds the collection)."
                )
            logger.info("Collection '%s' matches model dimensions — reusing.", self.collection_name)
            return

        logger.info(
            "Creating collection '%s' (text=%d, image=%d)",
            self.collection_name, self.text_dimension, self.image_dimension,
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._vectors_config(),
        )

        # Index the fields we actually filter on.
        for field in FILTER_FIELDS:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        logger.info("Collection '%s' created with named vectors and payload indexes.", self.collection_name)

    def upsert_batch(self, points: list[PointStruct]) -> int:
        if not points:
            logger.warning("upsert_batch called with empty list")
            return 0
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.debug("Upserted %d points into '%s'", len(points), self.collection_name)
        return len(points)

    def count(self) -> int:
        return self._client.count(self.collection_name).count

    def delete_collection(self):
        self._client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)
