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
}

# Named vectors. "text" = MiniLM description embeddings (existing),
# "image" = CLIP ViT-B/32 image embeddings (new). Both cosine.
TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"
TEXT_DIMENSION = 384
IMAGE_DIMENSION = 512


class Qdrant:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        text_dimension: int = TEXT_DIMENSION,
        image_dimension: int = IMAGE_DIMENSION,
    ):
        # Fall back to env so every call site (API, indexer, CLIs) shares one name.
        self.collection_name = collection_name or os.environ.get("QDRANT_COLLECTION_NAME", "products")
        self.text_dimension = text_dimension
        self.image_dimension = image_dimension
        # check_compatibility=False: silences the client/server minor-version warning.
        self._client = QdrantClient(url=url, api_key=api_key, timeout=60, check_compatibility=False)

    def _vectors_config(self) -> dict:
        return {
            TEXT_VECTOR: VectorParams(size=self.text_dimension, distance=Distance.COSINE),
            IMAGE_VECTOR: VectorParams(size=self.image_dimension, distance=Distance.COSINE),
        }

    def ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        logger.info("Existing Qdrant collections: %s", existing)

        if self.collection_name in existing:
            info = self._client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            # A pre-existing single-vector (unnamed) collection is incompatible.
            if not isinstance(vectors, dict) or TEXT_VECTOR not in vectors or IMAGE_VECTOR not in vectors:
                raise ValueError(
                    f"Collection '{self.collection_name}' exists without named vectors "
                    f"'{TEXT_VECTOR}'/'{IMAGE_VECTOR}'. Recreate it (delete_collection) to enable image search."
                )
            logger.info("Collection '%s' already exists with named vectors — reusing.", self.collection_name)
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
        for field in ["brand", "category", "available"]:
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
