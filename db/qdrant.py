import logging
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


class Qdrant:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "products",
        vector_dimension: int = 384,
    ):
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self._client = QdrantClient(url=url, api_key=api_key, timeout=60)

    def ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        logger.info("Existing Qdrant collections: %s", existing)

        if self.collection_name in existing:
            info = self._client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_dimension:
                raise ValueError(
                    "Collection '%s' exists with dimension %d, expected %d.",
                    self.collection_name, existing_size, self.vector_dimension,
                )
            logger.info("Collection '%s' already exists — reusing.", self.collection_name)
            return

        logger.info("Creating collection '%s' (dim=%d)", self.collection_name, self.vector_dimension)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE),
        )

        for field in ["store_name", "available", "product_type"]:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        logger.info("Collection '%s' created with payload indexes.", self.collection_name)

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
