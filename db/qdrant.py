import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import {
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
}

logger = logging.getLogger("qdrant")

PAYLOAD_SCHEMA = {
    "mongo_id",
    "store_name",
    "title",
    "product_type",
    "tags",
    "price_min",
    "price_max",
    "currency",
    "featured_image",
    "product_url",
    "available",
}

class Qdrant:
    def __init__(self, url: str, api_key: Optional[str] = None, collection_name: str = "products", vector_dimension: int = 384):
        self.collection_name = collection_name
        self._client = QdrantClient(url=url, api_key=api_key, timeout=60, verify=False)
        self.vector_dimension = vector_dimension
        
    def ensure_collection(self):
        existing_collections = [c.name for c in self._client.get_collections().collections]
        logger.info("Existing Qdrant collections: %s", existing_collections)
        
        if self.collection_name in existing_collections:
            info = self._client.get_collection(self.collection_name)
            logger.info("Collection '%s' already exists. Vector dimension: %d, Distance: %s",
                        self.collection_name, info.vectors.size, info.vectors.distance)
            
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_dimension:
                raise ValueError(f"Collection '{self.collection_name}' already exists with vector dimension {existing_size}, expected {self.vector_dimension}.")
            logger.info(f"Collection '{self.collection_name}' already exists — reusing.")
            return
        
        logger.info("Creating Qdrant collection '%s' with vector dimension %d", self.collection_name, self.vector_dimension)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dimension, 
                distance=Distance.COSINE
            ),
        )
        
        for field in ["store_name", "available", "product_type"]:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            
        logger.info(f"Collection '{self.collection_name}' created with payload indexes.")
    
    def upsert_points(self, points: list[PointStruct]) -> int: 
        if not points:
            logger.warning("upsert_points called with empty list")
            return 0
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        logger.info("Upserted %d points into collection '%s'", len(points), self.collection_name)
        return len(points)
    
    def count(self) -> int:
        info = self._client.count(self.collection_name).count
        return info
    
    def delete_collection(self):
        self._client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)