import logging
import time
import hashlib
from typing import Optional
import motor.motor_asyncio
from qdrant_client.models import PointStruct
from .embedder import Embedder
from ..db.qdrant import Qdrant, PAYLOAD_SCHEMA

logger = logging.getLogger("pipeline")

BATCH_SIZE = 500   # products fetched from MongoDB per iteration
EMBED_BATCH = 64   # texts passed to the model per forward pass

def _make_point_id(mongo_id: str) -> str:
    # Create a unique point ID by hashing the MongoDB ID
    return hashlib.sha256(mongo_id.encode()).hexdigest()