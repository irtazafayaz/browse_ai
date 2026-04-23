import os
from functools import lru_cache
from dotenv import load_dotenv

from vectorizer.embedder import Embedder
from db.qdrant import Qdrant

load_dotenv()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_qdrant() -> Qdrant:
    return Qdrant(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
