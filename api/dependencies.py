import os
from functools import lru_cache
from dotenv import load_dotenv
from pymongo.collection import Collection

from vectorizer.embedder import Embedder
from db.qdrant import Qdrant
from db.mongo import get_products_col

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


def get_collection() -> Collection:
    return get_products_col()
