import logging
import os
import certifi
from pymongo import MongoClient, TEXT
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

_client: MongoClient = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        _client = MongoClient(uri, tlsCAFile=certifi.where())
    return _client


def get_db():
    name = os.environ.get("MONGO_DB_NAME", "browse_ai")
    return get_client()[name]


def get_products_col() -> Collection:
    col = get_db()["products"]
    _ensure_text_index(col)
    return col


def _ensure_text_index(col: Collection):
    existing = [list(idx["key"].keys()) for idx in col.list_indexes()]
    has_text = any("_fts" in keys for keys in existing)
    if not has_text:
        logger.info("Creating text index on products collection")
        col.create_index(
            [("name", TEXT), ("brand", TEXT), ("description", TEXT), ("tags", TEXT)],
            default_language="english",
        )
        logger.info("Text index created")


def keyword_search(col: Collection, query: str, top_k: int = 100, brand: str = None) -> list[dict]:
    filter_ = {"$text": {"$search": query}}
    if brand:
        filter_["brand"] = brand

    cursor = col.find(
        filter_,
        {"score": {"$meta": "textScore"}, "_id": 1, "name": 1, "brand": 1,
         "category": 1, "price": 1, "imageUrl": 1, "productUrl": 1, "tags": 1},
    ).sort([("score", {"$meta": "textScore"})]).limit(top_k)

    return list(cursor)
