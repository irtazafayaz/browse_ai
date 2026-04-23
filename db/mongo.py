"""
MongoDB connection for browse-ai.
Connects to the same database as browse-ai-backend.
"""
import os
import certifi
from pymongo import MongoClient

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


def get_products_col():
    return get_db()["products"]
