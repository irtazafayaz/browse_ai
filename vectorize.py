import argparse
import asyncio
import logging
import os
from dotenv import load_dotenv

from vectorizer.embedder import Embedder
from vectorizer.clip_embedder import ClipEmbedder
from vectorizer.pipeline import VectorizationPipeline
from db.qdrant import Qdrant

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


async def main(limit=None):
    mongo_uri = os.environ["MONGO_URI"]
    # Match db/mongo.py + .env.example (MONGO_DB kept as a fallback for older envs).
    mongo_db = os.environ.get("MONGO_DB_NAME") or os.environ.get("MONGO_DB", "browse_ai")
    mongo_collection = os.environ.get("MONGO_COLLECTION", "products")
    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    embedder = Embedder()
    clip_embedder = ClipEmbedder()
    qdrant = Qdrant(url=qdrant_url, api_key=qdrant_api_key)

    pipeline = VectorizationPipeline(
        embedder=embedder,
        qdrant=qdrant,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
        clip_embedder=clip_embedder,
    )

    await pipeline.run(limit=limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N products (quick test)")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
