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


async def main(limit=None, recreate=False):
    mongo_uri = os.environ["MONGO_URI"]
    # Match db/mongo.py + .env.example (MONGO_DB kept as a fallback for older envs).
    mongo_db = os.environ.get("MONGO_DB_NAME") or os.environ.get("MONGO_DB", "browse_ai")
    mongo_collection = os.environ.get("MONGO_COLLECTION", "products")
    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    embedder = Embedder()
    clip_embedder = ClipEmbedder()
    # Collection dimensions follow the loaded models; a model swap -> --recreate.
    qdrant = Qdrant(
        url=qdrant_url, api_key=qdrant_api_key,
        text_dimension=embedder.vector_dimension,
        image_dimension=clip_embedder.vector_dimension,
    )

    pipeline = VectorizationPipeline(
        embedder=embedder,
        qdrant=qdrant,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
        clip_embedder=clip_embedder,
    )

    await pipeline.run(limit=limit, recreate=recreate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N products (quick test)")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and rebuild the Qdrant collection (required after changing embedding models)")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, recreate=args.recreate))
