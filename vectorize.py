import asyncio
import logging
import os
from dotenv import load_dotenv

from vectorizer.embedder import Embedder
from vectorizer.pipeline import VectorizationPipeline
from db.qdrant import Qdrant

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    mongo_uri = os.environ["MONGO_URI"]
    mongo_db = os.environ["MONGO_DB"]
    mongo_collection = os.environ.get("MONGO_COLLECTION", "products")
    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    embedder = Embedder()
    qdrant = Qdrant(url=qdrant_url, api_key=qdrant_api_key)

    pipeline = VectorizationPipeline(
        embedder=embedder,
        qdrant=qdrant,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
    )

    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
