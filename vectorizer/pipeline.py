import logging
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import motor.motor_asyncio
from qdrant_client.models import PointStruct

from .embedder import Embedder
from .clip_embedder import ClipEmbedder
from .attributes import enrichment_text, structured_attributes
from db.qdrant import Qdrant, PAYLOAD_SCHEMA, TEXT_VECTOR, IMAGE_VECTOR

logger = logging.getLogger(__name__)

BATCH_SIZE = 500
EMBED_BATCH = 64
IMAGE_DOWNLOAD_WORKERS = int(os.environ.get("IMAGE_DOWNLOAD_WORKERS", "16"))


def _make_point_id(mongo_id: str) -> int:
    return int(hashlib.md5(mongo_id.encode()).hexdigest(), 16) % (2 ** 63)


def _build_payload(product: dict) -> dict:
    payload = {}
    for field in PAYLOAD_SCHEMA:
        value = product.get(field)
        if value is not None:
            payload[field] = value

    tags = payload.get("tags", [])
    if isinstance(tags, str):
        payload["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    elif not isinstance(tags, list):
        payload["tags"] = []

    # Structured, filterable/boostable attributes.
    attrs = structured_attributes(product)
    payload["colors"] = attrs["colors"]
    payload["product_type"] = attrs["product_type"]
    if attrs["gender"]:
        payload["gender"] = attrs["gender"]

    return payload


def _build_text(product: dict) -> Optional[str]:
    parts = [
        product.get("name") or "",
        product.get("brand") or "",
        product.get("category") or "",
        product.get("description") or "",
        # Clean extracted attributes (color/gender/style) — puts color into the
        # vector even when it only appears in noisy tags.
        enrichment_text(product),
    ]
    text = " ".join(p.strip() for p in parts if p.strip())
    return text if text else None


class VectorizationPipeline:
    def __init__(
        self,
        embedder: Embedder,
        qdrant: Qdrant,
        mongo_uri: str,
        mongo_db: str,
        mongo_collection: str,
        clip_embedder: Optional[ClipEmbedder] = None,
    ):
        self.embedder = embedder
        self.clip_embedder = clip_embedder
        self.qdrant = qdrant
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection_name = mongo_collection

    async def run(self, limit: Optional[int] = None):
        start = time.time()

        self.qdrant.ensure_collection()

        client = motor.motor_asyncio.AsyncIOMotorClient(
            self.mongo_uri,
            serverSelectionTimeoutMS=10_000,
        )
        collection = client[self.mongo_db][self.mongo_collection_name]

        total_docs = await collection.count_documents({})
        if limit:
            total_docs = min(total_docs, limit)
            logger.info("LIMIT active: indexing only %d products", total_docs)
        logger.info("Total products to index: %d", total_docs)
        logger.info("Batch size: %d | Embed batch: %d", BATCH_SIZE, EMBED_BATCH)
        logger.info("Qdrant collection: %s", self.qdrant.collection_name)
        logger.info("Starting vectorization...")

        total_upserted = 0
        total_skipped = 0
        batch_num = 0
        batch_docs: list[dict] = []

        cursor = collection.find({}, batch_size=BATCH_SIZE)
        if limit:
            cursor = cursor.limit(limit)
        async for doc in cursor:
            batch_docs.append(doc)

            if len(batch_docs) < BATCH_SIZE:
                continue

            upserted, skipped = await self._process_batch(batch_docs, batch_num)
            total_upserted += upserted
            total_skipped += skipped
            batch_num += 1
            batch_docs = []

            elapsed = time.time() - start
            rate = total_upserted / elapsed if elapsed > 0 else 0
            eta = (total_docs - total_upserted - total_skipped) / rate if rate > 0 else 0
            logger.info(
                "Batch %d: upserted=%d skipped=%d | total=%d/%d | %.0f products/s | ETA %.0fs",
                batch_num, upserted, skipped, total_upserted, total_docs, rate, eta,
            )

        if batch_docs:
            upserted, skipped = await self._process_batch(batch_docs, batch_num)
            total_upserted += upserted
            total_skipped += skipped
            batch_num += 1

        client.close()

        duration = time.time() - start
        logger.info("--- VECTORIZATION COMPLETE ---")
        logger.info("  MongoDB products : %d", total_docs)
        logger.info("  Upserted         : %d", total_upserted)
        logger.info("  Skipped          : %d (empty titles)", total_skipped)
        logger.info("  Qdrant total     : %d", self.qdrant.count())
        logger.info("  Batches          : %d", batch_num)
        logger.info("  Time             : %.1fs", duration)
        logger.info("  Avg speed        : %.0f products/s", total_upserted / duration if duration > 0 else 0)

    async def _process_batch(self, docs: list[dict], batch_num: int) -> tuple[int, int]:
        valid_docs = []
        skipped = 0
        for doc in docs:
            text = _build_text(doc)
            if text:
                valid_docs.append((doc, text))
            else:
                skipped += 1

        if not valid_docs:
            return 0, skipped

        texts = [text for _, text in valid_docs]
        text_vectors = self.embedder.embed(texts)

        # CLIP image vectors (optional). Download images concurrently, then encode
        # only those that loaded. Products without a usable image just skip the
        # "image" named vector — their "text" vector is still stored.
        image_vectors: list[Optional[list[float]]] = [None] * len(valid_docs)
        if self.clip_embedder is not None:
            image_vectors = self._embed_images(valid_docs)

        points = []
        for i, (doc, _) in enumerate(valid_docs):
            mongo_id = str(doc.get("_id", ""))
            named_vectors = {TEXT_VECTOR: text_vectors[i]}
            if image_vectors[i] is not None:
                named_vectors[IMAGE_VECTOR] = image_vectors[i]
            points.append(PointStruct(
                id=_make_point_id(mongo_id),
                vector=named_vectors,
                payload={**_build_payload(doc), "mongo_id": mongo_id},
            ))

        upserted = self.qdrant.upsert_batch(points)
        return upserted, skipped

    def _embed_images(self, valid_docs: list[tuple[dict, str]]) -> list[Optional[list[float]]]:
        urls = [doc.get("imageUrl") or "" for doc, _ in valid_docs]

        with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
            images = list(pool.map(self.clip_embedder.load_image_url, urls))

        idx_with_img = [i for i, img in enumerate(images) if img is not None]
        result: list[Optional[list[float]]] = [None] * len(valid_docs)
        if not idx_with_img:
            logger.warning("Batch produced no loadable images (%d urls)", len(urls))
            return result

        vectors = self.clip_embedder.embed_images([images[i] for i in idx_with_img])
        for slot, vec in zip(idx_with_img, vectors):
            result[slot] = vec
        logger.info("Encoded %d/%d product images", len(idx_with_img), len(valid_docs))
        return result
