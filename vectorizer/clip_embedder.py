"""
CLIP embedder (free, local).

Uses sentence-transformers' `clip-ViT-B-32`, which encodes BOTH images and text
into the SAME 512-dim vector space. That shared space is what makes cross-modal
search work: a text query can be matched against product image vectors.

No paid API — the model downloads once (~350 MB) and runs on CPU.
"""
import io
import logging
from typing import Optional
import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_VECTOR_DIMENSION = 512
DEFAULT_BATCH_SIZE = 32
_DOWNLOAD_TIMEOUT = 15


class ClipEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE):
        logger.info("Loading CLIP model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_dimension = CLIP_VECTOR_DIMENSION
        self.batch_size = batch_size
        logger.info("CLIP ready. dimension=%d", CLIP_VECTOR_DIMENSION)

    # --- text side (shares the image space) ---
    def embed_text_one(self, text: str) -> list[float]:
        return self.embed_text([text])[0]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32).tolist()

    # --- image side ---
    def embed_image_one(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        vectors = self._model.encode(
            images,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32).tolist()

    def embed_image_url(self, url: str) -> Optional[list[float]]:
        img = self.load_image_url(url)
        return self.embed_image_one(img) if img is not None else None

    @staticmethod
    def load_image_url(url: str) -> Optional[Image.Image]:
        if not url:
            return None
        try:
            res = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
            res.raise_for_status()
            return Image.open(io.BytesIO(res.content)).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image %s: %s", url, e)
            return None

    @staticmethod
    def load_image_bytes(data: bytes) -> Optional[Image.Image]:
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            logger.warning("Failed to decode uploaded image: %s", e)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    clip = ClipEmbedder()
    tv = clip.embed_text_one("red floral maxi dress")
    print(f"text  → dim {len(tv)}, first 5: {tv[:5]}")
