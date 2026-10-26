"""
Image/text dual encoder (free, local). Encodes BOTH images and text into the
SAME vector space, which is what makes cross-modal search work: a text query
can be matched against product image vectors.

Default: `Marqo/marqo-fashionSigLIP` — a SigLIP ViT-B/16 fine-tuned on fashion
product catalogs (text<->image retrieval), 768-dim, ~0.8 GB. On fashion
benchmarks it beats general CLIP ViT-B/32 (the previous default) by a wide
margin, and it is small enough for MPS/CPU.

Two backends, picked from the model name:
  * "hf-hub:<org>/<repo>"  -> open_clip (SigLIP / laion CLIP / Marqo models)
  * anything else          -> sentence-transformers CLIP wrapper (e.g. "clip-ViT-B-32")
Set CLIP_MODEL to swap; the vector dimension is probed from the model.
"""
import io
import logging
import os
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image

from .device import DEVICE

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "hf-hub:Marqo/marqo-fashionSigLIP")
DEFAULT_BATCH_SIZE = 32
_DOWNLOAD_TIMEOUT = 15


class _OpenClipBackend:
    def __init__(self, name: str):
        import open_clip
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(name)
        self._model = self._model.to(DEVICE).eval()
        self._tokenizer = open_clip.get_tokenizer(name)

    @torch.inference_mode()
    def text(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self._tokenizer(texts[i:i + batch_size]).to(DEVICE)
            out.append(self._model.encode_text(tokens, normalize=True).float().cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def images(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self._preprocess(im) for im in images[i:i + batch_size]]).to(DEVICE)
            out.append(self._model.encode_image(x, normalize=True).float().cpu().numpy())
        return np.concatenate(out)


class _SentenceTransformersBackend:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(name, device=DEVICE)

    def text(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    def images(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        return self._model.encode(images, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


class ClipEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE):
        logger.info("Loading image/text model: %s on %s", model_name, DEVICE)
        if model_name.startswith("hf-hub:"):
            self._backend = _OpenClipBackend(model_name)
        else:
            self._backend = _SentenceTransformersBackend(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        # Probe instead of trusting a constant — a model swap can't desync Qdrant.
        self.vector_dimension = len(self.embed_text_one("probe"))
        logger.info("Image/text model ready. dimension=%d", self.vector_dimension)

    # --- text side (shares the image space) ---
    def embed_text_one(self, text: str) -> list[float]:
        return self.embed_text([text])[0]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._backend.text(texts, self.batch_size)
        return np.asarray(vectors, dtype=np.float32).tolist()

    # --- image side ---
    def embed_image_one(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        vectors = self._backend.images(images, self.batch_size)
        return np.asarray(vectors, dtype=np.float32).tolist()

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
