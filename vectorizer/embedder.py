import logging
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from .device import DEVICE, MODEL_KWARGS

logger = logging.getLogger(__name__)

# bge-small-en-v1.5 (384-dim). Benchmarked against mixedbread-ai/mxbai-embed-large-v1
# (1024-dim, MTEB retrieval 54.4 vs 51.7) on a 7.5k-product catalog sample
# (bench/): semantic-only P@10 0.313 vs 0.322, title->description MRR 0.158 vs
# 0.162, and end-to-end (hybrid + rerank) 0.735 vs 0.695 — the large model buys
# nothing here, at 9x the indexing time. Product text is short and the cross-
# encoder does the heavy lifting. TEXT_EMBED_MODEL swaps it (any bge/mxbai model
# is a drop-in: the dimension is read from the model, both families use the same
# QUERY-side instruction, and passages are indexed bare).
DEFAULT_MODEL_NAME = os.environ.get("TEXT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_BATCH_SIZE = 64
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE):
        logger.info("Loading text model: %s on %s", model_name, DEVICE)
        self._model = SentenceTransformer(model_name, device=DEVICE, model_kwargs=MODEL_KWARGS)
        self.model_name = model_name
        self.vector_dimension = self._model.get_embedding_dimension()
        self.batch_size = batch_size
        logger.info("Text model ready. dimension=%d", self.vector_dimension)

    def embed_one(self, text: str) -> list[float]:
        """Passage-side single embed (indexing)."""
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Query-side embed — prepends the bge retrieval instruction."""
        return self.embed([QUERY_INSTRUCTION + text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("embed() called with empty list")
            return []
        logger.debug("Embedding %d texts (batch_size=%d)", len(texts), self.batch_size)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(name)s | %(message)s")

    embedder = Embedder()

    v = embedder.embed_one("red cotton kurta for women")
    print(f"\nembed_one → length: {len(v)}, first 5: {v[:5]}")

    vectors = embedder.embed([
        "red cotton kurta for women",
        "blue silk dupatta",
        "printed lawn suit",
    ])
    print(f"\nembed     → {len(vectors)} vectors, each length {len(vectors[0])}")
