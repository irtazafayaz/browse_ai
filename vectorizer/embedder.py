import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# bge-small-en-v1.5: stronger retrieval than MiniLM, still free/local, same 384-dim.
# bge expects a short instruction on the QUERY side only (not on indexed passages).
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384
DEFAULT_BATCH_SIZE = 64
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE):
        logger.info("Loading model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_dimension = VECTOR_DIMENSION
        self.batch_size = batch_size
        logger.info("Model ready. dimension=%d", VECTOR_DIMENSION)

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
        logger.info("Embedding %d texts (batch_size=%d)", len(texts), self.batch_size)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = np.array(vectors, dtype=np.float32).tolist()
        logger.info("Done. shape=%dx%d", len(result), len(result[0]))
        return result


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
