"""
Cross-encoder re-ranker (free, local) + attribute-aware boosting.

Retrieval (bi-encoder + keyword) is recall-oriented; the cross-encoder scores
each (query, product) pair jointly for precision. On top of that we apply soft
attribute boosts: reward color/gender matches, demote KNOWN mismatches (e.g. a
green product for a "red" query), leave unknown-attribute products neutral.

Default model: `mixedbread-ai/mxbai-rerank-base-v1` (184M, BEIR NDCG@10 ~46.9 —
above bge-reranker-base and well above ms-marco-MiniLM-L-6 at ~41). Set
RERANK_MODEL to swap; CPU-only deploys can fall back to
`cross-encoder/ms-marco-MiniLM-L-6-v2`.

Boosts are expressed in standard deviations of the cross-encoder scores within
the candidate pool, NOT raw logits, so they mean the same thing whichever model
is loaded (ms-marco logits span ~-11..+11, mxbai's are a different scale).
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from vectorizer.attributes import structured_attributes, enrichment_text
from vectorizer.device import DEVICE, MODEL_KWARGS

logger = logging.getLogger(__name__)

RERANK_MODEL = os.environ.get("RERANK_MODEL", "mixedbread-ai/mxbai-rerank-base-v1")

# Boost weights in pool-std units (see module docstring).
COLOR_MATCH = 1.0
COLOR_MISMATCH = -1.5       # product has a KNOWN color, none matches the query
GENDER_MATCH = 0.5
GENDER_MISMATCH = -2.0      # product is a known, different gender
NON_APPAREL_PENALTY = -2.5  # garment/color query should not surface home/fragrance
_MIN_STD = 1e-3
DESCRIPTION_CHARS = 300


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        logger.info("Loading cross-encoder: %s on %s", model_name, DEVICE)
        # Identity activation: always work with raw logits. Some models (bge, mxbai)
        # would otherwise be sigmoid-squashed into [0,1] by sentence-transformers'
        # default, flattening the top of the ranking.
        self._model = CrossEncoder(model_name, device=DEVICE, activation_fn=torch.nn.Identity(),
                                   model_kwargs=MODEL_KWARGS)
        self.model_name = model_name
        logger.info("Cross-encoder ready.")

    def _doc_text(self, doc: dict) -> str:
        # The description's first few hundred chars hold the labelled facts
        # (Color/Fabric/Cut) the cross-encoder needs; the tail is boilerplate.
        parts = [doc.get("name") or "", doc.get("brand") or "", doc.get("category") or "",
                 enrichment_text(doc), (doc.get("description") or "")[:DESCRIPTION_CHARS]]
        return " ".join(p for p in parts if p).strip()

    def _boost(self, doc: dict, q_colors: list, q_gender: Optional[str]) -> float:
        # Semantic/pre-annotated docs always carry a "colors" key (payload), where
        # an absent gender legitimately means "unknown" — so don't recompute.
        # Only bare keyword docs need attributes derived on the fly.
        if "colors" in doc:
            colors = doc.get("colors") or []
            gender = doc.get("gender")
            ptype = doc.get("product_type")
        else:
            attrs = structured_attributes(doc)
            colors, gender, ptype = attrs["colors"], attrs["gender"], attrs["product_type"]

        boost = 0.0
        if q_colors:
            if colors and set(q_colors) & set(colors):
                boost += COLOR_MATCH
            elif colors:
                boost += COLOR_MISMATCH
        if q_gender and gender:
            boost += GENDER_MATCH if gender == q_gender else GENDER_MISMATCH
        if (q_colors or q_gender) and ptype == "non_apparel":
            boost += NON_APPAREL_PENALTY
        return boost

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int,
        q_colors: Optional[list] = None,
        q_gender: Optional[str] = None,
    ) -> list[dict]:
        if not docs:
            return []
        q_colors = q_colors or []
        pairs = [(query, self._doc_text(d)) for d in docs]
        ce_scores = np.asarray(self._model.predict(pairs, show_progress_bar=False), dtype=np.float64)

        # z-score within the pool so boosts are model-agnostic.
        std = float(ce_scores.std())
        z = (ce_scores - ce_scores.mean()) / (std if std > _MIN_STD else 1.0)

        scored = []
        for doc, ce, zs in zip(docs, ce_scores, z):
            boost = self._boost(doc, q_colors, q_gender)
            final = round(float(zs) + boost, 4)
            doc = {**doc, "ce_score": round(float(ce), 4), "boost": round(boost, 2),
                   "rerank_score": final, "score": final}
            scored.append(doc)

        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]
