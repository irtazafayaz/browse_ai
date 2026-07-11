"""
Cross-encoder re-ranker (free, local) + attribute-aware boosting.

Retrieval (bi-encoder + keyword) is recall-oriented; the cross-encoder scores
each (query, product) pair jointly for precision. On top of that we apply soft
attribute boosts: reward color/gender matches, demote KNOWN mismatches (e.g. a
green product for a "red" query), leave unknown-attribute products neutral.
"""
import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from vectorizer.attributes import structured_attributes, enrichment_text

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Boost weights, calibrated to the ms-marco logit scale (~ -11..+11).
COLOR_MATCH = 3.0
COLOR_MISMATCH = -5.0      # product has a KNOWN color, none matches the query
GENDER_MATCH = 1.5
GENDER_MISMATCH = -6.0     # product is a known, different gender
NON_APPAREL_PENALTY = -8.0  # garment/color query should not surface home/fragrance


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        logger.info("Loading cross-encoder: %s", model_name)
        self._model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Cross-encoder ready.")

    def _doc_text(self, doc: dict) -> str:
        parts = [doc.get("name") or "", doc.get("brand") or "", doc.get("category") or "", enrichment_text(doc)]
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
        ce_scores = self._model.predict(pairs, show_progress_bar=False)

        scored = []
        for doc, ce in zip(docs, ce_scores):
            boost = self._boost(doc, q_colors, q_gender)
            final = round(float(ce) + boost, 4)
            doc = {**doc, "ce_score": round(float(ce), 4), "boost": round(boost, 2),
                   "rerank_score": final, "score": final}
            scored.append(doc)

        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]
