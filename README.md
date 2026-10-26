# Browse AI — search backend

Hybrid text + image search over Pakistani fashion products. FastAPI on top of
MongoDB (catalog + keyword index) and Qdrant (text + image vectors), with free,
local models.

## Models

All models run locally (MPS on Apple silicon, CUDA, or CPU — auto-detected) and
download from Hugging Face on first use. Defaults were chosen with `bench/` on
a 7.5k-product sample pulled from the brand stores.

| Role | Default | Why |
|---|---|---|
| Text embedder (semantic channel) | `BAAI/bge-small-en-v1.5`, 384d | `mxbai-embed-large-v1` measured no better end-to-end on this catalog at 9× the indexing time |
| Image/text dual encoder | `hf-hub:Marqo/marqo-fashionSigLIP`, 768d | SigLIP fine-tuned on fashion catalogs; name→own-image MRR 0.188 vs 0.069 for CLIP ViT-B/32 (R@1 0.131 vs 0.038) |
| Cross-encoder reranker | `mixedbread-ai/mxbai-rerank-base-v1` | beats `ms-marco-MiniLM-L-6-v2` on every variant tested (end-to-end P@10 0.788 vs 0.722 at pool 80) |

End to end on 60 attribute queries, the old stack (no descriptions, MiniLM,
pool 50) scored P@10 0.398; the current one scores 0.788 (82% of the 0.962
ceiling the sample allows).

Override with `TEXT_EMBED_MODEL`, `CLIP_MODEL`, `RERANK_MODEL` (see
`.env.example`). The Qdrant collection stores the vector sizes of the models it
was built with and the API refuses to start on a mismatch — index and serve
with the same models.

## How a text query is answered (`GET /api/v1/search`)

1. Parse colour / gender from the query (`vectorizer/attributes.py`).
2. Two candidate channels of 80 each: bge semantic search in Qdrant (filtered to
   apparel when the query names a colour/gender) and the Mongo `$text` keyword
   index.
3. Reciprocal-rank-fusion merge → pool of 80.
4. Cross-encoder scores each (query, product) pair on name + brand + category +
   extracted attributes + the first 300 chars of the description. Scores are
   z-normalised within the pool, then colour/gender match/mismatch boosts (in
   std units, model-agnostic) are added. Top-k returned. `?debug=true` exposes
   timings and per-result scores.

`GET /api/v1/search/visual` (text → product images) and
`POST /api/v1/search/image` (photo → similar products) use the dual encoder's
shared space directly.

## Data notes that matter for relevance

* `scrapers/shopify.py` keeps the cleaned `body_html` as `description`. Brands
  put the facts search needs there (`Color: Fuchsia Pink`, `Fabric: Lawn`,
  `Dupatta: Included`) and nowhere else. It feeds the text vector, the Mongo
  text index, and the reranker.
* The colour facet reads the name, tags, **and** the description's `Color:`
  field (only the few words after the label — the free text around it names
  other colours). On the sample this took colour coverage from 32% to 62% of
  products with no name/field conflicts.

## Setup

```bash
python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in MONGO_URI / QDRANT_URL
```

Scrape, clean, index, serve:

```bash
python -m scrapers.runner --clear      # pulls descriptions now; re-run after upgrading
python clean.py --apply
python vectorize.py --recreate         # required whenever an embedding model changes
uvicorn api.main:app --port 8001       # /api/v1/dashboard shows model + DB status
```

## Benchmarking model swaps

`bench/` needs no database: it pulls a sample straight from the brand stores and
scores models on it.

```bash
python bench/fetch_sample.py                               # -> bench/sample/ (gitignored)
HF_HUB_OFFLINE=1 python bench/bench.py --task t2i  --models clip-ViT-B-32,hf-hub:Marqo/marqo-fashionSigLIP
HF_HUB_OFFLINE=1 python bench/bench.py --task text --gen-queries 60 --models BAAI/bge-small-en-v1.5 \
    --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2,mixedbread-ai/mxbai-rerank-base-v1
HF_HUB_OFFLINE=1 python bench/bench.py --task pipeline --gen-queries 60 \
    --models BAAI/bge-small-en-v1.5,hf-hub:Marqo/marqo-fashionSigLIP --rerankers mixedbread-ai/mxbai-rerank-base-v1
```

* `t2i`: product name → its own image, Recall@k / MRR (cross-modal alignment on
  this catalog).
* `text` / `pipeline`: attribute-precision@10 on colour × fabric × garment ×
  gender queries — every attribute the query states must be satisfied by the
  product (rule-based, so absolute numbers are strict; compare deltas).
  `pipeline` mirrors `search.py` end to end and reports candidate-pool variants.
