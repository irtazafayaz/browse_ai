"""
Model / pipeline benchmark on a real catalog sample (see fetch_sample.py).

Tasks
  t2i       text->image. query = product name, relevant = images of products with
            that exact name. Recall@1/5/10, MRR — cross-modal alignment on THIS
            catalog. Also attribute-precision of natural queries via the image side.
  text      natural attribute queries -> product text index (what pipeline._build_text
            indexes). Metric = attribute precision@10 (every colour/gender/garment/
            fabric term the query states must be satisfied). Optional cross-encoder
            rerank of the top-50 (CE only, no boosts). Plus title->description
            retrieval (lexical gap) when the sample has enough unique descriptions.
  pipeline  mirror of vectorizer/search.py end-to-end (semantic + BM25-as-Mongo-proxy
            -> RRF -> cross-encoder with z-scored attribute boosts), then candidate
            variants: colour/gender-filtered channels, bigger pools, image channel.

Usage
  python bench/bench.py --task t2i --models clip-ViT-B-32,hf-hub:Marqo/marqo-fashionSigLIP
  python bench/bench.py --task text --models BAAI/bge-small-en-v1.5 \
         --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2,mixedbread-ai/mxbai-rerank-base-v1
  python bench/bench.py --task pipeline --models <text-model>[,<dual-model>] --rerankers <ce>
Runs offline once models are cached (HF_HUB_OFFLINE=1).
"""
import argparse
import html
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorizer.attributes import (  # noqa: E402
    classify_type, enrichment_text, extract_colors, extract_gender, product_colors,
)
from vectorizer.device import DEVICE, MODEL_KWARGS  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
USE_DESC = True
CE_CHARS = int(os.environ.get("BENCH_CE_CHARS", "300"))  # mirrors reranker.DESCRIPTION_CHARS


# ----------------------------------------------------------------- model adapters
class STText:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=DEVICE, model_kwargs=MODEL_KWARGS)
        self.inst = "Represent this sentence for searching relevant passages: " if ("bge" in name or "mxbai" in name) else ""

    def docs(self, texts, bs=64):
        return self.m.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)

    def queries(self, texts):
        return self.m.encode([self.inst + t for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)


class STClip:
    """sentence-transformers CLIP wrapper (clip-ViT-B-32, clip-ViT-L-14)."""
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=DEVICE)

    def text(self, texts):
        return self.m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def images(self, imgs, bs=32):
        return self.m.encode(imgs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)


class OpenClip:
    """open_clip hf-hub models (Marqo fashion SigLIP/CLIP, laion CLIP)."""
    def __init__(self, name):
        import open_clip
        self.m, _, self.pre = open_clip.create_model_and_transforms(name)
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer(name)

    @torch.inference_mode()
    def text(self, texts, bs=64):
        out = [self.m.encode_text(self.tok(texts[i:i + bs]).to(DEVICE), normalize=True).float().cpu().numpy()
               for i in range(0, len(texts), bs)]
        return np.concatenate(out)

    @torch.inference_mode()
    def images(self, imgs, bs=32):
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.stack([self.pre(im) for im in imgs[i:i + bs]]).to(DEVICE)
            out.append(self.m.encode_image(x, normalize=True).float().cpu().numpy())
        return np.concatenate(out)


def dual(name):
    return OpenClip(name) if name.startswith("hf-hub:") else STClip(name)


class CE:
    def __init__(self, name):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(name, device=DEVICE, activation_fn=torch.nn.Identity(), model_kwargs=MODEL_KWARGS)

    def score(self, pairs):
        return np.asarray(self.m.predict(pairs, batch_size=32, show_progress_bar=False), dtype=np.float32)


# ----------------------------------------------------------------- data
def load():
    import json
    prods = json.load(open(os.path.join(SAMPLE, "products.json")))
    for p in prods:
        full_desc = p.get("description", "")
        p["_desc"] = re.sub(r"\s+", " ", html.unescape(full_desc)).strip()
        # Ground truth for the relevance rule always uses the full description,
        # so --no-desc changes only what gets INDEXED, not what counts as relevant.
        p["description"] = full_desc
        p["_colors"] = set(product_colors(p))
        p["_gender"] = extract_gender(p["category"], p["tags"], p["name"], p["productUrl"])
        p["_type"] = classify_type(p["category"], p["tags"], p["name"])
        p["_hay"] = f"{p['name']} {' '.join(p['tags'])} {p['category']} {p['_desc'][:300]}".lower()
        # what the pipeline sees
        p["description"] = full_desc if USE_DESC else ""
        # mirror pipeline._build_text (bi-encoder doc) and Reranker._doc_text (CE doc)
        p["_text"] = " ".join(x for x in [p["name"], p["brand"], p["category"], p["description"], enrichment_text(p)] if x)
        p["_ce"] = " ".join(x for x in [p["name"], p["brand"], p["category"], enrichment_text(p), p["description"][:CE_CHARS]] if x)
    return prods


def good_name(n):
    return len(re.findall(r"[A-Za-z]{3,}", n)) >= 3


def metrics(ranked_idx, relevant_sets, ks=(1, 5, 10)):
    rec = {k: 0 for k in ks}
    mrr = 0
    for r, rel in zip(ranked_idx, relevant_sets):
        hit = [i for i, d in enumerate(r) if d in rel]
        if hit:
            mrr += 1 / (hit[0] + 1)
            for k in ks:
                if hit[0] < k:
                    rec[k] += 1
    n = len(ranked_idx)
    return {f"R@{k}": round(rec[k] / n, 4) for k in ks} | {"MRR": round(mrr / n, 4), "n": n}


# ----------------------------------------------------------------- attribute queries
QUERIES = [
    "red embroidered lawn suit for women", "black abaya", "white cotton kurta for men",
    "blue printed lawn 3 piece unstitched", "pink chiffon formal dress for wedding", "green silk saree",
    "men shalwar kameez navy blue", "yellow kids frock", "mustard khaddar winter suit", "maroon velvet shawl",
    "grey waistcoat for men", "purple organza party wear", "brown linen kurta", "white embroidered dupatta",
    "orange printed kurti", "black trousers for women", "boys kurta shalwar white", "girls pink frock",
    "beige co-ord set", "golden festive eid outfit",
]
GARMENT = ["suit", "abaya", "kurta", "kurti", "kameez", "shalwar", "dress", "saree", "sari", "frock",
           "shawl", "waistcoat", "dupatta", "trousers", "trouser", "co-ord", "outfit", "lawn", "piece",
           "khaddar", "velvet", "chiffon", "silk", "organza", "linen", "cotton", "embroidered", "printed",
           "winter", "party", "formal", "festive", "eid", "wedding", "unstitched"]
GARMENT_SYN = {"suit": ["suit", "piece", "unstitched", "stitched", "shirt", "kameez"],
               "dress": ["dress", "frock", "maxi", "gown", "suit", "piece"],
               "outfit": ["suit", "piece", "shirt", "kurta", "dress", "kameez"],
               "trousers": ["trouser", "trousers", "pant", "pants", "culotte", "shalwar"],
               "co-ord": ["co-ord", "co ord", "coord", "set"],
               "saree": ["saree", "sari"], "sari": ["saree", "sari"],
               "kurta": ["kurta", "kurti", "kameez", "shirt"], "kurti": ["kurti", "kurta", "shirt"],
               "piece": ["piece", "pc", "suit"]}


_GARMENT_RX = {g: [re.compile(r"\b" + re.escape(s) + r"\b") for s in GARMENT_SYN.get(g, [g])] for g in GARMENT}
_Q_CACHE = {}


def _parse_q(q):
    if q not in _Q_CACHE:
        ql = q.lower()
        _Q_CACHE[q] = (set(extract_colors(q)), extract_gender(name=q),
                       [g for g in GARMENT if re.search(r"\b" + re.escape(g) + r"\b", ql)])
    return _Q_CACHE[q]


def attr_ok(q, p):
    """Rule relevance: every attribute the query states must be satisfied by the product."""
    if p["_type"] == "non_apparel":
        return False
    qc, qg, terms = _parse_q(q)
    if qc and not (qc & p["_colors"]):
        return False
    if qg and p["_gender"] and p["_gender"] != qg:
        return False
    hay = p["_hay"]
    for g in terms:
        if not any(rx.search(hay) for rx in _GARMENT_RX[g]):
            return False
    return True


def generate_queries(prods, n, min_rel=5, seed=0):
    """Template queries (colour x fabric/style x garment x gender) that have at
    least `min_rel` rule-relevant products in the sample, so P@10 isn't capped
    near zero by queries the catalog can't answer."""
    import random
    rng = random.Random(seed)
    colours = ["red", "pink", "blue", "green", "black", "white", "yellow", "brown", "grey", "purple",
               "orange", "maroon", "navy", "mustard", "beige", ""]
    styles = ["", "", "embroidered", "printed", "cotton", "silk", "khaddar", "linen", "velvet", "chiffon", "lawn"]
    garments = ["kurta", "suit", "dupatta", "trousers", "frock", "shawl", "shirt", "abaya", "kameez",
                "co-ord set", "waistcoat", "kurti", "3 piece unstitched", "saree", "dress"]
    genders = ["", "", "for women", "for men", "for kids", "for boys", "for girls"]
    seen, out = set(), []
    for _ in range(4000):
        q = " ".join(x for x in [rng.choice(colours), rng.choice(styles), rng.choice(garments), rng.choice(genders)] if x)
        if q in seen:
            continue
        seen.add(q)
        if sum(attr_ok(q, p) for p in prods) >= min_rel:
            out.append(q)
        if len(out) >= n:
            break
    return out


def p_at_k(ranked, prods, k=10):
    return float(np.mean([np.mean([attr_ok(q, prods[i]) for i in r[:k]]) for q, r in ranked]))


def _rrf(*rank_lists, k=60, n=50):
    sc = {}
    for lst in rank_lists:
        for r, d in enumerate(lst):
            sc[d] = sc.get(d, 0) + 1 / (k + r + 1)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])[:n]]


def _z(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-3 else 1.0)


def _load_images(prods):
    with_img = [i for i, p in enumerate(prods) if p["img"]]
    imgs = [Image.open(os.path.join(SAMPLE, prods[i]["img"])).convert("RGB") for i in with_img]
    return with_img, imgs


# ----------------------------------------------------------------- tasks
def task_t2i(prods, names):
    with_img, imgs = _load_images(prods)
    by_name = {}
    for row, i in enumerate(with_img):
        by_name.setdefault(prods[i]["name"].strip().lower(), set()).add(row)
    qnames = [n for n in by_name if good_name(n)]
    rel = [by_name[n] for n in qnames]
    print(f"[t2i] images={len(imgs)} unique-name queries={len(qnames)}")
    for name in names:
        t = time.time(); m = dual(name); tl = time.time() - t
        t = time.time(); I = m.images(imgs); ti = time.time() - t
        t = time.time(); Q = m.text(qnames); tq = time.time() - t
        ranked = np.argsort(-(Q @ I.T), axis=1)[:, :10]
        res = metrics(ranked, rel)
        rn = np.argsort(-(m.text(QUERIES) @ I.T), axis=1)[:, :10]
        pk = p_at_k([(q, [with_img[j] for j in r]) for q, r in zip(QUERIES, rn)], prods)
        print(f"  {name:45s} dim={I.shape[1]:4d} {res}  attrP@10={pk:.3f}  "
              f"| load {tl:.0f}s  img {len(imgs)/ti:.1f}/s  txt {len(qnames)/tq:.0f}/s")
        del m


def task_text(prods, names, rerankers):
    docs = [p["_text"] for p in prods]
    for name in names:
        t = time.time(); m = STText(name); tl = time.time() - t
        t = time.time(); D = m.docs(docs); td = time.time() - t
        t = time.time(); Q = m.queries(QUERIES); tq = time.time() - t
        order = np.argsort(-(Q @ D.T), axis=1)
        pk = p_at_k(list(zip(QUERIES, order[:, :10])), prods)
        print(f"  {name:36s} dim={D.shape[1]} semantic P@10={pk:.3f} | load {tl:.0f}s docs {len(docs)/td:.0f}/s q {tq/len(QUERIES)*1000:.0f}ms/q")
        for rn in rerankers:
            ce = CE(rn)
            t = time.time(); ranked, allscores = [], []
            for q, pool in zip(QUERIES, order[:, :50]):
                sc = ce.score([(q, prods[i]["_ce"]) for i in pool])
                allscores.append(sc)
                ranked.append((q, [pool[j] for j in np.argsort(-sc)]))
            tr = (time.time() - t) / len(QUERIES) * 1000
            a = np.concatenate(allscores)
            print(f"      + {rn:38s} P@10={p_at_k(ranked, prods):.3f}  {tr:.0f}ms/q  "
                  f"logits min/med/max={a.min():.2f}/{np.median(a):.2f}/{a.max():.2f} std={a.std():.2f}")
            del ce
        # title -> unique description (lexical gap)
        descs = {}
        for i, p in enumerate(prods):
            if len(p["_desc"].split()) >= 15:
                descs.setdefault(p["_desc"][:400], []).append(i)
        uniq = [(d, ids) for d, ids in descs.items() if len({prods[i]["name"].lower() for i in ids}) == 1]
        if len(uniq) >= 100:
            qn = [prods[ids[0]]["name"] for _, ids in uniq]
            keep = [k for k, n in enumerate(qn) if good_name(n)]
            Dd = m.docs([d for d, _ in uniq]); Qd = m.queries([qn[k] for k in keep])
            rk = np.argsort(-(Qd @ Dd.T), axis=1)[:, :10]
            print(f"      t2d (title->desc, {len(keep)} q / {len(uniq)} docs): {metrics(rk, [{k} for k in keep])}")
        del m


def task_pipeline(prods, text_name, ce_name, dual_name=None):
    from rank_bm25 import BM25Okapi
    from vectorizer.reranker import Reranker

    def tok(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    bm25 = BM25Okapi([tok(p["_text"]) for p in prods])
    tm = STText(text_name); St = tm.queries(QUERIES) @ tm.docs([p["_text"] for p in prods]).T
    ce = CE(ce_name)
    rr = Reranker.__new__(Reranker); rr._model = ce.m; rr.model_name = ce_name  # reuse loaded CE for its boost logic
    attrs = [{"colors": product_colors(p),
              "gender": extract_gender(p["category"], p["tags"], p["name"], p["productUrl"]),
              "product_type": classify_type(p["category"], p["tags"], p["name"])} for p in prods]
    Si = img_row = None
    if dual_name:
        with_img, imgs = _load_images(prods)
        dm = dual(dual_name); Si = dm.text(QUERIES) @ dm.images(imgs).T
        img_row = {i: r for r, i in enumerate(with_img)}
    rel = {q: {i for i, p in enumerate(prods) if attr_ok(q, p)} for q in QUERIES}
    ceiling = sum(min(len(rel[q]), 10) for q in QUERIES) / (10 * len(QUERIES))
    print(f"[pipeline] text={text_name} ce={ce_name} dual={dual_name}  P@10 ceiling={ceiling:.3f}")

    def run(cand_k, pool_n, filters, img, img_w=0.0):
        rows, t_ce = [], 0.0
        for qi, q in enumerate(QUERIES):
            qc, qg = extract_colors(q), extract_gender(name=q)
            apparel = bool(qc or qg)
            ok = np.array([not (apparel and a["product_type"] == "non_apparel") for a in attrs])
            sem = np.where(ok, St[qi], -9)
            ch = [list(np.argsort(-sem)[:cand_k]),
                  list(np.argsort(-np.where(ok, bm25.get_scores(tok(q)), -9))[:cand_k])]
            if filters and qc:
                m = np.array([bool(set(qc) & set(a["colors"])) for a in attrs])
                ch.append(list(np.argsort(-np.where(m, sem, -9))[:cand_k]))
            if filters and qg:
                m = np.array([a["gender"] == qg for a in attrs])
                ch.append(list(np.argsort(-np.where(m, sem, -9))[:cand_k]))
            if img and Si is not None:
                si = np.full(len(prods), -9.0)
                for i, r in img_row.items():
                    if ok[i]:
                        si[i] = Si[qi][r]
                ch.append(list(np.argsort(-si)[:cand_k]))
            pool = _rrf(*ch, n=pool_n)
            t = time.time(); cez = _z(ce.score([(q, prods[i]["_ce"]) for i in pool])); t_ce += time.time() - t
            score = cez + np.array([rr._boost(attrs[i], qc, qg) for i in pool])
            if img and img_w and Si is not None:
                score = score + img_w * _z([Si[qi][img_row[i]] if i in img_row else -1 for i in pool])
            rows.append((q, [pool[j] for j in np.argsort(-score)]))
        return p_at_k(rows, prods), p_at_k(rows, prods, 5), t_ce / len(QUERIES) * 1000

    variants = [
        ("production shape: sem+kw, k=40, pool=50", 40, 50, False, False, 0.0),
        ("+colour/gender channels", 40, 50, True, False, 0.0),
        ("+channels, k=60 pool=60", 60, 60, True, False, 0.0),
        ("+channels, k=80 pool=80", 80, 80, True, False, 0.0),
        ("+channels, k=100 pool=100", 100, 100, True, False, 0.0),
        ("no channels, k=80 pool=80", 80, 80, False, False, 0.0),
    ]
    if dual_name:
        variants += [
            ("+channels +image channel, pool=60", 60, 60, True, True, 0.0),
            ("+channels +image channel +img_w 0.5", 60, 60, True, True, 0.5),
            ("+channels +image channel +img_w 1.0", 60, 60, True, True, 1.0),
        ]
    for label, k, n, f, im, w in variants:
        p10, p5, ms = run(k, n, f, im, w)
        print(f"  {label:42s} P@10={p10:.3f} ({p10/ceiling:.0%} of ceiling)  P@5={p5:.3f}  rerank {ms:.0f}ms/q")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t2i", "text", "pipeline"])
    ap.add_argument("--models", default="", help="comma-separated; for pipeline: <text-model>[,<dual-model>]")
    ap.add_argument("--rerankers", default="", help="comma-separated cross-encoders")
    ap.add_argument("--no-desc", action="store_true", help="index without descriptions (old scraper behaviour)")
    ap.add_argument("--gen-queries", type=int, default=0,
                    help="replace the hand-written queries with N generated template queries (>=5 relevant each)")
    a = ap.parse_args()
    USE_DESC = not a.no_desc
    prods = load()
    if a.gen_queries:
        QUERIES[:] = generate_queries(prods, a.gen_queries)
    print(f"products={len(prods)} device={DEVICE} descriptions indexed={USE_DESC} queries={len(QUERIES)}")
    models = [m for m in a.models.split(",") if m]
    rr = [m for m in a.rerankers.split(",") if m]
    if a.task == "t2i":
        task_t2i(prods, models)
    elif a.task == "text":
        task_text(prods, models, rr)
    elif a.task == "pipeline":
        task_pipeline(prods, models[0], rr[0], models[1] if len(models) > 1 else None)
