"""
JSON query CLI — one entry point for all three search modes. Prints a JSON
object to stdout so it can be driven programmatically (tests, workflows).

Usage:
    python query_cli.py --mode text   --q "blue lawn suit"
    python query_cli.py --mode visual --q "red floral dress"     # text -> image
    python query_cli.py --mode image  --image-url "https://..."  # image -> image
    python query_cli.py --sample-image-url                        # print a product image URL from Mongo
"""
import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def _lazy():
    from vectorizer.embedder import Embedder
    from vectorizer.clip_embedder import ClipEmbedder
    from db.qdrant import Qdrant
    from db.mongo import get_products_col
    return Embedder, ClipEmbedder, Qdrant, get_products_col


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["text", "visual", "image"])
    p.add_argument("--q")
    p.add_argument("--image-url")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--brand")
    p.add_argument("--sample-image-url", action="store_true",
                   help="Print one product's imageUrl from Mongo and exit")
    args = p.parse_args()

    Embedder, ClipEmbedder, Qdrant, get_products_col = _lazy()

    if args.sample_image_url:
        col = get_products_col()
        doc = col.find_one({"imageUrl": {"$exists": True, "$ne": ""}},
                           {"imageUrl": 1, "name": 1, "brand": 1})
        print(json.dumps({"imageUrl": doc.get("imageUrl"), "name": doc.get("name"),
                          "brand": doc.get("brand")}))
        return

    from vectorizer.search import search, text_to_image_search, image_search

    qdrant = Qdrant(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))

    if args.mode == "text":
        embedder = Embedder()
        col = get_products_col()
        results = search(args.q, embedder=embedder, qdrant=qdrant, col=col,
                         top_k=args.top_k, brand=args.brand)
    elif args.mode == "visual":
        clip = ClipEmbedder()
        results = text_to_image_search(args.q, clip_embedder=clip, qdrant=qdrant,
                                       top_k=args.top_k, brand=args.brand)
    elif args.mode == "image":
        clip = ClipEmbedder()
        vec = clip.embed_image_url(args.image_url)
        if vec is None:
            print(json.dumps({"error": "could not load image", "url": args.image_url}))
            sys.exit(1)
        results = image_search(vec, qdrant=qdrant, top_k=args.top_k, brand=args.brand)
    else:
        print(json.dumps({"error": "specify --mode or --sample-image-url"}))
        sys.exit(2)

    slim = [{"brand": r.get("brand"), "name": r.get("name"),
             "price": r.get("price"), "score": r.get("score")} for r in results]
    print(json.dumps({"mode": args.mode, "query": args.q or args.image_url,
                      "count": len(slim), "results": slim}, ensure_ascii=False))


if __name__ == "__main__":
    main()
