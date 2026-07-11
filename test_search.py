"""
Quick search test — exercises all three search modes directly (no HTTP server).
Usage:
    python test_search.py                       # text + text->image
    python test_search.py --image path/to.jpg   # also image->image
"""
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from vectorizer.embedder import Embedder
from vectorizer.clip_embedder import ClipEmbedder
from db.qdrant import Qdrant
from db.mongo import get_products_col
from vectorizer.search import search, text_to_image_search, image_search

TEXT_QUERIES = [
    "blue lawn suit casual",
    "embroidered formal dress for wedding",
    "men kurta shalwar white",
]
VISUAL_QUERIES = [
    "red floral maxi dress",
    "black embroidered outfit",
]


def _show(results):
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.get('brand')}] {(r.get('name') or '')[:55]}")
        print(f"     Rs.{r.get('price')}  |  score {r.get('score')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to an image for image->image search")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    embedder = Embedder()
    clip = ClipEmbedder()
    qdrant = Qdrant(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    col = get_products_col()

    print("\n########## TEXT (hybrid: semantic + keyword) ##########")
    for q in TEXT_QUERIES:
        print(f'\nQuery: "{q}"')
        _show(search(q, embedder=embedder, qdrant=qdrant, col=col, top_k=args.top_k))

    print("\n########## TEXT -> IMAGE (cross-modal CLIP) ##########")
    for q in VISUAL_QUERIES:
        print(f'\nQuery: "{q}"')
        _show(text_to_image_search(q, clip_embedder=clip, qdrant=qdrant, top_k=args.top_k))

    if args.image:
        print(f"\n########## IMAGE -> IMAGE ({args.image}) ##########")
        with open(args.image, "rb") as f:
            img = clip.load_image_bytes(f.read())
        if img is None:
            print("  Could not decode image.")
        else:
            _show(image_search(clip.embed_image_one(img), qdrant=qdrant, top_k=args.top_k))


if __name__ == "__main__":
    main()
