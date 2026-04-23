"""
Quick search test — run a few queries and print top results.
Usage: python test_search.py
"""
import sys, json
sys.path.insert(0, '.')

from vectorizer.search import search

queries = [
    "blue lawn suit casual",
    "embroidered formal dress for wedding",
    "men kurta shalwar white",
    "black trousers wide leg",
    "kids summer outfit",
]

for q in queries:
    print(f"\n{'='*50}")
    print(f"Query: \"{q}\"")
    print('='*50)
    results = search(q, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['brand']}] {r['name']}")
        print(f"     Price: Rs.{r['price']}  |  Score: {r['score']:.3f}")
        print(f"     Tags: {', '.join(r.get('tags', [])[:4])}")
