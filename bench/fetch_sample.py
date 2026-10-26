"""
Pull a real product sample straight from the brand stores (public Shopify
products.json, no DB needed) plus thumbnails, for bench.py.

    python bench/fetch_sample.py                 # -> bench/sample/{products.json, img/}
    python bench/fetch_sample.py --pages 2       # fewer products per brand

Products are mapped exactly like scrapers/shopify.py so the benchmark indexes
what the pipeline would index.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.brands import SCRAPERS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
THUMB_WIDTH = 320  # SigLIP/CLIP inputs are 224-256px; smaller thumbnails download faster


def fetch_brand(scraper_cls, pages: int, session: requests.Session) -> list[dict]:
    s = scraper_cls()
    s.delay = 0.3
    products = []
    for page in range(1, pages + 1):
        try:
            res = session.get(f"{s.base_url}/products.json?limit=250&page={page}", timeout=20)
            if res.status_code != 200:
                print(f"  {s.brand_name} p{page}: HTTP {res.status_code}", flush=True)
                break
            data = res.json().get("products", [])
        except Exception as e:
            print(f"  {s.brand_name} p{page}: {e}", flush=True)
            break
        if not data:
            break
        for p in data:
            mapped = s._map(p)
            if mapped and mapped["imageUrl"]:
                products.append(mapped)
        print(f"  {s.brand_name} p{page}: {len(data)}", flush=True)
        time.sleep(0.3)
    return products


def download_thumb(args) -> tuple[int, bool]:
    i, url, session = args
    sep = "&" if "?" in url else "?"
    try:
        r = session.get(f"{url}{sep}width={THUMB_WIDTH}", timeout=20)
        r.raise_for_status()
        with open(os.path.join(OUT, "img", f"{i}.jpg"), "wb") as f:
            f.write(r.content)
        return i, True
    except Exception:
        return i, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="pages of 250 per brand")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"})

    products = []
    for name, cls in SCRAPERS.items():
        print(f"→ {name}", flush=True)
        products += fetch_brand(cls, a.pages, session)
    print(f"total products: {len(products)}", flush=True)

    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(download_thumb, [(i, p["imageUrl"], session) for i, p in enumerate(products)]))
    ok = {i for i, success in results if success}
    for i, p in enumerate(products):
        p["img"] = f"img/{i}.jpg" if i in ok else None
    print(f"images ok: {len(ok)}/{len(products)}", flush=True)

    with open(os.path.join(OUT, "products.json"), "w") as f:
        json.dump(products, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
