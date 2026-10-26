"""
Generic Shopify scraper.
All 10 brands use Shopify — this handles pagination and field mapping.
Each brand scraper just sets brand_name, base_url, and source.
"""
import re
from typing import Optional
from bs4 import BeautifulSoup
from scrapers.base import BaseScraper

MAX_DESCRIPTION_CHARS = 1200


class ShopifyScraper(BaseScraper):
    source: str = ""
    delay: float = 0.8

    def scrape(self) -> list[dict]:
        products = []
        page = 1

        while True:
            url = f"{self.base_url}/products.json?limit=250&page={page}"
            res = self.get(url)

            if res.status_code != 200:
                print(f"  Request failed: {res.status_code}")
                break

            data = res.json().get("products", [])
            if not data:
                break

            for p in data:
                product = self._map(p)
                if product:
                    products.append(product)

            print(f"  Page {page}: {len(data)} products")
            page += 1

        return products

    def _map(self, p: dict) -> Optional[dict]:
        title = p.get("title", "").strip()
        if not title:
            return None

        # Price — prefer first available variant
        price = None
        original_price = None
        for v in p.get("variants", []):
            if v.get("available", True):
                price = float(v["price"]) if v.get("price") else None
                cp = v.get("compare_at_price")
                original_price = float(cp) if cp else None
                break

        # Fall back to first variant
        if price is None and p.get("variants"):
            v = p["variants"][0]
            price = float(v["price"]) if v.get("price") else None
            cp = v.get("compare_at_price")
            original_price = float(cp) if cp else None

        # Main image
        image_url = p["images"][0]["src"] if p.get("images") else ""

        # Tags — strip internal Shopify upload/date tags
        skip_prefixes = ("uploaded-", "uploaded_", "upload-")
        tags = [
            t for t in p.get("tags", [])
            if not any(t.lower().startswith(s) for s in skip_prefixes)
        ]

        handle = p.get("handle", "")

        return {
            "brand":         self.brand_name,
            "name":          title,
            "price":         price,
            "originalPrice": original_price,
            "imageUrl":      image_url,
            "productUrl":    f"{self.base_url}/products/{handle}",
            "category":      p.get("product_type", ""),
            "tags":          tags,
            "description":   self._clean_description(p.get("body_html")),
            "source":        self.source,
        }

    @staticmethod
    def _clean_description(body_html: Optional[str]) -> str:
        """Shopify body_html -> plain text. Brands put the facts that matter for
        search here (Color, Fabric, Cut, Dupatta/Trouser included...) and nowhere
        else, so this feeds the embedding text, the Mongo text index and the
        reranker. Capped so a boilerplate care-guide can't swamp the vector."""
        if not body_html:
            return ""
        text = BeautifulSoup(body_html, "html.parser").get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_DESCRIPTION_CHARS]
