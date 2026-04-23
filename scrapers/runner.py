"""
Scraper runner.
Imports all brand scrapers, runs them, and saves results to MongoDB.

Usage:
    python -m scrapers.runner
    python -m scrapers.runner --brand sana_safinaz
    python -m scrapers.runner --clear
"""
import argparse
from db.mongo import get_products_col
from scrapers.brands import SCRAPERS


def run(brand_filter: str = None, clear: bool = False):
    col = get_products_col()

    if clear:
        col.delete_many({})
        print("Cleared products collection.")

    scrapers = SCRAPERS
    if brand_filter:
        scrapers = {k: v for k, v in SCRAPERS.items() if k == brand_filter}
        if not scrapers:
            print(f"Unknown brand: {brand_filter}. Available: {list(SCRAPERS)}")
            return

    total = 0
    for name, ScraperClass in scrapers.items():
        print(f"\n→ Scraping {name}...")
        try:
            products = ScraperClass().scrape()
            if products:
                col.insert_many(products, ordered=False)
                print(f"  ✓ {len(products)} products saved")
                total += len(products)
            else:
                print(f"  ⚠ No products returned")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nDone. Total products saved: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", help="Run a single brand scraper")
    parser.add_argument("--clear", action="store_true", help="Clear collection before scraping")
    args = parser.parse_args()
    run(brand_filter=args.brand, clear=args.clear)
