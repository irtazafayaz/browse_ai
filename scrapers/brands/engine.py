from scrapers.shopify import ShopifyScraper


class EngineScraper(ShopifyScraper):
    brand_name = "Engine"
    base_url   = "https://www.engine.com.pk"
    source     = "engine"
