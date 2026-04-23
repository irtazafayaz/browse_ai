from scrapers.shopify import ShopifyScraper


class GenerationScraper(ShopifyScraper):
    brand_name = "Generation"
    base_url   = "https://www.generation.com.pk"
    source     = "generation"
