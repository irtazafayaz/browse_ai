from scrapers.shopify import ShopifyScraper


class LimelightScraper(ShopifyScraper):
    brand_name = "Limelight"
    base_url   = "https://www.limelight.pk"
    source     = "limelight"
