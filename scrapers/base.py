"""
Base scraper class.
All brand scrapers inherit from this and implement `scrape()`.
"""
import time
import requests
from abc import ABC, abstractmethod


class BaseScraper(ABC):
    brand_name: str = ""
    base_url: str = ""
    delay: float = 1.0  # seconds between requests

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    @abstractmethod
    def scrape(self) -> list[dict]:
        """
        Scrape products from the brand site.
        Must return a list of product dicts with these fields:
            name        str
            brand       str
            price       float
            imageUrl    str
            productUrl  str
            tags        list[str]
            description str  (optional)
        """

    def get(self, url: str, **kwargs) -> requests.Response:
        time.sleep(self.delay)
        return self.session.get(url, timeout=15, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        time.sleep(self.delay)
        return self.session.post(url, timeout=15, **kwargs)
