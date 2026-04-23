from scrapers.brands.sana_safinaz import SanaSafinazScraper
from scrapers.brands.alkaram     import AlkaramScraper
from scrapers.brands.gul_ahmed   import GulAhmedScraper
from scrapers.brands.nishat_linen import NishatLinenScraper
from scrapers.brands.maria_b     import MariaBScraper
from scrapers.brands.limelight   import LimelightScraper
from scrapers.brands.generation  import GenerationScraper
from scrapers.brands.bonanza     import BonanzaScraper
from scrapers.brands.one         import ONEScraper
from scrapers.brands.engine      import EngineScraper

SCRAPERS = {
    "sana_safinaz":  SanaSafinazScraper,
    "alkaram":       AlkaramScraper,
    "gul_ahmed":     GulAhmedScraper,
    "nishat_linen":  NishatLinenScraper,
    "maria_b":       MariaBScraper,
    "limelight":     LimelightScraper,
    "generation":    GenerationScraper,
    "bonanza":       BonanzaScraper,
    "one":           ONEScraper,
    "engine":        EngineScraper,
}
