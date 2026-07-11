"""
Attribute extraction — shared by indexing (enrich text + structured payload)
and querying (parse the query into filters). Pure functions, no I/O, no models.

Grounded in the real catalog: color is usually in the product NAME, tags are
mostly marketing noise, and gender/type live in category + a few tags.
"""
import re
from typing import Optional

# --- Colors: canonical family -> synonyms. Query and product colors are both
# normalized to families, so "maroon"/"crimson" match a search for "red". ---
COLOR_FAMILIES = {
    "red":    ["red", "crimson", "scarlet", "maroon", "wine", "burgundy", "ruby", "cherry"],
    "pink":   ["pink", "fuchsia", "magenta", "rose", "blush", "coral"],
    "orange": ["orange", "rust", "peach", "apricot", "tangerine", "terracotta"],
    "yellow": ["yellow", "mustard", "lemon", "ochre"],
    "gold":   ["gold", "golden"],
    "green":  ["green", "olive", "mint", "sea foam", "sea green", "emerald", "pistachio", "teal"],
    "blue":   ["blue", "navy", "cobalt", "denim", "sky", "turquoise", "azure", "indigo"],
    "purple": ["purple", "lilac", "mauve", "lavender", "violet", "plum", "aubergine"],
    "brown":  ["brown", "tan", "beige", "camel", "coffee", "chocolate", "taupe", "khaki", "mocha", "sand"],
    "white":  ["white", "ivory", "cream", "off white", "off-white", "pearl"],
    "black":  ["black", "charcoal", "jet", "onyx"],
    "grey":   ["grey", "gray", "silver", "ash", "slate", "steel"],
}
# synonym -> family, longest-first so multiword ("sea green") wins over "green"
_COLOR_SYNONYMS = sorted(
    ((syn, fam) for fam, syns in COLOR_FAMILIES.items() for syn in syns),
    key=lambda x: -len(x[0]),
)

_GENDER_KIDS = re.compile(r"\b(kidswear|kid|kids|child|children|boy|boys|girl|girls|infant|toddler|junior)\b")
_GENDER_MEN = re.compile(r"\b(menswear|men|man|mens|gent|gents|gentleman|gentlemen|male)\b")
_GENDER_WOMEN = re.compile(r"\b(womenswear|women|woman|womens|ladies|lady|female)\b")

# Categories/keywords that are NOT clothing — a garment/color search should not
# surface these. NB: "Salt" is a Sana Safinaz clothing line, so it's NOT here.
_NON_APPAREL = re.compile(
    r"\b(home|homeware|fragrance|fragrances|perfume|candle|bedding|bedsheet|cushion|"
    r"decor|kitchen|tableware|pottery|crockery|towel)\b"
)

# Light style/fabric/garment lexicon — clean signal to enrich the embedding text.
_STYLE_TERMS = [
    "embroidered", "printed", "plain", "dyed", "woven", "handwork",
    "lawn", "silk", "cotton", "chiffon", "organza", "net", "khaddar", "linen", "velvet", "jamawar",
    "kurta", "kurti", "kameez", "shalwar", "trousers", "dupatta", "saree", "sari", "lehenga",
    "frock", "gharara", "sharara", "culotte", "co-ord", "waistcoat", "shirt", "abaya",
    "unstitched", "stitched", "pret", "ready to wear",
    "formal", "casual", "party", "wedding", "bridal", "eid", "festive", "luxury",
]
_STYLE_RE = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in _STYLE_TERMS}


def extract_colors(text: str) -> list[str]:
    """Return canonical color families found in text (word-boundary matched)."""
    if not text:
        return []
    t = text.lower()
    found = []
    for syn, fam in _COLOR_SYNONYMS:
        if fam in found:
            continue
        pattern = syn.replace(" ", r"[\s-]")
        if re.search(r"\b" + pattern + r"\b", t):
            found.append(fam)
    return sorted(found)


def extract_gender(category: str = "", tags=None, name: str = "", url: str = "") -> Optional[str]:
    hay = " ".join([category or "", " ".join(tags or []), name or "", url or ""]).lower()
    if _GENDER_KIDS.search(hay):
        return "kids"
    if _GENDER_MEN.search(hay):
        return "men"
    if _GENDER_WOMEN.search(hay):
        return "women"
    return None


def classify_type(category: str = "", tags=None, name: str = "") -> str:
    hay = " ".join([category or "", " ".join(tags or []), name or ""]).lower()
    return "non_apparel" if _NON_APPAREL.search(hay) else "apparel"


def _extract_styles(text: str) -> list[str]:
    t = (text or "").lower()
    return [term for term, rx in _STYLE_RE.items() if rx.search(t)]


def enrichment_text(product: dict) -> str:
    """Clean, deduped attribute words to APPEND to the embedding text — colors,
    gender, and style/fabric/garment — pulled from name + tags, minus the noise."""
    name = product.get("name") or ""
    tagblob = " ".join(product.get("tags") or [])
    blob = f"{name} {tagblob} {product.get('category') or ''}"

    colors = extract_colors(blob)
    gender = extract_gender(product.get("category", ""), product.get("tags"), name, product.get("productUrl", ""))
    styles = _extract_styles(blob)

    parts = colors + ([gender] if gender else []) + styles
    # de-dupe preserving order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return " ".join(out)


def structured_attributes(product: dict) -> dict:
    """Filterable fields to store in the Qdrant payload."""
    name = product.get("name") or ""
    return {
        "colors": extract_colors(f"{name} {' '.join(product.get('tags') or [])}"),
        "gender": extract_gender(product.get("category", ""), product.get("tags"), name, product.get("productUrl", "")),
        "product_type": classify_type(product.get("category", ""), product.get("tags"), name),
    }


def parse_query(query: str) -> dict:
    """Extract structured filters from a search query."""
    return {
        "colors": extract_colors(query),
        "gender": extract_gender(name=query),
    }
