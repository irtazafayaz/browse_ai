"""
Data cleaner for scraped products.

Fixes:
  1. Remove products with no image or zero price
  2. Remove duplicates (same brand + name — keep best record)
  3. Clean product names (strip "RTS |", "RTW |" prefixes, title case)
  4. Clean tags (remove SKUs, internal nav tags, normalize case, dedupe)
  5. Normalize category (title case)

Run:
    python clean.py          # preview only (dry run)
    python clean.py --apply  # apply changes to MongoDB
"""
from __future__ import annotations
import re
import sys
sys.path.insert(0, '.')

from db.mongo import get_products_col

DRY_RUN   = "--apply"     not in sys.argv
TAGS_ONLY = "--tags-only" in sys.argv

# ── Tag cleaning rules ────────────────────────────────────────────────────────

# Tags that are internal navigation / junk (exact match, lowercased)
JUNK_TAG_EXACT = {
    "shop by product", "man", "woman", "new in",
    "women rts", "men rtw", "women rtw", "men-rtw", "women-rts",
    "all summer articles", "all winter articles",
}

# Tags matching these patterns are internal codes / dates / SKUs
JUNK_TAG_PATTERNS = [
    re.compile(r"^[A-Z0-9]{2,}-[A-Z0-9\-]+$"),              # SKU: MKS-STY26-006
    re.compile(r"simplifiedsizechart", re.I),                 # size chart tags
    re.compile(r"^(new|sale)-\d+$", re.I),                   # New-30, Sale-20
    re.compile(r"vol-\d+", re.I),                             # Vol-1, Vol-2
    re.compile(r"^\d{1,4}%?$"),                               # bare numbers or "18%"
    re.compile(r"essentials$", re.I),                         # WOMEN-RTS-ESSENTIALS
    re.compile(r"^[a-z]+-\d{2}$", re.I),                     # Summer-26, Winter-25
    re.compile(r"^\w+-\d{2}\s+vol", re.I),                   # Winter-25 Vol-1
    re.compile(r"\d{1,2}[-/](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I),  # 2-Mar-Live
    re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-/]\d{2}", re.I),    # Nov-25
    re.compile(r"\d+(st|nd|rd|th)[-/]", re.I),               # 25Th-Feb-26
    re.compile(r"dhlcode|dhldes", re.I),                      # DHL shipping codes
    re.compile(r"^_"),                                         # _Label_New
    re.compile(r".*:.*"),                                      # anything with colon: Dhlcode:6204
    re.compile(r"^[a-z]\d{2}-\d{2}", re.I),                  # F20-40, C-Report
    re.compile(r"c-report", re.I),                            # internal report tags
    re.compile(r"-live$|-to$|-pret$", re.I),                  # schedule tags: 2-Mar-Live
    re.compile(r"pret-to|pret-live", re.I),                   # 26-Feb-Pret-To
    re.compile(r"^all-\w+$", re.I),                          # All-Jalabiyas navigation
]

# Normalize synonyms → single canonical form
TAG_SYNONYMS = {
    "womens": "Women",
    "woman":  "Women",
    "mens":   "Men",
    "man":    "Men",
    "readytowear": "Ready to Wear",
    "readytostitch": "Ready to Stitch",
    "unstitched": "Unstitched",
    "stitched": "Stitched",
}


def clean_tag(tag: str) -> str | None:
    tag = tag.strip()
    if not tag:
        return None

    lower = tag.lower()

    # Exact junk
    if lower in JUNK_TAG_EXACT:
        return None

    # Pattern junk
    for pattern in JUNK_TAG_PATTERNS:
        if pattern.search(tag):
            return None

    # Synonym normalization
    if lower.replace(" ", "") in TAG_SYNONYMS:
        return TAG_SYNONYMS[lower.replace(" ", "")]

    # Title case, max length guard
    cleaned = tag.title().strip()
    if len(cleaned) > 50 or len(cleaned) < 2:
        return None

    return cleaned


def clean_tags(raw: list) -> list:
    seen = set()
    result = []
    for t in raw:
        cleaned = clean_tag(str(t))
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result


# ── Name cleaning ─────────────────────────────────────────────────────────────

NAME_PREFIX_RE = re.compile(
    r"^(RTS|RTW|USS|US|RTW-[A-Z]+|RTS-[A-Z]+)\s*[|\-]\s*",
    re.IGNORECASE,
)


def clean_name(name: str) -> str:
    # Strip internal prefixes like "RTS | ", "RTW | "
    name = NAME_PREFIX_RE.sub("", name).strip()
    # Title case (but preserve all-caps abbreviations under 4 chars)
    words = []
    for word in name.split():
        if word.isupper() and len(word) <= 3:
            words.append(word)        # keep "II", "XL", etc.
        else:
            words.append(word.title())
    return " ".join(words)


# ── Main cleaner ──────────────────────────────────────────────────────────────

def _clean_fields(col) -> int:
    """Clean names, tags and categories on all docs. Returns count updated."""
    from pymongo import UpdateOne
    print("\n[4] Cleaning names, tags, categories...")
    cursor = col.find({}, {"name": 1, "tags": 1, "category": 1})
    ops = []
    for doc in cursor:
        ops.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {
                "name":     clean_name(doc.get("name", "")),
                "tags":     clean_tags(doc.get("tags", [])),
                "category": doc.get("category", "").title().strip(),
            }}
        ))
    print(f"    Documents to update: {len(ops)}")
    if not DRY_RUN:
        batch_size = 500
        for i in range(0, len(ops), batch_size):
            col.bulk_write(ops[i:i + batch_size], ordered=False)
            if i % 5000 == 0 and i > 0:
                print(f"    Updated {i}/{len(ops)}...")
        print(f"    Done.")
    return len(ops)


def run():
    col = get_products_col()
    total_before = col.count_documents({})
    print(f"Products before: {total_before}")

    removed_no_image = 0
    removed_zero_price = 0
    removed_duplicates = 0
    cleaned_fields = 0

    if TAGS_ONLY:
        print("--tags-only: skipping steps 1–3, re-cleaning fields only.\n")
        _clean_fields(col)
        return

    # 1. Remove products with no image
    ids_no_image = [d["_id"] for d in col.find({"imageUrl": ""})]
    print(f"\n[1] No image: {len(ids_no_image)} products")
    if not DRY_RUN and ids_no_image:
        col.delete_many({"_id": {"$in": ids_no_image}})
        removed_no_image = len(ids_no_image)

    # 2. Remove products with zero price
    ids_zero_price = [d["_id"] for d in col.find({"price": 0})]
    print(f"[2] Zero price: {len(ids_zero_price)} products")
    if not DRY_RUN and ids_zero_price:
        col.delete_many({"_id": {"$in": ids_zero_price}})
        removed_zero_price = len(ids_zero_price)

    # 3. Deduplicate — same brand + name, keep first seen
    print("\n[3] Finding duplicates (in-memory scan)...")
    seen = {}       # (brand, name) → _id to keep
    ids_to_delete = []

    cursor = col.find({}, {"brand": 1, "name": 1}, batch_size=500)
    for doc in cursor:
        key = (doc.get("brand", ""), doc.get("name", ""))
        if key in seen:
            ids_to_delete.append(doc["_id"])
        else:
            seen[key] = doc["_id"]

    print(f"    Unique products:          {len(seen)}")
    print(f"    Duplicate docs to remove: {len(ids_to_delete)}")
    if not DRY_RUN and ids_to_delete:
        # Delete in batches
        batch_size = 1000
        for i in range(0, len(ids_to_delete), batch_size):
            col.delete_many({"_id": {"$in": ids_to_delete[i:i+batch_size]}})
        removed_duplicates = len(ids_to_delete)

    # 4. Clean names, tags, categories on remaining docs
    cleaned_fields = _clean_fields(col)

    # 5. Summary
    total_after = col.count_documents({}) if not DRY_RUN else total_before
    print(f"""
{'='*45}
{'DRY RUN — no changes made' if DRY_RUN else 'DONE'}
{'='*45}
Removed (no image):    {removed_no_image}
Removed (zero price):  {removed_zero_price}
Removed (duplicates):  {removed_duplicates}
Fields cleaned:        {cleaned_fields}
Products before:       {total_before}
Products after:        {total_after}
""")

    # Show sample cleaned product
    print("--- Sample cleaned product ---")
    import json
    sample = col.find_one({})
    if sample:
        sample.pop("_id", None)
        print(json.dumps({k: v for k, v in sample.items() if k != "embedding"}, indent=2))


if __name__ == "__main__":
    if DRY_RUN:
        print("DRY RUN mode — pass --apply to make changes\n")
    run()
