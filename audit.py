import sys, json
sys.path.insert(0, '.')
from db.mongo import get_products_col
col = get_products_col()

print(f"Total products:       {col.count_documents({})}")
print(f"Missing imageUrl:     {col.count_documents({'imageUrl': ''})}")
print(f"Price is null:        {col.count_documents({'price': None})}")
print(f"Price is 0:           {col.count_documents({'price': 0})}")
print(f"Missing name:         {col.count_documents({'name': ''})}")

print("\n--- Sample products (3) ---")
for doc in col.find({}, {'name':1,'brand':1,'tags':1,'price':1,'originalPrice':1,'category':1}).limit(3):
    print(json.dumps({
        'brand': doc.get('brand'),
        'name': doc.get('name'),
        'price': doc.get('price'),
        'originalPrice': doc.get('originalPrice'),
        'category': doc.get('category'),
        'tags': doc.get('tags', [])
    }, indent=2))

pipeline = [
    {"$group": {"_id": {"brand": "$brand", "name": "$name"}, "count": {"$sum": 1}}},
    {"$match": {"count": {"$gt": 1}}},
    {"$count": "duplicate_groups"}
]
result = list(col.aggregate(pipeline))
print(f"\nDuplicate groups (same brand+name): {result[0]['duplicate_groups'] if result else 0}")

# Per brand count
print("\n--- Products per brand ---")
pipeline2 = [
    {"$group": {"_id": "$brand", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for r in col.aggregate(pipeline2):
    print(f"  {r['_id']:<25} {r['count']}")
