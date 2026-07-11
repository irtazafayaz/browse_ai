"""
Health / status dashboard.

- GET /api/v1/status  -> JSON health of every component (each checked in isolation
  so one failure never breaks the rest of the report)
- GET /dashboard      -> self-contained HTML page that polls the JSON endpoint
"""
import logging
import os
import time

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse

from api.dependencies import get_embedder, get_clip_embedder, get_qdrant
from db.mongo import get_db
from db.qdrant import TEXT_VECTOR, IMAGE_VECTOR

logger = logging.getLogger(__name__)

router = APIRouter()


def _timed(fn) -> dict:
    """Run a check, capturing latency and turning any exception into a
    structured 'down' report rather than a 500."""
    start = time.time()
    try:
        result = fn()
        result["ok"] = True
    except Exception as e:
        logger.warning("Health check failed: %s", e)
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result


def _mongo_status() -> dict:
    db = get_db()
    col = db["products"]
    # estimated_document_count() reads collection metadata — instant, no scan.
    count = col.estimated_document_count()
    index_keys = [list(idx["key"].keys()) for idx in col.list_indexes()]
    has_text_index = any("_fts" in keys for keys in index_keys)
    has_brand_index = any(keys == ["brand"] for keys in index_keys)
    # Bounded so a slow filtered scan can't hang the dashboard poll.
    try:
        with_images = col.count_documents({"imageUrl": {"$nin": ["", None]}}, maxTimeMS=4000)
    except Exception:
        with_images = None
    return {
        "database": db.name,
        "collection": "products",
        "products": count,
        "products_with_image": with_images,
        "text_index": has_text_index,
        "brand_index": has_brand_index,
    }


def _qdrant_status() -> dict:
    qdrant = get_qdrant()
    collections = [c.name for c in qdrant._client.get_collections().collections]
    exists = qdrant.collection_name in collections
    out = {
        "collection": qdrant.collection_name,
        "collection_exists": exists,
    }
    if exists:
        info = qdrant._client.get_collection(qdrant.collection_name)
        vectors = info.config.params.vectors
        named = isinstance(vectors, dict)
        out["points"] = qdrant._client.count(qdrant.collection_name).count
        out["named_vectors"] = named
        if named:
            out["vectors"] = {name: params.size for name, params in vectors.items()}
            out["image_search_ready"] = IMAGE_VECTOR in vectors and TEXT_VECTOR in vectors
        else:
            out["vectors"] = {"(unnamed)": getattr(vectors, "size", None)}
            out["image_search_ready"] = False
    return out


def _models_status() -> dict:
    # cache_info().currsize > 0 means the singleton is already warmed —
    # read it WITHOUT forcing a (heavy) load.
    return {
        "text_embedder_loaded": get_embedder.cache_info().currsize > 0,
        "clip_embedder_loaded": get_clip_embedder.cache_info().currsize > 0,
    }


@router.get("/status", tags=["system"])
def status(response: Response):
    components = {
        "mongodb": _timed(_mongo_status),
        "qdrant": _timed(_qdrant_status),
        "models": _timed(_models_status),
    }
    all_ok = all(c["ok"] for c in components.values())
    if not all_ok:
        response.status_code = 503  # readiness probe: degraded backend
    return {
        "status": "ok" if all_ok else "degraded",
        "components": components,
    }


@router.get("/dashboard", response_class=HTMLResponse, tags=["system"])
def dashboard():
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Browse AI — Status</title>
<style>
  :root { --ok:#16a34a; --bad:#dc2626; --warn:#d97706; --bg:#0b1120; --card:#111a2e; --line:#1f2b45; --muted:#93a4c4; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:#e6edf6; padding:32px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:24px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; }
  .card h2 { font-size:16px; margin:0 0 12px; display:flex; align-items:center; gap:8px; text-transform:capitalize; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .row { display:flex; justify-content:space-between; padding:5px 0; border-top:1px solid var(--line); }
  .row:first-of-type { border-top:none; }
  .k { color:var(--muted); }
  .v { font-variant-numeric:tabular-nums; font-weight:600; }
  .pill { font-size:12px; padding:2px 8px; border-radius:999px; font-weight:600; }
  .banner { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; margin-bottom:20px; }
  .err { color:var(--bad); font-size:13px; word-break:break-word; }
  .muted { color:var(--muted); font-size:13px; }
  button { background:#1d2b4a; color:#e6edf6; border:1px solid var(--line); border-radius:8px; padding:6px 12px; cursor:pointer; }
</style>
</head>
<body>
  <h1>Browse AI — System Status</h1>
  <div class="sub">Live health of MongoDB, Qdrant and the embedding models.
     <button onclick="load()">Refresh</button>
     <span class="muted" id="ts"></span></div>
  <div id="banner"></div>
  <div class="grid" id="grid"></div>

<script>
const OK='var(--ok)', BAD='var(--bad)', WARN='var(--warn)';
function bool(v){ return `<span class="pill" style="background:${v?OK:BAD}22;color:${v?OK:BAD}">${v?'yes':'no'}</span>`; }
function num(v){ return (v==null)?'—':Number(v).toLocaleString(); }

function rows(name, c){
  if(!c.ok) return `<div class="row"><span class="k">error</span></div><div class="err">${c.error||'unreachable'}</div>`;
  const skip = new Set(['ok','latency_ms']);
  let html='';
  for(const [k,val] of Object.entries(c)){
    if(skip.has(k)) continue;
    let disp;
    if(typeof val==='boolean') disp=bool(val);
    else if(typeof val==='number') disp=`<span class="v">${num(val)}</span>`;
    else if(val && typeof val==='object') disp=`<span class="v">${Object.entries(val).map(([a,b])=>a+':'+b).join('  ')}</span>`;
    else disp=`<span class="v">${val}</span>`;
    html+=`<div class="row"><span class="k">${k.replace(/_/g,' ')}</span>${disp}</div>`;
  }
  html+=`<div class="row"><span class="k">latency</span><span class="v">${c.latency_ms} ms</span></div>`;
  return html;
}

async function load(){
  const grid=document.getElementById('grid');
  const banner=document.getElementById('banner');
  try{
    const r=await fetch('/api/v1/status');
    const d=await r.json();
    const ok = d.status==='ok';
    banner.innerHTML=`<span class="banner" style="background:${ok?OK:WARN}22;color:${ok?OK:WARN}">${d.status.toUpperCase()}</span>`;
    grid.innerHTML=Object.entries(d.components).map(([name,c])=>
      `<div class="card"><h2><span class="dot" style="background:${c.ok?OK:BAD}"></span>${name}</h2>${rows(name,c)}</div>`
    ).join('');
    document.getElementById('ts').textContent='updated '+new Date().toLocaleTimeString();
  }catch(e){
    banner.innerHTML=`<span class="banner" style="background:${BAD}22;color:${BAD}">API UNREACHABLE</span>`;
    grid.innerHTML=`<div class="card err">${e}</div>`;
  }
}
load();
setInterval(load, 10000);
</script>
</body>
</html>"""
