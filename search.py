import os, json, time, hashlib

CACHE_FILE = "cache/search_cache.json"
_cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as f:
        _cache = json.load(f)


def _save_cache():
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f)


def web_search(query, k=5):
    """Return [{'title','url','text'}]. Results are cached on disk."""
    key = hashlib.md5(f"{query}|{k}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    results = []
    try:
        from ddgs import DDGS
        raw = DDGS().text(query, max_results=k)
        results = [{"title": r.get("title", ""), "url": r.get("href", ""),
                    "text": r.get("body", "")} for r in raw]
    except Exception as e:
        print("search failed:", e)      # rate limit etc. -> empty result
    if results:                          # never cache failures
        _cache[key] = results
        _save_cache()
    time.sleep(1.0)                      # be polite to the search engine
    return results
