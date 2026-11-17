"""Local demo server. Run from proj/:  python -m uvicorn demo.server:app --port 8000
Open http://localhost:8000
Live mode needs the Kaggle model server (KAGGLE_URL in .env) and internet.
Replay mode uses the recorded evaluation results in results/ and needs neither."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="Hallucination Auditor demo")


def _jl(name):
    p = os.path.join(ROOT, "results", name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "demo", "static", "index.html"))


@app.get("/api/status")
async def status():
    def check():
        import httpx
        from dotenv import dotenv_values
        u = dotenv_values(os.path.join(ROOT, ".env")).get("KAGGLE_URL", "").rstrip("/")
        if not u:
            return {"live": False, "why": "KAGGLE_URL not set"}
        try:
            r = httpx.get(u + "/health", timeout=5)
            return {"live": r.status_code == 200, "why": "" if r.status_code == 200 else f"HTTP {r.status_code}"}
        except Exception as e:
            return {"live": False, "why": "model server unreachable (is the Kaggle notebook running?)"}
    return await run_in_threadpool(check)


@app.get("/api/recorded")
def recorded():
    tq = json.load(open(os.path.join(ROOT, "results", "truthfulqa.json"), encoding="utf-8"))
    out = {"truthfulqa": tq}
    for tag in ("search", "nosearch"):
        out["halueval_" + tag] = _jl(f"halueval_qa_{tag}.jsonl")
    return out


class Q(BaseModel):
    question: str
    use_search: bool = True


def clean_draft(t):
    """GPT-Neo has no stop token and loops. Drop repeated sentences and a cut-off tail (demo only)."""
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", t) if p.strip()]
    seen, keys = [], set()
    for p in parts:
        if p.lower() not in keys:
            keys.add(p.lower()); seen.append(p)
    if len(seen) > 1 and not re.search(r"[.!?]$", seen[-1]):
        seen.pop()
    return " ".join(seen[:3])


def _ev(**k):
    return json.dumps(k, default=str) + "\n"


@app.post("/api/ask_stream")
def ask_stream(q: Q):
    """Runs the real pipeline step by step and streams one JSON line per stage."""
    def gen():
        import config
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from upstream import call_upstream
        import pipeline as pl
        from server import apply_action
        try:
            t = time.time()
            yield _ev(stage=1, status="start")
            st, body = call_upstream({"messages": [{"role": "user", "content": q.question}], "max_tokens": 70})
            if st != 200:
                yield _ev(error=body["error"]["message"]); return
            raw = body["choices"][0]["message"]["content"].strip()
            draft = clean_draft(raw) or raw
            yield _ev(stage=1, status="done", raw=raw, draft=draft, sec=round(time.time() - t, 2))

            t = time.time(); yield _ev(stage=2, status="start")
            claims = pl.extract_claims(q.question, draft)
            yield _ev(stage=2, status="done", claims=claims, sec=round(time.time() - t, 2))

            t = time.time(); yield _ev(stage=3, status="start")
            kept = pl.filter_claims(claims)[:config.MAX_CLAIMS]
            yield _ev(stage=3, status="done", kept=kept, sec=round(time.time() - t, 2))

            t = time.time(); yield _ev(stage=4, status="start", n=len(kept))
            results = []
            if kept:
                with ThreadPoolExecutor(config.MAX_WORKERS) as ex:
                    futs = {ex.submit(pl.verify_claim, c, q.use_search): c for c in kept}
                    for f in as_completed(futs):
                        r = f.result()
                        r["evidence"] = [{"title": e.get("title", ""), "url": e.get("url", ""),
                                          "text": e.get("text", "")[:240]} for e in r["evidence"]]
                        results.append(r)
                        yield _ev(stage=4, status="claim", claim=r)
                order = {c: i for i, c in enumerate(kept)}
                results.sort(key=lambda r: order[r["claim"]])
            yield _ev(stage=4, status="done", sec=round(time.time() - t, 2))

            R = pl.risk_score(results)
            yield _ev(stage=5, status="done", R=round(R, 3), n_con=sum(r["verdict"] == "CON" for r in results),
                      n_nei=sum(r["verdict"] == "NEI" for r in results), n=len(results))
            t = time.time(); action = pl.decide_action(R, results)
            final = None
            if action == "CORRECTED":
                final = pl.rewrite(q.question, draft, results)
            res = {"action": action, "R": round(R, 3), "final_text": final if action == "CORRECTED" else (draft if action != "BLOCKED" else None), "claims": results}
            yield _ev(stage=6, status="done", action=action, delivered=apply_action(draft, res), sec=round(time.time() - t, 2))
        except Exception as e:
            yield _ev(error=f"{type(e).__name__}: {e}")
    return StreamingResponse(gen(), media_type="application/x-ndjson")
