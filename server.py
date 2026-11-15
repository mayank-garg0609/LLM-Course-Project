"""Auditing proxy (chat-completions style API).

Client (chat UI / app)  ->  this server (:9000)  ->  model server on Kaggle
                                   |
                     audit(query, draft) -> verified / corrected / flagged / blocked

Run:  uvicorn server:app --host 0.0.0.0 --port 9000
"""
import json, os, time, uuid
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from pipeline import audit
from upstream import call_upstream

MODE = os.environ.get("AUDIT_MODE", "enforce")          # enforce | monitor
FAIL_MODE = os.environ.get("FAIL_MODE", "open")         # open = deliver draft if auditor breaks
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")     # optional bearer token for clients
MIN_CHARS = int(os.environ.get("AUDIT_MIN_CHARS", "15"))
APPEND_SOURCES = os.environ.get("APPEND_SOURCES", "true").lower() == "true"
LOG_CONTENT = os.environ.get("LOG_CONTENT", "false").lower() == "true"
BLOCK_MESSAGE = os.environ.get(
    "BLOCK_MESSAGE",
    "This response was withheld because it could not be verified. "
    "Please check an official source.")
LOG_FILE = "results/audit_log.jsonl"

app = FastAPI(title="Hallucination Auditor Proxy")


def check_auth(authorization):
    if PROXY_API_KEY and authorization != f"Bearer {PROXY_API_KEY}":
        raise HTTPException(status_code=401, detail="invalid API key")


def last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def source_urls(result, limit=5):
    urls = []
    for c in result["claims"]:
        for e in c["evidence"][:1]:
            if e.get("url") and e["url"] not in urls:
                urls.append(e["url"])
    return urls[:limit]


def apply_action(draft, result):
    a = result["action"]
    urls = source_urls(result)
    src = ("\n\nSources checked:\n" + "\n".join(f"- {u}" for u in urls)
           if urls and APPEND_SOURCES else "")
    if a == "VERIFIED":
        return draft + src
    if a == "CORRECTED":
        return ((result["final_text"] or draft)
                + "\n\n[Auditor: this answer was corrected using web sources.]" + src)
    if a == "FLAGGED":
        unv = [c["claim"] for c in result["claims"] if c["verdict"] != "SUP"][:3]
        return (draft + "\n\n[Auditor warning: these statements could not be verified:\n"
                + "\n".join(f"- {c}" for c in unv) + "]" + src)
    return BLOCK_MESSAGE                                   # BLOCKED


def log_audit(query, draft, result):
    os.makedirs("results", exist_ok=True)
    rec = {"ts": int(time.time()), "action": result["action"], "R": result["R"],
           "n_claims": len(result["claims"]), "latency": result["latency"],
           "verdicts": [c["verdict"] for c in result["claims"]]}
    if LOG_CONTENT:
        rec.update({"query": query, "draft": draft,
                    "claims": [{"claim": c["claim"], "verdict": c["verdict"],
                                "urls": [e.get("url") for e in c["evidence"]]}
                               for c in result["claims"]]})
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def sse(resp, text):
    base = {"id": resp.get("id", "chatcmpl-" + uuid.uuid4().hex[:12]),
            "object": "chat.completion.chunk",
            "created": resp.get("created", int(time.time())),
            "model": resp.get("model", "audited")}
    def gen():
        yield "data: " + json.dumps({**base, "choices": [{"index": 0,
              "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({**base, "choices": [{"index": 0,
              "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE}


@app.get("/v1/models")
def models():
    name = os.environ.get("UPSTREAM_MODEL", "gpt-neo")
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "auditor-proxy"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: str = Header(default=None)):
    check_auth(authorization)
    body = await request.json()
    want_stream = bool(body.get("stream"))

    status, resp = await run_in_threadpool(call_upstream, body)
    if status != 200:
        return JSONResponse(resp, status_code=status)

    msg = resp["choices"][0]["message"]
    draft = msg.get("content") or ""
    skip = (request.headers.get("x-audit-skip") == "true"
            or bool(msg.get("tool_calls")) or len(draft) < MIN_CHARS)

    headers, final_text = {"X-Audit-Action": "SKIPPED"}, draft
    if not skip:
        query = last_user_text(body.get("messages", []))
        try:
            result = await run_in_threadpool(audit, query, draft)
            log_audit(query, draft, result)
            headers = {"X-Audit-Action": result["action"], "X-Audit-Risk": str(result["R"])}
            resp["audit"] = {"action": result["action"], "R": result["R"],
                             "latency": result["latency"], "mode": MODE,
                             "claims": [{"claim": c["claim"], "verdict": c["verdict"],
                                         "urls": [e.get("url") for e in c["evidence"]]}
                                        for c in result["claims"]]}
            if MODE == "enforce":
                final_text = apply_action(draft, result)
        except Exception as e:                              # auditor broke: fail open/closed
            print("audit error:", e)
            headers = {"X-Audit-Action": "ERROR"}
            if FAIL_MODE == "closed":
                final_text = BLOCK_MESSAGE

    resp["choices"][0]["message"]["content"] = final_text
    if want_stream:
        r = sse(resp, final_text)
        r.headers.update(headers)
        return r
    return JSONResponse(resp, headers=headers)
