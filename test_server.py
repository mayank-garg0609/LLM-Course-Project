"""Offline test of the proxy logic: upstream LLM and auditor are mocked.
Run: python test_server.py"""
import sys, types, json, os
os.environ.update({"KAGGLE_URL": "http://x", "AUDIT_MODE": "enforce"})

state = {"result": None}
fake_pipeline = types.ModuleType("pipeline")
fake_pipeline.audit = lambda q, d: state["result"]
sys.modules["pipeline"] = fake_pipeline

import upstream  # noqa
upstream.call_upstream = lambda body: (200, {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "company-llm",
    "choices": [{"index": 0, "message": {"role": "assistant",
                 "content": "The CEO of X is Alice, appointed in 2015."}, "finish_reason": "stop"}]})
import server
server.call_upstream = upstream.call_upstream
from fastapi.testclient import TestClient
c = TestClient(server.app)

ev = [{"title": "t", "url": "http://src/1", "text": "..."}]
def res(action, verdicts, final=None):
    return {"action": action, "R": 0.4, "final_text": final, "latency": 1.0, "skipped_private": 0,
            "claims": [{"claim": f"claim{i}", "verdict": v, "evidence": ev, "reason": ""}
                       for i, v in enumerate(verdicts)]}

def ask(**kw):
    return c.post("/v1/chat/completions", json={"model": "m", "messages":
                  [{"role": "user", "content": "Who is CEO of X?"}], **kw})

state["result"] = res("VERIFIED", ["SUP"])
r = ask(); assert r.headers["x-audit-action"] == "VERIFIED" and "Sources checked" in r.json()["choices"][0]["message"]["content"]

state["result"] = res("CORRECTED", ["CON"], final="The CEO of X is Bob.")
r = ask(); t = r.json()["choices"][0]["message"]["content"]; assert "Bob" in t and "corrected" in t

state["result"] = res("FLAGGED", ["NEI"])
t = ask().json()["choices"][0]["message"]["content"]; assert "could not be verified" in t

state["result"] = res("BLOCKED", ["CON", "CON"])
r = ask(); assert r.json()["choices"][0]["message"]["content"] == server.BLOCK_MESSAGE
assert r.json()["audit"]["action"] == "BLOCKED"

r = ask(stream=True); assert "data: [DONE]" in r.text and r.headers["x-audit-action"] == "BLOCKED"

server.MODE = "monitor"                      # monitor: original text is returned untouched
t = ask().json()["choices"][0]["message"]["content"]; assert t.startswith("The CEO of X is Alice") and "Auditor" not in t
server.MODE = "enforce"

r = c.post("/v1/chat/completions", headers={"x-audit-skip": "true"},
           json={"messages": [{"role": "user", "content": "hi"}]}); assert r.headers["x-audit-action"] == "SKIPPED"

server.audit = lambda q, d: (_ for _ in ()).throw(RuntimeError("search down"))
r = ask(); assert r.headers["x-audit-action"] == "ERROR" and "Alice" in r.json()["choices"][0]["message"]["content"]
server.FAIL_MODE = "closed"
assert ask().json()["choices"][0]["message"]["content"] == server.BLOCK_MESSAGE

server.PROXY_API_KEY = "secret"
assert ask().status_code == 401
assert c.get("/health").json()["status"] == "ok"
print("all proxy tests passed; log lines:", sum(1 for _ in open("results/audit_log.jsonl")))
