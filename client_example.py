"""Calls the auditing proxy (python -m uvicorn server:app --port 9000) with plain HTTP."""
import httpx

r = httpx.post("http://localhost:9000/v1/chat/completions", timeout=300,
               json={"messages": [{"role": "user", "content": "Who is the current CEO of Microsoft?"}]}).json()
print(r["choices"][0]["message"]["content"])
print(r.get("audit"))       # action, risk score, per-claim verdicts
