# Plugging the auditor into a company's own LLM

## The idea
A company's fine-tuned LLM is frozen at its training date and is rarely retrained. The auditor is a proxy in front of it: every answer is checked against fresh web evidence before users see it. The LLM itself is not modified.

```
Chat UI / app / internal tool
        |   (chat-completions style JSON request)
        v
  Auditor proxy  :9000  ------>  GPT-Neo on Kaggle (public tunnel URL)
        |                              |
        |  <---- draft answer ---------+
        |
        +--> claim extraction -> web search -> rerank -> verifier LLM -> risk score R
        |
        v
  VERIFIED / CORRECTED / FLAGGED / BLOCKED answer (+ sources, + audit JSON)
```

Two different open models are involved, both hosted by one server on Kaggle: GPT-Neo-1.3B writes the answers (`UPSTREAM_MODEL`), and Flan-T5-large (a T5 model) judges claims (`VERIFIER_MODEL`). No API keys of any commercial provider are used. Keep them separate: a stale or weak model should not be the judge of its own output.

## Step 1: start the model server on Kaggle
1. Upload `kaggle/kaggle_notebook.ipynb` to Kaggle (Create -> Notebook -> File -> Import Notebook).
2. Session options: Accelerator **GPU T4 x2**, **Internet: On** (needs a phone-verified account).
3. Run All. The last cells print `KAGGLE_URL` (a `https://....trycloudflare.com` address) and `KAGGLE_API_KEY`. Leave the notebook running.
   The URL changes every time you restart the notebook.

## Step 2: run the auditor locally
```bash
pip install -r requirements.txt
cp .env.example .env            # paste KAGGLE_URL and KAGGLE_API_KEY from the notebook
uvicorn server:app --host 0.0.0.0 --port 9000
python client_example.py        # plain HTTP POST to http://localhost:9000/v1/chat/completions
python test_server.py           # offline test, no Kaggle needed
```
The response carries `X-Audit-Action` and `X-Audit-Risk` headers and an extra `audit` field with per-claim verdicts and source URLs.

Notes: the Kaggle server exposes `POST /generate` (`model` = `flan-t5` or `gpt-neo`) and `GET /health`. `kaggle/kaggle_server.py` is the app; `kaggle/make_notebook.py` rebuilds the notebook from it. Kaggle sessions stop after a few hours of runtime or when idle.

## Behaviour switches (`.env`)
| Setting | Meaning |
|---|---|
| `AUDIT_MODE=monitor` | Answers pass through unchanged; audits are only logged. Start here. |
| `AUDIT_MODE=enforce` | Apply verify / correct / flag / block. |
| `FAIL_MODE=open|closed` | If search or the verifier fails: deliver the draft (open) or block (closed). |
| `PROXY_API_KEY` | Clients must send this as a bearer token. |
| `LOG_CONTENT=false` | Log only action, R and counts. Set true to store questions and answers. |
| `PRIVATE_REGEX` | Claims matching these patterns are never sent to web search. |
| `X-Audit-Skip: true` header | Skip auditing for one request. |

## Suggested rollout
1. Run in `monitor` mode on real traffic for a while and read `results/audit_log.jsonl`.
2. Check how many answers get flagged or blocked and read a sample by hand.
3. Adjust `T1` and `T2` in `config.py`, then switch to `enforce`.

## Privacy: read before using real company data
- Claims extracted from answers are sent to the web search engine, and claims plus evidence are sent to the verifier LLM. Anything in an answer can leave your network. Use `PRIVATE_REGEX`, and a self-hosted search service (the verifier model already runs on your own Kaggle session, but the tunnel is a public Cloudflare address) if that matters.
- Audit logs can contain questions and answers if `LOG_CONTENT=true`.

## What it cannot do
- It checks claims against the public web. Company-private facts (internal prices, policies, product specs) cannot be verified that way and will tend to come back "not enough info" and be flagged. To cover them, replace `web_search()` in `search.py` with a search over internal documents. It must return the same list of `{title, url, text}`; nothing else changes.
- Streaming is lost: the proxy waits for the full answer, audits it, then returns it (as one chunk if the client asked for streaming). Expect several extra seconds per answer.
- Only the first choice is audited, and tool-call responses are passed through unaudited.
- Flan-T5-large is a small model: its claim extraction and verdict JSON are much weaker than a large model's, so expect noisy verdicts. Not tested on Kaggle, with real search, or a real verifier. Only the proxy logic was tested, with mocks (`test_server.py`).
