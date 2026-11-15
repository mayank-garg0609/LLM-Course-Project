# Hallucination Auditor: Project Explained

A Python layer that sits after any LLM. It takes `(query, draft answer)`, checks the claims in the answer against web search results, computes a risk score, and returns one of four actions: VERIFIED, CORRECTED, FLAGGED or BLOCKED.

This file explains every file in `proj/`, how the parts fit together, and how to run and evaluate the project. The original specification is `Hallucination_Auditor_Implementation_Guide.md` (and `Hallucination_Auditor_Implementation_Claude_Guide.md` in the parent folder).

## 1. Implementation status vs. the guide

| Guide item | Status |
|---|---|
| `config.py`, `llm.py`, `search.py`, `rerank.py`, `pipeline.py`, `utils.py` | Implemented. Matches the guide, plus `PRIVATE_REGEX` support. |
| `eval_halueval.py`, `eval_truthfulqa.py` | Implemented exactly as in the guide. |
| `app.py` (Streamlit demo) | Implemented as in the guide. |
| `requirements.txt` | Implemented; adds `fastapi`, `uvicorn`, `httpx` for the proxy. |
| `data/qa_data.json`, `data/general_data.json` | Were missing. Downloaded from the HaluEval repo. |
| `cache/`, `results/` | Created automatically on first run. |
| `.env` | Not present (secrets). Copy `.env.example` to `.env` and fill it in. |
| Result tables (guide section 7) | Not filled. They need real experiment runs. |

Beyond the guide, the project also has a deployment extension: an auditing proxy (chat-completions style API) (`server.py`, `upstream.py`, `client_example.py`, `test_server.py`, `Dockerfile`, `README_DEPLOYMENT.md`).

**Not verified in this session:** the Python dependencies are not installed in this environment, so nothing was run end to end. All files pass a syntax check (`py_compile`). Like the guide says, the code has not been run against a real LLM or the live web.

## 2. How the pipeline works

Step numbers match the badges in `hallucination_auditor_flowchart_simple.png`.

```
query + draft
  -> 3 Claim Extraction (LLM)        atomic, self-contained claims
  -> 4 Claim Filtering (LLM)         keep web-checkable factual claims
        (claims matching PRIVATE_REGEX are removed here)
  -> 5 Web Search (ddgs, cached)     5 results per claim
  -> 6 Evidence Ranking (MiniLM)     top-3 passages per claim
  -> 7 Claim Verification (LLM)      SUP / CON / NEI per claim
  -> 8 Hallucination Detection       CON and NEI count as unsupported
  -> 9 Risk Score                    R = (1.0*#CON + 0.5*#NEI) / #claims
  -> 10 Guardrail Decision
        R < T1 (0.2)           VERIFIED   deliver as is
        R >= T2 (0.5)          BLOCKED    withhold
        between, any CON       CORRECTED  one LLM rewrite using the evidence
        between, no CON        FLAGGED    deliver with warning and sources
```

Verdicts: SUP = evidence supports the claim, CON = evidence contradicts it, NEI = not enough information.

## 3. Core files

### [config.py](config.py)
All tunable numbers: thresholds `T1 = 0.2` and `T2 = 0.5`, search and evidence sizes (5 and 3), claim weights (`W_CON = 1.0`, `W_NEI = 0.5`), `MAX_CLAIMS = 8` and `MAX_WORKERS = 4`. It also reads `PRIVATE_REGEX` from the environment (patterns separated by `;;`), which lists claim patterns that must never be sent to web search.

### [llm.py](llm.py)
- `llm(prompt, ...)` makes one call to the model server on Kaggle (`remote_generate`, using `KAGGLE_URL`, `KAGGLE_API_KEY`, `VERIFIER_MODEL`, default Flan-T5). It retries 3 times with growing delays.
- `llm_json(prompt, default)` calls `llm` and parses the first JSON object or array in the reply. It returns `default` if parsing fails.

### [search.py](search.py)
`web_search(query, k)` uses DuckDuckGo (`ddgs`) and returns `[{title, url, text}]`. Results are cached in `cache/search_cache.json`, keyed by an MD5 of the query and `k`. Failures and empty results are never cached. There is a 1 second pause per uncached query.

### [rerank.py](rerank.py)
`rerank(claim, docs, top)` embeds the claim and each result with `all-MiniLM-L6-v2` and keeps the `top` results by cosine similarity. It adds a `score` field. The model loads on first use.

### [pipeline.py](pipeline.py)
The auditor itself.
- `extract_claims(query, draft)`: LLM prompt that returns atomic claims. It uses the question to complete short answers and skips opinions.
- `filter_claims(claims)`: LLM picks the indices worth checking. It keeps everything if the reply is malformed (fail open).
- `verify_claim(claim, use_search)`: searches, reranks, then asks the LLM judge for SUP/CON/NEI using only the evidence. No search results gives NEI. With `use_search=False` (the ablation) the LLM judges from its own knowledge. Invalid verdicts become NEI.
- `risk_score(results)`: the formula in section 2. It returns 0 when there are no claims.
- `decide_action(R, results, t1, t2)`: maps R to an action.
- `rewrite(query, draft, results)`: one LLM call that fixes or removes the CON claims and leaves the rest unchanged.
- `audit(query, draft, use_search, t1, t2)`: the full pipeline. Claim verification runs in parallel with a thread pool. It returns `action`, `R`, `final_text` (None when BLOCKED), per-claim results, `skipped_private` and `latency`.

### [utils.py](utils.py)
- `load_json_any`: reads a JSON array or JSON lines.
- `detection_metrics`: accuracy, precision, recall, F1, FPR and FNR, with hallucinated = positive.
- `tune_tau`: picks the best-F1 threshold from a small grid, on the dev split only.
- `latency_stats`: mean and p95 latency.

## 4. Evaluation and demo

### [eval_halueval.py](eval_halueval.py)
Detection experiment on HaluEval.
- `--task qa`: for each of `n` questions, audits the right answer (label 0) and the hallucinated answer (label 1).
- `--task general`: balanced sample of human-labelled ChatGPT responses (`Yes` = hallucinated). This exercises claim extraction on long text.
- `--no_search`: the ablation, where the LLM judges from its own knowledge.
- It calls `audit` with `t1=0.0, t2=9.9`, so no rewrite call happens and only R is recorded.
- Results append to `results/halueval_<task>_<search|nosearch>.jsonl`, so runs are resumable.
- It splits the records 30% dev / 70% test, tunes `tau` (the R threshold) on dev, and reports metrics and latency on test.

### [eval_truthfulqa.py](eval_truthfulqa.py)
Before/after test on TruthfulQA (`generation` config, loaded from Hugging Face).
1. The plain LLM answers each question.
2. An LLM judge compares that answer with the reference correct and incorrect answers.
3. The auditor runs, and the final text is judged the same way (unless BLOCKED).
4. It prints the baseline truthful rate, action counts, and the rate of false answers delivered without a warning (VERIFIED or CORRECTED only). Always read the action counts next to the truthfulness number, because blocking alone raises truthfulness.

Output goes to `results/truthfulqa.json`.

### [app.py](app.py)
Streamlit demo (`streamlit run app.py`). Enter a question and optionally a draft answer; if the draft is empty the LLM writes one. It shows the action and R, the original answer, the final answer, and each claim with its verdict, reason and source links.

## 5. Deployment extension (auditing proxy)

For protecting a company's own LLM. Clients send chat-completions style JSON to the proxy. Details are in [README_DEPLOYMENT.md](README_DEPLOYMENT.md).

```
Chat UI / app  ->  server.py (:9000)  ->  GPT-Neo on Kaggle
                        |
                 audit(query, draft)
```

### [upstream.py](upstream.py)
`call_upstream(body)` turns the chat messages into a prompt and sends it to the Kaggle server's `/generate` with `UPSTREAM_MODEL` (default `gpt-neo`). Streaming is off, because the full answer must be seen before auditing. It returns `(status, json)`.

### [server.py](server.py)
A FastAPI app with three endpoints: `GET /health`, `GET /v1/models` and `POST /v1/chat/completions`.
- Gets the upstream answer, takes the last user message as the query, and audits the answer.
- `apply_action` produces the delivered text:
  - VERIFIED: the draft plus "Sources checked" URLs.
  - CORRECTED: the rewritten answer plus a notice.
  - FLAGGED: the draft plus a warning listing up to 3 unverified claims.
  - BLOCKED: `BLOCK_MESSAGE`.
- Adds `X-Audit-Action` and `X-Audit-Risk` headers and an `audit` field (per-claim verdicts and URLs) to the response.
- If the client asked for streaming, it returns the final text as one SSE chunk.
- Skips auditing when the `X-Audit-Skip: true` header is set, the answer has tool calls, or the answer is shorter than `AUDIT_MIN_CHARS`.
- Logs each audit to `results/audit_log.jsonl`. Content is logged only if `LOG_CONTENT=true`.
- Switches (environment): `AUDIT_MODE` (`monitor` logs only, `enforce` applies actions), `FAIL_MODE` (`open` delivers the draft if the auditor errors, `closed` blocks), `PROXY_API_KEY`, `APPEND_SOURCES`, `BLOCK_MESSAGE`.

### [client_example.py](client_example.py)
Shows a plain `httpx` call to the proxy, and prints the `audit` field.

### [test_server.py](test_server.py)
Offline test of the proxy with the upstream LLM and `audit` mocked. It covers all four actions, monitor mode, skip header, streaming, fail-open and fail-closed, and API key auth. It needs `fastapi` and `httpx` installed but no API keys.

### [Dockerfile](Dockerfile)
Python 3.11 slim image that installs `requirements.txt` and runs `uvicorn server:app` on port 9000.

## 6. Configuration

Copy `.env.example` to `.env`:

| Variable | Purpose |
|---|---|
| `KAGGLE_URL`, `KAGGLE_API_KEY`, `VERIFIER_MODEL` | The Kaggle model server and the model (Flan-T5) that does verification, extraction, filtering and rewriting. |
| `UPSTREAM_MODEL` | The model (GPT-Neo) whose answers the proxy audits. |
| `AUDIT_MODE`, `FAIL_MODE`, `PROXY_API_KEY`, `APPEND_SOURCES`, `LOG_CONTENT` | Proxy behaviour (section 5). |
| `PRIVATE_REGEX` | Claims matching these never go to web search. |

Keep the verifier and the company LLM separate, so a stale or weak model is not the judge of its own output.

## 7. Run order

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                            # then fill it in

# smoke test
python -c "from pipeline import audit; import json; print(json.dumps(audit('Who wrote Hamlet?', 'Hamlet was written by Christopher Marlowe in 1720.'), indent=1, default=str))"

streamlit run app.py                              # demo UI

# experiments (start small, e.g. --n 20)
python eval_halueval.py --task qa --n 100
python eval_halueval.py --task qa --n 100 --no_search
python eval_halueval.py --task general --n 100
python eval_truthfulqa.py --n 100

# proxy
python test_server.py                             # offline test
uvicorn server:app --host 0.0.0.0 --port 9000
python client_example.py
```

## 8. Folder layout

```
proj/
├── config.py  llm.py  search.py  rerank.py  pipeline.py  utils.py   core auditor
├── eval_halueval.py  eval_truthfulqa.py  app.py                     experiments + demo
├── server.py  upstream.py  client_example.py  test_server.py        proxy deployment
├── kaggle/  kaggle_server.py, make_notebook.py, kaggle_notebook.ipynb   model server for Kaggle
├── Dockerfile  requirements.txt  .env.example
├── data/        HaluEval json files (qa_data.json, general_data.json)
├── cache/       search cache (auto-created)
├── results/     evaluation outputs and audit log (auto-created)
├── hallucination_auditor_flowchart_simple.png
└── *.md         guide, deployment notes, this file
```

## 9. Known limitations

- Evidence is search snippets only, not full pages, so some true claims come back NEI.
- The verifier is an LLM and makes mistakes. NEI leading to a warning is the safety net.
- DuckDuckGo can rate-limit or return empty results. The cache helps, and empty results become NEI.
- Claims about company-private facts cannot be verified on the public web and will tend to be flagged. For those, replace `web_search()` with a search over internal documents that returns the same `{title, url, text}` list.
- HaluEval QA answers are short, so claim extraction is tested mainly on the general subset.
- The TruthfulQA judge is a prompt to your LLM, not the fine-tuned judge from the paper, so numbers are not comparable to published ones.
- With samples of about 100, differences of a few points are noise.
- The proxy loses streaming, audits only the first choice, and passes tool-call responses through unaudited.
- The code has not been run against a real LLM, real search or a real company model.
- Claim-level privacy: anything in an answer can leave your network through search and verifier calls. Use `PRIVATE_REGEX` or self-hosted services if that matters.

## 10. Remaining work

1. Create `.env` and run the smoke test; fix any first-run issues (keys, rate limits, prompt wording).
2. Run the three HaluEval experiments and TruthfulQA, then fill the result tables from the guide (section 7).
3. Error analysis of 5 to 10 failure cases, and slides.
