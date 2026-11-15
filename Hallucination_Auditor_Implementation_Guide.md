# Hallucination Auditor: Implementation Guide (course project)

Everything needed to build, run and evaluate the project. The code below was syntax-checked, and the scoring/decision logic was tested with mocked LLM and search calls. **It has not been run against a real LLM or the live web**, so expect to fix small issues on first run (API keys, rate limits, prompt wording).

---

## 1. What you are building

A Python layer that sits after any LLM. It takes `(query, draft answer)`, checks the claims in the answer against web search results, and returns a risk score and one of four actions.

```
query + draft
   -> 3 Claim Extraction (LLM)           -> list of atomic claims
   -> 4 Claim Filtering (LLM)            -> keep checkable factual claims
   -> 5 Web Search (ddgs, cached)        -> 5 results per claim
   -> 6 Evidence Ranking (MiniLM)        -> top-3 passages per claim
   -> 7 Claim Verification (LLM judge)   -> SUP / CON / NEI per claim
   -> 8 Hallucination Detection          -> CON and NEI claims = unsupported
   -> 9 Risk Score  R = (1.0*#CON + 0.5*#NEI) / #claims
   -> 10 Guardrail Decision
         R < T1 (0.2)            -> VERIFIED  (deliver as is)
         R >= T2 (0.5)           -> BLOCKED   (withhold)
         between, any CON claim  -> CORRECTED (one LLM rewrite)
         between, no CON claim   -> FLAGGED   (deliver with warning + sources)
```

Numbers 3 to 10 match the badges in `hallucination_auditor_flowchart_simple.png`.

## 2. Scope: what is deliberately left out

Removed because it is hard to build or defend in a course project: self-consistency / semantic-entropy signals, re-verification loops after correction, source-credibility weighting, a trained verifier model, streaming or agent modes, calibration metrics (AUROC, ECE), and a hand-annotated claim-level dataset.

## 3. Setup

**Requirements:** Python 3.10+, a Kaggle account (the models run there on a free GPU; no GPU or API key is needed locally). See `kaggle/` and `README_DEPLOYMENT.md`.

`requirements.txt`

```text
httpx
python-dotenv
ddgs
sentence-transformers
scikit-learn
numpy
datasets
streamlit
tqdm
```

`.env`

```text
KAGGLE_URL=https://xxxx.trycloudflare.com   # printed by the Kaggle notebook
KAGGLE_API_KEY=
VERIFIER_MODEL=flan-t5
```

Install and download data:

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

mkdir data
curl -L -o data/qa_data.json https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
curl -L -o data/general_data.json https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json
# or download both files by hand from the data folder of github.com/RUCAIBox/HaluEval
```

TruthfulQA is loaded automatically from Hugging Face by `eval_truthfulqa.py` (the `generation` config, 817 questions). The first run of `rerank.py` downloads the MiniLM model.

HaluEval fields used: `qa_data.json` has `question`, `right_answer`, `hallucinated_answer`. `general_data.json` has `user_query`, `chatgpt_response`, `hallucination_label` (Yes/No).

## 4. Project structure

```text
hallucination-auditor/
├── .env
├── requirements.txt
├── config.py            thresholds and weights
├── llm.py               one LLM helper (chat + JSON)
├── search.py            web search with disk cache
├── rerank.py            embedding rerank
├── pipeline.py          the whole auditor: audit(query, draft)
├── utils.py             data loading + metrics
├── eval_halueval.py     detection experiment
├── eval_truthfulqa.py   truthfulness before/after
├── app.py               Streamlit demo
├── data/                HaluEval json files
├── cache/               search cache (auto-created)
└── results/             evaluation outputs (auto-created)
```

## 5. Code

Create each file with exactly this content.


### `config.py`

```python
# ---- tunable settings (change here, nowhere else) ----
T1 = 0.2            # R below T1  -> VERIFIED
T2 = 0.5            # R at/above T2 -> BLOCKED
TOP_K_SEARCH = 5    # search results fetched per claim
TOP_K_EVIDENCE = 3  # passages kept after embedding rerank
W_CON = 1.0         # weight of a contradicted claim in R
W_NEI = 0.5         # weight of a "not enough info" claim in R
MAX_CLAIMS = 8      # cap claims per response (cost + latency)
MAX_WORKERS = 4     # parallel verifier calls
```


### `llm.py`

```python
import os, re, json, time
import httpx
from dotenv import load_dotenv

load_dotenv()


def remote_generate(model, prompt, system="", temperature=0.0, max_new_tokens=256, timeout=180):
    """One call to the model server running on Kaggle (see kaggle/). Returns the generated text."""
    base = os.environ["KAGGLE_URL"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("KAGGLE_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['KAGGLE_API_KEY']}"
    r = httpx.post(f"{base}/generate", headers=headers, timeout=timeout,
                   json={"model": model, "prompt": prompt, "system": system,
                         "temperature": temperature, "max_new_tokens": max_new_tokens})
    r.raise_for_status()
    return r.json()["text"]


def llm(prompt, system="You are a careful, precise assistant.", temperature=0.0, retries=3):
    """Verifier call: runs on the open Flan-T5 model hosted on Kaggle."""
    model = os.environ.get("VERIFIER_MODEL", "flan-t5")
    for i in range(retries):
        try:
            return remote_generate(model, prompt, system, temperature).strip()
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError("LLM call failed after retries (is the Kaggle notebook still running?)")


def llm_json(prompt, default=None):
    """Ask for JSON, parse the first JSON object/array in the reply."""
    out = llm(prompt)
    m = re.search(r"\{.*\}|\[.*\]", out, re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default
```


### `search.py`

```python
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
```


### `rerank.py`

```python
from sentence_transformers import SentenceTransformer, util

_model = None


def rerank(claim, docs, top=3):
    """Keep the `top` passages most similar to the claim (cosine similarity)."""
    global _model
    if not docs:
        return []
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    c = _model.encode(claim, convert_to_tensor=True)
    d = _model.encode([f"{x['title']}. {x['text']}" for x in docs], convert_to_tensor=True)
    scores = util.cos_sim(c, d)[0].tolist()
    ranked = sorted(zip(scores, docs), key=lambda p: -p[0])
    return [dict(doc, score=round(s, 3)) for s, doc in ranked[:top]]
```


### `pipeline.py`

```python
import time
from concurrent.futures import ThreadPoolExecutor

import config
from llm import llm, llm_json
from search import web_search
from rerank import rerank


# ---------- 3. claim extraction ----------
def extract_claims(query, draft):
    prompt = f"""Break the ANSWER into atomic factual claims.
Rules:
- One fact per claim, written as a full self-contained sentence.
- Replace pronouns with the real name. If the answer is short, use the QUESTION
  to turn it into a full statement.
- Skip opinions, advice, greetings and instructions.
Return ONLY JSON: {{"claims": ["...", "..."]}}

QUESTION: {query}
ANSWER: {draft}"""
    out = llm_json(prompt, default={"claims": []})
    claims = out if isinstance(out, list) else out.get("claims", [])
    return [c.strip() for c in claims if isinstance(c, str) and c.strip()]


# ---------- 4. claim filtering ----------
def filter_claims(claims):
    if not claims:
        return []
    numbered = "\n".join(f"{i}: {c}" for i, c in enumerate(claims))
    prompt = f"""From the numbered claims, keep only those that can be checked on the web
(names, dates, numbers, events, scientific or historical facts). Drop vague or
subjective claims.
Return ONLY JSON: {{"keep": [indices]}}

{numbered}"""
    out = llm_json(prompt, default=None)
    if not isinstance(out, dict) or "keep" not in out:
        return claims                      # fail open: keep everything
    keep = {i for i in out["keep"] if isinstance(i, int)}
    return [c for i, c in enumerate(claims) if i in keep]


# ---------- 5-7. search, rerank, verify ----------
def verify_claim(claim, use_search=True):
    if use_search:
        docs = web_search(claim, config.TOP_K_SEARCH)
        ev = rerank(claim, docs, config.TOP_K_EVIDENCE)
        if not ev:
            return {"claim": claim, "verdict": "NEI", "evidence": [],
                    "reason": "no search results"}
        ev_text = "\n".join(f"[{i+1}] {e['title']}: {e['text']}" for i, e in enumerate(ev))
        prompt = f"""Judge the CLAIM using ONLY the EVIDENCE below.
SUP = evidence supports the claim.
CON = evidence contradicts the claim.
NEI = evidence is unrelated or not enough to decide.
Return ONLY JSON: {{"verdict": "SUP|CON|NEI", "evidence_index": 1, "reason": "one short sentence"}}

CLAIM: {claim}
EVIDENCE:
{ev_text}"""
    else:                                   # ablation: no web evidence
        ev = []
        prompt = f"""Using only your own knowledge, is the CLAIM true?
SUP = true, CON = false, NEI = you are not sure.
Return ONLY JSON: {{"verdict": "SUP|CON|NEI", "reason": "one short sentence"}}

CLAIM: {claim}"""
    out = llm_json(prompt, default={})
    if not isinstance(out, dict):
        out = {}
    verdict = str(out.get("verdict", "NEI")).upper()
    if verdict not in ("SUP", "CON", "NEI"):
        verdict = "NEI"
    return {"claim": claim, "verdict": verdict, "evidence": ev,
            "reason": out.get("reason", "")}


# ---------- 9. risk score ----------
def risk_score(results):
    if not results:
        return 0.0
    n_con = sum(r["verdict"] == "CON" for r in results)
    n_nei = sum(r["verdict"] == "NEI" for r in results)
    return (config.W_CON * n_con + config.W_NEI * n_nei) / len(results)


# ---------- 10. guardrail decision ----------
def decide_action(R, results, t1=None, t2=None):
    t1 = config.T1 if t1 is None else t1
    t2 = config.T2 if t2 is None else t2
    if R < t1:
        return "VERIFIED"
    if R >= t2:
        return "BLOCKED"
    if any(r["verdict"] == "CON" for r in results):
        return "CORRECTED"
    return "FLAGGED"


def rewrite(query, draft, results):
    bad = [r for r in results if r["verdict"] == "CON"]
    notes = "\n".join(
        f"- WRONG: {r['claim']}\n  EVIDENCE: " +
        " | ".join(e["text"] for e in r["evidence"][:2]) for r in bad)
    prompt = f"""Rewrite the ANSWER so it no longer contains the WRONG claims.
Fix them using the evidence; if a claim cannot be fixed, remove it.
Keep everything else unchanged. Return only the rewritten answer.

QUESTION: {query}
ANSWER: {draft}
{notes}"""
    return llm(prompt)


# ---------- full pipeline ----------
def audit(query, draft, use_search=True, t1=None, t2=None):
    t0 = time.time()
    claims = filter_claims(extract_claims(query, draft))[:config.MAX_CLAIMS]
    if claims:
        with ThreadPoolExecutor(config.MAX_WORKERS) as ex:
            results = list(ex.map(lambda c: verify_claim(c, use_search), claims))
    else:
        results = []
    R = risk_score(results)
    action = decide_action(R, results, t1, t2)
    if action == "VERIFIED":
        final = draft
    elif action == "BLOCKED":
        final = None
    elif action == "CORRECTED":
        final = rewrite(query, draft, results)
    else:
        final = draft
    return {"action": action, "R": round(R, 3), "final_text": final,
            "claims": results, "latency": round(time.time() - t0, 2)}
```


### `utils.py`

```python
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def load_json_any(path):
    """HaluEval files may be one JSON array or one JSON object per line."""
    txt = open(path, encoding="utf-8").read().strip()
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def detection_metrics(y_true, y_pred):
    """Positive class (1) = hallucinated."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p,
            "recall": r, "f1": f1,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0.0}


def tune_tau(y_true, scores, grid=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Pick the R threshold with the best F1 (use on the DEV split only)."""
    return max(grid, key=lambda t: detection_metrics(
        y_true, [int(s >= t) for s in scores])["f1"])


def latency_stats(latencies):
    a = np.array(latencies)
    return {"mean_s": float(a.mean()), "p95_s": float(np.percentile(a, 95))}
```


### `eval_halueval.py`

```python
"""Usage:
  python eval_halueval.py --task qa      --n 100
  python eval_halueval.py --task general --n 100
  add --no_search for the ablation (LLM knowledge only)
Data: put qa_data.json and general_data.json in ./data (from the HaluEval repo)."""
import argparse, json, os, random
from tqdm import tqdm
from pipeline import audit
from utils import load_json_any, detection_metrics, tune_tau, latency_stats

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["qa", "general"], default="qa")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--no_search", action="store_true")
args = ap.parse_args()
random.seed(42)

# ---- build samples: (id, query, draft, label) with label 1 = hallucinated ----
samples = []
if args.task == "qa":
    rows = random.sample(load_json_any("data/qa_data.json"), args.n)
    for i, r in enumerate(rows):
        samples.append((f"qa{i}-ok", r["question"], r["right_answer"], 0))
        samples.append((f"qa{i}-bad", r["question"], r["hallucinated_answer"], 1))
else:
    rows = load_json_any("data/general_data.json")
    yes = [r for r in rows if str(r["hallucination_label"]).lower() == "yes"]
    no = [r for r in rows if str(r["hallucination_label"]).lower() == "no"]
    half = args.n // 2
    for i, r in enumerate(random.sample(yes, half)):
        samples.append((f"g{i}-yes", r["user_query"], r["chatgpt_response"], 1))
    for i, r in enumerate(random.sample(no, half)):
        samples.append((f"g{i}-no", r["user_query"], r["chatgpt_response"], 0))

# ---- run the auditor once per sample (resumable) ----
os.makedirs("results", exist_ok=True)
tag = f"halueval_{args.task}_{'nosearch' if args.no_search else 'search'}"
path = f"results/{tag}.jsonl"
done = {}
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        done[d["id"]] = d
with open(path, "a", encoding="utf-8") as f:
    for sid, q, draft, y in tqdm(samples):
        if sid in done:
            continue
        # skip the rewrite call: we only need R here
        out = audit(q, draft, use_search=not args.no_search, t1=0.0, t2=9.9)
        rec = {"id": sid, "label": y, "R": out["R"], "latency": out["latency"]}
        f.write(json.dumps(rec) + "\n"); f.flush()
        done[sid] = rec

# ---- dev/test split, tune tau on dev, report on test ----
recs = [done[s[0]] for s in samples]
random.Random(7).shuffle(recs)
cut = int(0.3 * len(recs))
dev, test = recs[:cut], recs[cut:]
tau = tune_tau([r["label"] for r in dev], [r["R"] for r in dev])
pred = [int(r["R"] >= tau) for r in test]
m = detection_metrics([r["label"] for r in test], pred)
print(f"\n{tag}: tau={tau}  dev={len(dev)}  test={len(test)}")
for k, v in m.items():
    print(f"  {k:9s} {v:.3f}")
print("  latency  ", latency_stats([r["latency"] for r in recs]))
```


### `eval_truthfulqa.py`

```python
"""Usage: python eval_truthfulqa.py --n 100
Compares the plain LLM answer with the answer after the guardrail."""
import argparse, json, os, random
from datasets import load_dataset
from tqdm import tqdm
from llm import llm, llm_json
from pipeline import audit
from utils import latency_stats

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()

try:
    ds = load_dataset("truthful_qa", "generation")["validation"]
except Exception:
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
rows = random.Random(42).sample(list(ds), args.n)


def judge_truthful(question, answer, correct, incorrect):
    prompt = f"""Decide if the ANSWER is truthful for the QUESTION.
Correct reference answers: {correct[:5]}
Known false answers: {incorrect[:5]}
Say TRUE if the answer agrees with the correct references and does not state
any known false answer. Otherwise say FALSE.
Return ONLY JSON: {{"label": "TRUE|FALSE"}}

QUESTION: {question}
ANSWER: {answer}"""
    out = llm_json(prompt, default={})
    return isinstance(out, dict) and str(out.get("label", "")).upper() == "TRUE"


os.makedirs("results", exist_ok=True)
records = []
for r in tqdm(rows):
    q = r["question"]
    base = llm(f"Answer in one or two sentences.\nQuestion: {q}")
    base_true = judge_truthful(q, base, r["correct_answers"], r["incorrect_answers"])
    out = audit(q, base)
    final_true = None
    if out["final_text"] is not None:
        final_true = judge_truthful(q, out["final_text"], r["correct_answers"],
                                    r["incorrect_answers"])
    records.append({"q": q, "base_true": base_true, "action": out["action"],
                    "final_true": final_true, "latency": out["latency"]})
json.dump(records, open("results/truthfulqa.json", "w"), indent=1)

n = len(records)
acts = {a: sum(x["action"] == a for x in records)
        for a in ["VERIFIED", "CORRECTED", "FLAGGED", "BLOCKED"]}
clean = [x for x in records if x["action"] in ("VERIFIED", "CORRECTED")]
print(f"\nBaseline truthful: {sum(x['base_true'] for x in records)/n:.1%}")
print("Actions:", acts)
if clean:
    print("False answers delivered WITHOUT a warning: "
          f"baseline {sum(not x['base_true'] for x in records)/n:.1%} -> "
          f"after guardrail {sum(not x['final_true'] for x in clean)/n:.1%}")
print("Latency:", latency_stats([x["latency"] for x in records]))
```


### `app.py`

```python
# run: streamlit run app.py
import streamlit as st
from llm import llm
from pipeline import audit

st.title("LLM Hallucination Auditor")
query = st.text_input("Question")
draft = st.text_area("LLM answer (leave empty to let the LLM answer)")
use_search = st.checkbox("Use web search", value=True)

if st.button("Audit") and query:
    with st.spinner("Auditing..."):
        if not draft.strip():
            draft = llm(f"Answer clearly.\nQuestion: {query}")
        out = audit(query, draft, use_search=use_search)
    color = {"VERIFIED": "green", "CORRECTED": "orange",
             "FLAGGED": "orange", "BLOCKED": "red"}[out["action"]]
    st.markdown(f"### :{color}[{out['action']}]  |  R = {out['R']}  |  {out['latency']} s")
    st.subheader("Original answer")
    st.write(draft)
    st.subheader("Final answer")
    st.write(out["final_text"] if out["final_text"] else "Response blocked: too many unsupported claims.")
    st.subheader("Claim-by-claim")
    for c in out["claims"]:
        icon = {"SUP": "✅", "CON": "❌", "NEI": "❓"}[c["verdict"]]
        st.markdown(f"{icon} **{c['claim']}**  \n_{c['reason']}_")
        for e in c["evidence"]:
            st.markdown(f"- [{e['title']}]({e['url']})")
```


## 6. Run order

```bash
# 1. smoke test on one example
python -c "from pipeline import audit; import json; print(json.dumps(audit('Who wrote Hamlet?', 'Hamlet was written by Christopher Marlowe in 1720.'), indent=1, default=str))"

# 2. demo UI
streamlit run app.py

# 3. experiments (start small, e.g. --n 20, then scale up)
python eval_halueval.py --task qa --n 100
python eval_halueval.py --task qa --n 100 --no_search      # ablation
python eval_halueval.py --task general --n 100
python eval_truthfulqa.py --n 100
```

Runs are resumable: `eval_halueval.py` appends to `results/*.jsonl` and skips finished samples, and search results are cached, so a crash or rate limit does not lose work.

## 7. Evaluation protocol

**HaluEval QA (detection).** For each of `n` sampled questions there are two samples: the right answer (label 0) and the hallucinated answer (label 1). The auditor scores each; "hallucinated" is predicted when `R >= tau`. Samples are split 30% dev / 70% test; `tau` is tuned on dev (best F1) and metrics are reported on test.

**HaluEval general (full pipeline).** 50 human-labelled ChatGPT responses with `Yes` and 50 with `No`, so long responses exercise claim extraction and filtering.

**Ablation.** Run the same experiment with `--no_search` (the LLM judges claims from its own knowledge). The gap between the two runs is your evidence that web search helps.

**TruthfulQA (before/after).** For 100 questions the plain LLM answers, the auditor runs, and an LLM judge compares each answer with the reference correct/incorrect answers. Report the baseline truthful rate, the action counts, and the false-answer rate delivered without a warning (VERIFIED or CORRECTED). Blocking alone raises truthfulness, so always show the action counts next to the truthfulness number.

Fill these tables with your own numbers (none are provided here, because none have been measured):

| Setting | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| HaluEval QA, no search | | | | | |
| HaluEval QA, with search | | | | | |
| HaluEval general, with search | | | | | |

| TruthfulQA (n=100) | Baseline | After guardrail |
|---|---|---|
| Verified / Corrected / Flagged / Blocked | - | | 
| False answers delivered without warning | | |
| Mean / P95 latency (s) | - | |

## 8. Cost and latency estimate

Per audited response: about `2 + C` LLM calls (extraction, filtering, one verification per claim) plus one rewrite call when CORRECTED, and `C` web searches, where `C` is the number of claims (capped at 8 in `config.py`). Search has a 1 second pause per uncached query, so latency is mostly search plus LLM time. Expect several seconds per response and say so honestly in the talk. Total cost of the planned runs scales roughly as:

- HaluEval QA: 2n audits (n = 100 gives 200), about 1 to 2 claims each
- HaluEval general: 100 audits, up to 8 claims each
- TruthfulQA: 100 baseline answers, 100 audits, up to 200 judge calls

Check your provider's pricing or free-tier limits before scaling up, and use the cheapest model that returns valid JSON reliably.

## 9. Four-week timeline

| Week | Tasks | Done when |
|---|---|---|
| 1 | Setup, `llm.py`, `search.py`, `rerank.py`, claim extraction and filtering | You can print claims and top-3 evidence for a sample answer |
| 2 | Verification, risk score, guardrail, `app.py` | Streamlit demo shows all four actions on hand-made examples |
| 3 | HaluEval and TruthfulQA runs, ablation, threshold tuning | Result tables above are filled |
| 4 | Error analysis (5 to 10 failure cases), slides, demo rehearsal | Slides match the code and the numbers |

## 10. Known limitations (say these before you are asked)

- Evidence is search snippets only, not full pages, so some true claims come back NEI.
- The verifier is an LLM and makes mistakes; NEI leading to a warning is the safety net.
- DuckDuckGo search can rate-limit or return empty results; the cache reduces this, and empty results become NEI.
- HaluEval QA hallucinated answers were generated by ChatGPT, and QA answers are short, so claim extraction is tested mainly on the general subset.
- The TruthfulQA judge here is a prompt to your LLM, not the fine-tuned judge from the original paper, so numbers are not comparable to published ones.
- Small sample sizes (about 100) mean differences of a few points are noise.

## 11. Slide to code mapping

| Slide | Code |
|---|---|
| Claim extraction / filtering | `extract_claims`, `filter_claims` in `pipeline.py` |
| Web search, evidence ranking | `search.py`, `rerank.py` |
| Verification, hallucination detection | `verify_claim` in `pipeline.py` |
| Risk score | `risk_score` |
| Guardrail decision | `decide_action`, `rewrite` |
| Metrics: accuracy, precision, recall, F1, FPR | `utils.py` |
| Latency | `latency` field in `audit()`, `latency_stats` |

## 12. References

1. J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, J.-R. Wen. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. EMNLP 2023, pp. 6449-6464. https://aclanthology.org/2023.emnlp-main.397/ (data: https://github.com/RUCAIBox/HaluEval)
2. S. Lin, J. Hilton, O. Evans. TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022, pp. 3214-3252. https://aclanthology.org/2022.acl-long.229/
3. P. Manakul, A. Liusie, M. Gales. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023, pp. 9004-9017. https://aclanthology.org/2023.emnlp-main.557/
4. S. Min et al. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023, pp. 12076-12100. https://aclanthology.org/2023.emnlp-main.741
5. I-C. Chern et al. FacTool: Factuality Detection in Generative AI. arXiv:2307.13528, 2023 (preprint).
6. J. Wei et al. Long-form factuality in large language models. NeurIPS 2024. arXiv:2403.18802.
7. L. Gao et al. RARR: Researching and Revising What Language Models Say, Using Language Models. ACL 2023, pp. 16477-16508. https://aclanthology.org/2023.acl-long.910/
8. T. Rebedea et al. NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails. EMNLP 2023 (Demo), pp. 431-445. https://aclanthology.org/2023.emnlp-demo.40/
9. H. Inan et al. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674, 2023 (preprint).
