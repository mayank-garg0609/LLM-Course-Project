import re, time
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
    claims = filter_claims(extract_claims(query, draft))
    private = [c for c in claims if any(re.search(pt, c, re.I) for pt in config.PRIVATE_PATTERNS)]
    claims = [c for c in claims if c not in private][:config.MAX_CLAIMS]
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
            "claims": results, "skipped_private": len(private),
            "latency": round(time.time() - t0, 2)}
