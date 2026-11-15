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
