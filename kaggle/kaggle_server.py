"""Model server for Kaggle (free T4 GPU). Open models only, no API keys.

Serves two open-source models behind one FastAPI app:
  flan-t5   -> google/flan-t5-large   (T5-based, instruction tuned: the VERIFIER)
  gpt-neo   -> EleutherAI/gpt-neo-1.3B (GPT-Neo: the "company LLM" being audited)

Endpoints:
  GET  /health
  POST /generate   {"model": "flan-t5"|"gpt-neo", "prompt": "...", "system": "...",
                    "max_new_tokens": 256, "temperature": 0.0}
                   -> {"model": "...", "text": "..."}

Used by kaggle_notebook.ipynb, which starts this app and exposes it publicly.
"""
import os
import threading
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer)

API_KEY = os.environ.get("MODEL_API_KEY", "")          # optional shared secret
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = torch.float16 if DEVICE == "cuda" else torch.float32

SPECS = {
    # name: (hf id, seq2seq?, dtype). flan-t5 stays fp32 because fp16 overflows in T5.
    "flan-t5": ("google/flan-t5-large", True, torch.float32),
    "gpt-neo": ("EleutherAI/gpt-neo-1.3B", False, HALF),
}
MODELS = {}
LOCK = threading.Lock()          # one generation at a time on a single GPU


def load_all():
    for name, (hf_id, seq2seq, dtype) in SPECS.items():
        print(f"loading {name} ({hf_id}) ...", flush=True)
        tok = AutoTokenizer.from_pretrained(hf_id)
        cls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
        model = cls.from_pretrained(hf_id, torch_dtype=dtype).to(DEVICE).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        MODELS[name] = (tok, model, seq2seq)
    print("models ready on", DEVICE, flush=True)


app = FastAPI(title="Kaggle model server")


class GenReq(BaseModel):
    model: str = "flan-t5"
    prompt: str
    system: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.0


@app.on_event("startup")
def _startup():
    load_all()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "models": list(MODELS)}


@app.post("/generate")
def generate(req: GenReq, authorization: str = Header(default=None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid API key")
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model; use one of {list(MODELS)}")
    tok, model, seq2seq = MODELS[req.model]
    max_new = max(1, min(req.max_new_tokens, 512))

    if seq2seq:   # T5 takes an instruction directly
        text_in = (req.system + "\n\n" if req.system else "") + req.prompt
    else:         # GPT-Neo is a plain LM: use a chat-like transcript and stop at the next turn
        text_in = (req.system + "\n\n" if req.system else "") + f"User: {req.prompt}\nAssistant:"

    inputs = tok(text_in, return_tensors="pt", truncation=True, max_length=1536).to(DEVICE)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    if req.temperature > 0:
        kw.update(do_sample=True, temperature=req.temperature, top_p=0.95)
    else:
        kw.update(do_sample=False)
        if not seq2seq:
            kw.update(repetition_penalty=1.15)

    with LOCK, torch.no_grad():
        out = model.generate(**inputs, **kw)
    if not seq2seq:
        out = out[:, inputs["input_ids"].shape[1]:]          # drop the prompt
    text = tok.decode(out[0], skip_special_tokens=True)
    if not seq2seq:
        text = text.split("\nUser:")[0]                       # cut runaway next turn
    return {"model": req.model, "text": text.strip()}
