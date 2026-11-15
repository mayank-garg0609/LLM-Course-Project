"""Builds kaggle_notebook.ipynb (embeds kaggle_server.py). Run: python make_notebook.py"""
import json, pathlib

here = pathlib.Path(__file__).parent
server_src = (here / "kaggle_server.py").read_text(encoding="utf-8")


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Model server on Kaggle (GPT-Neo + Flan-T5) with a public URL
Before running: right panel -> **Session options** -> Accelerator **GPU T4 x2** (or P100), and **Internet: On**
(Internet needs a phone-verified Kaggle account). Then **Run All**. The public URL is printed by the last cell.
No API keys are needed. Keep this notebook running while you use the URL.
"""),
    code("!pip install -q fastapi uvicorn nest_asyncio"),
    code("%%writefile kaggle_server.py\n" + server_src),
    code('''
import os, secrets
# Shared secret so strangers who find the URL cannot use your GPU. Copy it into your local .env.
os.environ["MODEL_API_KEY"] = secrets.token_urlsafe(16)
print("MODEL_API_KEY =", os.environ["MODEL_API_KEY"])
'''),
    code('''
# Start the API in the background and wait until both models are loaded.
import subprocess, time, urllib.request, json
server = subprocess.Popen(["python", "-m", "uvicorn", "kaggle_server:app", "--host", "0.0.0.0", "--port", "8000"])
for _ in range(120):
    try:
        print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/health")))
        break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError("server did not start; check the output above")
'''),
    code('''
# Public URL via a Cloudflare quick tunnel (free, no account, no token).
import re, subprocess, time
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
public_url = None
for line in tunnel.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        public_url = m.group(0)
        break
print("\\nKAGGLE_URL =", public_url)
print("KAGGLE_API_KEY =", os.environ["MODEL_API_KEY"])
'''),
    code('''
# Self-test through the public URL (the tunnel can take ~10 s to become reachable).
import requests, time
time.sleep(10)
h = {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}
for m, p in [("flan-t5", "Answer yes or no: is Paris the capital of France?"),
             ("gpt-neo", "Who wrote the play Hamlet?")]:
    r = requests.post(public_url + "/generate", headers=h, json={"model": m, "prompt": p}, timeout=120)
    print(m, "->", r.json())
'''),
    code('''
# Keep the notebook alive (run this last; stop it to shut the server down).
while True:
    time.sleep(60)
'''),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
(here / "kaggle_notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote kaggle_notebook.ipynb")
