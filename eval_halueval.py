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
