# Research Scope: Hallucination Auditor for a Hosted LLM

## Problem

A company LLM (GPT-Neo, hosted on Kaggle) sometimes gives fluent but false answers. This project tests whether a post-hoc auditor can detect those answers and reduce how often false answers reach the user, without a fine-tuned judge and without retraining the model.

## Research questions

1. **Truthfulness.** Does passing the model's answers through the auditor raise the share of truthful answers on TruthfulQA, compared with the model's plain answers? The plain-answer baseline was 54% in an earlier run.
2. **Unwarned errors.** After the guardrail, how often does a false answer still reach the user with no warning (delivered as VERIFIED or CORRECTED)?
3. **Detection.** On HaluEval, how well does the auditor's risk score R separate hallucinated answers from correct ones? Does web search improve this over the LLM judging from its own knowledge alone?

## Setup

| Part | Choice |
|---|---|
| Model being audited | GPT-Neo (`UPSTREAM_MODEL`), Kaggle |
| Auditor / judge | Flan-T5 (`VERIFIER_MODEL`), Kaggle. Different model from the one audited. |
| Evidence | DuckDuckGo search, top 5 results per claim, top 3 kept by MiniLM rerank |
| Guardrail thresholds | T1 = 0.2 (VERIFIED below), T2 = 0.5 (BLOCKED at or above) |
| TruthfulQA | `generation` config, validation split, random sample of 100 (seed 42) |
| HaluEval | QA and general subsets, 100 samples each (seed 42). QA uses both right and hallucinated answers. |
| Ablation | HaluEval QA with `--no_search` (judge uses its own knowledge only) |

## Measures

- **TruthfulQA:** truthful rate of the plain answer (baseline) and of the final answer after the guardrail. Also the action counts (VERIFIED, CORRECTED, FLAGGED, BLOCKED) and the unwarned-error rate.
- **HaluEval:** accuracy, precision, recall, F1, FPR and FNR for a hallucinated-vs-correct decision. The threshold tau is tuned on a 30% dev split and reported on the 70% test split. Latency mean and p95.

## Scope

**In scope:** short factual questions (TruthfulQA), short QA answers and long ChatGPT responses (HaluEval), English, the two Kaggle-hosted models above, and the default thresholds.

**Out of scope:** tuning thresholds for the TruthfulQA run, fine-tuned judges, private company documents (the proxy can route such claims elsewhere via `PRIVATE_REGEX`, but this is not evaluated here), streaming, and tool-call responses.

## Limitations

- **Sample size.** With n = 100, the 95% margin on a proportion near 50% is about ±10 points. Smaller differences are not reliable.
- **Weak judge.** Flan-T5 both verifies claims and judges TruthfulQA answers. Its errors affect both the baseline and the guarded numbers. The TruthfulQA judge is a prompt, not the fine-tuned judge from the paper, so the numbers are not comparable to published results.
- **Search quality.** Evidence is short search snippets, not full pages. True claims can come back NEI and get flagged.
- **Dependence on the web.** DuckDuckGo results change over time and can be rate-limited. Results from different days may not match exactly.
- **Single split and seed.** No repeated runs or confidence intervals were computed.

## Results

_To be filled in after the runs finish (see the files below)._

Raw outputs: `results/truthfulqa.json`, `results/halueval_qa_search.jsonl`, `results/halueval_qa_nosearch.jsonl`, `results/halueval_general_search.jsonl`. Console logs: `results/run_*.log`.
