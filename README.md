# Accord — Negotiation Intelligence API

An LLM-powered system that analyzes negotiation transcripts — detecting escalation
and extreme-behavior indicators, retrieving relevant precedent cases, predicting
breakdown risk, and recommending de-escalation moves — served as a deployed,
**measured**, containerized API.

> **Status: Phases 0–4 scaffolded (data spine + analysis + RAG + agent + serve);
> serving benchmarks measured on Modal H100 + SGLang; deploy not yet run
> end-to-end.** This is a portfolio project built in the open, phase by phase.
> Each phase ends with a self-contained commit and a committed evaluation. See
> [DESIGN.md](DESIGN.md) for the authoritative architecture, [SPEC.md](SPEC.md)
> for the historical original plan, and [RUN.md](RUN.md) for the deploy sequence.
>
> **What's real today:** `results/batching_curve.csv`, `results/coldstart.csv`,
> `results/per_request_trace.csv` — measured continuous-batching Pareto for
> Qwen2.5-7B-Instruct on H100 (154 → 4,699 output tok/s at concurrency 1→64;
> $0.27/1M output tokens at saturation; 98 s cold start).
> **What's still promises:** live deploy URL, task-quality evals (sentiment F1,
> retrieval recall@k, RAG vs no-RAG ablation), outcome-model artifact. See the
> phase table below for the honest per-phase status.

## Credit & framing

The **problem framing and behavioral taxonomy** come from Y. Sawant's 2024 MSc thesis
*Enhancing Negotiation Advantage* (Cranfield School of Management). This repository is
the **systems/engineering build** — schema, ingestion, models, retrieval, agent,
serving, and evaluation — authored by **Sameer Rajendra**. The concept is credited;
the code is mine.

## Design principle: measure everything

The differentiator isn't that it uses LLMs — everything does. It's that **every claim
is backed by a committed benchmark** (in `results/`), the same honest-benchmarking
approach as my GPU-kernel repo. The evals answer, with numbers:

- Does a served/fine-tuned sentiment model beat an LLM zero-shot baseline — by how much, at what latency/cost?
- Does RAG actually improve recommendations, or just add latency? (**RAG vs no-RAG ablation**, LLM-judge scored.)
- Retrieval recall@k / MRR on held-out cases?
- Outcome-prediction F1, and is the probability calibrated?
- p50/p95 latency and $/request of the full pipeline?

An ablation that *could* show my own feature doesn't help — and reports it either way —
is the signal. If a reviewer reads everything, it should get stronger, not weaker.

## Data

**CaSiNo** (Chawla et al., NAACL 2021, *CaSiNo: A Corpus of Campsite Negotiation Dialogues
for Automatic Negotiation Systems*) — 1,030 dialogues in which two campers barter over Food,
Water, and Firewood packages, with per-party priority rankings, personality profiles
(Big-Five, Social Value Orientation), and per-party outcome points; 396 dialogues additionally
carry utterance-level persuasion-strategy annotations. Read from the HuggingFace parquet mirror
[`kchawla123/casino`](https://huggingface.co/datasets/kchawla123/casino) — script-free, no
external worksheet dependency (see `data/ingest_casino.py`).

> **Why CaSiNo (and not CraigslistBargain, which earlier versions used):** CraigslistBargain's
> only host, the CodaLab worksheet, went permanently `HTTP 500` in 2026-07. CaSiNo is reliably
> mirrored on HuggingFace, carries far richer annotations (strategies, personality, satisfaction),
> and is the corpus this repo's `schema.py` was originally designed for. Its multi-issue structure
> is also closer to multi-term contract negotiation than single-price haggling was.

> Raw data is **not committed** (see `.gitignore`); regenerate it locally.
> Verify the dataset's license before committing any data files.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install deps. Either path works — run all commands from the repo root.
pip install -e ".[dev]"                              # modern pip (>= 21.3)
# ...or on older pip (no editable/PEP 660 install needed):
pip install -r requirements.txt -r requirements-dev.txt

# Download CaSiNo (HuggingFace parquet) and normalize -> data/processed/casino.jsonl
python -m data.ingest_casino --download

# Build the RAG case corpus (cases + strategy playbook) -> data/processed/case_corpus.jsonl
python -m data.build_case_corpus

pytest -q && ruff check .
```

`ingest_casino` emits one normalized [`Transcript`](data/schema.py) per line of JSONL. The
schema is source-agnostic on purpose: swapping negotiation corpora means a new ingestion
script, not a new pipeline — which is exactly how this project moved from CraigslistBargain
back to CaSiNo when CraigslistBargain's host went dark.

## Build phases

| Phase | Scope | Ships | Status |
|------:|-------|-------|:------:|
| 0 | Data spine: schema + CaSiNo ingestion + RAG case corpus (+ strategy playbook) | normalized transcripts, case corpus, tests | ✅ done |
| 1 | Analysis: LangChain + SGLang, XGBoost outcome model, baseline evals | `results/sentiment.csv`, `results/outcome.csv` | 🟡 code done, eval unrun |
| 2 | RAG (Neon Postgres + pgvector HNSW, LangChain PGVector) | `results/retrieval.csv` (recall@k, MRR) | 🟡 code done, DB not provisioned |
| 3 | Agent (LangGraph) + MCP server + **RAG vs no-RAG ablation** | the ablation table | 🟡 code done, ablation unrun |
| 4 | Serve + deploy (FastAPI + Streamlit UI on Modal, scale-to-zero H100) | live endpoint + UI URL | 🟡 code done, `modal deploy` not run |
| 5a | **Serving benchmarks** (H100 batching Pareto, cold-start, $/token) | `results/batching_curve.csv`, `coldstart.csv`, `per_request_trace.csv` | ✅ done |
| 5b | Task-quality benchmarks (sentiment F1, retrieval recall@k, RAG vs no-RAG) | `results/{sentiment,retrieval,agent_eval,outcome}.csv` | ⏳ |
| 6 | Polish + CI | full results tables, honest limitations | ⏳ |

## Repository layout

```
data/         schema.py (normalized transcript) + ingestion scripts   ← Phase 0
analysis/     sentiment, behaviors, XGBoost outcome model + service    ← Phase 1
rag/          pgvector schema, embedding, retrieval (Neon)             ← Phase 2
agent/        LangGraph graph + tools + LLM factory + Langfuse hook    ← Phase 3
mcp_server/   FastMCP tool layer (same impls as agent/tools.py)        ← Phase 3
api/          FastAPI app + Pydantic contracts                         ← Phase 4
ui/           Streamlit UI (Modal ASGI, calls the API)                 ← Phase 4
evals/        per-component eval harnesses + report                    ← Phase 1+
infra/modal/  Modal deploy (SGLang + FastAPI + Streamlit)              ← Phase 4
infra/neon/   Neon setup notes                                         ← Phase 2
results/      committed eval outputs (reproducibility)
```

## Limitations (stated up front, and growing)

- Trained/evaluated on **cooperative campsite resource-negotiation** (CaSiNo). Sawant's
  thesis frames the problem around **business-contract** negotiation, based on qualitative
  interviews with practitioners in that setting — a real domain gap between the framing
  source and the training data, stated plainly rather than hidden. CaSiNo's *multi-issue*
  bartering (Food/Water/Firewood) is closer to multi-term contract negotiation than
  single-price haggling, but generalization is still **untested**; measuring that gap is
  itself an honest finding this repo intends to report.
- **The breakdown-risk target is near-degenerate on this corpus — measured, and reported here
  rather than published as a flattering number.** CaSiNo is a *cooperative* task: 97.6% of
  dialogues reach agreement, so "predict breakdown" is a ~2.4% event. On the held-out test split
  (102 dialogues: 98 agreements, 4 breakdowns) the XGBoost model scored **97.1% accuracy against
  a 96.1% majority-class baseline** — a lift of exactly *one dialogue* — and caught **1 of the 4**
  actual breakdowns. The isotonic calibrator collapsed to near-constant output (100 of 102
  predictions in a single bin at ~0.99), so the calibration claim in
  [DESIGN.md](DESIGN.md) §4 is **not** currently supported by evidence on this target.
  ROC-AUC (0.93) is the one number that isn't dominated by the class ratio, but it's computed
  over 4 negatives — the confidence interval is far too wide to lean on.
  `results/outcome.csv` is therefore **deliberately not committed yet**: the eval harness
  (`evals/outcome_eval.py`) reports `base_rate`, `accuracy_lift`, `breakdown_recall` and the raw
  confusion counts so this is reproducible in one command, but publishing a degenerate artifact
  would be the exact overclaim this repo exists to avoid. Reframing the target onto an outcome
  with real variance (joint points / point imbalance / satisfaction) is the tracked next step,
  and results land when they mean something.
- The thesis itself is a **qualitative study** (9 semi-structured practitioner
  interviews), not a dataset or a validated model — it explicitly states formal
  validation was beyond its scope. Accord is the systems build that operationalizes and
  measures the framework it proposes; nothing about the thesis's own findings is being
  reused as ground truth.
- Persuasion-strategy annotations cover only **396 of 1,030** dialogues — they power the
  RAG corpus and analysis but are deliberately **excluded from the outcome model's
  features**, since their presence is a dataset-selection artifact, not a negotiation signal.

## References

- K. Chawla, J. Ramirez, R. Clever, G. Lucas, J. May, J. Gratch, *CaSiNo: A Corpus of
  Campsite Negotiation Dialogues for Automatic Negotiation Systems*, NAACL 2021.
  (CaSiNo dataset — [`kchawla123/casino`](https://huggingface.co/datasets/kchawla123/casino).)
- Y. Sawant, *Enhancing Negotiation Advantage: An AI-Driven Framework for Predicting and
  Mitigating Extreme Negotiation Behaviour in Business Contracts using Sentiment Analysis
  and Predictive Modelling*, MSc thesis, Cranfield School of Management, 2024.
