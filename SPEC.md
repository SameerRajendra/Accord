# Accord — Negotiation Intelligence API

**An LLM-powered system that analyzes negotiation transcripts in real time — detecting escalation and extreme-behavior indicators, retrieving relevant precedent cases, predicting breakdown risk, and recommending de-escalation moves — served as a deployed, measured, containerized API.**

> Engineering implementation of a negotiation-analysis framework proposed in Y. Sawant's 2024 MSc thesis (*Enhancing Negotiation Advantage*, Cranfield School of Management). The thesis contributes the domain framing and behavioral taxonomy; this repository is the systems/engineering build — models, retrieval, agent, serving, and evaluation — authored by Sameer Rajendra.

---

## Design principle: measure everything

This repo's differentiator is not that it uses LLMs — everyone's does. It's that **every claim is backed by a committed benchmark**, the same way the LLM-kernel repo benchmarks honestly against FlashAttention. Specifically it answers, with numbers in `results/`:

- Does the fine-tuned/served sentiment model beat an LLM zero-shot baseline? By how much, at what latency/cost?
- Does RAG actually improve recommendation quality, or just add latency? (RAG vs no-RAG ablation, LLM-judge scored.)
- What's retrieval recall@k / MRR on held-out cases?
- What's the outcome-prediction F1, and is the probability calibrated?
- What's the p50/p95 latency and $/request of the full pipeline?

Unmeasured LLM demos are the norm. A measured one is the signal.

---

## Skill gaps this closes

| Component | Gap closed |
|---|---|
| Analysis pipeline + LangGraph agent | **#1 LLM apps & agents** |
| pgvector retrieval over case corpus | **#5 Vector DB & embeddings** |
| FastAPI → Docker → Cloud Run/App Runner | **#2 Cloud platforms** |
| Eval harness + ablations + tracking | **#6 Experiment tracking & evals** |
| Docker/compose, CI, tracing | **#3 MLOps (partial)** |
| Dataset ingestion + XGBoost outcome model | **#4 Data eng + classical ML (partial)** |

Four gaps closed hard, two touched. This is the single highest-leverage project on the gap list.

---

## Architecture

```
                       ┌─────────────────────────────────────────┐
  transcript (JSON) ──►│  FastAPI  /analyze                       │
                       │                                          │
                       │   1. sentiment/escalation (per turn)     │──► Claude / served model
                       │   2. extreme-behavior detection          │──► structured output
                       │   3. outcome predictor (breakdown risk)  │──► XGBoost
                       │   4. LangGraph agent:                    │
                       │        state ─► retrieve precedent ──────┼──► pgvector (case corpus)
                       │              ─► recommend next move       │──► Claude
                       │                                          │
                       │   trace + cost/latency  ─────────────────┼──► Langfuse
                       └─────────────────────────────────────────┘
                                        │
                                   results/ (committed eval CSV/JSON)
```

---

## Tech stack (2026, industry-standard)

- **Language:** Python 3.11
- **LLM:** Anthropic Claude API (default). *Optional differentiator:* self-host the sentiment classifier on **vLLM** — ties this project directly to your GPU/inference background and gives you a real "served open model vs API" latency/cost comparison.
- **Orchestration:** LangGraph (agent), direct SDK for single-shot calls
- **Vector store:** Postgres + **pgvector** (one service, real SQL, industry-common)
- **Embeddings:** `text-embedding-3-large` (default) or BGE via sentence-transformers (open)
- **Classical ML:** XGBoost outcome predictor (+ scikit-learn metrics/calibration)
- **API:** FastAPI + Pydantic v2 contracts
- **Serving:** Docker; local via docker-compose (api + postgres); deploy to **Google Cloud Run** (simplest containerized path) or **AWS App Runner**
- **Evals/tracking:** custom harness → `results/`, **Weights & Biases** for run tracking, **Langfuse** (self-hostable) for LLM tracing
- **CI:** GitHub Actions — lint + tests + a fast eval subset on every push
- **Data:** a real annotated negotiation corpus — **CaSiNo** (Chawla et al., NAACL 2021; 1030 dialogues with strategy annotations + per-party outcome points). Alternatives: DealOrNoDeal (FAIR), CraigslistBargain. *Verify license before committing data.*

---

## Repo structure

```
accord/
├── data/
│   ├── ingest_casino.py       # parse CaSiNo → normalized transcript schema
│   ├── build_case_corpus.py   # RAG corpus: annotated cases + strategy playbook
│   └── synthetic_gen.py       # LLM-generated hard cases (augmentation, labeled)
├── analysis/
│   ├── sentiment.py           # per-turn emotion/escalation (LLM baseline + served model)
│   ├── behaviors.py           # extreme-behavior detection (structured LLM output)
│   └── outcome_model.py       # XGBoost breakdown-risk predictor + calibration
├── rag/
│   ├── schema.sql             # pgvector DDL
│   ├── embed.py               # embed + upsert corpus
│   └── retriever.py           # top-k retrieval
├── agent/
│   ├── graph.py               # LangGraph: state → retrieve → recommend
│   └── tools.py               # sentiment / retrieve_precedent / predict_outcome
├── api/
│   ├── main.py                # FastAPI app, /analyze + /health
│   └── models.py              # Pydantic request/response
├── evals/
│   ├── sentiment_eval.py      # accuracy/F1 vs labels; model vs LLM-baseline
│   ├── retrieval_eval.py      # recall@k, MRR
│   ├── outcome_eval.py        # F1, ROC-AUC, calibration curve
│   ├── agent_eval.py          # LLM-judge quality; RAG vs no-RAG ablation
│   └── report.py              # emit results table (mirrors kernel-repo style)
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml     # api + postgres/pgvector
│   └── deploy_cloudrun.sh
├── results/                   # committed eval outputs (reproducibility)
├── .github/workflows/ci.yml
├── requirements.txt
├── pyproject.toml
└── README.md                  # honest, measured, credits the thesis
```

---

## Build order — 7 phases, each independently shippable

Each phase ends with a commit that stands on its own. Don't start a phase before the previous one's eval is green.

**Phase 0 — Data spine (day 1).** `ingest_casino.py` + `build_case_corpus.py`. Ship: normalized transcripts + case corpus, a `tests/` sanity check on the schema.

**Phase 1 — Analysis + baseline evals (day 2–3).** Sentiment (start with the LLM zero-shot baseline), behavior detection, outcome model. Ship: `results/sentiment.csv`, `results/outcome.csv` with real F1. *Your measurement brand starts here.*

**Phase 2 — RAG (day 3–4).** pgvector schema, embed corpus, retriever + `retrieval_eval.py`. Ship: `results/retrieval.csv` (recall@k, MRR).

**Phase 3 — Agent + the signature ablation (day 4–5).** LangGraph agent; `agent_eval.py` with the **RAG vs no-RAG** ablation, LLM-judge scored. Ship: the ablation table answering "does retrieval actually help?" — this is your FlashAttention-comparison equivalent.

**Phase 4 — Serve + deploy (day 5–6).** FastAPI, Dockerfile, docker-compose local, deploy to Cloud Run. Ship: a live endpoint URL + curl example in README.

**Phase 5 — Observability + cost (day 6–7).** Langfuse tracing; latency/cost harness. Ship: `results/latency_cost.csv` (p50/p95, $/request).

**Phase 6 — Polish + CI (day 7).** GitHub Actions (lint + tests + fast eval subset), README with the full results tables and honest limitations section.

---

## What makes it "industry," not "student"

- Real labeled dataset, not toy strings
- Committed, reproducible eval outputs in `results/`
- Ablations that could have shown your own feature doesn't help (and you report it either way)
- Pydantic-typed API contract, health check, containerized
- Live deployed URL
- Tracing + cost/latency budget (the thing production teams actually care about)
- CI that runs evals, so regressions are caught
- A README that names limitations instead of hiding them

---

## Honest framing (do this from commit 1)

- README credits Y. Sawant's thesis for the **problem framing and behavioral taxonomy**; everything you list as your skill is code you wrote.
- If you use CaSiNo or any dataset, cite it and check its license before committing data files.
- The limitations section says plainly what the system can't do (e.g., trained on campsite-negotiation dialogues; generalization to business contracts is untested — which is *itself* an honest, interesting finding to state).

The same razor as the kernel repo: if a reviewer clicks in and reads everything, it should get **stronger**, not weaker.
```
