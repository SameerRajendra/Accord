# Accord — Negotiation Intelligence API

An LLM-powered system that analyzes negotiation transcripts — detecting escalation
and extreme-behavior indicators, retrieving relevant precedent cases, predicting
breakdown risk, and recommending de-escalation moves — served as a deployed,
**measured**, containerized API.

> **Status: Phase 0 (data spine) complete.** This is a portfolio project built in
> the open, phase by phase. Each phase ends with a self-contained commit and a
> committed evaluation. See [SPEC.md](SPEC.md) for the full architecture and build plan.

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

**CaSiNo** (Chawla et al., NAACL 2021) — 1030 two-party campsite resource-negotiation
dialogues (Food / Water / Firewood), 396 with utterance-level strategy annotations.

> Raw data is **not committed** (see `.gitignore`); regenerate it locally.
> Verify the [CaSiNo license](https://github.com/kushalchawla/CaSiNo) before committing any data files.

## Quickstart (Phase 0)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Download raw CaSiNo and normalize -> data/processed/casino.jsonl
python -m data.ingest_casino --download

# Only the 396 strategy-annotated dialogues:
python -m data.ingest_casino --download --annotated-only

pytest -q
```

`ingest_casino` emits one normalized [`Transcript`](data/schema.py) per line of JSONL.
The schema is source-agnostic on purpose: adding DealOrNoDeal or CraigslistBargain later
means a new ingestion script, not a new pipeline.

## Build phases

| Phase | Scope | Ships | Status |
|------:|-------|-------|:------:|
| 0 | Data spine: schema + CaSiNo ingestion | normalized transcripts, schema tests | ✅ done |
| 1 | Analysis + baseline evals | `results/sentiment.csv`, `results/outcome.csv` | ⏳ |
| 2 | RAG (pgvector) | `results/retrieval.csv` (recall@k, MRR) | ⏳ |
| 3 | Agent + **RAG vs no-RAG ablation** | the ablation table | ⏳ |
| 4 | Serve + deploy (FastAPI → Docker → Cloud Run) | live endpoint + curl | ⏳ |
| 5 | Observability + cost | `results/latency_cost.csv` | ⏳ |
| 6 | Polish + CI | full results tables, honest limitations | ⏳ |

## Repository layout

```
data/      schema.py (normalized transcript) + ingestion scripts   ← Phase 0
analysis/  sentiment, extreme-behavior, XGBoost outcome model       ← Phase 1
rag/       pgvector schema, embedding, retrieval                    ← Phase 2
agent/     LangGraph graph + tools                                  ← Phase 3
api/       FastAPI app + Pydantic contracts                         ← Phase 4
evals/     per-component eval harnesses + report                    ← Phase 1+
infra/     Dockerfile, docker-compose (api + pgvector), deploy      ← Phase 4
results/   committed eval outputs (reproducibility)
```

## Limitations (stated up front, and growing)

- Trained/evaluated on **campsite-resource** dialogues. Generalization to business
  contracts is **untested** — measuring that gap is itself an honest finding this repo intends to report.
- Strategy annotations cover only 396/1030 dialogues.

## References

- K. Chawla et al., *CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems*, NAACL 2021.
- Y. Sawant, *Enhancing Negotiation Advantage*, MSc thesis, Cranfield School of Management, 2024.
