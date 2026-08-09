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

**CraigslistBargain** (He et al., EMNLP 2018, *Decoupling Strategy and Generation in
Negotiation Dialogues*) — 6,682 real buyer/seller price-haggling dialogues scraped from
Craigslist across six categories (housing, furniture, electronics, bike, car, phone),
with per-turn dialogue-act intents and a final agreed price (or no deal). Hosted on
CodaLab; ingested directly from the raw source (see `data/ingest_craigslist.py` for the
verified JSON structure notes).

> Raw data is **not committed** (see `.gitignore`); regenerate it locally.
> Verify the dataset's license before committing any data files.

## Quickstart (Phase 0)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install deps. Either path works — run all commands from the repo root.
pip install -e ".[dev]"                              # modern pip (>= 21.3)
# ...or on older pip (no editable/PEP 660 install needed):
pip install -r requirements.txt -r requirements-dev.txt

# Download raw CraigslistBargain (train+validation+test) and normalize
# -> data/processed/craigslist_bargain.jsonl
python -m data.ingest_craigslist --download

# Build the RAG case corpus -> data/processed/case_corpus.jsonl
python -m data.build_case_corpus

pytest -q && ruff check .
```

`ingest_craigslist` emits one normalized [`Transcript`](data/schema.py) per line of
JSONL. The schema is source-agnostic on purpose: adding another negotiation corpus later
means a new ingestion script, not a new pipeline — this is exactly what happened when
this project moved from an initial CaSiNo-based ingestion to CraigslistBargain.

## Build phases

| Phase | Scope | Ships | Status |
|------:|-------|-------|:------:|
| 0 | Data spine: schema + CraigslistBargain ingestion + RAG case corpus | normalized transcripts, case corpus, tests | ✅ done |
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

- Trained/evaluated on **consumer marketplace price-haggling** (Craigslist listings).
  Sawant's thesis frames the problem around **business-contract** negotiation, based on
  qualitative interviews with practitioners in that setting — a real domain gap between
  the framing source and the training data, stated plainly rather than hidden.
  Generalization from one to the other is **untested**; measuring that gap is itself an
  honest finding this repo intends to report.
- The thesis itself is a **qualitative study** (9 semi-structured practitioner
  interviews), not a dataset or a validated model — it explicitly states formal
  validation was beyond its scope. Accord is the systems build that operationalizes and
  measures the framework it proposes; nothing about the thesis's own findings is being
  reused as ground truth.
- Per-turn dialogue-act intents (`init-price`, `counter-price`, `agree`, ...) are a
  coarser signal than a persuasion-strategy taxonomy — they mark discourse moves, not
  rhetorical tactics. A dedicated sentiment/behavior taxonomy is planned for Phase 1,
  not assumed to already exist in this data.

## References

- M. He, D. He, D. Chapman, P. Liang, C. D. Manning, *Decoupling Strategy and Generation
  in Negotiation Dialogues*, EMNLP 2018. (CraigslistBargain dataset.)
- Y. Sawant, *Enhancing Negotiation Advantage: An AI-Driven Framework for Predicting and
  Mitigating Extreme Negotiation Behaviour in Business Contracts using Sentiment Analysis
  and Predictive Modelling*, MSc thesis, Cranfield School of Management, 2024.
