# Accord — System Design

**Scope:** the systems/engineering design for Accord, a negotiation-intelligence API whose
**entire LLM stack is self-hosted open models** (target: TensorRT-LLM/Triton on H100/H200; see §3
for current-vs-target reality) and whose orchestration layer is **LangChain + LangGraph, with an
MCP server exposing its own tools** (§5). No hosted LLM API is used at inference time. This
document is the authoritative architecture reference; where it differs from [SPEC.md](SPEC.md)
(which framed a Claude-API-default build, "direct SDK for single-shot calls," and no MCP at all),
**this document supersedes it** on serving, orchestration, and tool exposition.

Design ethos, unchanged from the SPEC: **measure everything.** Every capability claim — accuracy,
latency, throughput, cost — is backed by a committed benchmark in `results/`, the same honest
approach as the GPU-kernel repo.

---

## 1. Requirements

### Functional
Given a negotiation transcript (normalized [`Transcript`](data/schema.py)), `/analyze` returns:
1. **Per-turn sentiment/escalation** — emotion + escalation score for each utterance.
2. **Extreme-behavior detection** — structured flags (threats, ultimatums, stonewalling, …).
3. **Breakdown-risk prediction** — calibrated probability the negotiation fails to reach agreement.
4. **Precedent retrieval** — top-k similar past cases from the corpus.
5. **Recommended next move** — a de-escalation / value-creating action, grounded in retrieved precedent.

### Non-functional (design targets — validated in Phase 5, not asserted here)
| Dimension | Target | Rationale |
|---|---|---|
| Latency (full `/analyze`) | p50 ≤ ~10 s, p95 ≤ ~18 s, streamed | Analysis endpoint, not a chat turn; recommendation dominates. |
| Throughput | saturate GPU: maximize tok/s at fixed p95 | The number that justifies self-hosting. |
| Cost | report $/request and $/1M tokens **amortized from GPU-hour** | Honest self-host economics (see §8). |
| Reproducibility | committed eval outputs + pinned engine build configs | A reviewer can rebuild the engines and re-run. |
| Observability | per-stage trace, TTFT/TPOT, GPU util, cost | What production inference teams actually watch. |

### Constraints
- **Hardware:** H100/H200 (Hopper) — FP8 tensor cores, the reason TensorRT-LLM is the right serving stack.
- **Serving:** TensorRT-LLM engines behind Triton Inference Server (in-flight batching, paged KV cache).
- **Models:** open-weight only. No Claude/OpenAI/etc. at inference *or* as an eval judge.
- **Team:** solo. Favor one serving path done rigorously over many done shallowly.
- **Data:** CraigslistBargain (real buyer/seller price-haggling; He et al., EMNLP 2018) — replaced
  an earlier CaSiNo-based ingestion. Generalization to Sawant's thesis's business-contract framing
  is explicitly untested (§9).

---

## 2. High-level architecture

```
                                   ┌───────────────────────────── FastAPI  /analyze ─────────────────────────────┐
  transcript (JSON) ──► Pydantic ─►│                                                                              │
                                   │  ┌──────────── run concurrently ────────────┐                               │
                                   │  │ 1. sentiment/escalation (batched, guided) │──┐                           │
                                   │  │ 2. extreme-behavior (guided JSON)         │──┤   both hit the SAME        │
                                   │  └───────────────────────────────────────────┘  │   small-model engine      │
                                   │  ┌────────────────────────────┐                  ▼                           │
                                   │  │ 3. retrieve precedent       │──► pgvector (case corpus, HNSW)             │
                                   │  └────────────────────────────┘                                              │
                                   │  ┌────────────────────────────┐                                              │
                                   │  │ 4. outcome model (XGBoost)  │  (features from transcript + sentiment)     │
                                   │  └────────────────────────────┘                                              │
                                   │  ┌──────────────────────────────────────────────┐                           │
                                   │  │ 5. LangGraph agent → recommend next move      │──► large-model engine     │
                                   │  │    (state: analysis + retrieved cases)        │    (guided JSON, streamed)│
                                   │  └──────────────────────────────────────────────┘                           │
                                   └───────────────────┬──────────────────────────────────────────────────────┘
                                                       │ every LLM call ──► Triton (TensorRT-LLM backend)
                                                       ▼
        ┌───────────────────── GPU serving layer (H100/H200) ─────────────────────┐
        │  Triton Inference Server                                                 │
        │   ├─ TRT-LLM engine: small instruct model (classification/behavior)      │
        │   ├─ TRT-LLM engine: large instruct model (recommendation agent)         │
        │   └─ embedding engine: TensorRT-optimized encoder (BGE) — retrieval      │
        │  in-flight (continuous) batching · paged KV cache · FP8 weights/KV        │
        └──────────────────────────────────────────────────────────────────────────┘
                                                       │ traces + metrics
                                                       ▼
                             Langfuse (self-host) + Prometheus/Grafana + results/*.csv
```

**Model tiering** (defaults — swappable):
- **Small engine** (classification + behavior): a 7–8B instruct model (e.g. Qwen2.5-7B-Instruct). High call volume, latency-sensitive, cheap. One engine serves both stages.
- **Large engine** (recommendation agent): a 32B on a single H100, or a 70B (FP8) on H200 / 2×H100. Quality-sensitive, lower volume.
- **Embeddings**: BGE-family encoder, TensorRT-optimized (TRT-LLM is generative-only, so embeddings run on a sibling TensorRT engine / Triton model, not the LLM backend — stated plainly so the boundary is clear).

---

## 3. Serving layer (the centerpiece)

This is where the project earns its "inference systems" claim. The engine-build pipeline is committed
and reproducible.

> **Target vs. current reality:** everything below is the target architecture. What's actually
> running on the cluster today is **SGLang** serving instruct models directly (Qwen2.5-7B-Instruct
> for Accord's analysis calls, on its own port; a separate Llama-3.1-8B instance for a different
> project shares the same node). No TensorRT-LLM engine has been built yet — that remains the
> stretch goal for the quantization-ablation story (§7), tracked honestly rather than implied as
> done. SGLang is a legitimate, fast serving choice in its own right (RadixAttention prefix
> caching, continuous batching), just not the one this section describes.

### Engine build pipeline
```
HF checkpoint ─► quantize (ModelOpt: FP8 / INT4-AWQ) ─► trtllm-build (engine) ─► Triton model repo ─► serve
```
- **Quantization on Hopper:** FP8 (E4M3) weights **and** KV cache is the headline config — H100/H200 FP8 tensor cores are the reason to be on this hardware. INT4-AWQ built as the memory-savings comparison point.
- **Build knobs pinned in `infra/`:** `max_batch_size`, `max_num_tokens`, `max_input_len`, `max_seq_len`, paged-KV block size, `use_paged_context_fmha`, tensor-parallel degree. These *are* the experiment surface — committed so results are reproducible.
- **In-flight (continuous) batching** via the TRT-LLM Triton backend: new requests join the running batch at token granularity instead of waiting for a static batch to drain — this is what turns idle GPU into throughput.

### Structured / guided decoding
Behavior detection and the recommendation both require schema-valid JSON that maps 1:1 to the
[`api/models.py`](api/models.py) Pydantic contracts. TensorRT-LLM's **grammar-constrained decoding
(XGrammar)** enforces the JSON schema at decode time, so outputs parse without a repair loop. Per-turn
sentiment uses a constrained label set the same way. This removes an entire class of "LLM returned
almost-JSON" failures and is a concrete correctness lever, not a nicety.

### Two engines, one GPU (single-H100 layout)
| Engine | Precision | ~Weights | Notes |
|---|---|---|---|
| 7–8B classification/behavior | FP8 | ~8 GB | high concurrency, short outputs |
| 32B recommendation | FP8 | ~34 GB | streamed, longer outputs |
| BGE embedder | FP16/FP8 | ~1–2 GB | retrieval |
| **Paged KV cache** | FP8 | remainder of 80 GB | sized from `max_num_tokens` × concurrency |

On **H200 (141 GB)** the large engine becomes a 70B at FP8. On a single H100, keep the 32B or split
across 2×H100 (classification+embeddings on GPU0, recommendation on GPU1) to avoid KV-cache contention.

---

## 4. Component deep-dives

- **Sentiment/escalation** — one *batched* guided-decoding call over all utterances, not one call per turn. This is the key latency decision: an N-turn dialogue is one request with a compact per-turn JSON array out, not N round-trips.
- **Extreme-behavior** — one guided-JSON call producing typed flags; shares the small engine.
- **Outcome model** — **XGBoost**, not an LLM: features = priority profile + per-party strategy counts + sentiment trajectory + turn/word counts. ~1–5 ms, and it gives a *calibrated* probability (isotonic/Platt) with a committed calibration curve. This is deliberately the one non-LLM component — cheaper, faster, and honestly benchmarkable against an LLM zero-shot baseline (open model) as an ablation.
- **RAG / pgvector** — corpus = normalized cases (setup + strategies + outcome + lesson) built by `build_case_corpus.py`, embedded and stored in Postgres/pgvector with an **HNSW** index. `retrieval_eval.py` reports recall@k / MRR on held-out cases.
- **Agent (LangGraph)** — state machine: `analyze → retrieve → recommend`. Precedent cases are injected into the recommendation prompt. The **RAG-vs-no-RAG ablation** toggles the retrieve node — the signature experiment answering "does retrieval actually help the recommendation, or just add latency?"

---

## 5. Agent orchestration & tool exposition: LangChain, LangGraph, MCP

Decided 2026-07-12, reversing SPEC.md's original "direct SDK for single-shot calls" framing.
LangChain and MCP are now used throughout the pipeline, not just LangGraph for the agent —
recorded here because it changes several already-described components above.

- **LangChain for single-shot calls too.** `analysis/sentiment.py` and `analysis/behaviors.py`
  use LangChain's `ChatOpenAI` client pointed at the self-hosted SGLang endpoint (`base_url`
  override, dummy API key). The class name is the wire-protocol it speaks (OpenAI-compatible
  REST), **not** the provider — no OpenAI API is involved anywhere in this project, consistent
  with the self-hosted-only constraint (§1). Structured output goes through
  `.with_structured_output(PydanticModel)` rather than a hand-rolled JSON-parsing/repair loop.
  This replaces the originally-planned raw `openai` client calls, trading a heavier dependency
  for one consistent structured-output pattern across every LLM call in the repo.
- **LangChain's `PGVector`** (the `langchain_postgres` package, not the older deprecated
  `langchain_community` one) is the retrieval vector store — `rag/retriever.py` wraps
  `PGVector(embeddings=..., connection=...).as_retriever(search_kwargs={"k": ...})` rather than
  a hand-rolled `psycopg` + raw-SQL client.
- **MCP server** (`mcp_server/tools.py`, built with FastMCP) exposes Accord's own capabilities —
  `retrieve_precedent`, `predict_outcome`, `analyze_sentiment` — as MCP tools. Any MCP client
  (Claude Desktop, another agent) can call Accord's negotiation-intelligence pipeline directly.
  The LangGraph agent calls the **same underlying Python functions directly**, not through its
  own MCP server as a client — chosen over a full server+client round-trip for the agent itself
  (see the trade-off table, §8) to avoid a pointless self-network-hop. This mirrors the
  tool-exposition pattern from the author's other project (AEGOF v2's FastMCP GPU-profiling
  harness), applied here to a different domain.
- **Embeddings still go through the self-hosted stack** — `rag/embed.py` calls
  `langchain_openai.OpenAIEmbeddings` (again: protocol name, not provider) pointed at an
  embedding-serving SGLang instance, not a bare `sentence-transformers` script — consistent with
  running the embedder through the same LLM-inference-engine path as the rest of the pipeline.

---

## 6. Scale & reliability

### Latency budget (single `/analyze`, design targets to validate)
| Stage | Model | ~Output tok | Est. latency | On critical path? |
|---|---|---|---|---|
| sentiment (batched) | 7–8B FP8 | ~300–500 | ~1.5–3 s | yes (parallel w/ behavior+retrieve) |
| behavior | 7–8B FP8 | ~150 | ~1–1.5 s | overlaps sentiment (same engine → may queue) |
| retrieve | BGE + pgvector | — | ~0.1 s | overlaps |
| outcome | XGBoost | — | ~5 ms | after sentiment |
| **recommend** | 32B/70B FP8 | ~400 | **~5–10 s** | yes — **dominant cost** |
| **critical path** | | | **~8–12 s p50** | streamed to cut perceived latency |

The recommendation is the bottleneck by design (biggest model, longest output). Mitigations:
**stream** the recommendation token-by-token; run the analysis stages concurrently; keep the classification
engine warm so TTFT stays low.

### Throughput & GPU memory
- Continuous batching means per-request latency rises under load but aggregate tok/s climbs — the trade the benchmark quantifies (latency–throughput Pareto curve at fixed p95).
- KV cache is the real capacity limit, not weights. `max_num_tokens` and concurrency are tuned against 80 GB (H100) / 141 GB (H200); FP8 KV roughly halves cache footprint vs FP16, directly raising max concurrency.

### Reliability / graceful degradation
- **Retrieval down** → still return sentiment + behavior + outcome + a no-precedent recommendation (agent's retrieve node degrades to empty context) rather than failing the request.
- **Guided decoding** guarantees parseable structured output; a hard schema-validation failure returns a typed error, never malformed JSON.
- **Triton health/readiness** gates `/health`; engine OOM under load is prevented by the pinned `max_batch_size`/`max_num_tokens` rather than discovered in production.
- **Retries** only on idempotent read stages (retrieve); generation is not blindly retried (cost + latency).

### Observability
Langfuse traces each stage; Prometheus scrapes Triton (queue time, compute time, TTFT, TPOT, GPU util, KV-cache utilization); `results/latency_cost.csv` commits p50/p95/TTFT/TPOT and $/request.

---

## 7. Measurement plan (the honest benchmarks)

Two layers — **serving** (the GPU/inference story) and **task** (does it actually work):

**Serving / inference**
- **Quantization ablation:** FP8 vs FP16 vs INT4-AWQ — latency (TTFT/TPOT), throughput (tok/s), peak memory, **and task-accuracy delta** (the honest part: quantization that tanks F1 is reported, not hidden).
- **TRT-LLM vs baseline:** compiled TRT-LLM engine vs a naive HF `transformers` serving loop on the same model/GPU — the speedup the engine buys.
- **Batching curve:** throughput vs concurrency at fixed p95 (the latency–throughput Pareto).

**Task**
- `sentiment.csv` — F1 vs labels (served model vs open zero-shot baseline).
- `retrieval.csv` — recall@k, MRR.
- `outcome.csv` — F1, ROC-AUC, calibration curve (XGBoost vs LLM zero-shot).
- `agent_eval.csv` — **RAG vs no-RAG**, LLM-judge scored (judge = a *different, larger* open model than the one under test, to blunt self-preference bias; judge reliability is named as a limitation).
- `latency_cost.csv` — p50/p95, $/request, $/1M tokens amortized.

---

## 8. Trade-off analysis

| Decision | Chosen | Alternative | Why / cost |
|---|---|---|---|
| Serving stack | **TensorRT-LLM + Triton** (target) / **SGLang** (current, §3 caveat) | vLLM | Max latency/throughput ceiling on Hopper (FP8, compiled engines); cost is a heavier build pipeline (checkpoint convert → quantize → `trtllm-build`) vs vLLM's near-zero setup. Deliberate: the build pipeline *is* part of the showcase — but SGLang is what's actually running today (see §3), since no engine has been built yet. |
| LLM provider | **Self-hosted open** | Claude/OpenAI API | Full control of latency/cost/quantization and a real GPU-serving artifact; cost is that recommendation quality must be earned from open weights, and there's no managed autoscaling. |
| Recommendation model | 32B (H100) / 70B (H200) | 7B everywhere | Reasoning-heavy step needs the capacity; cost is it's the latency + memory bottleneck. |
| Outcome predictor | **XGBoost** | LLM classifier | ~1000× cheaper/faster, calibrated probabilities, honestly benchmarkable; cost is feature engineering. |
| Vector store | **pgvector (HNSW) via LangChain's `PGVector`** | Pinecone/Weaviate/Qdrant, or a hand-rolled psycopg client | One service, real SQL, self-hostable, plenty for corpus scale; going through LangChain's abstraction (§5) costs some transparency vs. hand-written SQL but keeps one consistent library across the pipeline. |
| Single-shot LLM calls | **LangChain (`ChatOpenAI` + structured output)** | raw `openai` client | Consistent structured-output pattern across every LLM call in the repo, and is itself a demonstrable piece of the #1 skill gap (LLM apps & agents); cost is an extra dependency layer over a plain HTTP client that SPEC.md originally called for. |
| Tool exposition | **MCP server (FastMCP)**, agent calls tools directly (not as its own MCP client) | Internal-only Python functions, no MCP; or agent-as-MCP-client | Makes the pipeline consumable by any MCP client (Claude Desktop, other agents) — a real, demonstrable integration point beyond the REST API; the agent skips a self-network-hop by calling the same functions directly rather than round-tripping through its own server. |
| Structured output | **Grammar-constrained (XGrammar)** | prompt + JSON-repair loop | Guarantees valid contracts at decode time, removes a failure class; cost is a small decode overhead + grammar setup. |
| Cost model | **amortized GPU-hour** | per-token API price | Honest: self-hosting wins **only at high utilization** — at ~$2.5/H100-hr and a saturated ~3k tok/s that's ≈ $0.2/1M output tokens (well under hosted-API rates), but an **idle** GPU is pure burn. The benchmark reports the utilization break-even, not a flattering single number. |

### What I'd revisit as it grows
- **Speculative decoding** (draft-model or Medusa/EAGLE) on the large engine to cut recommendation latency.
- **Disaggregated serving** — separate prefill/decode pools, or split classification vs recommendation onto dedicated GPUs, once one GPU's KV cache is the bottleneck.
- **Autoscaling / scale-to-zero** to fix the idle-GPU cost problem for bursty traffic (the self-host Achilles heel).
- **A fine-tuned small classifier** (LoRA) for sentiment/behavior if the zero-shot open baseline underperforms the labels.
- **Multi-corpus generalization** — the consumer-haggling→business-contract gap is untested (§9); adding a second corpus is the honest next experiment.
- **Actually build the TensorRT-LLM engines** — §3's engine-build pipeline is still the target architecture; SGLang is the pragmatic Phase 1 stand-in. Revisit once the quantization-ablation story (§7) needs the real comparison.

---

## 9. Limitations (stated up front)
- Trained/evaluated on **consumer marketplace price-haggling** (CraigslistBargain). Generalization
  to Sawant's thesis's **business-contract** framing is **untested** — and measuring that gap is
  itself a finding this repo will report, not hide.
- CraigslistBargain's per-turn dialogue-act intents (`init-price`, `counter-price`, `agree`, ...)
  are rule-based, not human-annotated, and are a coarser signal than a persuasion-strategy or
  sentiment taxonomy — treated as a weak proxy in early evals, explicitly caveated as such, not
  gold-standard labels.
- **No hosted-API baseline** by design — comparisons are open-vs-open (quantized vs full-precision, served vs zero-shot, RAG vs no-RAG), so "beats a frontier API" is explicitly *not* a claim made here.
- The eval judge is itself an open model; LLM-judge scores carry self-consistency and bias caveats.
