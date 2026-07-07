# Accord — System Design

**Scope:** the systems/engineering design for Accord, a negotiation-intelligence API whose
**entire LLM stack is self-hosted open models served on TensorRT-LLM (Triton), on H100/H200 GPUs.**
No hosted LLM API is used at inference time. This document is the authoritative architecture
reference; where it differs from [SPEC.md](SPEC.md) (which framed a Claude-API-default build with
optional vLLM), **this document supersedes it** on the serving decision.

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
| Cost | report $/request and $/1M tokens **amortized from GPU-hour** | Honest self-host economics (see §7). |
| Reproducibility | committed eval outputs + pinned engine build configs | A reviewer can rebuild the engines and re-run. |
| Observability | per-stage trace, TTFT/TPOT, GPU util, cost | What production inference teams actually watch. |

### Constraints
- **Hardware:** H100/H200 (Hopper) — FP8 tensor cores, the reason TensorRT-LLM is the right serving stack.
- **Serving:** TensorRT-LLM engines behind Triton Inference Server (in-flight batching, paged KV cache).
- **Models:** open-weight only. No Claude/OpenAI/etc. at inference *or* as an eval judge.
- **Team:** solo. Favor one serving path done rigorously over many done shallowly.
- **Data:** CaSiNo (campsite-resource negotiation). Generalization is explicitly untested (§8).

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

## 5. Scale & reliability

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

## 6. Measurement plan (the honest benchmarks)

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

## 7. Trade-off analysis

| Decision | Chosen | Alternative | Why / cost |
|---|---|---|---|
| Serving stack | **TensorRT-LLM + Triton** | vLLM | Max latency/throughput ceiling on Hopper (FP8, compiled engines); cost is a heavier build pipeline (checkpoint convert → quantize → `trtllm-build`) vs vLLM's near-zero setup. Deliberate: the build pipeline *is* part of the showcase. |
| LLM provider | **Self-hosted open** | Claude/OpenAI API | Full control of latency/cost/quantization and a real GPU-serving artifact; cost is that recommendation quality must be earned from open weights, and there's no managed autoscaling. |
| Recommendation model | 32B (H100) / 70B (H200) | 7B everywhere | Reasoning-heavy step needs the capacity; cost is it's the latency + memory bottleneck. |
| Outcome predictor | **XGBoost** | LLM classifier | ~1000× cheaper/faster, calibrated probabilities, honestly benchmarkable; cost is feature engineering. |
| Vector store | **pgvector (HNSW)** | Pinecone/Weaviate/Qdrant | One service, real SQL, self-hostable, plenty for corpus scale; cost is less turnkey ANN tuning. |
| Structured output | **Grammar-constrained (XGrammar)** | prompt + JSON-repair loop | Guarantees valid contracts at decode time, removes a failure class; cost is a small decode overhead + grammar setup. |
| Cost model | **amortized GPU-hour** | per-token API price | Honest: self-hosting wins **only at high utilization** — at ~$2.5/H100-hr and a saturated ~3k tok/s that's ≈ $0.2/1M output tokens (well under hosted-API rates), but an **idle** GPU is pure burn. The benchmark reports the utilization break-even, not a flattering single number. |

### What I'd revisit as it grows
- **Speculative decoding** (draft-model or Medusa/EAGLE) on the large engine to cut recommendation latency.
- **Disaggregated serving** — separate prefill/decode pools, or split classification vs recommendation onto dedicated GPUs, once one GPU's KV cache is the bottleneck.
- **Autoscaling / scale-to-zero** to fix the idle-GPU cost problem for bursty traffic (the self-host Achilles heel).
- **A fine-tuned small classifier** (LoRA) for sentiment/behavior if the zero-shot open baseline underperforms the labels.
- **Multi-corpus generalization** — the campsite→business-contract gap is untested (§8); adding a second corpus is the honest next experiment.

---

## 8. Limitations (stated up front)
- Trained/evaluated on **campsite-resource** dialogues (CaSiNo). Generalization to business contracts is **untested** — and measuring that gap is itself a finding this repo will report, not hide.
- Strategy annotations cover only 396/1030 dialogues.
- **No hosted-API baseline** by design — comparisons are open-vs-open (quantized vs full-precision, served vs zero-shot, RAG vs no-RAG), so "beats a frontier API" is explicitly *not* a claim made here.
- The eval judge is itself an open model; LLM-judge scores carry self-consistency and bias caveats.
