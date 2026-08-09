# Accord — System Design

**Scope:** the systems/engineering design for Accord, a negotiation-intelligence API whose
**entire LLM stack is self-hosted open weights** (no hosted LLM API is ever called at inference
time) and whose orchestration layer is **LangChain + LangGraph, with an MCP server exposing its
own tools** (§5). This document is the authoritative architecture reference; where it differs
from [SPEC.md](SPEC.md) (which framed a Claude-API-default build, "direct SDK for single-shot
calls," and no MCP), **this document supersedes it** on serving, orchestration, and tool
exposition.

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
| Latency (full `/analyze`, warm) | p50 ≤ ~10 s, p95 ≤ ~18 s, streamed | Analysis endpoint, not a chat turn; recommendation dominates. |
| Cold-start (first request after idle) | ≤ ~90 s | Explicit tradeoff of scale-to-zero (§3); documented, not hidden. |
| Throughput | saturate GPU: maximize tok/s at fixed p95 | The number that justifies self-hosting. |
| Cost | report $/request and $/1M tokens **amortized from GPU-second** | Honest self-host economics on serverless GPU (see §8). |
| Reproducibility | committed eval outputs + pinned engine build configs + `modal deploy` from committed source | A reviewer can redeploy and re-run. |
| Observability | Langfuse traces per stage, GPU-second/request | What production inference teams actually watch. |

### Constraints
- **Serving surface:** rented cloud GPU (Modal serverless, H100), scale-to-zero. No always-on
  GPU rental — the demo idles at ~$0 and pays only per request.
- **Serving stack:** SGLang serving Qwen2.5-7B-Instruct on the GPU container. TensorRT-LLM +
  Triton remains the stretch-goal engine story (§7) for a quantization-ablation section — not
  built.
- **Models:** open-weight only. No Claude/OpenAI/etc. at inference *or* as an eval judge. Managed
  OSS-inference endpoints (Together / Fireworks / Cerebras / Groq) are a documented fallback if
  the demo cold-start proves unusable, but the default is self-hosted SGLang on rented GPU.
- **Team:** solo. Favor one serving path done rigorously over many done shallowly.
- **Data:** CraigslistBargain (real buyer/seller price-haggling; He et al., EMNLP 2018) — replaced
  an earlier CaSiNo-based ingestion. Generalization to Sawant's thesis's business-contract framing
  is explicitly untested (§9).

---

## 2. High-level architecture

```
     ┌──────── Streamlit UI (Modal ASGI, CPU, scale-to-zero) ────────┐
     │  transcript textarea → POST /analyze → results table          │
     └───────────────────────────────┬──────────────────────────────┘
                                     │ HTTPS
                                     ▼
     ┌──────── FastAPI /analyze (Modal ASGI, colocated on GPU) ─────┐
     │                                                                │
     │  ┌─ run concurrently ────────────────────────────────────┐    │
     │  │ 1. sentiment/escalation (batched, structured output)  │──┐ │
     │  │ 2. extreme-behavior (structured output)               │──┤ │
     │  └──────────────────────────────────────────────────────┘  │ │
     │  ┌──────────────────────────┐                              ▼ │
     │  │ 3. retrieve precedent    │──► Neon Postgres + pgvector    │
     │  └──────────────────────────┘    (HNSW, managed, free tier)   │
     │  ┌──────────────────────────┐                                 │
     │  │ 4. outcome model (XGBoost)│                                │
     │  └──────────────────────────┘                                 │
     │  ┌────────────────────────────────────────┐                   │
     │  │ 5. LangGraph agent → recommend         │──► SGLang         │
     │  └────────────────────────────────────────┘                   │
     │                                                                │
     └──────────────────────┬───────────────────────────────────────┘
                            │ localhost (same container)
                            ▼
     ┌──────── SGLang server (H100, scale-to-zero) ────────┐
     │  Qwen2.5-7B-Instruct                                 │
     │  RadixAttention prefix cache · continuous batching   │
     │  weights cached on Modal Volume (no repeat download) │
     └──────────────────────────────────────────────────────┘
                            │
                            ▼
                Langfuse (managed cloud, free tier)
```

**Model tiering (current — demo scale):** one 7B instruct model serves all LLM stages
(sentiment, behavior, recommendation). Growing to a small+large tier (7B classification + 32B
recommendation, or 70B on H200) is a §8 revisit-as-it-grows item — not built until the demo has
traffic that warrants two engines and their KV-cache footprint.

**Embeddings:** hosted `sentence-transformers/all-MiniLM-L6-v2` (384-d) called via LangChain's
`HuggingFaceEmbeddings`, loaded once per container. This is the pragmatic Phase 2 choice — a
sibling SGLang embedding-server (as originally described) is deferred until measured latency
warrants a second serving process on the GPU container.

---

## 3. Serving layer (the centerpiece)

This is where the project earns its "inference systems" claim. The engine-build pipeline is
committed and reproducible.

### Current runtime — Modal serverless, SGLang, scale-to-zero

- **Where:** [Modal](https://modal.com) serverless GPU container, one H100 per container,
  `max_containers=1` for the demo. Deploy config: [infra/modal/app.py](infra/modal/app.py).
- **What:** [SGLang](https://github.com/sgl-project/sglang) starts on container-enter, serves
  Qwen2.5-7B-Instruct on localhost:30000, and stays hot for the container's lifetime. FastAPI
  runs as a Modal ASGI app inside the same container and talks to SGLang over localhost.
- **Cold-start economics:** the Qwen2.5-7B FP16 weights (~15 GB) are cached on a Modal Volume
  (`qwen25-7b-cache`), so the first-ever download happens once and subsequent cold starts avoid
  re-download. SGLang startup on H100 is roughly 60–90 s from a cached weights volume; the first
  request after idle pays that cost, all subsequent requests are warm (~seconds).
- **Idle cost:** ~$0. Modal charges per GPU-second and scales to zero after `scaledown_window`
  (default 5 min). This is the tradeoff Q4 of the pivot decision optimized for: near-zero when
  no one is looking at the demo, pay-per-use when Yash (or a recruiter) actually clicks.

### Target — TensorRT-LLM + Triton (stretch, not built)

The engine-build pipeline below is the target architecture for the quantization-ablation story
(§7). Nothing here is currently running; kept as an honest north-star, not implied as done.

```
HF checkpoint ─► quantize (ModelOpt: FP8 / INT4-AWQ) ─► trtllm-build (engine) ─► Triton model repo ─► serve
```

- **Quantization on Hopper:** FP8 (E4M3) weights **and** KV cache is the headline config — H100
  FP8 tensor cores are the reason to be on this hardware. INT4-AWQ built as the memory-savings
  comparison point.
- **Build knobs pinned in `infra/`:** `max_batch_size`, `max_num_tokens`, `max_input_len`,
  `max_seq_len`, paged-KV block size, `use_paged_context_fmha`, tensor-parallel degree. These
  *are* the experiment surface — committed so results are reproducible.
- **In-flight (continuous) batching** via the TRT-LLM Triton backend: new requests join the
  running batch at token granularity instead of waiting for a static batch to drain.

### Structured output

Behavior detection and the recommendation both require schema-valid JSON that maps 1:1 to the
[`api/models.py`](api/models.py) Pydantic contracts. In the current SGLang runtime this goes
through LangChain's `.with_structured_output(PydanticModel)`, which uses SGLang's
`response_format={"type": "json_schema"}` support to constrain decode — a structured-output
guarantee at the serving layer, not a post-hoc JSON-repair loop. If the TensorRT-LLM engine is
built later, XGrammar becomes the constrained-decoding path with the same public contract.

---

## 4. Component deep-dives

- **Sentiment/escalation** — one *batched* structured-output call over all utterances, not one
  call per turn. This is the key latency decision: an N-turn dialogue is one request with a
  compact per-turn JSON array out, not N round-trips.
- **Extreme-behavior** — one structured-JSON call producing typed flags; same engine.
- **Outcome model** — **XGBoost**, not an LLM: features = priority profile + per-party dialogue-
  act counts + turn/word counts. ~1–5 ms, and it gives a *calibrated* probability (isotonic) with
  a committed calibration curve. This is deliberately the one non-LLM component — cheaper,
  faster, and honestly benchmarkable against an LLM zero-shot baseline (open model) as an
  ablation.
- **RAG / pgvector on Neon** — corpus = normalized cases (setup + strategies + outcome + lesson)
  built by [`data/build_case_corpus.py`](data/build_case_corpus.py), embedded with
  `all-MiniLM-L6-v2`, stored in Neon Postgres with an **HNSW** index
  (`m=16, ef_construction=64`). `retrieval_eval.py` reports recall@k / MRR on held-out cases.
- **Agent (LangGraph)** — state machine: `analyze → retrieve → recommend`. Precedent cases are
  injected into the recommendation prompt. The **RAG-vs-no-RAG ablation** toggles the retrieve
  node — the signature experiment answering "does retrieval actually help the recommendation, or
  just add latency?"

---

## 5. Agent orchestration & tool exposition: LangChain, LangGraph, MCP

Decided 2026-07-12, reversing SPEC.md's original "direct SDK for single-shot calls" framing.
LangChain and MCP are used throughout the pipeline.

- **LangChain for single-shot calls.** [`analysis/sentiment.py`](analysis/sentiment.py) and
  [`analysis/behaviors.py`](analysis/behaviors.py) use LangChain's `ChatOpenAI` client pointed
  at the self-hosted SGLang endpoint (`base_url` override, dummy API key). The class name is the
  wire protocol it speaks (OpenAI-compatible REST), **not** the provider — no OpenAI API is
  involved anywhere in this project, consistent with the self-hosted-only constraint (§1).
  Structured output goes through `.with_structured_output(PydanticModel)` rather than a
  hand-rolled JSON-parsing/repair loop.
- **LangChain's `PGVector`** (the `langchain_postgres` package, not the deprecated
  `langchain_community` one) is the retrieval vector store —
  [`rag/retriever.py`](rag/retriever.py) wraps
  `PGVector(embeddings=..., connection=...).as_retriever(search_kwargs={"k": ...})` rather than
  a hand-rolled `psycopg` + raw-SQL client.
- **MCP server** ([`mcp_server/tools.py`](mcp_server/tools.py), FastMCP) exposes Accord's own
  capabilities — `retrieve_precedent`, `predict_outcome`, `analyze_sentiment` — as MCP tools.
  Any MCP client (Claude Desktop, another agent) can call Accord's negotiation-intelligence
  pipeline directly. The LangGraph agent calls the **same underlying Python functions
  directly**, not through its own MCP server as a client — chosen over a full server+client
  round-trip for the agent itself (see the trade-off table, §8) to avoid a pointless
  self-network-hop. This mirrors the tool-exposition pattern from the author's other project
  (AEGOF v2's FastMCP GPU-profiling harness), applied here to a different domain.
- **Embeddings** use LangChain's `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` loaded in-
  process on the Modal container — kept out of the SGLang path in the current phase to avoid
  standing up a second serving process for a small model. Moving embeddings behind a sibling
  SGLang instance is a §8 revisit-as-it-grows item.
- **Observability via Langfuse (managed cloud).** Every LangChain call and LangGraph node emits
  a Langfuse trace; the callback handler is wired in [`agent/callbacks.py`](agent/callbacks.py)
  and env-gated (no `LANGFUSE_PUBLIC_KEY` → no-op, so local runs without a Langfuse account
  still work). This replaces the "Langfuse self-host" line from the pre-pivot design;
  self-hosting Langfuse alongside a scale-to-zero serving story is overkill for a demo.

---

## 6. Scale & reliability

### Latency budget (single `/analyze`, warm — design targets to validate)
| Stage | Model | ~Output tok | Est. latency | On critical path? |
|---|---|---|---|---|
| sentiment (batched) | Qwen2.5-7B | ~300–500 | ~1.5–3 s | yes (parallel w/ behavior+retrieve) |
| behavior | Qwen2.5-7B | ~150 | ~1–1.5 s | overlaps sentiment (same engine → may queue) |
| retrieve | MiniLM + Neon pgvector | — | ~0.1–0.3 s (Neon RTT included) | overlaps |
| outcome | XGBoost | — | ~5 ms | after sentiment |
| **recommend** | Qwen2.5-7B | ~400 | **~3–6 s** | yes — **dominant cost** |
| **critical path** | | | **~5–9 s p50 (warm)** | streamed to cut perceived latency |
| **cold-start penalty** | container + SGLang boot | — | **~60–90 s** | first request after idle only |

The recommendation is the bottleneck by design (longest output). Mitigations: **stream** the
recommendation token-by-token; run the analysis stages concurrently; SGLang's RadixAttention
prefix cache keeps repeated-prompt overhead low.

### Throughput & GPU memory
- SGLang's continuous batching means per-request latency rises under load but aggregate tok/s
  climbs — the trade the benchmark quantifies (latency–throughput Pareto curve at fixed p95).
- With `max_containers=1` there's no autoscale-out — the demo intentionally caps concurrency at
  one container's worth. Raising `max_containers` on Modal is a config change, not a code change,
  if the demo ever needs to fan out.

### Reliability / graceful degradation
- **Retrieval down (Neon unavailable)** → still return sentiment + behavior + outcome + a
  no-precedent recommendation (agent's retrieve node degrades to empty context) rather than
  failing the request.
- **Structured output** guarantees parseable JSON at the serving layer; a hard schema-validation
  failure returns a typed error, never malformed JSON.
- **SGLang health** gated on container startup; the ASGI app doesn't accept traffic until
  SGLang's `/v1/models` returns.
- **Retries** only on idempotent read stages (retrieve); generation is not blindly retried
  (cost + latency).

### Observability
Langfuse traces each stage (sentiment / behavior / retrieve / outcome / recommend), with
LangGraph nodes visible as spans. `results/latency_cost.csv` commits p50/p95 warm-latency and
$/request derived from Modal GPU-second billing.

---

## 7. Measurement plan (the honest benchmarks)

Two layers — **serving** (the GPU/inference story) and **task** (does it actually work):

**Serving / inference**
- **Cold-start vs warm:** first-request-after-idle latency vs subsequent-request latency —
  reported as two distinct numbers, not averaged. Documents the scale-to-zero tradeoff.
- **Batching curve:** throughput vs concurrency at fixed p95 on SGLang (the latency–throughput
  Pareto for the current runtime).
- **[Stretch] Quantization ablation:** FP8 vs FP16 vs INT4-AWQ — latency (TTFT/TPOT), throughput
  (tok/s), peak memory, **and task-accuracy delta**. Requires the TensorRT-LLM engine build
  (§3); explicitly not done in the demo phase.
- **[Stretch] TRT-LLM vs SGLang:** the speedup a compiled engine buys over SGLang on the same
  model/GPU. Same caveat.

**Task**
- `sentiment.csv` — F1 vs labels (Qwen2.5-7B via SGLang vs open zero-shot baseline).
- `retrieval.csv` — recall@k, MRR on held-out cases.
- `outcome.csv` — F1, ROC-AUC, calibration curve (XGBoost vs LLM zero-shot).
- `agent_eval.csv` — **RAG vs no-RAG**, LLM-judge scored (judge = a *different, larger* open
  model than the one under test, to blunt self-preference bias; judge reliability named as a
  limitation).
- `latency_cost.csv` — p50/p95 warm, cold-start, $/request (Modal GPU-second amortized).

---

## 8. Trade-off analysis

| Decision | Chosen | Alternative | Why / cost |
|---|---|---|---|
| Serving host | **Modal serverless GPU (scale-to-zero)** | Always-on rented H100 (RunPod/Lambda dedicated) | Idle cost ~$0 for a demo nobody's hitting; cost is a 60–90 s cold-start penalty on the first request after idle — explicitly documented (§3, §6), not hidden. Always-on would be ~$1500+/mo, wrong shape for "shareable portfolio demo." |
| Serving stack | **SGLang** (current) / **TensorRT-LLM + Triton** (target) | vLLM | SGLang: RadixAttention prefix caching, continuous batching, cheap to run under Modal; TRT-LLM the eventual quantization-ablation vehicle. vLLM would work too; SGLang chosen because the same instance can serve chat + structured output cleanly. |
| LLM provider | **Self-hosted open weights on rented GPU** | Managed OSS-inference API (Together / Fireworks / Cerebras / Groq); or hosted API (Claude/OpenAI) | Full control of latency/cost/quantization and a real GPU-serving artifact; cost is recommendation quality must be earned from open weights and there's no managed autoscaling. Managed OSS-inference is the documented fallback if cold-start proves unusable in practice. Hosted API is explicitly out of scope. |
| Vector store | **Neon Postgres + pgvector (HNSW) via LangChain's `PGVector`** | Pinecone/Weaviate/Qdrant, or self-hosted Apptainer pgvector | Managed free tier fits demo scale (< 1M vectors); no infra to maintain vs the Apptainer instance used pre-pivot; one shareable connection string vs a per-Slurm-allocation container. Cost: another external dependency (Neon) added to the runtime path. |
| Embeddings | **`all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`, in-process** | Sibling SGLang embedding-server (BGE-family) | Smaller footprint on the GPU container, no second serving process to manage; cost is embedding-quality ceiling of MiniLM (384-d) vs BGE (768-d). Revisit if retrieval recall@k underperforms. |
| Outcome predictor | **XGBoost** | LLM classifier | ~1000× cheaper/faster, calibrated probabilities, honestly benchmarkable; cost is feature engineering. |
| Single-shot LLM calls | **LangChain (`ChatOpenAI` + structured output)** | raw `openai` client | Consistent structured-output pattern across every LLM call in the repo, and demonstrable evidence of the #1 skill gap (LLM apps & agents); cost is an extra dependency layer. |
| Tool exposition | **MCP server (FastMCP)**, agent calls tools directly (not as its own MCP client) | Internal-only Python functions, no MCP; or agent-as-MCP-client | Makes the pipeline consumable by any MCP client (Claude Desktop, other agents) — a real, demonstrable integration point beyond the REST API; the agent skips a self-network-hop by calling the same functions directly rather than round-tripping through its own server. |
| UI | **Streamlit on Modal (separate ASGI app)** | Next.js on Vercel; FastAPI + HTMX | One Python file, deploys as its own Modal ASGI so the UI can be updated without redeploying the GPU container. Ugly but shareable in an afternoon. Right choice for "minimal demo for one reviewer." |
| Observability | **Langfuse managed cloud (free tier)** | Self-hosted Langfuse + Prometheus/Grafana | Zero infra to run alongside the scale-to-zero serving story; free tier covers demo scale; env-gated so local runs without keys are no-ops. Self-hosting would require an always-on box which defeats the point of the pivot. |
| Cost model | **amortized GPU-second (Modal invoice)** | per-token API price | Honest for serverless: cost = GPU-seconds-billed / requests-served. When idle: $0. When busy: reports the real number Modal charges, not a hypothetical utilization scenario. |

### What I'd revisit as it grows
- **Two-tier model serving** — 7B classification + 32B recommendation (or 70B on H200), once
  demo traffic warrants two engines and their KV-cache footprint.
- **Speculative decoding** (draft-model or Medusa/EAGLE) on the recommendation to cut latency.
- **Disaggregated serving** — separate prefill/decode pools, once one GPU's KV cache is the
  bottleneck.
- **Sibling SGLang embedding-server (BGE)** if MiniLM retrieval recall@k underperforms.
- **A fine-tuned small classifier** (LoRA) for sentiment/behavior if the zero-shot open baseline
  underperforms the labels.
- **Multi-corpus generalization** — the consumer-haggling→business-contract gap is untested
  (§9); adding a second corpus is the honest next experiment.
- **Actually build the TensorRT-LLM engines** — §3's engine-build pipeline is still the target
  architecture; SGLang is the pragmatic demo runtime. Revisit once the quantization-ablation
  story (§7) needs the real comparison.
- **Raise `max_containers`** on Modal beyond 1, once concurrent demo traffic matters. Config
  change, not code change.

---

## 9. Limitations (stated up front)
- Trained/evaluated on **consumer marketplace price-haggling** (CraigslistBargain). Generalization
  to Sawant's thesis's **business-contract** framing is **untested** — and measuring that gap is
  itself a finding this repo will report, not hide.
- CraigslistBargain's per-turn dialogue-act intents (`init-price`, `counter-price`, `agree`, ...)
  are rule-based, not human-annotated, and are a coarser signal than a persuasion-strategy or
  sentiment taxonomy — treated as a weak proxy in early evals, explicitly caveated as such, not
  gold-standard labels.
- **No hosted-API baseline** by design — comparisons are open-vs-open (served vs zero-shot,
  RAG vs no-RAG), so "beats a frontier API" is explicitly *not* a claim made here.
- The eval judge is itself an open model; LLM-judge scores carry self-consistency and bias caveats.
- **Cold-start latency is real.** A reviewer hitting the demo cold pays ~60–90 s on the first
  request; subsequent requests are seconds. The `/health` endpoint pre-warms the container if a
  reviewer wants to avoid it.
