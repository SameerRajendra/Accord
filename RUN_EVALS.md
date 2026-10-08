# Running the evals

Every number in the README and in [`results/SCORECARD.md`](results/SCORECARD.md)
traces to a committed artifact under `results/`. This is how to reproduce each
one, and how to refresh the public demo link.

Every eval has a **Modal entrypoint** (rented GPU/CPU, one command each) and a
**local/cluster** path. Pick per step. The two LLM evals boot SGLang for you on
Modal, so you don't need to launch it yourself there.

Each step says what it needs, the command, and the artifact it writes. The
GPU steps need a Neon `DATABASE_URL` and an H100-class GPU.

---

## 0. One-time prerequisites

```bash
export DATABASE_URL="postgresql://user:pw@host-pooler.region.aws.neon.tech/neondb?sslmode=require"
python -m data.ingest_casino --download        # -> data/processed/casino.jsonl
python -m data.build_case_corpus               # -> data/processed/case_corpus.jsonl
```

For the Modal path these files ship into the image via `add_local_dir`, so build
them locally once before `modal deploy`/`modal run`. The `accord` Modal secret
must already carry `DATABASE_URL` (it does — the deploy uses it).

**Pulling results off Modal.** Every Modal eval writes its CSVs to the
`accord-artifacts` Volume and prints its JSON summary to the `modal run` output
(that's where the numbers are). To also pull the CSVs:

```bash
modal volume get accord-artifacts results ./results
```

---

## Step 1 — Outcome model  (CPU · no Neon · no GPU · ~1 min)

The cheapest real number. The eval is built to expose the near-degenerate
target, so read it as a rigor check, not a predictive win.

```bash
python -m evals.outcome_eval                          # local
modal run infra/modal/app.py::eval_outcome            # or Modal (CPU)
```

Writes `results/outcome.csv` + `outcome_calibration.csv`. Read: `base_rate`,
`accuracy`, `accuracy_lift`, `breakdown_recall`, confusion `tn/fp/fn/tp`.

---

## Step 2 — Retrieval recall@k  (Neon · no GPU · ~5 min)

The headline retrieval number. No LLM — embeddings + pgvector (+ graph).

**2a. Load the graph** (needed only for the `graph`/`hybrid` arms):

```bash
python -m rag.graph_ingest --dry-run                  # verify cases_unmatched == 0
python -m rag.graph_ingest                            # local, into Neon
modal run infra/modal/app.py::graph_ingest            # or Modal (CPU)
```

**2b. Run the eval:**

```bash
# local
python -m evals.retrieval_eval --retriever pgvector graph hybrid tfidf random
# or Modal (CPU)
modal run infra/modal/app.py::eval_retrieval
# vector-only (skip if the graph isn't loaded):
modal run infra/modal/app.py::eval_retrieval --retrievers "pgvector tfidf random"
```

Writes `results/retrieval.csv` + `retrieval_queries.csv`. Read the `dialogue`
and `strategy` rows: `recall_at_k`, `mrr_at_k`, `structural_match_at_k` vs
`structural_match_baseline`.

Report `pgvector` recall@k as the deployed number. **Do not** claim graph > vector unless the numbers show it —
read `struct@k` against its baseline, not recall alone (graph is structurally
handicapped by single-gold recall; `infra/graph/README.md` §7).

---

## Step 3 — Sentiment reliability + convergent validity  (SGLang · GPU)

**Not an F1.** CaSiNo has no emotion labels, so accuracy is not measurable and
`sentiment.csv` reports `f1_vs_gold: None` by design. This measures whether the
classifier is *reliable* (self-consistent) and *convergently valid* (associated
with human strategy annotations) — never claim a sentiment accuracy number.

```bash
# Modal
modal run infra/modal/app.py::eval_sentiment --limit 40

# Cluster
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000 &
export SGLANG_BASE_URL="http://127.0.0.1:30000/v1"
python -m evals.sentiment_eval
```

Writes `results/sentiment.csv`. Read: `neutral_default_rate` (how often the
model fell back to neutral — if high, everything else describes the fallback),
the consistency metrics, and the Cramér's V proxy. For inter-model agreement add
`--mode judge --judge-model Qwen/Qwen2.5-32B-Instruct` (a *second* model — the
eval refuses to judge against the model under test).

---

## Step 4 — RAG vs no-RAG ablation + citation grounding  (SGLang + Neon · GPU)

The signature experiment. ~140–180 LLM calls at `--limit 20`; budget tens of
minutes on a warm H100 on top of cold start + SGLang boot.

```bash
# Modal
modal run infra/modal/app.py::eval_agent --limit 20

# Cluster (SGLang already up from Step 3)
python -m evals.agent_eval --limit 20
```

Writes `results/agent_eval.csv`. Read: `rag_win_rate` (+ its Wilson interval and
coin-flip p-value), the lexical citation-grounding rate, and the no-RAG
fabrication-control rate.

If the judge is the same 7B under test, say so; a bigger self-hosted judge (`--judge-model Qwen/Qwen2.5-32B-Instruct`)
is the stronger claim.

---

## Step 4b — Safety suite  (isolation/PII no-GPU · injection needs SGLang)

The dimension a company can't ship without. Two of three parts need no GPU.

```bash
# No-GPU subset (tenant isolation + PII) — validates the Phase A namespace boundary
modal run infra/modal/app.py::eval_safety_nogpu
python -m evals.safety_eval --checks isolation pii      # or local

# Full suite incl. prompt-injection resistance (SGLang)
modal run infra/modal/app.py::eval_safety
```

Writes `results/safety_{injection,isolation,pii}.csv`. **`isolation_violations`
MUST be 0** (any cross-namespace leak is a shipping blocker). Injection
`compliance_rate` is fraction obeyed — lower is safer, and it's resistance to
*those* payloads, not a proof.

## Step 4c — Reference-free RAG triad  (SGLang + Neon · GPU)

Faithfulness / answer-relevance / context-precision — **no gold labels**, so it
runs on the live path too. Use a stronger judge or the numbers grade their own
homework.

```bash
modal run infra/modal/app.py::eval_rag_triad --limit 10
modal run infra/modal/app.py::eval_rag_triad --namespace acme-legal --limit 10   # LIVE path
```

Writes `results/rag_triad.csv` (+ consistency). Read `faithfulness_mean`,
`answer_relevance_mean_1to5`, `context_precision_mean` per arm, and
`self_consistency.unanimous_rate` (discount the metrics if it's low).

## Step 4d — Render the scorecard  (no infra · pure)

The single summary artifact. Reads whatever CSVs exist; shows measured /
not-measured / by-design honestly.

```bash
python -m evals.scorecard --out results/SCORECARD.md
```

## Step 5 — Refresh the live demo link  (Modal)

The deployed build predates thread input, stance/trajectory, the graph arm, and
all of Phase A. One deploy brings the public URL to current code. Phase A is
deploy-safe: new routes ship on the existing FastAPI app, the embedding model is
already in the image, and live namespaces create their pgvector collections on
demand.

```bash
modal run infra/modal/app.py::build_corpus            # re-embed benchmark corpus (if changed)
modal run infra/modal/app.py::graph_ingest            # load graph (if not already)
modal run infra/modal/app.py::seed_demo               # in-domain demo precedent (REQUIRED for a good demo)
modal deploy infra/modal/app.py                        # deploys API + UI, both scale-to-zero
```

UI URL: `https://<workspace>--accord-ui.modal.run`.

### Why the demo holds up for a first-time visitor (done in code)

- **No 500s.** Every analysis node degrades to a safe default — sentiment →
  neutral, behaviors → all-absent, stance → unknown, and (newly hardened)
  **retrieval → empty precedent + a "degraded, not an error" note** if Neon is
  asleep. A visitor always gets a complete analysis, never a stack trace.
- **Relevant precedent, not campsite bartering.** `seed_demo` loads
  business-negotiation precedents that overlap the default MSA-renewal thread,
  and the UI defaults to that `demo` namespace — so the first click shows
  on-topic precedent *with provenance*, not a cross-domain warning.
- **Cold start hidden.** The UI fires a `/health` ping on page load, so the GPU
  starts warming (~90 s) while the visitor reads, not after they click.

**Verify before sharing** (do this once after deploy):

```bash
curl -s "https://<workspace>--accord-accordserver-api.modal.run/health?probe_graph=true"
# expect: sglang_ready:true, rag_configured:true, graph_populated:true
```

Then open the UI, wait out the one-time warm, and click **Analyze thread** on the
prefilled thread. You should see trajectory + stances + a recommendation citing
`MSA renewal…` / `Auto-renewal clause…` demo precedents with "why this case"
lines. If precedent is empty, `seed_demo` didn't run.

**Remaining caveat:** the first visitor after idle still waits ~90 s for the warm
(the spinner explains it). A keep-warm cron would remove it but defeats the
~$0-idle H100 story — not worth it. A static "click to wake (~90 s)" landing page
is the middle ground.

---

## Summary

| Step | Infra | Modal command |
|------|-------|---------------|
| 1 Outcome | CPU | `eval_outcome` |
| 2 Retrieval | Neon | `graph_ingest` → `eval_retrieval` |
| 3 Sentiment | GPU | `eval_sentiment` |
| 4 RAG ablation | GPU + Neon | `eval_agent` |
| 5 Deploy | Modal | `modal deploy` |

Steps 1, 2, 5 need no GPU. Steps 3–4 are one `modal run` each (SGLang auto-boots)
or one cluster session with SGLang up.
