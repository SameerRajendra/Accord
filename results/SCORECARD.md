# Accord — eval scorecard

**12/15 metrics measured** · 0 failing · 1 warnings · 3 not yet run.

Statuses: ✅ pass · ⚠️ warn · ❌ fail · ◐ by-design (not measurable, documented) · • info (characterization) · — not measured. Thresholds are stated per row and are conservative; disagree with the call, trust the number.

| Dimension | Metric | Value | Status | Threshold / note | Source |
|---|---|---|:---:|---|---|
| Serving | Peak throughput | 4699 tok/s @ c=64 | • | characterization, not a pass/fail | `batching_curve.csv` |
| Serving | Cost at saturation | $0.27 / 1M out-tok | • | amortized GPU-second | `batching_curve.csv` |
| Serving | Cold start | 98 s | • | scale-to-zero wake; first-visit UX caveat | `coldstart.csv` |
| Retrieval | recall@5 (pgvector, dialogue) | 1.0% | • | vs random 0.5%; struct-match 29.4% vs chance 26.9% | `retrieval.csv` |
| Retrieval | recall@5 (graph, dialogue) | 0.0% | • | vs random 0.5%; struct-match 22.1% vs chance 26.9% | `retrieval.csv` |
| Retrieval | recall@5 (hybrid, dialogue) | 1.0% | • | vs random 0.5%; struct-match 22.7% vs chance 26.9% | `retrieval.csv` |
| RAG ablation | RAG vs no-RAG win-rate | 26.7% (95% CI 10.9%–52.0%) | ⚠️ | pass only if the CI clears 50% | `agent_eval.csv` |
| RAG ablation | Citation fabrication (deterministic) | 1.8% | ✅ | lexical grounding floor; lower is better | `agent_eval.csv` |
| RAG triad | faithfulness | — | — | run: python -m evals.rag_triad_eval | `rag_triad.csv` |
| Sentiment | reliability | — | — | run: python -m evals.sentiment_eval | `sentiment.csv` |
| Outcome model | Accuracy lift over base rate | 0.020 | ◐ | near-degenerate target (base rate 96.1%); reported, not claimed | `outcome.csv` |
| Outcome model | Breakdown recall (minority class) | 50.0% | ◐ | few positives — read the confusion counts, not this alone | `outcome.csv` |
| Safety | Prompt-injection | — | — | run: python -m evals.safety_eval --checks injection | `safety_injection.csv` |
| Safety | Tenant/corpus isolation violations | 0 | ✅ | MUST be 0 — any cross-namespace leak is a shipping blocker | `safety_isolation.csv` |
| Safety | High-severity PII hits | 0 | ✅ | SSN/card/IBAN/account patterns in scanned surface | `safety_pii.csv` |

## — Not yet run
These are gaps, not passes. See RUN_EVALS.md.
- RAG triad / faithfulness — run: python -m evals.rag_triad_eval
- Sentiment / reliability — run: python -m evals.sentiment_eval
- Safety / Prompt-injection — run: python -m evals.safety_eval --checks injection

> Honesty note: LLM-judged rows (RAG triad, some agent-eval) are only as trustworthy as the judge. Use a stronger self-hosted judge than the model under test and read the self-consistency number before quoting them. No human-vs-judge calibration exists yet.
