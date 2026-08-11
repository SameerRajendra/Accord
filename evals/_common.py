"""Shared plumbing for the eval harnesses — loading, preflight, honest statistics.

Kept deliberately small. Three things live here because getting them wrong in
three places would be three different kinds of dishonest:

1. **Preflight.** An eval that needs SGLang or Neon must say *what* is missing
   and *what command fixes it* (`EvalUnavailable`), not emit a stack trace that
   a reader mistakes for "the eval failed, so the system failed."
2. **Deterministic selection.** Every harness samples transcripts the same way,
   seeded, so two runs of the same command compare like-for-like.
3. **Small-sample statistics.** These evals run on tens of items, not
   thousands. A win rate without a confidence interval and a coin-flip
   baseline is a number that reads as evidence and isn't. `wilson_interval`
   and `binomial_two_sided_p` exist so no harness can report the former
   without the latter.

`outcome_eval.py` predates this module and keeps its own loader; it is the
committed reference implementation and is deliberately left untouched.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from data.schema import CaseDocument, Transcript

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_TRANSCRIPTS = Path("data/processed/casino.jsonl")
DEFAULT_CORPUS = Path("data/processed/case_corpus.jsonl")

# Matches agent.llm.DEFAULT_BASE_URL; duplicated only as a fallback for the
# preflight message so a missing `agent` import can't break error reporting.
FALLBACK_BASE_URL = "http://127.0.0.1:30000/v1"


class EvalUnavailable(RuntimeError):
    """A dependency this eval needs is unavailable, with a fix in the message.

    Raised for: missing data files, unset `DATABASE_URL`, an unreachable SGLang
    endpoint, an unpopulated vector store. Harness `main()`s catch it, print the
    message to stderr, and exit 2 — a clean "can't run" distinct from exit 1
    ("ran and something broke").
    """


# --------------------------------------------------------------------------
# Loading + deterministic selection
# --------------------------------------------------------------------------


def load_transcripts(path: Path) -> List[Transcript]:
    """Read normalized `Transcript` JSONL, or explain how to build it."""
    if not path.exists():
        raise EvalUnavailable(
            f"{path} not found. Build the normalized transcripts first:\n"
            f"  python -m data.ingest_casino --download"
        )
    out: List[Transcript] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Transcript.model_validate_json(line))
    if not out:
        raise EvalUnavailable(
            f"{path} is empty — re-run `python -m data.ingest_casino --download`."
        )
    return out


def load_corpus(path: Path) -> List[CaseDocument]:
    """Read the RAG case corpus JSONL, or explain how to build it."""
    if not path.exists():
        raise EvalUnavailable(
            f"{path} not found. Build the case corpus first:\n"
            f"  python -m data.ingest_casino --download\n"
            f"  python -m data.build_case_corpus"
        )
    out: List[CaseDocument] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(CaseDocument.model_validate_json(line))
    if not out:
        raise EvalUnavailable(f"{path} is empty — re-run `python -m data.build_case_corpus`.")
    return out


def select_transcripts(
    transcripts: Sequence[Transcript],
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 0,
    annotated_only: bool = False,
) -> List[Transcript]:
    """Deterministically pick an evaluation subset.

    `split="all"` disables split filtering. When `limit` truncates the pool the
    subset is a *seeded random sample*, not the first N — taking the first N of
    a hash-ordered file would silently correlate the sample with the split
    hash. The result is re-sorted by `dialogue_id` so the run order (and any
    prefix-cache effects it causes) is stable across runs.
    """
    pool = [
        t
        for t in transcripts
        if split in ("all", "") or t.metadata.get("split") == split
    ]
    if annotated_only:
        pool = [t for t in pool if t.has_strategy_annotations]
    pool.sort(key=lambda t: t.dialogue_id)
    if limit is not None and 0 < limit < len(pool):
        rng = random.Random(seed)  # noqa: S311 — sampling, not cryptography
        pool = sorted(rng.sample(pool, limit), key=lambda t: t.dialogue_id)
    return pool


def text_turns(transcript: Transcript) -> List:
    """Free-text turns only — protocol events (Submit-Deal, …) are not utterances."""
    return [t for t in transcript.turns if t.action is None]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    """Write a CSV, creating parents. `None` cells become empty strings.

    An empty cell is the on-disk signal for "this metric was not computed in
    this run"; `evals/report.py` renders those as `not measured` rather than
    dropping the row, so a missing number stays visible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])
    return path


def emit(summary: Dict[str, Any]) -> None:
    """JSON summary to stdout — the shape `outcome_eval.py` established."""
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def fail(exc: EvalUnavailable) -> int:
    """Print an actionable message to stderr and return the exit code."""
    sys.stderr.write(f"\nEVAL CANNOT RUN\n---------------\n{exc}\n\n")
    return 2


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight_llm(base_url: Optional[str] = None, timeout: float = 15.0) -> Dict[str, Any]:
    """Check the self-hosted LLM answers, and report which model it is serving.

    Returns `{"base_url": ..., "models": [...], "model_id": ...}`. The model id
    comes from the server's own `/models` response rather than from a flag, so
    the eval records what actually answered — the difference matters when the
    judge is supposed to be a *different* model from the one under test.

    There is no hosted-API fallback by design (DESIGN.md §1); an unreachable
    endpoint is a hard stop.
    """
    import httpx

    url = (base_url or os.environ.get("SGLANG_BASE_URL", FALLBACK_BASE_URL)).rstrip("/")
    key = os.environ.get("SGLANG_API_KEY", "sglang-no-auth")
    try:
        response = httpx.get(
            f"{url}/models", timeout=timeout, headers={"Authorization": f"Bearer {key}"}
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 — any failure means "cannot run"
        raise EvalUnavailable(
            f"Cannot reach the self-hosted LLM at {url}/models ({type(exc).__name__}: {exc}).\n"
            f"This eval needs a running SGLang server. Either:\n"
            f"  # a) point at an already-running instance\n"
            f"  export SGLANG_BASE_URL=http://127.0.0.1:30000/v1\n"
            f"  # b) start one locally / on the cluster\n"
            f"  python -m sglang.launch_server \\\n"
            f"      --model-path Qwen/Qwen2.5-7B-Instruct --port 30000\n"
            f"  # c) run this eval inside the Modal GPU container, where SGLang is colocated\n"
            f"  #    (see RUN.md; SGLang listens on localhost:30000 there)\n"
            f"No hosted-API fallback exists by design (DESIGN.md §1: open weights only)."
        ) from exc

    models = [m.get("id", "") for m in (payload.get("data") or []) if isinstance(m, dict)]
    return {"base_url": url, "models": models, "model_id": models[0] if models else ""}


def preflight_retrieval(probe: str = "campsite negotiation firewood") -> Dict[str, Any]:
    """Check Neon + pgvector answer a trivial query and the collection is populated."""
    if not os.environ.get("DATABASE_URL"):
        raise EvalUnavailable(
            "DATABASE_URL is not set — this eval retrieves from Neon Postgres + pgvector.\n"
            "  export DATABASE_URL='postgresql://user:pw@host-pooler.region.aws.neon.tech/"
            "neondb?sslmode=require'\n"
            "See infra/neon/README.md for provisioning, RUN.md §1 for the sequence."
        )
    from rag.retriever import retrieve

    try:
        hits = retrieve(probe, k=1)
    except Exception as exc:  # noqa: BLE001 — connection/driver/collection errors
        raise EvalUnavailable(
            f"Retrieval probe failed ({type(exc).__name__}: {exc}).\n"
            f"Check, in order:\n"
            f"  1. DATABASE_URL points at a reachable Neon branch (psql \"$DATABASE_URL\" -c "
            f"'select 1')\n"
            f"  2. the corpus is embedded:\n"
            f"       psql \"$DATABASE_URL\" -c 'SELECT COUNT(*) FROM langchain_pg_embedding;'\n"
            f"       modal run infra/modal/app.py::build_corpus   # if that count is 0\n"
            f"  3. pgvector + the HNSW index exist: psql \"$DATABASE_URL\" -f rag/schema.sql"
        ) from exc

    if not hits:
        raise EvalUnavailable(
            "Retrieval returned 0 results for a generic probe — the `accord_cases` collection "
            "is empty.\n"
            "  modal run infra/modal/app.py::build_corpus   # embeds data/processed/"
            "case_corpus.jsonl\n"
            "  psql \"$DATABASE_URL\" -f rag/schema.sql      # (re-)create the HNSW index"
        )
    return {"probe": probe, "top1_case_id": hits[0].case_id, "top1_score": hits[0].score}


# --------------------------------------------------------------------------
# Judge model construction
# --------------------------------------------------------------------------


@contextmanager
def _env(**overrides: Optional[str]) -> Iterator[None]:
    """Temporarily set/unset environment variables, restoring on exit."""
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def judge_chat_model(
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
):
    """Build a judge client through `agent.llm.chat_model` pointed at a second model.

    `chat_model` reads `SGLANG_BASE_URL`/`SGLANG_MODEL` at construction time and
    memoizes on `(model, temperature, max_tokens)`, so pointing a judge at a
    different endpoint means constructing it under overridden env with the cache
    cleared on both sides of the call: once before (so a memoized
    system-under-test client isn't returned for the judge) and once after (so
    later system-under-test calls don't inherit judge env). The returned client
    has the judge's base_url/model baked in and is unaffected by the reset.

    Consequence, and the reason harnesses run in phases: never construct a judge
    while system-under-test calls are in flight. Every harness here finishes all
    generation, *then* judges.

    Still self-hosted — this points at another SGLang (or SGLang-compatible)
    endpoint serving open weights. No hosted API (DESIGN.md §1).
    """
    from agent.llm import chat_model

    overrides: Dict[str, Optional[str]] = {"SGLANG_MODEL": model}
    if base_url:
        overrides["SGLANG_BASE_URL"] = base_url
    chat_model.cache_clear()
    try:
        with _env(**overrides):
            client = chat_model(model=model, temperature=temperature, max_tokens=max_tokens)
    finally:
        chat_model.cache_clear()
    return client


# --------------------------------------------------------------------------
# Small-sample statistics
# --------------------------------------------------------------------------


def safe_div(numerator: float, denominator: float) -> float:
    """Division that yields NaN instead of raising — an undefined rate is not 0.0."""
    return float(numerator) / float(denominator) if denominator else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (`q` in [0, 100]). NaN on empty input."""
    data = sorted(float(v) for v in values)
    if not data:
        return float("nan")
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return data[int(position)]
    return data[low] + (data[high] - data[low]) * (position - low)


def mean(values: Sequence[float]) -> float:
    return safe_div(sum(float(v) for v in values), len(values))


def stdev(values: Sequence[float]) -> float:
    """Population standard deviation; 0.0 for a single value, NaN when empty."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Used instead of the normal approximation because these harnesses run on
    tens of items, where the normal interval runs off the end of [0, 1] and
    reads as far more precise than the data supports.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denominator
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def binomial_two_sided_p(successes: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (method of small p-values).

    Answers the only question a 12-vs-8 win rate can honestly answer: is this
    distinguishable from a coin flip? NaN when there are no trials.
    """
    if n <= 0:
        return float("nan")

    def pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))

    observed = pmf(successes) * (1.0 + 1e-9)  # tolerance for float ties
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed))


def _ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks with ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on averaged ranks). NaN if degenerate.

    Hand-rolled because scipy is not a declared dependency and adding one for a
    single correlation is not worth it.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else float("nan")


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC with the degenerate cases handled explicitly.

    Returns NaN when one class is absent (AUC undefined) rather than raising.
    Note a constant scorer returns exactly 0.5 — which is why every caller must
    report the constant-output rate next to this number.
    """
    if len(set(labels)) < 2 or not labels:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(list(labels), list(scores)))


def cramers_v(table: Sequence[Sequence[int]]) -> float:
    """Cramér's V for an r×c contingency table — association strength in [0, 1].

    No p-value is reported: that would need scipy, and with these cell counts a
    p-value would be the least trustworthy number on the page anyway.
    """
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows < 2 or cols < 2:
        return float("nan")
    total = sum(sum(row) for row in table)
    if total <= 0:
        return float("nan")
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = row_sums[r] * col_sums[c] / total
            if expected > 0:
                chi2 += (table[r][c] - expected) ** 2 / expected
    return math.sqrt(chi2 / (total * min(rows - 1, cols - 1)))


def normalized_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy of a label distribution, scaled to [0, 1].

    1.0 = uniform across the taxonomy, 0.0 = one label always. The companion
    number to any self-consistency metric: a model that answers "neutral" every
    time is perfectly self-consistent and has entropy 0.
    """
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return float("nan")
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy / math.log(len(counts))
