"""Retrieval eval: recall@k and MRR@k over the pgvector corpus, against baselines.

There is no human relevance-judgement set for this corpus, so the query set is
constructed from CaSiNo's own structure. **How that labeling works, and where it
is weak, is the most important thing in this file** — the numbers mean nothing
without it.

Labeling strategy
-----------------
Three query modes, each with exactly one document labeled relevant:

``dialogue`` (default; production-faithful)
    Query = the last ≤4 free-text turns of a held-out dialogue, built by
    ``agent.graph._default_query`` — the *same function the agent calls in
    production*, imported rather than reimplemented so this eval cannot drift
    into measuring a query the system never issues. Gold = the case document
    built from that same dialogue (``casino-<dialogue_id>``), which is in the
    index because ``rag/embed.py`` embeds the whole corpus.
    Rationale: the document derived from dialogue D is, by construction, the
    single most relevant document in the corpus for a query drawn from D. No
    human judgement is invented; the label falls out of the build.

``summary`` (index sanity check / upper bound)
    Query = the gold document's own text. Gold = itself. Recall@1 must be ~1.0;
    anything less means the index or the embedding path is broken. **This is not
    a quality claim** — it is a control, and the gap between ``summary`` and
    ``dialogue`` is the quantity of interest: it isolates how much retrieval
    loses purely to the register mismatch between raw dialogue (the query) and
    templated structured summaries (the documents).

``strategy`` (human labels)
    Query = one utterance that CaSiNo's *human annotators* labeled with exactly
    one persuasion strategy. Gold = that strategy's playbook document
    (``strategy-<name>``). This is the only mode whose relevance label comes
    from a human rather than from the corpus build.

Weaknesses of this labeling — all of which push the reported numbers around,
and none of which are fixed by running more queries
---------------------------------------------------
1. **Single-gold assumption.** Exactly one document counts as relevant. Many
   other cases share a priority profile and outcome and would be equally good
   precedent; retrieving those scores as a miss. Recall@k here is therefore a
   *lower bound* on retrieval usefulness, not an estimate of it.
2. **Self-retrieval is not precedent retrieval.** In ``dialogue`` mode the gold
   is derived from the query's own dialogue. In production the live negotiation
   has no document in the corpus, so the real task ("find a *different*, similar
   past case") is strictly harder than what is measured here. Treat ``dialogue``
   recall as an optimistic bound on the production behaviour.
3. **The corpus is templated.** All 1,030 case documents share a near-identical
   skeleton; only priority labels, strategy counts, points and the lesson vary.
   Every document is similar to every other document, which compresses the
   similarity range — absolute cosine scores are close to meaningless on this
   corpus, and rank is the only trustworthy signal. (The compressed range is
   what made a top-5 of amicable agreements look plausible for a hostile query
   in the citation-fabrication failure; see ``evals/agent_eval.py``.)
4. **Register mismatch is measured, not controlled for.** ``dialogue`` queries
   are conversational text; documents are structured summaries. That is the
   production configuration, so it is the honest default — but a low number here
   is a statement about the corpus rendering, not only about the embedding model.
5. **``strategy`` mode excludes multi-label utterances**, which biases it toward
   prototypical, unambiguous examples — an easier task than the annotation
   distribution as a whole.
6. **"Held-out" refers to the queries, not the index.** Nothing is trained, so
   test-split documents are deliberately present in the corpus; the split only
   controls which dialogues are used as queries.
7. **Single-gold labeling structurally disadvantages the graph retrievers**, and
   the size of that handicap is unknown. See the next section — this is the one
   caveat that can invert a conclusion rather than just shift a number.

Retrievers (all reported on every row, because recall@10 over a 1,040-document
corpus sounds impressive until you see what chance and bag-of-words get)
------------------------------------------------------------------------------
- ``random`` — analytic expectation for a uniformly random ranking with one
  relevant document: recall@k = k/N, MRR@k = (Σ_{r≤k} 1/r)/N. No queries issued.
- ``tfidf`` — plain lexical TF-IDF cosine over the same corpus, computed locally
  with no database. If MiniLM embeddings do not beat bag-of-words, that is the
  finding, and DESIGN.md §8's "revisit if retrieval recall@k underperforms"
  (BGE via a sibling SGLang server) is the indicated next step.
- ``pgvector`` — the deployed path: MiniLM (384-d) + Neon HNSW via
  ``rag.retriever.retrieve``.
- ``graph`` — knowledge-graph traversal only (``graph_retrieve(...,
  use_vector=False)``): the query is mapped onto structural anchors — contested
  issues, persuasion tactics, outcome class, conflict structure.
- ``hybrid`` — graph and vector fused (``--fusion weighted --alpha 0.6`` by
  default; ``--fusion rrf`` and an ``--alpha`` sweep are the other arms
  ``infra/graph/README.md`` §7 asks for, reachable by flag).

**Read the graph rows against the right question.** Recall@k here asks "did you
return *this exact document*". Vector search can do that: the gold document and
the query share surface wording, so retrieving a dialogue's own case doc is
squarely the task. Graph retrieval cannot, structurally — it anchors on facts
(same conflict structure, same contested issue, same outcome class) that dozens
of dialogues share by construction, so it returns the gold document's
*equivalence class*, and single-gold scoring counts every other member of that
class as a miss. A low graph recall@k on this labeling is therefore expected and
is **not** evidence that graph retrieval finds worse precedent; it is evidence
that this labeling measures self-retrieval.

That is why every row also carries ``structural_match_at_k``: of the case
documents returned, the share whose dialogue has the *same conflict structure*
as the query's dialogue — a proxy for "is this usable precedent" that does not
presuppose one right answer. It is reported next to
``structural_match_baseline``, the rate a uniformly random draw would get given
how common that structure is in the corpus, because on a corpus where one
structure dominates a high match rate means nothing on its own. Neither number
is a relevance judgement; both are cheap, deterministic, and honest about what
they are.

Outputs
-------
- ``results/retrieval.csv`` — one row per (query_mode, retriever, k).
- ``results/retrieval_queries.csv`` — one row per query with the gold rank and
  the top-1 hit, so every aggregate above is auditable and recomputable.

Usage::

    python -m evals.retrieval_eval                       # all modes, all retrievers
    python -m evals.retrieval_eval --offline             # tfidf + random only, no Neon
    python -m evals.retrieval_eval --query-mode dialogue --k 1 5 20
    python -m evals.retrieval_eval --split all --max-queries 500
    python -m evals.retrieval_eval --retriever pgvector graph hybrid --fusion rrf
    python -m evals.retrieval_eval --retriever hybrid --alpha 0.25   # alpha sweep
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.ingest_casino import STRATEGY_VOCAB
from data.schema import CaseDocument, Transcript
from evals._common import (
    DEFAULT_CORPUS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRANSCRIPTS,
    EvalUnavailable,
    emit,
    fail,
    load_corpus,
    load_transcripts,
    mean,
    preflight_retrieval,
    safe_div,
    select_transcripts,
    text_turns,
    write_csv,
)
from rag.graph_schema import STRUCT_UNKNOWN, conflict_structure

QUERY_MODES = ("dialogue", "summary", "strategy")
RETRIEVERS = ("pgvector", "graph", "hybrid", "tfidf", "random")
#: Retrievers that need a populated knowledge graph, not just the vector store.
GRAPH_RETRIEVERS = ("graph", "hybrid")
#: Retrievers that need Neon at all (as opposed to running purely locally).
DATABASE_RETRIEVERS = ("pgvector",) + GRAPH_RETRIEVERS
DEFAULT_KS = (1, 3, 5, 10)

_MODE_NOTES = {
    "dialogue": "production-faithful: agent.graph._default_query over held-out dialogues; "
    "gold = that dialogue's own case doc (self-retrieval, single-gold)",
    "summary": "CONTROL, not a quality claim: query text == gold doc text; recall@1 ~1.0 is "
    "the expected sanity result",
    "strategy": "human-annotated single-strategy utterance -> that strategy's playbook doc",
}

#: Carried into every row so a reader of the CSV alone cannot mistake a low
#: graph recall for a quality verdict. The caveat has to travel with the
#: number, not live only in this module's docstring.
_GRAPH_CAVEAT = (
    "single-gold recall UNDERSTATES this arm by construction: graph anchors retrieve the "
    "gold document's structural equivalence class, and every member but the gold scores as "
    "a miss — read structural_match_at_k vs structural_match_baseline alongside it"
)
_RETRIEVER_NOTES = {
    "pgvector": "deployed path: MiniLM 384-d + Neon HNSW cosine",
    "graph": "graph traversal only (use_vector=False). " + _GRAPH_CAVEAT,
    "hybrid": "graph + vector fused. " + _GRAPH_CAVEAT,
    "tfidf": "lexical bag-of-words reference point, stock TfidfVectorizer, no database",
    "random": "analytic expectation, no queries issued",
}


@dataclass
class Query:
    """One evaluation query and its single labeled-relevant document."""

    query_id: str
    text: str
    gold_case_id: str
    structure: Optional[str] = None
    """Conflict structure of the dialogue this query came from, when it came
    from one. Powers `structural_match_at_k`, the diagnostic that reads graph
    retrieval on its own terms instead of only through single-gold recall."""


@dataclass
class Ranking:
    """A retriever's ranked answer to one query (truncated at max_k)."""

    query: Query
    case_ids: List[str]
    scores: List[float]

    def gold_rank(self) -> Optional[int]:
        """1-based rank of the gold document, or None if outside the truncation."""
        for position, case_id in enumerate(self.case_ids, start=1):
            if case_id == self.query.gold_case_id:
                return position
        return None


# --------------------------------------------------------------------------
# Structural diagnostic
# --------------------------------------------------------------------------
#
# Single-gold recall asks "did you return this exact document". That is the
# vector retriever's task by construction and not the graph retriever's, so a
# head-to-head on recall alone would answer a question only one of the two is
# playing. These three helpers add a second reading that does not presuppose
# one right answer: of the cases returned, how many stand in the same
# structural relation as the query's dialogue — and how many would a random
# draw have got, given how common that structure is.


def transcript_structure(transcript: Transcript) -> Optional[str]:
    """The dialogue's conflict structure, or None when it isn't derivable.

    Uses `rag.graph_schema.conflict_structure` — the same function the graph
    ingest and the query planner use, so this diagnostic cannot drift into
    measuring a different notion of "same structure" than the retriever does.
    """
    parties = list(transcript.parties)
    if len(parties) != 2:
        return None
    structure = conflict_structure(parties[0].priorities, parties[1].priorities)
    return None if structure == STRUCT_UNKNOWN else structure


def structure_index(transcripts: Sequence[Transcript]) -> Dict[str, str]:
    """case_id -> conflict structure, for every dialogue with a derivable one.

    Keyed the way `build_queries` keys gold ids (`source-dialogue_id`), which
    is also how `data/build_case_corpus.py` composes `case_id` — including its
    doubled `casino-casino-N` form. Consistency with the corpus matters more
    than tidiness here; see infra/graph/README.md §6.
    """
    index: Dict[str, str] = {}
    for transcript in transcripts:
        structure = transcript_structure(transcript)
        if structure is not None:
            index[f"{transcript.source}-{transcript.dialogue_id}"] = structure
    return index


def structural_match(
    rankings: Sequence[Ranking], structures: Dict[str, str], k: int
) -> Dict[str, Any]:
    """Share of returned cases sharing the query dialogue's conflict structure.

    Playbook documents and any case whose structure is unknown are dropped
    from the denominator rather than counted as misses — they are not wrong
    answers, they are answers this diagnostic cannot judge. A query whose
    whole top-k is undecidable is excluded and counted in `n_scored`, so a
    thin denominator is visible instead of inflating the mean.

    `baseline` is what a uniform random draw would score: the corpus-wide
    prevalence of each query's own structure, averaged over queries. On a
    corpus where one structure dominates, the match rate alone means nothing.
    """
    if not structures:
        return {"match": None, "baseline": None, "n_scored": 0}

    prevalence: Dict[str, float] = {}
    total = float(len(structures))
    for structure in structures.values():
        prevalence[structure] = prevalence.get(structure, 0.0) + 1.0
    for structure in prevalence:
        prevalence[structure] /= total

    shares: List[float] = []
    baselines: List[float] = []
    for ranking in rankings:
        if ranking.query.structure is None:
            continue
        judged = [
            structures[case_id]
            for case_id in ranking.case_ids[:k]
            if case_id in structures
        ]
        if not judged:
            continue
        matches = sum(1 for s in judged if s == ranking.query.structure)
        shares.append(matches / len(judged))
        baselines.append(prevalence.get(ranking.query.structure, 0.0))

    if not shares:
        return {"match": None, "baseline": None, "n_scored": 0}
    return {"match": mean(shares), "baseline": mean(baselines), "n_scored": len(shares)}


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------


def _production_query_fn():
    """Import the agent's own query builder — fail loudly if it moved.

    Deliberately not reimplemented: a local copy would keep passing after
    `agent/graph.py` changed how it builds the retrieval query, and this eval
    would quietly start measuring a query production never issues.
    """
    try:
        from agent.graph import _default_query
    except ImportError as exc:  # pragma: no cover - guards against upstream rename
        raise EvalUnavailable(
            "Could not import `agent.graph._default_query`, the function that builds the "
            "retrieval query in production.\n"
            "`dialogue` mode exists to measure exactly what production retrieves, so it "
            "refuses to fall back to a local copy.\n"
            "Fix: update the import in evals/retrieval_eval.py to the renamed function, or "
            "run with --query-mode summary strategy."
        ) from exc
    return _default_query


def build_queries(
    mode: str,
    transcripts: Sequence[Transcript],
    corpus_by_id: Dict[str, CaseDocument],
    max_queries: int,
) -> Tuple[List[Query], int]:
    """Build the query set for one mode. Returns (queries, n_skipped_missing_gold)."""
    queries: List[Query] = []
    skipped = 0

    if mode in ("dialogue", "summary"):
        default_query = _production_query_fn() if mode == "dialogue" else None
        for transcript in transcripts:
            gold_id = f"{transcript.source}-{transcript.dialogue_id}"
            document = corpus_by_id.get(gold_id)
            if document is None:
                # Corpus is stale relative to the transcripts — count it rather
                # than silently shrinking the denominator.
                skipped += 1
                continue
            if mode == "dialogue":
                text = default_query(transcript)
                if not text.strip():
                    skipped += 1
                    continue
            else:
                text = document.text
            queries.append(
                Query(
                    query_id=transcript.dialogue_id,
                    text=text,
                    gold_case_id=gold_id,
                    structure=transcript_structure(transcript),
                )
            )

    elif mode == "strategy":
        for transcript in transcripts:
            for turn in text_turns(transcript):
                labels = [s for s in turn.strategies if s in STRATEGY_VOCAB]
                # Exactly one label: a multi-label utterance has more than one
                # defensible gold document, which single-gold scoring cannot
                # represent honestly.
                if len(labels) != 1 or not turn.text.strip():
                    continue
                gold_id = f"strategy-{labels[0]}"
                if gold_id not in corpus_by_id:
                    skipped += 1
                    continue
                queries.append(
                    Query(
                        query_id=f"{transcript.dialogue_id}#{turn.index}",
                        text=turn.text,
                        gold_case_id=gold_id,
                    )
                )
    else:
        raise ValueError(f"unknown query mode: {mode}")

    if 0 < max_queries < len(queries):
        # Evenly spaced rather than the first N: utterance-level queries arrive
        # grouped by dialogue, so a head-slice would sample a handful of
        # dialogues instead of the whole pool.
        step = len(queries) / max_queries
        queries = [queries[int(i * step)] for i in range(max_queries)]
    return queries, skipped


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------


def rank_with_pgvector(queries: Sequence[Query], max_k: int) -> List[Ranking]:
    """The deployed path: MiniLM + Neon pgvector through `rag.retriever.retrieve`."""
    from rag.retriever import retrieve

    rankings: List[Ranking] = []
    for query in queries:
        hits = retrieve(query.text, k=max_k)
        rankings.append(
            Ranking(
                query=query,
                case_ids=[h.case_id for h in hits],
                scores=[float(h.score) for h in hits],
            )
        )
    return rankings


def rank_with_graph(
    queries: Sequence[Query],
    max_k: int,
    use_vector: bool,
    fusion: str,
    alpha: float,
) -> List[Ranking]:
    """Graph traversal, optionally fused with vector similarity.

    Deliberately calls `graph_retrieve` on the query *text*, not
    `graph_retrieve_for_transcript`, even though the transcript is available
    here: the agent's transcript-anchored path (`agent/tools.py`) is a
    different retriever with strictly more information, and scoring it against
    the text arms would compare two different tasks. Benchmarking that path is
    its own experiment — it belongs with the agent eval, which has the
    transcript in hand for a live case.

    `graph_retrieve` degrades to vector-only rather than raising if the graph
    is unreachable (infra/graph/README.md §6), which is why `run_eval` probes
    `graph_is_populated()` up front: a silently degraded graph arm here would
    publish a benchmark of the vector retriever under the graph's name.
    """
    from rag.graph_retriever import graph_retrieve

    rankings: List[Ranking] = []
    for query in queries:
        hits = graph_retrieve(
            query.text,
            k=max_k,
            use_vector=use_vector,
            fusion=fusion,
            alpha=alpha,
        )
        rankings.append(
            Ranking(
                query=query,
                case_ids=[h.case_id for h in hits],
                scores=[float(h.score) for h in hits],
            )
        )
    return rankings


def rank_with_tfidf(
    queries: Sequence[Query], documents: Sequence[CaseDocument], max_k: int
) -> List[Ranking]:
    """Lexical TF-IDF cosine baseline — no database, no embedding model.

    Stock `TfidfVectorizer` settings (lowercase, l2-normalized, no stopword
    list) so this is a plain bag-of-words reference point and not a tuned
    competitor. l2 normalization makes the dot product the cosine.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([d.text for d in documents])
    case_ids = [d.case_id for d in documents]
    similarity_rows = (vectorizer.transform([q.text for q in queries]) @ matrix.T).toarray()

    rankings: List[Ranking] = []
    for query, similarities in zip(queries, similarity_rows):
        order = sorted(range(len(case_ids)), key=similarities.__getitem__, reverse=True)[:max_k]
        rankings.append(
            Ranking(
                query=query,
                case_ids=[case_ids[i] for i in order],
                scores=[float(similarities[i]) for i in order],
            )
        )
    return rankings


def random_baseline(corpus_size: int, k: int) -> Tuple[float, float]:
    """Analytic (recall@k, MRR@k) for a uniformly random ranking, one gold doc.

    Closed form, so it costs nothing and cannot be accidentally tuned:
    recall@k = k/N, MRR@k = (Σ_{r=1..k} 1/r) / N.
    """
    if corpus_size <= 0:
        return (float("nan"), float("nan"))
    recall = min(k, corpus_size) / corpus_size
    mrr = sum(1.0 / r for r in range(1, min(k, corpus_size) + 1)) / corpus_size
    return (recall, mrr)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_rankings(rankings: Sequence[Ranking], k: int) -> Dict[str, float]:
    """recall@k and MRR@k over a set of single-gold rankings."""
    if not rankings:
        return {"recall": float("nan"), "mrr": float("nan"), "n": 0}
    hits = 0
    reciprocal_sum = 0.0
    for ranking in rankings:
        rank = ranking.gold_rank()
        if rank is not None and rank <= k:
            hits += 1
            reciprocal_sum += 1.0 / rank
    return {
        "recall": hits / len(rankings),
        "mrr": reciprocal_sum / len(rankings),
        "n": len(rankings),
    }


def _rank_stats(rankings: Sequence[Ranking]) -> Dict[str, float]:
    """Diagnostics that survive the single-gold caveat: score range and found-ranks."""
    found = [r.gold_rank() for r in rankings]
    found_ranks = [r for r in found if r is not None]
    top1_scores = [r.scores[0] for r in rankings if r.scores]
    gold_scores = [
        r.scores[r.gold_rank() - 1]
        for r in rankings
        if r.gold_rank() is not None and r.scores
    ]
    return {
        "mean_top1_score": mean(top1_scores) if top1_scores else float("nan"),
        "mean_gold_score_when_found": mean(gold_scores) if gold_scores else float("nan"),
        "mean_gold_rank_when_found": mean(found_ranks) if found_ranks else float("nan"),
        "found_rate_within_max_k": safe_div(len(found_ranks), len(rankings)),
    }


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def preflight_graph() -> Dict[str, object]:
    """Refuse to run a graph arm against a graph that isn't there.

    `graph_retrieve` catches its own failures and returns the vector result
    (infra/graph/README.md §6) — correct for serving, disastrous for a
    benchmark: the graph rows would be the vector retriever wearing the
    graph's name, and nothing in the output would say so.
    """
    from rag.graph_db import graph_is_populated

    try:
        populated = graph_is_populated()
    except Exception as exc:  # noqa: BLE001 — driver / DSN / missing tables
        raise EvalUnavailable(
            f"Could not query the knowledge-graph tables ({type(exc).__name__}: {exc}).\n"
            f"The graph arms need them loaded in the same database as the vector store:\n"
            f"  psql \"$DATABASE_URL\" -f rag/graph_schema.sql\n"
            f"  python -m rag.graph_ingest --dry-run   # verify cases_unmatched == 0 first\n"
            f"  python -m rag.graph_ingest\n"
            f"Or drop the graph arms: --retriever pgvector tfidf random"
        ) from exc

    if not populated:
        raise EvalUnavailable(
            "The knowledge-graph tables exist but are empty, so a graph arm would silently "
            "measure the vector retriever instead.\n"
            "  python -m rag.graph_ingest --dry-run   # check cases_unmatched == 0\n"
            "  python -m rag.graph_ingest\n"
            "Or drop the graph arms: --retriever pgvector tfidf random"
        )
    return {"graph_populated": True}


def run_eval(
    transcripts_path: Path,
    corpus_path: Path,
    results_dir: Path,
    query_modes: Sequence[str],
    retrievers: Sequence[str],
    ks: Sequence[int],
    split: str,
    limit: int,
    max_queries: int,
    seed: int,
    fusion: str = "weighted",
    alpha: float = 0.6,
) -> dict:
    corpus = load_corpus(corpus_path)
    corpus_by_id = {d.case_id: d for d in corpus}
    corpus_size = len(corpus)
    all_transcripts = load_transcripts(transcripts_path)
    max_k = max(ks)
    # Built over ALL transcripts, not the query pool: the diagnostic asks about
    # the documents that came back, which are drawn from the whole corpus.
    structures = structure_index(all_transcripts)

    probe: Optional[dict] = None
    graph_probe: Optional[dict] = None
    if any(r in DATABASE_RETRIEVERS for r in retrievers):
        # Fail here, with instructions, rather than mid-loop with a driver traceback.
        probe = preflight_retrieval()
    if any(r in GRAPH_RETRIEVERS for r in retrievers):
        graph_probe = preflight_graph()

    summary_rows: List[List[object]] = []
    per_query_rows: List[List[object]] = []
    modes_summary: Dict[str, dict] = {}

    for mode in query_modes:
        transcript_pool = select_transcripts(
            all_transcripts,
            split=split,
            limit=limit or None,
            seed=seed,
            annotated_only=(mode == "strategy"),
        )
        queries, skipped = build_queries(mode, transcript_pool, corpus_by_id, max_queries)
        if not queries:
            modes_summary[mode] = {
                "n_queries": 0,
                "skipped_missing_gold": skipped,
                "note": "no queries could be built for this mode (empty pool after filtering)",
            }
            continue

        mode_result: Dict[str, dict] = {}
        for retriever in retrievers:
            structural: Dict[int, Dict[str, Any]] = {
                k: {"match": None, "baseline": None, "n_scored": 0} for k in ks
            }
            if retriever == "random":
                stats = {
                    "mean_top1_score": None,
                    "mean_gold_score_when_found": None,
                    "mean_gold_rank_when_found": None,
                    "found_rate_within_max_k": None,
                }
                scored = {
                    k: dict(zip(("recall", "mrr"), random_baseline(corpus_size, k))) for k in ks
                }
            else:
                if retriever == "pgvector":
                    rankings = rank_with_pgvector(queries, max_k)
                elif retriever in GRAPH_RETRIEVERS:
                    rankings = rank_with_graph(
                        queries,
                        max_k,
                        use_vector=(retriever == "hybrid"),
                        fusion=fusion,
                        alpha=alpha,
                    )
                else:
                    rankings = rank_with_tfidf(queries, corpus, max_k)
                stats = _rank_stats(rankings)
                scored = {k: score_rankings(rankings, k) for k in ks}
                structural = {k: structural_match(rankings, structures, k) for k in ks}
                for ranking in rankings:
                    rank = ranking.gold_rank()
                    per_query_rows.append(
                        [
                            mode,
                            retriever,
                            ranking.query.query_id,
                            ranking.query.gold_case_id,
                            rank,
                            ranking.case_ids[0] if ranking.case_ids else None,
                            round(ranking.scores[0], 6) if ranking.scores else None,
                            round(ranking.scores[rank - 1], 6)
                            if rank is not None and ranking.scores
                            else None,
                        ]
                    )

            for k in ks:
                random_recall, random_mrr = random_baseline(corpus_size, k)
                recall = scored[k]["recall"]
                summary_rows.append(
                    [
                        mode,
                        retriever,
                        len(queries),
                        corpus_size,
                        k,
                        recall,
                        scored[k]["mrr"],
                        random_recall,
                        random_mrr,
                        (recall - random_recall) if not math.isnan(recall) else None,
                        structural[k]["match"],
                        structural[k]["baseline"],
                        structural[k]["n_scored"] or None,
                        stats["mean_top1_score"],
                        stats["mean_gold_score_when_found"],
                        stats["mean_gold_rank_when_found"],
                        stats["found_rate_within_max_k"],
                        fusion if retriever in GRAPH_RETRIEVERS else None,
                        alpha if retriever in GRAPH_RETRIEVERS else None,
                        skipped,
                        _MODE_NOTES.get(mode, ""),
                        _RETRIEVER_NOTES.get(retriever, ""),
                    ]
                )
            mode_result[retriever] = {
                f"recall@{k}": scored[k]["recall"] for k in ks
            }
            mode_result[retriever].update({f"mrr@{k}": scored[k]["mrr"] for k in ks})
            mode_result[retriever].update(
                {
                    f"structural_match@{k}": structural[k]["match"]
                    for k in ks
                    if structural[k]["match"] is not None
                }
            )
            mode_result[retriever].update(
                {key: value for key, value in stats.items() if value is not None}
            )
            if retriever in GRAPH_RETRIEVERS:
                mode_result[retriever]["fusion"] = fusion
                mode_result[retriever]["alpha"] = alpha
                mode_result[retriever]["note"] = _RETRIEVER_NOTES[retriever]

        modes_summary[mode] = {
            "n_queries": len(queries),
            "skipped_missing_gold": skipped,
            "retrievers": mode_result,
            "note": _MODE_NOTES.get(mode, ""),
        }

    summary_path = write_csv(
        results_dir / "retrieval.csv",
        [
            "query_mode",
            "retriever",
            "n_queries",
            "corpus_size",
            "k",
            "recall_at_k",
            "mrr_at_k",
            "random_recall_at_k",
            "random_mrr_at_k",
            "recall_lift_over_random",
            "structural_match_at_k",
            "structural_match_baseline",
            "structural_match_n_queries",
            "mean_top1_score",
            "mean_gold_score_when_found",
            "mean_gold_rank_when_found",
            "found_rate_within_max_k",
            "fusion",
            "alpha",
            "queries_skipped_missing_gold",
            "labeling_note",
            "retriever_note",
        ],
        summary_rows,
    )
    queries_path = write_csv(
        results_dir / "retrieval_queries.csv",
        [
            "query_mode",
            "retriever",
            "query_id",
            "gold_case_id",
            "gold_rank",
            "top1_case_id",
            "top1_score",
            "gold_score",
        ],
        per_query_rows,
    )

    out = {
        "corpus": str(corpus_path),
        "corpus_size": corpus_size,
        "split": split,
        "ks": list(ks),
        "retrievers": list(retrievers),
        "retrieval_probe": probe,
        "modes": modes_summary,
        "summary_csv": str(summary_path),
        "per_query_csv": str(queries_path),
        "labeling_caveats": [
            "single-gold: one relevant doc per query; equally-good precedents score as misses",
            "dialogue mode is self-retrieval (gold derived from the query's own dialogue) — "
            "an optimistic bound on production, where the live case is not in the corpus",
            "summary mode is a control, not a quality claim",
            "templated corpus compresses similarity scores; trust rank, not absolute score",
        ],
    }
    if graph_probe is not None:
        out["graph_probe"] = graph_probe
        out["fusion"] = fusion
        out["alpha"] = alpha
        out["labeling_caveats"].append(
            "single-gold recall structurally understates the graph arms: they retrieve the "
            "gold document's equivalence class (same structure / issue / outcome), and every "
            "member of it other than the gold itself scores as a miss. Read "
            "structural_match_at_k against structural_match_baseline alongside recall, and do "
            "not report 'graph is worse than vector' from recall alone."
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval recall@k / MRR@k against random and lexical baselines."
    )
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--query-mode",
        nargs="+",
        choices=list(QUERY_MODES),
        default=list(QUERY_MODES),
        help="Which query sets to build (default: all three).",
    )
    parser.add_argument(
        "--retriever",
        nargs="+",
        choices=list(RETRIEVERS),
        default=list(RETRIEVERS),
        help="Which retrievers to score (default: all three).",
    )
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument(
        "--split",
        default="test",
        help="Transcript split used to build queries ('all' for the whole corpus).",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap on transcripts sampled (0 = the whole split)."
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=200,
        help="Cap on queries per mode (0 = uncapped). Bounds Neon round-trips.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fusion",
        choices=("weighted", "rrf"),
        default="weighted",
        help="Graph/vector fusion for the graph arms. 'rrf' ignores score magnitudes, which "
        "this corpus's narrow 0.42-0.44 similarity band argues for — unmeasured either way.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Graph's share of the hybrid blend (1.0 = graph only, 0.0 = vector only). Note "
        "alpha=0.0 is NOT the pgvector baseline — the candidate pool is still the union of "
        "both lists. Compare against the pgvector row directly.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip every retriever that needs Neon (pgvector, graph, hybrid) — run only the "
        "tfidf and random baselines, no DATABASE_URL needed.",
    )
    args = parser.parse_args(argv)

    retrievers = [
        r for r in args.retriever if not (args.offline and r in DATABASE_RETRIEVERS)
    ]
    if not retrievers:
        parser.error("--offline removed every selected retriever; add --retriever tfidf random")
    ks = sorted({k for k in args.k if k > 0})
    if not ks:
        parser.error("--k needs at least one positive value")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")

    try:
        summary = run_eval(
            transcripts_path=args.transcripts,
            corpus_path=args.corpus,
            results_dir=args.results_dir,
            query_modes=args.query_mode,
            retrievers=retrievers,
            ks=ks,
            split=args.split,
            limit=args.limit,
            max_queries=args.max_queries,
            seed=args.seed,
            fusion=args.fusion,
            alpha=args.alpha,
        )
    except EvalUnavailable as exc:
        return fail(exc)

    emit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
