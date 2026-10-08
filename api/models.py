"""Pydantic v2 request/response contracts for the API.

Public API contract: `AnalyzeRequest` (a `Transcript` + the two retrieval
ablation knobs) → `AnalyzeResponse` (per-turn sentiment, per-party stance,
discussion trajectory, extreme-behavior flags, outcome probability, retrieved
precedents with their graph provenance, recommendation). Kept intentionally
close to the domain models — the API is a thin surface over `agent.graph.run`,
not a translation layer.

The one place it is *not* thin is `RetrievedCasePayload`; see its docstring.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from agent.graph import RetrievalInfo, RetrievalMode
from analysis.behaviors import BehaviorFlag
from analysis.sentiment import PerTurnSentiment
from analysis.stance import PartyStance, Trajectory
from data.schema import Transcript
from rag.documents import SourceType

_MODE_HELP = (
    "Which retriever answers when `use_rag` is on: 'vector' (pgvector baseline, the "
    "deployed default), 'graph' (graph traversal only), or 'hybrid' (both, fused). "
    "Ignored when `use_rag` is false."
)

_NAMESPACE_HELP = (
    "Live knowledge-base namespace to retrieve precedent from — the documents ingested "
    "via POST /corpus/documents. Omit it to search the frozen CaSiNo research corpus "
    "instead, which is what the committed evals score against and is campsite "
    "resource-bartering, not your domain."
)

_INDEX_RESULT_HELP = (
    "Keep this negotiation in the namespace as precedent for future analyses. Requires "
    "`namespace`. Opt-in rather than automatic: indexing every request would fill the "
    "corpus with half-finished threads and demo traffic, and the decision to retain "
    "someone's negotiation is not one an API should make on their behalf."
)


class AnalyzeRequest(BaseModel):
    transcript: Transcript
    use_rag: bool = Field(True, description="Toggle RAG for the ablation. Defaults to on.")
    retrieval_mode: RetrievalMode = Field(RetrievalMode.VECTOR, description=_MODE_HELP)
    retrieval_query: Optional[str] = Field(
        default=None,
        description=(
            "Override the query used for precedent retrieval. Defaults to the last few turns."
        ),
    )
    namespace: Optional[str] = Field(default=None, description=_NAMESPACE_HELP)
    index_result: bool = Field(default=False, description=_INDEX_RESULT_HELP)


class AnalyzeThreadRequest(BaseModel):
    """Raw-text entry point: paste an email thread instead of authoring JSON."""

    thread_text: str = Field(
        ...,
        description="A raw email thread, or a plain `Speaker: message` transcript.",
    )
    use_rag: bool = Field(True, description="Toggle RAG for the ablation. Defaults to on.")
    retrieval_mode: RetrievalMode = Field(RetrievalMode.VECTOR, description=_MODE_HELP)
    retrieval_query: Optional[str] = Field(
        default=None,
        description=(
            "Override the query used for precedent retrieval. Defaults to the last few turns."
        ),
    )
    namespace: Optional[str] = Field(default=None, description=_NAMESPACE_HELP)
    index_result: bool = Field(default=False, description=_INDEX_RESULT_HELP)


class ParsedTurnPayload(BaseModel):
    """What the parser extracted — echoed back so a caller can verify it."""

    index: int
    speaker: str
    text: str


class RecommendationPayload(BaseModel):
    next_move: str
    tactic: str
    rationale: str
    grounded_case_ids: List[str] = Field(default_factory=list)


class IndexedPayload(BaseModel):
    """Result of keeping an analyzed thread as precedent. Never fatal.

    Defined above `AnalyzeResponse` rather than beside the other knowledge-base
    models because that response references it, and a forward reference here
    would leave the model unresolved until an explicit `model_rebuild()`.
    """

    doc_id: str = ""
    namespace: str = ""
    chunks: int = 0
    replaced: int = Field(
        0, description="Chunks of a previous version of this document that were displaced."
    )
    error: Optional[str] = Field(
        None,
        description="Set when indexing failed. The analysis itself still succeeded — a "
        "knowledge-base write must never take down the answer the caller asked for.",
    )


class RetrievedCasePayload(BaseModel):
    """A retrieved precedent as it goes over the wire, graph fields included.

    This model exists because FastAPI serializes through the declared
    `response_model`. The graph arms return `GraphRetrievedCase`, a *subclass*
    of `RetrievedCase` — and a field typed `List[RetrievedCase]` would validate
    those instances happily while dropping every subclass-only field on the
    way out. The provenance would vanish silently at the API boundary, which
    is the one boundary where it matters most: the reason the graph layer
    exists is a fabricated citation, and `matched_by` is what lets a reader
    check a citation against the relation that produced it.

    Structured `evidence` (anchor ids, traversal paths, per-anchor
    contributions) is deliberately **not** carried here — it is verbose and
    every in-process consumer (`evals/agent_eval.py`, the MCP tools) holds the
    real `GraphRetrievedCase` objects and can read it directly. The wire
    format carries the human-readable summary of it instead.
    """

    case_id: str
    source: str
    kind: str
    text: str
    score: float = Field(
        ...,
        description="Vector arm: 1 - cosine. Graph arms: the FUSED score, not a similarity — "
        "the raw similarity stays in `vector_score`. Not comparable across arms.",
    )
    metadata: dict = Field(default_factory=dict)

    graph_score: Optional[float] = Field(
        None, description="Summed weighted anchor contributions. Null on the vector arm."
    )
    vector_score: Optional[float] = Field(
        None, description="Raw 1 - cosine. Null when the case was found by graph traversal alone."
    )
    graph_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    fusion: Optional[str] = Field(None, description="'weighted' | 'rrf'. Null on the vector arm.")
    matched_by: List[str] = Field(
        default_factory=list,
        description="One plain-English line per piece of graph evidence — why this case was "
        "retrieved. Empty on the vector arm (cosine similarity has no such explanation).",
    )

    @classmethod
    def from_case(cls, case: Any) -> RetrievedCasePayload:
        """Build from either a `RetrievedCase` or a `GraphRetrievedCase`.

        `getattr` with defaults rather than an isinstance branch: the two arms
        differ only by which optional fields are present, and a branch would
        need updating every time the graph adds one.
        """
        return cls(
            case_id=case.case_id,
            source=case.source,
            kind=case.kind,
            text=case.text,
            score=float(case.score),
            metadata=dict(getattr(case, "metadata", {}) or {}),
            graph_score=getattr(case, "graph_score", None),
            vector_score=getattr(case, "vector_score", None),
            graph_rank=getattr(case, "graph_rank", None),
            vector_rank=getattr(case, "vector_rank", None),
            fusion=getattr(case, "fusion", None),
            matched_by=list(getattr(case, "matched_by", None) or []),
        )


class AnalyzeResponse(BaseModel):
    sentiment: List[PerTurnSentiment]
    party_stances: List[PartyStance] = Field(
        default_factory=list,
        description="Whole-thread stance per participant, least flexible first. A party the "
        "model skipped comes back with mood/flexibility 'unknown' rather than a guess.",
    )
    trajectory: Optional[Trajectory] = Field(
        None,
        description="Where the discussion is heading. `direction='unknown'` (confidence 0.0) "
        "means the stance stage returned no reading; null means the stage never ran.",
    )
    behaviors: List[BehaviorFlag]
    outcome_prob: Optional[float] = Field(
        None,
        description=(
            "Calibrated P(agreement_reached); null if the outcome model artifact is missing."
        ),
    )
    retrieved: List[RetrievedCasePayload]
    retrieval: Optional[RetrievalInfo] = Field(
        None,
        description="What the retrieve node actually did, as opposed to what was asked of it. "
        "`graph_effective=false` on a graph/hybrid request means the traversal contributed "
        "nothing — read `note` before concluding the graph underperformed, because a missing "
        "or unloaded graph degrades to vector-only without raising.",
    )
    recommendation: RecommendationPayload
    indexed: Optional[IndexedPayload] = Field(
        None,
        description="Present only when `index_result` was requested. Check `error` — a failed "
        "knowledge-base write does not fail the analysis.",
    )
    parsed: List[ParsedTurnPayload] = Field(
        default_factory=list,
        description="Turns extracted from a raw thread. Empty when a Transcript was supplied "
        "directly — populated only by /analyze/thread, so the caller can check what the "
        "parser understood before trusting the analysis built on it.",
    )


# --------------------------------------------------------------------------
# Live knowledge base
# --------------------------------------------------------------------------


class IngestDocumentRequest(BaseModel):
    """Add one document to a namespace's knowledge base."""

    namespace: str = Field(..., description="Knowledge base to write to. Created on first write.")
    title: str = Field(
        ...,
        description="Human-readable name. This is what a citation shows, so make it something "
        "a reader could go and find.",
    )
    text: str = Field(..., description="Full document text. Chunked server-side by source type.")
    source_type: SourceType = Field(
        SourceType.NOTE,
        description="Drives chunking: 'email_thread'/'analyzed_thread' split by message, "
        "'contract'/'playbook' split by numbered clause, 'note' packs paragraphs.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary tags stored on every chunk (counterparty, deal value, year, ...). "
        "Filterable at retrieval time.",
    )


class IngestDocumentResponse(BaseModel):
    doc_id: str = Field(
        ...,
        description="Content-addressed. Re-ingesting identical text under the same title "
        "returns the same id and replaces the previous version rather than duplicating it.",
    )
    namespace: str
    chunks: int
    replaced: int
    title: str
    source_type: SourceType


class CorpusDocumentSummary(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source_type: Optional[str] = None
    created_at: Optional[str] = None
    chunks: int = 0


class CorpusStatsResponse(BaseModel):
    namespace: str
    collection: str
    documents: int
    chunks: int
    source_types: int
    note: str = Field(
        default="",
        description="Set when the namespace is empty, explaining that retrieval will return "
        "nothing and why that is a state rather than a fault.",
    )


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    namespace: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: str
    sglang_ready: bool
    outcome_model_loaded: bool
    rag_configured: bool
    graph_populated: Optional[bool] = Field(
        None,
        description="Knowledge-graph tables loaded? Null unless `?probe_graph=true` was passed "
        "(the probe opens a database connection, so it is opt-in) — and also null when the "
        "probe ran but could not reach the database, which is a different problem from an "
        "empty graph.",
    )
