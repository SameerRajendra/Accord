"""Top-k precedent retrieval over Neon Postgres + pgvector.

Uses LangChain's `PGVector` (`langchain_postgres`) with the same embedding
model as `rag.embed`. Wrapped by `mcp_server/tools.py`'s `retrieve_precedent`
MCP tool and `agent/tools.py`'s LangGraph-facing tool — both call this
module's `retrieve` directly (see DESIGN.md §5 for the shared-implementation
pattern).

Two collections, one retriever
------------------------------
`retrieve(query, k)` with no namespace reads the **frozen CaSiNo benchmark**
(`accord_cases`) — that is what `evals/retrieval_eval.py` scores, and it must
not move when a user ingests a document. Passing `namespace=` reads that
namespace's **live** collection instead: user-ingested contracts, threads,
playbooks, and analyzed threads kept as precedent (`rag/documents.py`).

They are never merged in one query. A blended result would mix documents with
gold labels and documents without, and no downstream number could say which
kind it measured.

Retrieving from an empty namespace returns `[]` rather than raising. A fresh
deployment has ingested nothing, and "no institutional knowledge yet" is a
real state the recommendation node already handles — it is the same path as
the no-RAG ablation arm.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field

from rag.documents import BENCHMARK_COLLECTION, collection_for
from rag.embed import _normalize_pg_url, build_embeddings

logger = logging.getLogger(__name__)


class RetrievedCase(BaseModel):
    """One retrieved precedent — schema is the retrieval contract."""

    case_id: str
    source: str = Field(
        ..., description="Originating dataset or document type, e.g. 'casino', 'contract'."
    )
    kind: str = Field(
        ..., description="'case' or 'strategy' (benchmark) / 'chunk' (live namespace)."
    )
    text: str = Field(..., description="Full embeddable text of the retrieved document.")
    score: float = Field(
        ..., description="Similarity score — larger means more similar (1 - cosine distance)."
    )
    metadata: dict = Field(default_factory=dict)

    def citation_label(self) -> str:
        """How this hit should be named when cited.

        A benchmark case is identified by its id; a live chunk is identified
        by its document title and clause heading, because "doc-9f3a1c#4" is
        not a thing a negotiator can look up, and an unverifiable citation is
        how the fabrication problem started.
        """
        title = self.metadata.get("title")
        heading = self.metadata.get("heading")
        if title and heading:
            return f'{title} — {heading}'
        if title:
            return str(title)
        return self.case_id


@lru_cache(maxsize=8)
def _get_store(collection_name: str):
    from langchain_postgres import PGVector

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — export the Neon connection string first "
            "(see infra/neon/README.md)"
        )
    return PGVector(
        embeddings=build_embeddings(),
        collection_name=collection_name,
        connection=_normalize_pg_url(url),
        use_jsonb=True,
    )


def retrieve(
    query: str,
    k: int = 5,
    filter: Optional[dict] = None,
    namespace: Optional[str] = None,
) -> List[RetrievedCase]:
    """Return the top-k most similar precedents to `query`.

    `namespace=None` reads the frozen CaSiNo benchmark collection; a namespace
    string reads that live collection instead. `filter` is a metadata
    predicate in LangChain PGVector's JSONB filter syntax — e.g.
    `{"kind": "case"}` or `{"source_type": {"$eq": "contract"}}`.

    The `score` returned is `1 - cosine_distance`, so larger is more similar.
    PGVector's `similarity_search_with_score` returns distance (smaller =
    closer); we invert here so the retrieval contract matches how a caller
    intuitively expects to sort ("descending score = most relevant first").

    A namespace with nothing in it returns `[]`. Any other retrieval failure
    still raises — an unreachable database is a fault, an empty knowledge base
    is not, and collapsing the two would hide a broken deployment behind a
    plausible-looking empty result.
    """
    collection = BENCHMARK_COLLECTION if namespace is None else collection_for(namespace)
    store = _get_store(collection)
    try:
        results = store.similarity_search_with_score(query, k=k, filter=filter)
    except Exception as exc:  # noqa: BLE001
        if namespace is not None and _looks_like_missing_collection(exc):
            logger.info(
                "namespace %r has no collection yet — returning no precedents", namespace
            )
            return []
        raise

    out: List[RetrievedCase] = []
    for doc, distance in results:
        meta = dict(doc.metadata or {})
        # Live chunks key on `chunk_id`; benchmark documents key on `case_id`.
        case_id = meta.pop("chunk_id", None) or meta.pop("case_id", None) or (doc.id or "")
        meta.pop("case_id", None)
        source = meta.pop("source", None) or meta.get("source_type") or ""
        kind = meta.pop("kind", None) or ("chunk" if namespace is not None else "")
        out.append(
            RetrievedCase(
                case_id=case_id,
                source=source,
                kind=kind,
                text=doc.page_content,
                score=float(1.0 - distance),
                metadata=meta,
            )
        )
    return out


def _looks_like_missing_collection(exc: Exception) -> bool:
    """Is this 'nothing has been ingested here yet' rather than a real fault?

    Matched on the message because `langchain_postgres` raises plain
    `ValueError`/`sqlalchemy` errors for a missing collection and gives no
    typed signal. Deliberately narrow: anything unrecognized re-raises, so a
    connection failure is never mistaken for an empty namespace.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return "collection" in text and ("not found" in text or "does not exist" in text)
