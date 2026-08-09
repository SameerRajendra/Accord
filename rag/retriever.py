"""Top-k precedent retrieval over Neon Postgres + pgvector.

Uses LangChain's `PGVector` (`langchain_postgres`) with the same embedding
model as `rag.embed`. Wrapped by `mcp_server/tools.py`'s `retrieve_precedent`
MCP tool and `agent/tools.py`'s LangGraph-facing tool — both call this
module's `retrieve` directly (see DESIGN.md §5 for the shared-implementation
pattern).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field

from rag.embed import COLLECTION_NAME, _normalize_pg_url, build_embeddings

logger = logging.getLogger(__name__)


class RetrievedCase(BaseModel):
    """One retrieved precedent — schema is the retrieval contract."""

    case_id: str
    source: str = Field(..., description="Originating dataset, e.g. 'craigslist_bargain'.")
    kind: str = Field(..., description="'case' or 'strategy'.")
    text: str = Field(..., description="Full embeddable text of the retrieved document.")
    score: float = Field(..., description="Similarity score — larger means more similar (1 - cosine distance).")
    metadata: dict = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _get_store():
    from langchain_postgres import PGVector

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — export the Neon connection string first "
            "(see infra/neon/README.md)"
        )
    return PGVector(
        embeddings=build_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=_normalize_pg_url(url),
        use_jsonb=True,
    )


def retrieve(query: str, k: int = 5, filter: Optional[dict] = None) -> List[RetrievedCase]:
    """Return the top-k most similar precedents to `query`.

    `filter` is a metadata predicate in LangChain PGVector's JSONB filter
    syntax — e.g. `{"kind": "case"}` or `{"source": {"$eq": "craigslist_bargain"}}`.

    The `score` returned is `1 - cosine_distance`, so larger is more similar.
    PGVector's `similarity_search_with_score` returns distance (smaller =
    closer); we invert here so the retrieval contract matches how a caller
    intuitively expects to sort ("descending score = most relevant first").
    """
    store = _get_store()
    results = store.similarity_search_with_score(query, k=k, filter=filter)
    out: List[RetrievedCase] = []
    for doc, distance in results:
        meta = dict(doc.metadata or {})
        case_id = meta.pop("case_id", doc.id or "")
        source = meta.pop("source", "")
        kind = meta.pop("kind", "")
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
