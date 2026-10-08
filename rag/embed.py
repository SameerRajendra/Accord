"""Embedding and indexing — for the frozen benchmark corpus *and* live namespaces.

Two write paths, because the two corpora have different jobs (see
`rag/documents.py` for the full split):

- **`embed_case_corpus`** — the CaSiNo benchmark. Batch, wipe-and-rebuild.
  That is the right default for a fixed corpus the evals score against: a
  diff-based upsert could leave a stale document behind and silently change a
  committed retrieval number.
- **`upsert_document` / `delete_document`** — a live namespace. Incremental,
  called at runtime, must never wipe anything. A knowledge base you can only
  write by rebuilding it from a JSONL file on disk is not a live knowledge
  base.

Both use `all-MiniLM-L6-v2` (DESIGN.md §4) so a chunk ingested at runtime is
comparable to a benchmark document embedded offline.

Run the benchmark build:
    python -m rag.embed

Requires env `DATABASE_URL` (Neon connection string, e.g.
`postgresql+psycopg://user:pass@host/dbname?sslmode=require`). PGVector
requires the `+psycopg` driver prefix; a bare `postgresql://...` URL from the
Neon dashboard is normalized to that form by `_normalize_pg_url` below.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from data.schema import CaseDocument
from rag.documents import (
    BENCHMARK_COLLECTION,
    Chunk,
    LiveDocument,
    chunk_document,
    collection_for,
)

logger = logging.getLogger(__name__)

#: Back-compat alias. `rag.documents.BENCHMARK_COLLECTION` is the name to use;
#: this one is imported by `rag/retriever.py` and older call sites.
COLLECTION_NAME = BENCHMARK_COLLECTION
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS_PATH = Path("data/processed/case_corpus.jsonl")


def _normalize_pg_url(url: str) -> str:
    """LangChain's PGVector needs the SQLAlchemy `postgresql+psycopg` prefix.

    Neon hands out URLs starting with `postgresql://` or `postgres://`; both
    are rewritten to `postgresql+psycopg://` so the same env var works whether
    it was copy-pasted from Neon or set manually. `sslmode=require` (Neon
    default) is preserved as-is.
    """
    if url.startswith("postgresql+psycopg://"):
        return url
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _load_corpus(path: Path) -> List[CaseDocument]:
    if not path.exists():
        raise FileNotFoundError(
            f"case corpus not found at {path}; run `python -m data.build_case_corpus` first"
        )
    docs: List[CaseDocument] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs.append(CaseDocument.model_validate_json(line))
    return docs


def _to_langchain_documents(cases: Iterable[CaseDocument]):
    from langchain_core.documents import Document

    for case in cases:
        # PGVector's metadata is JSON — CaseDocument.metadata is already a
        # plain dict, and case_id/source/kind get lifted in so filtering by
        # them post-hoc works without re-parsing.
        meta = {
            "case_id": case.case_id,
            "source": case.source,
            "kind": case.kind,
            **case.metadata,
        }
        yield Document(page_content=case.text, metadata=meta, id=case.case_id)


@lru_cache(maxsize=1)
def build_embeddings():
    """The `HuggingFaceEmbeddings` instance used across the pipeline.

    Cached because it loads a sentence-transformer into memory. Before live
    ingestion existed this was called once per process at startup; now it is
    on the request path (every `/corpus/documents` call embeds), and
    re-loading the model per request would put seconds of CPU in front of
    every ingest.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def embed_case_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    database_url: str = "",
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Wipe + re-embed the corpus. Returns the number of documents upserted."""
    from langchain_postgres import PGVector

    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — export the Neon connection string first "
            "(see infra/neon/README.md)"
        )

    logger.info("Loading corpus from %s", corpus_path)
    cases = _load_corpus(corpus_path)
    logger.info("Loaded %d case documents", len(cases))

    embeddings = build_embeddings()
    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=_normalize_pg_url(url),
        use_jsonb=True,
    )

    # Idempotent: drop the collection so re-runs don't accumulate duplicates.
    # `delete_collection` also removes the embedding rows for this collection.
    logger.info("Wiping existing collection %r", collection_name)
    try:
        store.delete_collection()
    except Exception as exc:  # noqa: BLE001 — first-run: collection doesn't exist
        logger.info("delete_collection skipped (%s)", exc)

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=_normalize_pg_url(url),
        use_jsonb=True,
    )

    documents = list(_to_langchain_documents(cases))
    ids = [d.id for d in documents]
    logger.info("Upserting %d documents into %r", len(documents), collection_name)
    store.add_documents(documents, ids=ids)
    logger.info("Done. Run rag/schema.sql again to (re-)create the HNSW index.")
    return len(documents)


# --------------------------------------------------------------------------
# Live namespaces — incremental, runtime, never wipes
# --------------------------------------------------------------------------
#
# The three functions below reach past LangChain into `langchain_pg_embedding`
# with raw SQL. That coupling is deliberate and is not new: `rag/schema.sql`
# already creates the HNSW index on that table by name. The alternative —
# `PGVector.delete(ids=...)` — needs the caller to already know every chunk id
# in a document, which is exactly the thing being looked up, and PGVector's
# metadata-filtered delete has moved across versions. One documented join
# against a stable table beats a moving API here.


def _resolve_url(database_url: str = "") -> str:
    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — export the Neon connection string first "
            "(see infra/neon/README.md)"
        )
    return url


@lru_cache(maxsize=8)
def _store_cached(collection_name: str, url: str):
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=build_embeddings(),
        collection_name=collection_name,
        connection=_normalize_pg_url(url),
        use_jsonb=True,
    )


def _store(collection_name: str, database_url: str = ""):
    """A PGVector handle for one collection, cached per (collection, url).

    Bounded at 8 because each entry holds a connection pool; a deployment with
    more live namespaces than that will churn the cache rather than leak
    pools. Revisit if namespaces ever outnumber that in practice.
    """
    return _store_cached(collection_name, _normalize_pg_url(_resolve_url(database_url)))


#: Deletes every embedding row belonging to one doc_id within one collection.
#: Scoped by collection so a doc_id that somehow collides across namespaces
#: cannot delete another tenant's rows.
_DELETE_DOC_SQL = """
DELETE FROM langchain_pg_embedding e
 USING langchain_pg_collection c
 WHERE e.collection_id = c.uuid
   AND c.name = %(collection)s
   AND e.cmetadata->>'doc_id' = %(doc_id)s
"""

_NAMESPACE_STATS_SQL = """
SELECT COUNT(*)                                   AS chunks,
       COUNT(DISTINCT e.cmetadata->>'doc_id')     AS documents,
       COUNT(DISTINCT e.cmetadata->>'source_type') AS source_types
  FROM langchain_pg_embedding e
  JOIN langchain_pg_collection c ON c.uuid = e.collection_id
 WHERE c.name = %(collection)s
"""

_NAMESPACE_DOCS_SQL = """
SELECT e.cmetadata->>'doc_id'      AS doc_id,
       MIN(e.cmetadata->>'title')  AS title,
       MIN(e.cmetadata->>'source_type') AS source_type,
       MIN(e.cmetadata->>'created_at')  AS created_at,
       COUNT(*)                    AS chunks
  FROM langchain_pg_embedding e
  JOIN langchain_pg_collection c ON c.uuid = e.collection_id
 WHERE c.name = %(collection)s
 GROUP BY 1
 ORDER BY created_at DESC NULLS LAST, doc_id
 LIMIT %(limit)s
"""


def _to_langchain_chunks(chunks: Iterable[Chunk]):
    from langchain_core.documents import Document

    for chunk in chunks:
        meta = dict(chunk.metadata)
        meta.update({"chunk_id": chunk.chunk_id, "heading": chunk.heading})
        # `embed_text()` prepends the heading — see rag/documents.py for why
        # the clause title carries most of the topical signal.
        yield Document(page_content=chunk.embed_text(), metadata=meta, id=chunk.chunk_id)


def delete_document(doc_id: str, namespace: str, database_url: str = "") -> int:
    """Remove every chunk of one document. Returns rows deleted.

    Safe to call for a document that was never ingested — returns 0. That
    matters because `upsert_document` calls it unconditionally.
    """
    from rag.graph_db import connect

    collection = collection_for(namespace)
    with connect(_resolve_url(database_url), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(_DELETE_DOC_SQL, {"collection": collection, "doc_id": doc_id})
            return int(cur.rowcount or 0)


def upsert_document(
    document: LiveDocument,
    database_url: str = "",
) -> Dict[str, object]:
    """Chunk, embed and index one document into its namespace.

    **Delete-then-insert, not insert-with-upsert.** Chunk ids are positional
    (`doc-abc#0`, `#1`, …), so re-ingesting an edited document that now splits
    into fewer chunks would leave the tail of the previous version behind —
    orphaned text, still retrievable, attributed to a document that no longer
    contains it. Deleting first is the only way the index reflects the
    document as it is now.

    Returns `{"doc_id", "namespace", "chunks", "replaced"}`; `replaced` is the
    chunk count of the version this call displaced, so a caller can tell an
    update from a first ingest.
    """
    chunks = chunk_document(document)
    replaced = delete_document(document.doc_id, document.namespace, database_url)

    store = _store(collection_for(document.namespace), database_url)
    documents = list(_to_langchain_chunks(chunks))
    store.add_documents(documents, ids=[d.id for d in documents])

    logger.info(
        "indexed %s (%d chunks, replaced %d) into namespace %r",
        document.doc_id,
        len(chunks),
        replaced,
        document.namespace,
    )
    return {
        "doc_id": document.doc_id,
        "namespace": document.namespace,
        "chunks": len(chunks),
        "replaced": replaced,
    }


def namespace_stats(namespace: str, database_url: str = "") -> Dict[str, object]:
    """Document and chunk counts for a namespace. Zeroes for an unused one.

    An empty namespace is a normal state, not an error: a fresh deployment has
    ingested nothing, and retrieval over it correctly returns nothing. The API
    surfaces this so "no precedents" can be explained rather than looking like
    a broken retriever.
    """
    from rag.graph_db import fetch_all

    collection = collection_for(namespace)
    rows = fetch_all(
        _NAMESPACE_STATS_SQL, {"collection": collection}, url=_resolve_url(database_url)
    )
    row = rows[0] if rows else {}
    return {
        "namespace": namespace,
        "collection": collection,
        "documents": int(row.get("documents") or 0),
        "chunks": int(row.get("chunks") or 0),
        "source_types": int(row.get("source_types") or 0),
    }


def namespace_documents(
    namespace: str, limit: int = 100, database_url: str = ""
) -> List[Dict[str, object]]:
    """List what a namespace holds — newest first. For the UI and for debugging."""
    from rag.graph_db import fetch_all

    rows = fetch_all(
        _NAMESPACE_DOCS_SQL,
        {"collection": collection_for(namespace), "limit": int(limit)},
        url=_resolve_url(database_url),
    )
    return [
        {
            "doc_id": row.get("doc_id"),
            "title": row.get("title"),
            "source_type": row.get("source_type"),
            "created_at": row.get("created_at"),
            "chunks": int(row.get("chunks") or 0),
        }
        for row in rows
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    n = embed_case_corpus()
    print(f"Upserted {n} documents into `{COLLECTION_NAME}`.")


if __name__ == "__main__":
    main()
