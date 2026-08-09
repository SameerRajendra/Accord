"""Embed the case corpus and upsert into Neon Postgres via LangChain's PGVector.

Reads normalized `CaseDocument` records from `data/processed/case_corpus.jsonl`
(built by `data.build_case_corpus`), embeds them with `all-MiniLM-L6-v2`
(DESIGN.md §4), and upserts into the `accord_cases` collection.

Idempotent: re-running this module wipes the collection and re-embeds. That's
the safer default for a demo corpus (< 10k docs) than a diff-based upsert.

Run:
    python -m rag.embed

Requires env `DATABASE_URL` (Neon connection string, e.g.
`postgresql+psycopg://user:pass@host/dbname?sslmode=require`). PGVector
requires the `+psycopg` driver prefix; a bare `postgresql://...` URL from the
Neon dashboard is normalized to that form by `_normalize_pg_url` below.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List

from data.schema import CaseDocument

logger = logging.getLogger(__name__)

COLLECTION_NAME = "accord_cases"
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


def build_embeddings():
    """Construct the HuggingFaceEmbeddings instance used across the pipeline."""
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    n = embed_case_corpus()
    print(f"Upserted {n} documents into `{COLLECTION_NAME}`.")


if __name__ == "__main__":
    main()
