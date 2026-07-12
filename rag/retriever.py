"""Top-k precedent retrieval over pgvector.

Status: not yet implemented. Planned in Phase 2 (see SPEC.md, build order;
architecture in DESIGN.md §5).

Uses LangChain's `PGVector` (the `langchain_postgres` package) rather than a
hand-rolled psycopg client — `PGVector(embeddings=..., connection=...).as_retriever(...)`.
Wrapped by `mcp_server/tools.py`'s `retrieve_precedent` MCP tool and
`agent/tools.py`'s LangGraph-facing tool, both calling this module's
implementation directly rather than duplicating it.
"""
