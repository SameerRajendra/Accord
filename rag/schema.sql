-- Accord case corpus — pgvector DDL for Neon Postgres.
--
-- LangChain's PGVector (`langchain_postgres`) manages the collection tables
-- (`langchain_pg_collection`, `langchain_pg_embedding`) itself on first use.
-- This file exists to:
--   1. install the pgvector extension (idempotent),
--   2. create the HNSW index once LangChain has created the embedding table,
--   3. document the exact index parameters used, so Phase-5 latency numbers
--      are reproducible from this repo alone.
--
-- Apply order (see infra/neon/README.md):
--   psql "$DATABASE_URL" -f rag/schema.sql          # step 1: extension only
--   python -m rag.embed                              # step 2: LangChain creates tables + upserts vectors
--   psql "$DATABASE_URL" -f rag/schema.sql          # step 3: re-run to create the HNSW index (idempotent)

CREATE EXTENSION IF NOT EXISTS vector;

-- HNSW index on the embedding column. Parameters:
--   m               = 16  (edges per node — default; balances recall vs memory)
--   ef_construction = 64  (build-time candidate list — default; larger = higher recall, slower build)
-- Distance operator: <=> is cosine distance, matching PGVector's default
-- `distance_strategy=COSINE`. Do NOT switch to <-> (L2) without also changing
-- the PGVector construction in rag/retriever.py.
--
-- Wrapped in DO block so this file is safe to run before LangChain has
-- created langchain_pg_embedding (step 1 above) — the index creation is
-- skipped rather than erroring.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'langchain_pg_embedding'
  ) THEN
    CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_cosine
      ON langchain_pg_embedding
      USING hnsw (embedding vector_cosine_ops)
      WITH (m = 16, ef_construction = 64);
  END IF;
END $$;
