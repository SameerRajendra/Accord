-- Accord knowledge graph — DDL for Neon Postgres.
--
-- Companion to rag/schema.sql (which owns the pgvector side). The graph lives
-- in the SAME database as the vector store, reached through the SAME
-- DATABASE_URL — see rag/graph_db.py for why, and infra/graph/README.md for
-- the technology decision that led here.
--
-- Two tables, property-graph shaped:
--   accord_graph_nodes  — one row per vertex, typed, with JSONB properties
--   accord_graph_edges  — one row per directed, typed, weighted relation
--
-- This is a *narrow* property graph, not a general triple store: node and
-- relation types are a closed vocabulary defined in rag/graph_schema.py, and
-- ingestion is the only writer. The relational shape is what makes the
-- retriever's traversal a plain recursive CTE rather than an ORM walk.
--
-- Apply order (see infra/graph/README.md):
--   psql "$DATABASE_URL" -f rag/graph_schema.sql     # create tables + indexes
--   python -m rag.graph_ingest                       # build + load the graph
--
-- Idempotent: safe to re-run. Re-running never drops data; `graph_ingest`
-- owns the wipe-and-reload.

CREATE TABLE IF NOT EXISTS accord_graph_nodes (
    node_id    TEXT PRIMARY KEY,
    node_type  TEXT  NOT NULL,
    label      TEXT  NOT NULL,
    props      JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS accord_graph_edges (
    edge_id    BIGSERIAL PRIMARY KEY,
    src_id     TEXT NOT NULL REFERENCES accord_graph_nodes(node_id) ON DELETE CASCADE,
    dst_id     TEXT NOT NULL REFERENCES accord_graph_nodes(node_id) ON DELETE CASCADE,
    rel        TEXT NOT NULL,
    weight     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    props      JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- One edge per (source, relation, target). A party who used `vouch-fair`
    -- four times gets ONE edge with props->>'count' = '4', not four rows —
    -- which is what makes the loader's ON CONFLICT upsert idempotent and lets
    -- the retriever read usage counts without an aggregate.
    CONSTRAINT accord_graph_edges_unique UNIQUE (src_id, rel, dst_id)
);

-- Node lookups are almost always "give me every node of type X" (the retriever
-- anchors on the ~22 global issue/strategy/outcome/conflict nodes) or a
-- primary-key hit.
CREATE INDEX IF NOT EXISTS accord_graph_nodes_type_idx
    ON accord_graph_nodes (node_type);

-- JSONB containment (`props @> '{"split":"test"}'::jsonb`) is how the
-- retriever applies metadata filters. jsonb_path_ops is the smaller, faster
-- GIN variant and supports exactly the @> operator we use.
CREATE INDEX IF NOT EXISTS accord_graph_nodes_props_idx
    ON accord_graph_nodes USING gin (props jsonb_path_ops);

-- Traversal goes both directions:
--   forward  (negotiation -> party -> strategy)  uses src_id
--   backward (strategy <- party <- negotiation)  uses dst_id
-- The retriever's hot path is the *backward* walk from an anchored strategy,
-- so the dst index is not optional.
CREATE INDEX IF NOT EXISTS accord_graph_edges_src_idx
    ON accord_graph_edges (src_id, rel);
CREATE INDEX IF NOT EXISTS accord_graph_edges_dst_idx
    ON accord_graph_edges (dst_id, rel);
CREATE INDEX IF NOT EXISTS accord_graph_edges_rel_idx
    ON accord_graph_edges (rel);

-- Verification queries live in infra/graph/README.md.
