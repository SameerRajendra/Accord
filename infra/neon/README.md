# Neon setup for Accord

Accord's pgvector store lives on [Neon](https://neon.tech) (managed
Postgres, free tier includes pgvector). One connection string is all the
downstream code needs.

## One-time setup

1. **Create a project + database** on Neon. Any region works; pick the one
   closest to your Modal region to keep retrieval RTT low. The default
   database name (`neondb`) is fine — Accord uses a single collection
   (`accord_cases`), not per-project databases.
2. **Copy the pooled connection string** from the Neon dashboard. It looks
   like `postgresql://<user>:<pw>@<host>-pooler.<region>.aws.neon.tech/neondb?sslmode=require`.
   Prefer the *pooler* endpoint over the direct one — Modal containers open
   short-lived connections, and the pooler handles that pattern better.
3. **Install the pgvector extension.** Neon supports it but doesn't enable
   by default. Either run the SQL below in the Neon SQL editor, or from your
   machine:

   ```bash
   psql "$DATABASE_URL" -f rag/schema.sql
   ```

   The DDL is idempotent — running it before the embedding table exists is
   safe (the HNSW index creation is skipped and only runs on the second pass
   after `rag/embed.py` has created the table).

## Wire the URL into Modal

```bash
modal secret create accord \
  DATABASE_URL="postgresql://user:pw@host-pooler.region.aws.neon.tech/neondb?sslmode=require" \
  LANGFUSE_PUBLIC_KEY="pk-lf-..." \
  LANGFUSE_SECRET_KEY="sk-lf-..." \
  LANGFUSE_HOST="https://cloud.langfuse.com"
```

Everything downstream reads `DATABASE_URL` from the environment — no config
file to edit.

## Populate the corpus

Two paths, both use the same Neon URL:

- **From Modal (recommended, no local pgvector needed):**
  ```bash
  modal run infra/modal/app.py::build_corpus
  ```
- **From your laptop / the cluster:**
  ```bash
  export DATABASE_URL="postgresql://..."
  python -m rag.embed
  ```

Then re-run `rag/schema.sql` to create the HNSW index (skipped on first pass
before the table existed):

```bash
psql "$DATABASE_URL" -f rag/schema.sql
```

## Verify

```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM langchain_pg_embedding;"
psql "$DATABASE_URL" -c "\\d+ langchain_pg_embedding"    # should show the HNSW index
```

## Free-tier limits (Neon, at time of writing)

- 0.5 GB storage per project — one Accord corpus (~10k docs @ 384-d MiniLM =
  ~15 MB of vectors + text) fits comfortably.
- Compute autosuspends after 5 min of inactivity, wakes on connect — a
  second or two of extra latency on the first Modal request after idle. This
  stacks with SGLang cold-start (see DESIGN.md §6); acceptable for a demo.
