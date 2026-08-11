"""Postgres connection helpers for the knowledge-graph layer.

The graph lives in the **same Neon database as the vector store**, reached
through the **same `DATABASE_URL`** env var that `rag/embed.py` reads (which in
turn comes from the `accord` Modal Secret — see `infra/neon/README.md`). That
is deliberate:

* No new secret. Modal has no `secret update`; you re-create with `--force`,
  which replaces the *whole* secret — `infra/modal/app.py` documents a past
  incident where adding one key silently wiped `DATABASE_URL`. Reusing the
  existing key means the graph layer needs no secret change at all.
* No new dependency. `psycopg[binary]` is already pinned in `requirements.txt`
  for the vector path, and already installed in the Modal images.
* Graph tables and `langchain_pg_embedding` sit in one database, so a future
  single-query hybrid rerank (SQL join instead of a cross-service merge) stays
  available. The shipped retriever does not do this yet — it fuses in Python
  so it can call `rag.retriever.retrieve` unmodified — but the option is only
  open because both live in one Postgres.

`ACCORD_GRAPH_DATABASE_URL` overrides `DATABASE_URL` if the graph is ever moved
to its own instance. Nothing is hardcoded; both are read from the environment.

Driver note: LangChain's `PGVector` wants the SQLAlchemy-flavored
`postgresql+psycopg://` prefix (`rag.embed._normalize_pg_url` adds it). Raw
psycopg wants the plain `postgresql://` form, so `psycopg_dsn` strips it back
off. The same Neon connection string therefore works for both paths.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Env vars consulted, in order.
DATABASE_URL_ENV = "DATABASE_URL"
GRAPH_DATABASE_URL_ENV = "ACCORD_GRAPH_DATABASE_URL"

_SQLALCHEMY_PREFIX = "postgresql+psycopg://"
_PLAIN_PREFIX = "postgresql://"


def psycopg_dsn(url: str = "") -> str:
    """Resolve and normalize the connection string for raw psycopg.

    Precedence: explicit `url` argument > `ACCORD_GRAPH_DATABASE_URL` >
    `DATABASE_URL`. Raises rather than silently connecting to nothing.
    """
    resolved = (
        url
        or os.environ.get(GRAPH_DATABASE_URL_ENV, "")
        or os.environ.get(DATABASE_URL_ENV, "")
    )
    if not resolved:
        raise RuntimeError(
            "{} not set — export the Neon connection string first "
            "(see infra/neon/README.md; the graph layer reuses the same one, "
            "or set {} to point the graph elsewhere)".format(
                DATABASE_URL_ENV, GRAPH_DATABASE_URL_ENV
            )
        )
    if resolved.startswith(_SQLALCHEMY_PREFIX):
        return _PLAIN_PREFIX + resolved[len(_SQLALCHEMY_PREFIX):]
    return resolved


@contextmanager
def connect(url: str = "", autocommit: bool = False):
    """Yield a psycopg connection to the graph database.

    psycopg is imported lazily so that importing `rag.graph_schema` /
    `rag.graph_ingest`'s pure builders in a test never requires the driver.
    """
    import psycopg  # local import: keeps the pure build path dependency-free

    conn = psycopg.connect(psycopg_dsn(url), autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(sql: str, params: Optional[Dict[str, Any]] = None, url: str = "") -> List[Dict[str, Any]]:
    """Run a read query and return rows as dicts. One connection, one round trip."""
    from psycopg.rows import dict_row

    with connect(url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # `None`, not `{}` — psycopg only runs client-side placeholder
            # substitution when params is not None, and a parameterless query
            # should skip it entirely.
            cur.execute(sql, params if params else None)
            return [dict(row) for row in cur.fetchall()]


def execute_script(sql: str, url: str = "") -> None:
    """Run a multi-statement DDL/utility script in one transaction.

    Exists so the DDL can be applied without `psql` on PATH — a real obstacle
    on Windows, where the Neon README's `psql "$DATABASE_URL" -f ...` step
    assumes a client that is not installed by default. See
    `rag.graph_ingest.apply_schema`.
    """
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def graph_is_populated(url: str = "") -> bool:
    """Cheap liveness probe: do the graph tables exist and hold nodes?

    Diagnostic, not a gate. `graph_retrieve` degrades to vector-only by catching
    the traversal's exception rather than pre-flighting with this (one round
    trip instead of two on the happy path). Call this before trusting a
    benchmark: because the degradation is silent by design, an unloaded graph
    looks like "the graph isn't helping" rather than like an error.
    """
    from rag.graph_schema import NODE_TABLE

    try:
        rows = fetch_all(
            "SELECT COUNT(*) AS n FROM {}".format(NODE_TABLE), url=url
        )
    except Exception as exc:  # noqa: BLE001 — missing table / unreachable DB are both "not populated"
        logger.info("graph not available (%s: %s)", type(exc).__name__, exc)
        return False
    return bool(rows and rows[0].get("n"))
