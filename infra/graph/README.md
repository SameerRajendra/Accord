# Knowledge-graph setup for Accord

Accord's graph retrieval layer lives in **the same Neon Postgres database as the
pgvector store**, as two relational tables traversed with recursive CTEs. One
connection string — the one you already have — is all the downstream code needs.

This is additive. Nothing about `rag/embed.py` or `rag/retriever.py` changes,
and `graph_retrieve` degrades to the pure-vector result if the graph tables are
missing or unreachable.

---

## 0. Decision record — why Postgres and not Neo4j

**Status:** accepted. **Decision:** build the graph in the existing Neon
Postgres instance. **Alternatives weighed:** Neo4j Aura Free; Apache AGE.

### Context

The graph is small and the traversals are shallow. Roughly **3.1k nodes and
~23k edges** (1,030 negotiations, 2,060 parties, 22 shared issue / strategy /
outcome / structure nodes), and the deepest query is three hops:
`negotiation → party → strategy`. Nothing in the retrieval design needs
variable-length path finding, shortest paths, or graph algorithms.

### Options

| | Neo4j Aura Free | **Neon Postgres (chosen)** | Apache AGE |
|---|---|---|---|
| New managed service | yes | **no** | yes (self-hosted) |
| New runtime dependency | `neo4j` driver, in `requirements.txt` **and** `infra/modal/app.py` | **none — `psycopg` is already pinned** | AGE extension + self-hosted PG |
| New secret | yes (URI + user + password) | **no — reuses `DATABASE_URL`** | yes |
| Fits the data | comfortably (caps are 200k nodes / 400k rels) | comfortably | comfortably |
| Idle behavior | **pauses after ~3 days idle; a paused free instance is deleted after ~30 days** | **autosuspends, wakes automatically on connect** | n/a |
| Hybrid graph+vector rerank | two systems to merge, or embeddings duplicated into Neo4j's vector index | **both live in one database — a SQL join stays on the table** | one database |
| Query language | Cypher (better fit for deep/variable-length paths) | SQL + recursive CTEs (fine for ≤3 fixed hops) | Cypher |
| Résumé keyword "Neo4j" | **yes** | no | no |

### Decision

**Postgres.** Three things decided it, in order of weight:

1. **The idle story.** Accord's whole shape is "scale to zero, cost ~$0, share
   a link, a recruiter clicks it weeks later." Neon autosuspends and **wakes on
   connect**; an Aura Free instance **pauses after a few days of inactivity and
   must be resumed by hand from the console**, and a long-paused free instance
   is eventually deleted. A demo whose graph path silently 404s two weeks after
   you share it is worse than no graph path. This is the opposite of the
   trade-off Modal's scale-to-zero was chosen for in DESIGN.md §8.
2. **The Modal image is not mine to edit.** Adding the `neo4j` driver means
   editing `_common_pip` in `infra/modal/app.py`, which this work does not own.
   A Neo4j build would ship broken on Modal until someone else changed that
   file. Postgres adds **zero** new dependencies — `psycopg[binary]` is already
   installed in every image.
3. **One database keeps hybrid rerank cheap.** Graph tables and
   `langchain_pg_embedding` are in the same Postgres, so combining a graph
   score with vector similarity in a single query stays available as an option.
   (`rag/graph_retriever.py` does not take it yet — it fuses in Python so it can
   call `rag.retriever.retrieve` unmodified — but the option only exists because
   both live in one place.) With Neo4j, you either duplicate all 1,040
   embeddings into its vector index and accept the drift, or you always pay a
   cross-service merge.

There is also a secret-management hazard specific to this repo: Modal has no
`secret update`, so adding a key means re-creating the secret with `--force`,
which replaces the whole thing. `infra/modal/app.py` documents an incident where
exactly that silently wiped `DATABASE_URL`. Reusing `DATABASE_URL` means the
graph layer needs **no secret change at all**.

**Apache AGE was ruled out immediately:** it is not in Neon's supported
extension list, so it would mean self-hosting Postgres — strictly worse than
both options on operational cost.

### What this decision costs

**You cannot put "Neo4j" on a résumé from this build.** That is a real cost and
worth naming plainly, because "Neo4j / knowledge graph" is an explicit target.
What you *can* claim honestly is the substance the JD is actually screening for:
an entity–relation schema over a real corpus, deterministic ingestion, multi-hop
traversal with query expansion, graph+vector hybrid reranking, and
provenance-carrying retrieval. That is the transferable skill; Cypher syntax is
a week.

Two secondary costs: Cypher expresses variable-length paths (`[:REL*1..3]`) more
compactly than a recursive CTE, and Neo4j Browser gives a graph visualization
that would look good in a demo. Neither changes what the retriever can answer.

### When to revisit

Flip to Neo4j if any of these becomes true:

- A target job description names Neo4j or Cypher as a **hard** requirement. The
  port is small and mostly mechanical — see the appendix; `rag/graph_schema.py`
  is already backend-neutral (uppercase Cypher-style relation names, one
  `HAS_STRUCTURE`-per-node modelling), and only `graph_db.py` plus the two SQL
  constants in `graph_retriever.py` are Postgres-specific.
- Retrieval starts needing genuinely variable-depth paths, shortest paths,
  community detection, or PageRank-style centrality over the corpus.
- The graph grows past a few hundred thousand edges *and* traversal shows up as
  a real latency line in `results/`.

Do **not** flip on the strength of "graph databases are for graphs." At this
size and depth that is an aesthetic argument, and it costs an always-awake
managed service the demo doesn't otherwise need.

---

## 1. Graph schema

Defined declaratively in [`rag/graph_schema.py`](../../rag/graph_schema.py);
DDL in [`rag/graph_schema.sql`](../../rag/graph_schema.sql).

### Nodes

| Type | Count | Id | Key properties |
|---|---:|---|---|
| `negotiation` | 1,030 | `negotiation:casino-0` | `outcome_class`, `conflict_structure`, `contested_issues`, `point_gap`, `joint_points`, `split`, `case_id`, `text` |
| `party` | 2,060 | `party:casino-0:agent_1` | `priorities`, `points`, `role`, `satisfaction`, `svo` |
| `issue` | 3 | `issue:Firewood` | `n_contested`, `n_traded` |
| `strategy` | 10 | `strategy:uv-part` | `polarity`, `definition`, `text`, `n_annotated_dialogues` |
| `outcome` | 4 | `outcome:no_agreement` | `n_negotiations`, `share` |
| `conflict` | 5 | `conflict:high_clash` | `n_negotiations`, `share` |

Negotiation and strategy nodes carry the **verbatim document text from
`case_corpus.jsonl`**. That is a benchmarking requirement, not convenience: a
head-to-head against the vector retriever where the two return
differently-worded text for the same case would measure the rendering, not the
retrieval.

### Edges

Per-negotiation facts, read directly off one dialogue:

```
(negotiation)-[:HAS_PARTY]->(party)
(party)-[:PRIORITIZES {level}]->(issue)            # High | Medium | Low
(party)-[:ALLOCATED {quantity, own_priority}]->(issue)
(party)-[:USED_STRATEGY {count, first_turn, last_turn}]->(strategy)
(party)-[:NEGOTIATED_WITH]->(party)                # stored both ways
(negotiation)-[:CONTESTED]->(issue)                # both parties ranked it High
(negotiation)-[:TRADED]->(issue)                   # one party's High is the other's Low
(negotiation)-[:HAS_STRUCTURE]->(conflict)
(negotiation)-[:RESULTED_IN]->(outcome)
```

Corpus-level aggregates — relations *between* dialogues, which a flat document
store structurally cannot hold:

```
(strategy)-[:CO_OCCURS_WITH {count, pmi, p_given_src}]->(strategy)
(strategy)-[:PRECEDED {support, confidence, base_rate, lift}]->(outcome)
```

There is deliberately **no `INVOLVES` edge**. Every CaSiNo dialogue involves all
three issues, so such an edge would match every negotiation and carry exactly
zero retrieval signal. Only `CONTESTED` and `TRADED` are informative.

### Derived classes

Both are deterministic, thresholds chosen and documented rather than fitted:

- **`outcome_class`** — `no_agreement`; `agreement_lopsided` when the point gap
  is ≥ 6; `agreement_balanced` below that; `agreement_unscored` when points are
  missing. The raw `point_gap` is stored on the node so you can re-bucket
  without re-ingesting.
- **`conflict_structure`** — `identical_rankings` (both parties ranked all three
  issues the same way), `high_clash` (same top priority), `complementary` (each
  party's top is the other's bottom), `partial_overlap`, `unknown_structure`.
  Only the most specific class is stored; the retriever expands `high_clash` to
  also match `identical_rankings` at query time.

---

## 2. What this answers that vector search cannot

The vector baseline scored 0.42–0.44 on a hostile-conflict query and returned
five amicable balanced-deal cases, and the recommendation node then fabricated a
citation about one of them. Two mechanisms caused that, and the graph addresses
both:

- Every case document is rendered from one template
  (`data/build_case_corpus.py`), so most of each document's tokens are
  boilerplate shared by all 1,030 cases. Cosine similarity is dominated by that
  shared frame rather than by the facts that separate the cases — which is what
  a 0.02-wide score band across the whole corpus looks like.
- **The discriminating words are not in the corpus at all.** CaSiNo renders an
  adversarial negotiation in the same neutral register as a friendly one; the
  strings "hostile", "aggressive", "walked away angry" appear nowhere. No
  embedding can recover a distinction the text never makes.

Queries the graph answers and flat retrieval cannot:

| Question | How | Why vector search fails |
|---|---|---|
| "Precedents where the other side was hostile" | plan → adversarial polarity → `uv-part` + `no_agreement`/`agreement_lopsided` + `high_clash` | The word never appears in any document. |
| "Which tactics preceded breakdowns?" | `(strategy)-[:PRECEDED]->(outcome:no_agreement)` sorted by lift | An aggregate over 396 dialogues, not a similarity lookup over one. |
| "Cases with the same conflict structure as this live transcript" | `plan_from_transcript` → exact `HAS_STRUCTURE` + `CONTESTED` match | The structure is computed from priority rankings, not stated in prose. |
| "Cases where a party surrendered their own top priority" | `ALLOCATED {quantity: 0, own_priority: 'High'}` | Requires joining the deal to the priorities; the text states both but relates neither. |
| "What did the *counterpart* of a `uv-part` player do?" | `strategy ← party → NEGOTIATED_WITH → party → strategy` | Cross-party relation, not a document property. |
| "Only breakdowns from the held-out test split" | `filters={"split": "test", ...}` on JSONB containment | Metadata filtering exists in pgvector, but cannot be combined with structural traversal. |

Every hit carries `evidence`: which anchor matched, along which path, worth how
much. Given that the failure this layer answers to was a **fabricated
citation**, traceability is a requirement.

**Honest status: unproven.** No benchmark has yet compared graph retrieval
against the vector baseline on this corpus. The scoring weights in
`RetrievalWeights` are hand-set priors; the choice between weighted-sum and RRF
fusion is a guess; the planner's lexicon is hand-written and its recall against
real phrasing is unmeasured. See §7.

---

## 3. One-time setup

The graph reuses the Neon project and the `DATABASE_URL` from
[`infra/neon/README.md`](../neon/README.md). If that is already done, there is
**nothing new to provision** — no new account, no new secret, no `modal secret
create --force`.

```bash
export DATABASE_URL="postgresql://user:pw@host-pooler.region.aws.neon.tech/neondb?sslmode=require"
```

Create the tables and indexes:

```bash
psql "$DATABASE_URL" -f rag/graph_schema.sql
```

No `psql` on PATH (the normal case on Windows)? Do it from Python instead:

```bash
python -c "from rag.graph_ingest import apply_schema; print('applied', apply_schema())"
```

Either way it is idempotent — every statement is `CREATE ... IF NOT EXISTS`, and
re-running never drops data.

> **Optional:** point the graph at a *different* database by setting
> `ACCORD_GRAPH_DATABASE_URL`. It takes precedence over `DATABASE_URL`
> everywhere in the graph layer. Leave it unset for the normal one-database
> setup.

---

## 4. Build and load the graph

Prerequisites — the same two files the vector path uses:

```bash
python -m data.ingest_casino --download      # -> data/processed/casino.jsonl
python -m data.build_case_corpus             # -> data/processed/case_corpus.jsonl
```

Then, **first do a dry run** — it builds the whole graph in memory, prints the
statistics, and never opens a database connection:

```bash
python -m rag.graph_ingest --dry-run
```

Expect output shaped like this (exact counts come from your corpus):

```json
{
  "transcripts": 1030,
  "cases_matched": 1030,
  "cases_unmatched": 0,
  "annotated_dialogues": 396,
  "nodes_total": 3112,
  "edges_total": 23000,
  "nodes_by_type":  {"issue": 3, "strategy": 10, "outcome": 4, "conflict": 5, "negotiation": 1030, "party": 2060},
  "edges_by_rel":   {"HAS_PARTY": 2060, "PRIORITIZES": 6180, "...": 0},
  "outcome_distribution":   {"agreement_balanced": 0, "agreement_lopsided": 0, "no_agreement": 0},
  "structure_distribution": {"high_clash": 0, "complementary": 0, "partial_overlap": 0}
}
```

**Check `cases_unmatched` is 0** before loading. Anything else means the case
corpus and the transcripts are out of sync, and those negotiation nodes will
carry no document text — graph-only hits would then return an empty `text` to
the recommendation prompt.

Load it:

```bash
python -m rag.graph_ingest          # applies the DDL, truncates, loads, in ONE transaction
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Build and report only; never connects to a database. |
| `--dump PATH` | Also write every node and edge as JSONL, so builds can be diffed. |
| `--no-wipe` | Upsert into the existing graph instead of truncating first. |
| `--no-ddl` | Skip applying `rag/graph_schema.sql` before loading. |
| `--min-cooccurrence N` | Dialogue-count floor for a `CO_OCCURS_WITH` edge (default 5). |

The load is **all-or-nothing**: DDL, truncate, nodes and edges run in a single
transaction. A partially loaded graph would score candidates against missing
edges and hand back provenance paths that do not exist — precisely the failure
this layer exists to prevent.

### Running it from Modal instead

There is no Modal entrypoint for graph ingestion yet, because
`infra/modal/app.py` is not owned by this work. Adding one is four lines
alongside the existing `build_corpus` function:

```python
@app.function(image=corpus_image, secrets=[accord_secret], timeout=1800)
def build_graph_fn() -> int:
    from rag.graph_ingest import build_graph, load_graph, load_cases, load_transcripts
    build = build_graph(
        load_transcripts(Path("/app/data/processed/casino.jsonl")),
        load_cases(Path("/app/data/processed/case_corpus.jsonl")),
    )
    result = load_graph(build)
    print("[accord] loaded", result)
    return result["nodes"] + result["edges"]
```

Until then, run it from your machine against the same Neon URL — the graph is
~23k rows and loads in seconds.

---

## 5. Verify

### Structure

```sql
-- Nodes by type. Expect 1030 negotiation, 2060 party, 3 issue, 10 strategy,
-- 4 outcome, 5 conflict.
SELECT node_type, COUNT(*) FROM accord_graph_nodes GROUP BY node_type ORDER BY 2 DESC;

-- Edges by relation.
SELECT rel, COUNT(*) FROM accord_graph_edges GROUP BY rel ORDER BY 2 DESC;

-- Dangling endpoints. MUST be 0 (the FK enforces it, so a non-zero result
-- means you are looking at the wrong database).
SELECT COUNT(*) AS dangling
  FROM accord_graph_edges e
  LEFT JOIN accord_graph_nodes n ON n.node_id = e.src_id
 WHERE n.node_id IS NULL;

-- Indexes present.
\d+ accord_graph_edges
```

### Sanity against known corpus facts

```sql
-- Outcome mix. Should mirror CaSiNo's ~97.6% agreement rate: no_agreement is a
-- small minority. If it is not, the ingestion read the wrong field.
SELECT label,
       (props->>'n_negotiations')::int AS n,
       props->>'share'                 AS share
  FROM accord_graph_nodes
 WHERE node_type = 'outcome'
 ORDER BY n DESC;

-- Strategy support. Should sum over ~396 annotated dialogues, not 1030.
SELECT label, (props->>'n_annotated_dialogues')::int AS n
  FROM accord_graph_nodes
 WHERE node_type = 'strategy'
 ORDER BY n DESC;
```

### The queries vector search cannot do

```sql
-- Which tactics preceded breakdowns? Read `support` before believing `lift` —
-- the no_agreement denominator is a couple of dozen dialogues.
SELECT src_id AS strategy,
       props->>'support'   AS support,
       props->>'lift'      AS lift,
       props->>'base_rate' AS base_rate
  FROM accord_graph_edges
 WHERE rel = 'PRECEDED' AND dst_id = 'outcome:no_agreement'
 ORDER BY weight DESC;

-- Two-hop traversal: negotiations where a party played `uv-part` AND no deal
-- was reached. This is the answer set the hostile query should have returned.
SELECT DISTINCT hp.src_id AS negotiation
  FROM accord_graph_edges us
  JOIN accord_graph_edges hp ON hp.dst_id = us.src_id AND hp.rel = 'HAS_PARTY'
  JOIN accord_graph_edges ro ON ro.src_id = hp.src_id AND ro.rel = 'RESULTED_IN'
 WHERE us.rel = 'USED_STRATEGY' AND us.dst_id = 'strategy:uv-part'
   AND ro.dst_id = 'outcome:no_agreement';

-- Parties who surrendered the issue they ranked highest.
SELECT src_id AS party, dst_id AS issue
  FROM accord_graph_edges
 WHERE rel = 'ALLOCATED'
   AND props->>'own_priority' = 'High'
   AND (props->>'quantity')::int = 0
 LIMIT 20;
```

### From Python

```bash
# Is the graph there at all?
python -c "from rag.graph_db import graph_is_populated; print(graph_is_populated())"

# What does the planner think a query is asking for? (no database needed)
python -c "from rag.graph_retriever import plan_query; print(plan_query('hostile fight over firewood').describe())"

# End-to-end, with provenance. (Heredoc — bash / Git Bash. On PowerShell, paste
# the body into `python` interactively instead.)
python - <<'PY'
from rag.graph_retriever import graph_retrieve
for hit in graph_retrieve("precedents where the other side was hostile", k=5):
    print(hit.case_id, round(hit.score, 3), "| graph", round(hit.graph_score, 2),
          "| vector", hit.vector_score)
    for line in hit.matched_by:
        print("   -", line)
PY
```

Expect the hostile query to return cases where a party actually played
`uv-part`, or that ended `no_agreement` / `agreement_lopsided`, or that had both
campers fighting over the same resource — with a `matched_by` line naming which
of those it was. If it instead returns amicable balanced deals with no evidence
lines, the graph is not loaded; check `graph_is_populated()` above.

---

## 6. Gotchas

- **`case_id` in the corpus is doubled.** `data/build_case_corpus.py` composes
  `case_id` as `f"{source}-{dialogue_id}"` while `dialogue_id` already carries
  the source, so real ids look like `casino-casino-0`. Ingestion therefore joins
  transcripts to case documents on `metadata['dialogue_id']`, not on `case_id`.
  Do not "fix" the ids in `data/` without re-running the graph ingest — the
  vector store's ids would change too.
- **Strategy annotations cover 396 of 1,030 dialogues.** Everything downstream
  of `USED_STRATEGY` — `CO_OCCURS_WITH`, `PRECEDED`, strategy-anchored
  retrieval — sees only that subset. A strategy-only query can therefore never
  reach 60% of the corpus. Issue and structure anchors cover all of it, which is
  why the planner emits several anchor families rather than one.
- **`PRECEDED` is correlational and is not used for ranking by default.** Both
  its numerator and its base rate are computed over the annotated subset (mixing
  a 396-dialogue numerator with a 1,030-dialogue base rate would inflate every
  lift), and the weight is clamped at 5.0 so a lift computed off three dialogues
  cannot dominate a score. Read `support` first.
- **A live transcript has no strategy annotations.** `plan_from_transcript`
  says so in `notes` and anchors on structure and issues instead. This is the
  expected path in production, not an error.
- **Neon autosuspend stacks with SGLang cold start.** First request after idle
  pays both (DESIGN.md §6). The graph adds no new suspend penalty — it is the
  same database connection the vector path already wakes.
- **Use the pooled Neon endpoint.** Same reasoning as
  `infra/neon/README.md`: Modal containers open short-lived connections.
- **`graph_retrieve` never raises on a missing graph.** If the tables do not
  exist it logs a warning and returns the pure-vector result. That is deliberate
  (DESIGN.md §6), but it means a silent misconfiguration looks like "the graph
  isn't helping" rather than an error. Check `graph_is_populated()` before
  concluding anything from a benchmark.
- **Re-ingest after any change to `data/build_case_corpus.py`.** The document
  text is denormalized onto the nodes; stale text would make graph and vector
  hits disagree about the same case.

---

## 7. Benchmarking it honestly

`graph_retrieve(query, k)` is signature-compatible with
`rag.retriever.retrieve(query, k)` specifically so the two can be swapped in one
harness. `evals/retrieval_eval.py` (recall@k / MRR on held-out cases, currently
a stub) is where the comparison belongs.

The arms worth running, all reachable by parameter and none needing a code
change:

| Arm | Call |
|---|---|
| Vector baseline | `retrieve(q, k)` |
| Graph only | `graph_retrieve(q, k, use_vector=False)` |
| Hybrid, weighted | `graph_retrieve(q, k, fusion="weighted", alpha=0.6)` |
| Hybrid, RRF | `graph_retrieve(q, k, fusion="rrf")` |
| Alpha sweep | `alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` |
| Expansion off | `max_hops=0` |
| Transcript-anchored | `graph_retrieve_for_transcript(t, k=5)` |

Two things to hold onto when reading the results:

1. `alpha=0.0` is *not* identical to the vector baseline — it still restricts
   the candidate pool to the union of both lists. Compare against `retrieve`
   directly, not against `alpha=0.0`.
2. Under `fusion="weighted"`, a graph-only candidate has no vector score and is
   scored 0.0 on that axis, which systematically understates it. `fusion="rrf"`
   does not have that bias. If weighted wins, check it is not winning because of
   this.

Until those numbers exist, "graph retrieval improves precedent quality" is a
hypothesis this code implements — not a result it demonstrates. Say it that way.

---

## 8. Integration point for the agent (documented, not wired)

`agent/graph.py` and `agent/tools.py` are owned elsewhere, so nothing here is
wired into the LangGraph agent. The hook is deliberately small.

`GraphRetrievedCase` **subclasses** `rag.retriever.RetrievedCase`, so
`AgentState.retrieved: List[RetrievedCase]` needs no change. Adding a tool in
`agent/tools.py`:

```python
from rag.graph_retriever import GraphRetrievedCase, graph_retrieve_for_transcript

def retrieve_precedent_graph_tool(
    transcript: Transcript, query: str, k: int = 5
) -> List[GraphRetrievedCase]:
    """Graph-anchored precedent. Falls back to vector-only if the graph is absent."""
    return graph_retrieve_for_transcript(transcript, query=query, k=k)
```

and in `_node_retrieve`:

```python
query = state.get("retrieval_query") or _default_query(state["transcript"])
if state.get("use_graph", False):                       # third ablation arm
    return {"retrieved": retrieve_precedent_graph_tool(state["transcript"], query, k=5)}
return {"retrieved": retrieve_precedent_tool(query, k=5)}
```

Two notes for whoever wires it:

- **Prefer `graph_retrieve_for_transcript` over `graph_retrieve`.** The agent
  holds the actual `Transcript`, so the priority rankings, contested issue and
  conflict structure are *known facts* rather than words to be guessed from the
  last four turns. That is the strongest anchor set available anywhere in the
  pipeline.
- **Surface the provenance in the recommendation prompt.** Each hit carries
  `matched_by` — one plain-English line per piece of evidence. Appending those
  under each precedent in `_format_analysis` gives the model the *relation* that
  justified the case, not just its text, which is the direct countermeasure to
  the citation-fabrication that motivated this layer. It also makes a fabricated
  citation checkable after the fact: if the model cites a case, the evidence
  list says exactly what relation it actually stands in.

`use_graph` as a third arm of the existing RAG-vs-no-RAG ablation (no-RAG /
vector-RAG / graph-RAG) costs one boolean and makes the comparison free.

---

## Appendix — porting to Neo4j

Kept current so §0's "when to revisit" is a real option rather than a promise.
`rag/graph_schema.py` is already backend-neutral: uppercase Cypher-style
relation names, one node label per type, properties as flat maps. Only
`rag/graph_db.py` and the two SQL constants in `rag/graph_retriever.py` are
Postgres-specific.

### Constraints and indexes

```cypher
CREATE CONSTRAINT negotiation_id IF NOT EXISTS
  FOR (n:Negotiation) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT party_id IF NOT EXISTS
  FOR (p:Party) REQUIRE p.node_id IS UNIQUE;
CREATE CONSTRAINT strategy_id IF NOT EXISTS
  FOR (s:Strategy) REQUIRE s.node_id IS UNIQUE;
CREATE CONSTRAINT issue_id IF NOT EXISTS
  FOR (i:Issue) REQUIRE i.node_id IS UNIQUE;
CREATE CONSTRAINT outcome_id IF NOT EXISTS
  FOR (o:Outcome) REQUIRE o.node_id IS UNIQUE;
CREATE CONSTRAINT conflict_id IF NOT EXISTS
  FOR (c:Conflict) REQUIRE c.node_id IS UNIQUE;

CREATE INDEX negotiation_outcome IF NOT EXISTS
  FOR (n:Negotiation) ON (n.outcome_class);
CREATE INDEX negotiation_structure IF NOT EXISTS
  FOR (n:Negotiation) ON (n.conflict_structure);
```

### Loading

`build_graph()` already returns backend-neutral `GraphNode` / `GraphEdge`
objects; only `load_graph()` would be replaced:

```cypher
UNWIND $nodes AS row
CALL apoc.merge.node([row.node_type], {node_id: row.node_id}, row.props) YIELD node
RETURN count(node);

UNWIND $edges AS row
MATCH (a {node_id: row.src_id}), (b {node_id: row.dst_id})
CALL apoc.merge.relationship(a, row.rel, {}, row.props, b, {}) YIELD rel
RETURN count(rel);
```

(Without APOC, generate one statement per node label and relation type — the
vocabulary is closed and small, so that is 6 + 11 statements.)

### The traversal

The whole of `_TRAVERSAL_SQL` becomes roughly:

```cypher
MATCH (seed:Strategy) WHERE seed.node_id IN $strategy_ids
MATCH path = (seed)-[:CO_OCCURS_WITH*0..1]->(s:Strategy)
WITH s, max(reduce(w = 1.0, r IN relationships(path) | w * r.weight * $hop_decay)) AS sw
MATCH (n:Negotiation)-[:HAS_PARTY]->(p:Party)-[u:USED_STRATEGY]->(s)
WITH n, sum(sw * $w_strategy) AS strategy_score
OPTIONAL MATCH (n)-[:CONTESTED]->(i:Issue) WHERE i.node_id IN $issue_ids
OPTIONAL MATCH (n)-[:RESULTED_IN]->(o:Outcome) WHERE o.node_id IN $outcome_ids
OPTIONAL MATCH (n)-[:HAS_STRUCTURE]->(c:Conflict) WHERE c.node_id IN $structure_ids
RETURN n.node_id AS node_id,
       strategy_score
         + count(i) * $w_issue
         + count(o) * $w_outcome
         + count(c) * $w_structure AS graph_score
ORDER BY graph_score DESC
LIMIT $candidate_k;
```

Shorter than the SQL, which is the honest point in Cypher's favor — and also the
whole of it at this depth. Everything else in `rag/graph_retriever.py` (the
planner, the fusion, the provenance model) is backend-independent and would not
change.

### Additional operational steps a Neo4j port would need

1. `modal secret create accord --force NEO4J_URI=... NEO4J_USERNAME=...
   NEO4J_PASSWORD=... DATABASE_URL=... LANGFUSE_*=...` — note `--force` replaces
   the **entire** secret, so every existing key must be repeated in the same
   command or it is wiped.
2. Add `neo4j>=5` to `requirements.txt` **and** to `_common_pip` in
   `infra/modal/app.py`.
3. A keep-alive for the free instance's idle-pause policy, or accept that the
   graph path goes dark between demos.
