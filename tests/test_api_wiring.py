"""Regression tests for `api/main.py` — plumbing, not the graph itself.

The graph, LLM calls, and Neon RAG are all mocked. Two classes of bug are
under test, both of which are invisible at runtime:

1. **Request fields that never arrive.** `retrieval_query` was silently
   dropped on its way into the graph once already (bug 1); `retrieval_mode`
   is the same shape of risk, and a request that says "graph" while the
   vector retriever runs would quietly invalidate an ablation.
2. **Response fields that get stripped.** The graph arms return a
   `RetrievedCase` *subclass*, and FastAPI serializes through the declared
   response_model — so the provenance can disappear at the API boundary while
   every in-process test still passes.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

TRANSCRIPT_PAYLOAD = {
    "dialogue_id": "test-1",
    "source": "test",
    "domain": "unit-test",
    "parties": [
        {"party_id": "buyer", "metadata": {}},
        {"party_id": "seller", "metadata": {}},
    ],
    "turns": [
        {"index": 0, "speaker": "seller", "text": "hi"},
        {"index": 1, "speaker": "buyer", "text": "hello"},
    ],
    "outcome": {"agreement_reached": False, "final_deal": None, "points": {}},
    "has_strategy_annotations": False,
    "metadata": {},
}


def _graph_result_stub():
    """Minimal AgentState-shaped dict that satisfies AnalyzeResponse."""
    from agent.graph import Recommendation

    return {
        "sentiment": [],
        "behaviors": [],
        "outcome_prob": None,
        "retrieved": [],
        "recommendation": Recommendation(
            next_move="stub",
            tactic="other",
            rationale="stub",
            grounded_case_ids=[],
        ),
    }


def test_analyze_forwards_retrieval_query_to_graph():
    """Bug-1 regression: req.retrieval_query must reach run_graph."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())

        payload = {
            "transcript": TRANSCRIPT_PAYLOAD,
            "use_rag": True,
            "retrieval_query": "custom query from UI",
        }
        r = client.post("/analyze", json=payload)

    assert r.status_code == 200, r.text
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("retrieval_query") == "custom query from UI"
    assert kwargs.get("use_rag") is True


def test_analyze_defaults_retrieval_query_to_none():
    """When the client omits retrieval_query, run_graph gets None (not missing)."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())

        r = client.post("/analyze", json={"transcript": TRANSCRIPT_PAYLOAD})

    assert r.status_code == 200, r.text
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("retrieval_query") is None


def test_analyze_forwards_use_rag_false():
    """The RAG ablation toggle must reach the graph."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())

        r = client.post(
            "/analyze",
            json={"transcript": TRANSCRIPT_PAYLOAD, "use_rag": False},
        )

    assert r.status_code == 200, r.text
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("use_rag") is False


def test_analyze_defaults_to_the_vector_arm():
    """An old client that never heard of retrieval_mode keeps the deployed behaviour."""
    from agent.graph import RetrievalMode
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())

        r = client.post("/analyze", json={"transcript": TRANSCRIPT_PAYLOAD})

    assert r.status_code == 200, r.text
    assert mock_run.call_args.kwargs.get("retrieval_mode") is RetrievalMode.VECTOR


def test_analyze_forwards_retrieval_mode():
    """The third ablation arm has to reach the graph, or the arm isn't real."""
    from agent.graph import RetrievalMode
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())

        r = client.post(
            "/analyze",
            json={"transcript": TRANSCRIPT_PAYLOAD, "retrieval_mode": "hybrid"},
        )

    assert r.status_code == 200, r.text
    assert mock_run.call_args.kwargs.get("retrieval_mode") is RetrievalMode.HYBRID


def test_analyze_rejects_an_unknown_retrieval_mode():
    """422, not a silent fallback to vector — a mislabeled arm corrupts a benchmark."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())

        r = client.post(
            "/analyze",
            json={"transcript": TRANSCRIPT_PAYLOAD, "retrieval_mode": "neo4j"},
        )

    assert r.status_code == 422
    mock_run.assert_not_called()


def test_graph_provenance_survives_response_serialization():
    """Regression: `List[RetrievedCase]` would silently drop every subclass field.

    The graph arms return `GraphRetrievedCase`. FastAPI serializes through the
    declared response_model, so typing the field as the base class validates
    those instances happily and strips `matched_by` on the way out — deleting
    exactly the provenance the graph layer was built to provide.
    """
    from api.main import build_app
    from rag.graph_retriever import GraphEvidence, GraphRetrievedCase

    evidence = GraphEvidence(
        kind="outcome",
        anchor_id="outcome:no_agreement",
        anchor_label="no_agreement",
        contribution=0.8,
    )
    hit = GraphRetrievedCase(
        case_id="casino-casino-204",
        source="casino",
        kind="case",
        text="precedent text",
        score=0.66,
        graph_score=0.8,
        vector_score=0.43,
        fusion="weighted",
        evidence=[evidence],
        matched_by=[evidence.summary()],
    )
    stub = _graph_result_stub()
    stub["retrieved"] = [hit]

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = stub
        client = TestClient(build_app())
        r = client.post(
            "/analyze",
            json={"transcript": TRANSCRIPT_PAYLOAD, "retrieval_mode": "hybrid"},
        )

    assert r.status_code == 200, r.text
    case = r.json()["retrieved"][0]
    assert case["matched_by"] == ["ended as no_agreement"]
    assert case["graph_score"] == 0.8
    assert case["vector_score"] == 0.43
    assert case["fusion"] == "weighted"


def test_vector_arm_response_leaves_graph_fields_null():
    """A plain `RetrievedCase` must not acquire invented graph provenance."""
    from api.main import build_app
    from rag.retriever import RetrievedCase

    stub = _graph_result_stub()
    stub["retrieved"] = [
        RetrievedCase(case_id="casino-casino-1", source="casino", kind="case", text="t", score=0.42)
    ]

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = stub
        client = TestClient(build_app())
        r = client.post("/analyze", json={"transcript": TRANSCRIPT_PAYLOAD})

    assert r.status_code == 200, r.text
    case = r.json()["retrieved"][0]
    assert case["graph_score"] is None
    assert case["vector_score"] is None
    assert case["matched_by"] == []


def test_retrieval_info_reaches_the_response():
    """`graph_effective=false` is the client's only signal that the graph was absent."""
    from agent.graph import RetrievalInfo
    from api.main import build_app

    stub = _graph_result_stub()
    stub["retrieval_info"] = RetrievalInfo(
        mode="graph",
        n_retrieved=5,
        n_graph_grounded=0,
        graph_effective=False,
        note="check the tables",
    )

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = stub
        client = TestClient(build_app())
        r = client.post(
            "/analyze", json={"transcript": TRANSCRIPT_PAYLOAD, "retrieval_mode": "graph"}
        )

    assert r.status_code == 200, r.text
    info = r.json()["retrieval"]
    assert info["mode"] == "graph"
    assert info["graph_effective"] is False
    assert info["note"] == "check the tables"


def test_analyze_forwards_namespace_to_the_graph():
    """Without this the live knowledge base is unreachable from the API."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())
        r = client.post(
            "/analyze", json={"transcript": TRANSCRIPT_PAYLOAD, "namespace": "acme"}
        )

    assert r.status_code == 200, r.text
    assert mock_run.call_args.kwargs.get("namespace") == "acme"


def test_analyze_defaults_to_the_benchmark_corpus():
    """No namespace means the frozen CaSiNo corpus — evals must not move."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())
        r = client.post("/analyze", json={"transcript": TRANSCRIPT_PAYLOAD})

    assert r.status_code == 200, r.text
    assert mock_run.call_args.kwargs.get("namespace") is None


def test_index_result_without_a_namespace_is_refused_not_silently_dropped():
    """There is no default knowledge base, and the benchmark corpus is read-only."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run:
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())
        r = client.post(
            "/analyze", json={"transcript": TRANSCRIPT_PAYLOAD, "index_result": True}
        )

    assert r.status_code == 200, r.text
    indexed = r.json()["indexed"]
    assert indexed["error"] and "namespace" in indexed["error"]


def test_a_failed_index_write_does_not_fail_the_analysis():
    """The caller asked for an analysis and got one; losing it to a write is worse."""
    from api.main import build_app

    with patch("api.main.run_graph") as mock_run, patch(
        "rag.embed.upsert_document", side_effect=RuntimeError("neon unreachable")
    ):
        mock_run.return_value = _graph_result_stub()
        client = TestClient(build_app())
        r = client.post(
            "/analyze",
            json={
                "transcript": TRANSCRIPT_PAYLOAD,
                "namespace": "acme",
                "index_result": True,
            },
        )

    assert r.status_code == 200, r.text
    assert r.json()["recommendation"]["next_move"] == "stub"
    assert "neon unreachable" in r.json()["indexed"]["error"]


def test_ingest_endpoint_reports_chunks_and_replacement():
    from api.main import build_app

    with patch(
        "rag.embed.upsert_document",
        return_value={"doc_id": "doc-abc", "namespace": "acme", "chunks": 4, "replaced": 2},
    ) as upsert:
        client = TestClient(build_app())
        r = client.post(
            "/corpus/documents",
            json={
                "namespace": "acme",
                "title": "Acme MSA 2025",
                "text": "7.1 Liability. " + ("Clause text. " * 40),
                "source_type": "contract",
            },
        )

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["chunks"] == 4 and body["replaced"] == 2
    assert upsert.call_args.args[0].title == "Acme MSA 2025"


def test_unusable_ingest_input_is_a_400_not_a_500():
    """Bad input is the user's to fix; a 500 would send them to the logs instead."""
    from api.main import build_app

    client = TestClient(build_app())
    r = client.post(
        "/corpus/documents",
        json={"namespace": "not a slug", "title": "T", "text": "content"},
    )
    assert r.status_code == 400


def test_empty_namespace_stats_explain_themselves():
    """An empty knowledge base is a state, not a fault — say so where it shows."""
    from api.main import build_app

    with patch(
        "rag.embed.namespace_stats",
        return_value={
            "namespace": "acme",
            "collection": "accord_ns_acme",
            "documents": 0,
            "chunks": 0,
            "source_types": 0,
        },
    ):
        client = TestClient(build_app())
        r = client.get("/corpus/stats", params={"namespace": "acme"})

    assert r.status_code == 200, r.text
    assert "empty" in r.json()["note"].lower()


def test_deleting_a_document_that_was_never_there_is_not_an_error():
    """The caller's intent is satisfied either way; a retry should not start 404ing."""
    from api.main import build_app

    with patch("rag.embed.delete_document", return_value=0):
        client = TestClient(build_app())
        r = client.delete("/corpus/documents/doc-nope", params={"namespace": "acme"})

    assert r.status_code == 200, r.text
    assert r.json()["deleted_chunks"] == 0


def test_health_skips_the_graph_probe_by_default():
    """The default health check must not open a database connection.

    A poll that wakes a suspended Neon branch every time defeats the
    scale-to-zero design the whole deployment is built around.
    """
    from api.main import build_app

    with patch("api.main._check_sglang_ready", return_value=True), patch(
        "api.main._check_outcome_model_loaded", return_value=False
    ), patch("api.main._check_graph_populated") as probe:
        client = TestClient(build_app())
        r = client.get("/health")

    assert r.status_code == 200, r.text
    probe.assert_not_called()
    assert r.json()["graph_populated"] is None


def test_health_probes_the_graph_when_asked():
    from api.main import build_app

    with patch("api.main._check_sglang_ready", return_value=True), patch(
        "api.main._check_outcome_model_loaded", return_value=False
    ), patch("api.main._check_graph_populated", return_value=True) as probe:
        client = TestClient(build_app())
        r = client.get("/health?probe_graph=true")

    assert r.status_code == 200, r.text
    probe.assert_called_once()
    assert r.json()["graph_populated"] is True
