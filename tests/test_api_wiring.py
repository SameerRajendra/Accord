"""Regression tests for `api/main.py` — plumbing, not the graph itself.

The graph, LLM calls, and Neon RAG are all mocked. What's under test here is
whether `AnalyzeRequest` fields actually reach `agent.graph.run` (bug 1
regression: `retrieval_query` was silently dropped before it got into the
graph).
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
