"""FastAPI app exposing /analyze and /health.

Runs as a Modal ASGI app colocated with SGLang in the same GPU container
(see [infra/modal/app.py](../infra/modal/app.py)). The FastAPI process talks
to SGLang over localhost, so no network hop leaves the container for
inference.

`build_app()` is called by Modal's ASGI adapter; running this file directly
also works for a local dev loop pointed at any SGLang endpoint (set
`SGLANG_BASE_URL`).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException

from agent.graph import RetrievalMode
from agent.graph import run as run_graph
from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeThreadRequest,
    CorpusDocumentSummary,
    CorpusStatsResponse,
    DeleteDocumentResponse,
    HealthResponse,
    IndexedPayload,
    IngestDocumentRequest,
    IngestDocumentResponse,
    ParsedTurnPayload,
    RecommendationPayload,
    RetrievedCasePayload,
)
from data.schema import Transcript
from rag.documents import IngestError, build_document, document_from_transcript

logger = logging.getLogger(__name__)


def _check_sglang_ready(base_url: Optional[str] = None) -> bool:
    url = base_url or os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")
    try:
        r = httpx.get(url.rstrip("/") + "/models", timeout=2.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _check_outcome_model_loaded() -> bool:
    from analysis.outcome_service import MissingModelError, _load

    try:
        _load()
        return True
    except MissingModelError:
        return False
    except Exception:  # noqa: BLE001
        return False


def _check_graph_populated() -> Optional[bool]:
    """Are the graph tables loaded? `None` if the question couldn't be answered.

    Distinguishing "no" from "couldn't tell" matters here: an unreachable
    database and an empty graph are different problems with different fixes,
    and the graph retriever degrades identically (and silently) under both.
    """
    try:
        from rag.graph_db import graph_is_populated

        return bool(graph_is_populated())
    except Exception as exc:  # noqa: BLE001 — driver, DSN, or missing tables
        logger.warning("graph probe failed (%s: %s)", type(exc).__name__, exc)
        return None


def build_app() -> FastAPI:
    app = FastAPI(
        title="Accord — Negotiation Intelligence API",
        version="0.1.0",
        description="Analyze negotiation transcripts: sentiment, per-party stance, where the "
        "discussion is heading, extreme-behavior flags, breakdown risk, precedent retrieval, "
        "and a de-escalation recommendation.\n\n"
        "Analysis is zero-shot and works on any negotiation, seen or unseen. Precedent "
        "retrieval reads a knowledge base you build: POST /corpus/documents to ingest your "
        "own contracts, threads and playbooks, then pass `namespace` on /analyze. Omitting "
        "`namespace` searches the frozen CaSiNo research corpus instead — that is the corpus "
        "the committed evals score against, and it is campsite resource-bartering, so treat "
        "those hits as a demo rather than as domain precedent.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health(probe_graph: bool = False) -> HealthResponse:
        """Liveness plus subsystem readiness.

        `?probe_graph=true` additionally queries the knowledge-graph tables.
        It is opt-in because that query opens a database connection, and the
        default health check should not wake a suspended Neon branch on every
        poll. Use it once after `python -m rag.graph_ingest` to confirm the
        graph arm has something to traverse.
        """
        return HealthResponse(
            status="ok",
            sglang_ready=_check_sglang_ready(),
            outcome_model_loaded=_check_outcome_model_loaded(),
            rag_configured=bool(os.environ.get("DATABASE_URL")),
            graph_populated=_check_graph_populated() if probe_graph else None,
        )

    def _index_analyzed(
        transcript: Transcript, namespace: Optional[str], final: dict
    ) -> IndexedPayload:
        """Keep an analyzed thread as precedent. Best-effort, never fatal.

        A failed knowledge-base write returns an `error` on the payload rather
        than a 500: the caller asked for an analysis, got one, and losing it
        because a follow-on write failed would be the wrong trade. The failure
        is still reported — silently not saving is worse than saying so.
        """
        if not namespace:
            return IndexedPayload(
                error="index_result requires `namespace` — there is no default knowledge base "
                "to write into, and writing to the frozen benchmark corpus is not allowed"
            )
        try:
            trajectory = final.get("trajectory")
            rec = final.get("recommendation")
            analysis = {
                "trajectory": trajectory.direction.value if trajectory is not None else None,
                "recommended_tactic": rec.tactic if rec is not None else None,
                "behaviors_flagged": [b.name for b in (final.get("behaviors") or []) if b.present],
            }
            from rag.embed import upsert_document

            document = document_from_transcript(
                transcript, namespace=namespace, analysis=analysis
            )
            result = upsert_document(document)
            return IndexedPayload(
                doc_id=str(result["doc_id"]),
                namespace=str(result["namespace"]),
                chunks=int(result["chunks"]),
                replaced=int(result["replaced"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("indexing the analyzed thread failed")
            return IndexedPayload(error=f"{type(exc).__name__}: {exc}")

    def _run(
        transcript: Transcript,
        use_rag: bool,
        retrieval_query: Optional[str],
        retrieval_mode: RetrievalMode = RetrievalMode.VECTOR,
        namespace: Optional[str] = None,
        index_result: bool = False,
        parsed: Optional[list] = None,
    ) -> AnalyzeResponse:
        """Shared analysis path for both entry points."""
        try:
            final = run_graph(
                transcript,
                use_rag=use_rag,
                retrieval_query=retrieval_query,
                retrieval_mode=retrieval_mode,
                namespace=namespace,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph invocation failed")
            raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

        # After the analysis, so a thread is never indexed on a run that failed
        # — and so the stored metadata reflects a completed reading.
        indexed = _index_analyzed(transcript, namespace, final) if index_result else None

        rec = final.get("recommendation")
        rec_payload = RecommendationPayload(
            next_move=rec.next_move if rec else "",
            tactic=rec.tactic if rec else "other",
            rationale=rec.rationale if rec else "recommendation missing",
            grounded_case_ids=rec.grounded_case_ids if rec else [],
        )
        return AnalyzeResponse(
            sentiment=final.get("sentiment", []),
            party_stances=final.get("party_stances", []),
            # None (not a synthesized "unknown") when the node never ran — the
            # stance stage builds its own explicit unknown when it merely failed.
            trajectory=final.get("trajectory"),
            behaviors=final.get("behaviors", []),
            outcome_prob=final.get("outcome_prob"),
            # Converted explicitly: the graph arms return a RetrievedCase
            # *subclass*, and serializing those through the base type would
            # silently drop the provenance. See RetrievedCasePayload.
            retrieved=[RetrievedCasePayload.from_case(c) for c in final.get("retrieved", [])],
            retrieval=final.get("retrieval_info"),
            recommendation=rec_payload,
            indexed=indexed,
            parsed=parsed or [],
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
        """Typed entry point — caller supplies an already-normalized Transcript."""
        return _run(
            req.transcript,
            req.use_rag,
            req.retrieval_query,
            retrieval_mode=req.retrieval_mode,
            namespace=req.namespace,
            index_result=req.index_result,
        )

    @app.post("/analyze/thread", response_model=AnalyzeResponse)
    def analyze_thread(req: AnalyzeThreadRequest) -> AnalyzeResponse:
        """Raw-text entry point — paste an email thread, the LLM structures it.

        Parse failures return 422 (the caller's input is unusable) rather than
        500, so a malformed paste is distinguishable from a broken pipeline.
        """
        from analysis.parse_thread import ThreadParseError, parse_thread

        try:
            transcript = parse_thread(req.thread_text)
        except ThreadParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("unexpected thread-parsing failure")
            raise HTTPException(status_code=500, detail=f"thread parsing failed: {exc}") from exc

        parsed = [
            ParsedTurnPayload(index=t.index, speaker=t.speaker, text=t.text)
            for t in transcript.turns
        ]
        return _run(
            transcript,
            req.use_rag,
            req.retrieval_query,
            retrieval_mode=req.retrieval_mode,
            namespace=req.namespace,
            index_result=req.index_result,
            parsed=parsed,
        )

    # ----------------------------------------------------------------------
    # Live knowledge base
    # ----------------------------------------------------------------------
    #
    # The endpoints that make this system work on data it has never seen.
    # Everything above analyzes an unseen negotiation already — sentiment,
    # stance, trajectory and behaviours are zero-shot and need no corpus. What
    # needed unseen-data support was the *grounding*: precedent retrieval was
    # welded to a fixed research corpus of campsite bartering. These let a
    # deployment build its own.

    @app.post("/corpus/documents", response_model=IngestDocumentResponse, status_code=201)
    def ingest_document(req: IngestDocumentRequest) -> IngestDocumentResponse:
        """Add a document to a namespace's knowledge base.

        Chunked by source type, embedded with the same model the benchmark
        corpus uses, and indexed immediately — no rebuild step, no file on
        disk. 400 for input that cannot become a document (empty text, bad
        namespace, oversized upload); 5xx only for a genuine backend failure.
        """
        try:
            document = build_document(
                title=req.title,
                text=req.text,
                namespace=req.namespace,
                source_type=req.source_type,
                metadata=req.metadata,
            )
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            from rag.embed import upsert_document

            result = upsert_document(document)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingestion failed")
            raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc

        return IngestDocumentResponse(
            doc_id=str(result["doc_id"]),
            namespace=str(result["namespace"]),
            chunks=int(result["chunks"]),
            replaced=int(result["replaced"]),
            title=document.title,
            source_type=document.source_type,
        )

    @app.get("/corpus/stats", response_model=CorpusStatsResponse)
    def corpus_stats(namespace: str) -> CorpusStatsResponse:
        """What a namespace holds. An empty one is a state, not an error."""
        try:
            from rag.embed import namespace_stats

            stats = namespace_stats(namespace)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("corpus stats failed")
            raise HTTPException(status_code=500, detail=f"corpus stats failed: {exc}") from exc

        note = ""
        if not stats["documents"]:
            note = (
                "This namespace is empty, so retrieval against it returns nothing and the "
                "recommendation runs ungrounded. That is the expected state before anything "
                "is ingested — POST /corpus/documents to give it precedent."
            )
        return CorpusStatsResponse(note=note, **stats)  # type: ignore[arg-type]

    @app.get("/corpus/documents", response_model=List[CorpusDocumentSummary])
    def list_corpus_documents(namespace: str, limit: int = 100) -> List[CorpusDocumentSummary]:
        """List a namespace's documents, newest first."""
        try:
            from rag.embed import namespace_documents

            rows = namespace_documents(namespace, limit=limit)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("listing corpus documents failed")
            raise HTTPException(status_code=500, detail=f"listing failed: {exc}") from exc
        return [CorpusDocumentSummary(**row) for row in rows]  # type: ignore[arg-type]

    @app.delete("/corpus/documents/{doc_id}", response_model=DeleteDocumentResponse)
    def delete_corpus_document(doc_id: str, namespace: str) -> DeleteDocumentResponse:
        """Remove every chunk of a document.

        Returns 200 with `deleted_chunks: 0` for a document that was not
        there, rather than 404 — the caller's intent ("this must not be in the
        knowledge base") is satisfied either way, and a retry after a partial
        failure should not start erroring.
        """
        try:
            from rag.embed import delete_document

            deleted = delete_document(doc_id, namespace)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("deleting corpus document failed")
            raise HTTPException(status_code=500, detail=f"delete failed: {exc}") from exc
        return DeleteDocumentResponse(
            doc_id=doc_id, namespace=namespace, deleted_chunks=deleted
        )

    return app


app = build_app()
