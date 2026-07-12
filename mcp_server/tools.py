"""FastMCP tool layer exposing Accord's negotiation-intelligence pipeline.

Status: not yet implemented. Planned alongside Phase 3 (see DESIGN.md §5,
"Agent orchestration & tool exposition: LangChain, LangGraph, MCP").

Design (decided 2026-07-12, not yet built):

- Three `@mcp.tool()`-decorated functions: `retrieve_precedent`,
  `predict_outcome`, `analyze_sentiment`. Each is a **thin wrapper** around an
  existing (or planned) implementation elsewhere — this module must not
  duplicate logic:
    - `predict_outcome` wraps `analysis.outcome_model.predict_calibrated`
      (already implemented — see analysis/outcome_model.py).
    - `analyze_sentiment` wraps `analysis.sentiment.analyze` (not yet
      implemented — Phase 1, uses LangChain's `ChatOpenAI` against
      self-hosted SGLang).
    - `retrieve_precedent` wraps `rag.retriever.retrieve` (not yet
      implemented — Phase 2, uses LangChain's `PGVector`).
- `agent/tools.py`'s LangGraph-facing tool wrappers call the **same**
  underlying `analysis.*`/`rag.*` functions directly — the agent does not
  route through this MCP server as a client. Two adapters over one
  implementation, not two implementations.
- Server built with FastMCP (mirrors the tool-exposition pattern from the
  author's other project, AEGOF v2's GPU-profiling harness).

Mirrors AEGOF v2's `mcp_server/` naming convention for consistency across
the author's projects.
"""
