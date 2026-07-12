"""LangGraph-facing tool wrappers: sentiment / retrieve_precedent / predict_outcome.

Status: not yet implemented. Planned in Phase 3 (see SPEC.md, build order;
architecture in DESIGN.md §5).

These wrap the **same** underlying `analysis.*`/`rag.*` implementations that
`mcp_server/tools.py` exposes as MCP tools — the agent calls those functions
directly (not through the MCP server as a client), so this module and
`mcp_server/tools.py` are two thin adapters over one implementation, not two
implementations. See `mcp_server/tools.py`'s docstring for the full mapping.
"""
