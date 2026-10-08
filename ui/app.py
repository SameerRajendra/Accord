"""Streamlit UI — paste an email thread, see Accord's analysis.

Deployed as the `ui` web server on the `accord` Modal app (CPU image, separate
from the GPU container); calls the API over HTTPS. `ACCORD_API_URL` points at
the API.

Two input modes:

- **Email thread** (default) — paste raw text; `/analyze/thread` has the LLM
  structure it. This is the demo path: nobody should have to hand-author JSON
  to try a product.
- **Transcript JSON** (advanced) — the typed `/analyze` contract, for a
  pre-normalized `Transcript`. Kept because it's the reproducible path for
  evals and the RAG ablation.

The sidebar exposes the retrieval ablation directly: RAG on/off, and which
retriever answers when it's on (vector / graph / hybrid). The graph arm needs
the parties' priority rankings to anchor well, so it has more to work with on
the Transcript-JSON tab than on a parsed email thread — the result panel says
which of the two happened rather than leaving it to be inferred.

Kept intentionally minimal — one page, no auth. This is the "shareable in an
afternoon" UI DESIGN.md §8 committed to, not a product.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
import streamlit as st

st.set_page_config(page_title="Accord — Negotiation Intelligence", layout="wide")

API_URL = os.environ.get("ACCORD_API_URL", "").rstrip("/")

# Pre-seeded demo knowledge base (see data/demo_seed.py). Defaulting the demo to
# it means a recruiter's first click grounds in relevant business-negotiation
# precedent, not campsite bartering. Set ACCORD_DEMO_NAMESPACE="" to default to
# the benchmark corpus instead.
DEMO_NAMESPACE = os.environ.get("ACCORD_DEMO_NAMESPACE", "demo")


def _prewarm() -> None:
    """Kick off the GPU cold start once per session, while the page is being read.

    The heavy latency a recruiter feels is the ~90 s SGLang cold start, paid on
    the first request after idle. Firing a health check on page load routes a
    request to the GPU container, which starts booting immediately — so by the
    time they finish reading and click Analyze, it is warm or warming. The short
    timeout means we never block the UI on it; even a timeout still triggers the
    boot on Modal's side.
    """
    if not API_URL or st.session_state.get("_prewarmed"):
        return
    st.session_state["_prewarmed"] = True
    try:
        httpx.get(f"{API_URL}/health", timeout=3.0)
    except Exception:  # noqa: BLE001 — fire-and-forget; the point is to trigger the boot
        pass

# A realistic contract-renewal thread: newest message on top (as most mail
# clients stack them), quoted history, signatures — so the parser has to
# reverse the order and de-duplicate quoted blocks to get this right.
DEFAULT_THREAD = """From: Daniel Okafor <d.okafor@cormorant-legal.com>
Sent: Thursday, 7 August 2026 18:22
To: Priya Raman <p.raman@northwind.io>
Subject: RE: Master Services Agreement - renewal terms

Priya,

I've gone as far as I intend to. The uplift stands at 38% and the auto-renewal
clause is not negotiable. Either sign by Friday or we let the agreement lapse
and you can find another provider on thirty days' notice.

Daniel Okafor
Cormorant Legal

> From: Priya Raman
> Sent: Thursday, 7 August 2026 14:05
>
> Daniel, this is starting to feel less like a negotiation and more like a
> hostage situation. You've moved 2% in three weeks while refusing to explain
> the underlying cost basis. Frankly it's hard to take the "partnership"
> language in your last email seriously.
>
> Priya

> From: Daniel Okafor
> Sent: Wednesday, 6 August 2026 11:30
>
> The 40% figure reflects market rates. I can come down to 38% but I'm not
> going to itemise our internal costs, and I'd remind you that you're already
> operating past the original term.

> From: Priya Raman
> Sent: Tuesday, 5 August 2026 09:12
>
> Thanks for the redline. We're aligned on most of it, but a 40% uplift is
> well outside what we budgeted for this cycle. Our usage is flat year on
> year - can you walk me through how you arrived at that number? Happy to
> look at a longer term if it helps the economics.

> From: Daniel Okafor
> Sent: Monday, 4 August 2026 16:40
>
> Hi Priya - attached is our proposed renewal for the MSA. Headline change is
> a 40% uplift on the annual licence plus a twelve month auto-renewal. Let me
> know if you'd like to discuss.
"""

DEFAULT_TRANSCRIPT: dict[str, Any] = {
    "dialogue_id": "demo-1",
    "source": "manual",
    "domain": "campsite_resources",
    "parties": [
        {
            "party_id": "agent_1",
            "priorities": {"Firewood": "High", "Food": "Medium", "Water": "Low"},
            "metadata": {},
        },
        {
            "party_id": "agent_2",
            "priorities": {"Firewood": "High", "Water": "Medium", "Food": "Low"},
            "metadata": {},
        },
    ],
    "turns": [
        {
            "index": 0,
            "speaker": "agent_1",
            "text": (
                "Hi! I'm hoping to grab extra firewood - our group has a lot of seniors "
                "who need to stay warm."
            ),
        },
        {
            "index": 1,
            "speaker": "agent_2",
            "text": (
                "I need firewood too, my dog has fleas and the fire keeps them off. "
                "I can't give it up."
            ),
        },
        {
            "index": 2,
            "speaker": "agent_1",
            "text": (
                "There's no way you need it more than a group of elderly people. "
                "That's a bit ridiculous."
            ),
        },
        {
            "index": 3,
            "speaker": "agent_2",
            "text": "Take it or leave it - I'm keeping all three firewood or there's no deal.",
        },
    ],
    "outcome": {"agreement_reached": False, "final_deal": None, "points": {}},
    "has_strategy_annotations": False,
    "metadata": {"split": "demo"},
}


# Mirrors `agent.graph.RetrievalMode`. Spelled out here rather than imported
# because the UI container ships without the analysis package — it is an HTTP
# client, not a second copy of the pipeline.
_RETRIEVAL_MODES = {
    "vector": "Vector (pgvector baseline)",
    "graph": "Knowledge graph only",
    "hybrid": "Hybrid (graph + vector)",
}


st.title("Accord")
st.caption(
    "Negotiation intelligence: where it's heading · who stands where · sentiment · "
    "behaviors · risk · precedent · recommendation."
)

# Start the GPU warming while the page is read, so the first Analyze click is fast.
_prewarm()

if not API_URL:
    st.warning(
        "`ACCORD_API_URL` is not set on this UI container. Point it at the deployed API, e.g. "
        "`https://<workspace>--accord-accordserver-api.modal.run`."
    )

with st.sidebar:
    st.subheader("Knowledge base")
    namespace = st.text_input(
        "Namespace",
        value=DEMO_NAMESPACE,
        placeholder="e.g. acme-legal",
        help="Your own precedent — the documents you ingest on the 'Knowledge base' tab. "
        "Pre-set to the built-in **demo** knowledge base (business-negotiation precedents). "
        "Clear it to search the CaSiNo research corpus instead (campsite bartering — there "
        "to demo the mechanics, not to advise on a business deal).",
    ).strip()
    if namespace == DEMO_NAMESPACE and DEMO_NAMESPACE:
        st.caption(
            "Grounding in the **demo knowledge base** — business-negotiation precedents and "
            "playbook rules. Every cited precedent shows *why* it was retrieved."
        )
    elif namespace:
        st.caption(f"Retrieving from **{namespace}**.")
    else:
        st.caption(
            "Retrieving from the **CaSiNo research corpus** (campsite bartering). "
            "Set a namespace to ground answers in your own documents."
        )

    st.markdown("---")
    st.subheader("Options")
    use_rag = st.toggle(
        "Enable RAG (retrieval)",
        value=True,
        help="Off = ablation baseline. The recommendation runs without precedent grounding.",
    )
    # Two knobs, not three radio options: "no retrieval" is the use_rag axis,
    # so there is no incoherent "RAG off, graph on" state to guard against.
    retrieval_mode = st.radio(
        "Retriever",
        options=list(_RETRIEVAL_MODES),
        format_func=lambda m: _RETRIEVAL_MODES[m],
        index=0,
        disabled=not use_rag,
        help=(
            "vector — pgvector cosine over the case corpus (the deployed baseline).\n\n"
            "graph — knowledge-graph traversal only: anchors on the parties' priority "
            "rankings, contested issues and outcome class, which the case text never states "
            "in words.\n\n"
            "hybrid — both, fused. Which of the three is actually best on this corpus is "
            "not yet measured."
        ),
    )
    retrieval_query = st.text_area(
        "Custom retrieval query (optional)",
        value="",
        help="If empty, the last few turns are used as the query.",
    )
    index_result = st.toggle(
        "Keep this negotiation as precedent",
        value=False,
        disabled=not namespace,
        help="Index the analyzed thread into your namespace so future analyses can retrieve "
        "it. Off by default and requires a namespace — retaining someone's negotiation is "
        "not a decision to make silently.",
    )
    st.markdown("---")
    st.caption(
        "Cold-start note: the first request after idle pays ~90 s while the GPU wakes and "
        "loads the model. Subsequent requests take seconds."
    )


def _request(method: str, path: str, **kwargs):
    """Call the API, surfacing errors as readable Streamlit messages.

    4xx is treated as "your input needs fixing" and shown as a warning with
    the server's own explanation; 5xx is an error. The distinction matters
    because most 4xx here are things a user can act on (empty namespace, bad
    document) and rendering those as red failures teaches people the app is
    broken when it is telling them something useful.
    """
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=300.0, **kwargs)
    except httpx.RequestError as exc:
        st.error(f"Couldn't reach the API at {API_URL}: {exc}")
        return None
    if 400 <= r.status_code < 500:
        try:
            st.warning(r.json().get("detail", r.text))
        except Exception:  # noqa: BLE001
            st.warning(r.text[:500])
        return None
    if r.status_code >= 500:
        st.error(f"API returned {r.status_code}: {r.text[:500]}")
        return None
    return r.json()


def _post(path: str, payload: dict):
    return _request("POST", path, json=payload)


# Direction values come from `analysis.stance.Direction`. Spelling them out here
# (rather than title-casing the enum) is what makes the headline readable to
# someone who has never seen the taxonomy.
_DIRECTION_HEADLINE = {
    "converging": "Converging — the parties are moving toward each other",
    "holding": "Holding — positions are steady; neither side is moving",
    "stalling": "Stalling — repetition without progress",
    "escalating": "Escalating — tension is rising turn over turn",
    "breaking_down": "Breaking down — heading toward no deal",
}

# Streamlit's status boxes double as the severity signal, so the headline reads
# at a glance without a custom colour system.
_DIRECTION_BOX = {
    "converging": st.success,
    "holding": st.info,
    "stalling": st.warning,
    "escalating": st.warning,
    "breaking_down": st.error,
}


def _render_trajectory(trajectory: Optional[dict]) -> None:
    """The headline answer: where is this discussion going?"""
    st.subheader("Where this is heading")
    direction = (trajectory or {}).get("direction")
    if not trajectory or direction in (None, "unknown"):
        st.info(
            "No reading returned for this thread — treat the sections below as the "
            "only evidence, not as a calm verdict."
        )
        reasoning = (trajectory or {}).get("reasoning")
        if reasoning:
            st.caption(reasoning)
        return

    _DIRECTION_BOX.get(direction, st.info)(
        f"**{_DIRECTION_HEADLINE.get(direction, direction)}**"
    )
    if trajectory.get("reasoning"):
        st.markdown(trajectory["reasoning"])

    notes = [
        f"Confidence {trajectory.get('confidence', 0.0):.2f} — model self-reported, "
        "uncalibrated (no trajectory labels exist to score it against)."
    ]
    turned = trajectory.get("turning_point_turn")
    if turned is not None:
        notes.append(f"Tone turned at turn {turned}.")
    st.caption(" ".join(notes))


def _render_party_stances(stances: list) -> None:
    """One card per participant — who has hardened, who still has room to move."""
    st.subheader("Where each party stands")
    if not stances:
        st.caption("No per-party reading returned.")
        return

    st.caption(
        "A whole-thread reading per participant — not an average of the per-turn sentiment "
        "below. Ordered least flexible first: the party with no room to move is the one "
        "holding up the deal."
    )
    # Three across keeps each card readable; threads with more participants wrap.
    for start in range(0, len(stances), 3):
        row = stances[start:start + 3]
        for col, s in zip(st.columns(len(row)), row):
            with col, st.container(border=True):
                st.markdown(f"#### {s.get('party', '—')}")
                st.markdown(f"**Mood:** `{s.get('mood', 'unknown')}`")
                st.markdown(f"**Flexibility:** `{s.get('flexibility', 'unknown')}`")
                st.markdown(f"**Holding:** {s.get('position') or '—'}")
                if s.get("rationale"):
                    st.caption(s["rationale"])
                turns = s.get("evidence_turns") or []
                cited = ", ".join(f"turn {t}" for t in turns) if turns else "none cited"
                st.caption(f"Evidence: {cited}")


def _render(result: dict) -> None:
    parsed = result.get("parsed") or []
    if parsed:
        with st.expander(f"What the parser read — {len(parsed)} messages", expanded=False):
            st.caption(
                "Check this before trusting the analysis. Messages should be in chronological "
                "order with quoted history removed."
            )
            st.dataframe(parsed, use_container_width=True, hide_index=True)

    _render_trajectory(result.get("trajectory"))
    _render_party_stances(result.get("party_stances") or [])

    top1, top2 = st.columns([3, 2])
    with top1:
        st.subheader("Recommendation")
        rec = result.get("recommendation", {})
        st.markdown(f"**Next move:** {rec.get('next_move', '—')}")
        st.markdown(f"**Tactic:** `{rec.get('tactic', '—')}`")
        st.markdown(f"**Rationale:** {rec.get('rationale', '—')}")
        cases = rec.get("grounded_case_ids") or []
        if cases:
            st.markdown("**Cites:** " + ", ".join(f"`{c}`" for c in cases))

    with top2:
        st.subheader("Breakdown risk")
        prob = result.get("outcome_prob")
        if prob is None:
            st.caption(
                "Not available on this path — the outcome model needs the priority and "
                "personality features an email thread doesn't carry."
            )
        else:
            st.metric("P(agreement reached)", f"{prob:.2f}")
            st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Per-turn sentiment")
    st.dataframe(result.get("sentiment", []), use_container_width=True, hide_index=True)

    st.subheader("Extreme-behavior flags")
    flags = result.get("behaviors", [])
    present = [f for f in flags if f.get("present")]
    if present:
        st.dataframe(present, use_container_width=True, hide_index=True)
        with st.expander("All categories, including those not flagged"):
            st.dataframe(flags, use_container_width=True, hide_index=True)
    else:
        st.success("No extreme behaviors flagged.")
        with st.expander("All categories"):
            st.dataframe(flags, use_container_width=True, hide_index=True)

    _render_precedents(result)

    indexed = result.get("indexed")
    if indexed:
        if indexed.get("error"):
            # Surfaced rather than swallowed: the analysis succeeded, but the
            # user asked for this to be retained and it was not.
            st.warning(f"Not saved to the knowledge base — {indexed['error']}")
        else:
            note = f"Saved as precedent ({indexed['chunks']} chunks) in `{indexed['namespace']}`."
            if indexed.get("replaced"):
                note += f" Replaced a previous version ({indexed['replaced']} chunks)."
            st.success(note)


def _render_precedents(result: dict) -> None:
    """Retrieved cases, plus which retriever found them and on what evidence."""
    info = result.get("retrieval") or {}
    mode = info.get("mode", "vector")

    st.subheader("Retrieved precedents")

    # Which corpus answered decides what these hits are worth, so it is said
    # before the hits rather than left to be inferred from their content.
    if info.get("corpus") == "live":
        st.caption(f"Grounded in your knowledge base: **{info.get('namespace')}**.")
    elif mode != "none" and result.get("retrieved"):
        st.warning(
            "These come from the built-in **CaSiNo research corpus** — campsite "
            "resource-bartering, not your organisation's history. Useful for tactical "
            "structure; not comparable deals. Set a namespace and ingest your own "
            "documents on the Knowledge base tab to ground this properly."
        )

    if mode in ("graph", "hybrid"):
        # The graph layer degrades to vector-only silently by design, so an
        # unloaded graph otherwise looks exactly like a graph that didn't help.
        # Say which one happened rather than letting the reader assume.
        if info.get("graph_effective"):
            st.caption(
                f"Retriever: **{_RETRIEVAL_MODES.get(mode, mode)}** — "
                f"{info.get('n_graph_grounded', 0)} of {info.get('n_retrieved', 0)} hits carry "
                "graph evidence."
            )
        else:
            st.warning(
                "The graph arm was requested but contributed nothing — these results are "
                "vector-only, and are **not** a measurement of graph retrieval quality."
            )
            if info.get("note"):
                st.caption(info["note"])
        if info.get("plan"):
            st.caption(f"Query plan: `{info['plan']}`")
    elif mode == "vector":
        st.caption(f"Retriever: **{_RETRIEVAL_MODES['vector']}**.")

    retrieved = result.get("retrieved", [])
    if not retrieved:
        st.info("No precedents returned (RAG disabled, or the query matched nothing).")
        return

    for r_ in retrieved:
        meta = r_.get("metadata") or {}
        # A live chunk is named by its document and clause; a benchmark case
        # only has an id. An unverifiable citation is how the fabrication
        # problem started, so show the checkable name where one exists.
        name = meta.get("title") or r_["case_id"]
        if meta.get("heading"):
            name = f"{name} — {meta['heading']}"
        label = f"{name} · score={r_['score']:.3f} · {r_['source']}/{r_['kind']}"
        if r_.get("vector_score") is not None and r_.get("graph_score") is not None:
            # Under fusion `score` is a blend, not a similarity — showing the
            # two components keeps that from reading as a cosine.
            label += f" · graph={r_['graph_score']:.2f} vector={r_['vector_score']:.3f}"
        with st.expander(label):
            why = r_.get("matched_by") or []
            if why:
                st.markdown("**Why this case was retrieved**")
                for line in why:
                    st.markdown(f"- {line}")
                st.markdown("---")
            st.write(r_["text"])


def _analysis_payload(base: dict) -> dict:
    """Common request fields for both analysis tabs."""
    payload = dict(base)
    payload["use_rag"] = use_rag
    payload["retrieval_mode"] = retrieval_mode
    if retrieval_query.strip():
        payload["retrieval_query"] = retrieval_query.strip()
    if namespace:
        payload["namespace"] = namespace
        payload["index_result"] = index_result
    return payload


tab_thread, tab_json, tab_kb = st.tabs(
    ["Email thread", "Transcript JSON (advanced)", "Knowledge base"]
)

with tab_thread:
    st.caption("Paste a thread. Newest-first order and quoted replies are handled.")
    thread_text = st.text_area(
        "Thread", value=DEFAULT_THREAD, height=340, label_visibility="collapsed"
    )
    if st.button("Analyze thread", type="primary"):
        payload = _analysis_payload({"thread_text": thread_text})
        with st.spinner("Parsing and analyzing (cold start can take ~90 s)…"):
            result = _post("/analyze/thread", payload)
        if result:
            _render(result)

with tab_json:
    st.caption("The typed contract — a pre-normalized Transcript, as the evals use.")
    transcript_json = st.text_area(
        "Transcript JSON", value=json.dumps(DEFAULT_TRANSCRIPT, indent=2), height=340,
        label_visibility="collapsed",
    )
    if st.button("Analyze transcript"):
        try:
            transcript = json.loads(transcript_json)
        except json.JSONDecodeError as exc:
            st.error(f"That isn't valid JSON: {exc}")
            st.stop()
        payload = _analysis_payload({"transcript": transcript})
        with st.spinner("Analyzing (cold start can take ~90 s)…"):
            result = _post("/analyze", payload)
        if result:
            _render(result)

with tab_kb:
    st.caption(
        "Precedent retrieval is only as good as what it can retrieve from. Add your own "
        "past threads, signed contracts and playbooks here — they are chunked by structure "
        "(messages, numbered clauses) and indexed immediately."
    )
    if not namespace:
        st.info("Set a **Namespace** in the sidebar to create or open a knowledge base.")
    else:
        stats = _request("GET", "/corpus/stats", params={"namespace": namespace})
        if stats:
            left, right = st.columns(2)
            left.metric("Documents", stats.get("documents", 0))
            right.metric("Chunks", stats.get("chunks", 0))
            if stats.get("note"):
                st.info(stats["note"])

        with st.form("ingest"):
            kb_title = st.text_input(
                "Title", placeholder="Acme MSA 2025 — signed",
                help="What a citation will show. Make it something you could go and find.",
            )
            kb_type = st.selectbox(
                "Document type",
                options=["contract", "email_thread", "playbook", "note"],
                help="Chooses how the text is split: contracts and playbooks by numbered "
                "clause, threads by message, notes by paragraph.",
            )
            kb_text = st.text_area("Text", height=260, placeholder="Paste the document…")
            if st.form_submit_button("Add to knowledge base", type="primary"):
                if not kb_title.strip() or not kb_text.strip():
                    st.warning("Both a title and some text are required.")
                else:
                    with st.spinner("Chunking and embedding…"):
                        added = _post(
                            "/corpus/documents",
                            {
                                "namespace": namespace,
                                "title": kb_title.strip(),
                                "text": kb_text,
                                "source_type": kb_type,
                            },
                        )
                    if added:
                        message = f"Indexed **{added['title']}** as {added['chunks']} chunks."
                        if added.get("replaced"):
                            message += (
                                f" Replaced a previous version ({added['replaced']} chunks) —"
                                " re-ingesting the same document updates it rather than"
                                " duplicating it."
                            )
                        st.success(message)

        docs = _request("GET", "/corpus/documents", params={"namespace": namespace})
        if docs:
            st.markdown("#### In this knowledge base")
            st.dataframe(docs, use_container_width=True, hide_index=True)
