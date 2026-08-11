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
        {"party_id": "agent_1", "priorities": {"Firewood": "High", "Food": "Medium", "Water": "Low"}, "metadata": {}},
        {"party_id": "agent_2", "priorities": {"Firewood": "High", "Water": "Medium", "Food": "Low"}, "metadata": {}},
    ],
    "turns": [
        {"index": 0, "speaker": "agent_1", "text": "Hi! I'm hoping to grab extra firewood - our group has a lot of seniors who need to stay warm."},
        {"index": 1, "speaker": "agent_2", "text": "I need firewood too, my dog has fleas and the fire keeps them off. I can't give it up."},
        {"index": 2, "speaker": "agent_1", "text": "There's no way you need it more than a group of elderly people. That's a bit ridiculous."},
        {"index": 3, "speaker": "agent_2", "text": "Take it or leave it - I'm keeping all three firewood or there's no deal."},
    ],
    "outcome": {"agreement_reached": False, "final_deal": None, "points": {}},
    "has_strategy_annotations": False,
    "metadata": {"split": "demo"},
}


st.title("Accord")
st.caption(
    "Negotiation intelligence: where it's heading · who stands where · sentiment · "
    "behaviors · risk · precedent · recommendation."
)

if not API_URL:
    st.warning(
        "`ACCORD_API_URL` is not set on this UI container. Point it at the deployed API, e.g. "
        "`https://<workspace>--accord-accordserver-api.modal.run`."
    )

with st.sidebar:
    st.subheader("Options")
    use_rag = st.toggle(
        "Enable RAG (retrieval)",
        value=True,
        help="Off = ablation baseline. The recommendation runs without precedent grounding.",
    )
    retrieval_query = st.text_area(
        "Custom retrieval query (optional)",
        value="",
        help="If empty, the last few turns are used as the query.",
    )
    st.markdown("---")
    st.caption(
        "Cold-start note: the first request after idle pays ~90 s while the GPU wakes and "
        "loads the model. Subsequent requests take seconds."
    )
    st.caption(
        "Precedent corpus is CaSiNo (campsite resource negotiations). Cross-domain relevance "
        "to business threads is untested — see the retrieved scores."
    )


def _post(path: str, payload: dict):
    """POST to the API, surfacing errors as readable Streamlit messages."""
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=300.0)
    except httpx.RequestError as exc:
        st.error(f"Couldn't reach the API at {API_URL}: {exc}")
        return None
    if r.status_code == 422:
        # The parser rejected the input — the message explains what to fix.
        try:
            st.warning(r.json().get("detail", r.text))
        except Exception:  # noqa: BLE001
            st.warning(r.text[:500])
        return None
    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:500]}")
        return None
    return r.json()


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

    st.subheader("Retrieved precedents")
    retrieved = result.get("retrieved", [])
    if not retrieved:
        st.info("No precedents returned (RAG disabled, or the query matched nothing).")
    else:
        for r_ in retrieved:
            with st.expander(f"[{r_['case_id']}] score={r_['score']:.3f} · {r_['source']}/{r_['kind']}"):
                st.write(r_["text"])


tab_thread, tab_json = st.tabs(["Email thread", "Transcript JSON (advanced)"])

with tab_thread:
    st.caption("Paste a thread. Newest-first order and quoted replies are handled.")
    thread_text = st.text_area("Thread", value=DEFAULT_THREAD, height=340, label_visibility="collapsed")
    if st.button("Analyze thread", type="primary"):
        payload = {"thread_text": thread_text, "use_rag": use_rag}
        if retrieval_query.strip():
            payload["retrieval_query"] = retrieval_query.strip()
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
        payload = {"transcript": transcript, "use_rag": use_rag}
        if retrieval_query.strip():
            payload["retrieval_query"] = retrieval_query.strip()
        with st.spinner("Analyzing (cold start can take ~90 s)…"):
            result = _post("/analyze", payload)
        if result:
            _render(result)
