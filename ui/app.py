"""Streamlit UI — paste a transcript, see Accord's full analysis.

Deployed as its own Modal ASGI app (`accord-ui`); calls the `accord` API's
`/analyze` endpoint over HTTPS. `ACCORD_API_URL` env var points at the API;
default falls back to a hint if unset so a first run surfaces the miswiring
instead of a cryptic connection error.

Kept intentionally minimal — one page, three panels, no auth. This is the
"shareable in an afternoon" UI DESIGN.md §8 committed to, not a product.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

st.set_page_config(page_title="Accord — Negotiation Intelligence", layout="wide")

API_URL = os.environ.get("ACCORD_API_URL", "").rstrip("/")

DEFAULT_TRANSCRIPT: dict[str, Any] = {
    "dialogue_id": "demo-1",
    "source": "manual",
    "domain": "consumer-marketplace",
    "parties": [
        {"party_id": "buyer", "metadata": {"target": 300, "item_listed_price": 450}},
        {"party_id": "seller", "metadata": {"target": 400, "item_listed_price": 450}},
    ],
    "turns": [
        {"index": 0, "speaker": "seller", "text": "Hi, I have the bike listed for $450. It's barely used."},
        {"index": 1, "speaker": "buyer", "text": "Would you take $280? That's my max."},
        {"index": 2, "speaker": "seller", "text": "$280 is way too low. I could do $420, that's already a big drop."},
        {"index": 3, "speaker": "buyer", "text": "I'm not going anywhere near $420. Meet me at $310."},
        {"index": 4, "speaker": "seller", "text": "You clearly don't understand what this bike is worth."},
    ],
    "outcome": {"agreement_reached": False, "final_deal": None, "points": {}},
    "has_strategy_annotations": False,
    "metadata": {"split": "demo", "category": "bike"},
}


st.title("Accord")
st.caption("Negotiation intelligence: sentiment · behaviors · risk · precedent · recommendation.")

if not API_URL:
    st.warning(
        "`ACCORD_API_URL` is not set on this UI container. "
        "Set it to the deployed FastAPI URL, e.g. `https://<workspace>--accord-accordserver-api.modal.run`."
    )

with st.sidebar:
    st.subheader("Options")
    use_rag = st.toggle("Enable RAG (retrieval)", value=True, help="Off = ablation baseline (no precedent grounding).")
    retrieval_query = st.text_area(
        "Custom retrieval query (optional)",
        value="",
        help="If empty, the last few turns are used as the query.",
    )
    st.markdown("---")
    st.caption("Cold-start note: the first request after idle pays ~60–90 s for SGLang to boot. Subsequent requests are seconds.")

st.subheader("Transcript")
transcript_json = st.text_area(
    "Paste a normalized Transcript JSON, or edit the demo below:",
    value=json.dumps(DEFAULT_TRANSCRIPT, indent=2),
    height=380,
)

col_run, _ = st.columns([1, 5])
with col_run:
    run = st.button("Analyze", type="primary", use_container_width=True)

if run:
    try:
        transcript = json.loads(transcript_json)
    except json.JSONDecodeError as exc:
        st.error(f"Transcript is not valid JSON: {exc}")
        st.stop()

    payload = {"transcript": transcript, "use_rag": use_rag}
    if retrieval_query.strip():
        payload["retrieval_query"] = retrieval_query.strip()

    with st.spinner("Analyzing (cold start can take ~90 s)…"):
        try:
            r = httpx.post(f"{API_URL}/analyze", json=payload, timeout=180.0)
        except httpx.RequestError as exc:
            st.error(f"Failed to reach the API at {API_URL}: {exc}")
            st.stop()

    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:500]}")
        st.stop()

    result = r.json()

    top1, top2 = st.columns(2)
    with top1:
        st.subheader("Recommendation")
        rec = result.get("recommendation", {})
        st.markdown(f"**Next move:** {rec.get('next_move', '—')}")
        st.markdown(f"**Tactic:** `{rec.get('tactic', '—')}`")
        st.markdown(f"**Rationale:** {rec.get('rationale', '—')}")
        cases = rec.get("grounded_case_ids") or []
        if cases:
            st.markdown("**Grounded in:** " + ", ".join(f"`{c}`" for c in cases))

    with top2:
        st.subheader("Breakdown risk")
        prob = result.get("outcome_prob")
        if prob is None:
            st.info("Outcome model artifact not loaded — deploy needs `train_outcome` to have been run.")
        else:
            st.metric("P(agreement_reached)", f"{prob:.2f}")
            st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Per-turn sentiment")
    st.dataframe(result.get("sentiment", []), use_container_width=True)

    st.subheader("Extreme-behavior flags")
    st.dataframe(result.get("behaviors", []), use_container_width=True)

    st.subheader("Retrieved precedents")
    retrieved = result.get("retrieved", [])
    if not retrieved:
        st.info("No precedents returned (RAG disabled or empty result).")
    else:
        for r_ in retrieved:
            with st.expander(f"[{r_['case_id']}] score={r_['score']:.3f} · {r_['source']}/{r_['kind']}"):
                st.write(r_["text"])
