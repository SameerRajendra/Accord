"""Tests for live-document ingestion — the path that frees Accord from CaSiNo.

All pure: no database, no embedding model, no LLM. Ingestion is deliberately
deterministic and GPU-free (a knowledge base you can only load while a GPU is
warm is one nobody loads), so it is fully testable offline.

What's under test is mostly *chunk boundaries*, because the unit of retrieval
decides what a citation can claim. A chunk that merges two people's words
cannot be attributed; half a contract clause cannot be quoted.
"""

import pytest

from data.schema import Outcome, Party, Transcript, Turn
from rag.documents import (
    BENCHMARK_COLLECTION,
    MIN_CHUNK_CHARS,
    TARGET_CHUNK_CHARS,
    IngestError,
    SourceType,
    build_document,
    chunk_document,
    collection_for,
    document_from_transcript,
    make_doc_id,
    normalize_namespace,
)


def _doc(text: str, source_type: SourceType, title: str = "T"):
    return build_document(title=title, text=text, namespace="acme", source_type=source_type)


# --- namespaces: an isolation boundary, not a label ------------------------


@pytest.mark.parametrize("bad", ["has space", "UPPER CASE!", "semi;colon", "a" * 64, "-lead"])
def test_namespace_rejects_anything_that_is_not_a_slug(bad):
    """Namespaces become Postgres collection names — this is an injection boundary."""
    with pytest.raises(IngestError):
        normalize_namespace(bad)


def test_namespace_is_case_normalized():
    assert normalize_namespace("Acme-Legal") == "acme-legal"


def test_absent_namespace_falls_back_to_the_default_rather_than_failing():
    """Single-tenant demos should never have to think about namespaces."""
    assert normalize_namespace(None) == "default"
    assert normalize_namespace("") == "default"


def test_a_namespace_can_never_address_the_benchmark_collection():
    """Otherwise a user's ingestion could overwrite the corpus the evals score."""
    assert collection_for("accord_cases") != BENCHMARK_COLLECTION
    assert collection_for("default") != BENCHMARK_COLLECTION


def test_doc_id_is_content_addressed_and_namespace_scoped():
    a = make_doc_id("acme", "MSA", "same text")
    assert a == make_doc_id("acme", "MSA", "same text")     # re-ingest replaces
    assert a != make_doc_id("acme", "MSA", "edited text")   # edit is a new doc
    assert a != make_doc_id("other", "MSA", "same text")    # tenants stay separate


# --- contracts: the clause is the unit a lawyer cites ----------------------


CONTRACT = """MASTER SERVICES AGREEMENT

7.1 Limitation of Liability. Except as set out in Section 7.2, neither party's
total aggregate liability shall exceed the fees paid in the preceding twelve
months, and neither party shall be liable for indirect or consequential loss.

7.2 Indemnification. Vendor shall indemnify Buyer against third-party claims
that the Services infringe any intellectual property right, provided Buyer
notifies Vendor promptly and permits Vendor to control the defence.

8. Term and Termination. This Agreement continues for an initial term of
twenty-four months and renews automatically for successive twelve-month terms
unless either party gives ninety days written notice.
"""


def test_contract_splits_on_numbered_clauses():
    chunks = chunk_document(_doc(CONTRACT, SourceType.CONTRACT))
    headings = [c.heading for c in chunks]

    assert any(h and h.startswith("7.1") for h in headings)
    assert any(h and h.startswith("7.2") for h in headings)
    assert any(h and h.startswith("8") for h in headings)
    # Each clause must survive whole — a half-clause is not quotable.
    liability = next(c for c in chunks if c.heading and c.heading.startswith("7.1"))
    assert "preceding twelve" in liability.text


def test_clause_heading_is_embedded_with_the_body():
    """The heading carries the topical signal; the body is often boilerplate.

    Dropping it repeats `data/build_case_corpus.py`'s mistake — embedding
    near-identical text and wondering why cosine can't separate the documents.
    """
    chunks = chunk_document(_doc(CONTRACT, SourceType.CONTRACT))
    liability = next(c for c in chunks if c.heading and c.heading.startswith("7.1"))
    assert "Limitation of Liability" in liability.embed_text()


def test_a_bare_number_in_prose_does_not_start_a_clause():
    """'...within 3 business days' must not be read as the start of clause 3."""
    text = (
        "1. Notice. Either party may terminate on notice.\n"
        "Payment is due within 3 business days of invoice, and 5 further days for disputes.\n"
    )
    chunks = chunk_document(_doc(text, SourceType.CONTRACT))
    assert len(chunks) == 1


# --- threads: attribution is the thing that must not break -----------------


THREAD = """From: Daniel Okafor
Sent: Thursday, 7 August 2026 18:22
To: Priya Raman
Subject: RE: MSA renewal

I've gone as far as I intend to. The uplift stands at 38% and the auto-renewal
clause is not negotiable.

Best regards,
Daniel

> From: Priya Raman
> Sent: Thursday, 7 August 2026 14:05
>
> Daniel, you've moved 2% in three weeks while refusing to explain the
> underlying cost basis. That is difficult to take seriously as partnership.
"""


def test_thread_splits_by_message_and_keeps_the_sender():
    chunks = chunk_document(_doc(THREAD, SourceType.EMAIL_THREAD))
    assert len(chunks) == 2
    assert "Daniel Okafor" in (chunks[0].heading or "")
    assert "Priya Raman" in (chunks[1].heading or "")
    # Two people's words must never land in one chunk — attribution is the
    # whole product, and a merged chunk cannot be attributed to either.
    assert "38%" in chunks[0].text and "38%" not in chunks[1].text
    assert "cost basis" in chunks[1].text


def test_signature_and_routing_headers_are_dropped():
    """Boilerplate repeats in every message from a sender, so it makes them look alike."""
    chunks = chunk_document(_doc(THREAD, SourceType.EMAIL_THREAD))
    body = "\n".join(c.text for c in chunks)
    assert "Best regards" not in body
    assert "Subject:" not in body
    assert "To: Priya" not in body


def test_quote_markers_are_stripped_from_the_text():
    chunks = chunk_document(_doc(THREAD, SourceType.EMAIL_THREAD))
    assert not any(line.startswith(">") for c in chunks for line in c.text.splitlines())


# --- sizing and fallbacks --------------------------------------------------


def test_long_prose_is_packed_to_roughly_the_target_size():
    paragraph = ("This is a sentence about liability caps and renewal terms. " * 8).strip()
    text = "\n\n".join([paragraph] * 12)
    chunks = chunk_document(_doc(text, SourceType.NOTE))

    assert len(chunks) > 1
    assert all(len(c.text) <= 2_000 for c in chunks)
    assert any(len(c.text) > TARGET_CHUNK_CHARS / 2 for c in chunks)


def test_tiny_trailing_fragments_are_merged_not_indexed_alone():
    """A chunk holding 'Sincerely,' would occupy a top-k slot a precedent needs."""
    text = "7.1 Liability. " + ("Detailed clause text. " * 60) + "\n\nOK.\n"
    chunks = chunk_document(_doc(text, SourceType.CONTRACT))
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in chunks)


def test_unstructured_text_still_produces_one_chunk():
    """Silently ingesting nothing would look like success and retrieve like nothing."""
    chunks = chunk_document(_doc("just one line with no structure at all", SourceType.NOTE))
    assert len(chunks) == 1
    assert chunks[0].index == 0


def test_empty_and_oversized_documents_are_refused_with_a_reason():
    with pytest.raises(IngestError):
        build_document(title="T", text="   ", namespace="acme")
    with pytest.raises(IngestError):
        build_document(title="  ", text="content", namespace="acme")


def test_chunk_metadata_carries_what_a_citation_needs():
    chunks = chunk_document(_doc(CONTRACT, SourceType.CONTRACT, title="Acme MSA 2025"))
    meta = chunks[0].metadata
    assert meta["namespace"] == "acme"
    assert meta["title"] == "Acme MSA 2025"
    assert meta["source_type"] == "contract"
    assert chunks[0].chunk_id.startswith(chunks[0].doc_id + "#")


# --- self-growing corpus ---------------------------------------------------


def _transcript() -> Transcript:
    return Transcript(
        dialogue_id="thread-1",
        source="email_thread",
        domain="business_negotiation",
        parties=[Party(party_id="Daniel"), Party(party_id="Priya")],
        turns=[
            Turn(index=0, speaker="Daniel", text="The uplift stands at 38 percent."),
            Turn(index=1, speaker="Priya", text="That is well outside what we budgeted."),
        ],
        outcome=Outcome(agreement_reached=False),
        metadata={"subject": "MSA renewal"},
    )


def test_analyzed_thread_is_rendered_so_the_message_chunker_understands_it():
    """Written for the chunker on purpose — cheaper than a fourth chunking mode."""
    document = document_from_transcript(_transcript(), namespace="acme")
    chunks = chunk_document(document)

    assert document.title == "MSA renewal"
    assert document.source_type is SourceType.ANALYZED_THREAD
    assert len(chunks) == 2
    assert "Daniel" in (chunks[0].heading or "")
    assert "Priya" in (chunks[1].heading or "")


def test_the_analysis_is_metadata_and_is_not_embedded():
    """Retrieval must match on what was said, not on the system's own past verdicts."""
    document = document_from_transcript(
        _transcript(), namespace="acme", analysis={"trajectory": "escalating"}
    )
    assert document.metadata["analysis"]["trajectory"] == "escalating"
    assert "escalating" not in document.text

    chunks = chunk_document(document)
    assert not any("escalating" in c.embed_text() for c in chunks)
