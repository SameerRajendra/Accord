"""Live knowledge-base documents: the model and the chunker.

This is the module that decouples Accord from CaSiNo. Everything under
`data/` builds a *fixed academic corpus*: 1,030 campsite dialogues rendered
from one template, embedded in one batch, wiped and rebuilt on every run. That
corpus keeps an important job — it is the only body of text here with ground
truth, so the evals measure against it — but it cannot be the thing a live
negotiation retrieves from. A contract dispute grounded in precedent about
firewood is worse than no precedent at all.

So there are two corpora with two different jobs, and conflating them is the
mistake this module exists to prevent:

============  ==========================  ==================================
              CaSiNo (`accord_cases`)     live namespace (`accord_ns_<name>`)
============  ==========================  ==================================
role          frozen benchmark            the product
contents      1,030 templated case docs   whatever the user ingests
labels        gold case per query         none — unlabeled by nature
written by    `data/build_case_corpus`    this module, at runtime
measured by   `evals/retrieval_eval`      not measurable the same way
============  ==========================  ==================================

A live namespace starts **empty**. That is deliberate: seeding it with CaSiNo
would put campsite precedent into business results and quietly re-create the
domain mismatch. Retrieval over an empty namespace returns nothing and the
recommendation degrades to the no-RAG path, which is a legitimate answer for a
system that has not been given any institutional knowledge yet.

Chunking
--------
The unit of retrieval matters more than the embedding model does, and it is
not the same unit for every input:

- **Email threads** split by *message*. A thread is already a sequence of
  authored turns; splitting one on character count would cut mid-sentence and
  merge two people's words into one chunk, which destroys the attribution the
  whole product depends on.
- **Contracts** split by *clause*. Numbered sections are the unit lawyers
  actually cite and the unit a redline applies to. A clause that survives
  intact is quotable; half a clause is not.
- **Playbooks** split by *rule*, for the same reason.
- Anything else falls back to paragraph packing with overlap.

All of it is **deterministic and GPU-free**. Ingestion must work when SGLang
is cold or absent — a knowledge base you can only load while a GPU is warm is
a knowledge base nobody loads. `analysis/parse_thread.py` keeps the LLM path
for *analysis*, where the model is already running; this module deliberately
does not call it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

#: Vector-store collection holding the frozen CaSiNo benchmark corpus. Named
#: separately from the live namespaces so an eval can never accidentally score
#: against user-ingested documents, and a user's documents can never be wiped
#: by a corpus rebuild.
BENCHMARK_COLLECTION = "accord_cases"

#: Namespace used when a request doesn't name one. Single-tenant demos never
#: have to think about namespaces; multi-tenant deployments set it per request.
DEFAULT_NAMESPACE = "default"

_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

#: Target and hard ceiling for a chunk, in characters. ~1,200 chars is roughly
#: 300 tokens: comfortably inside all-MiniLM-L6-v2's 256-token window after
#: truncation, and large enough that a clause usually survives whole. The hard
#: cap only bites on pathological input (a contract with no paragraph breaks).
TARGET_CHUNK_CHARS = 1_200
MAX_CHUNK_CHARS = 2_000
#: Overlap when a long block has to be split mid-prose, so a sentence spanning
#: the boundary is retrievable from both sides.
CHUNK_OVERLAP_CHARS = 150
#: Chunks shorter than this are merged into their neighbour. A 12-character
#: chunk ("Sincerely,") is noise that still occupies a top-k slot.
MIN_CHUNK_CHARS = 80

#: Guardrail on a single ingested document. Higher than `parse_thread`'s
#: 24k thread limit because contracts are legitimately long, but bounded so a
#: mailbox export doesn't become one ingestion call.
MAX_DOCUMENT_CHARS = 400_000


class SourceType(str, Enum):
    """What kind of thing was ingested. Drives chunking and nothing else.

    Kept small and concrete. A taxonomy with fifteen entries would be
    unfillable by a user pasting text into a box, and every entry that doesn't
    change the chunking is a distinction the system can't act on.
    """

    EMAIL_THREAD = "email_thread"
    CONTRACT = "contract"
    PLAYBOOK = "playbook"
    #: A thread that went through `/analyze` and was kept as precedent. Its own
    #: type because it carries the analysis as metadata, which a raw thread
    #: does not, and because "what did we learn from our own past deals" is a
    #: different provenance claim from "here is a document someone uploaded".
    ANALYZED_THREAD = "analyzed_thread"
    NOTE = "note"


class IngestError(ValueError):
    """Input that cannot become a document. The message says what to fix."""


class Chunk(BaseModel):
    """One retrievable unit. `chunk_id` is the vector store's document id."""

    chunk_id: str
    doc_id: str
    index: int = Field(..., ge=0, description="0-based position within the document.")
    text: str
    heading: Optional[str] = Field(
        None,
        description="Clause heading, or 'Sender — date' for a message. Prepended to the "
        "embedded text so a clause number is searchable, and shown in citations.",
    )
    metadata: Dict = Field(default_factory=dict)

    def embed_text(self) -> str:
        """What actually gets embedded — heading included.

        The heading carries most of a clause's topical signal ("Limitation of
        Liability") while the body is often boilerplate legalese. Dropping it
        would be the same mistake `data/build_case_corpus.py` made with its
        shared template: embedding mostly-identical text and wondering why
        cosine can't tell the documents apart.
        """
        if self.heading:
            return f"{self.heading}\n\n{self.text}"
        return self.text


class LiveDocument(BaseModel):
    """A document in a live namespace, before chunking."""

    doc_id: str
    namespace: str = DEFAULT_NAMESPACE
    title: str
    source_type: SourceType = SourceType.NOTE
    text: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata: Dict = Field(default_factory=dict)


def normalize_namespace(namespace: Optional[str]) -> str:
    """Validate a namespace, or raise with the rule.

    Namespaces become collection names in Postgres, so this is an injection
    boundary as much as a formatting one — reject anything that isn't a plain
    slug rather than quoting it downstream and hoping.
    """
    value = (namespace or DEFAULT_NAMESPACE).strip().lower()
    if not _NAMESPACE_RE.match(value):
        raise IngestError(
            f"invalid namespace {namespace!r}: use 1-63 characters of a-z, 0-9, '-' or '_', "
            "starting with a letter or digit"
        )
    return value


def collection_for(namespace: str) -> str:
    """Vector-store collection backing a namespace.

    Prefixed so it can never collide with `BENCHMARK_COLLECTION`: a namespace
    literally named "accord_cases" would otherwise let a user's ingestion
    write into the frozen benchmark corpus and silently invalidate every
    committed retrieval number.
    """
    return f"accord_ns_{normalize_namespace(namespace)}"


def make_doc_id(namespace: str, title: str, text: str) -> str:
    """Content-addressed id: re-ingesting identical text is a no-op, not a duplicate.

    Hashing the *content* rather than minting a UUID means the natural user
    behaviour — pasting the same playbook again after editing one line —
    replaces the old chunks instead of leaving both versions in the index to
    compete. The namespace is in the hash so two tenants ingesting the same
    public document keep separate ids.
    """
    digest = hashlib.sha256()
    for part in (normalize_namespace(namespace), title.strip(), text.strip()):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return "doc-" + digest.hexdigest()[:16]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

#: A numbered clause heading: "7.", "7.2", "7.2.1", "Section 7 -", "ARTICLE IV".
#: Requires the number to be followed by a capitalized token, so a line reading
#: "3 business days from invoice" is not mistaken for the start of clause 3.
#:
#: The title capture stops at the first period rather than running to
#: end-of-line. Two reasons: it yields "Limitation of Liability" instead of the
#: clause's entire opening sentence, and — the part that actually matters —
#: anchoring the capture to `$` would make the whole pattern fail on any clause
#: whose text runs on the same line as its number, which is how most real
#: pasted contracts are formatted.
_CLAUSE_RE = re.compile(
    r"^\s*(?:(?:section|clause|article)\s+)?"
    r"(\d+(?:\.\d+)*|[IVXLC]+)"
    r"[.)]?\s+"
    r"(?=[A-Z\"'(])"
    r"([^.\n]{0,120})",
    re.IGNORECASE,
)

#: An ALL-CAPS or Title-Case standalone heading line ("LIMITATION OF LIABILITY").
_BARE_HEADING_RE = re.compile(r"^\s*([A-Z][A-Z \-/&']{4,80})\s*:?\s*$")

#: Start-of-message markers in a pasted thread. Deliberately narrow: a false
#: positive splits one person's message in two and attributes half of it to
#: nobody, which is worse than a false negative (an over-large chunk).
_MESSAGE_HEADER_RES = (
    re.compile(r"^\s*>*\s*From:\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*>*\s*On .{3,80}?,?\s*(.+?)\s+wrote:\s*$", re.IGNORECASE),
)

#: Signature and disclaimer openers. Everything from here to the end of a
#: message is dropped: it is the same boilerplate in every message from that
#: sender, so keeping it makes every one of their messages look alike to
#: cosine similarity.
_SIGNATURE_RES = (
    re.compile(r"^\s*--\s*$"),
    re.compile(
        r"^\s*(best regards|kind regards|regards|sincerely|thanks|thank you|cheers)\s*[,.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(this (e-?mail|message) (is|and any)|confidentiality notice|disclaimer:)",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*sent from my \w+", re.IGNORECASE),
)


def _strip_quote_markers(line: str) -> str:
    return re.sub(r"^\s*(?:>\s?)+", "", line)


def _is_signature_start(line: str) -> bool:
    return any(pattern.match(line) for pattern in _SIGNATURE_RES)


def _split_email_messages(text: str) -> List[Dict[str, str]]:
    """Split a pasted thread into messages. Deterministic, no LLM.

    Returns `[{"heading": ..., "text": ...}]` in the order they appear, which
    for most clients is newest-first. **Order is deliberately not corrected
    here.** Chronology matters for escalation scoring, which is
    `analysis/parse_thread.py`'s job on the analysis path; for retrieval each
    message is embedded independently, so their order in the list changes
    nothing. Re-implementing (and disagreeing with) the LLM's ordering logic
    would be strictly worse than not doing it.

    Quoted history is *not* de-duplicated either, for a reason specific to
    retrieval: a quoted block is the same text as an earlier message, so it
    hashes and embeds to a near-identical chunk. Dropping quotes entirely is
    handled by the caller merging short chunks and by content-addressed ids;
    aggressively removing them here would delete the only copy when someone
    pastes only the latest reply *with* its history.
    """
    lines = text.splitlines()
    messages: List[Dict[str, str]] = []
    current_heading: Optional[str] = None
    buffer: List[str] = []
    in_signature = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            messages.append({"heading": current_heading or "", "text": body})
        buffer.clear()

    for raw in lines:
        line = _strip_quote_markers(raw)
        header_match = None
        for pattern in _MESSAGE_HEADER_RES:
            header_match = pattern.match(line)
            if header_match:
                break
        if header_match:
            flush()
            sender = header_match.group(1).strip()
            current_heading = f"Message from {sender}" if sender else "Message"
            in_signature = False
            continue
        # Header continuation lines carry routing, not content.
        if re.match(r"^\s*(to|cc|bcc|sent|date|subject):\s", line, re.IGNORECASE):
            continue
        if _is_signature_start(line):
            in_signature = True
            continue
        if in_signature:
            # A blank line ends the signature block; real content can follow it
            # when a thread continues below a quoted signature.
            if not line.strip():
                in_signature = False
            continue
        buffer.append(line)

    flush()
    return messages


def _split_clauses(text: str) -> List[Dict[str, str]]:
    """Split a contract or playbook into numbered clauses / rules.

    Falls back to returning a single block when no numbering is found, which
    the caller then paragraph-packs. That fallback is the common case for
    prose playbooks and is not a failure.
    """
    lines = text.splitlines()
    blocks: List[Dict[str, str]] = []
    heading: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or heading:
            blocks.append({"heading": heading or "", "text": body})
        buffer.clear()

    for line in lines:
        clause = _CLAUSE_RE.match(line)
        bare = _BARE_HEADING_RE.match(line) if not clause else None
        if clause:
            flush()
            number, title = clause.group(1), clause.group(2).strip()
            heading = f"{number} {title}".strip()
            # The heading line's trailing text is often the clause's first
            # sentence rather than a title; keep it in the body too so nothing
            # is lost if it was content.
            buffer.append(line.strip())
        elif bare:
            flush()
            heading = bare.group(1).strip()
        else:
            buffer.append(line)

    flush()
    return [b for b in blocks if b["text"].strip()]


def _pack_paragraphs(text: str) -> List[str]:
    """Group paragraphs up to `TARGET_CHUNK_CHARS`, splitting any that exceed it.

    Paragraph-first rather than character-first: a paragraph break is a real
    semantic boundary an author put there, and respecting it costs nothing.
    Only a paragraph that is itself over the ceiling gets cut, and then with
    overlap so a sentence spanning the cut is still retrievable.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    packed: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHUNK_CHARS:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_long_block(paragraph))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > TARGET_CHUNK_CHARS and current:
            packed.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        packed.append(current)
    return packed


def _split_long_block(block: str) -> List[str]:
    """Cut an over-long block at sentence boundaries, with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > MAX_CHUNK_CHARS:
            # A single "sentence" this long is unpunctuated text; hard-cut it.
            if current:
                out.append(current)
                current = ""
            for start in range(0, len(sentence), MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS):
                out.append(sentence[start:start + MAX_CHUNK_CHARS])
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > TARGET_CHUNK_CHARS and current:
            out.append(current)
            tail = current[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
            current = f"{tail} {sentence}".strip()
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _merge_short(blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fold undersized *continuation* fragments into the previous block.

    A chunk holding "Sincerely," or a lone clause number occupies a top-k slot
    that a real precedent should have. But merging must never cross an
    attribution boundary: two short messages from different senders, or two
    short clauses with different numbers, are distinct authored units, and
    combining them would put one party's words under another's name — the
    exact failure message-level and clause-level chunking exists to avoid.

    So a block is merged only when it is a continuation of the previous one:
    it carries no heading of its own, or the same heading. A distinct,
    non-empty heading is a hard boundary regardless of how short the block is.
    """
    merged: List[Dict[str, str]] = []
    for block in blocks:
        prev = merged[-1] if merged else None
        is_continuation = prev is not None and (
            not block["heading"] or block["heading"] == prev["heading"]
        )
        if (
            is_continuation
            and len(block["text"]) < MIN_CHUNK_CHARS
            and len(prev["text"]) + len(block["text"]) <= MAX_CHUNK_CHARS
        ):
            prev["text"] = f"{prev['text']}\n\n{block['text']}".strip()
            continue
        merged.append(dict(block))
    return merged


def chunk_document(document: LiveDocument) -> List[Chunk]:
    """Split a document into retrievable chunks, by source type.

    Never returns an empty list for non-empty text: a document that resists
    every structural heuristic still comes back as one chunk. Silently
    ingesting nothing would look like success and retrieve like an empty
    corpus.
    """
    text = (document.text or "").strip()
    if not text:
        raise IngestError("document text is empty")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise IngestError(
            f"document is {len(text):,} characters; the limit is {MAX_DOCUMENT_CHARS:,}. "
            "Split it into separate documents — a whole mailbox export is not one document."
        )

    if document.source_type in (SourceType.EMAIL_THREAD, SourceType.ANALYZED_THREAD):
        blocks = _split_email_messages(text)
    elif document.source_type in (SourceType.CONTRACT, SourceType.PLAYBOOK):
        blocks = _split_clauses(text)
    else:
        blocks = []

    if not blocks:
        blocks = [{"heading": "", "text": text}]

    # Any block still over the target gets paragraph-packed. Structural
    # splitting decides *where* boundaries are; this decides how big.
    sized: List[Dict[str, str]] = []
    for block in blocks:
        if len(block["text"]) <= TARGET_CHUNK_CHARS:
            sized.append(block)
            continue
        for piece in _pack_paragraphs(block["text"]):
            sized.append({"heading": block["heading"], "text": piece})

    sized = _merge_short(sized)

    chunks: List[Chunk] = []
    for index, block in enumerate(sized):
        body = block["text"].strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{index}",
                doc_id=document.doc_id,
                index=index,
                text=body,
                heading=block["heading"].strip() or None,
                metadata={
                    "namespace": document.namespace,
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "source_type": document.source_type.value,
                    "created_at": document.created_at,
                    "chunk_index": index,
                    **document.metadata,
                },
            )
        )

    if not chunks:  # pragma: no cover — `text` was non-empty, so this can't be reached
        raise IngestError("document produced no usable chunks")
    return chunks


def build_document(
    title: str,
    text: str,
    namespace: str = DEFAULT_NAMESPACE,
    source_type: SourceType = SourceType.NOTE,
    metadata: Optional[Dict] = None,
) -> LiveDocument:
    """Validate input and mint a content-addressed document."""
    title = (title or "").strip()
    if not title:
        raise IngestError("title is required — it is what a citation shows the user")
    namespace = normalize_namespace(namespace)
    if not (text or "").strip():
        raise IngestError("document text is empty")
    return LiveDocument(
        doc_id=make_doc_id(namespace, title, text),
        namespace=namespace,
        title=title,
        source_type=source_type,
        text=text,
        metadata=dict(metadata or {}),
    )


def document_from_transcript(
    transcript,
    namespace: str = DEFAULT_NAMESPACE,
    title: Optional[str] = None,
    analysis: Optional[Dict] = None,
) -> LiveDocument:
    """Turn an analyzed negotiation into a precedent document for the corpus.

    This is the self-growing half of the knowledge base: a thread that has
    been through `/analyze` is exactly the kind of thing that should ground
    the *next* one, and it costs one write to keep it.

    The turns are rendered with `From: <speaker>` headers rather than
    `Speaker: text` on purpose — that is the form `_split_email_messages`
    already recognizes, so each turn becomes its own chunk with correct
    attribution instead of collapsing into one paragraph-packed blob. Writing
    for the chunker beats adding a fourth chunking strategy.

    `analysis` (trajectory, recommended tactic, flagged behaviours) is stored
    as metadata, not folded into the text. Retrieval should match on what was
    *said*; the analysis is a derived judgement, and embedding it would let
    the system retrieve its own past conclusions and mistake them for evidence.
    """
    lines: List[str] = []
    for turn in transcript.turns:
        if not (turn.text or "").strip():
            continue
        lines.append(f"From: {turn.speaker}\n\n{turn.text.strip()}\n")
    body = "\n".join(lines).strip()
    if not body:
        raise IngestError("transcript has no text turns to index")

    subject = (transcript.metadata or {}).get("subject")
    resolved_title = title or subject or f"Negotiation {transcript.dialogue_id}"

    metadata: Dict = {
        "dialogue_id": transcript.dialogue_id,
        "parties": [p.party_id for p in transcript.parties],
        "n_turns": len(transcript.turns),
    }
    if analysis:
        metadata["analysis"] = analysis

    return LiveDocument(
        doc_id=make_doc_id(namespace, resolved_title, body),
        namespace=normalize_namespace(namespace),
        title=resolved_title,
        source_type=SourceType.ANALYZED_THREAD,
        text=body,
        metadata=metadata,
    )


__all__ = [
    "BENCHMARK_COLLECTION",
    "DEFAULT_NAMESPACE",
    "MAX_DOCUMENT_CHARS",
    "Chunk",
    "IngestError",
    "LiveDocument",
    "SourceType",
    "build_document",
    "chunk_document",
    "collection_for",
    "document_from_transcript",
    "make_doc_id",
    "normalize_namespace",
]
