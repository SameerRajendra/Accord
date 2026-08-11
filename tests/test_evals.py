"""Tests for the eval harnesses — the parts that must be right without a GPU.

Nothing here touches SGLang, Neon, or the network. What is covered is the
machinery that decides whether a metric is honest: citation grounding, retrieval
scoring, the small-sample statistics, and the report's treatment of missing
numbers.

The centerpiece is `test_casino_204_*`: a regression test built from the exact
fabricated citation this project documented — a real, retrieved case credited
with a "walk-away strategy" it never contained. If the grounding check stops
catching that, the project's most valuable metric has silently died, and these
tests are how that gets noticed.
"""

import math

import pytest

from data.schema import CaseDocument, Outcome, Party, Transcript, Turn
from evals._common import (
    binomial_two_sided_p,
    cramers_v,
    normalized_entropy,
    percentile,
    safe_div,
    select_transcripts,
    spearman,
    stdev,
    wilson_interval,
    write_csv,
)
from evals.agent_eval import (
    CitationCheck,
    lexical_grounding_check,
    normalize_case_id,
    resolve_citation,
)
from evals.report import NOT_MEASURED, UNDEFINED, build_report, format_value
from evals.retrieval_eval import Query, Ranking, build_queries, random_baseline, score_rankings
from evals.sentiment_eval import strategy_group

# --------------------------------------------------------------------------
# Fixtures drawn from the real corpus
# --------------------------------------------------------------------------

# Verbatim from data/processed/case_corpus.jsonl — note the corpus id carries a
# duplicated source prefix (`casino-casino-204`) while the model cited
# `casino-204`.
CASINO_204_TEXT = (
    "Case casino-casino-204 — a campsite negotiation between two campers bartering over "
    "Food, Water, and Firewood packages.\n\n"
    "agent_1 priorities: High=Food, Medium=Water, Low=Firewood agent_2 priorities: "
    "High=Food, Medium=Water, Low=Firewood\n\n"
    "Outcome: Agreement reached — agent_2 gets 0 Firewood, 2 Water, 2 Food; agent_1 gets "
    "3 Firewood, 1 Water, 1 Food. Points: agent_1=18, agent_2=18.\n\n"
    "Lesson: A balanced deal — both parties scored 18 points."
)

FABRICATED_RATIONALE = (
    "The partner has issued an ultimatum, so hold firm and prepare to walk away, as seen "
    "in [casino-204], where a walk-away strategy led to an agreement that balanced both "
    "parties' needs."
)

BREAKDOWN_CASE_TEXT = (
    "Case casino-casino-7 — a campsite negotiation between two campers bartering over "
    "Food, Water, and Firewood packages.\n\n"
    "Outcome: No agreement — the negotiation ended without a deal.\n\n"
    "Lesson: The negotiation broke down without agreement — a cautionary example of a "
    "value gap the parties couldn't bridge."
)


def _transcript(dialogue_id: str, split: str = "test", strategies=None) -> Transcript:
    strategies = strategies or [[], []]
    return Transcript(
        dialogue_id=dialogue_id,
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(party_id="agent_1", priorities={"Food": "High"}),
            Party(party_id="agent_2", priorities={"Water": "High"}),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="I need the food packages.",
                 strategies=strategies[0]),
            Turn(index=1, speaker="agent_2", text="Water matters more to me.",
                 strategies=strategies[1]),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=any(strategies),
        metadata={"split": split},
    )


# --------------------------------------------------------------------------
# Citation grounding — the regression that matters
# --------------------------------------------------------------------------


def test_casino_204_shortened_id_resolves_to_the_corpus_id():
    resolved, how = resolve_citation("casino-204", ["casino-casino-204", "casino-casino-9"])
    assert resolved == "casino-casino-204"
    # Real case, but the id as written does not exist in the corpus — counted
    # separately from fabrication, never silently normalized away.
    assert how == "normalized"


def test_casino_204_walk_away_claim_is_flagged_as_unsupported():
    tactics, contradiction, evidence = lexical_grounding_check(
        rationale=FABRICATED_RATIONALE,
        next_move="Hold firm on firewood.",
        cited_raw="casino-204",
        resolved_id="casino-casino-204",
        case_text=CASINO_204_TEXT,
        sole_citation=True,
    )
    assert "walk-away" in tactics
    # The case really did reach agreement, so the outcome half of the claim is
    # true. Flagging it would be the eval overreaching.
    assert contradiction is False
    assert "casino-204" in evidence


def test_casino_204_ultimatum_is_not_charged_to_the_cited_case():
    """Clause scoping: `ultimatum` describes the live transcript, not the case."""
    tactics, _, _ = lexical_grounding_check(
        rationale=FABRICATED_RATIONALE,
        next_move="",
        cited_raw="casino-204",
        resolved_id="casino-casino-204",
        case_text=CASINO_204_TEXT,
        sole_citation=True,
    )
    assert "ultimatum" not in tactics


def test_supported_claim_is_not_flagged():
    tactics, contradiction, evidence = lexical_grounding_check(
        rationale="Precedent [casino-casino-204] shows both parties scoring 18 points.",
        next_move="Propose an even split.",
        cited_raw="casino-casino-204",
        resolved_id="casino-casino-204",
        case_text=CASINO_204_TEXT,
        sole_citation=True,
    )
    assert tactics == []
    assert contradiction is False
    assert evidence == ""


def test_outcome_claim_contradicting_the_case_is_flagged():
    _, contradiction, evidence = lexical_grounding_check(
        rationale="In [casino-casino-7], the parties reached an agreement after trading.",
        next_move="",
        cited_raw="casino-casino-7",
        resolved_id="casino-casino-7",
        case_text=BREAKDOWN_CASE_TEXT,
        sole_citation=True,
    )
    assert contradiction is True
    assert evidence


def test_corpus_rendering_of_walk_away_counts_as_support():
    """`_outcome_line` writes `ended via walk_away`; that must count as evidence."""
    case_text = (
        "Case casino-casino-8 — a campsite negotiation.\n\n"
        "Outcome: No agreement — the negotiation ended without a deal (ended via walk_away)."
    )
    tactics, _, _ = lexical_grounding_check(
        rationale="In [casino-casino-8] the walk-away ended talks.",
        next_move="",
        cited_raw="casino-casino-8",
        resolved_id="casino-casino-8",
        case_text=case_text,
        sole_citation=True,
    )
    assert tactics == []


def test_negated_reference_is_exempt_from_tactic_flags():
    tactics, _, _ = lexical_grounding_check(
        rationale="Unlike [casino-casino-204], where nobody walked away, hold your position.",
        next_move="",
        cited_raw="casino-casino-204",
        resolved_id="casino-casino-204",
        case_text=CASINO_204_TEXT,
        sole_citation=True,
    )
    assert tactics == []


def test_invented_case_id_is_unresolved():
    resolved, how = resolve_citation("casino-999", ["casino-casino-204"])
    assert resolved == ""
    assert how == "unresolved"


def test_exact_citation_resolution():
    resolved, how = resolve_citation("casino-casino-204", ["casino-casino-204"])
    assert (resolved, how) == ("casino-casino-204", "exact")


def test_normalize_case_id_strips_citation_punctuation():
    assert normalize_case_id("[casino-204],") == "casino-204"
    assert normalize_case_id(" `Casino_204` ") == "casino-204"


def test_unresolved_and_unsupported_both_count_as_fabrication():
    unresolved = CitationCheck(
        dialogue_id="d", arm="rag", cited_raw="casino-999", resolved_case_id="",
        resolution="unresolved",
    )
    unsupported = CitationCheck(
        dialogue_id="d", arm="rag", cited_raw="casino-casino-204",
        resolved_case_id="casino-casino-204", resolution="exact",
        lexical_unsupported_tactics=["walk-away"],
    )
    clean = CitationCheck(
        dialogue_id="d", arm="rag", cited_raw="casino-casino-204",
        resolved_case_id="casino-casino-204", resolution="exact",
    )
    assert unresolved.lexically_fabricated and unresolved.judge_fabricated
    assert unsupported.lexically_fabricated
    assert not clean.lexically_fabricated and not clean.judge_fabricated


# --------------------------------------------------------------------------
# Retrieval scoring
# --------------------------------------------------------------------------


def _ranking(gold_position: int, depth: int = 5) -> Ranking:
    query = Query(query_id="q", text="text", gold_case_id="gold")
    ids = [f"other-{i}" for i in range(depth)]
    if gold_position <= depth:
        ids[gold_position - 1] = "gold"
    return Ranking(query=query, case_ids=ids, scores=[1.0 - 0.1 * i for i in range(depth)])


def test_recall_and_mrr_at_k():
    rankings = [_ranking(1), _ranking(4)]
    at_1 = score_rankings(rankings, 1)
    assert at_1["recall"] == pytest.approx(0.5)
    assert at_1["mrr"] == pytest.approx(0.5)
    at_5 = score_rankings(rankings, 5)
    assert at_5["recall"] == pytest.approx(1.0)
    assert at_5["mrr"] == pytest.approx((1.0 + 0.25) / 2)


def test_gold_outside_the_truncation_scores_zero():
    assert score_rankings([_ranking(99)], 5)["recall"] == 0.0


def test_random_baseline_is_the_closed_form():
    recall, mrr = random_baseline(1000, 10)
    assert recall == pytest.approx(0.01)
    assert mrr == pytest.approx(sum(1.0 / r for r in range(1, 11)) / 1000)


def test_random_baseline_is_tiny_next_to_any_real_retriever():
    """The number that keeps recall@10 honest on a 1,040-doc corpus."""
    recall, _ = random_baseline(1040, 10)
    assert recall < 0.01


def _corpus_doc(case_id: str, text: str, kind: str = "case") -> CaseDocument:
    return CaseDocument(case_id=case_id, source="casino", kind=kind, text=text)


def test_summary_mode_queries_with_the_document_text():
    corpus = {"casino-d1": _corpus_doc("casino-d1", "rendered case text")}
    queries, skipped = build_queries("summary", [_transcript("d1")], corpus, max_queries=0)
    assert skipped == 0
    assert len(queries) == 1
    assert queries[0].text == "rendered case text"
    assert queries[0].gold_case_id == "casino-d1"


def test_dialogue_mode_uses_the_agents_own_query_builder():
    corpus = {"casino-d1": _corpus_doc("casino-d1", "rendered case text")}
    queries, _ = build_queries("dialogue", [_transcript("d1")], corpus, max_queries=0)
    assert len(queries) == 1
    # Production concatenates the trailing text turns as "speaker: text".
    assert "agent_2: Water matters more to me." in queries[0].text
    assert queries[0].gold_case_id == "casino-d1"


def test_missing_gold_document_is_counted_not_dropped_silently():
    queries, skipped = build_queries("summary", [_transcript("d1")], {}, max_queries=0)
    assert queries == []
    assert skipped == 1


def test_strategy_mode_uses_only_single_label_utterances():
    corpus = {
        "strategy-uv-part": _corpus_doc("strategy-uv-part", "Strategy: uv-part. ...", "strategy"),
        "strategy-self-need": _corpus_doc("strategy-self-need", "Strategy: self-need.", "strategy"),
    }
    transcript = _transcript("d1", strategies=[["uv-part"], ["uv-part", "self-need"]])
    queries, _ = build_queries("strategy", [transcript], corpus, max_queries=0)
    assert len(queries) == 1  # the multi-label turn is excluded
    assert queries[0].gold_case_id == "strategy-uv-part"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_binomial_p_flags_a_sweep_and_forgives_a_coin_flip():
    assert binomial_two_sided_p(10, 10) == pytest.approx(2 / 1024)
    assert binomial_two_sided_p(5, 10) == pytest.approx(1.0)
    assert math.isnan(binomial_two_sided_p(0, 0))


def test_wilson_interval_brackets_the_estimate_and_stays_in_range():
    low, high = wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
    # A tiny sample must produce a wide interval, not a confident one.
    wide_low, wide_high = wilson_interval(2, 2)
    assert wide_high == pytest.approx(1.0)
    assert wide_low < 0.5


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 95) == pytest.approx(5.0)
    assert math.isnan(percentile([], 50))


def test_spearman_handles_monotone_and_reversed():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_entropy_exposes_a_constant_labeler():
    assert normalized_entropy([5, 5]) == pytest.approx(1.0)
    assert normalized_entropy([10, 0]) == pytest.approx(0.0)


def test_cramers_v_of_perfect_association_is_one():
    assert cramers_v([[10, 0], [0, 10]]) == pytest.approx(1.0)


def test_safe_div_and_stdev_edges():
    assert math.isnan(safe_div(1, 0))
    assert stdev([2.0]) == 0.0
    assert math.isnan(stdev([]))


def test_select_transcripts_is_deterministic_and_split_aware():
    pool = [_transcript(f"d{i}", split="test") for i in range(5)]
    pool.append(_transcript("train-1", split="train"))
    first = select_transcripts(pool, split="test", limit=3, seed=1)
    second = select_transcripts(pool, split="test", limit=3, seed=1)
    assert [t.dialogue_id for t in first] == [t.dialogue_id for t in second]
    assert len(first) == 3
    assert all(t.metadata["split"] == "test" for t in first)


def test_strategy_groups_follow_the_playbook():
    assert strategy_group(["uv-part"]) == "adversarial"
    assert strategy_group(["showing-empathy", "small-talk"]) == "cooperative"
    assert strategy_group(["self-need"]) == "distributive"
    # Contrasting labels on one turn are mixed evidence, not a label.
    assert strategy_group(["uv-part", "showing-empathy"]) == "unclassified"
    assert strategy_group([]) == "unclassified"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def test_blank_and_nan_render_differently():
    assert format_value("", "pct") == NOT_MEASURED
    assert format_value(None, "float") == NOT_MEASURED
    assert format_value("nan", "float") == UNDEFINED
    assert format_value("0.25", "pct") == "25.0%"


def test_report_marks_every_unrun_benchmark(tmp_path):
    report = build_report(tmp_path)
    assert "MISSING" in report
    # The command to produce each missing artifact must be in the report.
    for command in (
        "python -m evals.retrieval_eval",
        "python -m evals.agent_eval",
        "python -m evals.sentiment_eval",
        "python -m evals.outcome_eval",
    ):
        assert command in report
    assert "Not measured yet" in report


def test_report_prints_accuracy_next_to_its_base_rate(tmp_path):
    write_csv(
        tmp_path / "outcome.csv",
        ["n_test", "f1", "accuracy", "roc_auc", "base_rate", "accuracy_lift",
         "breakdown_recall", "tn", "fp", "fn", "tp"],
        [[102, 0.985, 0.97, 0.93, 0.96, 0.01, 0.25, 1, 3, 0, 98]],
    )
    report = build_report(tmp_path)
    assert "97.0%" in report  # accuracy
    assert "96.0%" in report  # the base rate it must be read against
    assert "`1 / 3 / 0 / 98`" in report  # raw confusion counts


def test_report_marks_a_metric_the_eval_did_not_compute(tmp_path):
    write_csv(tmp_path / "sentiment.csv", ["n_dialogues", "f1_vs_gold"], [[40, None]])
    report = build_report(tmp_path)
    assert NOT_MEASURED in report
    # Present-but-unmeasured is not the same as an unrun eval.
    assert "`sentiment.csv` | Task quality | present" in report


def test_report_lists_unmanifested_files_rather_than_hiding_them(tmp_path):
    write_csv(tmp_path / "some_side_experiment.csv", ["a"], [[1]])
    report = build_report(tmp_path)
    assert "some_side_experiment.csv" in report
