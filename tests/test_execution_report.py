"""Execution comparisons retain paired identities and distinguish drift from quality."""

from copy import deepcopy
import json

import pytest

from experiments.tas_v7.execution_report import comparison_report


def caption(
    image_id,
    *,
    method="F4",
    seed=0,
    mode="sample",
    mentions=4,
    false=1,
    words=12,
    seconds=3.0,
    tokens=(7, 8, 2),
    coverage=0.5,
):
    return {
        "image_id": image_id,
        "query_id": image_id,
        "seed": seed,
        "method": method,
        "generation_mode": mode,
        "protocol": "HARMONIZED",
        "query": "Describe this image.",
        "prompt": "USER: <image> Describe this image.",
        "caption": "A cat beside a dog.",
        "token_ids": list(tokens),
        "generated_tokens": len(tokens),
        "stop_reason": "eos_or_stop",
        "object_mentions": mentions,
        "false_mentions": false,
        "true_mentions": mentions - false,
        "CHAIRs": int(false > 0),
        "category_coverage": coverage,
        "covered_categories": 1 if coverage else 0,
        "annotated_categories": 2 if coverage is not None else 0,
        "length_words": words,
        "amortized_generation_seconds": seconds,
        "batch_latency_seconds": seconds * 4,
        "peak_memory_bytes": 1000,
    }


def report(reference, candidate):
    return comparison_report(
        reference, candidate, bootstrap_repeats=100, bootstrap_seed=17
    )


def test_pairs_reordered_outputs_and_reports_token_drift_without_equivalence():
    before = [caption(1, false=0), caption(2)]
    after = [
        caption(2, false=0, words=9, tokens=(7, 9, 2)),
        caption(1, false=1, words=14),
    ]
    after[0]["caption"] = "A dog."
    after[0]["stop_reason"] = "max_new_tokens"
    original = deepcopy((before, after))
    result = report(before, after)
    row = result["summary"][0]
    assert result["status"] == "descriptive_only"
    assert "No equivalence claim" in result["interpretation"]
    assert row["exact_token_match_rate"] == row["exact_caption_match_rate"] == 0.5
    assert row["chair_status_transitions"] == {
        "0->0": 0,
        "0->1": 1,
        "1->0": 1,
        "1->1": 0,
    }
    assert row["paired_length_changes"] == {
        "shorter": 1,
        "same": 0,
        "longer": 1,
        "mean_candidate_minus_reference": -0.5,
    }
    assert (
        row["candidate"]["token_limit_rate"]
        == row["candidate"]["eos_or_stop_rate"]
        == 0.5
    )
    assert row["paired"]["CHAIRs"]["n_paired_records"] == 2
    assert (before, after) == original
    json.dumps(result, allow_nan=False)


def test_cost_uses_amortized_batch_time_once_and_preserves_micro_chair():
    before = [
        caption(1, mentions=10, false=1, seconds=4),
        caption(2, mentions=2, false=1, seconds=4),
    ]
    after = [
        caption(1, mentions=10, false=0, seconds=1),
        caption(2, mentions=2, false=1, seconds=1),
    ]
    for row in before:
        row["batch_latency_seconds"] = 4  # B=1 reference
    for row in after:
        row["batch_latency_seconds"] = 100  # Must not enter throughput arithmetic.
    row = report(before, after)["summary"][0]
    assert row["reference"]["amortized_generation_seconds"] == 8
    assert row["candidate"]["amortized_generation_seconds"] == 2
    assert row["speedup"] == 4
    assert row["reference"]["CHAIRi"] == pytest.approx(2 / 12)
    assert row["candidate"]["CHAIRi"] == pytest.approx(1 / 12)
    assert row["CHAIRi_delta"] == pytest.approx(-1 / 12)


def test_zero_denominators_stay_undefined_and_missing_tokens_do_not_look_identical():
    before = [caption(1, mentions=0, false=0, coverage=None)]
    after = deepcopy(before)
    del after[0]["token_ids"]
    row = report(before, after)["summary"][0]
    for metric in ("CHAIRi", "category_coverage"):
        assert row["reference"][metric] is None
        assert row["paired"][metric]["status"] == "undefined"
        assert "denominator" in row["paired"][metric]["reason"]
        assert row["paired"][metric]["ci_low"] is None
    assert row["exact_token_match_rate"] is None
    assert row["token_pairs_compared"] == 0
    assert row["exact_caption_match_rate"] == 1


def test_methods_modes_and_three_seeds_remain_separate_with_image_clusters():
    before = [
        caption(image, method=method, seed=seed, mode=mode)
        for method in ("E1-H", "F4")
        for mode in ("sample", "greedy")
        for seed in (0, 1, 2)
        for image in (1, 2)
    ]
    after = deepcopy(before)
    result = report(before, after)
    assert len(result["summary"]) == 4
    for row in result["summary"]:
        assert row["n_pairs"] == row["paired"]["CHAIRi"]["n_paired_records"] == 6
        assert row["n_image_clusters"] == 2
        assert (
            row["paired"]["CHAIRi"]["resampling_unit"]
            == "image; all queries/seeds retained"
        )
        assert row["speedup"] == row["exact_token_match_rate"] == 1
    assert before == after


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_id", 2),
        ("query_id", 2),
        ("method", "E1-H"),
        ("seed", 3),
        ("generation_mode", "greedy"),
        ("query", "Other question"),
        ("prompt", "Other prompt"),
        ("protocol", "LEGACY"),
        ("generated_tokens", 100),
        ("annotated_categories", 3),
    ],
)
def test_mismatched_pairs_or_generation_metadata_are_rejected(field, value):
    before = [caption(1)]
    after = deepcopy(before)
    after[0][field] = value
    with pytest.raises(ValueError):
        report(before, after)


@pytest.mark.parametrize(
    "field", ["image_id", "query_id", "method", "seed", "generation_mode", "prompt"]
)
def test_missing_identity_fields_are_rejected(field):
    before = [caption(1)]
    after = deepcopy(before)
    del after[0][field]
    with pytest.raises(ValueError):
        report(before, after)


def test_missing_duplicate_and_nonfinite_records_are_rejected():
    before = [caption(1), caption(2)]
    for after in ([], before[:1], before + [before[0]]):
        with pytest.raises(ValueError):
            report(before, after)
    for field in (
        "amortized_generation_seconds",
        "false_mentions",
        "category_coverage",
    ):
        after = deepcopy(before)
        after[0][field] = float("nan")
        with pytest.raises(ValueError):
            report(before, after)


def test_unavailable_or_zero_costs_do_not_fabricate_speedup():
    before = [caption(1)]
    after = [caption(1, seconds=0)]
    assert report(before, after)["summary"][0]["speedup"] is None
    del after[0]["amortized_generation_seconds"]
    row = report(before, after)["summary"][0]
    assert row["candidate"]["amortized_generation_seconds"] is None
    assert row["speedup"] is None
