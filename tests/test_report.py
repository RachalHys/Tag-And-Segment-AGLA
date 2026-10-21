"""Scientific selection and timing regressions, independent of model outputs."""

import pytest

from experiments.tas_v7.report import select_finalists, summarize, write_report
from tas_runtime.config import StudyConfig


def caption(method, *, coverage=0.9, false=10, mentions=100, image_id=1, length=100):
    return {
        "image_id": image_id,
        "query_id": image_id,
        "seed": 0,
        "method": method,
        "protocol": "HARMONIZED",
        "generation_mode": "sample",
        "object_mentions": mentions,
        "false_mentions": false,
        "true_mentions": mentions - false,
        "CHAIRs": int(false > 0),
        "CHAIRi": false / mentions,
        "category_coverage": coverage,
        "covered_categories": round(coverage * 50),
        "annotated_categories": 50,
        "length_words": length,
    }


def test_shortlist_rejects_coverage_collapse_and_dominated_candidate():
    records = [
        caption("E1-H"),
        caption("E2", coverage=0.02, false=0),
        caption("E3", coverage=0.02, false=1),
        caption("F4", false=2),
    ]
    result = select_finalists(records, coverage_tolerance=0.0)
    assert result["selected"] == ["F4"]
    assert result["coverage_tolerance"] == 0.0
    by_method = {row["method"]: row for row in result["candidates"]}
    assert not by_method["E2"]["coverage_pass"]
    assert by_method["E3"]["pareto_dominated"]


def test_shortlist_uses_absolute_percentage_points_and_matched_protocol():
    records = [caption("E1-H"), caption("F4", coverage=0.89, false=2)]
    assert select_finalists(records, coverage_tolerance=0.0)["selected"] == []
    assert select_finalists(records, coverage_tolerance=0.01)["selected"] == ["F4"]
    legacy = {**caption("E1"), "protocol": "LEGACY"}
    with pytest.raises(ValueError, match="E1-H"):
        select_finalists([legacy, records[1]], coverage_tolerance=0.01)


def test_approved_two_point_rule_includes_boundary_and_excludes_larger_loss():
    baseline = caption("E1-H", coverage=0.6)
    assert StudyConfig("unused").coverage_tolerance == 0.02
    assert select_finalists([baseline, caption("F4", coverage=0.58, false=2)])[
        "selected"
    ] == ["F4"]
    assert (
        select_finalists([baseline, caption("F4", coverage=0.579, false=2)])["selected"]
        == []
    )


def test_no_improvement_or_pareto_dominated_policy_does_not_force_a_finalist():
    tied = [caption("E1-H"), caption("F4")]
    assert select_finalists(tied)["selected"] == []
    dominated = [
        caption("E1-H"),
        caption("F0", false=1),
        caption("F4", false=2),
    ]
    assert select_finalists(dominated)["selected"] == []


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_coverage_rule_rejected_before_execution(value):
    with pytest.raises(ValueError, match="coverage"):
        StudyConfig("unused", coverage_tolerance=value)
    with pytest.raises(ValueError, match="coverage"):
        select_finalists([], coverage_tolerance=value)


def test_batched_latency_is_separate_from_amortized_throughput():
    records = [
        {
            "batch_latency_seconds": 80.0,
            "amortized_generation_seconds": 10.0,
            "generated_tokens": 100,
        }
        for _ in range(8)
    ]
    summary = summarize(records)
    assert summary["p50_batch_latency_seconds"] == 80.0
    assert summary["p95_batch_latency_seconds"] == 80.0
    assert summary["p50_amortized_seconds_per_image"] == 10.0
    assert summary["captions_per_hour_warm"] == 360.0
    assert summary["generation_seconds"] == 80.0


def test_old_amortized_results_do_not_fabricate_request_latency():
    summary = summarize([{"generation_seconds": 10.0, "generated_tokens": 100}])
    assert summary["p50_batch_latency_seconds"] is None
    assert summary["p95_batch_latency_seconds"] is None
    assert summary["p50_amortized_seconds_per_image"] == 10.0


def test_report_equal_length_comparison_pairs_images_and_seeds(tmp_path):
    baseline = [caption("E1-H", image_id=i, length=20) for i in (1, 2)]
    candidate = [
        caption("F4", image_id=1, length=20, false=0),
        caption("F4", image_id=2, length=5, false=0),
    ]
    report = write_report(tmp_path, baseline + candidate, bootstrap_repeats=50)
    result = report["equal_length_comparisons"][0]
    assert result["paired_records"] == 2 and result["matched_records"] == 1
    assert result["baseline"]["CHAIRi"] == 0.1
    assert result["candidate"]["CHAIRi"] == 0.0
    assert result["matched_image_ids"] == ["1"]
    assert "descriptive" in result["interpretation"]


def test_equal_length_report_handles_no_matching_captions(tmp_path):
    report = write_report(
        tmp_path,
        [caption("E1-H", length=20), caption("F4", length=5)],
        bootstrap_repeats=50,
    )
    comparison = report["equal_length_comparisons"][0]
    assert comparison["matched_records"] == 0
    assert comparison["baseline"] is None and comparison["candidate"] is None
