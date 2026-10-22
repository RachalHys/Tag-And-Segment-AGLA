"""Annotation provenance and within-method diagnostic comparisons."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import test_study
from experiments.tas_v7.methods import method_by_id
from experiments.tas_v7.study_diagnostics import (
    paired_intervention_report,
    run_diagnostics,
)
from tas_runtime.perception import PerceptionResult

study = test_study.study


def scored(method, image_id, seed, false):
    return {
        "method": method,
        "protocol": "HARMONIZED",
        "generation_mode": "sample",
        "image_id": image_id,
        "query_id": image_id,
        "seed": seed,
        "object_mentions": 2,
        "false_mentions": false,
        "true_mentions": 2 - false,
        "CHAIRs": int(false > 0),
        "covered_categories": 1,
        "annotated_categories": 1,
        "category_coverage": 1.0,
        "length_words": 4,
        "perception_metadata": {
            "intervention_details": {"applied": image_id == 1},
            "union_mask_changed": image_id == 1,
        },
    }


def test_paired_diagnostics_keep_methods_separate_and_cluster_all_seeds():
    reference = [
        scored(method, image, seed, false)
        for method, false in (("E1-H", 1), ("F4", 0))
        for image in (1, 2)
        for seed in (0, 1)
    ]
    changed = [
        scored(
            record["method"],
            record["image_id"],
            record["seed"],
            0 if record["method"] == "E1-H" else 1,
        )
        for record in reversed(reference)
    ]
    report = paired_intervention_report(reference, {"D-MASK": changed}, repeats=100)
    assert report["reference"] == "D-PRED" and "no causal" in report["scope"]
    for comparison in report["comparisons"]:
        expected = -0.5 if comparison["method"] == "E1-H" else 0.5
        chair = next(row for row in comparison["paired"] if row["metric"] == "CHAIRi")
        assert chair["delta_candidate_minus_baseline"] == expected
        assert chair["ci_low"] == chair["ci_high"] == expected
        assert chair["n_image_clusters"] == 2 and chair["n_paired_records"] == 4
        assert comparison["applied_image_ids"] == [1]
        assert comparison["mask_changed_image_ids"] == [1]
        assert comparison["intervention_metrics"]["n_captions"] == 4


@pytest.mark.parametrize("change", ["missing", "duplicate", "query", "protocol", "prompt", "annotation"])
def test_paired_diagnostics_reject_misaligned_or_mixed_inputs(change):
    reference = [scored("F4", image, 0, 0) for image in (1, 2)]
    candidate = [dict(row) for row in reference]
    if change == "missing":
        candidate.pop()
    elif change == "duplicate":
        candidate.append(candidate[0])
    elif change == "query":
        candidate[0]["query_id"] = "another-query"
    elif change == "prompt":
        candidate[0]["prompt"] = "a different prompt"
    elif change == "annotation":
        candidate[0]["annotated_categories"] = 99
    else:
        candidate[0]["protocol"] = "LEGACY"
    with pytest.raises(ValueError, match="Diagnostic"):
        paired_intervention_report(reference, {"D-MASK": candidate}, repeats=100)


def test_undefined_diagnostic_metrics_are_recorded_without_dropping_images():
    record = {
        **scored("F4", 1, 0, 0),
        "object_mentions": 0,
        "true_mentions": 0,
        "annotated_categories": 0,
        "covered_categories": 0,
        "category_coverage": None,
    }
    report = paired_intervention_report([record], {"D-MISS": [record]}, repeats=100)
    comparison = report["comparisons"][0]
    assert comparison["reference_metrics"]["n_captions"] == 1
    paired = {result["metric"]: result for result in comparison["paired"]}
    assert paired["CHAIRi"]["status"] == "undefined"
    assert paired["category_coverage"]["status"] == "undefined"
    assert paired["length_words"]["delta_candidate_minus_baseline"] == 0


def test_empty_diagnostics_cannot_be_reported_as_a_completed_comparison():
    with pytest.raises(ValueError, match="requires reference"):
        paired_intervention_report([], {"D-MASK": []}, repeats=100)


@pytest.mark.parametrize("batched_execution", [False, True])
def test_diagnostics_report_perfect_mask_match_and_unmatched_noop(study, monkeypatch, batched_execution):
    run, rows = study
    if batched_execution:
        from test_batched_study import batched

        run.config = batched(run.config)
        run.policy = run.config.fixed_execution_policy
        run.runtime.set_attention("sdpa")
    monkeypatch.setattr(run, "rows", lambda cohort: rows)
    original_perception = run.perception_for

    def perception(row):
        result, key, hit = original_perception(row)
        if row["image_id"] == rows[1]["image_id"]:
            result.metadata["proposed_labels"] = ["object"]
        return result, key, hit

    monkeypatch.setattr(run, "perception_for", perception)
    sam_calls = []

    def rerun(image, *, inventory_override, boxes_override):
        sam_calls.append(boxes_override)
        mask = np.zeros((image.height, image.width), bool)
        mask[2:7, 3:10] = True
        return PerceptionResult(
            mask, mask[None], boxes_override, np.ones(1), {"sam_predicted_iou": [0.9]}
        )

    run.perception = SimpleNamespace(run=rerun, close=lambda: None)
    run_diagnostics(run, method_by_id("F4"))
    assert len(sam_calls) == 1
    directory = run.persistent_run / "diagnostics"
    audit = [
        json.loads(line)
        for line in (directory / "perception-audit.jsonl").read_text().splitlines()
    ]
    match = audit[0]["matches"][0]
    assert match["box_iou"] == match["mask_iou"] == 1.0
    assert (
        match["predicted_mask_area_pixels"]
        == match["annotation_mask_area_pixels"]
        == 35
    )
    assert match["mask_intersection_pixels"] == match["mask_union_pixels"] == 35
    assert audit[1]["matches"] == []

    for variant in ("D-MASK", "D-BOX", "D-MISS"):
        path = next(run.persistent_run.glob(f"diagnostics-{variant}-*/captions.jsonl"))
        captions = [json.loads(line) for line in path.read_text().splitlines()]
        for caption in captions:
            metadata = caption["perception_metadata"]
            details = metadata["intervention_details"]
            applied = caption["image_id"] == rows[0]["image_id"]
            assert details["applied"] == applied
            assert details["status"] == ("applied" if applied else "no_match")
            assert details["instance_count_before"] == 1
            after = 0 if applied and variant == "D-MISS" else 1
            assert details["instance_count_after"] == after
            assert metadata["same_instance_count_preserved"] == (after == 1)
            assert metadata["union_mask_changed"] == (after == 0)
            if applied and variant == "D-MISS":
                assert details["verified_present"] is False
                assert "verified_present_annotation_id" not in metadata
    paired = json.loads((directory / "paired-interventions.json").read_text())
    assert len(paired["comparisons"]) == 9
    assert all(
        row["reference_metrics"]["n_images"] == 2 for row in paired["comparisons"]
    )
    completed = json.loads((directory / "diagnostics.json").read_text())
    assert completed["paired_interventions"] == "paired-interventions.json"
    assert completed["caption_execution_protocol"] == run.config.execution_protocol
    assert completed["caption_execution_policy"] == run.policy
    assert completed["probe_execution"]["batch_images"] == 1
    assert completed["probe_execution"]["views"] == ["ORIG", "OBJ-H", "BH", "FG-U"]
    influence = [json.loads(line) for line in (directory / "fixed-prefix-influence.jsonl").read_text().splitlines()]
    assert all(row["probe_execution"] == completed["probe_execution"] for row in influence)
    run.close()
