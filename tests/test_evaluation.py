"""COCO fixtures and official POPE quirks, independent of generation code."""

import json

import pytest

from experiments.tas_v7.evaluation import (
    ChairLexicon,
    CocoChairEvaluator,
    aggregate_caption_metrics,
    align_pope_answers,
    official_pope_prediction,
    paired_cluster_bootstrap,
    score_pope,
)


@pytest.fixture(scope="module")
def lexicon():
    return ChairLexicon()


@pytest.fixture
def evaluator(tmp_path, lexicon):
    instances = {
        "images": [{"id": 42}, {"id": 70}],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 18, "name": "dog"},
            {"id": 62, "name": "chair"},
        ],
        "annotations": [
            {"image_id": 42, "category_id": 1},
            {"image_id": 42, "category_id": 18},
            {"image_id": 70, "category_id": 62},
        ],
    }
    # A reference-mentioned bicycle is legitimate for CHAIR but MUST NOT
    # inflate our separate instance-annotation category coverage denominator.
    captions = {
        "images": [{"id": 42}, {"id": 70}],
        "annotations": [
            {"image_id": 42, "caption": "A person and dog beside a bicycle."},
            {"image_id": 70, "caption": "A chair stands in the room."},
        ],
    }
    (tmp_path / "instances_val2017.json").write_text(json.dumps(instances))
    (tmp_path / "captions_val2017.json").write_text(json.dumps(captions))
    return CocoChairEvaluator.from_annotation_dir(
        tmp_path, image_ids=[42, 70], lexicon=lexicon
    )


def test_chair_original_phrase_and_singularization_fixtures(lexicon):
    fixture = lexicon.fixture_check()
    assert fixture["passed"] == 8
    assert fixture["canonical_categories"] == 80
    assert lexicon.caption_to_words("Children watch cats and dogs.")[1] == (
        "person",
        "cat",
        "dog",
    )


def test_chair_uses_reference_objects_and_preserves_mention_multiplicity(evaluator):
    score = evaluator.score(42, "A dog, another dog, a bicycle and a cat.", seed=0)
    assert score["object_mentions"] == 4
    assert score["true_mentions"] == 3 and score["false_mentions"] == 1
    assert score["CHAIRs"] == 1 and score["CHAIRi"] == 0.25
    assert score["category_coverage"] == 0.5
    assert score["covered_categories"] == 1 and score["annotated_categories"] == 2
    assert "bicycle" in score["chair_gt_categories"]
    assert "bicycle" not in score["instance_gt_categories"]


def test_micro_chair_i_is_not_average_per_caption_rate(evaluator):
    long = evaluator.score(42, "A dog, a person and a bicycle.", seed=0)
    short = evaluator.score(70, "A cat.", seed=0)
    aggregate = aggregate_caption_metrics([long, short])
    assert aggregate["CHAIRi"] == 0.25
    assert aggregate["CHAIRs"] == 0.5
    assert aggregate["CHAIRi"] != (long["CHAIRi"] + short["CHAIRi"]) / 2


def test_empty_mentions_are_not_perfect_aggregate_hallucination_evidence(evaluator):
    score = evaluator.score(42, "This is a beautiful scene.")
    aggregate = aggregate_caption_metrics([score])
    assert score["CHAIRi"] == 0
    assert aggregate["CHAIRi"] is None
    assert aggregate["zero_object_caption_rate"] == 1
    assert aggregate["category_coverage"] == 0
    with pytest.raises(ValueError, match="empty"):
        aggregate_caption_metrics([])


def test_unknown_image_and_missing_reference_annotation_fail(evaluator, tmp_path):
    with pytest.raises(KeyError):
        evaluator.score(999, "A dog.")
    with pytest.raises(ValueError, match="both instance AND"):
        CocoChairEvaluator([tmp_path / "unused.json"], [])


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("No, there is a dog.", "no"),
        ("There is not a dog.", "no"),
        ("Yes. No dog appears.", "yes"),
        ("NO", "yes"),
        ("unclear", "yes"),
        ("", "yes"),
        ("no!", "yes"),
    ],
)
def test_official_pope_parser_keeps_original_case_and_punctuation_quirks(
    answer, expected
):
    assert official_pope_prediction(answer) == expected


def test_pope_invalid_responses_do_not_disappear_from_official_scores():
    records = [
        {"label": "yes", "answer": "Yes, a dog is visible."},
        {"label": "no", "answer": "No, no dog."},
        {"label": "no", "answer": "Unclear."},
        {"label": "yes", "answer": "not visible"},
    ]
    score = score_pope(records, variant="adversarial")
    assert (score["tp"], score["tn"], score["fp"], score["fn"]) == (1, 1, 1, 1)
    assert (
        score["accuracy"] == score["precision"] == score["recall"] == score["f1"] == 0.5
    )
    assert score["yes_rate"] == score["invalid_rate"] == 0.5
    with pytest.raises(ValueError, match="length"):
        align_pope_answers([], [{"label": "yes"}])
    with pytest.raises(ValueError, match="mismatch"):
        align_pope_answers(
            [{"question_id": 1, "answer": "yes"}], [{"question_id": 2, "label": "yes"}]
        )


def test_paired_bootstrap_retains_seed_clusters_and_ratio_metric(evaluator):
    baseline, candidate = [], []
    for seed in [0, 1, 2]:
        baseline.extend(
            [
                evaluator.score(42, "A cat.", seed=seed),
                evaluator.score(70, "A dog.", seed=seed),
            ]
        )
        candidate.extend(
            [
                evaluator.score(42, "A dog.", seed=seed),
                evaluator.score(70, "A chair.", seed=seed),
            ]
        )
    result = paired_cluster_bootstrap(baseline, candidate, repeats=150, seed=17)
    assert result["delta_candidate_minus_baseline"] == -1
    assert result["ci_low"] == result["ci_high"] == -1
    assert result["n_image_clusters"] == 2 and result["n_paired_records"] == 6
    assert result == paired_cluster_bootstrap(baseline, candidate, repeats=150, seed=17)
    with pytest.raises(ValueError, match="different"):
        paired_cluster_bootstrap(baseline, candidate[:-1], repeats=150)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_cluster_bootstrap(baseline + [baseline[0]], candidate, repeats=150)


def test_report_evaluation_preserves_identity_and_records_annotation_sources(evaluator):
    result = evaluator.evaluate(
        [{"image_id": 42, "caption": "A dog.", "method": "E0", "seed": 1}]
    )
    assert result["records"][0]["method"] == "E0"
    assert result["records"][0]["seed"] == 1
    assert result["metadata"]["annotation_files"]["instances"] == [
        "instances_val2017.json"
    ]
    assert "not AMBER COVER" in result["overall"]["coverage_scope"]
