"""Scientific invariant fixtures; no model checkpoint or GPU is required."""

from dataclasses import replace
import math

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.tas_v7.diagnostics import (
    InstanceEvidence,
    add_verified_false_region,
    fixed_prefix_influence,
    js_from_logits,
    remove_verified_instance,
    replace_same_instance_evidence,
    swap_labels_only,
    teacher_forced_phrase_log_probability,
    union_instances,
)
from experiments.tas_v7.methods import (
    build_controls,
    fuse_logits,
    harmonized_baselines,
    initial_methods,
    method_by_id,
    plausibility_mask,
)
from experiments.tas_v7.views import (
    area_matched_random_mask,
    create_views,
    gaussian_full_image,
    legacy_object_view,
    processor_geometry_canvas,
)


def distribution_logits():
    return {
        "ORIG": torch.tensor([[0.7, 0.2, 0.1]]).log(),
        "OBJ-H": torch.tensor([[0.1, 0.3, 0.6]]).log(),
        "FG-U": torch.tensor([[0.2, 0.4, 0.4]]).log(),
    }


def test_initial_method_table_and_protocol_identities():
    methods = initial_methods()
    assert [method.id for method in methods] == [
        "E0",
        "C0",
        "C1",
        "C2",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "F0",
        "F1",
        "F2",
        "F3",
        "F4",
    ]
    assert method_by_id("E0").support is False
    assert method_by_id("E1").protocol == "LEGACY"
    assert method_by_id("F4").views[-1] == "FG-U"
    assert [method.id for method in harmonized_baselines()] == ["E0-H", "E1-H"]
    assert all(method.protocol == "HARMONIZED" for method in harmonized_baselines())


def test_probability_mixtures_have_expected_weights_not_logit_average():
    logits = distribution_logits()
    assert torch.allclose(
        fuse_logits(logits, method_by_id("F2")).exp(),
        torch.tensor([[0.47, 0.25, 0.28]]),
        atol=1e-6,
    )
    assert torch.allclose(
        fuse_logits(logits, method_by_id("F0")).exp(),
        torch.tensor([[0.46, 0.24, 0.30]]),
        atol=1e-6,
    )


def test_selective_penalty_matches_independent_numpy_calculation():
    logits = {
        "ORIG": torch.tensor([[0.8, 0.15, 0.05]]).log(),
        "OBJ-H": torch.tensor([[0.7, 0.25, 0.05]]).log(),
        "FG-U": torch.tensor([[0.01, 0.04, 0.95]]).log(),
    }
    p0, po, pc = (
        array.numpy()[0]
        for array in [logits["ORIG"].exp(), logits["OBJ-H"].exp(), logits["FG-U"].exp()]
    )
    penalty = np.minimum(
        np.maximum(np.log(pc) - np.maximum(np.log(p0), np.log(po)) - 0.2, 0), 1.5
    )
    q = (0.75 * po + 0.25 * pc) * np.exp(-penalty)
    expected = 0.6 * p0 + 0.4 * q / q.sum()
    actual = fuse_logits(logits, method_by_id("F4")).exp().numpy()[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual[2] < fuse_logits(logits, method_by_id("F2")).exp()[0, 2]
    assert (actual >= 0.6 * p0 - 1e-7).all()


@pytest.mark.parametrize("method_id", ["F0", "F2", "F4"])
def test_eta_zero_is_original_and_identical_views_do_not_change_distribution(method_id):
    logits = distribution_logits()
    zero = fuse_logits(logits, replace(method_by_id(method_id), eta=0)).softmax(-1)
    assert torch.allclose(zero, logits["ORIG"].softmax(-1))
    identical = {view: logits["ORIG"] for view in ["ORIG", "OBJ-H", "FG-U"]}
    output = fuse_logits(identical, method_by_id(method_id)).softmax(-1)
    assert torch.allclose(output, logits["ORIG"].softmax(-1), atol=1e-6)


def test_kappa_zero_recovers_mixture_and_large_logits_stay_finite():
    logits = distribution_logits()
    assert torch.equal(
        fuse_logits(logits, replace(method_by_id("F4"), kappa=0)),
        fuse_logits(logits, method_by_id("F2")),
    )
    extreme = {
        "ORIG": torch.tensor([[10000.0, -10000.0, 0.0]], dtype=torch.float16),
        "OBJ-H": torch.tensor([[-10000.0, 10000.0, 0.0]], dtype=torch.float16),
        "FG-U": torch.tensor([[0.0, -10000.0, 10000.0]], dtype=torch.float16),
    }
    score = fuse_logits(extreme, method_by_id("F4"))
    assert score.dtype == torch.float32
    assert torch.isfinite(score).all()
    assert torch.allclose(score.exp().sum(-1), torch.ones(1))


def test_legacy_tag_keeps_native_precision_and_operation_order():
    logits = {
        "ORIG": torch.tensor([[3.333, -0.4998, 15.01]], dtype=torch.float16),
        "OBJ-H": torch.tensor([[-0.7777, 3.141, -7.511]], dtype=torch.float16),
    }
    original = {view: tensor.clone() for view, tensor in logits.items()}
    expected = logits["ORIG"] + torch.tensor(2.0, dtype=torch.float16) * logits["OBJ-H"]
    actual = fuse_logits(logits, method_by_id("E1"))
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)
    assert all(torch.equal(logits[view], original[view]) for view in logits)
    harmonized = fuse_logits(logits, harmonized_baselines()[1])
    assert harmonized.dtype == torch.float32
    assert torch.equal(harmonized, logits["ORIG"].float() + 2 * logits["OBJ-H"].float())


def test_linear_gamma_zero_and_duplicate_context_recover_tag_in_fp32():
    logits = distribution_logits()
    expected = logits["ORIG"] + 2 * logits["OBJ-H"]
    assert torch.allclose(
        fuse_logits(logits, replace(method_by_id("E4"), gamma=0)), expected
    )
    duplicate = build_controls(method_by_id("E4"))[1]
    assert duplicate.context_view == "ORIG" and duplicate.views == ("ORIG", "OBJ-H")
    assert torch.allclose(fuse_logits(logits, duplicate), expected)


def test_duplicate_context_control_matches_thirty_percent_object_not_f0():
    logits = distribution_logits()
    duplicate = build_controls(method_by_id("F2"))[1]
    actual = fuse_logits(logits, duplicate).exp()
    expected = 0.7 * logits["ORIG"].exp() + 0.3 * logits["OBJ-H"].exp()
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual, fuse_logits(logits, method_by_id("F0")).exp())


def test_context_only_controls_remove_object_evidence_and_keep_declared_scale():
    logits = distribution_logits()
    base = method_by_id("E4")
    linear = replace(
        base,
        id="context-linear",
        fusion="linear_context",
        views=("ORIG", "FG-U"),
        context="FG-U",
    )
    score = fuse_logits(logits, linear)
    assert torch.allclose(score, 2.5 * logits["ORIG"] + 0.5 * logits["FG-U"])
    mixture = replace(linear, id="context-mixture", fusion="mixture_context")
    assert torch.allclose(
        fuse_logits(logits, mixture).exp(),
        0.9 * logits["ORIG"].exp() + 0.1 * logits["FG-U"].exp(),
    )
    changed_object = {**logits, "OBJ-H": torch.zeros_like(logits["OBJ-H"])}
    assert torch.equal(fuse_logits(changed_object, linear), score)
    selective = replace(mixture, id="context-selective", fusion="selective_context")
    expected = fuse_logits({**logits, "OBJ-H": logits["ORIG"]}, method_by_id("F4"))
    assert torch.equal(fuse_logits(logits, selective), expected)


def test_execution_audit_rejects_a_changed_plausibility_boundary():
    from experiments.tas_v7.audit import compare_snapshots

    first = [{"ORIG": torch.tensor([[0.0, math.log(0.5) + 0.001]])}]
    second = [{"ORIG": torch.tensor([[0.0, math.log(0.5) - 0.001]])}]
    with pytest.raises(AssertionError, match="plausibility support"):
        compare_snapshots(first, second)


def test_support_edges_and_native_vs_fp32_boundary():
    logits = torch.tensor([[4.0, 2.0, -5.0], [1.0, 1.0, -1.0]])
    assert plausibility_mask(logits, 0).all()
    assert plausibility_mask(logits, 1).sum(-1).tolist() == [1, 2]
    boundary = torch.tensor([[10.0, 9.3046875]], dtype=torch.float16)
    assert plausibility_mask(boundary, 0.5, "legacy")[0, 1]
    assert not plausibility_mask(boundary, 0.5, "harmonized")[0, 1]
    masked = logits.masked_fill(~plausibility_mask(logits, 0.5), -float("inf"))
    assert masked.softmax(-1)[0, 1:].eq(0).all()
    with pytest.raises(ValueError):
        plausibility_mask(logits, -0.1)


@pytest.fixture
def rgb_fixture():
    cells = (np.indices((40, 48)).sum(0) % 2 * 255).astype(np.uint8)
    pixels = np.stack([cells, np.roll(cells, 1, axis=1), np.full_like(cells, 100)], -1)
    return Image.fromarray(pixels, mode="RGB")


def test_fgu_keeps_background_exact_and_uses_full_image_gaussian(rgb_fixture):
    mask = np.zeros((40, 48), bool)
    mask[10:30, 12:36] = True
    views = create_views(rgb_fixture, mask)
    original = np.asarray(rgb_fixture)
    blurred = np.asarray(views["FG-U"])
    full = np.asarray(gaussian_full_image(rgb_fixture, sigma=0.04 * 40))
    np.testing.assert_array_equal(blurred[~mask], original[~mask])
    np.testing.assert_array_equal(blurred[mask], full[mask])
    assert not np.array_equal(blurred[mask], original[mask])
    assert np.asarray(views["OBJ-H"])[~mask].max() == 0
    assert np.asarray(views["BH"])[mask].max() == 0


def test_view_empty_full_zero_sigma_and_explicit_legacy_fallback(rgb_fixture):
    empty, full = np.zeros((40, 48), bool), np.ones((40, 48), bool)
    original = np.asarray(rgb_fixture)
    views = create_views(rgb_fixture, empty)
    assert not np.asarray(views["OBJ-H"]).any()
    np.testing.assert_array_equal(views["FG-U"], original)
    np.testing.assert_array_equal(views["BH"], original)
    views = create_views(rgb_fixture, full)
    np.testing.assert_array_equal(views["OBJ-H"], original)
    assert not np.asarray(views["BH"]).any()
    np.testing.assert_array_equal(views["FG-U"], gaussian_full_image(rgb_fixture, 1.6))
    np.testing.assert_array_equal(
        create_views(rgb_fixture, full, sigma_ratio=0)["FG-U"], original
    )
    np.testing.assert_array_equal(
        legacy_object_view(rgb_fixture, empty, fallback=True), original
    )
    assert not np.asarray(legacy_object_view(rgb_fixture, empty, fallback=False)).any()
    with pytest.raises(ValueError, match="binary|boolean"):
        create_views(rgb_fixture, np.full((40, 48), 0.5))


@pytest.mark.parametrize("size", [(37, 53), (53, 37), (32, 64), (61, 59)])
def test_processor_canvas_matches_actual_clip_pixel_values(size):
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    random = np.random.default_rng(42)
    image = Image.fromarray(
        random.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    )
    mask = np.zeros((size[1], size[0]), bool)
    mask[3:15, 5:20] = True
    canvas, transformed_mask, metadata = processor_geometry_canvas(
        image, mask, processor
    )
    normal = processor(images=image, return_tensors="np")["pixel_values"]
    prepared = processor(
        images=canvas, do_resize=False, do_center_crop=False, return_tensors="np"
    )["pixel_values"]
    np.testing.assert_array_equal(normal, prepared)
    assert transformed_mask.dtype == bool and transformed_mask.shape == (32, 32)
    assert metadata["normalization_applied"] is False


def test_random_intervention_exact_area_and_reproducibility():
    for area in [0, 1, 7, 25, 89, 120]:
        mask = np.zeros((10, 12), bool)
        mask.flat[:area] = True
        a = area_matched_random_mask(mask, np.random.default_rng(7))
        b = area_matched_random_mask(mask, np.random.default_rng(7))
        assert a.sum() == area
        np.testing.assert_array_equal(a, b)


def test_same_instance_controls_do_not_add_all_annotation_objects():
    mask = np.zeros((5, 6), bool)
    mask[1:3, 1:3] = True
    predicted = [InstanceEvidence("p1", "dog", (1, 1, 3, 3), mask)]
    better = np.zeros((5, 6), bool)
    better[1:4, 1:4] = True
    annotations = {
        "gt1": InstanceEvidence("gt1", "dog", (1, 1, 4, 4), better),
        "gt2": InstanceEvidence("gt2", "person", (0, 0, 1, 1), ~better),
    }
    replaced, metadata = replace_same_instance_evidence(
        predicted, annotations, {"p1": "gt1"}
    )
    assert len(replaced) == 1 and replaced[0].label == "dog"
    np.testing.assert_array_equal(union_instances(replaced, mask.shape), better)
    boxed, metadata = replace_same_instance_evidence(
        predicted, annotations, {"p1": "gt1"}, component="box"
    )
    assert boxed[0].mask is None and metadata["requires_sam_rerun"]
    with pytest.raises(ValueError, match="segmentation"):
        union_instances(boxed, mask.shape)
    swapped, meta = swap_labels_only(predicted, {"p1": "elephant"})
    np.testing.assert_array_equal(union_instances(swapped, mask.shape), mask)
    assert meta["expected_pixel_generation_change"] is False


def test_presence_absence_interventions_require_independent_verification():
    evidence = InstanceEvidence("p1", "dog", (0, 0, 2, 2), np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="verified"):
        remove_verified_instance([evidence], "p1", verified_present_ids=[])
    removed, _ = remove_verified_instance([evidence], "p1", verified_present_ids=["p1"])
    assert removed == []
    with pytest.raises(ValueError, match="absence"):
        add_verified_false_region(
            [], evidence, absence_verified=False, verification_note="COCO missing label"
        )


def test_influence_is_symmetric_bounded_and_rejects_different_prefixes():
    first = torch.tensor([[100.0, -100.0]])
    second = -first
    assert torch.allclose(js_from_logits(first, first), torch.zeros(1))
    assert torch.allclose(js_from_logits(first, second), torch.tensor([math.log(2)]))
    assert torch.equal(js_from_logits(first, second), js_from_logits(second, first))
    views = {view: first for view in ["ORIG", "OBJ-H", "BH", "FG-U"]}
    prefixes = {view: [1, 4, 8] for view in views}
    influence = fixed_prefix_influence(views, prefix_ids_by_view=prefixes)
    assert influence["S_blur"].item() == 0
    prefixes["FG-U"] = [1, 4, 9]
    with pytest.raises(ValueError, match="identical"):
        fixed_prefix_influence(views, prefix_ids_by_view=prefixes)


def test_phrase_score_uses_all_subwords_and_aligned_teacher_forcing():
    logits = torch.tensor([[0.8, 0.2], [0.1, 0.9]]).log()
    score = teacher_forced_phrase_log_probability(logits, [0, 1])
    assert score.item() == pytest.approx(math.log(0.8 * 0.9))
    batched = teacher_forced_phrase_log_probability(logits.repeat(2, 1, 1), [0, 1])
    assert batched.shape == (2,) and torch.equal(batched[0], score)
    with pytest.raises(ValueError, match="aligned"):
        teacher_forced_phrase_log_probability(logits, [0])
