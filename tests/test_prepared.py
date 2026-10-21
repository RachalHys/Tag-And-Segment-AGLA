"""View selection preserves the baseline geometry across method controls."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import CLIPImageProcessor

from experiments.tas_v7.methods import (
    build_controls,
    harmonized_baselines,
    method_by_id,
)
from experiments.tas_v7.prepared import (
    method_features,
    prepare_feature_batch,
    prepare_image,
)
from experiments.tas_v7.views import area_matched_random_mask, legacy_object_view
from tas_runtime.decoding import record_seed
from tas_runtime.perception import PerceptionResult
from tas_runtime.runtime import LlavaRuntime


class PixelRuntime:
    """Use the actual CLIP preprocessing; retain pixels as test features."""

    pixels = LlavaRuntime.pixels

    def __init__(self):
        self.image_processor = CLIPImageProcessor(
            size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
        )

    def features(self, images):
        return images.flatten(2).transpose(1, 2)


@pytest.fixture
def image_case(tmp_path):
    y, x = np.indices((49, 71))
    image = Image.fromarray(
        np.stack((x * 3 % 256, y * 5 % 256, (x + y) % 2 * 255), -1).astype(np.uint8)
    )
    path = tmp_path / "image.png"
    image.save(path)
    mask = np.zeros((49, 71), bool)
    mask[9:35, 25:49] = True
    return path, image, mask


def prepare_case(image_case, *, empty=False, fallback=False, seed=0):
    path, image, mask = image_case
    if empty:
        mask = np.zeros_like(mask)
    perception = PerceptionResult(
        mask,
        mask[None],
        np.empty((0, 4), np.float32),
        np.empty(0, np.float32),
        {"legacy_fallback": fallback},
    )
    runtime = PixelRuntime()
    prepared = prepare_image(
        runtime,
        path,
        perception,
        SimpleNamespace(sigma_ratio=0.04),
        random_seed=seed,
    )
    return runtime, prepared, image, mask


@pytest.mark.parametrize("fallback", [False, True])
def test_renamed_and_filter_off_baselines_preserve_geometry(image_case, fallback):
    runtime, prepared, _, _ = prepare_case(
        image_case, empty=fallback, fallback=fallback
    )
    legacy = method_by_id("E1")
    renamed = replace(legacy, id="renamed-baseline")
    nofilter = build_controls(legacy)[0]
    harmonized = harmonized_baselines()[1]
    harmonized_nofilter = build_controls(harmonized)[0]
    features = prepare_feature_batch(
        runtime,
        [prepared],
        [legacy, renamed, nofilter, harmonized, harmonized_nofilter],
    )
    native_object = method_features(features, legacy)["OBJ-H"]
    assert torch.equal(native_object, method_features(features, renamed)["OBJ-H"])
    assert torch.equal(native_object, method_features(features, nofilter)["OBJ-H"])
    canvas_object = method_features(features, harmonized)["OBJ-H"]
    assert torch.equal(
        canvas_object, method_features(features, harmonized_nofilter)["OBJ-H"]
    )
    # Raw masking before bicubic resize differs from a hard mask on the canvas;
    # a legacy empty-inventory fallback also remains different from hard black.
    assert not torch.equal(native_object, canvas_object)


def test_explicit_black_and_random_controls_preserve_their_geometry(image_case):
    seed = 17
    runtime, prepared, image, mask = prepare_case(image_case, seed=seed)
    legacy = method_by_id("E1")
    black = replace(legacy, id="black-object", context="OBJ-H", control="black_context")
    legacy_random = replace(legacy, id="random-legacy", control="random_region")
    harmonized_random = replace(
        harmonized_baselines()[1], id="random-canvas", control="random_region"
    )
    features = prepare_feature_batch(
        runtime, [prepared], [legacy, black, legacy_random, harmonized_random]
    )
    assert torch.equal(method_features(features, black)["OBJ-H"], features["BLACK"])
    image_digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    raw_random_mask = area_matched_random_mask(
        mask,
        np.random.default_rng(
            record_seed(seed, image_digest, "legacy-random-region-v1")
        ),
    )
    assert raw_random_mask.sum() == mask.sum()
    expected = runtime.features(
        runtime.pixels(legacy_object_view(image, raw_random_mask))[None]
    )
    assert torch.equal(method_features(features, legacy_random)["OBJ-H"], expected)
    assert torch.equal(
        method_features(features, harmonized_random)["OBJ-H"],
        features["RANDOM_OBJ-H"],
    )
    assert not torch.equal(
        method_features(features, legacy_random)["OBJ-H"],
        method_features(features, harmonized_random)["OBJ-H"],
    )
    for method in (black, legacy_random, harmonized_random):
        assert torch.equal(
            method_features(features, method)["ORIG"],
            method_features(features, legacy)["ORIG"],
        )


def test_random_legacy_empty_mask_is_black_without_native_fallback(image_case):
    runtime, prepared, _, _ = prepare_case(image_case, empty=True, fallback=True)
    legacy = method_by_id("E1")
    random = replace(legacy, id="random-empty", control="random_region")
    features = prepare_feature_batch(runtime, [prepared], [legacy, random])
    assert torch.equal(
        method_features(features, random)["OBJ-H"],
        runtime.features(prepared["pixels"]["BLACK"][None]),
    )
    assert not torch.equal(
        method_features(features, legacy)["OBJ-H"],
        method_features(features, random)["OBJ-H"],
    )
