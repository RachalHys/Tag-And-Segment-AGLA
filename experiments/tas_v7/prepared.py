"""Prepare explicit legacy/canvas views and immutable minibatch visual features."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
import torch

from tas_runtime.decoding import record_seed
from tas_runtime.perception import PerceptionResult
from .views import (
    ViewConfig,
    create_views,
    processor_geometry_canvas,
    legacy_object_view,
    area_matched_random_mask,
)


def prepare_image(
    runtime, image_path, perception: PerceptionResult, config, *, random_seed=0
):
    with Image.open(image_path) as handle:
        raw = handle.convert("RGB")
    legacy = legacy_object_view(
        raw, perception.mask, fallback=perception.metadata["legacy_fallback"]
    )
    canvas, mask, geometry = processor_geometry_canvas(
        raw, perception.mask, runtime.image_processor
    )
    view_config = ViewConfig(sigma_ratio=config.sigma_ratio)
    views = create_views(canvas, mask, config=view_config)
    pixels = {name: runtime.pixels(image, canvas=True) for name, image in views.items()}
    pixels["LEGACY_OBJ-H"] = runtime.pixels(legacy)
    # ORIG must not silently acquire a different native preprocessing contract.
    native_original = runtime.pixels(raw)
    if not torch.equal(native_original, pixels["ORIG"]):
        error = (native_original - pixels["ORIG"]).abs().max().item()
        raise AssertionError(
            f"Model-canvas ORIG differs from native CLIP preprocessing ({error})."
        )
    pixels["BLACK"] = runtime.pixels(Image.new("RGB", canvas.size), canvas=True)
    image_digest = hashlib.sha256(np.asarray(raw).tobytes()).hexdigest()
    rng = np.random.default_rng(
        record_seed(random_seed, image_digest, "random-region-v1")
    )
    random_mask = area_matched_random_mask(mask, rng)
    random_views = create_views(canvas, random_mask, config=view_config)
    for name in ("OBJ-H", "BH", "FG-U"):
        pixels["RANDOM_" + name] = runtime.pixels(random_views[name], canvas=True)
    legacy_random_mask = area_matched_random_mask(
        perception.mask,
        np.random.default_rng(
            record_seed(random_seed, image_digest, "legacy-random-region-v1")
        ),
    )
    pixels["LEGACY_RANDOM_OBJ-H"] = runtime.pixels(
        legacy_object_view(raw, legacy_random_mask)
    )
    return {
        "pixels": pixels,
        "views": views,
        "canvas_mask": mask,
        "metadata": {
            "view_config": view_config.to_dict(),
            "geometry": geometry,
            "legacy_object_geometry": "mask raw RGB before CLIP resize/crop; preserve original fallback",
            "legacy_fallback": perception.metadata["legacy_fallback"],
            "mask_fraction_raw": float(perception.mask.mean()),
            "mask_fraction_canvas": float(mask.mean()),
            "mask_empty": not bool(mask.any()),
            "mask_full": bool(mask.all()),
        },
    }


def pixel_key(method, view):
    # Controls replace their named input first. Renaming a method or removing
    # its support filter must otherwise preserve its protocol's geometry.
    if method.control == "black_context" and view == method.context_view:
        return "BLACK"
    if method.control == "random_region" and view != "ORIG":
        if method.protocol == "LEGACY" and view == "OBJ-H":
            return "LEGACY_RANDOM_OBJ-H"
        return "RANDOM_" + view
    if method.protocol == "LEGACY" and view == "OBJ-H":
        return "LEGACY_OBJ-H"
    return view


@torch.inference_mode()
def prepare_feature_batch(runtime, prepared, methods):
    """Reuse exact tensors with the native single-image vision encoder shape.

    FP16 encoding can depend on batch shape. Keep each encoded view independent
    of the other images/methods in this minibatch, then batch language decoding.
    """
    needed = sorted(
        {pixel_key(method, view) for method in methods for view in method.views}
    )
    unique, locations = {}, {}
    for name in needed:
        for index, item in enumerate(prepared):
            pixel = item["pixels"][name]
            digest = hashlib.sha256(pixel.numpy().tobytes()).hexdigest()
            if digest not in unique:
                unique[digest] = pixel
            locations[(name, index)] = digest
    features = {
        key: runtime.features(pixel.unsqueeze(0))[0] for key, pixel in unique.items()
    }
    return {
        name: torch.stack(
            [features[locations[(name, index)]] for index in range(len(prepared))]
        )
        for name in needed
    }


def method_features(feature_batch, method):
    return {view: feature_batch[pixel_key(method, view)] for view in method.views}


def input_fingerprints(prepared, method):
    """CPU pixel identities for exact reuse of completed diagnostic captions.

    Checking every named view is conservative: even differences that disappear
    on conversion to FP16 cause regeneration. No prefix or KV cache is shared.
    """
    # Scope this cache to one call. Research callers may edit/reassign pixels
    # between experiments while retaining the same prepared dictionary.
    cached = {}
    result = {}
    for view in method.views:
        name = pixel_key(method, view)
        if name not in cached:
            pixel = prepared["pixels"][name]
            if pixel.device.type != "cpu":
                raise ValueError("Diagnostic input fingerprints require CPU preprocessing tensors.")
            cached[name] = {
                "shape": list(pixel.shape), "stride": list(pixel.stride()), "dtype": str(pixel.dtype),
                "sha256": hashlib.sha256(pixel.numpy().tobytes()).hexdigest(),
            }
        result[view] = cached[name]
    return result
