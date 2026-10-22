"""Network-free integration with the real pinned LLaVA/Llama implementation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.tas_v7.methods import fuse_logits, method_by_id, plausibility_mask
from tas_runtime.attention import set_attention_backend
from tas_runtime.decoding import (
    DecodeOptions,
    decode_branches,
    forced_prefix_logits,
    record_seed,
)
from tas_runtime.runtime import LlavaBackend, legacy_sampler


def inputs(batch=1):
    torch.manual_seed(8)
    return torch.tensor([[1, 8, -200, 4, 6]]).repeat(batch, 1), torch.rand(
        batch, 3, 4, 4
    )


def test_prefill_features_and_last_position_match_original_forward(model):
    ids, pixels = inputs(2)
    original = model(
        ids, images=pixels, attention_mask=torch.ones_like(ids), use_cache=True
    ).logits[:, -1]
    backend = LlavaBackend(model)
    pixels_state = backend.prefill(ids, pixels)
    features_state = backend.prefill(ids, model.encode_images(pixels))
    torch.testing.assert_close(pixels_state.logits, original)
    torch.testing.assert_close(features_state.logits, original)
    assert pixels_state.length == ids.shape[1] + 1  # one token replaced by two patches
    advanced = backend.advance(torch.tensor([[5], [7]]), pixels_state)
    assert advanced.length == pixels_state.length + 1
    assert pixels_state.past[0][0].shape[-2] == pixels_state.length


def test_fp16_native_clip_features_match_raw_pixels_and_cached_decode(model, tmp_path):
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    from experiments.tas_v7.prepared import prepare_feature_batch
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    from tas_runtime.runtime import LlavaRuntime

    vision_dir = tmp_path / "vision"
    config = CLIPVisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, image_size=8, patch_size=2,
    )
    CLIPVisionModel(config).save_pretrained(vision_dir)
    CLIPImageProcessor(size=8, crop_size=8).save_pretrained(vision_dir)
    model.model.vision_tower = CLIPVisionTower(
        str(vision_dir), SimpleNamespace(mm_vision_select_layer=-2)
    )
    model.model.mm_projector = torch.nn.Linear(8, model.config.hidden_size)
    model.half()
    runtime = LlavaRuntime(model, None, None, tmp_path, last_position_logits=False)
    ids = torch.tensor([[1, 8, -200, 4, 6]])
    generator = torch.Generator().manual_seed(91)
    prepared = []
    for _ in range(2):
        pixels = torch.rand(3, 8, 8, generator=generator)
        prepared.append({"pixels": {
            "ORIG": pixels, "OBJ-H": pixels * 0.4, "FG-U": pixels * 0.8,
        }})
    method = method_by_id("F4")
    features = prepare_feature_batch(runtime, prepared, [method])
    for index, item in enumerate(prepared):
        raw = {name: item["pixels"][name][None].half() for name in method.views}
        cached = {name: features[name][index:index + 1] for name in method.views}
        expected = forced_prefix_logits(
            runtime.backend, ids, raw, [3, 4], execution="serial"
        )
        actual = forced_prefix_logits(
            runtime.backend, ids, cached, [3, 4], execution="serial"
        )
        for first, second in zip(expected, actual):
            for name in method.views:
                assert first[name].dtype == torch.float16
                assert torch.equal(first[name], second[name])


@pytest.mark.parametrize("identifier", ["E0", "E1"])
def test_generation_reproduces_actual_legacy_sampler(model, identifier):
    ids, pixels = inputs()
    method = method_by_id(identifier)
    views = {"ORIG": pixels}
    kwargs = {}
    if identifier == "E1":
        views["OBJ-H"] = pixels * 0.4
        kwargs.update(images_cd=views["OBJ-H"], cd_alpha=2.0, cd_beta=0.5)
    seed = record_seed(0, 123, identifier)
    with (
        torch.random.fork_rng(),
        legacy_sampler(Path(__file__).resolve().parents[1], enabled=identifier == "E1"),
    ):
        torch.manual_seed(seed)
        reference = model.generate(
            ids,
            images=pixels,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=12,
            use_cache=True,
            **kwargs,
        )[0, ids.shape[1] :].tolist()
    result = decode_branches(
        LlavaBackend(model, last_position_logits=False),
        ids,
        views,
        lambda z: fuse_logits(z, method),
        (lambda z: plausibility_mask(z, 0.5)) if method.support else None,
        DecodeOptions(max_new_tokens=12, protocol="LEGACY"),
        [seed],
        execution="serial",
    )
    assert result.token_ids == [reference]


def test_serial_packed_independent_histories_and_schedule_rng(model):
    ids, pixels = inputs(2)
    views = {"ORIG": pixels, "OBJ-H": pixels * 0.5, "FG-U": pixels * 0.9}
    method = method_by_id("F4")
    backend = LlavaBackend(model)
    a = forced_prefix_logits(backend, ids, views, [7, 8, 9], execution="serial")
    b = forced_prefix_logits(backend, ids, views, [7, 8, 9], execution="packed")
    for first, second in zip(a, b):
        for name in views:
            torch.testing.assert_close(first[name], second[name], atol=2e-7, rtol=2e-5)
    seeds = [record_seed(9, image, "F4") for image in (123, 456)]
    options = DecodeOptions(max_new_tokens=10)
    fusion = lambda z: fuse_logits(z, method)
    batch = decode_branches(backend, ids, views, fusion, None, options, seeds)
    single = []
    for index in (1, 0):
        row = decode_branches(
            backend,
            ids[index : index + 1],
            {k: v[index : index + 1] for k, v in views.items()},
            fusion,
            None,
            options,
            [seeds[index]],
            execution="serial",
        )
        single.insert(0, row.token_ids[0])
    assert single == batch.token_ids


def test_sdpa_square_cached_and_non_square_chunk_match_eager(model):
    ids, pixels = inputs(2)
    backend = LlavaBackend(model)
    eager = backend.prefill(ids, pixels)
    eager_step = backend.advance(torch.tensor([[3], [4]]), eager)
    eager_chunk = backend.advance(torch.tensor([[3, 7], [4, 9]]), eager)
    set_attention_backend(model, "sdpa", unpadded=True)
    sdpa = backend.prefill(ids, pixels)
    step = backend.advance(torch.tensor([[3], [4]]), sdpa)
    chunk = backend.advance(torch.tensor([[3, 7], [4, 9]]), sdpa)
    for first, second in ((eager, sdpa), (eager_step, step), (eager_chunk, chunk)):
        torch.testing.assert_close(first.logits, second.logits, atol=2e-7, rtol=2e-5)
    set_attention_backend(model, "eager")
    torch.testing.assert_close(backend.prefill(ids, pixels).logits, eager.logits)


def test_per_image_eos_and_token_broadcast():
    class Backend:
        def __init__(self):
            self.advances = []

        def prefill(self, ids, pixels):
            from tas_runtime.decoding import ForwardState

            # First image ends immediately; second ends after one continuation.
            return ForwardState(
                torch.tensor([[0.0, 0.0, 20.0, 0.0], [0.0, 0.0, 0.0, 20.0]]).repeat(
                    2, 1
                ),
                None,
                5,
            )

        def advance(self, tokens, state):
            from tas_runtime.decoding import ForwardState

            self.advances.append(tokens.clone())
            return ForwardState(
                torch.tensor([[0.0, 0.0, 20.0, 0.0]]).repeat(4, 1), None, 6
            )

    backend = Backend()
    result = decode_branches(
        backend,
        torch.ones(2, 3, dtype=torch.long),
        {"ORIG": torch.zeros(2, 3, 4, 4), "OBJ-H": torch.zeros(2, 3, 4, 4)},
        lambda z: z["ORIG"],
        None,
        DecodeOptions(do_sample=False, max_new_tokens=5),
        [1, 2],
    )
    assert result.token_ids == [[2], [3, 2]]
    assert result.stop_reasons == ["eos_or_stop", "eos_or_stop"]
    assert backend.advances[0].squeeze().tolist() == [2, 3, 2, 3]


@pytest.mark.parametrize("execution", ["serial", "packed"])
@pytest.mark.parametrize("sample", [False, True])
def test_stop_metadata_handles_early_eos_last_step_sequence_and_token_limit(execution, sample):
    from tas_runtime.decoding import ForwardState

    class Backend:
        def scores(self, rows, step):
            logits = torch.full((len(rows), 8), -100.0)
            for index, row in enumerate(rows.tolist()):
                token = 2 if row == 0 else (5 + step) if row == 1 else 3
                logits[index, token] = 100.0
            return logits

        def prefill(self, ids, images):
            return ForwardState(self.scores(ids[:, 0], 0), ids[:, 0], 1)

        def advance(self, tokens, state):
            # The terminal token enters the cache, then subsequent positions pad.
            assert tokens[state.past == 0].eq(2 if state.length == 1 else 0).all()
            return ForwardState(self.scores(state.past, state.length), state.past, state.length + 1)

    result = decode_branches(
        Backend(), torch.arange(3).reshape(3, 1),
        {"ORIG": torch.zeros(3, 1), "OBJ-H": torch.zeros(3, 1)},
        lambda logits: logits["ORIG"], None,
        DecodeOptions(do_sample=sample, max_new_tokens=3, stop_token_sequences=((5, 6, 7),)),
        [11, 22, 33], execution=execution,
    )
    assert result.token_ids == [[2], [5, 6, 7], [3, 3, 3]]
    assert result.stop_reasons == ["eos_or_stop", "eos_or_stop", "max_new_tokens"]
