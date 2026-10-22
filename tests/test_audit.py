"""P0 validates the actual optimized path against raw-pixel baselines."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.tas_v7.audit import audit_and_pilot, compare_snapshots
from tas_runtime.config import GenerationConfig, StudyConfig
from tas_runtime.runtime import LlavaRuntime


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "</s>"

    def __call__(self, text):
        return SimpleNamespace(input_ids=[self.eos_token_id])

    def batch_decode(self, ids, **kwargs):
        return [" ".join(map(str, row.tolist())) for row in ids]


@pytest.fixture
def pilot_case(model, tmp_path):
    runtime = LlavaRuntime(
        model, TinyTokenizer(), None, Path(__file__).resolve().parents[1]
    )
    runtime.prompt = lambda query, mode: (
        query,
        torch.tensor([1, 8, -200, 4, 6]),
        "</s>",
    )
    generator = torch.Generator().manual_seed(12)
    rows, prepared = [], []
    for index in range(2):
        pixels = torch.rand((3, 4, 4), generator=generator)
        rows.append({"image_id": index, "query": "Describe this image."})
        prepared.append(
            {
                "pixels": {
                    "ORIG": pixels,
                    "OBJ-H": pixels * 0.6,
                    "LEGACY_OBJ-H": pixels * 0.3,
                    "FG-U": pixels * 0.9,
                },
                "metadata": {"fixture": "different raw/canvas object views"},
            }
        )
    config = StudyConfig(
        persistent_root=str(tmp_path),
        generation=GenerationConfig(top_k=0),
        pilot_images=2,
        pilot_tokens=3,
        batch_sizes=(2,),
        attention_backends=("eager",),
        execution="packed",
    )
    return runtime, config, rows, prepared


@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_optimized_policy_validates_native_and_harmonized_paths(
    pilot_case, tmp_path, sample, backend
):
    runtime, config, rows, prepared = pilot_case
    config = replace(
        config,
        attention_backends=(backend,),
        generation=replace(
            config.generation, do_sample=sample, temperature=1.0 if sample else 0.0
        ),
    )
    report = audit_and_pilot(runtime, config, rows, prepared, tmp_path / "audit")
    assert report["status"] == "passed"
    assert all(check["native_token_match"] for check in report["baseline_checks"])
    assert report["native_reference_generation"]["do_sample"] is True
    assert report["native_reference_generation"]["temperature"] == 1.0
    assert report["execution_reference"]["methods"] == ["E0", "E1", "F4"]
    assert len(report["pilot"]) == 1
    checks = report["pilot"][0]["method_checks"]
    for identifier in ("E0", "E1", "F4"):
        assert checks[identifier]["greedy_match"]
        assert checks[identifier]["support_token_changes"] == 0
        if sample and identifier != "F4":
            assert checks[identifier]["sample_match"]
        else:
            assert "sample_match" not in checks[identifier]
    assert runtime.backend.last_position_logits is True


def test_native_feature_corruption_cannot_hide_behind_f4_pilot(
    pilot_case, tmp_path, monkeypatch
):
    runtime, config, rows, prepared = pilot_case
    original_features = runtime.features
    legacy_pixels = [item["pixels"]["LEGACY_OBJ-H"] for item in prepared]

    def corrupted_features(images):
        features = original_features(images).clone()
        for index, image in enumerate(images):
            if any(torch.equal(image, expected) for expected in legacy_pixels):
                # F4 never uses this legacy-only input. Its own feature-based
                # reference would remain internally consistent with the old gate.
                features[index] = torch.linspace(
                    -1000, 1000, features.shape[-1], dtype=features.dtype
                )
        return features

    monkeypatch.setattr(runtime, "features", corrupted_features)
    directory = tmp_path / "audit"
    with pytest.raises(RuntimeError, match="No execution policy passed"):
        audit_and_pilot(runtime, config, rows, prepared, directory)
    report = json.loads((directory / "audit.json").read_text())
    assert all(check["native_token_match"] for check in report["baseline_checks"])
    candidate = report["pilot"][0]
    assert candidate["status"] == "rejected"
    assert candidate["method_checks"]["E0"]["greedy_match"]
    assert "E1" in candidate["method_checks"]
    assert not (directory / "execution-policy.json").exists()


def test_locked_policy_rechecks_only_the_locked_option(pilot_case, tmp_path):
    runtime, config, rows, prepared = pilot_case
    directory = tmp_path / "audit"
    first = audit_and_pilot(runtime, config, rows, prepared, directory)
    expanded = replace(
        config,
        attention_backends=("eager", "sdpa"),
        execution="auto",
        batch_sizes=(1, 2),
    )
    second = audit_and_pilot(runtime, expanded, rows, prepared, directory)
    assert second["selected_policy"] == first["selected_policy"]
    assert len(second["pilot"]) == 1


def test_snapshot_support_uses_original_native_dtype():
    first = [{"ORIG": torch.tensor([[10.0, 9.3046875]], dtype=torch.float16)}]
    second = [{"ORIG": torch.tensor([[10.0, 9.296875]], dtype=torch.float16)}]
    # Both are excluded under FP32 support, but only the second is excluded
    # under the rounded native-FP16 threshold used by legacy E1.
    compare_snapshots(first, second, protocol="HARMONIZED")
    with pytest.raises(AssertionError, match="plausibility support"):
        compare_snapshots(first, second, protocol="LEGACY")


def test_failed_head_optimization_uses_checked_full_head_policy(
    pilot_case, tmp_path, monkeypatch
):
    runtime, config, rows, prepared = pilot_case
    finish = runtime.backend._finish

    def shape_sensitive_finish(output):
        state = finish(output)
        if runtime.backend.last_position_logits:
            # A common offset preserves sampled/greedy decisions but exceeds
            # the fixed-prefix tolerance. Native-token parity alone misses it.
            state.logits = state.logits + 1.0
        return state

    monkeypatch.setattr(runtime.backend, "_finish", shape_sensitive_finish)
    directory = tmp_path / "audit"
    report = audit_and_pilot(runtime, config, rows, prepared, directory)
    assert report["baseline_checks"][0]["last_position_sample_match"]
    assert report["pilot"][0]["status"] == "rejected"
    assert "step 0, view ORIG" in report["pilot"][0]["error"]
    assert report["selected_policy"] == {
        "backend": "eager",
        "execution": "serial",
        "batch_images": 1,
        "last_position_logits": False,
    }
    fallback = report["pilot"][-1]
    assert fallback["status"] == "passed"
    assert set(fallback["method_checks"]) == {"E0", "E1", "F4"}
    assert all(
        check["max_absolute_logit_error"] == 0
        for check in fallback["method_checks"].values()
    )
    assert runtime.backend.last_position_logits is False
    # Reconnects must retain the selected head and recheck just that policy.
    runtime.backend.last_position_logits = True  # fresh loader default
    second = audit_and_pilot(runtime, config, rows, prepared, directory)
    assert second["selected_policy"] == report["selected_policy"]
    assert len(second["pilot"]) == 1
    assert second["pilot"][0]["last_position_logits"] is False


def test_last_position_token_mismatch_rejects_optimization_only(
    pilot_case, tmp_path, monkeypatch
):
    runtime, config, rows, prepared = pilot_case
    finish = runtime.backend._finish

    def changed_head(output):
        state = finish(output)
        if runtime.backend.last_position_logits:
            state.logits = state.logits.clone()
            state.logits[:, -1] += 100
        return state

    monkeypatch.setattr(runtime.backend, "_finish", changed_head)
    report = audit_and_pilot(runtime, config, rows, prepared, tmp_path / "audit")
    assert all(check["native_token_match"] for check in report["baseline_checks"])
    assert not any(
        check["last_position_sample_match"] for check in report["baseline_checks"]
    )
    assert report["selected_policy"]["last_position_logits"] is False
    assert runtime.backend.last_position_logits is False


def test_native_decoder_mismatch_still_blocks(pilot_case, tmp_path, monkeypatch):
    runtime, config, rows, prepared = pilot_case
    reference = runtime.reference

    def changed_reference(*args, **kwargs):
        tokens = reference(*args, **kwargs).clone()
        tokens[:, 0] = (tokens[:, 0] + 1) % runtime.model.config.vocab_size
        return tokens

    monkeypatch.setattr(runtime, "reference", changed_reference)
    directory = tmp_path / "audit"
    with pytest.raises(AssertionError, match="does not reproduce"):
        audit_and_pilot(runtime, config, rows, prepared, directory)
    report = json.loads((directory / "audit.json").read_text())
    assert report["status"] == "failed"
    assert report["pilot"] == []
    assert not (directory / "execution-policy.json").exists()


def test_failed_locked_head_is_never_replaced(pilot_case, tmp_path, monkeypatch):
    runtime, config, rows, prepared = pilot_case
    directory = tmp_path / "audit"
    first = audit_and_pilot(runtime, config, rows, prepared, directory)
    assert first["selected_policy"]["last_position_logits"] is True
    frozen = (directory / "execution-policy.json").read_bytes()
    finish = runtime.backend._finish

    def changed_head(output):
        state = finish(output)
        if runtime.backend.last_position_logits:
            state.logits = state.logits.clone()
            state.logits[:, -1] += 100
        return state

    monkeypatch.setattr(runtime.backend, "_finish", changed_head)
    with pytest.raises(RuntimeError, match="No execution policy passed"):
        audit_and_pilot(runtime, config, rows, prepared, directory)
    assert (directory / "execution-policy.json").read_bytes() == frozen
    report = json.loads((directory / "audit.json").read_text())
    assert len(report["pilot"]) == 1
    assert "locked last-position head" in report["pilot"][0]["error"]
