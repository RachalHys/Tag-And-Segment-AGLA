"""Measured batching is explicit; native correctness and run isolation remain."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

import test_audit
import test_study
from experiments.tas_v7 import audit
from experiments.tas_v7.methods import harmonized_baselines, initial_methods, method_by_id
from experiments.tas_v7.report import select_finalists, write_report
from tas_runtime.artifacts import ArtifactError

pilot_case = test_audit.pilot_case
study = test_study.study


def batched(config, **changes):
    return replace(
        config, **{
            "execution_protocol": "tas-batched-study-v1",
            "batch_sizes": (4,), "attention_backends": ("sdpa",),
            "execution": "packed", "last_position_logits": True,
            "pilot_images": 4, **changes,
        }
    )


@pytest.fixture
def batched_pilot(pilot_case):
    runtime, config, rows, prepared = pilot_case
    rows += [{**row, "image_id": row["image_id"] + 2} for row in rows]
    prepared = [
        {**item, "pixels": {**item["pixels"], "BH": item["pixels"]["ORIG"] * 0.1}}
        for item in prepared + prepared
    ]
    return runtime, batched(config), rows, prepared


@pytest.mark.parametrize("change", [
    {"execution": "serial"}, {"batch_sizes": (1, 4)},
    {"attention_backends": ("eager",)}, {"last_position_logits": False},
    {"pilot_images": 2}, {"stage": "batch_compare"},
    {"dataset": "amber", "stage": "external"},
    {"execution_protocol": "unknown"},
])
def test_invalid_batched_configuration_fails_before_model_load(pilot_case, change):
    with pytest.raises(ValueError):
        batched(pilot_case[1], **change)


@pytest.mark.parametrize("half", [False, True])
def test_batched_audit_checks_all_methods_with_one_policy_and_reuses_snapshots(
    batched_pilot, tmp_path, monkeypatch, half
):
    runtime, config, rows, prepared = batched_pilot
    if half:
        runtime.model.half()
        runtime.dtype = torch.float16
    forced = audit.forced_prefix_logits
    raw_calls = []

    def observed(*args, **kwargs):
        if next(iter(args[2].values())).ndim == 4:
            raw_calls.append(tuple(args[2]))
        return forced(*args, **kwargs)

    monkeypatch.setattr(audit, "forced_prefix_logits", observed)
    report = audit.audit_and_pilot(runtime, config, rows, prepared, tmp_path / "audit")
    assert report["status"] == "passed"
    assert report["execution_protocol"] == config.execution_protocol
    assert report["selected_policy"] == config.fixed_execution_policy
    assert len(report["pilot"]) == 1
    checks = report["pilot"][0]["method_checks"]
    assert set(checks) == {m.id for m in initial_methods() + harmonized_baselines()}
    assert all(check["support_token_changes"] == 0 for check in checks.values())
    assert all(check["native_token_match"] for check in report["baseline_checks"])
    assert set(name for name, check in checks.items() if "sample_match" in check) == {"E0", "E1", "F4"}
    # Four images × five unique raw views, rather than rerunning each view for
    # every formula. Immutable snapshots are shared; generated prefixes are not.
    assert len(raw_calls) == 20
    again = audit.audit_and_pilot(runtime, config, rows, prepared, tmp_path / "audit")
    assert again["selected_policy"] == report["selected_policy"]
    assert len(again["pilot"]) == 1


def test_batched_free_token_drift_is_reported_without_weakening_native_gate(
    batched_pilot, tmp_path, monkeypatch
):
    runtime, config, rows, prepared = batched_pilot
    generate = audit.generate

    def drift(*args, **kwargs):
        result = generate(*args, **kwargs)
        if next(iter(args[4].values())).ndim == 3:
            result.token_ids[0][0] = (result.token_ids[0][0] + 1) % runtime.model.config.vocab_size
        return result

    monkeypatch.setattr(audit, "generate", drift)
    report = audit.audit_and_pilot(runtime, config, rows, prepared, tmp_path / "audit")
    assert report["status"] == "passed"
    assert report["selected_policy"] == config.fixed_execution_policy
    drift = report["pilot"][0]["token_drift"]
    assert set(drift) == {"E0", "E1", "F4"}
    assert all(set(value) == {"greedy", "sample"} for value in drift.values())
    assert all(check["native_token_match"] for check in report["baseline_checks"])


@pytest.mark.parametrize("failure", ["native", "head", "features"])
def test_batched_audit_has_no_silent_fallback_on_blocking_failure(
    batched_pilot, tmp_path, monkeypatch, failure
):
    runtime, config, rows, prepared = batched_pilot
    if failure == "native":
        original = runtime.reference

        def changed(*args, **kwargs):
            tokens = original(*args, **kwargs).clone()
            tokens[:, 0] = (tokens[:, 0] + 1) % runtime.model.config.vocab_size
            return tokens

        monkeypatch.setattr(runtime, "reference", changed)
    elif failure == "head":
        original = runtime.backend._finish

        def changed(output):
            state = original(output)
            if runtime.backend.last_position_logits:
                state.logits = state.logits.clone()
                state.logits[:, -1] += 100
            return state

        monkeypatch.setattr(runtime.backend, "_finish", changed)
    else:
        original = runtime.features
        black_holes = [item["pixels"]["BH"] for item in prepared]

        def changed(images):
            features = original(images).clone()
            if any(torch.equal(images[0], pixel) for pixel in black_holes):
                features[:] = torch.linspace(-1000, 1000, features.shape[-1])
            return features

        monkeypatch.setattr(runtime, "features", changed)
    directory = tmp_path / "audit"
    with pytest.raises((RuntimeError, AssertionError)):
        audit.audit_and_pilot(runtime, config, rows, prepared, directory)
    report = json.loads((directory / "audit.json").read_text())
    assert report["status"] == "failed"
    assert len(report["pilot"]) <= 1
    assert not (directory / "execution-policy.json").exists()


def test_batched_locked_policy_and_shape_cannot_change(batched_pilot, tmp_path):
    runtime, config, rows, prepared = batched_pilot
    directory = tmp_path / "audit"
    with pytest.raises(ValueError, match="four prepared"):
        audit.audit_and_pilot(runtime, config, rows[:2], prepared[:2], directory)
    directory.mkdir(exist_ok=True)
    (directory / "execution-policy.json").write_text(json.dumps({
        **config.fixed_execution_policy, "batch_images": 2,
    }))
    with pytest.raises(ValueError, match="locked policy differs"):
        audit.audit_and_pilot(runtime, config, rows, prepared, directory)


def test_batched_run_identity_separates_shortlist_and_policy_locks(study, monkeypatch):
    from experiments.tas_v7 import study as module
    from tas_runtime import perception, profiling
    from tas_runtime.runtime import LlavaRuntime

    run, _ = study
    runtime = run.runtime
    work = {
        **run.work, "paths": {},
        "assets": {"identity": "frozen-models", "assets": {}},
        "preparation": {"dataset_identity": "frozen-data", "nlp": "frozen-nlp"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        name="NVIDIA A100-SXM4-40GB", total_memory=40 * 1024**3, major=8,
    ))
    monkeypatch.setattr(module, "restore_working_set", lambda _: work)
    monkeypatch.setattr(LlavaRuntime, "load", lambda *a, **kw: runtime)
    monkeypatch.setattr(profiling, "environment_manifest", lambda: {"packages": {}})
    monkeypatch.setattr(perception, "perception_identity", lambda *a: {})
    run.initialize()
    strict_id, strict_root = run.run_id, run.persistent_run
    (strict_root / "shortlist.json").write_text('{"selected":["old"]}')
    run.config = batched(run.config)
    run.initialize()
    assert run.run_id != strict_id
    assert not (run.persistent_run / "shortlist.json").exists()
    assert run.identity["execution"] == {
        "protocol": "tas-batched-study-v1", "fixed_policy": run.config.fixed_execution_policy,
    }
    identifier = run.run_id
    run.initialize()
    assert run.run_id == identifier
    assert json.loads((strict_root / "shortlist.json").read_text()) == {"selected": ["old"]}


def test_batched_partial_collection_is_labelled_and_cannot_reuse_strict_outputs(study):
    run, rows = study
    methods = [method_by_id(name) for name in ("E0", "E1", "F4")]
    source = run.collection("screen", rows, methods, seeds=(0,))
    run.config = batched(run.config)
    run.policy = run.config.fixed_execution_policy
    run.runtime.set_attention("sdpa")
    with pytest.raises(ArtifactError):
        run.collection("controls", rows, methods, seeds=(0,), reuse_from=source)
    records, report = run.collection("smoke", rows, methods, seeds=(0,))
    assert len(records) == report["expected_outputs"] == 6
    for row in records:
        assert row["execution_protocol"] == "tas-batched-study-v1"
        assert row["execution_policy"] == run.config.fixed_execution_policy
        assert row["actual_batch_images"] == row["padding_images"] == 2
        assert row["execution_batch_images"] == 4
        assert row["seed"] == 0
    again, _ = run.collection("smoke", rows, methods, seeds=(0,))
    assert again == records


def test_batched_flow_preserves_policy_through_confirmation_and_controls(study, monkeypatch):
    from experiments.tas_v7 import report, study_diagnostics

    run, rows = study
    run.config = batched(run.config)
    run.policy = run.config.fixed_execution_policy
    run.runtime.set_attention("sdpa")
    monkeypatch.setattr(run, "initialize", lambda: run)
    monkeypatch.setattr(run, "audit", lambda: None)
    monkeypatch.setattr(run, "rows", lambda cohort: rows)
    monkeypatch.setattr(report, "select_finalists", lambda *a, **kw: {"selected": ["F4"]})
    diagnostics = []
    monkeypatch.setattr(study_diagnostics, "run_diagnostics", lambda *a: diagnostics.append(dict(run.policy)))
    original = run.collection
    outputs = {}

    def collection(stage, *args, **kwargs):
        result = original(stage, *args, **kwargs)
        outputs[stage] = result
        assert all(row["execution_policy"] == run.config.fixed_execution_policy for row in result[0])
        assert all(row["execution_protocol"] == "tas-batched-study-v1" for row in result[0])
        return result

    monkeypatch.setattr(run, "collection", collection)
    result = run.run()
    assert result["status"] == "complete_first_round"
    assert set(outputs) == {"smoke", "screen", "confirm", "controls", "temperature-0.7", "temperature-1.0", "temperature-1.3"}
    assert len(outputs["smoke"][0]) == 28
    assert len(outputs["screen"][0]) == 32
    assert {row["method"] for row in outputs["screen"][0]} >= {"E0-H", "E1-H"}
    assert {row["seed"] for row in outputs["confirm"][0]} == {0, 1, 2}
    assert {row["generation_mode"] for row in outputs["confirm"][0]} == {"sample", "greedy"}
    assert outputs["controls"][1]["metadata"]["reused_outputs"] > 0
    assert diagnostics == [run.config.fixed_execution_policy]
    shortlist = json.loads((run.persistent_run / "shortlist.json").read_text())
    assert shortlist["execution_protocol"] == result["execution_protocol"] == "tas-batched-study-v1"


@pytest.mark.parametrize("kind", ["protocol", "policy"])
def test_report_and_shortlist_reject_mixed_execution(tmp_path, kind):
    from test_report import caption

    rows = [caption("E1-H"), caption("F4", false=1)]
    if kind == "protocol":
        rows[1]["execution_protocol"] = "tas-batched-study-v1"
    else:
        rows[1]["execution_policy"] = {"batch_images": 4}
    with pytest.raises(ValueError, match="cannot mix execution"):
        select_finalists(rows)
    with pytest.raises(ValueError, match="cannot mix execution"):
        write_report(tmp_path, rows)
