"""Execution comparisons preserve reference gates, identities and natural prefixes."""

from dataclasses import replace
import json

import pytest
import torch
import test_study

from experiments.tas_v7 import batch_compare
from experiments.tas_v7.methods import method_by_id
from experiments.tas_v7.prepared import method_features, prepare_feature_batch

# Reuse the miniature real-model/data fixture without duplicating its setup.
study = test_study.study


@pytest.mark.parametrize("execution", ["serial", "packed"])
def test_history_replay_keeps_distinct_image_prefixes(study, execution):
    run, rows = study
    prepared, _ = run.prepare_rows(rows)
    method = method_by_id("F4")
    views = method_features(
        prepare_feature_batch(run.runtime, prepared, [method]), method
    )
    ids = torch.stack([run.runtime.prompt(row["query"])[1] for row in rows])
    histories = [[8, 9, 10], [11]]  # second image already stopped; pad its row
    result = batch_compare.history_logits(
        run.runtime.backend, ids, views, histories, 3, execution=execution, pad_id=2
    )
    for index, history in enumerate(([8, 9, 10], [11, 2, 2])):
        reference = batch_compare.history_logits(
            run.runtime.backend,
            ids[index : index + 1],
            {key: value[index : index + 1] for key, value in views.items()},
            [history],
            3,
            execution="serial",
            pad_id=2,
        )
        for name in method.views:
            torch.testing.assert_close(result[name][index : index + 1], reference[name])


def test_policy_context_restores_reference_after_failure(study):
    run, _ = study
    run.policy = {
        "backend": "eager",
        "execution": "serial",
        "batch_images": 1,
        "last_position_logits": True,
    }
    original = dict(run.policy)
    with pytest.raises(RuntimeError, match="failure"):
        with batch_compare.using_policy(
            run,
            {
                **original,
                "backend": "sdpa",
                "batch_images": 2,
                "last_position_logits": False,
            },
        ):
            assert run.runtime.backend.last_position_logits is False
            raise RuntimeError("failure")
    assert run.policy == original
    assert run.runtime.attention_backend == "eager"
    assert run.runtime.backend.last_position_logits is True


def test_comparison_runs_full_captions_in_separate_ledgers_and_resumes(
    study, monkeypatch
):
    from tas_runtime import profiling
    from experiments.tas_v7.study import SessionBudgetReached

    run, rows = study
    run.config = replace(
        run.config,
        stage="batch_compare",
        comparison_images=2,
        pilot_tokens=3,
        batch_sizes=(1, 2),
        attention_backends=("eager",),
        execution="auto",
    )
    monkeypatch.setattr(run, "rows", lambda cohort: rows)
    profiles = []

    def profile(call, directory, *, device):
        profiles.append(directory)
        assert len(call().token_ids) == 1
        return {"status": "fixture"}

    monkeypatch.setattr(profiling, "profile_call", profile)
    collection = run.collection

    def pause_before_candidate(stage, *args, **kwargs):
        if stage == "batch-compare-candidate":
            raise SessionBudgetReached("pause after reference")
        return collection(stage, *args, **kwargs)

    monkeypatch.setattr(run, "collection", pause_before_candidate)
    with pytest.raises(SessionBudgetReached, match="pause after reference"):
        batch_compare.run_batch_comparison(run)
    assert len(run.ledgers) == 1
    assert len(run.ledgers[0][0].records(latest=True)) == 8
    # Recheck of strict P0 is tested above. Isolate resume here: frozen discovery
    # and successful B=1 captions must survive a stop before candidate generation.
    monkeypatch.setattr(run, "audit", lambda: None)
    monkeypatch.setattr(run, "collection", collection)
    generate = batch_compare.generate

    def only_candidate(*args, **kwargs):
        assert len(args[3]) == 2
        return generate(*args, **kwargs)

    monkeypatch.setattr("experiments.tas_v7.audit.generate", only_candidate)
    batch_compare.run_batch_comparison(run)
    assert len(profiles) == 2
    report_path = next(run.persistent_run.glob("batch-comparison-*/comparison.json"))
    report = json.loads(report_path.read_text())
    assert report["status"] == "descriptive_only"
    assert report["n_pairs"] == 8
    assert report["reference_policy"]["batch_images"] == 1
    assert report["candidate_policy"]["batch_images"] == 2
    assert run.policy == report["reference_policy"]
    assert set(item["method"] for item in report["summary"]) == {
        "E0",
        "E1",
        "E1-H",
        "F4",
    }
    outputs = {}
    for ledger, _ in run.ledgers:
        role = ledger.identity["execution_protocol"].rsplit("/", 1)[-1]
        outputs[role] = [item["payload"] for item in ledger.records(latest=True)]
    assert set(outputs) == {"reference", "candidate"}
    assert all(
        row["generated_tokens"] <= run.config.generation.max_new_tokens
        for rows_ in outputs.values()
        for row in rows_
    )
    assert {
        (row["image_id"], row["method"], row["seed"]) for row in outputs["reference"]
    } == {(row["image_id"], row["method"], row["seed"]) for row in outputs["candidate"]}
    # Strict P0 rechecks on real resume. Skip only that already covered gate here
    # to prove saved comparison discovery/captions cause no repeat generation.
    monkeypatch.setattr(run, "audit", lambda: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed comparison outputs must not be regenerated")

    monkeypatch.setattr(batch_compare, "discover", forbidden)
    monkeypatch.setattr(run.runtime.backend, "prefill", forbidden)
    batch_compare.run_batch_comparison(run)
    assert json.loads(report_path.read_text())["summary"] == report["summary"]


def test_comparison_failure_cannot_start_the_study_smoke(study, monkeypatch):
    run, _ = study
    run.config = replace(run.config, stage="batch_compare")
    monkeypatch.setattr(run, "initialize", lambda: run)
    calls = []

    def comparison(study_):
        calls.append("comparison")
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(batch_compare, "run_batch_comparison", comparison)
    monkeypatch.setattr(run, "audit", lambda: calls.append("strict study audit"))
    monkeypatch.setattr(
        run, "collection", lambda *args, **kwargs: calls.append("smoke")
    )
    with pytest.raises(RuntimeError, match="candidate failed"):
        run.run()
    assert calls == ["comparison"]


@pytest.mark.parametrize(
    "invalid",
    [
        {"comparison_images": 1},
        {"comparison_images": 31},
    ],
)
def test_comparison_configuration_is_bounded(study, invalid):
    run, _ = study
    with pytest.raises(ValueError):
        replace(run.config, **invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        {"comparison_images": 2, "batch_sizes": (1, 4)},
        {"batch_sizes": (1,)},
        {"methods": ("E0", "F4")},
        {"methods": ("E0", "E1", "F4", "F4")},
    ],
)
def test_invalid_comparison_stops_before_any_model_load(study, invalid):
    run, _ = study
    with pytest.raises(ValueError):
        replace(run.config, stage="batch_compare", **invalid)


def test_comparison_cannot_bypass_a_native_reference_failure(study, monkeypatch):
    run, rows = study
    original_config = run.config = replace(
        run.config,
        stage="batch_compare",
        comparison_images=2,
        batch_sizes=(1, 2),
        pilot_tokens=3,
    )
    monkeypatch.setattr(run, "rows", lambda cohort: rows)

    def failed_audit():
        raise AssertionError("native baseline mismatch")

    monkeypatch.setattr(run, "audit", failed_audit)
    monkeypatch.setattr(
        batch_compare,
        "discover",
        lambda *args: pytest.fail("discovery after failed P0"),
    )
    with pytest.raises(AssertionError, match="native baseline"):
        batch_compare.run_batch_comparison(run)
    assert run.config is original_config
    assert not list(run.persistent_run.glob("batch-comparison-*"))


def test_natural_prefix_report_distinguishes_close_choices_from_support_changes(study):
    run, _ = study
    method = method_by_id("F4")
    first = {name: torch.tensor([[2.0, 1.999, -3.0]]) for name in method.views}
    second = {name: torch.tensor([[1.999, 2.0, -3.0]]) for name in method.views}
    report = batch_compare._distribution_check(
        run.runtime,
        run.config,
        method,
        first,
        second,
        {"reference_token": 0, "candidate_token": 1},
    )
    assert report["strict_numerical_check"]["status"] == "within_tolerance"
    assert report["replay_matches_recorded_choices"]
    assert 0 < report["reference_top_two"]["top_two_margin"] < 0.01
    assert 0 < report["sample_distribution_total_variation"] < 0.01
    assert report["support_token_changes"] == 0
    second["ORIG"][0, 2] = 2.0
    report = batch_compare._distribution_check(
        run.runtime,
        run.config,
        method,
        first,
        second,
        {"reference_token": 0, "candidate_token": 1},
    )
    assert report["support_token_changes"] == 1
    assert report["strict_numerical_check"]["status"] == "outside_strict_checks"


def test_discovery_records_drift_without_making_it_strictly_eligible(
    study, monkeypatch
):
    from tas_runtime import profiling
    from experiments.tas_v7.methods import harmonized_baselines

    run, rows = study
    run.config = replace(
        run.config, pilot_tokens=3, batch_sizes=(1, 2), execution="serial"
    )
    run.policy = {
        "backend": "eager",
        "execution": "serial",
        "batch_images": 1,
        "last_position_logits": True,
    }
    reference = dict(run.policy)
    generate = batch_compare.generate

    def changed_caption(*args, **kwargs):
        result = generate(*args, **kwargs)
        if len(args[3]) == 2 and kwargs.get("greedy"):
            result.token_ids[0][0] = (
                result.token_ids[0][0] + 1
            ) % run.runtime.model.config.vocab_size
        return result

    monkeypatch.setattr(batch_compare, "generate", changed_caption)
    monkeypatch.setattr(
        profiling, "profile_call", lambda *args, **kwargs: {"status": "fixture"}
    )
    methods = [
        method_by_id("E0"),
        method_by_id("E1"),
        harmonized_baselines()[1],
        method_by_id("F4"),
    ]
    result = batch_compare.discover(run, rows, methods, reference, run.persistent_run)
    assert result["candidate_policy"]["batch_images"] == 2
    check = result["candidates"][0]["methods"]["E1"]
    assert check["exact_greedy_match"] is False
    difference = check["first_differences"][0]
    assert difference["position"] == 0
    assert difference["natural_prefix"]["replay_matches_recorded_choices"] is False
    assert run.policy == reference
    assert not (run.persistent_run / "execution-policy.json").exists()


def test_comparison_plan_is_explicit_about_cost_and_no_promotion(tmp_path):
    from experiments.tas_v7.cli import plan
    from tas_runtime.config import StudyConfig

    result = plan(
        StudyConfig(str(tmp_path), stage="batch_compare", batch_sizes=(1, 2, 4))
    )
    assert result["complete_captions"] == 96
    assert result["promotion"].startswith("none")
    assert result["methods"] == ["E0", "E1", "E1-H", "F4"]


def test_profile_generation_failure_does_not_mark_discovery_complete(
    study, monkeypatch
):
    from tas_runtime import profiling

    run, rows = study
    run.config = replace(
        run.config, pilot_tokens=3, batch_sizes=(1, 2), execution="serial"
    )
    run.policy = {
        "backend": "eager",
        "execution": "serial",
        "batch_images": 1,
        "last_position_logits": True,
    }

    def failed(*args, **kwargs):
        raise RuntimeError("profile generation failed")

    monkeypatch.setattr(profiling, "profile_call", failed)
    with pytest.raises(RuntimeError, match="profile generation"):
        batch_compare.discover(
            run,
            rows,
            [method_by_id(name) for name in ("E0", "E1", "F4")],
            dict(run.policy),
            run.persistent_run,
        )
    report = json.loads((run.persistent_run / "discovery.json").read_text())
    assert report["status"] == "selected"
    assert "candidate_policy" in report
    assert run.policy["batch_images"] == 1
