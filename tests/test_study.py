import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from transformers import CLIPImageProcessor

from experiments.tas_v7.methods import method_by_id
from experiments.tas_v7.study import SessionBudgetReached, Study
from tas_runtime.config import GenerationConfig, StudyConfig
from tas_runtime.perception import PerceptionResult
from tas_runtime.persistence import PersistentNpzCache
from tas_runtime.runtime import LlavaRuntime


class Tokenizer:
    eos_token, eos_token_id, bos_token_id, pad_token_id = "</s>", 2, 1, 0

    def __call__(self, text, add_special_tokens=True):
        if text == self.eos_token:
            ids = [2]
        else:
            ids = [3 + len(word) % 30 for word in text.split()]
        return SimpleNamespace(input_ids=([1] if add_special_tokens else []) + ids)

    def decode(self, tokens, skip_special_tokens=True):
        return "A cat."

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]


@pytest.fixture
def study(tmp_path, model):
    config = StudyConfig(
        str(tmp_path / "drive"),
        local_root=str(tmp_path / "vm"),
        generation=GenerationConfig(max_new_tokens=4),
        snapshot_every=1,
        seeds=(0, 1, 2),
        bootstrap_replicates=100,
    )
    result = Study(config)
    data = tmp_path / "data"
    annotation_dir = data / "scoring_annotations"
    annotation_dir.mkdir(parents=True)
    rows = []
    for identifier in (123, 456):
        image = Image.fromarray(
            np.random.default_rng(identifier).integers(
                0, 255, (9, 13, 3), dtype=np.uint8
            )
        )
        path = data / f"{identifier}.png"
        image.save(path)
        rows.append(
            {
                "id": identifier,
                "image_id": identifier,
                "image": path.name,
                "query": "Describe this image.",
                "split": "val2017",
            }
        )
    annotation = {
        "images": [{"id": row["id"]} for row in rows],
        "categories": [{"id": 17, "name": "cat"}],
        "annotations": [
            {
                "id": i,
                "image_id": row["id"],
                "category_id": 17,
                "bbox": [3, 2, 7, 5],
                "segmentation": [[3, 2, 10, 2, 10, 7, 3, 7]],
            }
            for i, row in enumerate(rows)
        ],
    }
    (annotation_dir / "instances_val2017.json").write_text(json.dumps(annotation))
    captions = {
        "images": annotation["images"],
        "annotations": [
            {"id": i, "image_id": row["id"], "caption": "A cat."}
            for i, row in enumerate(rows)
        ],
    }
    (annotation_dir / "captions_val2017.json").write_text(json.dumps(captions))
    result.runtime = LlavaRuntime(
        model,
        Tokenizer(),
        CLIPImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
        ),
        Path(__file__).resolve().parents[1],
    )
    result.work = {
        "data_root": data,
        "local_root": tmp_path / "vm",
        "persistent_root": tmp_path / "drive",
    }
    result.identity = {"test_scientific_identity": 1}
    result.local_run, result.persistent_run = (
        tmp_path / "vm/run",
        tmp_path / "drive/runs/run",
    )
    result.local_run.mkdir(parents=True)
    result.persistent_run.mkdir(parents=True)
    result.policy = {"backend": "eager", "execution": "packed", "batch_images": 2}
    result.cache = PersistentNpzCache(tmp_path / "vm/cache", tmp_path / "drive/cache")

    def perception(row):
        mask = np.zeros((9, 13), bool)
        mask[2:7, 3:10] = True
        return (
            PerceptionResult(
                mask,
                mask[None],
                np.array([[3, 2, 10, 7]], np.float32),
                np.array([0.8], np.float32),
                {
                    "legacy_fallback": False,
                    "inventory": ["cat"],
                    "proposed_labels": ["cat"],
                    "sam_predicted_iou": [0.8],
                },
            ),
            {"image": row["image_id"]},
            True,
        )

    result.perception_for = perception
    return result, rows


def test_real_tiny_multimodal_collection_resume_and_cpu_reports(study, monkeypatch):
    from experiments.tas_v7.cli import score_saved

    run, rows = study
    methods = [method_by_id(name) for name in ("E0", "E1", "F4")]
    original_prompt = run.runtime.prompt
    prompt_queries = []

    def prompt(query, conv_mode):
        prompt_queries.append(query)
        return original_prompt(query, conv_mode)

    monkeypatch.setattr(run.runtime, "prompt", prompt)
    records, report = run.collection("smoke-fixture", rows, methods, seeds=(0,))
    assert prompt_queries == ["Describe this image."]
    assert len(records) == 6
    assert report["completed_outputs"] == report["expected_outputs"] == 6
    assert all(
        record["CHAIRs"] == 0 and record["actual_batch_images"] == 2
        for record in records
    )
    assert all(
        record["batch_latency_seconds"]
        == pytest.approx(record["amortized_generation_seconds"] * 2)
        for record in records
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed captions must not be recomputed")

    monkeypatch.setattr(run.runtime.backend, "prefill", forbidden)
    monkeypatch.setattr("experiments.tas_v7.study._scorer", forbidden)
    resumed, _ = run.collection("smoke-fixture", rows, methods, seeds=(0,))
    assert resumed == records
    assert prompt_queries == ["Describe this image."]
    run.close()
    rescored = score_saved(run.config)
    assert rescored["count"] == 1
    restored_report = json.loads(Path(rescored["reports"][0]).read_text())
    assert restored_report["expected_outputs"] == 6
    assert (
        restored_report["metadata"]["shared_stage_cost"]
        == report["metadata"]["shared_stage_cost"]
    )
    assert restored_report["metadata"]["complete"] is True


def test_cpu_report_keeps_incomplete_and_failed_outputs_visible(tmp_path):
    from experiments.tas_v7.cli import score_saved
    from tas_runtime.artifacts import ResultLedger
    from tas_runtime.persistence import publish_ledger

    identity = {
        "rows": [{"image_id": 1}, {"image_id": 2}],
        "methods": [{"id": "E0"}],
        "seeds": [0, 1, 2],
        "modes": ["sample", "greedy"],
    }
    ledger = ResultLedger(tmp_path / "vm", identity)
    ledger.record({"image_id": 1}, {"error": "recorded failure"}, status="error")
    publish_ledger(ledger, tmp_path / "drive/runs/fixture/smoke/snapshots")
    result = score_saved(StudyConfig(str(tmp_path / "drive")))
    report = json.loads(Path(result["reports"][0]).read_text())
    assert report["expected_outputs"] == 8 and report["completed_outputs"] == 0
    assert report["metadata"]["complete"] is False
    assert len(report["failures"]) == 1


@pytest.mark.parametrize("finalist", [None, "F4"])
def test_diagnostic_pipeline_reruns_sam_and_keeps_mask_quality_metadata_aligned(
    study, monkeypatch, finalist
):
    from experiments.tas_v7.study_diagnostics import run_diagnostics

    run, rows = study
    monkeypatch.setattr(run, "rows", lambda cohort: rows)
    calls = []

    def sam_rerun(image, *, inventory_override, boxes_override):
        calls.append(np.array(boxes_override, copy=True))
        mask = np.zeros((image.height, image.width), bool)
        mask[1:8, 2:11] = True
        return PerceptionResult(
            mask, mask[None], boxes_override, np.ones(1), {"sam_predicted_iou": [0.91]}
        )

    run.perception = SimpleNamespace(run=sam_rerun, close=lambda: None)
    method = method_by_id(finalist) if finalist else None
    run_diagnostics(run, method)
    assert len(calls) == len(rows)
    expected = {"D-PRED": [0.8], "D-MASK": [None], "D-BOX": [0.91], "D-MISS": []}
    for variant, qualities in expected.items():
        saved = next(run.persistent_run.glob(f"diagnostics-{variant}-*/captions.jsonl"))
        records = [json.loads(line) for line in saved.read_text().splitlines()]
        assert len(records) == len(rows) * (3 if finalist else 2)
        for record in records:
            metadata = record["perception_metadata"]
            assert metadata["sam_predicted_iou"] == qualities
            if variant == "D-MISS":
                assert metadata["proposed_labels"] == [] and metadata["mask_empty"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed diagnostic inference must be reused")

    monkeypatch.setattr(run.runtime.backend, "prefill", forbidden)
    run_diagnostics(run, method)
    assert len(calls) == len(rows)
    (run.persistent_run / "diagnostics/fixed-prefix-influence.jsonl").write_text(
        "corrupted"
    )
    with pytest.raises(AssertionError, match="Completed diagnostic inference"):
        run_diagnostics(run, method)
    run.close()


def test_budget_is_a_resumable_stop(study):
    run, _ = study
    run.deadline = 0
    with pytest.raises(SessionBudgetReached):
        run.check_budget()


def test_wrong_dataset_or_silent_prompt_change_rejected(tmp_path):
    with pytest.raises(ValueError):
        StudyConfig(str(tmp_path), dataset="pope")
    with pytest.raises(ValueError):
        StudyConfig(
            str(tmp_path), generation=GenerationConfig(prompt="Invent a story.")
        )


def test_duplicate_seed_and_precision_change_rejected():
    with pytest.raises(ValueError):
        GenerationConfig(precision="bf16")
    with pytest.raises(ValueError):
        StudyConfig("/tmp/unused", seeds=(0, 0))


def test_study_cli_plan_is_network_free(tmp_path):
    from experiments.tas_v7.cli import main

    path = tmp_path / "config.json"
    path.write_text(json.dumps(StudyConfig(str(tmp_path)).as_dict()))
    result = main(["--config", str(path), "--phase", "plan"])
    assert result["smoke"] == 420
    assert result["screen_initial"] == 3360
    assert result["shortlist"]["maximum_coverage_loss_percentage_points"] == 2


@pytest.mark.parametrize("improved", [False, True])
def test_automatic_flow_obeys_scientific_shortlist_before_dependent_stages(
    study, monkeypatch, improved
):
    from test_report import caption

    from experiments.tas_v7 import study_diagnostics

    run, rows = study
    monkeypatch.setattr(run, "initialize", lambda: run)
    monkeypatch.setattr(run, "audit", lambda: None)
    monkeypatch.setattr(run, "rows", lambda cohort: rows)
    calls = []

    def collection(stage, rows, methods, **kwargs):
        calls.append(stage)
        # Only the stage boundary is stubbed. The actual scientific selector
        # receives valid pooled counts and sees either a tie or a true gain.
        records = [
            {
                **caption(
                    method.id,
                    false=2
                    if stage == "screen" and improved and method.id == "F4"
                    else 10,
                ),
                "protocol": method.protocol,
            }
            for method in methods
        ]
        return records, {}

    monkeypatch.setattr(run, "collection", collection)
    monkeypatch.setattr(
        study_diagnostics, "run_diagnostics", lambda *_: calls.append("diagnostics")
    )
    result = run.run()
    if improved:
        assert result["status"] == "complete_first_round"
        assert calls == [
            "smoke",
            "screen",
            "confirm",
            "temperature-0.7",
            "temperature-1.0",
            "temperature-1.3",
            "controls",
            "diagnostics",
        ]
    else:
        assert result["status"] == "no_eligible_finalist"
        assert calls == ["smoke", "screen", "diagnostics"]


def test_partial_batch_keeps_locked_shape_without_padding_results(study, monkeypatch):
    run, rows = study
    methods = [method_by_id(name) for name in ("E0", "F4")]
    complete, _ = run.collection("full", rows, methods, seeds=(0,))
    sizes = []
    prefill = run.runtime.backend.prefill

    def observed_prefill(ids, features):
        sizes.append(len(ids))
        return prefill(ids, features)

    monkeypatch.setattr(run.runtime.backend, "prefill", observed_prefill)
    partial, report = run.collection("tail", rows[:1], methods, seeds=(0,))
    assert sizes == [2, 6]  # locked B=2, then packed three-view B=2
    assert len(partial) == report["completed_outputs"] == report["expected_outputs"] == 2
    reference = {row["method"]: row for row in complete if row["image_id"] == rows[0]["image_id"]}
    for row in partial:
        assert row["token_ids"] == reference[row["method"]]["token_ids"]
        assert row["actual_batch_images"] == 1
        assert row["execution_batch_images"] == 2
        assert row["padding_images"] == 1


def test_full_flow_reuses_screen_samples_in_controls(study, monkeypatch):
    from dataclasses import replace
    import shutil

    from experiments.tas_v7 import audit, report

    run, rows = study
    run.config = replace(run.config, run_confirmation=False, run_diagnostics=False)
    monkeypatch.setattr(run, "initialize", lambda: run)
    monkeypatch.setattr(run, "audit", lambda: None)
    monkeypatch.setattr(run, "rows", lambda cohort: rows)
    # Selection itself has separate scientific tests. Fix a finalist here to
    # exercise the actual full controls collection with the miniature model.
    monkeypatch.setattr(
        report, "select_finalists", lambda *args, **kwargs: {"selected": ["F4"]}
    )
    baseline_ids = {"E0", "E1", "F0", "E0-H", "E1-H", "F4"}
    active_stage, saved, calls = None, {}, []
    collection, generate = run.collection, audit.generate

    def observed_collection(stage, *args, **kwargs):
        nonlocal active_stage
        active_stage = stage
        result = collection(stage, *args, **kwargs)
        saved[stage] = result
        return result

    def observed_generate(*args, **kwargs):
        method = args[2]
        if active_stage == "controls" and method.id in baseline_ids:
            assert kwargs.get("greedy"), (
                "Controls regenerated an already completed screen sample"
            )
        calls.append((active_stage, method.id, kwargs.get("greedy")))
        return generate(*args, **kwargs)

    monkeypatch.setattr(run, "collection", observed_collection)
    monkeypatch.setattr(audit, "generate", observed_generate)
    assert run.run()["status"] == "complete_first_round"
    screen = {(row["image_id"], row["method"]): row for row in saved["screen"][0]}
    controls, controls_report = saved["controls"]
    reused = [row for row in controls if "reused_from_stage" in row]
    assert (
        len(reused)
        == controls_report["metadata"]["reused_outputs"]
        == len(rows) * len(baseline_ids)
    )
    for row in reused:
        expected = screen[(row["image_id"], row["method"])]
        assert row == {**expected, "reused_from_stage": row["reused_from_stage"]}
        assert row["reused_from_stage"] == Path(run.stage_reports["screen"]).parent.name
    assert any(stage == "controls" and greedy for stage, _, greedy in calls)
    assert any(stage == "controls" and method.endswith("_BLACK") for stage, method, _ in calls)
    assert any(stage == "temperature-1.0" for stage, _, _ in calls)
    before = len(calls)
    shutil.rmtree(run.local_run)
    run.ledgers = []  # restore every stage from verified Drive snapshots
    assert run.run()["status"] == "complete_first_round"
    assert len(calls) == before  # completed controls, imports and reports resume


def test_screen_reuse_avoids_all_generation_and_preserves_science(study, monkeypatch):
    from experiments.tas_v7.cli import score_saved

    run, rows = study
    methods = [method_by_id(name) for name in ("E0", "E1", "F4")]
    source = run.collection("screen", rows, methods, seeds=(0,))
    reference, _ = run.collection("controls-reference", rows, methods, seeds=(0,))
    assert [row["token_ids"] for row in source[0]] == [
        row["token_ids"] for row in reference
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Exact completed captions must not prepare features or generate again"
        )

    monkeypatch.setattr(run.runtime.backend, "prefill", forbidden)
    monkeypatch.setattr(run, "prepare_rows", forbidden)
    monkeypatch.setattr("experiments.tas_v7.study._scorer", forbidden)
    reused, report = run.collection("controls", rows, methods, seeds=(0,), reuse_from=source)
    assert len(reused) == report["metadata"]["reused_outputs"] == 6
    assert report["metadata"]["shared_stage_cost"]["new_generation_seconds"] == 0
    assert all(row["reused_captions"] == 2 for row in report["summary"])
    for before, after in zip(source[0], reused):
        assert after == {**before, "reused_from_stage": after["reused_from_stage"]}
    assert sum(row["generation_seconds"] for row in report["summary"]) > 0
    run.close()
    report_path = Path(run.stage_reports["controls"])
    report_path.unlink()
    score_saved(run.config)
    rebuilt = json.loads(report_path.read_text())
    assert rebuilt["metadata"]["reused_outputs"] == 6
    assert all(row["reused_captions"] == 2 for row in rebuilt["summary"])


@pytest.mark.parametrize(
    "field",
    ["study", "rows", "generation", "seeds", "execution", "partial_batch", "interventions"],
)
def test_screen_reuse_rejects_incompatible_inputs_before_inference(study, monkeypatch, field):
    from copy import deepcopy

    from tas_runtime.artifacts import ArtifactError

    run, rows = study
    methods = [method_by_id("E0")]
    records, report = run.collection("screen", rows, methods, seeds=(0,))
    report = deepcopy(report)
    previous = report["metadata"]["identity"]
    changes = {
        "study": {"another_scientific_identity": 2},
        "rows": previous["rows"][::-1],
        "generation": {**previous["generation"], "temperature": 0.7},
        "seeds": [1],
        "execution": {**previous["execution"], "last_position_logits": False},
        "partial_batch": "a different padding rule",
        "interventions": {str(rows[0]["image_id"]): {"modified": True}},
    }
    previous[field] = changes[field]

    def forbidden(*args, **kwargs):
        raise AssertionError("Incompatible reuse must fail before inference")

    monkeypatch.setattr(run, "prepare_rows", forbidden)
    with pytest.raises(ArtifactError):
        run.collection("controls", rows, methods, seeds=(0,), reuse_from=(records, report))
    assert run.ledgers[-1][0].records() == []
    run.close()


@pytest.mark.parametrize(
    "problem",
    ["incomplete", "missing", "duplicate", "unexpected", "wrong_mode", "wrong_stage"],
)
def test_screen_reuse_rejects_invalid_source(study, problem):
    from copy import deepcopy

    from tas_runtime.artifacts import ArtifactError

    run, rows = study
    methods = [method_by_id("E0")]
    records, report = deepcopy(run.collection("screen", rows, methods, seeds=(0,)))
    if problem == "incomplete":
        report["metadata"]["complete"] = False
    elif problem == "missing":
        records.pop()
    elif problem == "duplicate":
        records[-1] = records[0]
    elif problem == "unexpected":
        records[-1]["query"] = "another prompt"
    elif problem == "wrong_mode":
        report["metadata"]["identity"]["modes"] = ["greedy"]
    elif problem == "wrong_stage":
        report["metadata"]["identity"]["stage"] = "confirm"
    with pytest.raises(ArtifactError):
        run.collection("controls", rows, methods, seeds=(0,), reuse_from=(records, report))
    assert run.ledgers[-1][0].records() == []
    run.close()


def test_changed_method_is_generated_instead_of_reusing_screen(study, monkeypatch):
    from dataclasses import replace

    from experiments.tas_v7 import audit

    run, rows = study
    method = method_by_id("F4")
    source = run.collection("screen", rows, [method], seeds=(0,))
    generate, calls = audit.generate, []

    def observed_generate(*args, **kwargs):
        calls.append(args[2].to_dict())
        return generate(*args, **kwargs)

    monkeypatch.setattr(audit, "generate", observed_generate)
    changed = replace(method, eta=0.6)
    records, report = run.collection(
        "controls", rows, [changed], seeds=(0,), reuse_from=source
    )
    assert calls == [changed.to_dict()]
    assert report["metadata"]["reused_outputs"] == 0
    assert not any("reused_from_stage" in row for row in records)
    run.close()


def test_interrupted_controls_restore_imports_and_successful_greedy_outputs(study, monkeypatch):
    import shutil

    from experiments.tas_v7 import audit

    run, rows = study
    methods = [method_by_id(name) for name in ("E0", "E1")]
    source = run.collection("screen", rows, methods, seeds=(0,))
    generate = audit.generate

    def interrupted_generation(*args, **kwargs):
        assert kwargs["greedy"]
        if args[2].id == "E1":
            raise KeyboardInterrupt("simulated session interruption")
        return generate(*args, **kwargs)

    monkeypatch.setattr(audit, "generate", interrupted_generation)
    with pytest.raises(KeyboardInterrupt, match="simulated session interruption"):
        run.collection(
            "controls", rows, methods, seeds=(0,),
            modes=("sample", "greedy"), reuse_from=source,
        )
    assert len(run.ledgers[-1][0].records(latest=True)) == 6
    shutil.rmtree(run.local_run)
    run.ledgers = []
    calls = []

    def remaining_generation(*args, **kwargs):
        assert kwargs["greedy"] and args[2].id == "E1"
        calls.append(args[2].id)
        return generate(*args, **kwargs)

    monkeypatch.setattr(audit, "generate", remaining_generation)
    records, report = run.collection(
        "controls", rows, methods, seeds=(0,),
        modes=("sample", "greedy"), reuse_from=source,
    )
    assert calls == ["E1"]
    assert len(records) == report["completed_outputs"] == report["expected_outputs"] == 8
    assert report["metadata"]["reused_outputs"] == 4
    assert all(row["attempt"] == 1 for row in run.ledgers[-1][0].records(latest=True))
    run.close()
