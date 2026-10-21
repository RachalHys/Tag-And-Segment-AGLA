import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from transformers import CLIPImageProcessor

from tas_runtime.config import GenerationConfig, StudyConfig
from tas_runtime.persistence import PersistentNpzCache
from tas_runtime.perception import PerceptionResult
from tas_runtime.runtime import LlavaRuntime
from experiments.tas_v7.study import Study, SessionBudgetReached
from experiments.tas_v7.methods import method_by_id


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
    records, report = run.collection("smoke-fixture", rows, methods, seeds=(0,))
    assert len(records) == 6
    assert report["completed_outputs"] == report["expected_outputs"] == 6
    assert all(
        record["CHAIRs"] == 0 and record["actual_batch_images"] == 2
        for record in records
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed captions must not be recomputed")

    monkeypatch.setattr(run.runtime.backend, "prefill", forbidden)
    resumed, _ = run.collection("smoke-fixture", rows, methods, seeds=(0,))
    assert resumed == records
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


def test_diagnostic_pipeline_reruns_sam_and_keeps_mask_quality_metadata_aligned(
    study, monkeypatch
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
    run_diagnostics(run, method_by_id("F4"))
    assert len(calls) == len(rows)
    expected = {"D-PRED": [0.8], "D-MASK": [None], "D-BOX": [0.91], "D-MISS": []}
    for variant, qualities in expected.items():
        saved = next(run.persistent_run.glob(f"diagnostics-{variant}-*/captions.jsonl"))
        records = [json.loads(line) for line in saved.read_text().splitlines()]
        assert len(records) == len(rows) * 3
        for record in records:
            metadata = record["perception_metadata"]
            assert metadata["sam_predicted_iou"] == qualities
            if variant == "D-MISS":
                assert metadata["proposed_labels"] == [] and metadata["mask_empty"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed diagnostic inference must be reused")

    monkeypatch.setattr(run.runtime.backend, "prefill", forbidden)
    run_diagnostics(run, method_by_id("F4"))
    assert len(calls) == len(rows)
    (run.persistent_run / "diagnostics/fixed-prefix-influence.jsonl").write_text(
        "corrupted"
    )
    with pytest.raises(AssertionError, match="Completed diagnostic inference"):
        run_diagnostics(run, method_by_id("F4"))
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
