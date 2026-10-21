"""Repository-backed, resumable v7 study for CPU preparation and one GPU.

The automatic first-round flow is an explicit user override of the brief's
original smoke-review pause. P0 correctness gates remain mandatory. Adaptive
P3/P4/P5 and training are not enabled by this runner.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil
import time
from collections import defaultdict

from tas_runtime.artifacts import (
    ArtifactError,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from tas_runtime.config import StudyConfig


class SessionBudgetReached(Exception):
    """A planned resumable stop, never a failed image or skipped example."""


def source_identity(project_root):
    root = Path(project_root)
    files = {}
    for directory in ("tas_runtime", "experiments/tas_v7"):
        for path in sorted((root / directory).glob("*.py")):
            files[str(path.relative_to(root))] = sha256_file(path)
    for name in (
        "llava/model/language_model/llava_llama.py",
        "llava/model/llava_arch.py",
        "llava/model/multimodal_encoder/clip_encoder.py",
        "llava/model/multimodal_encoder/builder.py",
        "llava/model/multimodal_projector/builder.py",
        "llava/conversation.py",
        "llava/constants.py",
        "llava/mm_utils.py",
        "eval_tas/sample.py",
        "eval_tas/inventory_detector.py",
        "eval_tas/augmentation.py",
        "requirements/colab-base.txt",
        "requirements/colab-perception.txt",
        "requirements/colab-amber.txt",
    ):
        files[name] = sha256_file(root / name)
    return {"files": files, "sha256": canonical_hash(files)}


def prepare(config: StudyConfig, *, models=True, cohorts=("A", "B", "C", "D")):
    """CPU phase. Fetch artifacts without constructing any CUDA model."""
    from tas_runtime.assets import prepare_assets, download_verified, MODEL_SPECS
    from tas_runtime.bootstrap import prepare_nlp
    from tas_runtime.data import prepare_coco_data, POPE_URLS, pope_image_ids

    root = Path(config.persistent_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    pending_bytes = 0
    if models:
        for name, spec in MODEL_SPECS.items():
            for item in spec["files"]:
                path = (
                    root / "assets/models" / name / spec["revision"] / item["filename"]
                )
                if not path.exists() or path.stat().st_size != item["size"]:
                    partial = path.with_name(path.name + ".part")
                    pending_bytes += max(
                        0,
                        item["size"]
                        - (partial.stat().st_size if partial.exists() else 0),
                    )
        free = shutil.disk_usage(root).free
        if pending_bytes and free < pending_bytes + 1024**3:
            raise RuntimeError(
                f"Persistent storage needs about {(pending_bytes + 1024**3) / 1e9:.1f} GB more free space for the model/data preparation, but reports {free / 1e9:.1f} GB. Use a larger persistent folder before downloading."
            )
        print(
            f"Missing checkpoint transfer: {pending_bytes / 1e9:.2f} GB; verified existing files are reused.",
            flush=True,
        )
    model_manifest = prepare_assets(root / "assets") if models else None
    nlp = prepare_nlp(root / "nlp")
    excluded, provenance = [], {}
    if config.exclusions_file:
        source = Path(config.exclusions_file)
        excluded += json.loads(source.read_text())
        provenance["user_file_sha256"] = sha256_file(source)
    if config.reserve_pope:
        files = []
        for variant, url in POPE_URLS.items():
            path = download_verified(
                url, root / "data" / "pope" / f"coco_{variant}.jsonl", progress=print
            )
            files.append(path)
        excluded += pope_image_ids(files)
        provenance["official_pope_manifests_sha256"] = {
            path.name: sha256_file(path) for path in files
        }
    cohort_root = (
        root
        / "data"
        / "coco"
        / f"cohorts-{config.split_seed}-{canonical_hash(sorted(set(excluded)))[:12]}"
    )
    data = prepare_coco_data(
        cohort_root,
        seed=config.split_seed,
        exclusions=excluded,
        exclusion_provenance=provenance,
        cohorts=cohorts,
        audit_count=config.diagnostic_count,
    )
    from tas_runtime.persistence import bundle_dataset

    data_bundle = bundle_dataset(cohort_root, root)
    output = {
        "schema": "tas-preparation-v1",
        "assets": model_manifest["identity"] if model_manifest else None,
        "split_seed": config.split_seed,
        "diagnostic_count": config.diagnostic_count,
        "data_bundle": data_bundle,
        "dataset_root_relative": str(cohort_root.relative_to(root)),
        "dataset_identity": data["identity"],
        "nlp": nlp,
        "reserve_pope": config.reserve_pope,
        "exclusions": sorted(set(excluded)),
        "training_overlap": "unknown; split names do not establish unseen training data",
    }
    identity = canonical_hash(output)
    atomic_json(root / "preparations" / f"{identity}.json", output)
    # A convenience pointer only; the fully resolved content/identity is checked
    # again in each study and saved with its outputs.
    atomic_json(root / "preparation.json", output)
    print(f"CPU preparation complete: {root}", flush=True)
    return output


def restore_working_set(config):
    from tas_runtime.assets import asset_paths, prepare_assets
    from tas_runtime.persistence import (
        copy_verified,
        restore_dataset_bundle,
        restore_nlp,
    )

    persistent, local = (
        Path(config.persistent_root).expanduser(),
        Path(config.local_root).expanduser(),
    )
    manifest_path = persistent / "preparation.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "CPU preparation is missing. Run the notebook in CPU/prepare mode first."
        )
    preparation = json.loads(manifest_path.read_text())
    if not preparation.get("assets"):
        raise RuntimeError(
            "CPU preparation did not include model files. Run prepare without --skip-models before allocating the GPU."
        )
    for key in ("split_seed", "diagnostic_count", "reserve_pope"):
        if preparation.get(key) != getattr(config, key):
            raise RuntimeError(
                f"Prepared {key} differs from this configuration. Run CPU prepare for the intended cohorts."
            )
    assets = prepare_assets(
        persistent / "assets"
    )  # checks receipts, reuses verified files
    local.mkdir(parents=True, exist_ok=True)
    required = sum(
        item["size"]
        for name, spec in assets["assets"].items()
        for item in spec["files"]
        if not (
            local / "assets/models" / name / spec["revision"] / item["filename"]
        ).exists()
    )
    if shutil.disk_usage(local).free < required + 1024**3:
        raise RuntimeError(
            "Local runtime disk is too small for verified weights plus working data. Clear obsolete local sessions or choose a larger runtime disk."
        )
    for name, spec in assets["assets"].items():
        source_path = Path(assets["paths"][name])
        directory = source_path if "repo_id" in spec else source_path.parent
        relative = directory.relative_to(persistent / "assets")
        for item in spec["files"]:
            copy_verified(
                directory / item["filename"],
                local / "assets" / relative / item["filename"],
                sha256=item["sha256"],
            )
    portable = {"schema": "tas-model-assets-v1", "assets": assets["assets"]}
    atomic_json(local / "assets/assets-manifest.json", portable)
    source_data = persistent / preparation["dataset_root_relative"]
    destination = local / "data" / "coco" / source_data.name
    restore_dataset_bundle(persistent, local, destination, preparation["data_bundle"])
    restore_nlp(persistent / "nlp", local / "nlp")
    os.environ["NLTK_DATA"] = str(local / "nlp")
    import nltk

    nltk.data.path.insert(0, str(local / "nlp"))
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YOLO_AUTOINSTALL="false",
        HF_HOME=str(local / "hf-cache"),
        TORCH_HOME=str(local / "torch-cache"),
        YOLO_CONFIG_DIR=str(local / "ultralytics"),
    )
    return {
        "preparation": preparation,
        "assets": assets,
        "paths": asset_paths(local / "assets", portable),
        "data_root": destination,
        "local_root": local,
        "persistent_root": persistent,
    }


def _groups_by_prompt_length(runtime, rows, conv_mode):
    buckets = defaultdict(list)
    for row in rows:
        buckets[len(runtime.prompt(row["query"], conv_mode)[1])].append(row)
    return [group for _, group in sorted(buckets.items())]


def _scorer(data_root, rows):
    from .evaluation import CocoChairEvaluator

    split = rows[0]["split"]
    if any(row["split"] != split for row in rows):
        raise ValueError(
            "One caption scoring collection must use one explicit COCO split."
        )
    return CocoChairEvaluator.from_annotation_dir(
        Path(data_root) / "scoring_annotations",
        split=split,
        image_ids=[row["image_id"] for row in rows],
    )


class Study:
    def __init__(self, config: StudyConfig, project_root=None):
        self.config = config
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.source = source_identity(self.project_root)
        self.started = time.monotonic()
        self.deadline = self.started + config.max_session_minutes * 60
        self.runtime = None
        self.perception = None
        self.cache = None
        self.ledgers = []
        self.stage_reports = {}

    def check_budget(self):
        if time.monotonic() >= self.deadline:
            raise SessionBudgetReached(
                "Session time budget reached; completed records are persisted. Rerun to resume."
            )

    def initialize(self):
        import torch
        from tas_runtime.persistence import PersistentNpzCache
        from tas_runtime.perception import perception_identity
        from tas_runtime.profiling import environment_manifest
        from tas_runtime.runtime import LlavaRuntime
        from .evaluation import ChairLexicon
        from .methods import initial_methods

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Study.run needs GPU mode; CPU sessions should use prepare or score."
            )
        gpu = torch.cuda.get_device_properties(0)
        if self.config.require_a100 and (
            "A100" not in gpu.name or gpu.total_memory < 70 * 1024**3
        ):
            raise RuntimeError(
                f"Configured A100 80GB study received {gpu.name}/{gpu.total_memory / 1024**3:.1f}GiB. Select the intended runtime or explicitly revise the resource setting."
            )
        if gpu.major < 8:
            raise RuntimeError(
                "This optimized study profile targets Ampere or newer; T4 needs a separately validated profile."
            )
        self.work = restore_working_set(self.config)
        ChairLexicon().fixture_check()
        self.runtime = LlavaRuntime.load(
            self.work["paths"],
            self.project_root,
            last_position_logits=self.config.last_position_logits,
        )
        core = {
            "namespace": "tas-v7",
            "generation": dataclasses.asdict(self.config.generation),
            "last_position_logits": self.config.last_position_logits,
            "dataset_kind": self.config.dataset,
            "source": self.source,
            "model_assets": self.work["assets"]["identity"],
            "dataset": self.work["preparation"]["dataset_identity"],
            "sigma_ratio": self.config.sigma_ratio,
            "yolo_conf": self.config.yolo_conf,
            "expansion_ratio": self.config.expansion_ratio,
            "seeds": list(self.config.seeds),
            "methods": [method.to_dict() for method in initial_methods()],
            "nlp": self.work["preparation"]["nlp"],
            "environment": environment_manifest(),
            "stage_flow": {
                "automatic_after_smoke": self.config.auto_continue_after_smoke,
                "authorization": "owner explicitly overrode original smoke human-review pause; P0 remains blocking",
                "P3_P4_P5": "disabled",
            },
        }
        # Git commit/platform paths do not change the scientific source hash;
        # packages/GPU model do matter when replaying a run on another machine.
        core["environment"].pop("git_commit", None)
        core["environment"].pop("platform", None)
        self.identity = core
        self.run_id = canonical_hash(core)[:20]
        self.local_run = self.work["local_root"] / "runs" / self.run_id
        self.persistent_run = self.work["persistent_root"] / "runs" / self.run_id
        self.local_run.mkdir(parents=True, exist_ok=True)
        self.persistent_run.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.persistent_run / "resolved-config.json",
            {"configuration": self.config.as_dict(), "identity": core},
        )
        aux = {
            key: value
            for key, value in self.work["assets"]["assets"].items()
            if key in {"blip", "yolo", "mobile_sam", "clip_text"}
        }
        self.perception_config = perception_identity(aux, self.config.yolo_conf)
        self.cache = PersistentNpzCache(
            self.work["local_root"] / "perception-cache",
            self.work["persistent_root"] / "caches/perception",
            flush_every=16,
        )
        return self

    def rows(self, cohort):
        from tas_runtime.data import read_jsonl, verify_manifest_images

        manifest = self.work["data_root"] / "manifests" / f"{cohort}.jsonl"
        rows = read_jsonl(manifest)
        verify_manifest_images(rows, self.work["data_root"])
        return rows

    def perception_for(self, row):
        from PIL import Image
        from tas_runtime.perception import (
            PerceptionRuntime,
            PerceptionResult,
            perception_key,
        )

        path = self.work["data_root"] / row["image"]
        key = perception_key(path, self.perception_config)
        value = self.cache.get(key)
        if value is not None:
            return PerceptionResult.from_cache(value), key, True
        self.check_budget()
        if self.perception is None:
            self.perception = PerceptionRuntime(
                self.work["paths"], confidence=self.config.yolo_conf
            )
        with Image.open(path) as source:
            image = source.convert("RGB")
        from tas_runtime.profiling import synchronize

        synchronize()
        started = time.perf_counter()
        # Python hash order is fixed by the notebook subprocess environment.
        result = self.perception.run(image)
        synchronize()
        result.metadata["perception_seconds_uncached"] = time.perf_counter() - started
        self.cache.put(key, result.arrays(), result.metadata)
        return result, key, False

    def prepare_rows(self, rows):
        from .prepared import prepare_image

        output, elapsed, hits = [], 0.0, 0
        for row in rows:
            self.check_budget()
            result, key, cached = self.perception_for(row)
            item = prepare_image(
                self.runtime,
                self.work["data_root"] / row["image"],
                result,
                self.config,
                random_seed=self.config.split_seed,
            )
            item.update(perception_key=key, perception=result)
            elapsed += (
                result.metadata.get("perception_seconds_uncached", 0.0)
                if not cached
                else 0.0
            )
            hits += cached
            output.append(item)
        return output, {
            "new_perception_seconds": elapsed,
            "cache_hits": hits,
            "images": len(rows),
        }

    def audit(self):
        from .audit import audit_and_pilot

        rows = self.rows("A")[: max(self.config.pilot_images, 2)]
        prepared, perception_cost = self.prepare_rows(rows)
        # View gallery is actual RGB before normalization; tensor parity is also checked.
        from PIL import Image, ImageDraw

        tiles = []
        for row, item in zip(rows[:4], prepared[:4]):
            for name, image in item["views"].items():
                tile = image.resize((224, 224)).copy()
                ImageDraw.Draw(tile).text(
                    (4, 4),
                    f"{row['image_id']} {name}",
                    fill="red",
                    stroke_fill="white",
                    stroke_width=1,
                )
                tiles.append(tile)
        gallery = Image.new("RGB", (4 * 224, max(1, len(tiles) // 4) * 224))
        for index, tile in enumerate(tiles):
            gallery.paste(tile, ((index % 4) * 224, (index // 4) * 224))
        gallery.save(self.persistent_run / "views.png")
        result = audit_and_pilot(
            self.runtime, self.config, rows, prepared, self.persistent_run / "audit"
        )
        result["shared_perception_cost"] = perception_cost
        result["remaining_runtime_estimate"] = (
            "derive from complete smoke captions; short pilot is not a full-run promise"
        )
        atomic_json(self.persistent_run / "audit/audit.json", result)
        self.policy = result["selected_policy"]
        print(f"P0 passed. Locked execution policy: {self.policy}", flush=True)
        return result

    def collection(
        self,
        stage,
        rows,
        methods,
        *,
        seeds=None,
        modes=("sample",),
        prepared_overrides=None,
    ):
        import torch
        from tas_runtime.decoding import record_seed
        from tas_runtime.persistence import restore_ledger, publish_ledger
        from tas_runtime.profiling import synchronize
        from .audit import generate
        from .prepared import prepare_feature_batch, method_features
        from .report import write_report

        seeds = tuple(self.config.seeds if seeds is None else seeds)
        row_identity = [
            {
                "id": row["id"],
                "image_id": row["image_id"],
                "query": row["query"],
                "image": row["image"],
                "sha256": sha256_file(self.work["data_root"] / row["image"]),
            }
            for row in rows
        ]
        identity = {
            "study": self.identity,
            "stage": stage,
            "rows": row_identity,
            "methods": [method.to_dict() for method in methods],
            "seeds": seeds,
            "generation": dataclasses.asdict(self.config.generation),
            "modes": modes,
            "execution": self.policy,
            "interventions": {
                str(key): value["metadata"]
                for key, value in (prepared_overrides or {}).items()
            },
        }
        stage_id = stage + "-" + canonical_hash(identity)[:12]
        destination = self.persistent_run / stage_id
        snapshots = destination / "snapshots"
        ledger = restore_ledger(self.local_run / stage_id, snapshots, identity)
        self.ledgers.append((ledger, snapshots))
        expected = (
            len(rows)
            * len(methods)
            * sum(len(seeds) if mode == "sample" else 1 for mode in modes)
        )
        scorer = (
            _scorer(self.work["data_root"], rows)
            if self.config.dataset == "coco"
            else None
        )

        def key_for(row, method, seed, mode):
            return {
                "id": row["id"],
                "image_id": row["image_id"],
                "method": method.id,
                "seed": seed,
                "mode": mode,
                "query": row["query"],
            }

        shared_cost = {
            "new_perception_seconds": 0.0,
            "cache_hits": 0,
            "images": 0,
            "feature_seconds": 0.0,
        }
        count_since_flush = 0
        completed_count = sum(
            row["status"] == "ok" for row in ledger.records(latest=True)
        )
        report_path = destination / "report.json"
        if completed_count == expected and report_path.exists():
            report = json.loads(report_path.read_text())
            if (
                report["completed_outputs"] == expected
                and report["expected_outputs"] == expected
                and canonical_hash(report["metadata"]["identity"])
                == canonical_hash(identity)
            ):
                self.stage_reports[stage] = str(report_path)
                return [
                    row["payload"]
                    for row in ledger.records(latest=True)
                    if row["status"] == "ok"
                ], report
        try:
            for group in _groups_by_prompt_length(
                self.runtime, rows, self.config.generation.conv_mode
            ):
                for start in range(0, len(group), self.policy["batch_images"]):
                    batch = group[start : start + self.policy["batch_images"]]
                    pending = []
                    for method in methods:
                        for mode in modes:
                            for seed in seeds if mode == "sample" else (seeds[0],):
                                if any(
                                    not ledger.completed(
                                        key_for(row, method, seed, mode)
                                    )
                                    for row in batch
                                ):
                                    pending.append((method, seed, mode))
                    if not pending:
                        continue
                    self.check_budget()
                    prepared, costs = self.prepare_rows(batch)
                    for row, item in zip(batch, prepared):
                        if prepared_overrides and row["image_id"] in prepared_overrides:
                            override = prepared_overrides[row["image_id"]]
                            item.update(override)
                    for name in costs:
                        shared_cost[name] += costs[name]
                    ids = torch.stack(
                        [
                            self.runtime.prompt(
                                row["query"], self.config.generation.conv_mode
                            )[1]
                            for row in batch
                        ]
                    ).to(self.runtime.device)
                    feature_start = time.perf_counter()
                    unique_methods = list(
                        {method.id: method for method, _, _ in pending}.values()
                    )
                    features = prepare_feature_batch(
                        self.runtime, prepared, unique_methods
                    )
                    synchronize(self.runtime.device)
                    feature_seconds = time.perf_counter() - feature_start
                    shared_cost["feature_seconds"] += feature_seconds
                    for method, seed, mode in pending:
                        self.check_budget()
                        keys = [key_for(row, method, seed, mode) for row in batch]
                        attempts = max(
                            (ledger.latest(key) or {}).get("attempt", 0)
                            for key in keys
                            if not ledger.completed(key)
                        )
                        for _ in range(attempts, self.config.max_retries):
                            try:
                                result = generate(
                                    self.runtime,
                                    self.config,
                                    method,
                                    ids,
                                    method_features(features, method),
                                    [
                                        record_seed(
                                            seed,
                                            row["image_id"],
                                            method.id,
                                            row["query"],
                                        )
                                        for row in batch
                                    ],
                                    greedy=mode == "greedy",
                                    execution=self.policy["execution"],
                                )
                                elapsed = result.prefill_seconds + result.decode_seconds
                                for row, item, key, tokens, reason in zip(
                                    batch,
                                    prepared,
                                    keys,
                                    result.token_ids,
                                    result.stop_reasons,
                                ):
                                    if ledger.completed(key):
                                        continue
                                    text, _, stop = self.runtime.prompt(
                                        row["query"], self.config.generation.conv_mode
                                    )
                                    caption = self.runtime.decode_text(tokens, stop)
                                    if not caption:
                                        raise ValueError(
                                            f"Empty caption for image {row['image_id']}; record/retry instead of treating as a success."
                                        )
                                    payload = {
                                        "id": row["id"],
                                        "image_id": row["image_id"],
                                        "query_id": row["id"],
                                        "image": row["image"],
                                        "caption": caption,
                                        "response": caption,
                                        "method": method.id,
                                        "protocol": method.protocol,
                                        "seed": seed,
                                        "generation_mode": mode,
                                        "prompt": text,
                                        "token_ids": tokens,
                                        "generated_tokens": len(tokens),
                                        "stop_reason": reason,
                                        "generation_seconds": elapsed / len(batch),
                                        "prefill_seconds": result.prefill_seconds
                                        / len(batch),
                                        "decode_seconds": result.decode_seconds
                                        / len(batch),
                                        "peak_memory_bytes": result.peak_memory_bytes,
                                        "actual_batch_images": len(batch),
                                        "branches": result.branch_count,
                                        "view_metadata": item["metadata"],
                                        "perception_key": item["perception_key"],
                                        "perception_metadata": item[
                                            "perception"
                                        ].metadata,
                                    }
                                    if scorer:
                                        payload = scorer.score(
                                            row["image_id"],
                                            caption,
                                            **{
                                                k: v
                                                for k, v in payload.items()
                                                if k not in {"image_id", "caption"}
                                            },
                                        )
                                    if "label" in row:
                                        payload["label"] = row["label"]
                                    ledger.record(key, payload)
                                    count_since_flush += 1
                                    completed_count += 1
                                break
                            except (
                                RuntimeError,
                                ValueError,
                                FloatingPointError,
                            ) as exc:
                                for key in keys:
                                    if not ledger.completed(key):
                                        ledger.record(
                                            key,
                                            {
                                                "error": f"{type(exc).__name__}: {exc}",
                                                "stage": stage,
                                            },
                                            status="error",
                                        )
                                publish_ledger(ledger, snapshots)
                                # Preserve the locked execution/batch policy. OOM is
                                # a failed configuration, never a silent new protocol.
                                if isinstance(exc, torch.cuda.OutOfMemoryError):
                                    raise
                        if count_since_flush >= self.config.snapshot_every:
                            publish_ledger(ledger, snapshots)
                            count_since_flush = 0
                        print(
                            f"{stage}: {completed_count}/{expected} outputs", flush=True
                        )
                    del features, prepared
            latest = ledger.records(latest=True)
            records = [row["payload"] for row in latest if row["status"] == "ok"]
            failed = [row for row in latest if row["status"] != "ok"]
            report = write_report(
                destination,
                records,
                failures=failed,
                expected=expected,
                metadata={
                    "shared_stage_cost": shared_cost,
                    "identity": identity,
                    "shared_stage_cost_scope": "current session only; historical per-image uncached perception timings remain in result metadata",
                    "complete": len(records) == expected and not failed,
                },
                bootstrap_repeats=self.config.bootstrap_replicates,
                bootstrap_seed=self.config.bootstrap_seed,
            )
            self.stage_reports[stage] = str(destination / "report.json")
            if len(records) != expected or failed:
                raise RuntimeError(
                    f"Stage {stage} incomplete ({len(records)}/{expected}); inspect failures before dependent stages."
                )
            return records, report
        finally:
            publish_ledger(ledger, snapshots)
            self.cache.flush()

    def close(self):
        from tas_runtime.persistence import publish_ledger

        for ledger, snapshots in self.ledgers:
            publish_ledger(ledger, snapshots)
        if self.cache:
            self.cache.flush()
        if self.perception:
            self.perception.close()
            self.perception = None

    def run(self):
        from .methods import (
            initial_methods,
            method_by_id,
            harmonized_baselines,
            build_controls,
        )
        from .report import select_finalists

        try:
            self.initialize()
            self.audit()
            stage = self.config.stage
            methods = initial_methods()
            if self.config.methods:
                methods = [method_by_id(name) for name in self.config.methods]
            if stage == "external":
                from .external import run_external

                run_external(self, methods)
                return self.finish("external_complete")
            selection_path = self.persistent_run / "shortlist.json"
            if stage in {"flow", "smoke"}:
                self.collection(
                    "smoke", self.rows("A"), methods, seeds=(self.config.seeds[0],)
                )
                if stage == "smoke" or not self.config.auto_continue_after_smoke:
                    return self.finish("smoke_complete")
            if stage == "audit":
                return self.finish("audit_complete")
            if stage in {"flow", "screen"}:
                screen_methods = methods + harmonized_baselines()
                screen, _ = self.collection(
                    "screen",
                    self.rows("B"),
                    screen_methods,
                    seeds=(self.config.seeds[0],),
                )
                selection = select_finalists(screen, count=self.config.shortlist_count)
                if (
                    selection_path.exists()
                    and json.loads(selection_path.read_text())["selected"]
                    != selection["selected"]
                ):
                    raise ArtifactError(
                        "A frozen shortlist changed; use a new study identity."
                    )
                atomic_json(selection_path, selection)
                if stage == "screen":
                    return self.finish("screen_complete")
            if not selection_path.exists():
                raise RuntimeError(
                    "Dependent stage needs a completed screen/shortlist. Run flow or screen first."
                )
            finalists = [
                method_by_id(name)
                for name in json.loads(selection_path.read_text())["selected"]
            ]
            baselines = [
                method_by_id(name) for name in ("E0", "E1", "F0")
            ] + harmonized_baselines()
            if stage in {"flow", "confirm"} and self.config.run_confirmation:
                if len(self.config.seeds) < 3:
                    raise ValueError(
                        "Confirmation requires at least three predeclared seeds plus greedy."
                    )
                self.collection(
                    "confirm",
                    self.rows("C"),
                    baselines + finalists,
                    modes=("sample", "greedy"),
                )
            if stage in {"flow", "controls"} and self.config.run_controls:
                controls = []
                for method in finalists:
                    controls += build_controls(method)
                    controls += [
                        dataclasses.replace(
                            method, id=method.id + "_BLACK", control="black_context"
                        ),
                        dataclasses.replace(
                            method, id=method.id + "_RANDOM", control="random_region"
                        ),
                        dataclasses.replace(
                            method,
                            id=method.id + "_CONTEXT_ONLY",
                            fusion=method.fusion + "_context",
                            views=("ORIG", method.context_view),
                            control="context_only",
                            context=method.context_view,
                        ),
                    ]
                # Explicit temperature-only controls use native scores at matched
                # generation budgets; configuration-specific temperature is stored.
                for temperature in (0.7, 1.0, 1.3):
                    previous = self.config
                    self.config = dataclasses.replace(
                        previous,
                        generation=dataclasses.replace(
                            previous.generation, temperature=temperature
                        ),
                    )
                    try:
                        temp_baselines = [
                            dataclasses.replace(
                                method, id=f"{method.id}_T{temperature}"
                            )
                            for method in harmonized_baselines()
                        ]
                        self.collection(
                            f"temperature-{temperature}",
                            self.rows("B"),
                            temp_baselines,
                            seeds=(previous.seeds[0],),
                            modes=("sample",),
                        )
                    finally:
                        self.config = previous
                self.collection(
                    "controls",
                    self.rows("B"),
                    baselines + finalists + controls,
                    seeds=(self.config.seeds[0],),
                    modes=("sample", "greedy"),
                )
            if stage in {"flow", "diagnostics"} and self.config.run_diagnostics:
                from .study_diagnostics import run_diagnostics

                run_diagnostics(self, finalists[0])
            if stage == "final" or stage == "flow" and self.config.run_final:
                if not self.config.run_final:
                    raise ValueError(
                        "Opening final data requires explicit run_final=True and a frozen selected policy."
                    )
                self.collection(
                    "final",
                    self.rows("remaining"),
                    baselines + finalists,
                    modes=("sample", "greedy"),
                )
            return self.finish("complete_first_round")
        except SessionBudgetReached as exc:
            return self.finish("session_budget_reached", note=str(exc))
        finally:
            self.close()

    def finish(self, status, note=None):
        result = {
            "status": status,
            "note": note,
            "run_id": getattr(self, "run_id", None),
            "reports": self.stage_reports,
            "elapsed_seconds": time.monotonic() - self.started,
            "deferred": [
                "P3 adaptive",
                "P4 prompts/branch selection",
                "P5 learned controller",
                "FG-G conditional ablation",
                "second backbone validation",
            ],
            "scientific_claim": "No winning method/quality or significance claim is inferred from completion.",
        }
        if hasattr(self, "persistent_run"):
            atomic_json(self.persistent_run / "session.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return result
