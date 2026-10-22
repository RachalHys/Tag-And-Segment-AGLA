"""Small annotation-scoped perception audit and fixed-prefix influence probes.

Automatic class/box correspondences are recorded as provisional and inspectable;
they are not independent semantic verification or a causal upper bound. False
region insertion requires separately verified input and is not invented here.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from PIL import Image

from tas_runtime.artifacts import atomic_json, atomic_jsonl, canonical_hash, sha256_file
from tas_runtime.decoding import forced_prefix_logits
from tas_runtime.perception import PerceptionResult, PerceptionRuntime

from .diagnostics import (
    InstanceEvidence,
    fixed_prefix_influence,
    inventory_audit,
    remove_annotation_matched_instance,
    replace_same_instance_evidence,
)
from .evaluation import (
    ChairLexicon,
    _pairing_key,
    aggregate_caption_metrics,
    paired_cluster_bootstrap,
)
from .methods import method_by_id
from .prepared import prepare_feature_batch, prepare_image


def _box_iou(first, second):
    x0, y0 = np.maximum(first[:2], second[:2])
    x1, y1 = np.minimum(first[2:], second[2:])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return overlap / (a + b - overlap) if a + b > overlap else 0.0


def _mask(annotation, height, width):
    from pycocotools import mask as masks

    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        encoded = masks.merge(masks.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        encoded = masks.frPyObjects(segmentation, height, width)
    else:
        encoded = segmentation
    return masks.decode(encoded).astype(bool)


def matched_annotation_instances(
    prediction, annotations, categories, lexicon, *, min_iou=0.5
):
    """Deterministic one-to-one label+box correspondence, with explicit threshold."""
    pairs = []
    labels = prediction.metadata.get("proposed_labels", [])
    for index, (box, label) in enumerate(zip(prediction.boxes, labels)):
        category = lexicon.inverse.get(label)
        if category is None:
            continue
        for annotation in annotations:
            if (
                annotation.get("iscrowd", 0)
                or categories[annotation["category_id"]] != category
            ):
                continue
            x, y, w, h = annotation["bbox"]
            iou = _box_iou(box, np.array([x, y, x + w, y + h]))
            if iou >= min_iou:
                pairs.append((float(iou), index, annotation["id"]))
    assigned_predictions, assigned_annotations, matches = set(), set(), []
    for iou, index, annotation_id in sorted(
        pairs, key=lambda item: (-item[0], item[1], item[2])
    ):
        if (
            index not in assigned_predictions
            and annotation_id not in assigned_annotations
        ):
            matches.append(
                {
                    "prediction_index": index,
                    "annotation_id": annotation_id,
                    "box_iou": iou,
                }
            )
            assigned_predictions.add(index)
            assigned_annotations.add(annotation_id)
    return matches


def paired_intervention_report(reference, interventions, *, repeats=1000, seed=0):
    """Within-method supervised perturbation deltas, resampling whole images."""
    if not reference or not interventions:
        raise ValueError(
            "Diagnostic comparison requires reference and intervention captions"
        )

    def groups(records):
        grouped = defaultdict(list)
        for record in records:
            grouped[
                (record["method"], record["protocol"], record["generation_mode"])
            ].append(record)
        return grouped

    baseline = groups(reference)
    comparisons = []
    for variant, records in interventions.items():
        changed = groups(records)
        if changed.keys() != baseline.keys():
            raise ValueError(
                "Diagnostic intervention methods/protocols/modes differ from D-PRED"
            )
        for group, original in baseline.items():
            candidate = changed[group]
            first = {_pairing_key(record) for record in original}
            second = {_pairing_key(record) for record in candidate}
            if (
                len(first) != len(original)
                or len(second) != len(candidate)
                or first != second
            ):
                raise ValueError(
                    "Diagnostic comparison requires unique, aligned image/query/seed pairs"
                )
            indexed = {_pairing_key(record): record for record in original}
            for record in candidate:
                expected = indexed[_pairing_key(record)]
                for field in ("query", "prompt", "annotated_categories", "instance_gt_categories", "chair_gt_categories", "evaluator_version"):
                    if expected.get(field) != record.get(field):
                        raise ValueError(f"Diagnostic paired caption/scoring identity differs: {field}")
            paired = []
            for metric in (
                "CHAIRi",
                "CHAIRs",
                "category_coverage",
                "true_mentions",
                "false_mentions",
                "length_words",
            ):
                try:
                    result = paired_cluster_bootstrap(
                        original, candidate, metric=metric, repeats=repeats, seed=seed
                    )
                except ValueError as exc:
                    # Empty annotation/mention denominators are undefined,
                    # never perfect scores or a reason to exclude an image.
                    if "denominator" not in str(exc):
                        raise
                    result = {
                        "metric": metric,
                        "status": "undefined",
                        "reason": str(exc),
                    }
                paired.append(result)
            comparisons.append(
                {
                    "intervention": variant,
                    "method": group[0],
                    "protocol": group[1],
                    "generation_mode": group[2],
                    "applied_image_ids": sorted(
                        {
                            record["image_id"]
                            for record in candidate
                            if record["perception_metadata"]["intervention_details"][
                                "applied"
                            ]
                        }
                    ),
                    "mask_changed_image_ids": sorted(
                        {
                            record["image_id"]
                            for record in candidate
                            if record["perception_metadata"]["union_mask_changed"]
                        }
                    ),
                    "reference_metrics": aggregate_caption_metrics(original),
                    "intervention_metrics": aggregate_caption_metrics(candidate),
                    "paired": paired,
                }
            )
    return {
        "schema": "tas-perception-paired-v1",
        "reference": "D-PRED",
        "delta_direction": "intervention minus D-PRED within the same method/protocol/mode",
        "pairing": "image/query/seed; all records retained, including no-match interventions",
        "scope": "COCO annotation-supported, provisionally matched supervised perturbations; no causal attribution, automatic winner or guaranteed oracle bound",
        "comparisons": comparisons,
    }


def run_diagnostics(study, finalist=None):
    rows = study.rows("audit")
    directory = study.persistent_run / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    annotations_path = (
        study.work["data_root"] / "scoring_annotations/instances_val2017.json"
    )
    source = json.loads(annotations_path.read_text())
    annotations_sha256 = sha256_file(annotations_path)
    lexicon = ChairLexicon()
    categories = {
        item["id"]: lexicon.inverse[item["name"]] for item in source["categories"]
    }
    by_image = defaultdict(list)
    for annotation in source["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    audits, overrides = [], {"D-MASK": {}, "D-BOX": {}, "D-MISS": {}}
    for row in rows:
        study.check_budget()
        prediction, original_key, _ = study.perception_for(row)
        annotations = by_image[row["image_id"]]
        lookup = {item["id"]: item for item in annotations}
        matches = matched_annotation_instances(
            prediction, annotations, categories, lexicon
        )
        audit = {
            "image_id": row["image_id"],
            "matches": matches,
            "matching": "provisional one-to-one label/box-IoU>=0.5, reviewed with captions later; not independent semantic verification",
            "inventory": inventory_audit(
                prediction.metadata["inventory"],
                sorted({categories[item["category_id"]] for item in annotations}),
                lexicon,
            ),
            "predicted_box_count": len(prediction.boxes),
            "annotated_instance_count": len(annotations),
            "mask_fraction": float(prediction.mask.mean()),
            "fallback": prediction.metadata["legacy_fallback"],
        }
        with Image.open(study.work["data_root"] / row["image"]) as handle:
            image = handle.convert("RGB")
        labels = prediction.metadata.get("proposed_labels", [])
        predicted_evidence = [
            InstanceEvidence(
                str(index),
                labels[index] if index < len(labels) else "unknown",
                tuple(box),
                mask,
            )
            for index, (box, mask) in enumerate(
                zip(prediction.boxes, prediction.instances)
            )
        ]
        correspondence = {
            str(match["prediction_index"]): str(match["annotation_id"])
            for match in matches
        }
        annotation_evidence = {}
        for match in matches:
            annotation = lookup[match["annotation_id"]]
            x, y, w, h = annotation["bbox"]
            annotated_mask = _mask(annotation, image.height, image.width)
            predicted_mask = prediction.instances[match["prediction_index"]]
            intersection = int(np.logical_and(predicted_mask, annotated_mask).sum())
            union = int(np.logical_or(predicted_mask, annotated_mask).sum())
            match.update(
                mask_iou=intersection / union if union else None,
                predicted_mask_area_pixels=int(predicted_mask.sum()),
                annotation_mask_area_pixels=int(annotated_mask.sum()),
                mask_intersection_pixels=intersection,
                mask_union_pixels=union,
            )
            annotation_evidence[str(annotation["id"])] = InstanceEvidence(
                str(annotation["id"]),
                categories[annotation["category_id"]],
                (x, y, x + w, y + h),
                annotated_mask,
                provenance="COCO annotation",
            )
        audits.append(audit)
        for variant in overrides:
            key = {
                "base_perception": original_key,
                "diagnostic": variant,
                "annotations_sha256": annotations_sha256,
                "matches": matches,
                "version": "same-instance-provisional-v3",
            }
            value = study.cache.get(key)
            if value is not None:
                changed = PerceptionResult.from_cache(value)
            else:
                instances = prediction.instances.copy()
                boxes = prediction.boxes.copy()
                scores = prediction.scores.copy()
                metadata = {
                    **prediction.metadata,
                    "diagnostic": variant,
                    "matches": matches,
                    "annotation_scope": "matched predicted instances only; unmatched predictions retained",
                    "provisional_matching_requires_review": True,
                    "intervention_details": {
                        "intervention": variant,
                        "applied": False,
                        "status": "no_match",
                        "instance_count_before": len(prediction.instances),
                        "instance_count_after": len(prediction.instances),
                    },
                }
                if variant == "D-MASK":
                    changed_evidence, details = replace_same_instance_evidence(
                        predicted_evidence, annotation_evidence, correspondence
                    )
                    if changed_evidence:
                        instances = np.stack([item.mask for item in changed_evidence])
                    qualities = list(
                        metadata.get("sam_predicted_iou", [None] * len(instances))
                    )
                    for match in matches:
                        qualities[match["prediction_index"]] = None
                    metadata["sam_predicted_iou"] = qualities
                    metadata["intervention_details"] = details
                elif variant == "D-BOX" and matches:
                    changed_evidence, details = replace_same_instance_evidence(
                        predicted_evidence,
                        annotation_evidence,
                        correspondence,
                        component="box",
                    )
                    boxes = np.array(
                        [item.box_xyxy for item in changed_evidence], dtype=np.float32
                    )
                    metadata["intervention_details"] = details
                    if study.perception is None:
                        study.perception = PerceptionRuntime(
                            study.work["paths"], confidence=study.config.yolo_conf
                        )
                    rerun = study.perception.run(
                        image,
                        inventory_override=prediction.metadata["inventory"],
                        boxes_override=boxes,
                    )
                    instances = rerun.instances
                    metadata["sam_predicted_iou"] = rerun.metadata["sam_predicted_iou"]
                    metadata["sam_rerun_on_modified_boxes"] = True
                elif variant == "D-MISS" and matches:
                    removed = matches[0]
                    changed_evidence, details = remove_annotation_matched_instance(
                        predicted_evidence,
                        str(removed["prediction_index"]),
                        annotation_id=str(removed["annotation_id"]),
                    )
                    instances = (
                        np.stack([item.mask for item in changed_evidence])
                        if changed_evidence
                        else prediction.instances[:0]
                    )
                    metadata["intervention_details"] = details
                    boxes = np.delete(boxes, removed["prediction_index"], axis=0)
                    scores = np.delete(scores, removed["prediction_index"], axis=0)
                    if "proposed_labels" in metadata:
                        metadata["proposed_labels"] = [
                            label
                            for i, label in enumerate(metadata["proposed_labels"])
                            if i != removed["prediction_index"]
                        ]
                    if "sam_predicted_iou" in metadata:
                        metadata["sam_predicted_iou"] = [
                            quality
                            for i, quality in enumerate(metadata["sam_predicted_iou"])
                            if i != removed["prediction_index"]
                        ]
                    metadata["removed_prediction_index"] = removed["prediction_index"]
                    metadata["matched_present_annotation_id"] = removed["annotation_id"]
                mask = (
                    instances.any(axis=0)
                    if len(instances)
                    else np.zeros_like(prediction.mask)
                )
                metadata.update(
                    mask_fraction=float(mask.mean()),
                    mask_empty=not bool(mask.any()),
                    mask_full=bool(mask.all()),
                    same_instance_count_preserved=len(instances)
                    == len(prediction.instances),
                    union_mask_changed=not np.array_equal(mask, prediction.mask),
                )
                changed = PerceptionResult(mask, instances, boxes, scores, metadata)
                study.cache.put(key, changed.arrays(), changed.metadata)
            prepared = prepare_image(
                study.runtime,
                study.work["data_root"] / row["image"],
                changed,
                study.config,
                random_seed=study.config.split_seed,
            )
            prepared.update(perception=changed, perception_key=key)
            overrides[variant][row["image_id"]] = prepared
    atomic_jsonl(directory / "perception-audit.jsonl", audits)
    methods = [method_by_id("E0"), method_by_id("E1")]
    if finalist is not None:
        methods.append(finalist)
    original_records, original_report = study.collection(
        "diagnostics-D-PRED", rows, methods, seeds=(study.config.seeds[0],)
    )
    intervention_records = {}
    for variant, values in overrides.items():
        intervention_records[variant], _ = study.collection(
            "diagnostics-" + variant,
            rows,
            methods,
            seeds=(study.config.seeds[0],),
            prepared_overrides=values,
            reuse_from=(original_records, original_report),
        )
    probe_execution = {
        **study.policy,
        "batch_images": 1,
        "views": ["ORIG", "OBJ-H", "BH", "FG-U"],
        "scope": "fixed-prefix diagnostic; separate from caption batch numerics",
    }
    probe_inputs = canonical_hash(
        {
            "study": study.identity,
            "generation": study.config.generation,
            "execution": study.policy,
            "probe_execution": probe_execution,
            "methods": [method.to_dict() for method in methods],
            "rows": rows,
            "annotation_sha256": annotations_sha256,
            "bootstrap_replicates": study.config.bootstrap_replicates,
            "bootstrap_seed": study.config.bootstrap_seed,
            "images": {
                row["image"]: sha256_file(study.work["data_root"] / row["image"])
                for row in rows
            },
        }
    )
    completion_path = directory / "diagnostics.json"
    influence_path = directory / "fixed-prefix-influence.jsonl"
    comparison_path = directory / "paired-interventions.json"
    if (
        completion_path.exists()
        and influence_path.exists()
        and comparison_path.exists()
    ):
        completed = json.loads(completion_path.read_text())
        if (
            completed.get("probe_inputs") == probe_inputs
            and completed.get("influence_sha256") == sha256_file(influence_path)
            and completed.get("comparison_sha256") == sha256_file(comparison_path)
        ):
            study.stage_reports["diagnostics"] = str(completion_path)
            study.cache.flush()
            return
    atomic_json(
        comparison_path,
        paired_intervention_report(
            original_records,
            intervention_records,
            repeats=study.config.bootstrap_replicates,
            seed=study.config.bootstrap_seed,
        ),
    )
    # Use one original-generated prefix per image for ALL views. No cached E1
    # logits are treated as a caption generated by another policy.
    influence = []
    originals = {
        record["image_id"]: record
        for record in original_records
        if record["method"] == "E0"
    }
    for row in rows:
        study.check_budget()
        prepared, _ = study.prepare_rows([row])
        fake_methods = [method_by_id("F1"), method_by_id("F2")]
        feature_batch = prepare_feature_batch(study.runtime, prepared, fake_methods)
        views = {name: feature_batch[name] for name in ("ORIG", "OBJ-H", "BH", "FG-U")}
        ids = (
            study.runtime.prompt(row["query"], study.config.generation.conv_mode)[1]
            .unsqueeze(0)
            .to(study.runtime.device)
        )
        prefix = originals[row["image_id"]]["token_ids"][:8]
        snapshots = forced_prefix_logits(
            study.runtime.backend,
            ids,
            views,
            prefix,
            execution=study.policy["execution"],
        )
        for position, logits in enumerate(snapshots):
            prefixes = {name: prefix[:position] for name in views}
            probe = fixed_prefix_influence(
                logits, prefix_ids_by_view=prefixes, temperature=1.0
            )
            influence.append(
                {
                    "image_id": row["image_id"],
                    "prefix": prefix[:position],
                    **{
                        name: float(probe[name][0])
                        for name in ("S_context", "S_remove", "S_blur")
                    },
                    "measurement_temperature": 1.0,
                    "execution_protocol": study.config.execution_protocol,
                    "probe_execution": probe_execution,
                    "before_filter": True,
                    "interpretation": probe["interpretation"],
                }
            )
        del feature_batch
    atomic_jsonl(influence_path, influence)
    atomic_json(
        completion_path,
        {
            "cohort_size": len(rows),
            "methods": [method.id for method in methods],
            "annotation_source_sha256": annotations_sha256,
            "perception_audit": "perception-audit.jsonl",
            "influence": "fixed-prefix-influence.jsonl",
            "paired_interventions": comparison_path.name,
            "probe_inputs": probe_inputs,
            "caption_execution_protocol": study.config.execution_protocol,
            "caption_execution_policy": study.policy,
            "probe_execution": probe_execution,
            "influence_sha256": sha256_file(influence_path),
            "comparison_sha256": sha256_file(comparison_path),
            "automatically_run": ["D-PRED", *overrides],
            "not_automatically_run": {
                "D-INV": "optional downstream rerun; explicit supervised diagnostic",
                "D-FALSE": "needs an independently verified false-region manifest",
                "IG_correctness_or_gating": "requires verified present/absent phrases and held-out validation",
            },
            "scope": "annotation-scoped, provisional correspondences; no causal decomposition or guaranteed oracle upper bound",
        },
    )
    study.stage_reports["diagnostics"] = str(directory / "diagnostics.json")
    study.cache.flush()
