"""Small annotation-scoped perception audit and fixed-prefix influence probes.

Automatic class/box correspondences are recorded as provisional and inspectable;
they are not independent semantic verification or a causal upper bound. False
region insertion requires separately verified input and is not invented here.
"""

from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
from PIL import Image

from tas_runtime.artifacts import atomic_json, atomic_jsonl, canonical_hash, sha256_file
from tas_runtime.perception import PerceptionRuntime, PerceptionResult
from tas_runtime.decoding import forced_prefix_logits
from .diagnostics import (
    InstanceEvidence,
    inventory_audit,
    fixed_prefix_influence,
    replace_same_instance_evidence,
    remove_verified_instance,
)
from .evaluation import ChairLexicon
from .methods import method_by_id
from .prepared import prepare_image, prepare_feature_batch


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


def run_diagnostics(study, finalist):
    rows = study.rows("audit")
    directory = study.persistent_run / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    annotations_path = (
        study.work["data_root"] / "scoring_annotations/instances_val2017.json"
    )
    source = json.loads(annotations_path.read_text())
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
        prediction, original_key, cached = study.perception_for(row)
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
        audits.append(audit)
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
            annotation_evidence[str(annotation["id"])] = InstanceEvidence(
                str(annotation["id"]),
                categories[annotation["category_id"]],
                (x, y, x + w, y + h),
                _mask(annotation, image.height, image.width),
                provenance="COCO annotation",
            )
        for variant in overrides:
            key = {
                "base_perception": original_key,
                "diagnostic": variant,
                "annotations_sha256": sha256_file(annotations_path),
                "matches": matches,
                "version": "same-instance-provisional-v2",
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
                    "same_instance_count_preserved": variant != "D-MISS",
                    "annotation_scope": "matched predicted instances only; unmatched predictions retained",
                    "provisional_matching_requires_review": True,
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
                    changed_evidence, details = remove_verified_instance(
                        predicted_evidence,
                        str(removed["prediction_index"]),
                        verified_present_ids=correspondence,
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
                    metadata["verified_present_annotation_id"] = removed[
                        "annotation_id"
                    ]
                mask = (
                    instances.any(axis=0)
                    if len(instances)
                    else np.zeros_like(prediction.mask)
                )
                metadata.update(
                    mask_fraction=float(mask.mean()),
                    mask_empty=not bool(mask.any()),
                    mask_full=bool(mask.all()),
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
    methods = [method_by_id("E0"), method_by_id("E1"), finalist]
    original_records, _ = study.collection(
        "diagnostics-D-PRED", rows, methods, seeds=(study.config.seeds[0],)
    )
    for variant, values in overrides.items():
        study.collection(
            "diagnostics-" + variant,
            rows,
            methods,
            seeds=(study.config.seeds[0],),
            prepared_overrides=values,
        )
    probe_inputs = canonical_hash(
        {
            "study": study.identity,
            "generation": study.config.generation,
            "execution": study.policy,
            "methods": [method.to_dict() for method in methods],
            "rows": rows,
            "annotation_sha256": sha256_file(annotations_path),
            "images": {
                row["image"]: sha256_file(study.work["data_root"] / row["image"])
                for row in rows
            },
        }
    )
    completion_path = directory / "diagnostics.json"
    influence_path = directory / "fixed-prefix-influence.jsonl"
    if completion_path.exists() and influence_path.exists():
        completed = json.loads(completion_path.read_text())
        if completed.get("probe_inputs") == probe_inputs and completed.get(
            "influence_sha256"
        ) == sha256_file(influence_path):
            study.stage_reports["diagnostics"] = str(completion_path)
            study.cache.flush()
            return
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
            "annotation_source_sha256": sha256_file(annotations_path),
            "perception_audit": "perception-audit.jsonl",
            "influence": "fixed-prefix-influence.jsonl",
            "probe_inputs": probe_inputs,
            "influence_sha256": sha256_file(influence_path),
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
