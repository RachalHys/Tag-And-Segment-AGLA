"""Controlled perception interventions and fixed-prefix influence diagnostics.

Annotations are diagnostic supervision, never inference-time truth or a gate.
Changing an inventory/box requires rerunning downstream perception; this file
does not pretend that merely relabeling an existing mask performs that rerun.
Influence/JS are perturbation measures, not correctness or full-answer IG.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .views import binary_mask


@dataclass(frozen=True)
class InstanceEvidence:
    instance_id: str
    label: str
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None
    provenance: str = "prediction"
    detector_score: float | None = None
    mask_score: float | None = None
    annotation_id: str | None = None

    def __post_init__(self):
        if (
            not self.instance_id
            or len(self.box_xyxy) != 4
            or not all(math.isfinite(x) for x in self.box_xyxy)
        ):
            raise ValueError("Instances need an id and finite xyxy box")
        x0, y0, x1, y1 = self.box_xyxy
        if x1 < x0 or y1 < y0:
            raise ValueError("Invalid xyxy box ordering")
        if self.mask is not None:
            if np.asarray(self.mask).ndim != 2:
                raise ValueError("Instance masks must be HxW")
            binary_mask(self.mask, np.asarray(self.mask).shape)


def union_instances(
    instances: Sequence[InstanceEvidence], shape: tuple[int, int]
) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    ids = [instance.instance_id for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate instance IDs")
    for instance in instances:
        if instance.mask is None:
            raise ValueError(
                f"Instance {instance.instance_id} needs downstream segmentation before union"
            )
        union |= binary_mask(instance.mask, shape)
    return union


def replace_same_instance_evidence(
    predicted: Sequence[InstanceEvidence],
    annotations: Mapping[str, InstanceEvidence],
    matches: Mapping[str, str],
    *,
    component: str = "mask",
) -> tuple[list[InstanceEvidence], dict]:
    """D-MASK/D-BOX on an explicitly matched subset, preserving instance count.

    Matching is externally audited, not inferred from a label or pretty mask.
    Unmatched predictions are retained.  D-BOX clears matched masks so SAM must
    be rerun on the new boxes; returning stale masks would confound the audit.
    """
    if component not in {"mask", "box"}:
        raise ValueError("component must be mask or box")
    predicted_ids = {instance.instance_id for instance in predicted}
    if len(predicted_ids) != len(predicted) or not set(matches) <= predicted_ids:
        raise ValueError("Matches must name unique predicted instance IDs")
    if len(set(matches.values())) != len(matches):
        raise ValueError("One-to-one same-instance matching is required")
    missing = set(matches.values()) - set(annotations)
    if missing:
        raise ValueError(f"Missing annotation instances: {sorted(missing)}")
    output = []
    for instance in predicted:
        if instance.instance_id not in matches:
            output.append(instance)
            continue
        annotation_id = matches[instance.instance_id]
        annotation = annotations[annotation_id]
        if component == "mask":
            if annotation.mask is None:
                raise ValueError("D-MASK needs a real annotation mask")
            output.append(
                replace(
                    instance,
                    mask=np.array(annotation.mask, copy=True),
                    annotation_id=annotation_id,
                    provenance="D-MASK:same-instance-annotation",
                    mask_score=None,
                )
            )
        else:
            output.append(
                replace(
                    instance,
                    box_xyxy=annotation.box_xyxy,
                    mask=None,
                    annotation_id=annotation_id,
                    provenance="D-BOX:same-instance-annotation",
                    mask_score=None,
                )
            )
    return output, {
        "intervention": "D-MASK" if component == "mask" else "D-BOX",
        "same_instance_matches": dict(matches),
        "instance_count_before": len(predicted),
        "instance_count_after": len(output),
        "requires_sam_rerun": component == "box",
        "annotation_scope": "matched instances only; not all objects; not a guaranteed upper bound",
    }


def remove_verified_instance(
    instances: Sequence[InstanceEvidence],
    instance_id: str,
    *,
    verified_present_ids: Sequence[str],
) -> tuple[list[InstanceEvidence], dict]:
    if instance_id not in verified_present_ids:
        raise ValueError("D-MISS must name an independently verified present instance")
    if sum(instance.instance_id == instance_id for instance in instances) != 1:
        raise ValueError("D-MISS target must occur exactly once")
    return [
        instance for instance in instances if instance.instance_id != instance_id
    ], {
        "intervention": "D-MISS",
        "removed_instance_id": instance_id,
        "verified_present": True,
    }


def add_verified_false_region(
    instances: Sequence[InstanceEvidence],
    false_region: InstanceEvidence,
    *,
    absence_verified: bool,
    verification_note: str,
) -> tuple[list[InstanceEvidence], dict]:
    if not absence_verified or not verification_note.strip():
        raise ValueError(
            "D-FALSE needs explicit independent absence verification, not just missing COCO annotation"
        )
    if false_region.instance_id in {instance.instance_id for instance in instances}:
        raise ValueError("False-region id collides with a prediction")
    if false_region.mask is None:
        raise ValueError("D-FALSE needs the injected region mask")
    injected = replace(false_region, provenance="D-FALSE:independently-verified")
    return [*instances, injected], {
        "intervention": "D-FALSE",
        "added_instance_id": injected.instance_id,
        "absence_verified": True,
        "verification_note": verification_note,
    }


def swap_labels_only(
    instances: Sequence[InstanceEvidence],
    replacements: Mapping[str, str],
    *,
    labels_enter_decoder_prompt: bool = False,
) -> tuple[list[InstanceEvidence], dict]:
    if not set(replacements) <= {instance.instance_id for instance in instances}:
        raise ValueError("Unknown instance in label-swap")
    return [
        replace(instance, label=replacements.get(instance.instance_id, instance.label))
        for instance in instances
    ], {
        "intervention": "label-swap-fixed-pixels",
        "labels_enter_decoder_prompt": labels_enter_decoder_prompt,
        "expected_pixel_generation_change": labels_enter_decoder_prompt,
        "note": "With fixed pixels and no inventory in prompt this must be a no-op for decoding; inventory changes BEFORE detection are a different intervention.",
    }


def inventory_audit(
    predicted_labels: Sequence[str], annotated_categories: Sequence[str], lexicon
) -> dict:
    """Annotation-scoped inventory precision/recall; ambiguous labels retained."""
    expected = set(annotated_categories)
    mapped, unknown = set(), []
    for label in predicted_labels:
        direct = lexicon.inverse.get(label.lower())
        candidates = (
            {direct} if direct is not None else set(lexicon.caption_to_words(label)[1])
        )
        if len(candidates) == 1:
            mapped.update(candidates)
        else:
            unknown.append(label)
    correct = mapped & expected
    return {
        "mapped_inventory": sorted(mapped),
        "unknown_or_ambiguous_labels": unknown,
        "annotation_categories": sorted(expected),
        "matched_categories": sorted(correct),
        "missed_categories": sorted(expected - mapped),
        "annotation_unsupported_categories": sorted(mapped - expected),
        "precision_mapped_annotation_scope": len(correct) / len(mapped)
        if mapped
        else None,
        "recall_annotation_scope": len(correct) / len(expected) if expected else None,
        "scope": "COCO annotation categories; unannotated entities are not proven absent",
    }


def js_from_logits(first, second, *, temperature: float = 1.0):
    """Per-example Jensen-Shannon divergence in NATURAL-log units, FP32."""
    import torch
    import torch.nn.functional as F

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Measurement temperature must be positive and fixed")
    if first.shape != second.shape or first.ndim < 1:
        raise ValueError("Distributions must share shape and final vocabulary axis")
    lp = F.log_softmax(first.float() / temperature, dim=-1)
    lq = F.log_softmax(second.float() / temperature, dim=-1)
    lm = torch.logaddexp(lp, lq) - math.log(2)
    p, q = lp.exp(), lq.exp()
    klp = torch.where(p > 0, p * (lp - lm), torch.zeros_like(p)).sum(dim=-1)
    klq = torch.where(q > 0, q * (lq - lm), torch.zeros_like(q)).sum(dim=-1)
    return torch.clamp((klp + klq) / 2, min=0)


def fixed_prefix_influence(
    logits_by_view: Mapping,
    *,
    prefix_ids_by_view: Mapping[str, Sequence[int]],
    temperature: float = 1.0,
) -> dict:
    """Probe all views at identical teacher-forced prefix BEFORE filtering.

    This is not replayed free-run evaluation and never controls generation.
    Prefix identity is mandatory to catch the scientifically invalid comparison
    of distributions taken from independently generated captions.
    """
    import torch.nn.functional as F

    required = {"ORIG", "OBJ-H", "BH", "FG-U"}
    if not required <= set(logits_by_view) or not required <= set(prefix_ids_by_view):
        raise ValueError("Influence probe requires all four views and their prefix IDs")
    prefixes = [tuple(prefix_ids_by_view[view]) for view in sorted(required)]
    if any(prefix != prefixes[0] for prefix in prefixes[1:]):
        raise ValueError(
            "Influence distributions must use identical teacher-forced prefixes"
        )
    original = logits_by_view["ORIG"]
    lp0 = F.log_softmax(original.float() / temperature, dim=-1)
    output = {}
    for view, influence, difference in [
        ("OBJ-H", "S_context", "D_context"),
        ("BH", "S_remove", "D_remove"),
        ("FG-U", "S_blur", "D_blur"),
    ]:
        output[influence] = js_from_logits(
            original, logits_by_view[view], temperature=temperature
        )
        output[difference] = lp0 - F.log_softmax(
            logits_by_view[view].float() / temperature, dim=-1
        )
    output["temperature"] = temperature
    output["interpretation"] = (
        "fixed-prefix view influence; not correctness, hallucination probability, or full-answer IG"
    )
    return output


def teacher_forced_phrase_log_probability(
    logits, token_ids: Sequence[int], *, temperature: float = 1.0
):
    """Sum log probabilities of EVERY phrase token at its correct prefix.

    logits[...,t,:] must predict token_ids[t], after conditioning on the shared
    initial prefix and phrase tokens [:t].  The caller must actually run teacher
    forcing; repeating one next-token distribution is not phrase scoring.
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if logits.ndim not in {2, 3} or logits.shape[-2] != len(token_ids) or not token_ids:
        raise ValueError(
            "Need [phrase_tokens,vocab] or [batch,phrase_tokens,vocab] and a nonempty aligned phrase"
        )
    if any(token < 0 or token >= logits.shape[-1] for token in token_ids):
        raise ValueError("Phrase token outside vocabulary")
    ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    ids = ids.reshape(*([1] * (logits.ndim - 2)), len(token_ids), 1).expand(
        *logits.shape[:-1], 1
    )
    return (
        F.log_softmax(logits.float() / temperature, dim=-1)
        .gather(-1, ids)
        .squeeze(-1)
        .sum(-1)
    )


def matched_candidate_log_odds_change(
    original_present, original_absent, fused_present, fused_absent
):
    """Offline diagnostic from independently verified present/absent phrases.

    Inputs are full phrase log probabilities.  No ground-truth information may
    be used to choose a final-run inference policy through this function.
    """
    return (fused_present - fused_absent) - (original_present - original_absent)
