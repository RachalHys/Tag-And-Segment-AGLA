"""Descriptive paired execution comparison; no automatic equivalence decision."""

from __future__ import annotations

from collections import defaultdict
import json
import math

from .evaluation import aggregate_caption_metrics, paired_cluster_bootstrap
from .report import equal_length_comparison


def _index(records):
    indexed = {}
    required = {
        "image_id",
        "method",
        "seed",
        "generation_mode",
        "protocol",
        "query",
        "prompt",
        "caption",
        "object_mentions",
        "false_mentions",
        "true_mentions",
        "CHAIRs",
        "category_coverage",
        "covered_categories",
        "annotated_categories",
        "length_words",
    }
    for record in records:
        if required - record.keys():
            raise ValueError(
                f"Missing comparison fields: {sorted(required - record.keys())}"
            )
        query_id = record.get("question_id", record.get("query_id"))
        if query_id is None or any(
            record[name] is None
            for name in (
                "image_id",
                "method",
                "seed",
                "generation_mode",
                "protocol",
                "query",
                "prompt",
            )
        ):
            raise ValueError(
                "Comparison requires explicit image/query/seed/method/mode identities."
            )
        if not isinstance(record["caption"], str):
            raise ValueError("Comparison captions must be strings.")
        if record["generation_mode"] not in {"sample", "greedy"} or record[
            "protocol"
        ] not in {"LEGACY", "HARMONIZED"}:
            raise ValueError("Unknown generation mode or arithmetic protocol.")
        for name in (
            "object_mentions",
            "false_mentions",
            "true_mentions",
            "CHAIRs",
            "covered_categories",
            "annotated_categories",
            "length_words",
            "generated_tokens",
            "amortized_generation_seconds",
            "peak_memory_bytes",
        ):
            if name in record and (
                not isinstance(record[name], (int, float))
                or not math.isfinite(record[name])
                or record[name] < 0
            ):
                raise ValueError(
                    f"Invalid finite nonnegative comparison value: {name}."
                )
        if record["CHAIRs"] not in (0, 1):
            raise ValueError("CHAIRs must be a binary caption status.")
        coverage = record["category_coverage"]
        if coverage is not None and (
            not isinstance(coverage, (int, float))
            or not math.isfinite(coverage)
            or not 0 <= coverage <= 1
        ):
            raise ValueError("Invalid category coverage.")
        tokens = record.get("token_ids")
        if tokens is not None and (
            not isinstance(tokens, (list, tuple))
            or any(not isinstance(token, int) for token in tokens)
        ):
            raise ValueError("token_ids must be an integer sequence.")
        if (
            tokens is not None
            and "generated_tokens" in record
            and record["generated_tokens"] != len(tokens)
        ):
            raise ValueError("generated_tokens differs from token_ids length.")
        key = tuple(
            map(
                str,
                (
                    record["image_id"],
                    query_id,
                    record["seed"],
                    record["method"],
                    record["generation_mode"],
                ),
            )
        )
        if key in indexed:
            raise ValueError(f"Duplicate execution comparison pair: {key}.")
        indexed[key] = record
    if not indexed:
        raise ValueError("Execution comparison needs nonempty paired outputs.")
    return indexed


def _summary(records):
    n = len(records)
    costs = [record.get("amortized_generation_seconds") for record in records]
    seconds = sum(costs) if all(cost is not None for cost in costs) else None
    peaks = [
        record["peak_memory_bytes"]
        for record in records
        if "peak_memory_bytes" in record
    ]
    stops = [record.get("stop_reason") for record in records]
    tokens = [record.get("generated_tokens") for record in records]
    total_tokens = sum(tokens) if all(value is not None for value in tokens) else None
    return {
        **aggregate_caption_metrics(records),
        "amortized_generation_seconds": seconds,
        "captions_per_hour_warm": 3600 * n / seconds if seconds else None,
        "generated_tokens": total_tokens,
        "tokens_per_caption": total_tokens / n if total_tokens is not None else None,
        "tokens_per_second": total_tokens / seconds
        if total_tokens is not None and seconds
        else None,
        "peak_memory_bytes": max(peaks) if len(peaks) == n else None,
        "eos_or_stop_rate": stops.count("eos_or_stop") / n,
        "token_limit_rate": stops.count("max_new_tokens") / n,
        "unknown_stop_reason_count": sum(
            stop not in {"eos_or_stop", "max_new_tokens"} for stop in stops
        ),
    }


def comparison_report(
    reference_records, candidate_records, *, bootstrap_repeats=1000, bootstrap_seed=0
):
    """Compare matching methods/modes across policies without pooling trajectories.

    Callers separately freeze/validate policy, source, data and generation
    identities. Method IDs stay unchanged so both sides use the same RNG keys.
    """
    if bootstrap_repeats < 100:
        raise ValueError("Need bootstrap_repeats>=100.")
    reference, candidate = _index(reference_records), _index(candidate_records)
    if reference.keys() != candidate.keys():
        raise ValueError(
            "Execution policies have mismatched image/query/seed/method/mode pairs."
        )
    groups = defaultdict(list)
    for key, original in reference.items():
        alternate = candidate[key]
        for field in ("query", "prompt", "protocol"):
            if original[field] != alternate[field]:
                raise ValueError(f"Paired execution {field} mismatch: {key}.")
        if original["annotated_categories"] != alternate["annotated_categories"]:
            raise ValueError(f"Paired annotation denominator mismatch: {key}.")
        for field in (
            "instance_gt_categories",
            "chair_gt_categories",
            "evaluator_version",
        ):
            if original.get(field) != alternate.get(field):
                raise ValueError(f"Paired scoring identity mismatch: {field}.")
        groups[
            (original["method"], original["protocol"], original["generation_mode"])
        ].append((original, alternate))
    summaries = []
    for (method, protocol, mode), pairs in sorted(groups.items()):
        before, after = (list(items) for items in zip(*pairs))
        paired = {}
        for metric in (
            "CHAIRi",
            "CHAIRs",
            "category_coverage",
            "length_words",
            "true_mentions",
            "false_mentions",
        ):
            try:
                paired[metric] = paired_cluster_bootstrap(
                    before,
                    after,
                    metric=metric,
                    repeats=bootstrap_repeats,
                    seed=bootstrap_seed,
                )
            except ValueError as exc:
                if "denominator" not in str(exc):
                    raise
                paired[metric] = {
                    "metric": metric,
                    "status": "undefined",
                    "reason": str(exc),
                    "delta_candidate_minus_baseline": None,
                    "ci_low": None,
                    "ci_high": None,
                }
        baseline, alternate = _summary(before), _summary(after)
        lengths = [
            right["length_words"] - left["length_words"] for left, right in pairs
        ]
        transitions = {
            f"{a}->{b}": sum(
                left["CHAIRs"] == a and right["CHAIRs"] == b for left, right in pairs
            )
            for a in (0, 1)
            for b in (0, 1)
        }
        token_pairs = [
            (left["token_ids"], right["token_ids"])
            for left, right in pairs
            if left.get("token_ids") is not None and right.get("token_ids") is not None
        ]
        token_matches = sum(list(left) == list(right) for left, right in token_pairs)
        seconds_before, seconds_after = (
            row["amortized_generation_seconds"] for row in (baseline, alternate)
        )
        summaries.append(
            {
                "method": method,
                "protocol": protocol,
                "generation_mode": mode,
                "n_pairs": len(pairs),
                "n_image_clusters": len({str(row["image_id"]) for row in before}),
                "reference": baseline,
                "candidate": alternate,
                "paired": paired,
                **{
                    f"{metric}_delta": paired[metric]["delta_candidate_minus_baseline"]
                    for metric in (
                        "CHAIRi",
                        "CHAIRs",
                        "category_coverage",
                        "length_words",
                    )
                },
                "token_pairs_compared": len(token_pairs),
                "exact_token_matches": token_matches,
                "exact_token_match_rate": token_matches / len(pairs)
                if len(token_pairs) == len(pairs)
                else None,
                "exact_caption_match_rate": sum(
                    left["caption"] == right["caption"] for left, right in pairs
                )
                / len(pairs),
                "chair_status_transitions": transitions,
                "paired_length_changes": {
                    "shorter": sum(value < 0 for value in lengths),
                    "same": lengths.count(0),
                    "longer": sum(value > 0 for value in lengths),
                    "mean_candidate_minus_reference": sum(lengths) / len(lengths),
                },
                "speedup": seconds_before / seconds_after
                if seconds_before and seconds_after
                else None,
                "equal_length_comparison": equal_length_comparison(before, after),
            }
        )
    result = {
        "schema": "tas-execution-comparison-v1",
        "status": "descriptive_only",
        "n_pairs": len(reference),
        "summary": summaries,
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": bootstrap_seed,
        "interpretation": "No equivalence claim or automatic promotion; no equivalence margins were supplied. The shortlist's two-point coverage rule is not an execution-equivalence margin.",
        "cost_scope": "Sum per-caption amortized prefill+decode costs once per real image; duplicated batch latencies are not summed. Shared setup/perception/features are excluded.",
        "limitations": [
            "Small development cohorts and one seed cannot establish quality parity.",
            "Intervals resample images, retaining their seeds/queries; an interval containing zero does not establish equivalence.",
            "CHAIR uses incomplete annotations; coverage/length do not measure overall caption quality or relations.",
            "Equal-length subsets are descriptive; all paired captions remain in primary metrics.",
        ],
    }
    # Enforce JSON-safe results rather than serializing NaN/Infinity silently.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Execution report contains nonfinite or non-JSON values."
        ) from exc
    return result
