"""Scientific summaries, transparent pilot selection and paired image analyses."""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tas_runtime.artifacts import atomic_json, atomic_jsonl

from .evaluation import (
    _pairing_key,
    aggregate_caption_metrics,
    paired_cluster_bootstrap,
)


def summarize(records):
    if not records:
        return {"count": 0, "status": "no_outputs"}
    result = (
        aggregate_caption_metrics(records)
        if "object_mentions" in records[0]
        else {"n_captions": len(records)}
    )
    # Older saved records contain only amortized generation_seconds. They can
    # still supply throughput, but cannot establish request/batch latency.
    elapsed = [
        row.get("amortized_generation_seconds", row.get("generation_seconds", 0.0))
        for row in records
    ]
    latency = [
        row["batch_latency_seconds"]
        for row in records
        if "batch_latency_seconds" in row
    ]
    tokens = [row.get("generated_tokens", 0) for row in records]
    total = sum(elapsed)
    result.update(
        generation_seconds=total,
        captions_per_hour_warm=3600 * len(records) / total if total else None,
        generated_tokens=sum(tokens),
        tokens_per_caption=float(np.mean(tokens)),
        p50_batch_latency_seconds=float(np.percentile(latency, 50))
        if latency
        else None,
        p95_batch_latency_seconds=float(np.percentile(latency, 95))
        if latency
        else None,
        latency_record_count=len(latency),
        p50_amortized_seconds_per_image=float(np.percentile(elapsed, 50)),
        p95_amortized_seconds_per_image=float(np.percentile(elapsed, 95)),
        peak_memory_bytes=max(row.get("peak_memory_bytes", 0) for row in records),
        cost_scope="warm decoder prefill+decode; CLIP features, perception, storage and setup reported separately",
        latency_scope="elapsed prefill+decode until the whole image batch returns; no queue/setup/perception time",
    )
    return result


def grouped_summary(records):
    groups = defaultdict(list)
    for row in records:
        groups[
            (row["method"], row["protocol"], row.get("generation_mode", "sample"))
        ].append(row)
    return [
        {
            "method": method,
            "protocol": protocol,
            "generation_mode": mode,
            **summarize(rows),
        }
        for (method, protocol, mode), rows in sorted(groups.items())
    ]


def select_finalists(records, *, count=2, coverage_tolerance=0.02):
    """Shortlist observed hallucination improvements under a locked coverage rule.

    Coverage is compared with E1-H in the same sampled protocol. Exclude policies
    dominated by another candidate or matched control, then rank at most two.
    This development rule is not a significance or caption-quality guarantee.
    """
    if not math.isfinite(coverage_tolerance) or not 0 <= coverage_tolerance <= 1:
        raise ValueError("coverage_tolerance must be a finite fraction in [0,1].")
    if not 1 <= count <= 2:
        raise ValueError("Shortlist count must be one or two.")
    summaries = [
        row
        for row in grouped_summary(records)
        if row["protocol"] == "HARMONIZED" and row["generation_mode"] == "sample"
    ]
    baseline = next((row for row in summaries if row["method"] == "E1-H"), None)
    if baseline is None or any(
        baseline.get(key) is None for key in ("CHAIRi", "CHAIRs", "category_coverage")
    ):
        raise ValueError(
            "Automatic shortlisting requires scored HARMONIZED sampled E1-H with coverage."
        )
    candidates = [
        row
        for row in summaries
        if row["method"] in {"E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4"}
        and all(
            row.get(key) is not None
            for key in ("CHAIRi", "CHAIRs", "category_coverage")
        )
    ]
    if not candidates:
        raise ValueError(
            "No completely scored P2 candidate exists for automatic confirmation selection."
        )
    candidates.sort(
        key=lambda row: (
            row["CHAIRi"],
            row["CHAIRs"],
            -row["category_coverage"],
            -row["true_mentions"],
            row["method"],
        )
    )
    comparisons = [
        row
        for row in summaries
        if all(
            row.get(key) is not None
            for key in ("CHAIRi", "CHAIRs", "category_coverage")
        )
    ]
    for row in candidates:
        row["coverage_delta"] = row["category_coverage"] - baseline["category_coverage"]
        row["coverage_pass"] = row["coverage_delta"] >= -coverage_tolerance - 1e-12
        row["hallucination_improved"] = (
            row["CHAIRi"] < baseline["CHAIRi"] or row["CHAIRs"] < baseline["CHAIRs"]
        )
        row["pareto_dominated"] = any(
            other["CHAIRi"] <= row["CHAIRi"]
            and other["CHAIRs"] <= row["CHAIRs"]
            and other["category_coverage"] >= row["category_coverage"]
            and (
                other["CHAIRi"] < row["CHAIRi"]
                or other["CHAIRs"] < row["CHAIRs"]
                or other["category_coverage"] > row["category_coverage"]
            )
            for other in comparisons
            if other is not row
        )
        row["eligible"] = (
            row["coverage_pass"]
            and row["hallucination_improved"]
            and not row["pareto_dominated"]
        )
    selected = [row["method"] for row in candidates if row["eligible"]][:count]
    return {
        "schema": "tas-shortlist-v2",
        "selected": selected,
        "candidates": candidates,
        "baseline": baseline,
        "selection_rule": "coverage guard vs E1-H; improve CHAIRi or CHAIRs; nondominated among matched sampled policies; rank CHAIRi/CHAIRs, coverage/true mentions, ID",
        "coverage_tolerance": coverage_tolerance,
        "coverage_units": "absolute category-coverage fraction; 0.01 = one percentage point",
        "status": "shortlisted" if selected else "no_eligible_finalist",
        "interpretation": "development shortlist only; tradeoffs and caption quality require review; no automatic winner/significance claim",
    }


def equal_length_comparison(baseline, candidate):
    """Descriptive paired subset; conditioning on output length is not causal proof."""
    first = {_pairing_key(row): row for row in baseline}
    second = {_pairing_key(row): row for row in candidate}
    if (
        len(first) != len(baseline)
        or len(second) != len(candidate)
        or first.keys() != second.keys()
    ):
        raise ValueError(
            "Equal-length comparison needs unique, aligned image/query/seed records."
        )
    matched = [
        key
        for key in first
        if first[key]["length_words"] == second[key]["length_words"]
    ]
    return {
        "paired_records": len(first),
        "matched_records": len(matched),
        "matched_image_ids": sorted({key[0] for key in matched}),
        "baseline": aggregate_caption_metrics([first[key] for key in matched])
        if matched
        else None,
        "candidate": aggregate_caption_metrics([second[key] for key in matched])
        if matched
        else None,
        "matching_rule": "same image/query/seed and exactly equal word counts; no caption truncation",
        "interpretation": "descriptive selected subset; unequal lengths remain in the main report; does not establish caption quality or causal mechanism",
    }


def write_report(
    directory,
    records,
    *,
    failures=(),
    expected=None,
    metadata=None,
    bootstrap_repeats=1000,
    bootstrap_seed=0,
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    summary = grouped_summary(records)
    paired = []
    equal_length = []
    by_group = defaultdict(list)
    for row in records:
        by_group[
            (row["protocol"], row.get("generation_mode", "sample"), row["method"])
        ].append(row)
    for (protocol, mode, method), rows in by_group.items():
        baseline = "E1" if protocol == "LEGACY" else "E1-H"
        if (
            method == baseline
            or (protocol, mode, baseline) not in by_group
            or "object_mentions" not in rows[0]
        ):
            continue
        try:
            comparison = equal_length_comparison(
                by_group[(protocol, mode, baseline)], rows
            )
        except ValueError as exc:
            comparison = {"status": "unpaired", "reason": str(exc)}
        equal_length.append(
            {
                "baseline_method": baseline,
                "candidate_method": method,
                "protocol": protocol,
                "mode": mode,
                **comparison,
            }
        )
        for metric in ("CHAIRi", "CHAIRs", "category_coverage", "length_words"):
            try:
                result = paired_cluster_bootstrap(
                    by_group[(protocol, mode, baseline)],
                    rows,
                    metric=metric,
                    repeats=bootstrap_repeats,
                    seed=bootstrap_seed,
                )
                paired.append(
                    {
                        "baseline": baseline,
                        "candidate": method,
                        "protocol": protocol,
                        "mode": mode,
                        **result,
                    }
                )
            except ValueError as exc:
                paired.append(
                    {
                        "baseline": baseline,
                        "candidate": method,
                        "metric": metric,
                        "status": "unpaired",
                        "reason": str(exc),
                    }
                )
    output = {
        "summary": summary,
        "paired": paired,
        "equal_length_comparisons": equal_length,
        "failures": list(failures),
        "expected_outputs": expected,
        "completed_outputs": len(records),
        "metadata": metadata or {},
        "limitations": [
            "COCO annotation scope is incomplete; inspect ambiguous mentions.",
            "CHAIR/category coverage/length do not establish caption quality or relations.",
            "LEGACY and HARMONIZED use different geometry/numerics; no matched cross-protocol claim.",
            "Confirmation reused for tuning becomes development; final held-out data remains unopened.",
        ],
    }
    atomic_json(root / "report.json", output)
    atomic_jsonl(root / "captions.jsonl", records)
    atomic_jsonl(root / "failures.jsonl", failures)
    fields = [
        "method",
        "protocol",
        "generation_mode",
        "n_captions",
        "CHAIRi",
        "CHAIRs",
        "category_coverage",
        "true_mentions",
        "length_words",
        "tokens_per_caption",
        "captions_per_hour_warm",
        "p50_batch_latency_seconds",
        "p95_batch_latency_seconds",
        "p50_amortized_seconds_per_image",
        "peak_memory_bytes",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary)
    (root / "summary.csv").write_text(stream.getvalue(), encoding="utf-8")
    return output
