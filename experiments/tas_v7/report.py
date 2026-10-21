"""Scientific summaries, transparent pilot selection and paired image analyses."""

from __future__ import annotations

from collections import defaultdict
import csv
import io
from pathlib import Path

import numpy as np

from tas_runtime.artifacts import atomic_json, atomic_jsonl
from .evaluation import aggregate_caption_metrics, paired_cluster_bootstrap


def summarize(records):
    if not records:
        return {"count": 0, "status": "no_outputs"}
    result = (
        aggregate_caption_metrics(records)
        if "object_mentions" in records[0]
        else {"n_captions": len(records)}
    )
    elapsed = [row.get("generation_seconds", 0.0) for row in records]
    tokens = [row.get("generated_tokens", 0) for row in records]
    total = sum(elapsed)
    result.update(
        generation_seconds=total,
        captions_per_hour_warm=3600 * len(records) / total if total else None,
        generated_tokens=sum(tokens),
        tokens_per_caption=float(np.mean(tokens)),
        p50_seconds=float(np.percentile(elapsed, 50)),
        p95_seconds=float(np.percentile(elapsed, 95)),
        peak_memory_bytes=max(row.get("peak_memory_bytes", 0) for row in records),
        cost_scope="warm decoder prefill+decode; CLIP features, perception, storage and setup reported separately",
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


def select_finalists(records, *, count=2):
    """Predeclared exploratory ranking; never claims statistical significance.

    Rank P2 candidates by pooled CHAIRi, then CHAIRs, then coverage, then
    true-object mentions. Also report their Pareto status and baseline deltas.
    No unapproved coverage-loss threshold is silently imposed.
    """
    summaries = grouped_summary(records)
    candidates = [
        row
        for row in summaries
        if row["method"] in {"E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4"}
        and row.get("CHAIRi") is not None
    ]
    if not candidates:
        raise ValueError(
            "No completely scored P2 candidate exists for automatic confirmation selection."
        )
    candidates.sort(
        key=lambda row: (
            row["CHAIRi"],
            row["CHAIRs"],
            -(row["category_coverage"] or 0),
            -row["true_mentions"],
            row["method"],
        )
    )
    for row in candidates:
        row["pareto_dominated"] = any(
            other["CHAIRi"] <= row["CHAIRi"]
            and other["CHAIRs"] <= row["CHAIRs"]
            and (other["category_coverage"] or 0) >= (row["category_coverage"] or 0)
            and (
                other["CHAIRi"] < row["CHAIRi"]
                or other["CHAIRs"] < row["CHAIRs"]
                or (other["category_coverage"] or 0) > (row["category_coverage"] or 0)
            )
            for other in candidates
            if other is not row
        )
    return {
        "selected": [row["method"] for row in candidates[:count]],
        "candidates": candidates,
        "selection_rule": "CHAIRi, CHAIRs ascending; category coverage, true mentions descending; ID tie-break",
        "coverage_tolerance": None,
        "interpretation": "development shortlist only; coverage/quality tradeoffs require joint review; no automatic winner claim",
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
        "peak_memory_bytes",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary)
    (root / "summary.csv").write_text(stream.getvalue(), encoding="utf-8")
    return output
