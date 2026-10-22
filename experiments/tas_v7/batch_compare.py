"""A bounded execution comparison; never qualifies a policy for strict P0."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from itertools import zip_longest
import json

import torch

from tas_runtime.artifacts import (
    ArtifactError,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
)
from tas_runtime.decoding import final_scores, forced_prefix_logits, record_seed
from tas_runtime.progress import progress, step

from .audit import compare_snapshots, decode_options, generate
from .methods import fuse_logits, harmonized_baselines, method_by_id, plausibility_mask
from .prepared import method_features, prepare_feature_batch


@contextmanager
def using_policy(study, policy):
    previous = study.policy
    study.policy = dict(policy)
    study.runtime.set_attention(policy["backend"])
    study.runtime.backend.last_position_logits = policy["last_position_logits"]
    try:
        yield
    finally:
        study.policy = previous
        study.runtime.set_attention(previous["backend"])
        study.runtime.backend.last_position_logits = previous["last_position_logits"]


@torch.inference_mode()
def history_logits(backend, ids, views, histories, position, *, execution, pad_id):
    """Replay natural, per-image histories; retain only the requested final logits.

    Finished companion rows use pad tokens. They cannot attend to another image;
    keeping them preserves the candidate's execution shape and row positions.
    """
    batch, names = len(ids), list(views)
    if len(histories) != batch or position < 0:
        raise ValueError(
            "One history per image and a nonnegative position are required."
        )
    if execution == "packed":
        states = [
            backend.prefill(ids.repeat(len(names), 1), torch.cat(list(views.values())))
        ]
    elif execution == "serial":
        states = [backend.prefill(ids, views[name]) for name in names]
    else:
        raise ValueError("Unknown execution policy.")
    tokens = torch.tensor(
        [row[:position] + [pad_id] * max(0, position - len(row)) for row in histories],
        device=ids.device,
        dtype=torch.long,
    )
    for index in range(position):
        current = tokens[:, index : index + 1]
        if execution == "packed":
            states = [backend.advance(current.repeat(len(names), 1), states[0])]
        else:
            states = [backend.advance(current, state) for state in states]
    if execution == "packed":
        return {
            name: states[0].logits[i * batch : (i + 1) * batch]
            for i, name in enumerate(names)
        }
    return {name: state.logits for name, state in zip(names, states)}


def first_differences(reference, candidate, rows):
    result = []
    for index, (first, second) in enumerate(zip(reference, candidate)):
        for position, (left, right) in enumerate(zip_longest(first, second)):
            if left != right:
                result.append(
                    {
                        "row": index,
                        "image_id": rows[index]["image_id"],
                        "position": position,
                        "reference_token": left,
                        "candidate_token": right,
                    }
                )
                break
    return result


def _distribution_check(runtime, config, method, reference, candidate, difference):
    """All fusion/filter calculations stay on the inference device and dtype."""
    try:
        strict = {
            "status": "within_tolerance",
            **compare_snapshots(
                [reference],
                [candidate],
                protocol=method.protocol,
                beta=method.beta,
                support=method.support,
            ),
        }
    except AssertionError as exc:
        strict = {"status": "outside_strict_checks", "error": str(exc)}
    scores, probabilities = [], []
    masks = []
    for by_view in (reference, candidate):
        support = (
            (
                lambda original: plausibility_mask(
                    original, method.beta, mode=method.protocol
                )
            )
            if method.support
            else None
        )
        fused = fuse_logits(by_view, method)
        greedy = final_scores(
            fused,
            by_view["ORIG"],
            support,
            decode_options(runtime, config, method, greedy=True),
        )
        sampled = final_scores(
            fused, by_view["ORIG"], support, decode_options(runtime, config, method)
        )
        probability = sampled.softmax(-1)
        if not torch.isfinite(probability).all():
            raise FloatingPointError(
                "Nonfinite probabilities on a replayed natural prefix."
            )
        scores.append(greedy)
        probabilities.append(probability.float())
        masks.append(
            support(by_view["ORIG"])
            if support
            else torch.ones_like(greedy, dtype=torch.bool)
        )
    top = []
    for value in scores:
        values, indices = value[0].topk(2)
        top.append(
            {
                "token_ids": indices.tolist(),
                "scores": [float(v) if torch.isfinite(v) else None for v in values],
                "top_two_margin": float(values[0] - values[1])
                if torch.isfinite(values[1])
                else None,
            }
        )
    return {
        "strict_numerical_check": strict,
        "max_branch_logit_error": {
            name: float((reference[name].float() - candidate[name].float()).abs().max())
            for name in reference
        },
        "support_token_changes": int((masks[0] != masks[1]).sum()),
        "sample_distribution_total_variation": float(
            0.5 * (probabilities[0] - probabilities[1]).abs().sum()
        ),
        "reference_top_two": top[0],
        "candidate_top_two": top[1],
        "replay_matches_recorded_choices": (
            top[0]["token_ids"][0] == difference["reference_token"]
            and top[1]["token_ids"][0] == difference["candidate_token"]
        ),
    }


def _benchmark(study, method, ids, views, seeds, policy, token_limit):
    results = []
    with using_policy(study, policy):
        size = policy["batch_images"]
        # Warm the exact shape. Profiler measurements never enter selection.
        generate(
            study.runtime,
            study.config,
            method,
            ids[:size],
            {key: value[:size] for key, value in views.items()},
            seeds[:size],
            execution=policy["execution"],
            max_new_tokens=token_limit,
        )
        for start in range(0, len(ids), size):
            study.check_budget()
            result = generate(
                study.runtime,
                study.config,
                method,
                ids[start : start + size],
                {key: value[start : start + size] for key, value in views.items()},
                seeds[start : start + size],
                execution=policy["execution"],
                max_new_tokens=token_limit,
            )
            results.append(result)
    seconds = sum(item.prefill_seconds + item.decode_seconds for item in results)
    tokens = sum(len(row) for item in results for row in item.token_ids)
    return {
        "seconds": seconds,
        "generated_tokens": tokens,
        "tokens_per_second": tokens / seconds,
        "captions_per_hour": 3600 * len(ids) / seconds,
        "peak_memory_bytes": max(item.peak_memory_bytes for item in results),
        "token_limit": token_limit,
        "scope": "warmed sampled F4 only; not a complete-caption speed promise",
    }


def discover(study, rows, methods, reference_policy, directory):
    """Measure serial/packed image batches, retaining exact-token drift as data."""
    config, runtime = study.config, study.runtime
    count = max(config.batch_sizes)
    if count < 2 or count > len(rows):
        raise ValueError(
            "Batch comparison needs an image batch >=2 and at least that many comparison images."
        )
    rows = rows[:count]
    prepared, costs = study.prepare_rows(rows, label="Batch comparison views")
    with step("Shared singleton-encoded visual features"):
        features = prepare_feature_batch(runtime, prepared, methods)
    ids = torch.stack(
        [runtime.prompt(row["query"], config.generation.conv_mode)[1] for row in rows]
    ).to(runtime.device)
    forced = [
        int(runtime.tokenizer.bos_token_id or 1),
        int(runtime.tokenizer.eos_token_id),
        1,
    ]
    seeds = {
        method.id: [
            record_seed(config.seeds[0], row["image_id"], method.id, row["query"])
            for row in rows
        ]
        for method in methods
    }
    references = {}
    with using_policy(study, reference_policy):
        for method in progress(
            methods, desc="B=1 comparison references", unit="method"
        ):
            views = method_features(features, method)
            reference = {"snapshots": [], "greedy": []}
            for index in range(count):
                study.check_budget()
                single = {key: value[index : index + 1] for key, value in views.items()}
                reference["snapshots"].append(
                    forced_prefix_logits(
                        runtime.backend,
                        ids[index : index + 1],
                        single,
                        forced,
                        execution="serial",
                    )
                )
                reference["greedy"] += generate(
                    runtime,
                    config,
                    method,
                    ids[index : index + 1],
                    single,
                    [seeds[method.id][index]],
                    greedy=True,
                    execution="serial",
                    max_new_tokens=config.pilot_tokens,
                ).token_ids
            references[method.id] = reference
    f4 = next(method for method in methods if method.id == "F4")
    token_limit = min(config.generation.max_new_tokens, 48)
    report = {
        "status": "running",
        "reference_policy": reference_policy,
        "candidates": [],
        "shared_perception_cost": costs,
        "selection": "largest configured image batch; same reference attention/head; fastest sampled F4 tokens/second before quality scores",
    }
    report["reference_benchmark"] = _benchmark(
        study,
        f4,
        ids,
        method_features(features, f4),
        seeds[f4.id],
        reference_policy,
        token_limit,
    )
    executions = (
        ("serial", "packed") if config.execution == "auto" else (config.execution,)
    )
    for execution in progress(
        executions, desc="Batched execution probes", unit="policy"
    ):
        policy = {**reference_policy, "batch_images": count, "execution": execution}
        item = {"policy": policy, "methods": {}}
        try:
            with using_policy(study, policy):
                for method in methods:
                    study.check_budget()
                    views, reference = (
                        method_features(features, method),
                        references[method.id],
                    )
                    actual = forced_prefix_logits(
                        runtime.backend, ids, views, forced, execution=execution
                    )
                    expected = [
                        {
                            name: torch.cat(
                                [row[step][name] for row in reference["snapshots"]]
                            )
                            for name in method.views
                        }
                        for step in range(len(forced) + 1)
                    ]
                    check = compare_snapshots(
                        expected,
                        actual,
                        protocol=method.protocol,
                        beta=method.beta,
                        support=method.support,
                    )
                    candidate = generate(
                        runtime,
                        config,
                        method,
                        ids,
                        views,
                        seeds[method.id],
                        greedy=True,
                        execution=execution,
                        max_new_tokens=config.pilot_tokens,
                    ).token_ids
                    differences = first_differences(
                        reference["greedy"], candidate, rows
                    )
                    check.update(
                        exact_greedy_match=not differences,
                        first_differences=differences,
                    )
                    item["methods"][method.id] = check
                    for difference in differences:
                        study.check_budget()
                        index, position = difference["row"], difference["position"]
                        if (
                            difference["reference_token"] is None
                            or difference["candidate_token"] is None
                        ):
                            raise AssertionError(
                                "Token-length-only drift after a shared EOS is a stopping error."
                            )
                        # Companion histories use the actual candidate's prefixes;
                        # this target row has an identical prefix in both runs.
                        pad_id = decode_options(runtime, config, method).pad_token_id
                        predicted = history_logits(
                            runtime.backend,
                            ids,
                            views,
                            candidate,
                            position,
                            execution=execution,
                            pad_id=pad_id,
                        )
                        candidate_logits = {
                            key: value[index : index + 1]
                            for key, value in predicted.items()
                        }
                        with using_policy(study, reference_policy):
                            expected_logits = history_logits(
                                runtime.backend,
                                ids[index : index + 1],
                                {
                                    key: value[index : index + 1]
                                    for key, value in views.items()
                                },
                                [reference["greedy"][index]],
                                position,
                                execution="serial",
                                pad_id=pad_id,
                            )
                        difference["natural_prefix"] = _distribution_check(
                            runtime,
                            config,
                            method,
                            expected_logits,
                            candidate_logits,
                            difference,
                        )
                        del predicted, expected_logits, candidate_logits
            item["benchmark"] = _benchmark(
                study,
                f4,
                ids,
                method_features(features, f4),
                seeds[f4.id],
                policy,
                token_limit,
            )
            item["status"] = "measured"
        except (RuntimeError, AssertionError, FloatingPointError) as exc:
            item.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["candidates"].append(item)
        print(f"Batched comparison {execution}/B={count}: {item['status']}", flush=True)
        if "error" in item:
            print(item["error"], flush=True)
        else:
            speedup = (
                item["benchmark"]["tokens_per_second"]
                / report["reference_benchmark"]["tokens_per_second"]
            )
            print(
                f"  Short sampled F4 token throughput: {speedup:.2f}x B=1; exact-token drift remains recorded.",
                flush=True,
            )
        atomic_json(directory / "discovery.json", report)
    measured = [item for item in report["candidates"] if item["status"] == "measured"]
    if not measured:
        raise RuntimeError(
            "No batched comparison candidate passed the fixed-prefix numerical/support checks. See discovery.json."
        )
    report.update(
        status="selected",
        candidate_policy=max(
            measured, key=lambda item: item["benchmark"]["tokens_per_second"]
        )["policy"],
    )
    atomic_json(directory / "discovery.json", report)
    # Profile actual sampling, unlike a greedy-only performance investigation.
    from tas_runtime.profiling import profile_call

    report["operator_profiles"] = {}
    with using_policy(study, reference_policy):
        for method in (next(method for method in methods if method.id == "E0"), f4):
            study.check_budget()
            with step(f"Sampled B=1 operator profile: {method.id}"):
                report["operator_profiles"][method.id] = profile_call(
                    lambda: generate(
                        runtime,
                        config,
                        method,
                        ids[:1],
                        {
                            key: value[:1]
                            for key, value in method_features(features, method).items()
                        },
                        seeds[method.id][:1],
                        execution="serial",
                        max_new_tokens=config.pilot_tokens,
                    ),
                    directory / f"profile-{method.id}",
                    device=runtime.device,
                )
    report["status"] = "complete"
    atomic_json(directory / "discovery.json", report)
    return report


def run_batch_comparison(study):
    """Strict reference -> short batch discovery -> 96 default complete captions."""
    from .execution_report import comparison_report

    config = study.config
    rows = study.rows("A")[: config.comparison_images]
    if len(rows) != config.comparison_images:
        raise ValueError("The prepared cohort has too few comparison images.")
    names = config.methods or ("E0", "E1", "E1-H", "F4")
    available = {method.id: method for method in harmonized_baselines()}
    methods = [
        available[name] if name in available else method_by_id(name) for name in names
    ]
    if not {"E0", "E1", "F4"} <= set(names) or len(set(names)) != len(names):
        raise ValueError("Comparison requires unique methods including E0, E1 and F4.")
    if not config.generation.do_sample:
        raise ValueError(
            "The execution comparison evaluates the study's sampled captions."
        )
    if max(config.batch_sizes) > len(rows) or max(config.batch_sizes) < 2:
        raise ValueError(
            "Choose a comparison image count >= the requested batch size, which must be >=2."
        )
    try:
        # Preserve every strict P0 rule; limit it to the reference shape rather
        # than repeating the already rejected 24-option exact-token sweep.
        study.config = replace(
            config, stage="flow", batch_sizes=(1,), execution="serial", pilot_images=2
        )
        study.audit()
    finally:
        study.config = config
    reference = dict(study.policy)
    plan = {
        "schema": "tas-batched-comparison-v1",
        "rows": rows,
        "study": study.identity,
        "methods": [method.to_dict() for method in methods],
        "seed": config.seeds[0],
        "reference_policy": reference,
        "batch_images": max(config.batch_sizes),
        "execution": config.execution,
        "pilot_tokens": config.pilot_tokens,
        "quality_token_limit": config.generation.max_new_tokens,
        "claim": "descriptive development pilot; no equivalence margin or automatic promotion",
    }
    directory = study.persistent_run / ("batch-comparison-" + canonical_hash(plan)[:12])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "plan.json", plan)
    discovery_path = directory / "discovery.json"
    discovery = (
        json.loads(discovery_path.read_text()) if discovery_path.exists() else None
    )
    if discovery is None or discovery["status"] != "complete":
        discovery = discover(study, rows, methods, reference, directory)
    if discovery["reference_policy"] != reference:
        raise ArtifactError(
            "Comparison reference changed; never reuse a different execution policy."
        )
    candidate = discovery["candidate_policy"]
    print(
        f"Comparison candidate frozen before quality scoring: {candidate}", flush=True
    )
    with using_policy(study, reference):
        baseline_records, _ = study.collection(
            "batch-compare-reference", rows, methods, seeds=(config.seeds[0],)
        )
    with using_policy(study, candidate):
        candidate_records, _ = study.collection(
            "batch-compare-candidate", rows, methods, seeds=(config.seeds[0],)
        )
    report = comparison_report(
        baseline_records,
        candidate_records,
        bootstrap_repeats=config.bootstrap_replicates,
        bootstrap_seed=config.bootstrap_seed,
    )
    report.update(
        plan=plan,
        reference_policy=reference,
        candidate_policy=candidate,
        discovery="discovery.json",
    )
    atomic_json(directory / "comparison.json", report)
    pairs = {(row["id"], row["method"], row["seed"]): row for row in baseline_records}
    changed = []
    for row in candidate_records:
        previous = pairs[(row["id"], row["method"], row["seed"])]
        if previous["token_ids"] != row["token_ids"]:
            changed.append(
                {
                    "image_id": row["image_id"],
                    "image": row["image"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "reference_caption": previous["caption"],
                    "candidate_caption": row["caption"],
                    "reference_tokens": previous["token_ids"],
                    "candidate_tokens": row["token_ids"],
                }
            )
    atomic_jsonl(directory / "changed-captions.jsonl", changed)
    study.stage_reports["batch_comparison"] = str(directory / "comparison.json")
    print(f"Execution comparison saved: {directory / 'comparison.json'}", flush=True)
