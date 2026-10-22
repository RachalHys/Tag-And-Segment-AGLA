"""Blocking P0 checks and a measured execution-policy pilot on actual weights."""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import zip_longest

import torch

from tas_runtime.artifacts import atomic_json
from tas_runtime.decoding import (
    DecodeOptions,
    decode_branches,
    forced_prefix_logits,
    record_seed,
)
from tas_runtime.profiling import environment_manifest, synchronize
from tas_runtime.progress import progress, step

from .methods import fuse_logits, method_by_id, plausibility_mask
from .prepared import method_features, pixel_key, prepare_feature_batch


def decode_options(runtime, config, method, *, greedy=False, max_new_tokens=None):
    generation = config.generation
    return DecodeOptions(
        max_new_tokens=max_new_tokens or generation.max_new_tokens,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        do_sample=generation.do_sample and not greedy,
        eos_token_ids=(runtime.tokenizer.eos_token_id,),
        pad_token_id=runtime.tokenizer.pad_token_id or runtime.tokenizer.eos_token_id,
        protocol=method.protocol,
        stop_token_sequences=((runtime.tokenizer.eos_token_id,),),
    )


def generate(
    runtime,
    config,
    method,
    ids,
    views,
    seeds,
    *,
    greedy=False,
    execution="packed",
    max_new_tokens=None,
):
    return decode_branches(
        runtime.backend,
        ids,
        views,
        lambda logits: fuse_logits(logits, method),
        (
            lambda original: plausibility_mask(
                original, method.beta, mode=method.protocol
            )
        )
        if method.support
        else None,
        decode_options(
            runtime, config, method, greedy=greedy, max_new_tokens=max_new_tokens
        ),
        seeds,
        execution=execution,
    )


def compare_snapshots(
    reference,
    candidate,
    *,
    atol=0.04,
    rtol=0.01,
    protocol="HARMONIZED",
    beta=0.5,
    support=True,
):
    if len(reference) != len(candidate):
        raise AssertionError("Forced-prefix snapshot counts differ.")
    worst = 0.0
    argmax_changes, support_changes = 0, 0
    for step_index, (expected, actual) in enumerate(zip(reference, candidate)):
        for name in expected:
            first, second = expected[name].float(), actual[name].float()
            worst = max(worst, float((first - second).abs().max()))
            try:
                torch.testing.assert_close(first, second, atol=atol, rtol=rtol)
            except AssertionError as exc:
                raise AssertionError(
                    f"Forced-prefix step {step_index}, view {name}: {exc}"
                ) from exc
            argmax_changes += int((first.argmax(-1) != second.argmax(-1)).sum())
            if name == "ORIG" and support:
                support_changes += int(
                    (
                        plausibility_mask(expected[name], beta, mode=protocol)
                        != plausibility_mask(actual[name], beta, mode=protocol)
                    ).sum()
                )
    if support_changes:
        raise AssertionError(
            f"Original-view plausibility support changed for {support_changes} pilot tokens."
        )
    return {
        "max_absolute_logit_error": worst,
        "argmax_changes": argmax_changes,
        "support_token_changes": support_changes,
        "atol": atol,
        "rtol": rtol,
    }


def audit_and_pilot(runtime, config, rows, prepared, directory):
    """Eager baseline gate, then one-at-a-time candidates; never silent fallback.

    A faster kernel is eligible only if fixed-prefix logits agree within the
    recorded tolerance and greedy captions match on the pilot images. Failed
    candidates stay in the report; the reference eager/serial path remains.
    """
    import json
    from pathlib import Path

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    policy_path = root / "execution-policy.json"
    locked_policy = (
        json.loads(policy_path.read_text()) if policy_path.exists() else None
    )
    available = min(config.pilot_images, len(rows), len(prepared))
    if available < 1:
        raise ValueError("P0 needs at least one prepared pilot image.")
    reference_count = (
        locked_policy["batch_images"] if locked_policy is not None
        else max((count for count in config.batch_sizes if count <= available), default=1)
    )
    if reference_count > available:
        raise ValueError("The locked execution batch exceeds the available pilot images; preserve its shape when resuming.")
    report = {
        "environment": environment_manifest(),
        "baseline_checks": [],
        "pilot": [],
        "geometry": prepared[0]["metadata"],
        "caption_labels": "perception-only; no labels added to LLaVA prompt",
        "scientific_protocols": {
            "LEGACY": "native E0; raw-before-processor OBJ-H/fallback and native arithmetic E1",
            "HARMONIZED": "processor canvas; FP32 fusion/probability/support; one final output temperature",
        },
        "sampling": "per image/method/run generator, independent of job order; kernel changes can alter sampled captions",
        "status": "running",
    }
    _, ids, _ = runtime.prompt(rows[0]["query"], config.generation.conv_mode)
    ids = ids.unsqueeze(0).to(runtime.device)
    runtime.set_attention("eager")
    original_last = runtime.backend.last_position_logits
    head_native_match = True
    # The original E1 implementation defines sampling only. Audit that native
    # path even when a study asks for greedy decoding; do not call an unrelated
    # HF greedy_search path that would ignore E1's augmented image.
    native_config = replace(
        config,
        generation=replace(
            config.generation,
            do_sample=True,
            temperature=(
                config.generation.temperature
                if config.generation.temperature > 0
                else 1.0
            ),
        ),
    )
    report["native_reference_generation"] = {
        "do_sample": True,
        "temperature": native_config.generation.temperature,
        "scope": "original sampled E0/E1; separate from study greedy policy checks",
    }
    # Original model computes all prompt logits. Keep the reference comparison
    # independent of the optional last-position projection optimization.
    runtime.backend.last_position_logits = False
    try:
        for identifier in progress(("E0", "E1"), desc="P0 native baselines", unit="method"):
            method = method_by_id(identifier)
            pixels = {
                "ORIG": prepared[0]["pixels"]["ORIG"]
                .unsqueeze(0)
                .to(runtime.device, runtime.dtype)
            }
            if identifier == "E1":
                pixels["OBJ-H"] = (
                    prepared[0]["pixels"]["LEGACY_OBJ-H"]
                    .unsqueeze(0)
                    .to(runtime.device, runtime.dtype)
                )
            seed = record_seed(
                config.seeds[0], rows[0]["image_id"], identifier, rows[0]["query"]
            )
            reference = runtime.reference(
                ids,
                pixels["ORIG"],
                pixels.get("OBJ-H"),
                seed=seed,
                max_new_tokens=config.pilot_tokens,
                temperature=native_config.generation.temperature,
                top_p=config.generation.top_p,
                top_k=config.generation.top_k,
            )[0].tolist()
            candidate = generate(
                runtime,
                native_config,
                method,
                ids,
                pixels,
                [seed],
                execution="serial",
                max_new_tokens=config.pilot_tokens,
            )
            match = candidate.token_ids[0] == reference
            report["baseline_checks"].append(
                {
                    "method": identifier,
                    "native_token_match": match,
                    "reference_tokens": reference,
                    "runtime_tokens": candidate.token_ids[0],
                    "seed": seed,
                }
            )
            if not match:
                raise AssertionError(
                    f"P0 failed: {identifier} does not reproduce the original sampled reference at a fixed seed. See audit.json."
                )
            if original_last:
                runtime.backend.last_position_logits = True
                optimized = generate(
                    runtime,
                    native_config,
                    method,
                    ids,
                    pixels,
                    [seed],
                    execution="serial",
                    max_new_tokens=config.pilot_tokens,
                )
                runtime.backend.last_position_logits = False
                optimized_match = optimized.token_ids[0] == reference
                report["baseline_checks"][-1]["last_position_sample_match"] = (
                    optimized_match
                )
                if not optimized_match:
                    head_native_match = False
                    print(
                        f"Last-position projection rejected: {identifier} native sampled tokens changed.",
                        flush=True,
                    )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "audit.json", report)
        raise
    finally:
        runtime.backend.last_position_logits = original_last

    # Only these rows can be consumed by a candidate. Extra configured preview
    # images never contributed to a check; do not generate unused raw captions.
    pilot_rows = rows[:reference_count]
    pilot_prepared = prepared[: len(pilot_rows)]
    # Audit the native baselines as well as the expensive V=3 topology. Every
    # candidate uses the same cached-feature path as the actual study, compared
    # with per-image eager/full-head RAW PIXEL references, not cached features.
    pilot_methods = [method_by_id(name) for name in ("E0", "E1", "F4")]
    prompt_ids = [
        runtime.prompt(row["query"], config.generation.conv_mode)[1]
        for row in pilot_rows
    ]
    if len({len(tokens) for tokens in prompt_ids}) != 1:
        raise ValueError(
            "The captioning pilot requires equal-length prompts; external VQA uses length buckets later."
        )
    all_ids = torch.stack(prompt_ids).to(runtime.device)
    forced = [
        int(runtime.tokenizer.bos_token_id or 1),
        int(runtime.tokenizer.eos_token_id),
        1,
    ]
    references = {}
    runtime.backend.last_position_logits = False
    try:
        for method in progress(pilot_methods, desc="P0 reference methods", unit="method"):
            seeds = [
                record_seed(config.seeds[0], row["image_id"], method.id, row["query"])
                for row in pilot_rows
            ]
            reference = {"seeds": seeds, "logits": [], "greedy": [], "sample": []}
            for index, item in enumerate(progress(
                pilot_prepared, desc=f"P0 {method.id} raw references", unit="image", leave=False,
            )):
                single = {
                    name: item["pixels"][pixel_key(method, name)]
                    .unsqueeze(0)
                    .to(runtime.device, runtime.dtype)
                    for name in method.views
                }
                reference["logits"].append(
                    forced_prefix_logits(
                        runtime.backend,
                        all_ids[index : index + 1],
                        single,
                        forced,
                        execution="serial",
                    )
                )
                modes = (
                    ("greedy", "sample")
                    if method.protocol == "LEGACY" and config.generation.do_sample
                    else ("greedy",)
                )
                for mode in modes:
                    reference[mode] += generate(
                        runtime,
                        config,
                        method,
                        all_ids[index : index + 1],
                        single,
                        [seeds[index]],
                        greedy=mode == "greedy",
                        execution="serial",
                        max_new_tokens=config.pilot_tokens,
                    ).token_ids
            references[method.id] = reference
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "audit.json", report)
        raise
    finally:
        runtime.backend.last_position_logits = original_last
    with step("P0 shared visual features"):
        feature_start = time.perf_counter()
        features = prepare_feature_batch(runtime, pilot_prepared, pilot_methods)
        synchronize(runtime.device)
        report["feature_preparation_seconds"] = time.perf_counter() - feature_start
    report["execution_reference"] = {
        "image_ids": [row["image_id"] for row in pilot_rows],
        "methods": [method.id for method in pilot_methods],
        "input": "per-image raw pixels; eager attention; full-position vocabulary head",
        "candidate_input": "cached projected features; selected head, attention and batch policy",
        "vision_encoding": "each exact tensor encoded once at native image batch size 1",
        "checks": "fixed-prefix logits/support, greedy tokens, native sampled tokens when study samples",
        "scope": "pilot only; no bitwise equivalence guarantee for unseen images or longer captions",
    }
    eligible = []
    policies = [
        (
            {
                "backend": backend,
                "execution": execution,
                "batch_images": count,
                "last_position_logits": original_last and head_native_match,
            },
            False,
        )
        for backend in config.attention_backends
        for execution in (
            ["serial", "packed"] if config.execution == "auto" else [config.execution]
        )
        for count in config.batch_sizes
        if count <= len(pilot_rows)
    ]
    if locked_policy is not None:
        # The head is part of the frozen execution policy, including reconnects.
        policies = [(locked_policy, False)]
    else:
        if original_last and head_native_match:
            # Keep batching available when only the head-shape optimization
            # fails. Full-head variants are skipped when their faster-head
            # counterpart passes the same correctness checks.
            policies += [
                ({**policy, "last_position_logits": False}, False)
                for policy, _ in policies
            ]
        reference_policy = {
            "backend": "eager",
            "execution": "serial",
            "batch_images": 1,
            "last_position_logits": False,
        }
        if not any(policy == reference_policy for policy, _ in policies):
            policies.append((reference_policy, True))
    for candidate_policy, fallback in progress(
        policies, desc="P0 execution options", unit="option",
    ):
        if fallback and eligible:
            break
        if not candidate_policy["last_position_logits"] and any(
            passed["last_position_logits"] and all(
                passed[key] == candidate_policy[key]
                for key in ("backend", "execution", "batch_images")
            )
            for passed in eligible
        ):
            continue
        backend_name = candidate_policy["backend"]
        execution = candidate_policy["execution"]
        requested = candidate_policy["batch_images"]
        item = {**candidate_policy, "views": 3, "method_checks": {}}
        if fallback:
            item["reference_fallback"] = True
        try:
            if candidate_policy["last_position_logits"] and not head_native_match:
                raise AssertionError(
                    "The locked last-position head no longer reproduces native sampled tokens."
                )
            runtime.set_attention(backend_name)
            runtime.backend.last_position_logits = candidate_policy["last_position_logits"]
            count = requested
            for method in pilot_methods:
                reference = references[method.id]
                local_views = {
                    name: value[:count]
                    for name, value in method_features(features, method).items()
                }
                actual = forced_prefix_logits(
                    runtime.backend,
                    all_ids[:count],
                    local_views,
                    forced,
                    execution=execution,
                )
                expected = [
                    {
                        name: torch.cat(
                            [
                                reference["logits"][i][step][name]
                                for i in range(count)
                            ]
                        )
                        for name in method.views
                    }
                    for step in range(len(forced) + 1)
                ]
                checks = {"protocol": method.protocol}
                item["method_checks"][method.id] = checks
                checks.update(
                    compare_snapshots(
                        expected,
                        actual,
                        protocol=method.protocol,
                        beta=method.beta,
                        support=method.support,
                    )
                )
                for mode in ("greedy", "sample"):
                    if not reference[mode]:
                        continue
                    result = generate(
                        runtime,
                        config,
                        method,
                        all_ids[:count],
                        local_views,
                        reference["seeds"][:count],
                        greedy=mode == "greedy",
                        execution=execution,
                        max_new_tokens=config.pilot_tokens,
                    )
                    checks[mode + "_match"] = (
                        result.token_ids == reference[mode][:count]
                    )
                    seconds = result.prefill_seconds + result.decode_seconds
                    checks[mode + "_timing"] = {
                        "prefill_seconds": result.prefill_seconds,
                        "decode_seconds": result.decode_seconds,
                        "batch_seconds": seconds,
                        "generated_tokens": sum(map(len, result.token_ids)),
                        "peak_memory_bytes": result.peak_memory_bytes,
                    }
                    if not checks[mode + "_match"]:
                        differences = []
                        for row, (expected_tokens, actual_tokens) in enumerate(zip(
                            reference[mode][:count], result.token_ids,
                        )):
                            for position, (expected_token, actual_token) in enumerate(
                                zip_longest(expected_tokens, actual_tokens)
                            ):
                                if expected_token != actual_token:
                                    differences.append({
                                        "image_id": pilot_rows[row]["image_id"],
                                        "position": position,
                                        "reference_token": expected_token,
                                        "candidate_token": actual_token,
                                    })
                                    break
                        checks[mode + "_first_differences"] = differences
                        raise AssertionError(
                            f"{method.id} {mode} pilot captions changed: {differences}; candidate not selected automatically."
                        )
            # Warm one more run, with comparable maximum token budget.
            # The final method in pilot_methods is the V=3 F4 topology.
            result = generate(
                runtime,
                config,
                method,
                all_ids[:count],
                local_views,
                reference["seeds"][:count],
                greedy=True,
                execution=execution,
                max_new_tokens=config.pilot_tokens,
            )
            seconds = result.prefill_seconds + result.decode_seconds
            item.update(
                status="passed",
                seconds=seconds,
                outputs_per_hour=3600 * count / seconds,
                token_count=sum(map(len, result.token_ids)),
                peak_memory_bytes=result.peak_memory_bytes,
            )
            eligible.append(item)
        except (RuntimeError, AssertionError, FloatingPointError) as exc:
            item.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["pilot"].append(item)
        print(
            f"Pilot {backend_name}/{execution}/B={requested}/"
            f"last_position={candidate_policy['last_position_logits']}: {item['status']}",
            flush=True,
        )
        if item["status"] == "rejected":
            print(item["error"], flush=True)
        else:
            print(
                f"  {item['seconds']:.2f}s / {item['token_count']} tokens; "
                f"{item['outputs_per_hour']:.0f} outputs/hour (short pilot); "
                f"peak {item['peak_memory_bytes'] / 1024**3:.1f} GiB",
                flush=True,
            )
    if not eligible:
        report.update(status="failed", error="No execution policy passed P0/pilot.")
        atomic_json(root / "audit.json", report)
        raise RuntimeError(report["error"])
    best = max(eligible, key=lambda item: item["outputs_per_hour"])
    report["selected_policy"] = {
        key: best[key]
        for key in ("backend", "execution", "batch_images", "last_position_logits")
    }
    if locked_policy is None:
        atomic_json(policy_path, report["selected_policy"])
    report["status"] = "passed"
    report["pilot_scope"] = (
        "short warmed V=3 captions; real stage throughput and variable-length efficiency reported separately"
    )
    runtime.backend.last_position_logits = best["last_position_logits"]
    runtime.set_attention(best["backend"])
    report["last_position_logits"] = best["last_position_logits"]
    if config.stage == "audit":
        from tas_runtime.profiling import profile_call

        # Keep the completed P0 evidence even if the optional profiled call fails.
        atomic_json(root / "audit.json", report)
        method = method_by_id("F4")
        count = best["batch_images"]
        with step("Profile one warmed F4 pilot (diagnostic only)"):
            report["operator_profile"] = profile_call(
                lambda: generate(
                    runtime, config, method, all_ids[:count],
                    {name: value[:count] for name, value in method_features(features, method).items()},
                    references[method.id]["seeds"][:count], greedy=True,
                    execution=best["execution"], max_new_tokens=config.pilot_tokens,
                ),
                root, device=runtime.device,
            )
    atomic_json(root / "audit.json", report)
    return report
