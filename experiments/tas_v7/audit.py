"""Blocking P0 checks and a measured execution-policy pilot on actual weights."""

from __future__ import annotations

from dataclasses import replace
import time

import torch

from tas_runtime.artifacts import atomic_json
from tas_runtime.decoding import (
    DecodeOptions,
    decode_branches,
    forced_prefix_logits,
    record_seed,
)
from tas_runtime.profiling import environment_manifest, synchronize
from .methods import method_by_id, fuse_logits, plausibility_mask
from .prepared import prepare_feature_batch, method_features, pixel_key


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
    for expected, actual in zip(reference, candidate):
        for name in expected:
            first, second = expected[name].float(), actual[name].float()
            worst = max(worst, float((first - second).abs().max()))
            torch.testing.assert_close(first, second, atol=atol, rtol=rtol)
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
    prompt, ids, _ = runtime.prompt(rows[0]["query"], config.generation.conv_mode)
    ids = ids.unsqueeze(0).to(runtime.device)
    runtime.set_attention("eager")
    original_last = runtime.backend.last_position_logits
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
        for identifier in ("E0", "E1"):
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
                    raise AssertionError(
                        "Last-position projection changed the baseline pilot. Set last_position_logits=False and rerun the audit."
                    )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "audit.json", report)
        raise
    finally:
        runtime.backend.last_position_logits = original_last

    pilot_rows = rows[: config.pilot_images]
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
        for method in pilot_methods:
            seeds = [
                record_seed(config.seeds[0], row["image_id"], method.id, row["query"])
                for row in pilot_rows
            ]
            reference = {"seeds": seeds, "logits": [], "greedy": [], "sample": []}
            for index, item in enumerate(pilot_prepared):
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
    feature_start = time.perf_counter()
    features = prepare_feature_batch(runtime, pilot_prepared, pilot_methods)
    synchronize(runtime.device)
    report["feature_preparation_seconds"] = time.perf_counter() - feature_start
    report["execution_reference"] = {
        "methods": [method.id for method in pilot_methods],
        "input": "per-image raw pixels; eager attention; full-position vocabulary head",
        "candidate_input": "cached projected features; selected head, attention and batch policy",
        "checks": "fixed-prefix logits/support, greedy tokens, native sampled tokens when study samples",
        "scope": "pilot only; no bitwise equivalence guarantee for unseen images or longer captions",
    }
    eligible = []
    for backend_name in config.attention_backends:
        for execution in (
            ["serial", "packed"] if config.execution == "auto" else [config.execution]
        ):
            for requested in config.batch_sizes:
                candidate_policy = {
                    "backend": backend_name,
                    "execution": execution,
                    "batch_images": requested,
                }
                if locked_policy is not None and candidate_policy != locked_policy:
                    continue
                if (
                    requested > len(pilot_rows)
                    or execution == "serial"
                    and requested != 1
                ):
                    continue
                item = {
                    "backend": backend_name,
                    "execution": execution,
                    "batch_images": requested,
                    "views": 3,
                    "method_checks": {},
                }
                try:
                    runtime.set_attention(backend_name)
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
                            if not checks[mode + "_match"]:
                                raise AssertionError(
                                    f"{method.id} {mode} pilot captions changed; candidate not selected automatically."
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
                    f"Pilot {backend_name}/{execution}/B={requested}: {item['status']}",
                    flush=True,
                )
    if not eligible:
        report.update(status="failed", error="No execution policy passed P0/pilot.")
        atomic_json(root / "audit.json", report)
        raise RuntimeError(report["error"])
    best = max(eligible, key=lambda item: item["outputs_per_hour"])
    report["selected_policy"] = {
        key: best[key] for key in ("backend", "execution", "batch_images")
    }
    if locked_policy is None:
        atomic_json(policy_path, report["selected_policy"])
    report["status"] = "passed"
    report["pilot_scope"] = (
        "short warmed V=3 captions; real stage throughput and variable-length efficiency reported separately"
    )
    report["last_position_logits"] = original_last
    atomic_json(root / "audit.json", report)
    runtime.set_attention(best["backend"])
    return report
