"""Shared branch execution, explicit sampling and independent per-record RNG.

No experimental fusion formula is built into this module. Views are packed in
view-major order, then logits are fused to one distribution per real image.
The selected token is fed to every view of that image on the next step.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Callable, Mapping

import torch

from .profiling import synchronize


def record_seed(seed: int, image_id, method_id: str, query: str = "") -> int:
    payload = json.dumps(
        ["image-method-seed-v1", int(seed), str(image_id), method_id, query],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


@dataclasses.dataclass(frozen=True)
class DecodeOptions:
    max_new_tokens: int = 180
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    do_sample: bool = True
    eos_token_ids: tuple[int, ...] = (2,)
    pad_token_id: int = 0
    protocol: str = "HARMONIZED"
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if self.max_new_tokens < 1 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid token limit or top_p.")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("Sampling needs a positive temperature.")
        if self.protocol not in {"LEGACY", "HARMONIZED"}:
            raise ValueError(
                "A named LEGACY or HARMONIZED postprocessing protocol is required."
            )


@dataclasses.dataclass
class ForwardState:
    logits: torch.Tensor
    past: object
    length: int


@dataclasses.dataclass
class DecodeResult:
    token_ids: list[list[int]]
    stop_reasons: list[str]
    prefill_seconds: float
    decode_seconds: float
    peak_memory_bytes: int
    branch_count: int
    execution: str


def final_scores(logits, original, support, options: DecodeOptions):
    """Exactly one temperature; truncation matches the pinned HF warpers.

    LEGACY preserves native-dtype fusion/filter/warper order. HARMONIZED
    uses FP32 final scores, output temperature, support, then top-k/top-p.
    Probability calculations inside new fusion functions are separately FP32.
    """
    from transformers.generation.logits_process import (
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    scores = logits.clone() if options.protocol == "LEGACY" else logits.float().clone()
    mask = support(original) if support is not None else None
    if options.protocol == "LEGACY" and mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    if options.do_sample and options.temperature != 1.0:
        scores = scores / options.temperature
    if options.protocol == "HARMONIZED" and mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    if options.do_sample:
        if options.top_k:
            scores = TopKLogitsWarper(options.top_k)(None, scores)
        if options.top_p < 1.0:
            scores = TopPLogitsWarper(options.top_p)(None, scores)
    return scores


def _choose(scores, generators, sample):
    if not torch.isfinite(scores).any(dim=-1).all():
        raise FloatingPointError(
            "Fusion/filter left an image without a finite candidate token."
        )
    if not sample:
        return scores.argmax(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError(
            "Nonfinite sampling probabilities; stopping before writing an invalid caption."
        )
    return torch.stack(
        [
            torch.multinomial(row, 1, generator=generator)[0]
            for row, generator in zip(probabilities, generators)
        ]
    )


@torch.inference_mode()
def decode_branches(
    backend,
    prompt_ids: torch.Tensor,
    views: Mapping[str, torch.Tensor],
    fusion: Callable,
    support: Callable | None,
    options: DecodeOptions,
    seeds: list[int],
    *,
    execution: str = "packed",
) -> DecodeResult:
    """Generate one caption per image with either serial or packed view execution.

    Backend exposes prefill(prompt_ids, image_batch)->ForwardState and
    advance(tokens[:,None], prior_state)->ForwardState. It must not mutate
    another trajectory's cache. Only equal-length, unpadded prompts are accepted.
    """
    if execution not in {"serial", "packed"}:
        raise ValueError("execution must be serial or packed.")
    if prompt_ids.ndim != 2 or not views or "ORIG" not in views:
        raise ValueError("Expected [images,prompt_tokens] and at least the ORIG view.")
    names = list(views)
    batch = prompt_ids.shape[0]
    if len(seeds) != batch or any(value.shape[0] != batch for value in views.values()):
        raise ValueError("Seeds and every visual view must have one row per image.")
    device = prompt_ids.device
    generators = [
        torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    if execution == "packed":
        state = backend.prefill(
            prompt_ids.repeat(len(names), 1), torch.cat(list(views.values()), dim=0)
        )
        states = None
    else:
        states = [backend.prefill(prompt_ids, views[name]) for name in names]
        state = None
    synchronize(device)
    prefill_seconds = time.perf_counter() - started
    generated = torch.full(
        (batch, options.max_new_tokens),
        options.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch, dtype=torch.long, device=device)
    eos = torch.tensor(options.eos_token_ids, dtype=torch.long, device=device)
    stop_sequences = [
        torch.tensor(seq, dtype=torch.long, device=device)
        for seq in options.stop_token_sequences
        if seq
    ]
    decode_start = time.perf_counter()
    for step in range(options.max_new_tokens):
        if execution == "packed":
            by_view = {
                name: state.logits[index * batch : (index + 1) * batch]
                for index, name in enumerate(names)
            }
        else:
            by_view = {name: item.logits for name, item in zip(names, states)}
        scores = final_scores(fusion(by_view), by_view["ORIG"], support, options)
        tokens = _choose(scores, generators, options.do_sample)
        tokens = torch.where(finished, options.pad_token_id, tokens)
        generated[:, step] = tokens
        lengths += (~finished).long()
        stop = (
            (tokens[:, None] == eos[None, :]).any(dim=-1)
            if eos.numel()
            else torch.zeros_like(finished)
        )
        for sequence in stop_sequences:
            if sequence.numel() <= step + 1:
                stop |= (
                    generated[:, step + 1 - sequence.numel() : step + 1] == sequence
                ).all(dim=-1)
        finished |= stop
        if finished.all() or step + 1 == options.max_new_tokens:
            break
        if execution == "packed":
            state = backend.advance(tokens.repeat(len(names))[:, None], state)
        else:
            states = [backend.advance(tokens[:, None], item) for item in states]
    synchronize(device)
    elapsed = time.perf_counter() - decode_start
    token_rows = generated.cpu().tolist()
    size_rows = lengths.cpu().tolist()
    # Stop reasons are output metadata. Read them once after generation instead
    # of synchronizing the GPU for newly_finished.any()/nonzero() at every token.
    reasons = ["eos_or_stop" if stopped else "max_new_tokens" for stopped in finished.cpu().tolist()]
    return DecodeResult(
        [row[:length] for row, length in zip(token_rows, size_rows)],
        reasons,
        prefill_seconds,
        elapsed,
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        len(names),
        execution,
    )


@torch.inference_mode()
def forced_prefix_logits(
    backend, prompt_ids, views, continuation: list[int], *, execution="packed"
):
    """Collect logits on identical forced histories; never substitutes for free generation."""
    names, batch = list(views), prompt_ids.shape[0]
    if execution == "packed":
        states = [
            backend.prefill(
                prompt_ids.repeat(len(names), 1), torch.cat(list(views.values()), dim=0)
            )
        ]
    else:
        states = [backend.prefill(prompt_ids, views[name]) for name in names]
    snapshots = []
    for index in range(len(continuation) + 1):
        if execution == "packed":
            snapshots.append(
                {
                    name: states[0].logits[i * batch : (i + 1) * batch].detach().cpu()
                    for i, name in enumerate(names)
                }
            )
        else:
            snapshots.append(
                {
                    name: state.logits.detach().cpu()
                    for name, state in zip(names, states)
                }
            )
        if index < len(continuation):
            tokens = torch.full(
                (batch * (len(names) if execution == "packed" else 1), 1),
                continuation[index],
                dtype=torch.long,
                device=prompt_ids.device,
            )
            states = [backend.advance(tokens, state) for state in states]
    return snapshots
