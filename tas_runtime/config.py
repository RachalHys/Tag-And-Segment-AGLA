"""Explicit, serializable settings for the frozen v7 experimental protocol."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 180
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    do_sample: bool = True
    precision: str = "fp16"
    prompt: str = "Describe this image."
    conv_mode: str = "llava_v1"
    rng_policy: str = "image-method-seed-v1"

    def __post_init__(self):
        if self.max_new_tokens < 1 or not 0 < self.top_p <= 1:
            raise ValueError("max_new_tokens must be positive; top_p must be in (0,1].")
        if self.do_sample and self.temperature <= 0:
            raise ValueError(
                "Sampling requires positive temperature. Use do_sample=False for greedy."
            )
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be None or nonnegative.")
        if self.precision != "fp16":
            raise ValueError(
                "v7 reproduction is FP16. A different precision needs a named protocol."
            )
        if self.rng_policy != "image-method-seed-v1":
            raise ValueError(
                "Unsupported RNG policy; resume requires image/method/run-specific streams."
            )


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    persistent_root: str
    local_root: str = "/content/tas-v7-work"
    stage: str = "flow"
    dataset: str = "coco"
    methods: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0, 1, 2)
    generation: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    sigma_ratio: float = 0.04
    yolo_conf: float = 0.20
    expansion_ratio: float = 0.0
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    attention_backends: tuple[str, ...] = ("eager", "sdpa")
    execution: str = "auto"
    last_position_logits: bool = True
    pilot_tokens: int = 24
    pilot_images: int = 8
    comparison_images: int = 12
    max_retries: int = 2
    snapshot_every: int = 32
    split_seed: int = 20261021
    bootstrap_seed: int = 20261022
    bootstrap_replicates: int = 1000
    shortlist_count: int = 2
    coverage_tolerance: float = 0.02  # owner approved two absolute percentage points
    diagnostic_count: int = 75
    auto_continue_after_smoke: bool = True
    run_confirmation: bool = True
    run_controls: bool = True
    run_diagnostics: bool = True
    run_final: bool = False
    external_manifest: str | None = None
    external_image_root: str | None = None
    exclusions_file: str | None = None
    reserve_pope: bool = True
    amber_resources: str | None = None
    source_revision: str = ""
    require_a100: bool = False
    max_session_minutes: float = 180.0

    def __post_init__(self):
        if self.stage not in {
            "flow",
            "audit",
            "batch_compare",
            "smoke",
            "screen",
            "confirm",
            "controls",
            "diagnostics",
            "final",
            "external",
        }:
            raise ValueError(f"Unknown stage: {self.stage}")
        if self.dataset not in {"coco", "amber", "pope"}:
            raise ValueError("dataset must be coco, amber or pope.")
        if self.dataset != "coco" and self.stage != "external":
            raise ValueError(
                "AMBER/POPE use stage='external'; COCO cohorts are never relabeled as another benchmark."
            )
        if self.dataset == "coco" and self.generation.prompt != "Describe this image.":
            raise ValueError(
                "First-round COCO prompt is locked. Prompt changes require a separate named ablation."
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Use at least one unique seed.")
        if not self.batch_sizes or min(self.batch_sizes) < 1:
            raise ValueError("batch_sizes must contain positive image batch sizes.")
        if set(self.attention_backends) - {"eager", "sdpa"}:
            raise ValueError("Supported attention backends: eager, sdpa.")
        if self.execution not in {"auto", "serial", "packed"}:
            raise ValueError("execution must be auto, serial or packed.")
        if not 0 <= self.sigma_ratio <= 1 or not 0 < self.yolo_conf <= 1:
            raise ValueError("Invalid blur or detection configuration.")
        if self.expansion_ratio != 0:
            raise ValueError(
                "The initial v7 protocol locks expansion_ratio=0; define a separate ablation."
            )
        if not 50 <= self.diagnostic_count <= 100:
            raise ValueError(
                "P0-D uses 50-100 images from the locked diagnostic cohort."
            )
        if not 1 <= self.shortlist_count <= 2:
            raise ValueError("The initial screen shortlists at most two methods.")
        if not 0 <= self.coverage_tolerance <= 1:
            raise ValueError("coverage_tolerance must be a finite fraction in [0,1].")
        if (
            self.pilot_tokens < 2
            or self.pilot_images < 2
            or self.max_retries < 1
            or self.snapshot_every < 1
        ):
            raise ValueError("Invalid pilot, retry or snapshot settings.")
        if not isinstance(self.comparison_images, int) or not 2 <= self.comparison_images <= 30:
            raise ValueError("Batch comparison uses 2-30 development images from cohort A.")
        if self.stage == "batch_compare":
            if not self.generation.do_sample or not 2 <= max(self.batch_sizes) <= self.comparison_images:
                raise ValueError("Batch comparison requires sampling and 2 <= image batch <= comparison_images.")
            if self.methods and (
                len(self.methods) != len(set(self.methods))
                or not {"E0", "E1", "F4"} <= set(self.methods)
            ):
                raise ValueError("Comparison methods must be unique and include E0, E1 and F4.")
        if self.max_session_minutes <= 0:
            raise ValueError(
                "max_session_minutes must be positive; unfinished stages resume next session."
            )

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        generation = value.pop("generation", {})
        if isinstance(generation, GenerationConfig):
            value["generation"] = generation
        else:
            value["generation"] = GenerationConfig(**generation)
        for key in ("methods", "seeds", "batch_sizes", "attention_backends"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
