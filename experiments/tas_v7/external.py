"""Explicit adapters for official external query manifests and AMBER scoring."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import io
import os
from pathlib import Path
import re
from types import SimpleNamespace

from tas_runtime.artifacts import ArtifactError, atomic_json, sha256_file
from tas_runtime.data import load_external_jsonl, verify_manifest_images


def score_amber(project_root, records, destination, *, resources=None):
    """Call the existing evaluator with its original NLP semantics, cached.

    init() reads a global args in the legacy source. Supplying that explicit
    namespace fixes the importable invocation without editing benchmark code.
    """
    import spacy

    root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resources = Path(resources or root / "AMBER/data").resolve()
    path = destination / "amber-input.json"
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)) or any(
        type(identifier) is not int or not 1 <= identifier <= 1004 for identifier in ids
    ):
        raise ArtifactError(
            "AMBER generative scoring needs one valid response per official integer ID."
        )
    atomic_json(
        path,
        [
            {"id": row["id"], "response": row["caption"]}
            for row in sorted(records, key=lambda item: item["id"])
        ],
    )
    source = root / "AMBER/inference.py"
    spec = importlib.util.spec_from_file_location("_tas_amber_official", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Cache exactly the unchanged spaCy pipeline and its pair comparisons.
    module.nlp = functools.lru_cache(maxsize=8192)(module.nlp)
    module.check_synonyms_word = functools.lru_cache(maxsize=131072)(
        module.check_synonyms_word
    )
    module.args = SimpleNamespace(
        word_association=str(resources / "relation.json"),
        safe_words=str(resources / "safe_words.txt"),
        metrics=str(resources / "metrics.txt"),
        annotation=str(resources / "annotations.json"),
        inference_data=str(path),
        similarity_score=0.8,
        evaluation_type="g",
        top_k=30,
    )
    previous = Path.cwd()
    output = io.StringIO()
    try:
        os.chdir(destination)
        with contextlib.redirect_stdout(output):
            module.main(module.args)
    finally:
        os.chdir(previous)
    transcript = output.getvalue()
    (destination / "amber-official.txt").write_text(transcript)
    metrics = {}
    for key in ("CHAIR", "Cover", "Hal", "Cog"):
        match = re.search(rf"^{key}:\s+([\d.]+)", transcript, flags=re.MULTILINE)
        if match is None:
            raise ArtifactError(
                "Official AMBER evaluator did not produce all requested metrics."
            )
        metrics[key] = float(match.group(1))
    result = {
        "metrics_percent": metrics,
        "count": len(records),
        "evaluator_sha256": sha256_file(source),
        "resources_sha256": {
            name: sha256_file(resources / name)
            for name in (
                "relation.json",
                "safe_words.txt",
                "metrics.txt",
                "annotations.json",
            )
        },
        "spacy_version": spacy.__version__,
        "scope": "official frozen AMBER generative; legacy project benchmark, not blind",
    }
    atomic_json(destination / "amber-report.json", result)
    return result


def run_external(study, methods):
    from .evaluation import score_pope

    config = study.config
    if not config.external_manifest or not config.external_image_root:
        raise ValueError(
            "External stage needs external_manifest and external_image_root. CPU prepare/verify those files first."
        )
    external = load_external_jsonl(config.external_manifest, dataset=config.dataset)
    rows = external["rows"]
    root = Path(config.external_image_root).expanduser().resolve()
    verify_manifest_images(rows, root)
    previous_root = study.work["data_root"]
    study.work["data_root"] = root
    try:
        records, _ = study.collection(
            config.dataset, rows, methods, modes=("sample", "greedy")
        )
    finally:
        study.work["data_root"] = previous_root
    groups = {}
    for row in records:
        groups.setdefault(
            (row["method"], row["seed"], row["generation_mode"]), []
        ).append(row)
    reports = []
    directory = study.persistent_run / "external-metrics"
    directory.mkdir(parents=True, exist_ok=True)
    for (method, seed, mode), values in groups.items():
        if config.dataset == "pope":
            result = score_pope(
                values,
                answer_key="caption",
                variant=Path(config.external_manifest).stem,
            )
        elif config.dataset == "amber":
            result = score_amber(
                study.project_root,
                values,
                directory / f"{method}-{seed}-{mode}",
                resources=config.amber_resources,
            )
        else:
            raise ValueError("External stage supports AMBER or POPE.")
        reports.append(
            {"method": method, "seed": seed, "generation_mode": mode, "report": result}
        )
    atomic_json(
        directory / "report.json",
        {
            "dataset": config.dataset,
            "query_source": external,
            "groups": reports,
            "limitation": "Official manifests/resources are explicit; never tune at this final stage.",
        },
    )
    study.stage_reports[config.dataset] = str(directory / "report.json")
