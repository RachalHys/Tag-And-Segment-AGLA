"""Locked COCO study cohorts and optional external benchmark manifests.

Preparation is CPU-only. In particular, no perception model is instantiated to
prepare a manifest or download files. Cohorts are locked before model outputs
exist and image identity is retained across all queries/seeds.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .artifacts import (
    ArtifactError,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    sha256_file,
)
from .assets import POPE_REVISION, download_verified, safe_extract_zip


COCO_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ANNOTATIONS_URL = COCO_BASE_URL + "/annotations/annotations_trainval2017.zip"
COCO_ANNOTATIONS_BYTES = 252_907_541
# Official S3 single-part object's ETag. Actual SHA256 is also recorded locally.
COCO_ANNOTATIONS_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
DEFAULT_COHORT_SIZES = {"A": 30, "B": 240, "C": 300, "D": 150}
ANNOTATION_MEMBERS = tuple(
    f"annotations/{kind}_{split}2017.json"
    for split in ("train", "val")
    for kind in ("captions", "instances")
)
POPE_URLS = {
    variant: f"https://raw.githubusercontent.com/AoiDragon/POPE/{POPE_REVISION}"
    f"/output/coco/coco_pope_{variant}.json"
    for variant in ("random", "popular", "adversarial")
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ArtifactError(f"Invalid JSONL at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ArtifactError(f"Manifest row must be an object: {path}:{number}")
            rows.append(row)
    return rows


def _images(path: str | Path, split: str) -> dict[int, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    images = {}
    for image in data["images"]:
        identifier = image["id"]
        if (
            not isinstance(identifier, int)
            or isinstance(identifier, bool)
            or identifier in images
        ):
            raise ArtifactError("COCO annotation image IDs must be unique integers")
        filename = image["file_name"]
        if (
            not re.fullmatch(r"\d{12}\.jpg", filename)
            or int(filename[:-4]) != identifier
        ):
            raise ArtifactError(f"Unexpected COCO2017 image filename: {filename!r}")
        images[identifier] = {
            "id": identifier,
            "image_id": identifier,
            "image": f"images/{split}/{filename}",
            "dataset": "coco2017",
            "split": split,
            "width": image["width"],
            "height": image["height"],
            "source_url": f"{COCO_BASE_URL}/{split}/{filename}",
        }
    return images


def _ranked(ids: Iterable[int], *, seed: int, scope: str) -> list[int]:
    # Hash ordering avoids dependencies on RNG implementations or JSON order.
    return sorted(
        ids,
        key=lambda identifier: (
            canonical_hash(
                {
                    "namespace": "tas-v7",
                    "seed": seed,
                    "scope": scope,
                    "image_id": identifier,
                }
            ),
            identifier,
        ),
    )


def build_coco_manifests(
    annotations: Mapping[str, str | Path],
    root: str | Path,
    *,
    seed: int = 20261021,
    exclusions: Iterable[int] = (),
    exclusion_provenance: Any = None,
    audit_count: int = 75,
    cohort_sizes: Mapping[str, int] | None = None,
    query: str = "Describe this image.",
) -> dict[str, Any]:
    """Freeze A/train, disjoint B/C/D/remaining/val, and an audit subset of D.

    ``annotations`` must contain ``train`` and ``val`` COCO annotation JSONs
    with an ``images`` array (captions files are smaller for this purpose).
    A changed seed, annotations, exclusions, sizes or prompt is rejected at an
    existing root. Choose a new study root intentionally to create a new split.
    Training overlap is recorded as unknown, never claimed absent.
    """
    root = Path(root).expanduser().resolve()
    sizes = dict(DEFAULT_COHORT_SIZES if cohort_sizes is None else cohort_sizes)
    if set(sizes) != {"A", "B", "C", "D"} or any(
        type(value) is not int or value < 1 for value in sizes.values()
    ):
        raise ValueError("Cohort sizes must specify positive integer A/B/C/D counts")
    if not 1 <= audit_count <= sizes["D"]:
        raise ValueError("Audit count must be a nonempty subset of D")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A nonempty caption query is required")
    excluded = sorted(set(exclusions))
    if any(type(identifier) is not int or identifier < 0 for identifier in excluded):
        raise ValueError("Exclusions must be COCO integer image IDs")
    annotation_hashes = {
        name: sha256_file(annotations[name]) for name in ("train", "val")
    }
    config = {
        "namespace": "tas-v7",
        "schema": "coco-cohorts-v1",
        "seed": seed,
        "sizes": sizes,
        "audit_count": audit_count,
        "query": query,
        "annotations_sha256": annotation_hashes,
        "excluded_image_ids": excluded,
        "exclusion_provenance": exclusion_provenance,
        "training_overlap": "unknown; COCO split names do not establish model training separation",
        "pope_overlap": "explicit exclusions applied"
        if excluded
        else "not audited; do not claim POPE held-out",
    }
    identity = canonical_hash(config)
    directory = root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    registry_path = directory / "cohorts.json"
    if registry_path.exists():
        saved = json.loads(registry_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity or saved.get("configuration") != config:
            raise ArtifactError(
                "Locked COCO manifest configuration changed; use a new study root"
            )
        for cohort, info in saved["cohorts"].items():
            path = directory / info["file"]
            if not path.is_file() or sha256_file(path) != info["sha256"]:
                raise ArtifactError(f"Locked cohort was modified/missing: {cohort}")
        return _manifest_registry(root, saved)
    train = _images(annotations["train"], "train2017")
    val = _images(annotations["val"], "val2017")
    excluded_set = set(excluded)
    train_ids = _ranked(set(train) - excluded_set, seed=seed, scope="train2017")
    val_ids = _ranked(set(val) - excluded_set, seed=seed, scope="val2017")
    if len(train_ids) < sizes["A"] or len(val_ids) < sum(
        sizes[name] for name in ("B", "C", "D")
    ):
        raise ArtifactError("Insufficient eligible images for the frozen study sizes")
    ids = {"A": train_ids[: sizes["A"]]}
    offset = 0
    for cohort in ("B", "C", "D"):
        ids[cohort] = val_ids[offset : offset + sizes[cohort]]
        offset += sizes[cohort]
    ids["remaining"] = val_ids[offset:]
    ids["audit"] = _ranked(ids["D"], seed=seed, scope="P0-D-subset")[:audit_count]
    cohorts = {}
    for cohort, image_ids in ids.items():
        source = train if cohort == "A" else val
        rows = [
            {**source[identifier], "query": query, "cohort": cohort}
            for identifier in image_ids
        ]
        path = directory / f"{cohort}.jsonl"
        atomic_jsonl(path, rows)
        cohorts[cohort] = {
            "file": path.name,
            "count": len(rows),
            "sha256": sha256_file(path),
            "image_ids": image_ids,
        }
    saved = {"identity": identity, "configuration": config, "cohorts": cohorts}
    atomic_json(registry_path, saved)
    return _manifest_registry(root, saved)


def _manifest_registry(root: Path, saved: dict[str, Any]) -> dict[str, Any]:
    return {
        **saved,
        "root": str(root),
        "registry": str(root / "manifests" / "cohorts.json"),
        "manifests": {
            name: str(root / "manifests" / info["file"])
            for name, info in saved["cohorts"].items()
        },
    }


def verify_manifest_images(
    manifest: str | Path | Iterable[dict[str, Any]],
    root: str | Path,
    *,
    decode: bool = True,
) -> dict[str, Any]:
    """Check all referenced image files; failures are explicit, never omitted."""
    root = Path(root).resolve()
    rows = read_jsonl(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
    files = {}
    errors = []
    for row in rows:
        filename = row["image"]
        path = root / filename
        if Path(filename).is_absolute() or not path.resolve().is_relative_to(root):
            raise ArtifactError(f"Image path escapes dataset root: {filename}")
        if filename in files:
            continue
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise ArtifactError("missing/empty image")
            if decode:
                from PIL import Image

                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if (
                        "width" in row
                        and "height" in row
                        and image.size != (row["width"], row["height"])
                    ):
                        raise ArtifactError(f"unexpected dimensions {image.size}")
            files[filename] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        except (OSError, ValueError, ArtifactError) as exc:
            errors.append({"image": filename, "error": str(exc)})
    if errors:
        raise ArtifactError(
            f"Dataset image validation failed ({len(errors)} files): {errors[:10]}"
        )
    return {"files": files, "identity": canonical_hash(files), "count": len(files)}


def _subset_annotations(source: Path, destination: Path, selected: set[int]) -> None:
    """Produce compact scoring/diagnostic annotations during CPU preparation."""
    expected = {"source_sha256": sha256_file(source), "image_ids": sorted(selected)}
    receipt = destination.with_suffix(".source.json")
    if destination.exists() and receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("input") == expected and previous.get("sha256") == sha256_file(
            destination
        ):
            return
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    result = {
        name: value
        for name, value in data.items()
        if name not in {"images", "annotations"}
    }
    result["images"] = [image for image in data["images"] if image["id"] in selected]
    result["annotations"] = [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"] in selected
    ]
    atomic_json(destination, result)
    atomic_json(receipt, {"input": expected, "sha256": sha256_file(destination)})


def prepare_coco_data(
    root: str | Path,
    *,
    seed: int = 20261021,
    exclusions: Iterable[int] = (),
    exclusion_provenance: Any = None,
    cohorts: Iterable[str] = ("A", "B", "C", "D"),
    audit_count: int = 75,
    cohort_sizes: Mapping[str, int] | None = None,
    annotation_files: Mapping[str, str | Path] | None = None,
    download_images: bool = True,
    download_workers: int = 8,
    progress: Any = print,
) -> dict[str, Any]:
    """Prepare official annotations, locked cohorts, and only selected images.

    By default 30 train images and 690 val images are fetched individually; the
    18GB train archive is never downloaded. Select ``remaining`` explicitly for
    final data. A trusted local ``annotation_files`` mapping (four keys such as
    ``captions_train2017``) supports offline reuse and small test fixtures.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if annotation_files is None:
        archive = download_verified(
            COCO_ANNOTATIONS_URL,
            root / "downloads" / "annotations_trainval2017.zip",
            size=COCO_ANNOTATIONS_BYTES,
            md5=COCO_ANNOTATIONS_MD5,
            progress=progress,
        )
        extraction = root / "annotations-source.json"
        paths = {Path(member).stem: root / member for member in ANNOTATION_MEMBERS}
        source = {
            "archive_sha256": sha256_file(archive),
            "url": COCO_ANNOTATIONS_URL,
            "md5": COCO_ANNOTATIONS_MD5,
        }
        good = False
        if extraction.exists():
            saved = json.loads(extraction.read_text(encoding="utf-8"))
            good = saved.get("source") == source and all(
                path.exists() and saved.get("files", {}).get(name) == sha256_file(path)
                for name, path in paths.items()
            )
        if not good:
            safe_extract_zip(archive, root, members=ANNOTATION_MEMBERS)
            atomic_json(
                extraction,
                {
                    "source": source,
                    "files": {name: sha256_file(path) for name, path in paths.items()},
                },
            )
    else:
        paths = {name: Path(path) for name, path in annotation_files.items()}
        required = {Path(member).stem for member in ANNOTATION_MEMBERS}
        if required - paths.keys():
            raise ValueError(
                f"Missing annotation files: {sorted(required - paths.keys())}"
            )
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    dataset_manifest = root / "dataset-manifest.json"
    if dataset_manifest.exists():
        previous_data = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if previous_data.get("source_annotations_sha256") != source_hashes:
            raise ArtifactError(
                "Frozen scoring annotation source changed; use a new study root"
            )
    registry = build_coco_manifests(
        {"train": paths["captions_train2017"], "val": paths["captions_val2017"]},
        root,
        seed=seed,
        exclusions=exclusions,
        exclusion_provenance=exclusion_provenance,
        audit_count=audit_count,
        cohort_sizes=cohort_sizes,
    )
    cohorts = list(cohorts)
    if any(name not in registry["manifests"] for name in cohorts):
        raise ValueError(f"Unknown cohort selection: {cohorts}")
    unique = {}
    for cohort in cohorts:
        for row in read_jsonl(registry["manifests"][cohort]):
            unique[row["image"]] = row
    if download_images:
        if type(download_workers) is not int or not 1 <= download_workers <= 32:
            raise ValueError("Download workers must be between 1 and 32")

        def fetch(row: dict[str, Any]) -> Path:
            return download_verified(row["source_url"], root / row["image"])

        # File downloads are independent; generation RNG is never involved.
        with ThreadPoolExecutor(max_workers=download_workers) as workers:
            for index, _ in enumerate(workers.map(fetch, unique.values()), 1):
                if progress and (index % 50 == 0 or index == len(unique)):
                    progress(f"Verified COCO files: {index}/{len(unique)}")
        verification = verify_manifest_images(unique.values(), root)
        image_receipt = root / "images-verification.json"
        previous_files = {}
        if image_receipt.exists():
            previous_files = json.loads(image_receipt.read_text(encoding="utf-8"))[
                "files"
            ]
            for name, info in verification["files"].items():
                if name in previous_files and previous_files[name] != info:
                    raise ArtifactError(f"Frozen COCO image bytes changed: {name}")
        all_files = {**previous_files, **verification["files"]}
        atomic_json(
            image_receipt,
            {
                "files": all_files,
                "identity": canonical_hash(all_files),
                "count": len(all_files),
            },
        )
        registry["image_identity"] = verification["identity"]
    compact = {}
    for split in ("train2017", "val2017"):
        # Include every locked cohort, making scoring ready after approval while
        # avoiding repeated parsing of the large train instances annotation.
        selected = set()
        for cohort in registry["manifests"]:
            if (cohort == "A") == (split == "train2017"):
                selected.update(registry["cohorts"][cohort]["image_ids"])
        for kind in ("captions", "instances"):
            name = f"{kind}_{split}"
            destination = root / "scoring_annotations" / f"{name}.json"
            _subset_annotations(paths[name], destination, selected)
            compact[name] = str(destination)
    registry["annotations"] = compact
    registry["source_annotations"] = {name: str(path) for name, path in paths.items()}
    registry["selected_cohorts"] = cohorts
    # Portable summary for notebook restoration, with convenience paths rebased
    # by prepare_coco_data or the notebook's restore helper.
    atomic_json(
        dataset_manifest,
        {
            "schema": "tas-coco-data-v1",
            "identity": registry["identity"],
            "cohorts": registry["cohorts"],
            "selected_cohorts": cohorts,
            "annotations": {
                name: str(Path(path).relative_to(root))
                for name, path in compact.items()
            },
            "source_annotations_sha256": source_hashes,
            "image_identity": registry.get("image_identity"),
        },
    )
    return registry


def load_external_jsonl(
    path: str | Path, *, dataset: str, split: str = "legacy", image_prefix: str = ""
) -> dict[str, Any]:
    """Normalize existing AMBER/POPE query files without downloading images.

    Duplicate image IDs are permitted for multiple questions; duplicate query
    IDs are rejected. No evaluator resources or label prompts are injected.
    """
    rows = []
    identifiers = set()
    for item in read_jsonl(path):
        identifier = item.get("id", item.get("question_id"))
        query = item.get("query", item.get("text"))
        image = item.get("image")
        if identifier is None or canonical_hash(identifier) in identifiers:
            raise ArtifactError("External manifests require unique query IDs")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(image, str)
        ):
            raise ArtifactError(
                "External manifest requires image and nonempty query/text"
            )
        image_path = Path(image_prefix) / image
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ArtifactError(
                "External image paths must remain relative to the supplied image root"
            )
        identifiers.add(canonical_hash(identifier))
        image_identifier = item.get("image_id", image)
        rows.append(
            {
                "id": identifier,
                "image_id": image_identifier,
                "image": image_path.as_posix(),
                "query": query,
                "dataset": dataset,
                "split": split,
                **({"label": item["label"]} if "label" in item else {}),
            }
        )
    return {
        "rows": rows,
        "source_sha256": sha256_file(path),
        "identity": canonical_hash({"dataset": dataset, "split": split, "rows": rows}),
        "training_overlap": "unknown",
        "project_heldout": False,
    }


def pope_image_ids(paths: Iterable[str | Path]) -> list[int]:
    """Extract exact COCO IDs from selected official POPE manifests for exclusion."""
    ids = set()
    for path in paths:
        for row in read_jsonl(path):
            match = re.fullmatch(
                r"(?:COCO_val2014_)?(\d{12})\.jpg", Path(row["image"]).name
            )
            if match is None:
                raise ArtifactError(f"Unknown POPE COCO image alias: {row['image']!r}")
            ids.add(int(match.group(1)))
    return sorted(ids)
