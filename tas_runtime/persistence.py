"""Local working files with verified, bounded writes to persistent storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from .artifacts import (
    ArtifactError,
    NpzCache,
    ResultLedger,
    atomic_json,
    canonical_hash,
    sha256_file,
)


def bundle_dataset(source_root, persistent_root):
    """Prepare one local-disk archive so future GPU restores avoid tiny Drive reads."""
    source_root, persistent_root = Path(source_root), Path(persistent_root)
    dependency = {
        name: sha256_file(source_root / name)
        for name in ("dataset-manifest.json", "images-verification.json")
    }
    label = canonical_hash(dependency)
    relative = Path("dataset-bundles") / f"coco-{label}.tar"
    target = persistent_root / relative
    receipt = target.with_suffix(".json")
    if target.exists() and receipt.exists():
        info = json.loads(receipt.read_text())
        if (
            info.get("inputs") == dependency
            and info.get("size") == target.stat().st_size
        ):
            return info
    files = [source_root / name for name in dependency]
    for directory in ("manifests", "scoring_annotations", "images"):
        files += [
            path
            for path in (source_root / directory).rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".json", ".jsonl", ".jpg", ".jpeg", ".png"}
            and not path.name.endswith(".verified.json")
        ]
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "dataset.tar"
        with tarfile.open(archive_path, "w") as archive:
            for source in sorted(files):
                archive.add(
                    source,
                    arcname=str(source.relative_to(source_root)),
                    recursive=False,
                )
        digest = sha256_file(archive_path)
        copy_verified(archive_path, target, sha256=digest)
    info = {
        "file": str(relative),
        "sha256": digest,
        "size": target.stat().st_size,
        "inputs": dependency,
    }
    atomic_json(receipt, info)
    return info


def restore_dataset_bundle(persistent_root, local_root, destination, info):
    relative = Path(info["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactError("Unsafe dataset archive path.")
    archive_path = Path(local_root) / "downloads" / relative.name
    copy_verified(Path(persistent_root) / relative, archive_path, sha256=info["sha256"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            target = destination / path
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or not target.resolve().is_relative_to(destination.resolve())
            ):
                raise ArtifactError("Unsafe dataset archive member.")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".restoring")
            with archive.extractfile(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            os.replace(temporary, target)


def restore_nlp(source_root, destination):
    """Copy a few fixed ZIPs instead of hundreds of remote language-resource files."""
    from .assets import safe_extract_zip

    source_root, destination = Path(source_root), Path(destination)
    info = json.loads((source_root / "resources.json").read_text())
    for relative, digest in info["sha256"].items():
        source = source_root / relative
        target = destination / relative
        copy_verified(source, target, sha256=digest)
        if source.name in {
            "punkt.zip",
            "punkt_tab.zip",
            "averaged_perceptron_tagger_eng.zip",
        }:
            with zipfile.ZipFile(target) as archive:
                if source.name == "punkt.zip":
                    selected = [
                        name
                        for name in archive.namelist()
                        if name in {"punkt/english.pickle", "punkt/PY3/english.pickle"}
                    ]
                elif source.name == "punkt_tab.zip":
                    selected = [
                        name
                        for name in archive.namelist()
                        if name.startswith("punkt_tab/english/")
                        and not name.endswith("/")
                    ]
                else:
                    selected = [
                        name for name in archive.namelist() if not name.endswith("/")
                    ]
            safe_extract_zip(target, target.parent, members=selected)
    atomic_json(destination / "resources.json", info)
    return info


def copy_verified(source, destination, *, sha256=None):
    source, destination = Path(source), Path(destination)
    digest = sha256 or sha256_file(source)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        if sha256_file(destination) == digest:
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != digest:
        raise ArtifactError(f"Copy verification failed: {destination}")
    os.replace(temporary, destination)
    return destination


def restore_ledger(local_dir, persistent_dir, identity):
    local_dir, persistent_dir = Path(local_dir), Path(persistent_dir)
    digest = canonical_hash(identity)
    index = persistent_dir / "ledger-index.json"
    if not (local_dir / "results.jsonl").exists() and index.exists():
        snapshot = read_incremental_snapshot(index)
        if snapshot["identity_hash"] != digest:
            raise ArtifactError("Incremental snapshot run identity mismatch.")
        return ResultLedger.restore_records(local_dir, identity, snapshot["records"])
    return ResultLedger(local_dir, identity)


def read_incremental_snapshot(index_path):
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    if (
        index.get("schema") != "tas-ledger-chunks-v1"
        or canonical_hash(index["identity"]) != index["identity_hash"]
    ):
        raise ArtifactError("Invalid incremental ledger manifest.")
    rows = []
    for chunk in index["chunks"]:
        source = index_path.parent / chunk["file"]
        if source.parent != index_path.parent or sha256_file(source) != chunk["sha256"]:
            raise ArtifactError("Result chunk failed verification.")
        part = json.loads(source.read_text())
        if chunk["start"] != len(rows) or len(part) != chunk["count"]:
            raise ArtifactError("Result chunk sequence mismatch.")
        rows.extend(part)
    if len(rows) != index["records"]:
        raise ArtifactError("Result chunks are incomplete.")
    if rows and rows[-1]["checksum"] != index["last_checksum"]:
        raise ArtifactError("Result chunk history does not match the manifest.")
    return {
        "identity": index["identity"],
        "identity_hash": index["identity_hash"],
        "records": rows,
    }


def publish_ledger(ledger, destination):
    """Publish only new records; cumulative Drive traffic is linear in outputs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "ledger-index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {
            "schema": "tas-ledger-chunks-v1",
            "identity": ledger.identity,
            "identity_hash": ledger.identity_hash,
            "records": 0,
            "chunks": [],
        }
    )
    if index["identity_hash"] != ledger.identity_hash:
        raise ArtifactError("Result chunks belong to another run.")
    count = ledger.record_count
    if index["records"] > count:
        raise ArtifactError(
            "Persistent results are ahead of this writer; restore before continuing."
        )
    if (
        index["records"]
        and ledger.records_since(index["records"] - 1)[0]["checksum"]
        != index["last_checksum"]
    ):
        raise ArtifactError(
            "Persistent results have a different history; use one writer and restore before continuing."
        )
    new = ledger.records_since(index["records"])
    if not new:
        return index_path
    digest = canonical_hash(new)
    name = f"records-{index['records']:08d}-{count:08d}-{digest[:16]}.json"
    chunk_path = destination / name
    atomic_json(chunk_path, new)
    index["chunks"].append(
        {
            "file": name,
            "sha256": sha256_file(chunk_path),
            "start": index["records"],
            "count": len(new),
        }
    )
    index["records"] = count
    index["last_checksum"] = new[-1]["checksum"]
    atomic_json(index_path, index)
    return index_path


class PersistentNpzCache:
    """Bundle multiple immutable per-image caches to avoid many Drive writes.

    A compact index maps exact cache keys to SHA-verified tar shards. Mutable
    model/KV objects are never serialized. Only safe .npz files are restored.
    """

    def __init__(self, local_root, persistent_root, *, flush_every=16):
        self.local = NpzCache(local_root)
        self.persistent = Path(persistent_root)
        self.persistent.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persistent / "index.json"
        self.index = (
            json.loads(self.index_path.read_text())
            if self.index_path.exists()
            else {"schema": "tas-npz-shards-v1", "entries": {}}
        )
        if self.index.get("schema") != "tas-npz-shards-v1":
            raise ArtifactError("Unsupported persistent cache index.")
        self.pending, self.restored = {}, set()
        self.flush_every = flush_every

    def get(self, key):
        value = self.local.get(key)
        if value is not None:
            return value
        digest = canonical_hash(key)
        entry = self.index["entries"].get(digest)
        if entry is None:
            return None
        shard = entry["shard"]
        if shard not in self.restored:
            source = self.persistent / shard
            if (
                source.parent != self.persistent
                or sha256_file(source) != entry["shard_sha256"]
            ):
                raise ArtifactError("Persistent perception shard failed verification.")
            with tarfile.open(source, "r") as archive:
                for member in archive.getmembers():
                    relative = Path(member.name)
                    if (
                        not member.isfile()
                        or relative.is_absolute()
                        or ".." in relative.parts
                        or relative.suffix != ".npz"
                    ):
                        raise ArtifactError("Unsafe persistent cache archive member.")
                    target = self.local.root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    temporary = target.with_suffix(".restoring")
                    with stream, temporary.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    os.replace(temporary, target)
            self.restored.add(shard)
        path = self.local.path(key)
        if not path.exists() or sha256_file(path) != entry["file_sha256"]:
            raise ArtifactError("Perception cache member failed verification.")
        return self.local.get(key)

    def put(self, key, arrays, metadata=None):
        path = self.local.put(key, arrays, metadata)
        digest = canonical_hash(key)
        if digest not in self.index["entries"]:
            self.pending[digest] = path
        if len(self.pending) >= self.flush_every:
            self.flush()
        return path

    def flush(self):
        if not self.pending:
            return
        files = {digest: sha256_file(path) for digest, path in self.pending.items()}
        label = canonical_hash(files)
        shard_name = f"perception-{label}.tar"
        with tempfile.TemporaryDirectory(dir=self.local.root) as temporary:
            source = Path(temporary) / shard_name
            with tarfile.open(source, "w") as archive:
                for path in self.pending.values():
                    archive.add(
                        path,
                        arcname=str(path.relative_to(self.local.root)),
                        recursive=False,
                    )
            shard_sha = sha256_file(source)
            copy_verified(source, self.persistent / shard_name, sha256=shard_sha)
        # Single writer per cache index; fail if another session changed it.
        if self.index_path.exists():
            current = json.loads(self.index_path.read_text())
            if current != self.index:
                raise ArtifactError(
                    "Another session changed this cache index. Use one writer per study."
                )
        for digest, file_sha in files.items():
            self.index["entries"][digest] = {
                "shard": shard_name,
                "shard_sha256": shard_sha,
                "file_sha256": file_sha,
            }
        atomic_json(self.index_path, self.index)
        self.pending.clear()
