"""Small, CPU-only persistence primitives for reproducible experiments.

Keep the active ledger on the VM's local disk. The persistence module publishes
new records in immutable chunks. This module has no model or torch import.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Configuration mapping keys must be strings")
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers cannot identify an experiment")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def atomic_json(path: str | Path, value: Any) -> Path:
    return _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def atomic_jsonl(path: str | Path, rows: Any) -> Path:
    """Write a bounded manifest/report atomically, never an active append journal."""
    return _atomic_bytes(
        path, "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    )


class ArtifactError(RuntimeError):
    """An artifact is incomplete, corrupt, or belongs to another configuration."""


class NpzCache:
    """Versioned, content-addressed numpy arrays plus JSON metadata in one file.

    Keys must contain all dependencies (image digest, revisions, precision,
    perception settings, geometry, view version, etc.). Arrays with object dtype
    are rejected, and reads always use ``allow_pickle=False``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: Any) -> Path:
        digest = canonical_hash(key)
        return self.root / digest[:2] / f"{digest}.npz"

    def put(
        self,
        key: Any,
        arrays: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        import numpy as np

        if "__metadata__" in arrays:
            raise ValueError("__metadata__ is reserved")
        converted = {}
        for name, value in arrays.items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError("Cache array names must be simple nonempty strings")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("Object arrays are forbidden in experiment caches")
            converted[name] = array
        existing = self.get(key)
        if existing is not None:
            old_arrays, old_metadata = existing
            same = (
                old_metadata == _canonical(metadata or {})
                and old_arrays.keys() == converted.keys()
                and all(
                    np.array_equal(
                        old_arrays[name], value, equal_nan=value.dtype.kind in "fc"
                    )
                    for name, value in converted.items()
                )
            )
            if not same:
                raise ArtifactError(
                    "A populated cache key cannot change; version its dependencies"
                )
            return self.path(key)
        header = {
            "schema": "tas-arrays-v1",
            "key": _canonical(key),
            "key_hash": canonical_hash(key),
            "metadata": _canonical(metadata or {}),
        }
        converted["__metadata__"] = np.frombuffer(
            canonical_json(header).encode("utf-8"), dtype=np.uint8
        )
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".arrays-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **converted)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def get(self, key: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
        import numpy as np

        path = self.path(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                header = json.loads(archive["__metadata__"].tobytes().decode("utf-8"))
                if header.get("schema") != "tas-arrays-v1" or header.get(
                    "key_hash"
                ) != canonical_hash(key):
                    raise ArtifactError(f"Cache identity mismatch: {path}")
                if canonical_hash(header["key"]) != canonical_hash(key):
                    raise ArtifactError(f"Cache key mismatch: {path}")
                arrays = {
                    name: archive[name].copy()
                    for name in archive.files
                    if name != "__metadata__"
                }
            return arrays, header["metadata"]
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"Cannot safely read cache {path}: {exc}") from exc


class ResultLedger:
    """Single-writer append journal with configuration validation and retries.

    Every successful append is fsynced locally. A crash can leave a partial last
    record: its bytes are quarantined and only that uncommitted tail is removed.
    Previously committed records, including failed attempts, are never discarded.
    All identity-bearing fields must be supplied by the study runner; this class
    cannot infer omitted model/view/dataset/source revisions.
    """

    def __init__(self, directory: str | Path, identity: Any):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = _canonical(identity)
        self.identity_hash = canonical_hash(identity)
        manifest = self.directory / "identity.json"
        expected = {
            "schema": "tas-ledger-v1",
            "identity": self.identity,
            "identity_hash": self.identity_hash,
        }
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                raise ArtifactError(
                    f"Refusing incompatible results directory: {self.directory}"
                )
        else:
            atomic_json(manifest, expected)
        self.journal = self.directory / "results.jsonl"
        self._rows: list[dict[str, Any]] = []
        self._latest: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.journal.exists():
            return
        with self.journal.open("rb+") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                    checksum = row.get("checksum")
                    content = {
                        name: value for name, value in row.items() if name != "checksum"
                    }
                    if checksum != canonical_hash(content):
                        raise ValueError("record checksum mismatch")
                    if (
                        row["sequence"] != len(self._rows)
                        or row["identity_hash"] != self.identity_hash
                    ):
                        raise ValueError("record order or run identity mismatch")
                    if canonical_hash(row["key"]) != row["key_hash"] or row[
                        "status"
                    ] not in {"ok", "error"}:
                        raise ValueError("record key or status mismatch")
                    if row.get("attempt") != self._attempts.get(row["key_hash"], 0) + 1:
                        raise ValueError("record retry number mismatch")
                    previous = self._latest.get(row["key_hash"])
                    if previous is not None and previous["status"] == "ok":
                        raise ValueError(
                            "a committed successful result was overwritten"
                        )
                except (ValueError, KeyError, TypeError) as exc:
                    if line.endswith(b"\n"):
                        raise ArtifactError(
                            f"Corrupt committed journal record at byte {offset}"
                        ) from exc
                    tail = self.directory / f"interrupted-tail-{time.time_ns()}.bin"
                    _atomic_bytes(tail, line)
                    handle.seek(offset)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                self._rows.append(row)
                self._latest[row["key_hash"]] = row
                self._attempts[row["key_hash"]] = row["attempt"]
                if not line.endswith(b"\n"):
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    break

    def latest(self, key: Any) -> dict[str, Any] | None:
        row = self._latest.get(canonical_hash(key))
        return json.loads(canonical_json(row)) if row is not None else None

    def completed(self, key: Any) -> bool:
        row = self._latest.get(canonical_hash(key))
        return row is not None and row["status"] == "ok"

    def record(self, key: Any, payload: Any, status: str = "ok") -> dict[str, Any]:
        if status not in {"ok", "error"}:
            raise ValueError(
                "Only ok/error records are supported; failures must not silently skip images"
            )
        key_hash = canonical_hash(key)
        previous = self._latest.get(key_hash)
        if previous is not None and previous["status"] == "ok":
            if status == "ok" and previous["payload"] == _canonical(payload):
                return json.loads(canonical_json(previous))
            raise ArtifactError(
                "A completed result cannot be overwritten; create a new run identity"
            )
        row = {
            "sequence": len(self._rows),
            "identity_hash": self.identity_hash,
            "key": _canonical(key),
            "key_hash": key_hash,
            "status": status,
            "attempt": self._attempts.get(key_hash, 0) + 1,
            "time_ns": time.time_ns(),
            "payload": _canonical(payload),
        }
        row["checksum"] = canonical_hash(row)
        with self.journal.open("ab") as handle:
            handle.write((canonical_json(row) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        self._rows.append(row)
        self._latest[key_hash] = row
        self._attempts[key_hash] = row["attempt"]
        return json.loads(canonical_json(row))

    def records(self, latest: bool = False) -> list[dict[str, Any]]:
        rows = list(self._latest.values()) if latest else self._rows
        return json.loads(canonical_json(rows))

    @property
    def record_count(self) -> int:
        return len(self._rows)

    def records_since(self, start: int) -> list[dict[str, Any]]:
        if type(start) is not int or not 0 <= start <= len(self._rows):
            raise ValueError("Invalid journal position")
        return json.loads(canonical_json(self._rows[start:]))

    @classmethod
    def restore_records(
        cls, directory: str | Path, identity: Any, records: Any
    ) -> "ResultLedger":
        """Validate restored records using the same rules as the local journal."""
        ledger = cls(directory, identity)
        if ledger._rows:
            raise ArtifactError("Restore requires an empty local ledger")
        atomic_jsonl(ledger.journal, records)
        ledger._load()
        return ledger
