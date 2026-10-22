from pathlib import Path
import pytest
import numpy as np
from tas_runtime.artifacts import ArtifactError
from tas_runtime.persistence import (
    PersistentNpzCache,
    restore_ledger,
    publish_ledger,
    read_incremental_snapshot,
)


def test_sharded_cache_survives_fresh_vm_and_rejects_corruption(tmp_path):
    persistent = tmp_path / "drive"
    first = PersistentNpzCache(tmp_path / "vm1", persistent, flush_every=2)
    first.put({"image": 1}, {"mask": np.eye(4, dtype=bool)}, {"fallback": False})
    first.put({"image": 2}, {"mask": np.ones((4, 4), bool)})
    assert len(list(persistent.glob("*.tar"))) == 1
    second = PersistentNpzCache(tmp_path / "vm2", persistent)
    arrays, metadata = second.get({"image": 1})
    assert metadata["fallback"] is False
    np.testing.assert_array_equal(arrays["mask"], np.eye(4, dtype=bool))
    shard = next(persistent.glob("*.tar"))
    shard.write_bytes(b"broken")
    third = PersistentNpzCache(tmp_path / "vm3", persistent)
    with pytest.raises(ArtifactError):
        third.get({"image": 2})


def test_restore_rejects_a_different_run_or_corrupt_chunk(tmp_path):
    identity = {"science": 1}
    ledger = restore_ledger(tmp_path / "vm1", tmp_path / "drive", identity)
    ledger.record({"image": 1}, {"caption": "A cat."})
    ledger.record({"image": 2}, {"error": "recorded"}, status="error")
    publish_ledger(ledger, tmp_path / "drive")
    restored = restore_ledger(tmp_path / "vm2", tmp_path / "drive", identity)
    assert restored.completed({"image": 1})
    assert restored.records() == ledger.records()
    assert not restored.completed({"image": 2})
    with pytest.raises(ArtifactError):
        restore_ledger(tmp_path / "wrong", tmp_path / "drive", {"science": 2})
    chunk = next((tmp_path / "drive").glob("records-*.json"))
    chunk.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):
        restore_ledger(tmp_path / "corrupt", tmp_path / "drive", identity)


def test_different_writer_cannot_overwrite_published_history(tmp_path):
    identity = {"science": 1}
    first = restore_ledger(tmp_path / "first", tmp_path / "drive", identity)
    second = restore_ledger(tmp_path / "second", tmp_path / "drive", identity)
    first.record("one", {"caption": "A cat."})
    publish_ledger(first, tmp_path / "drive")
    second.record("one", {"caption": "A dog."})
    with pytest.raises(ArtifactError):
        publish_ledger(second, tmp_path / "drive")


def test_incremental_publication_does_not_rewrite_old_results(tmp_path):
    ledger = restore_ledger(tmp_path / "vm1", tmp_path / "drive", {"identity": 1})
    ledger.record({"image": 1}, {"caption": "A cat."})
    first = publish_ledger(ledger, tmp_path / "drive")
    chunks = list((tmp_path / "drive").glob("records-*.json"))
    assert len(chunks) == 1
    original_bytes, original_mtime = (
        chunks[0].read_bytes(),
        chunks[0].stat().st_mtime_ns,
    )
    ledger.record({"image": 2}, {"error": "interrupted"}, status="error")
    ledger.record({"image": 2}, {"caption": "A dog."})
    publish_ledger(ledger, tmp_path / "drive")
    assert (
        chunks[0].read_bytes() == original_bytes
        and chunks[0].stat().st_mtime_ns == original_mtime
    )
    snapshot = read_incremental_snapshot(first)
    assert len(snapshot["records"]) == 3
    restored = restore_ledger(tmp_path / "vm2", tmp_path / "drive", {"identity": 1})
    assert restored.completed({"image": 1}) and restored.completed({"image": 2})
    assert len(restored.records()) == 3
    publish_ledger(restored, tmp_path / "drive")
    assert len(list((tmp_path / "drive").glob("records-*.json"))) == 2


def test_dataset_archive_restores_without_per_file_remote_reads(tmp_path):
    from tas_runtime.persistence import bundle_dataset, restore_dataset_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset-manifest.json").write_text('{"dataset":"fixture"}')
    (source / "images-verification.json").write_text('{"files":{}}')
    (source / "images").mkdir()
    (source / "images/a.jpg").write_bytes(b"image-fixture")
    (source / "images/a.jpg.verified.json").write_text('{"ignored":true}')
    manifest = bundle_dataset(source, tmp_path / "drive")
    restored = tmp_path / "vm/data"
    restore_dataset_bundle(tmp_path / "drive", tmp_path / "vm", restored, manifest)
    assert (restored / "images/a.jpg").read_bytes() == b"image-fixture"
    assert not (restored / "images/a.jpg.verified.json").exists()
    assert bundle_dataset(source, tmp_path / "drive") == manifest


def test_nlp_archive_restore_keeps_real_english_tokenization(tmp_path):
    import os
    import nltk
    from tas_runtime.persistence import restore_nlp
    from tas_runtime.artifacts import atomic_json, sha256_file

    source = Path(os.environ.get("NLTK_DATA", "/nonexistent"))
    if not (source / "tokenizers/punkt_tab.zip").exists():
        pytest.skip("Preparation-owned NLTK ZIP fixture is unavailable")
    # Test against the actual cached tokenizer rather than a synthetic lookup.
    copied = tmp_path / "persistent"
    import shutil

    (copied / "tokenizers").mkdir(parents=True)
    path = copied / "tokenizers/punkt_tab.zip"
    shutil.copyfile(source / "tokenizers/punkt_tab.zip", path)
    atomic_json(
        copied / "resources.json",
        {"sha256": {"tokenizers/punkt_tab.zip": sha256_file(path)}},
    )
    restore_nlp(copied, tmp_path / "vm/nlp")
    assert (tmp_path / "vm/nlp/tokenizers/punkt_tab/english/collocations.tab").exists()
    previous = nltk.data.path[:]
    try:
        nltk.data.path[:] = [str(tmp_path / "vm/nlp")]
        assert nltk.word_tokenize("A cat sits.") == ["A", "cat", "sits", "."]
    finally:
        nltk.data.path[:] = previous


def test_copy_progress_preserves_verification_and_reuse(tmp_path):
    from tas_runtime.artifacts import sha256_file
    from tas_runtime.persistence import copy_verified

    source, destination = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"a" * (5 * 1024 * 1024))
    expected = sha256_file(source)
    chunks = []
    copy_verified(source, destination, sha256=expected, progress=chunks.append)
    assert sum(chunks) == source.stat().st_size
    assert len(chunks) > 1
    assert destination.read_bytes() == source.read_bytes()
    original_mtime = destination.stat().st_mtime_ns
    chunks.clear()
    copy_verified(source, destination, sha256=expected, progress=chunks.append)
    assert sum(chunks) == source.stat().st_size
    assert destination.stat().st_mtime_ns == original_mtime
    with pytest.raises(ArtifactError, match="verification failed"):
        copy_verified(source, destination, sha256="0" * 64, progress=chunks.append)
    assert destination.read_bytes() == source.read_bytes()
