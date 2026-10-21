from pathlib import Path
import tempfile
import unittest

from tas_runtime.artifacts import (
    ArtifactError,
    NpzCache,
    ResultLedger,
    canonical_hash,
    canonical_json,
)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_canonical_identity_rejects_ambiguous_and_nonfinite_values(self):
        self.assertEqual(
            canonical_hash({"b": [1, 2], "a": Path("x")}),
            canonical_hash({"a": "x", "b": (1, 2)}),
        )
        with self.assertRaises(ValueError):
            canonical_json({"coefficient": float("nan")})
        with self.assertRaises(TypeError):
            canonical_json({1: "ambiguous"})

    def test_ledger_retry_resume_and_completed_immutability(self):
        identity = {"model": "revision-a", "views": "v7", "batch": "serial"}
        key = {"image": 42, "method": "F4", "seed": 7}
        ledger = ResultLedger(self.root, identity)
        ledger.record(key, {"error": "temporary OOM"}, status="error")
        self.assertFalse(ledger.completed(key))
        ledger.record(key, {"caption": "A dog."})
        restored = ResultLedger(self.root, identity)
        self.assertTrue(restored.completed(key))
        self.assertEqual(restored.latest(key)["attempt"], 2)
        self.assertEqual(len(restored.records()), 2)
        self.assertEqual(len(restored.records(latest=True)), 1)
        restored.record(key, {"caption": "A dog."})  # Idempotent publication.
        self.assertEqual(len(restored.records()), 2)
        with self.assertRaises(ArtifactError):
            restored.record(key, {"caption": "A cat."})
        with self.assertRaises(ArtifactError):
            ResultLedger(self.root, {**identity, "views": "old-v6"})

    def test_only_partial_tail_is_recovered_without_losing_committed_outputs(self):
        ledger = ResultLedger(self.root, {"model": "a"})
        ledger.record("one", {"caption": "kept"})
        with ledger.journal.open("ab") as handle:
            handle.write(b'{"sequence":1,"payload":{"caption":"interrupt')
        resumed = ResultLedger(self.root, {"model": "a"})
        self.assertTrue(resumed.completed("one"))
        self.assertEqual(len(list(self.root.glob("interrupted-tail-*.bin"))), 1)
        resumed.record("two", {"caption": "next"})
        self.assertEqual(len(ResultLedger(self.root, {"model": "a"}).records()), 2)

    def test_corrupt_committed_record_fails_instead_of_skipping(self):
        ledger = ResultLedger(self.root, {"model": "a"})
        ledger.record("one", {"caption": "original"})
        ledger.journal.write_text(
            ledger.journal.read_text().replace("original", "modified")
        )
        with self.assertRaises(ArtifactError):
            ResultLedger(self.root, {"model": "a"})

    def test_restored_records_validate_checksums_and_preserve_errors(self):
        identity = {"source_hash": "a", "model": "revision"}
        ledger = ResultLedger(self.root / "local", identity)
        ledger.record("one", {"caption": "kept"})
        ledger.record("two", {"error": "recorded"}, status="error")
        restored = ResultLedger.restore_records(
            self.root / "new-vm", identity, ledger.records()
        )
        self.assertEqual(restored.records(), ledger.records())
        self.assertTrue(restored.completed("one"))
        self.assertFalse(restored.completed("two"))
        with self.assertRaises(ArtifactError):
            ResultLedger.restore_records(
                self.root / "wrong", {"model": "other"}, ledger.records()
            )
        corrupted = ledger.records()
        corrupted[0]["payload"] = {"caption": "changed"}
        with self.assertRaises(ArtifactError):
            ResultLedger.restore_records(self.root / "corrupt", identity, corrupted)

    def test_npz_cache_never_uses_pickle_and_keeps_metadata(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        cache = NpzCache(self.root / "masks")
        key = {"image_sha256": "a", "revision": "fixed", "geometry": "v7"}
        arrays = {
            "union": np.array([[0, 1], [1, 0]], dtype=bool),
            "instances": np.zeros((0, 2, 2), dtype=bool),
        }
        cache.put(key, arrays, {"labels": ["cat"], "fallback": False})
        cache.put(key, arrays, {"labels": ["cat"], "fallback": False})  # Idempotent.
        loaded, metadata = cache.get(key)
        np.testing.assert_array_equal(loaded["union"], arrays["union"])
        self.assertEqual(metadata["labels"], ["cat"])
        self.assertIsNone(cache.get({**key, "geometry": "old"}))
        with self.assertRaises(ArtifactError):
            cache.put(key, {"union": np.ones((2, 2), dtype=bool)}, {"labels": ["cat"]})
        with self.assertRaises(ValueError):
            cache.put(key, {"unsafe": np.array([{"value": 1}], dtype=object)})
        cache.path(key).write_bytes(b"damaged")
        with self.assertRaises(ArtifactError):
            cache.get(key)


if __name__ == "__main__":
    unittest.main()
