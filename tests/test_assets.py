import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
import zipfile

from tas_runtime.artifacts import ArtifactError, atomic_json
from tas_runtime.assets import (
    MODEL_SPECS,
    asset_paths,
    download_verified,
    prepare_assets,
    safe_extract_zip,
)


class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = b"deterministic-checkpoint-fixture\n" * 100
        cls.requests = []
        content = cls.content
        requests = cls.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requested_range = self.headers.get("Range")
                requests.append((self.path, requested_range))
                offset = (
                    int(requested_range.split("=")[1].split("-")[0])
                    if requested_range
                    else 0
                )
                if self.path == "/ignore-range":
                    offset = 0
                self.send_response(206 if offset else 200)
                if offset:
                    self.send_header(
                        "Content-Range",
                        f"bytes {offset}-{len(content) - 1}/{len(content)}",
                    )
                self.send_header("Content-Length", str(len(content) - offset))
                self.end_headers()
                self.wfile.write(content[offset:])

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sha = hashlib.sha256(self.content).hexdigest()

    def tearDown(self):
        self.temporary.cleanup()

    def test_pinned_download_cached_without_another_network_request(self):
        path = self.root / "model.bin"
        download_verified(
            self.base + "/model",
            path,
            sha256=self.sha,
            size=len(self.content),
            retries=1,
        )
        count = len(self.requests)
        download_verified(
            self.base + "/model",
            path,
            sha256=self.sha,
            size=len(self.content),
            retries=1,
        )
        self.assertEqual(path.read_bytes(), self.content)
        self.assertEqual(len(self.requests), count)
        self.assertEqual(
            json.loads(path.with_name("model.bin.verified.json").read_text())["sha256"],
            self.sha,
        )

    def test_resume_and_server_ignoring_range_both_preserve_correct_bytes(self):
        for endpoint in ("/model", "/ignore-range"):
            with self.subTest(endpoint=endpoint):
                path = self.root / endpoint[1:] / "model.bin"
                path.parent.mkdir()
                path.with_name("model.bin.part").write_bytes(self.content[:101])
                atomic_json(
                    path.with_name("model.bin.part.json"),
                    {
                        "url": self.base + endpoint,
                        "sha256": self.sha,
                        "md5": None,
                        "size": len(self.content),
                    },
                )
                download_verified(
                    self.base + endpoint,
                    path,
                    sha256=self.sha,
                    size=len(self.content),
                    retries=1,
                )
                self.assertEqual(path.read_bytes(), self.content)
                self.assertEqual(self.requests[-1][1], "bytes=101-")

    def test_wrong_checksum_never_publishes_final_file(self):
        path = self.root / "wrong.bin"
        with self.assertRaises(ArtifactError):
            download_verified(
                self.base + "/model",
                path,
                sha256="0" * 64,
                size=len(self.content),
                retries=1,
            )
        self.assertFalse(path.exists())
        with self.assertRaises(ValueError):
            download_verified("http://example.com/plaintext", path)

    def test_modified_cached_file_is_replaced_after_verification(self):
        path = self.root / "model.bin"
        download_verified(self.base + "/model", path, sha256=self.sha, retries=1)
        path.write_bytes(b"modified")
        download_verified(self.base + "/model", path, sha256=self.sha, retries=1)
        self.assertEqual(path.read_bytes(), self.content)
        self.assertEqual(len(list(self.root.glob("model.bin.invalid-*"))), 1)

    def test_portable_model_manifest_can_be_rebased(self):
        spec = {
            "fixture": {
                "repo_id": "official/test",
                "revision": "a" * 40,
                "files": [
                    {
                        "filename": "config.json",
                        "url": self.base + "/model",
                        "sha256": self.sha,
                        "size": len(self.content),
                    }
                ],
            }
        }
        with patch.dict(MODEL_SPECS, spec, clear=True):
            manifest = prepare_assets(self.root / "first", progress=None)
            second = prepare_assets(self.root / "second", progress=None)
        self.assertEqual(manifest["identity"], second["identity"])
        rebased = asset_paths(self.root / "second", manifest)
        self.assertEqual(rebased["fixture"], second["paths"]["fixture"])
        self.assertNotIn(str(self.root), json.dumps(manifest["assets"]))

    def test_selected_zip_files_and_traversal_rejection(self):
        archive = self.root / "safe.zip"
        with zipfile.ZipFile(archive, "w") as source:
            source.writestr("annotations/wanted.json", "{}")
            source.writestr("annotations/not-needed.json", "{}")
        files = safe_extract_zip(
            archive, self.root / "output", members=["annotations/wanted.json"]
        )
        self.assertEqual(len(files), 1)
        self.assertFalse((self.root / "output/annotations/not-needed.json").exists())
        with zipfile.ZipFile(archive, "w") as source:
            source.writestr("../escape", "bad")
        with self.assertRaises((ValueError, ArtifactError)):
            safe_extract_zip(archive, self.root / "output")
        self.assertFalse((self.root / "escape").exists())

    def test_symlink_zip_and_size_bomb_are_rejected(self):
        archive = self.root / "link.zip"
        with zipfile.ZipFile(archive, "w") as source:
            info = zipfile.ZipInfo("symlink")
            info.external_attr = 0o120777 << 16
            source.writestr(info, "/etc/passwd")
        with self.assertRaises(ArtifactError):
            safe_extract_zip(archive, self.root / "output")
        with zipfile.ZipFile(archive, "w") as source:
            source.writestr("large", b"0" * 100)
        with self.assertRaises(ArtifactError):
            safe_extract_zip(archive, self.root / "output", max_bytes=10)

    def test_model_specs_pin_one_weight_format_and_checksums(self):
        for name, model in MODEL_SPECS.items():
            self.assertNotIn(model["revision"], {"main", "master", "latest"})
            for file in model["files"]:
                self.assertRegex(file["sha256"], "^[0-9a-f]{64}$")
                self.assertGreater(file["size"], 0)
                self.assertTrue(file["url"].startswith("https://"))
            files = [file["filename"] for file in model["files"]]
            self.assertFalse(
                "model.safetensors" in files and "pytorch_model.bin" in files
            )


if __name__ == "__main__":
    unittest.main()
