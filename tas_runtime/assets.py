"""Pinned, reusable checkpoint downloads without importing inference packages.

Metadata was verified against the official model repositories/releases. Weight
formats are selected explicitly, so alternate TF/Flax/PyTorch copies are not
accidentally downloaded. The active GPU runtime can use these paths offline.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Callable, Iterable

from .artifacts import ArtifactError, atomic_json, canonical_hash

# Full revisions and file checksums, not moving main/latest references.
MODEL_SPECS = {
    "llava": {
        "repo_id": "liuhaotian/llava-v1.5-7b",
        "revision": "4481d270cc22fd5c4d1bb5df129622006ccd9234",
        "files": [
            {
                "filename": "config.json",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/config.json",
                "sha256": "f112de882e31171a105f0d3f69d9e6e8700a40149f0944f4b507d3a41789e492",
                "size": 1161,
            },
            {
                "filename": "generation_config.json",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/generation_config.json",
                "sha256": "e453057ea6a89d902d7326b3addbe312a7a6c8faabf777a521c9f2d563f79b8d",
                "size": 124,
            },
            {
                "filename": "pytorch_model-00001-of-00002.bin",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/pytorch_model-00001-of-00002.bin",
                "sha256": "492fdfa63cf314097d99da81a3586d54a97adc1471a0ae0535f5caa269b86314",
                "size": 9976634558,
            },
            {
                "filename": "pytorch_model-00002-of-00002.bin",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/pytorch_model-00002-of-00002.bin",
                "sha256": "c0eb94eb3fa578a4c63567f2eca6b0acf6a44a891c257aca0df88ffa891de72e",
                "size": 3542276251,
            },
            {
                "filename": "pytorch_model.bin.index.json",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/pytorch_model.bin.index.json",
                "sha256": "487578d3f7ec21c057596d0f9dc71084acc349eb8f13bd8b5af86c26460f2d31",
                "size": 27068,
            },
            {
                "filename": "special_tokens_map.json",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/special_tokens_map.json",
                "sha256": "719833ff26ac897a3ec8ed946028a135de2a351470af59b4008744ab1f0ee9b7",
                "size": 438,
            },
            {
                "filename": "tokenizer.model",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/tokenizer.model",
                "sha256": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
                "size": 499723,
            },
            {
                "filename": "tokenizer_config.json",
                "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/4481d270cc22fd5c4d1bb5df129622006ccd9234/tokenizer_config.json",
                "sha256": "cf01145c6de57f061991ff871e85d0402c38481f968daea8874c431cd0c7740f",
                "size": 749,
            },
        ],
    },
    "vision": {
        "repo_id": "openai/clip-vit-large-patch14-336",
        "revision": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": [
            {
                "filename": "config.json",
                "url": "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/config.json",
                "sha256": "51b1c14aabcdf639c4a0370eeda1010b773bbe1df78319c7d0f5882c22ac0ac0",
                "size": 4757,
            },
            {
                "filename": "preprocessor_config.json",
                "url": "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/preprocessor_config.json",
                "sha256": "d253881f65322dc546df59cf925a408e5538b8ecb5a1b496cdd36af9992686d4",
                "size": 316,
            },
            {
                "filename": "pytorch_model.bin",
                "url": "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/pytorch_model.bin",
                "sha256": "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801",
                "size": 1711974081,
            },
        ],
    },
    "blip": {
        "repo_id": "Salesforce/blip-vqa-base",
        "revision": "787b3d35d57e49572baabd22884b3d5a05acf072",
        "files": [
            {
                "filename": "config.json",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/config.json",
                "sha256": "689a09e2a9980b7fcad329271c032254fde23b1ee7a90c67b003e1867dc9c098",
                "size": 4559,
            },
            {
                "filename": "model.safetensors",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/model.safetensors",
                "sha256": "33786eed34def0c95fa948128cb4386be9b9219aa2c2e25f1c9c744692121bb7",
                "size": 1538800584,
            },
            {
                "filename": "preprocessor_config.json",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/preprocessor_config.json",
                "sha256": "0aa66e2e9ac3ea3b5cd4388c35072e22db4e1cc1f96c7872bed07749c712ade1",
                "size": 445,
            },
            {
                "filename": "special_tokens_map.json",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/special_tokens_map.json",
                "sha256": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
                "size": 125,
            },
            {
                "filename": "tokenizer.json",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/tokenizer.json",
                "sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                "size": 711396,
            },
            {
                "filename": "tokenizer_config.json",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/tokenizer_config.json",
                "sha256": "48d1c9120fe61c9741286050189b030e936f23797a65fd0a179078661e1c43bb",
                "size": 592,
            },
            {
                "filename": "vocab.txt",
                "url": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/787b3d35d57e49572baabd22884b3d5a05acf072/vocab.txt",
                "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
                "size": 231508,
            },
        ],
    },
    "yolo": {
        "revision": "ultralytics-assets-v8.3.0",
        "files": [
            {
                "sha256": "095f5266bb9b654bd5ad9e21e9cdeda78e0f2c8460f5d652eaf04bab7ee251cf",
                "size": 27169314,
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt",
                "filename": "yolov8s-world.pt",
            }
        ],
    },
    "mobile_sam": {
        "revision": "f706ad9c4eb7f219c00d9050e46328518ffb65d2",
        "files": [
            {
                "sha256": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
                "size": 40728226,
                "url": "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt",
                "filename": "mobile_sam.pt",
            }
        ],
    },
    "clip_text": {
        "revision": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "files": [
            {
                "filename": "ViT-B-32.pt",
                "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
                "size": 353976522,
            }
        ],
    },
}

MOBILE_SAM_REVISION = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
POPE_REVISION = "08d957b917e5a378a2f99d35b6293c536a66298b"


def _safe_relative(name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"Unsafe artifact filename: {name!r}")
    if "\\" in name or ":" in name:
        raise ValueError(f"Unsafe artifact filename: {name!r}")
    return Path(*relative.parts)


def _file_receipt(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    receipt = path.with_name(path.name + ".verified.json")
    try:
        info = path.stat()
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            data.get("expected") == expected
            and data.get("size") == info.st_size
            and data.get("mtime_ns") == info.st_mtime_ns
        ):
            return data
    except (OSError, ValueError):
        pass
    return None


def download_verified(
    url: str,
    path: str | Path,
    *,
    sha256: str | None = None,
    size: int | None = None,
    md5: str | None = None,
    progress: Callable[[str], None] | None = None,
    retries: int = 4,
    timeout: float = 90,
    verify_cache: bool = False,
) -> Path:
    """Resume a download, verify content, and atomically publish the final file.

    TLS is never disabled. HTTP is accepted only for localhost test fixtures.
    If a server ignores Range, the partial file is replaced instead of appended.
    Cached files use a checksum receipt only while size/mtime and expected source
    match; ``verify_cache=True`` forces a full rehash. Partial bytes remain on
    disk after network interruption and are reused on the next invocation.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in artifact URLs")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Artifact downloads require HTTPS")
    for name, value, length in (("sha256", sha256, 64), ("md5", md5, 32)):
        if value is not None and not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            raise ValueError(f"Invalid {name} checksum")
    if retries < 1 or (size is not None and size < 0):
        raise ValueError("Retries must be positive and expected size nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {"url": url, "sha256": sha256, "md5": md5, "size": size}

    def validate(candidate: Path) -> tuple[bool, str]:
        if size is not None and candidate.stat().st_size != size:
            return False, ""
        sha = hashlib.sha256()
        weak = hashlib.md5() if md5 is not None else None
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                sha.update(chunk)
                if weak is not None:
                    weak.update(chunk)
        digest = sha.hexdigest()
        return (
            (sha256 is None or digest == sha256)
            and (weak is None or weak.hexdigest() == md5)
        ), digest

    def publish_receipt(digest: str) -> None:
        info = path.stat()
        atomic_json(
            path.with_name(path.name + ".verified.json"),
            {
                "schema": "tas-file-v1",
                "expected": expected,
                "sha256": digest,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            },
        )

    if path.is_file():
        receipt = None if verify_cache else _file_receipt(path, expected)
        if receipt is not None:
            return path
        valid, digest = validate(path)
        if valid:
            publish_receipt(digest)
            return path
        # A final filename is never accepted on size alone when a checksum exists.
        invalid = path.with_name(f"{path.name}.invalid-{time.time_ns()}")
        os.replace(path, invalid)

    partial = path.with_name(path.name + ".part")
    partial_info = partial.with_name(partial.name + ".json")
    if partial.exists():
        try:
            same = json.loads(partial_info.read_text(encoding="utf-8")) == expected
        except (OSError, ValueError):
            same = False
        if not same or (size is not None and partial.stat().st_size > size):
            partial.unlink()
    atomic_json(partial_info, expected)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            # A complete interrupted download only needs verification.
            if size is None or offset != size:
                headers = {
                    "User-Agent": "TAS-reproducible-assets/1",
                    "Accept-Encoding": "identity",
                }
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = response.status
                    if status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+|\*)", content_range
                        )
                        if match is None or int(match.group(1)) != offset:
                            raise ArtifactError(
                                "Server returned an incompatible Content-Range"
                            )
                    elif status == 200:
                        offset = 0
                    else:
                        raise ArtifactError(f"Unexpected download HTTP status {status}")
                    received = 0
                    with partial.open("ab" if offset else "wb") as handle:
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            handle.write(chunk)
                            received += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and received != int(content_length):
                        raise ArtifactError(
                            "Interrupted transfer: Content-Length does not match received bytes"
                        )
            valid, digest = validate(partial)
            if not valid:
                # A truncated transfer can still be resumed. Wrong full content
                # must be removed, otherwise Range would keep retrying its tail.
                if size is None or partial.stat().st_size >= size:
                    partial.unlink(missing_ok=True)
                raise ArtifactError(f"Download checksum/size mismatch for {path.name}")
            os.replace(partial, path)
            publish_receipt(digest)
            partial_info.unlink(missing_ok=True)
            if progress:
                progress(f"Verified {path.name} ({path.stat().st_size:,} bytes)")
            return path
        except (OSError, ValueError, ArtifactError, http.client.HTTPException) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                # Unknown-length assets may already have a complete .part file.
                # Restart once rather than accepting content without a hash.
                partial.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise ArtifactError(
        f"Download failed after {retries} attempts: {url}: {last_error}"
    ) from last_error


def safe_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    members: Iterable[str] | None = None,
    max_bytes: int = 12 * 1024**3,
    max_members: int = 100_000,
) -> list[Path]:
    """Extract regular files with traversal, symlink, size and CRC checks."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    selected = set(members) if members is not None else None
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if len(infos) > max_members:
            raise ArtifactError("Archive has too many members")
        matched = set()
        total = 0
        seen = set()
        validated = []
        for info in infos:
            relative = _safe_relative(info.filename.rstrip("/"))
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ArtifactError(
                    f"Archive contains a link/special file: {info.filename}"
                )
            if info.filename in seen:
                raise ArtifactError(f"Duplicate archive member: {info.filename}")
            seen.add(info.filename)
            target = destination / relative
            if not target.resolve().is_relative_to(root):
                raise ArtifactError(f"Archive escapes destination: {info.filename}")
            if selected is None or info.filename in selected:
                matched.add(info.filename)
                total += info.file_size
                if total > max_bytes:
                    raise ArtifactError("Archive exceeds allowed uncompressed size")
                validated.append((info, target))
        if selected is not None and selected - matched:
            raise ArtifactError(
                f"Archive lacks required members: {sorted(selected - matched)}"
            )
        for info, target in validated:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".extract-", dir=target.parent
            )
            try:
                with (
                    os.fdopen(descriptor, "wb") as output,
                    source.open(info) as input_file,
                ):
                    shutil.copyfileobj(input_file, output, 4 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            extracted.append(target)
    return extracted


def prepare_assets(
    root: str | Path,
    names: Iterable[str] | None = None,
    *,
    progress: Callable[[str], None] | None = print,
    verify_cache: bool = False,
) -> dict[str, Any]:
    """Download selected pinned models; return paths plus portable provenance.

    The returned ``identity`` hashes relative paths and checksums, never the
    machine-specific cache root. ``paths`` are convenience absolute paths which
    a caller can rebase after restoring the directory to a Colab VM.
    """
    root = Path(root).expanduser().resolve()
    names = list(MODEL_SPECS) if names is None else list(names)
    if len(set(names)) != len(names) or any(name not in MODEL_SPECS for name in names):
        raise ValueError(f"Unknown/duplicate model selection: {names}")
    root.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for name in names:
        spec = MODEL_SPECS[name]
        relative = Path("models") / name / spec["revision"]
        directory = root / relative
        for file in spec["files"]:
            filename = _safe_relative(file["filename"])
            download_verified(
                file["url"],
                directory / filename,
                sha256=file["sha256"],
                size=file["size"],
                progress=progress,
                verify_cache=verify_cache,
            )
        model_path = (
            relative if "repo_id" in spec else relative / spec["files"][0]["filename"]
        )
        assets[name] = {**spec, "relative_path": model_path.as_posix()}
        paths[name] = str(root / model_path)
    portable = {"schema": "tas-model-assets-v1", "assets": assets}
    manifest = {**portable, "identity": canonical_hash(portable), "paths": paths}
    atomic_json(root / "assets-manifest.json", portable)
    return manifest


def asset_paths(
    root: str | Path, manifest: dict[str, Any] | None = None
) -> dict[str, str]:
    """Rebase a saved portable manifest after CPU→GPU restoration."""
    root = Path(root).expanduser().resolve()
    if manifest is None:
        manifest = json.loads(
            (root / "assets-manifest.json").read_text(encoding="utf-8")
        )
    return {
        name: str(root / _safe_relative(spec["relative_path"]))
        for name, spec in manifest["assets"].items()
    }
