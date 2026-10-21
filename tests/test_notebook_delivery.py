import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from tools.notebook_delivery import (
    BASE_COMMIT,
    _package_files,
    package_from_commit,
    restore_source_package,
)


@pytest.fixture
def source_package():
    root = Path(__file__).resolve().parents[1]
    receipt = root / ".tas-source.json"
    if receipt.exists():
        source = json.loads(receipt.read_text())

        def make():
            contents = {name: (root / name).read_bytes() for name in source["files"]}
            return _package_files(contents, source["source_commit"])

        return make(), root, make
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    def make():
        return package_from_commit(root, commit)

    return make(), root, make


def test_committed_source_package_is_deterministic_and_restores_without_uploads(
    source_package, tmp_path
):
    package, root, make = source_package
    assert package == make()
    checkout, persistent = tmp_path / "checkout", tmp_path / "drive"
    receipt = restore_source_package(package, checkout, persistent, str(root))
    assert receipt["source_commit"] == package["source_commit"]
    assert (checkout / "experiments/tas_v7/study.py").is_file()
    assert not (checkout / "tas_idea.md").exists()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == BASE_COMMIT
    )
    assert (
        subprocess.check_output(["git", "diff", "--name-only"], cwd=checkout, text=True)
        == ""
    )
    assert len(list((persistent / "code").glob("*.zip"))) == 1
    target = checkout / "tas_runtime/config.py"
    target.write_text("interrupted or changed local source")
    old_receipt = json.loads((checkout / ".tas-source.json").read_text())
    old_receipt["files"]["experiments/obsolete.py"] = "prior idea"
    (checkout / "experiments/obsolete.py").write_text("obsolete private source")
    (checkout / ".tas-source.json").write_text(json.dumps(old_receipt))
    assert restore_source_package(package, checkout, persistent, str(root)) == receipt
    assert not (checkout / "experiments/obsolete.py").exists()
    assert (
        hashlib.sha256(target.read_bytes()).hexdigest()
        == package["files"]["tas_runtime/config.py"]
    )


def test_corrupt_source_package_fails_before_clone_or_model_setup(
    source_package, tmp_path
):
    package, root, _ = source_package
    archive = package["archive_b85"]
    changed = ("1" if archive[0] != "1" else "2") + archive[1:]
    with pytest.raises(ValueError, match="checksum"):
        restore_source_package(
            {**package, "archive_b85": changed},
            tmp_path / "checkout",
            tmp_path / "drive",
            str(root),
        )
    assert not (tmp_path / "checkout").exists()


@pytest.mark.parametrize(
    "filename",
    ["../escape.py", "/outside.py", "README.md", "tas_runtime/../../escape.py"],
)
def test_source_package_rejects_traversal_and_baseline_paths(
    source_package, tmp_path, filename
):
    package, root, _ = source_package
    output = io.BytesIO()
    data = b"source fixture"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(filename, data)
    encoded = output.getvalue()
    changed = {
        **package,
        "files": {filename: hashlib.sha256(data).hexdigest()},
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "archive_b85": base64.b85encode(encoded).decode(),
    }
    with pytest.raises(ValueError, match="Unsafe"):
        restore_source_package(
            changed, tmp_path / "checkout", tmp_path / "drive", str(root)
        )
    assert not (tmp_path / "checkout").exists()
