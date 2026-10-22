"""Package committed experiment modules over the fixed public baseline."""

from __future__ import annotations

import base64
import hashlib
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

BASE_COMMIT = "624f74501bdafe9b7d6a05ae739d12b422ce8737"
SOURCE_PATHS = (
    "AGENTS.md",
    "docs/TAS_V7_COLAB.md",
    "tas_runtime",
    "experiments",
    "requirements",
    "tests",
    "tools",
)


def package_from_commit(root, commit):
    """Deterministic transport archive, never a second editable implementation."""
    root = Path(root)
    resolved = subprocess.check_output(
        ["git", "rev-parse", commit + "^{commit}"], cwd=root, text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", resolved, *SOURCE_PATHS], cwd=root
    )
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Experiment source must contain regular files only.")
            contents[member.name] = source.extractfile(member).read()
    return _package_files(contents, resolved)


def _package_files(contents, source_commit):
    files = {}
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as destination:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            destination.writestr(info, data, compresslevel=9)
            files[name] = hashlib.sha256(data).hexdigest()
    data = output.getvalue()
    return {
        "schema": "tas-private-source-v1",
        "base_commit": BASE_COMMIT,
        "source_commit": source_commit,
        "sha256": hashlib.sha256(data).hexdigest(),
        "files": files,
        "archive_b85": base64.b85encode(data).decode("ascii"),
    }


def restore_source_package(package, checkout, persistent, repo_url):
    """Notebook bootstrap; stdlib only, before installing scientific packages.

    This function is copied by the notebook generator so upload of the notebook
    alone is sufficient. Its authoritative source remains this Python module.
    Existing baseline files cannot be replaced by the source package.
    """
    import base64
    import hashlib
    import io
    import json
    import os
    import re
    import stat
    import subprocess
    import zipfile
    from pathlib import Path

    if package.get("schema") != "tas-private-source-v1" or any(
        re.fullmatch(r"[a-f0-9]{40}", package.get(key, "")) is None
        for key in ("base_commit", "source_commit")
    ):
        raise ValueError("Invalid versioned source package.")
    data = base64.b85decode(package["archive_b85"])
    if hashlib.sha256(data).hexdigest() != package["sha256"]:
        raise ValueError("Source archive checksum mismatch.")
    checkout, persistent = Path(checkout), Path(persistent)
    base = package["base_commit"]

    def owned_path(name):
        relative = Path(name)
        allowed = len(relative.parts) > 1 and relative.parts[0] in {
            "tas_runtime",
            "experiments",
            "requirements",
            "tests",
            "tools",
        }
        allowed |= name in {"AGENTS.md", "docs/TAS_V7_COLAB.md"}
        return (
            allowed
            and not relative.is_absolute()
            and ".." not in relative.parts
            and (checkout / relative).resolve().is_relative_to(checkout.resolve())
        )

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != len(package["files"]) or {
            item.filename for item in members
        } != set(package["files"]):
            raise ValueError("Source file manifest mismatch.")
        if sum(item.file_size for item in members) > 8 * 1024**2:
            raise ValueError("Source package exceeds its code-only size limit.")
        contents = {}
        for item in members:
            if not owned_path(item.filename) or stat.S_IFMT(
                item.external_attr >> 16
            ) not in {0, stat.S_IFREG}:
                raise ValueError("Unsafe source member.")
            content = archive.read(item)
            if hashlib.sha256(content).hexdigest() != package["files"][item.filename]:
                raise ValueError("Source member checksum mismatch.")
            contents[item.filename] = content
    if not (checkout / ".git").exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", str(repo_url), str(checkout)], check=True
        )
    found = subprocess.run(
        ["git", "cat-file", "-e", base + "^{commit}"], cwd=checkout, capture_output=True
    )
    if found.returncode:
        subprocess.run(["git", "fetch", "origin", base], cwd=checkout, check=True)
    for name in contents:
        baseline = subprocess.run(
            ["git", "cat-file", "-e", base + ":" + name],
            cwd=checkout,
            capture_output=True,
        )
        if baseline.returncode == 0:
            raise ValueError(
                "Source package cannot overwrite a public baseline file: " + name
            )
    subprocess.run(["git", "checkout", "--detach", base], cwd=checkout, check=True)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if actual != base:
        raise ValueError("Public baseline checkout mismatch.")
    previous = checkout / ".tas-source.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        for name in set(old.get("files", {})) - set(contents):
            baseline = subprocess.run(
                ["git", "cat-file", "-e", base + ":" + name],
                cwd=checkout,
                capture_output=True,
            )
            if not owned_path(name) or baseline.returncode == 0:
                raise ValueError("Unsafe obsolete source path.")
            (checkout / name).unlink(missing_ok=True)

    def save(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".installing")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    for name, content in contents.items():
        save(checkout / name, content)
    receipt = {key: value for key, value in package.items() if key != "archive_b85"}
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    save(checkout / ".tas-source.json", receipt_bytes)
    stored = (
        persistent
        / "code"
        / ("source-" + package["source_commit"] + "-" + package["sha256"][:16])
    )
    if (
        not stored.with_suffix(".zip").exists()
        or hashlib.sha256(stored.with_suffix(".zip").read_bytes()).hexdigest()
        != package["sha256"]
    ):
        save(stored.with_suffix(".zip"), data)
    save(stored.with_suffix(".json"), receipt_bytes)
    return receipt
