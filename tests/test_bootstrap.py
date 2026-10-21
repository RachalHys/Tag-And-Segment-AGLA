from pathlib import Path
import sys

import pytest
from tas_runtime.bootstrap import profile_identity, validate_environment


def test_environment_identity_distinguishes_cpu_cuda_and_source_profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cpu = profile_identity("cpu", root, False)
    gpu = profile_identity("gpu", root, True)
    assert cpu != gpu
    assert cpu["python"] == "3.10.16"
    assert gpu["torch"] == "2.4.1"
    assert len(gpu["requirements"]) == 2
    with pytest.raises(ValueError):
        profile_identity("tpu", root, False)


def test_actual_cpu_profile_imports():
    if sys.version_info[:2] != (3, 10):
        pytest.skip("The declared Colab interpreter is Python3.10.")
    report = validate_environment(sys.executable, "cpu")
    assert report["packages"]["transformers"] == "4.35.2"
    assert report["torch"].split("+")[0] == "2.4.1"


def test_prepared_nlp_is_reused_offline_and_detects_resource_drift(
    tmp_path, monkeypatch
):
    import os
    import shutil
    import nltk
    from tas_runtime.artifacts import atomic_json, sha256_file
    from tas_runtime.bootstrap import prepare_nlp
    from tas_runtime.persistence import restore_nlp

    source = Path(os.environ.get("NLTK_DATA", "/nonexistent"))
    monkeypatch.setenv("NLTK_DATA", str(source))
    if not (source / "tokenizers/punkt.zip").is_file():
        pytest.skip("Preparation-owned NLTK fixtures are unavailable")
    persistent = tmp_path / "drive"
    (persistent / "tokenizers").mkdir(parents=True)
    hashes = {}
    for name in ("punkt", "punkt_tab"):
        relative = f"tokenizers/{name}.zip"
        shutil.copyfile(source / relative, persistent / relative)
        hashes[relative] = sha256_file(persistent / relative)
    receipt = {
        "schema": "tas-nlp-v1",
        "nltk": nltk.__version__,
        "resources": ["punkt", "punkt_tab"],
        "sha256": hashes,
    }
    atomic_json(persistent / "resources.json", receipt)
    local = tmp_path / "vm/nlp"
    restore_nlp(persistent, local)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Verified resources must not require network access on rerun"
        )

    monkeypatch.setattr(nltk, "download", forbidden)
    monkeypatch.setattr(nltk.data, "path", nltk.data.path[:])
    assert prepare_nlp(local) == receipt
    (local / "tokenizers/punkt_tab.zip").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen NLP resource"):
        prepare_nlp(local)
