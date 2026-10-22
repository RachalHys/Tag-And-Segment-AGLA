import os
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_failed_child_error_is_visible_and_persisted(tmp_path, monkeypatch, capsys):
    import subprocess

    output = (
        "[preflight] import torchvision\n"
        "Traceback (most recent call last):\n"
        "ImportError: fixture shared library is missing\n"
    )

    def fail(command, **kwargs):
        assert command[1] == "-u"
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        return SimpleNamespace(returncode=1, stdout=output)

    monkeypatch.setattr("tas_runtime.bootstrap.subprocess.run", fail)
    log = tmp_path / "drive/diagnostics/environment/preflight.log"
    with pytest.raises(
        RuntimeError, match="fixture shared library is missing"
    ) as error:
        validate_environment("/managed/bin/python", "gpu", log_path=log)
    assert str(log) in str(error.value)
    assert log.read_text() == output
    assert output in capsys.readouterr().out


def test_cpu_preflight_handles_inherited_colab_backend(tmp_path, monkeypatch):
    if sys.version_info[:2] != (3, 10):
        pytest.skip("The declared Colab interpreter is Python3.10.")
    inline_backend = "module://matplotlib_inline.backend_inline"
    monkeypatch.setenv("MPLBACKEND", inline_backend)
    log = tmp_path / "preflight.log"
    report = validate_environment(sys.executable, "cpu", log_path=log)
    output = log.read_text()
    assert sys.executable in output
    assert "[preflight] import torch" in output
    assert "[preflight] import torchvision" in output
    assert "[preflight] import matplotlib.pyplot" in output
    assert "[preflight] passed" in output
    assert report["packages"]["numpy"] == "1.26.4"
    assert report["matplotlib_backend"].lower() == "agg"
    assert os.environ["MPLBACKEND"] == inline_backend


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
