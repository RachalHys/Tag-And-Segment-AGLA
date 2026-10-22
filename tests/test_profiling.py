import json

import pytest
import torch

from tas_runtime.profiling import profile_call
from tas_runtime.progress import progress, step


def test_cpu_operator_profile_calls_once_and_persists_small_summary(tmp_path):
    calls = []

    def work():
        calls.append(1)
        return torch.ones(4, 4) @ torch.ones(4, 4)

    result = profile_call(work, tmp_path, device="cpu")
    assert calls == [1]
    assert result["status"] == "complete"
    assert "aten::mm" in (tmp_path / "operators.txt").read_text()
    assert json.loads((tmp_path / "profile.json").read_text()) == result
    assert not list(tmp_path.glob("*.pt.trace.json"))


def test_unavailable_profiler_is_reported_without_claiming_measurement(tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("CUPTI unavailable")

    monkeypatch.setattr(torch.profiler, "profile", unavailable)
    assert profile_call(lambda: None, tmp_path, device="cpu") == {
        "status": "unavailable", "error": "CUPTI unavailable",
    }


def test_stage_names_and_bar_are_visible_on_redirected_stdout(capsys):
    with step("Test stage"):
        with progress(total=2, desc="Test images", unit="image") as bar:
            bar.update(2)
    output = capsys.readouterr().out
    assert "[stage] Test stage" in output
    assert "2/2" in output and "100%" in output
    with pytest.raises(ValueError, match="failure"), step("Failed stage"):
        raise ValueError("failure")
    output = capsys.readouterr().out
    assert "Failed stage: stopped after" in output
    assert "Failed stage: done" not in output
