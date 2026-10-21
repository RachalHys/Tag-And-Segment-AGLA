import ast
import nbformat
import pytest
from tools.build_colab_notebook import build
from tools.notebook_delivery import BASE_COMMIT


def fixture_package(commit):
    return {
        "schema": "tas-private-source-v1",
        "base_commit": BASE_COMMIT,
        "source_commit": commit,
        "files": {},
        "sha256": "",
        "archive_b85": "",
    }


def test_notebook_schema_all_cells_compile_and_no_model_code_duplication():
    value = build("a" * 40, source_package=fixture_package("a" * 40))
    notebook = nbformat.from_dict(value)
    nbformat.validate(notebook)
    sources = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"colab-cell-{index}")
            sources.append(cell.source)
    joined = "\n".join(sources)
    assert "class Llava" not in joined
    assert "def fuse_logits" not in joined
    assert 'git", "checkout", "--detach"' in joined
    assert 'PYTHONHASHSEED="0"' in joined
    assert "ensure_environment" in joined
    assert "restore_source_package(SOURCE_PACKAGE" in joined
    assert "PRIVATE_BUNDLE_PATH" not in joined
    assert 'run_phase("check")' in joined
    assert 'run_phase("run")' in joined


def test_generated_settings_are_plain_python_without_colab_import():
    value = build("b" * 40, source_package=fixture_package("b" * 40))
    parameters = next(
        cell["source"] for cell in value["cells"] if cell["cell_type"] == "code"
    )
    namespace = {}
    exec(compile(parameters, "parameters", "exec"), namespace)
    assert namespace["RESEARCH_COMMIT"] == "b" * 40
    assert namespace["BASELINE_COMMIT"] == BASE_COMMIT
    assert namespace["PHASE"] == "auto"
    assert namespace["STAGE"] == "flow"
    assert namespace["SEEDS"] == [0, 1, 2]
    assert namespace["RUN_FINAL_REMAINDER"] is False
    assert namespace["AUTO_SHUTDOWN"] is True
    assert namespace["COVERAGE_TOLERANCE_PP"] == 2.0


@pytest.fixture
def shutdown_notebook(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    notebook = build("c" * 40, source_package=fixture_package("c" * 40))
    calls = []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(flush_and_unmount=lambda: calls.append("flush"))
    colab.runtime = SimpleNamespace(unassign=lambda: calls.append("release"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {}
    parameters = next(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    exec(compile(parameters, "parameters", "exec"), namespace)
    return notebook, namespace, calls


def test_shutdown_flushes_drive_before_releasing_runtime(shutdown_notebook):
    notebook, namespace, calls = shutdown_notebook
    cell = notebook["cells"][-1]["source"]
    exec(compile(cell, "shutdown-cell", "exec"), namespace)
    assert calls == ["flush", "release"]
    calls.clear()
    namespace["AUTO_SHUTDOWN"] = False
    exec(compile(cell, "shutdown-disabled", "exec"), namespace)
    assert calls == []


@pytest.mark.parametrize("enabled", [True, False])
def test_failed_preflight_releases_runtime_and_preserves_failure(
    shutdown_notebook, enabled
):
    import time

    notebook, namespace, calls = shutdown_notebook
    cell = next(
        cell["source"]
        for cell in notebook["cells"]
        if "phase_started =" in cell["source"]
    )

    def fail(phase):
        calls.append(phase)
        raise RuntimeError("preflight fixture failed")

    namespace.update(
        AUTO_SHUTDOWN=enabled, effective_phase="run", run_phase=fail, time=time
    )
    with pytest.raises(RuntimeError, match="preflight fixture failed"):
        exec(compile(cell, "failed-preflight-cell", "exec"), namespace)
    assert calls == (["check", "flush", "release"] if enabled else ["check"])
