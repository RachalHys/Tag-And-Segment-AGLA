import ast
import nbformat
import pytest
from tools.build_colab_notebook import build
from tools.notebook_delivery import BASE_COMMIT


def test_builder_resolves_git_ref_to_fixed_commit(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import subprocess
    from tools.build_colab_notebook import main

    root = Path(__file__).resolve().parents[1]
    # Archive transport is tested separately. Keep this check focused on real
    # Git-ref resolution, including in the restored baseline checkout.
    monkeypatch.setattr(
        "tools.build_colab_notebook.package_from_commit",
        lambda root, commit: fixture_package(commit),
    )
    destination = tmp_path / "notebook.ipynb"
    monkeypatch.setattr(
        "sys.argv", ["builder", "--commit", "HEAD", "--output", str(destination)]
    )
    main()
    value = json.loads(destination.read_text())
    settings = next(
        cell["source"] for cell in value["cells"] if cell["cell_type"] == "code"
    )
    namespace = {}
    exec(settings, namespace)
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert namespace["RESEARCH_COMMIT"] == expected


def fixture_package(commit):
    return {
        "schema": "tas-private-source-v1",
        "base_commit": BASE_COMMIT,
        "source_commit": commit,
        "files": {},
        "sha256": "",
        "archive_b85": "",
    }


def notebook_configuration(notebook, namespace):
    """Resolve the actual JSON config assignment without installing Colab assets."""
    cell = next(cell["source"] for cell in notebook["cells"] if "configuration = {" in cell["source"])
    assignment = next(
        node for node in ast.walk(ast.parse(cell))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "configuration" for target in node.targets)
    )
    scope = dict(namespace, actual_commit=namespace["RESEARCH_COMMIT"])
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "configuration", "exec"), scope)
    return scope["configuration"]


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
    assert namespace["METHODS"] == []
    assert namespace["MAX_SESSION_MINUTES"] == 180
    assert namespace["SEEDS"] == [0, 1, 2]
    assert namespace["RUN_FINAL_REMAINDER"] is False
    assert namespace["AUTO_SHUTDOWN"] is True
    assert namespace["REQUIRE_A100_80GB"] is False
    assert namespace["EXECUTION_PROTOCOL"] == "tas-batched-study-v1"
    assert namespace["BATCH_SIZES"] == [4]
    assert namespace["EXECUTION"] == "packed"
    assert namespace["ATTENTION_BACKENDS"] == ["sdpa"]
    assert namespace["PILOT_IMAGES"] == 4
    assert namespace["COVERAGE_TOLERANCE_PP"] == 2.0
    from experiments.tas_v7.cli import plan
    from tas_runtime.config import StudyConfig

    config = StudyConfig.from_dict(notebook_configuration(value, namespace))
    resolved = plan(config)
    assert config.execution_protocol == "tas-batched-study-v1"
    assert config.batch_sizes == (4,)
    assert config.execution == "packed"
    assert config.attention_backends == ("sdpa",)
    assert config.pilot_images == 4
    assert config.last_position_logits is True
    assert config.generation.max_new_tokens == 180
    assert config.generation.precision == "fp16"
    assert config.run_confirmation and config.run_controls and config.run_diagnostics
    assert resolved["initial_methods"] == 14
    assert resolved["smoke"] == 420
    assert resolved["budget_minutes_per_session"] == 180


def test_full_notebook_can_explicitly_select_strict_before_outputs():
    notebook = build("b" * 40, source_package=fixture_package("b" * 40))
    parameters = next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    parameters = parameters.replace("EXECUTION_PROTOCOL = 'tas-batched-study-v1'", "EXECUTION_PROTOCOL = 'strict'")
    namespace = {}
    exec(parameters, namespace)
    from tas_runtime.config import StudyConfig

    config = StudyConfig.from_dict(notebook_configuration(notebook, namespace))
    assert config.execution_protocol == "strict"
    assert config.batch_sizes == (1, 2, 4)
    assert config.execution == "auto"
    assert config.attention_backends == ("eager", "sdpa")
    assert config.pilot_images == 8


def test_performance_notebook_stops_after_audit_and_keeps_flow_available():
    notebook = build(
        "b" * 40, source_package=fixture_package("b" * 40), stage="audit"
    )
    namespace = {}
    parameters = next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    exec(compile(parameters, "parameters", "exec"), namespace)
    assert namespace["STAGE"] == "audit"
    assert namespace["EXECUTION_PROTOCOL"] == "strict"
    assert namespace["BATCH_SIZES"] == [1, 2, 4]
    assert namespace["AUTO_SHUTDOWN"] is True
    assert namespace["EXECUTION"] == "auto"
    assert namespace["PILOT_IMAGES"] == 8
    assert any("does not start the 420-caption" in cell["source"] for cell in notebook["cells"])


def test_separate_batch_comparison_notebook_is_bounded_and_displays_paired_reports():
    notebook = build("b" * 40, source_package=fixture_package("b" * 40), stage="batch_compare")
    nbformat.validate(nbformat.from_dict(notebook))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(cell["source"], filename=f"comparison-cell-{index}")
    namespace = {}
    exec(next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"), namespace)
    assert namespace["STAGE"] == "batch_compare"
    assert namespace["COMPARISON_IMAGES"] == 4
    assert namespace["METHODS"] == ["E0", "E1", "F4"]
    assert namespace["MAX_SESSION_MINUTES"] == 20
    assert namespace["AUTO_SHUTDOWN"] is True
    assert namespace["EXECUTION_PROTOCOL"] == "strict"
    assert namespace["BATCH_SIZES"] == [1, 2, 4]
    assert namespace["EXECUTION"] == "auto"
    assert namespace["PILOT_IMAGES"] == 8
    assert any("24 captions total" in cell["source"] for cell in notebook["cells"])
    assert any("comparison.json" in cell["source"] for cell in notebook["cells"])
    from experiments.tas_v7.cli import plan
    from tas_runtime.config import StudyConfig

    config = StudyConfig.from_dict(notebook_configuration(notebook, namespace))
    resolved = plan(config)
    assert resolved["complete_captions"] == 24
    assert resolved["budget_minutes_per_session"] == 20
    assert config.generation.max_new_tokens == 180


def test_report_renders_old_audits_and_includes_newest_run(tmp_path, monkeypatch):
    import json
    import os
    import sys
    from types import ModuleType

    ipython, displays = ModuleType("IPython"), ModuleType("IPython.display")
    displays.Image = lambda **kwargs: kwargs
    ipython.display = displays
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", displays)
    for index, name in enumerate(("ff", "ee", "dd", "aa")):
        directory = tmp_path / "runs" / name
        (directory / "audit").mkdir(parents=True)
        config = directory / "resolved-config.json"
        config.write_text("{}")
        os.utime(config, (1000 + index, 1000 + index))
        (directory / "audit/audit.json").write_text(json.dumps({"pilot": [{
            "backend": "eager", "execution": "serial", "batch_images": 1,
            "status": "rejected",  # older reports have no head/memory/rate fields
        }]}))
    (tmp_path / "runs/aa/audit/audit.json").write_text(json.dumps({
        "execution_protocol": "tas-batched-study-v1",
        "selected_policy": {"backend": "sdpa", "execution": "packed", "batch_images": 4, "last_position_logits": True},
        "pilot": [{
            "backend": "sdpa", "execution": "packed", "batch_images": 4, "status": "passed",
            "method_checks": {"F4": {"protocol": "HARMONIZED", "greedy_match": False, "sample_match": True, "argmax_changes": 1, "support_token_changes": 0}},
        }],
    }))
    comparison = tmp_path / "runs/aa/batch-comparison-fixture"
    comparison.mkdir()
    (comparison / "comparison.json").write_text(json.dumps({
        "reference_policy": {"batch_images": 1}, "candidate_policy": {"batch_images": 4},
        "summary": [{"method": "F4", "generation_mode": "sample", "n_pairs": 12,
                     "exact_token_match_rate": 0.5, "speedup": 2.0,
                     "CHAIRi_delta": 0.0, "CHAIRs_delta": 0.0,
                     "category_coverage_delta": 0.0, "length_words_delta": 0.1}],
        "interpretation": "descriptive only",
    }))
    (comparison / "changed-captions.jsonl").write_text(json.dumps({
        "image_id": 1, "method": "F4", "reference_caption": "Original caption",
        "candidate_caption": "Batched caption",
    }) + "\n")
    diagnostics = tmp_path / "runs/aa/diagnostics"
    diagnostics.mkdir()
    (diagnostics / "paired-interventions.json").write_text(json.dumps({
        "scope": "provisional; no causal attribution", "comparisons": [{
            "intervention": "D-MASK", "method": "E1", "applied_image_ids": [1],
            "mask_changed_image_ids": [1], "paired": [
                {"metric": "CHAIRi", "delta_candidate_minus_baseline": -0.01, "ci_low": -0.02, "ci_high": 0.0},
                {"metric": "category_coverage", "status": "undefined"},
            ],
        }],
    }))
    notebook = build("d" * 40, source_package=fixture_package("d" * 40))
    cell = next(cell["source"] for cell in notebook["cells"] if "runs_root =" in cell["source"])
    output = []
    exec(compile(cell, "reports", "exec"), {
        "persistent": tmp_path, "json": json, "display": output.append,
        "Markdown": str, "HTML": str,
    })
    headings = [value for value in output if isinstance(value, str) and value.startswith("### Run")]
    assert headings == ["### Run `aa`", "### Run `dd`", "### Run `ee`"]
    assert any("**Execution protocol:** `tas-batched-study-v1`" in str(value) for value in output)
    assert any("**Execution protocol:** `strict`" in str(value) for value in output)
    assert any("Greedy match" in str(value) and "HARMONIZED" in str(value) for value in output)
    assert any("Paired execution comparison" in str(value) for value in output)
    assert any("Batched caption" in str(value) for value in output)
    assert any("Paired perception interventions" in str(value) for value in output)
    assert any("-1.00 [-2.00, +0.00]" in str(value) for value in output)


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


def test_child_output_is_forwarded_before_exit_without_waiting_for_newline(tmp_path):
    from contextlib import redirect_stdout
    import io
    import sys
    from tools.build_colab_notebook import run_logged

    marker = tmp_path / "release"

    class NotebookOutput(io.StringIO):
        def write(self, value):
            result = super().write(value)
            if "child-ready" in self.getvalue():
                marker.touch()
            return result

    script = """
import os, pathlib, sys, time
os.write(1,b'child-ready')
os.write(2,b' child-stderr\\n')
deadline=time.monotonic()+5
while not pathlib.Path(sys.argv[1]).exists():
 if time.monotonic()>deadline: raise RuntimeError('output was not forwarded live')
 time.sleep(.01)
print('child-finished: ✓')
"""
    output = NotebookOutput()
    with redirect_stdout(output):
        run_logged([sys.executable, "-u", "-c", script, marker])
    text = output.getvalue()
    assert "child-stderr" in text
    assert "child-finished: ✓" in text
    assert "[process] finished" in text


def test_quiet_child_shows_heartbeat_before_completion(tmp_path):
    from contextlib import redirect_stdout
    import io
    import sys
    from tools.build_colab_notebook import run_logged

    marker = tmp_path / "release"

    class NotebookOutput(io.StringIO):
        def write(self, value):
            result = super().write(value)
            if "still running" in self.getvalue():
                marker.touch()
            return result

    script = """
import pathlib, sys, time
deadline=time.monotonic()+5
while not pathlib.Path(sys.argv[1]).exists():
 if time.monotonic()>deadline: raise RuntimeError('heartbeat was not shown')
 time.sleep(.01)
"""
    output = NotebookOutput()
    with redirect_stdout(output):
        run_logged(
            [sys.executable, "-u", "-c", script, marker],
            label="run", heartbeat_seconds=.02,
        )
    assert "[run] still running; elapsed" in output.getvalue()


def test_streamed_child_error_remains_visible_and_propagates(capsys):
    import subprocess
    import sys
    from tools.build_colab_notebook import run_logged

    script = "import sys; print('fixture error',file=sys.stderr); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_logged([sys.executable, "-u", "-c", script])
    assert error.value.returncode == 3
    assert "fixture error" in capsys.readouterr().out


def test_interrupted_output_forwarding_reaps_child(tmp_path):
    from contextlib import redirect_stdout
    import io
    import os
    import sys
    from tools.build_colab_notebook import run_logged

    pid_file = tmp_path / "pid"

    class InterruptedNotebook(io.StringIO):
        def write(self, value):
            if "child-ready" in value:
                raise KeyboardInterrupt
            return super().write(value)

    script = """
import os, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
try:
    print('child-ready',flush=True)
    time.sleep(30)
finally:
    pathlib.Path(sys.argv[1] + '.flushed').write_text('saved')
"""
    with redirect_stdout(InterruptedNotebook()), pytest.raises(KeyboardInterrupt):
        run_logged([sys.executable, "-u", "-c", script, pid_file])
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert pid_file.with_name("pid.flushed").read_text() == "saved"
