import ast
import nbformat
from tools.build_colab_notebook import build


def test_notebook_schema_all_cells_compile_and_no_model_code_duplication():
    value = build("a" * 40)
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
    assert "PRIVATE_BUNDLE_PATH" in joined
    assert 'run_phase("check")' in joined
    assert 'run_phase("run")' in joined


def test_generated_settings_are_plain_python_without_colab_import():
    value = build("b" * 40)
    parameters = next(
        cell["source"] for cell in value["cells"] if cell["cell_type"] == "code"
    )
    namespace = {}
    exec(compile(parameters, "parameters", "exec"), namespace)
    assert namespace["REPO_COMMIT"] == "b" * 40
    assert namespace["PHASE"] == "auto"
    assert namespace["STAGE"] == "flow"
    assert namespace["SEEDS"] == [0, 1, 2]
    assert namespace["RUN_FINAL_REMAINDER"] is False
