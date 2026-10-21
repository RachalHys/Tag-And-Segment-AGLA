"""Generate the repository-backed notebook; source remains in Python modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import textwrap


def build(commit, repo_url="https://github.com/RachalHys/Tag-And-Segment-AGLA.git"):
    cells = []

    def markdown(source):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n",
            }
        )

    def code(source):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": textwrap.dedent(source).strip() + "\n",
            }
        )

    markdown("""
    # TAS v7 — CPU preparation / A100 experiments

    Upload this notebook to Colab. Use **CPU → Run all** first, then select your
    **A100 80 GB → Run all**. `PHASE="auto"` selects preparation on CPU and the
    study on GPU. Models/data/caches/results live in the same persistent Drive
    folder across sessions and future notebooks. CPU/GPU environments are
    recreated locally; the notebook kernel is left intact.

    The notebook imports versioned Python from the experiment branch. For
    unpublished/private work, put the supplied `tas-v7-source.bundle` in the
    Drive path below; it replaces the GitHub transport without changing code.
    Otherwise make this exact experiment commit available at `REPO_URL` first.
    No research code or results are uploaded to GitHub by this notebook.

    Automatic flow: blocking P0 audit → 30-image smoke → 240-image screen →
    predeclared shortlist → confirmation (3 seeds + greedy) → controls → P0-D.
    This implements the owner's explicit override of the earlier smoke review
    pause. Final remainder and P3/P4/P5 remain disabled by default. A session
    budget saves progress; the next Run all continues completed stages.

    Full A100 weights have not been tested by the local CPU suite. P0 rejects
    baseline mismatches before bulk generation. No scientific gains are assumed.
    """)
    code(f"""
    # Edit these settings once. Reuse PERSISTENT_ROOT in future idea notebooks.
    REPO_URL = {repo_url!r}
    REPO_BRANCH = "experiment/tas-v7-colab"
    REPO_COMMIT = {commit!r}
    PRIVATE_BUNDLE_PATH = "/content/drive/MyDrive/TAS-Shared/code/tas-v7-source.bundle"
    PERSISTENT_ROOT = "/content/drive/MyDrive/TAS-Shared"
    LOCAL_ROOT = "/content/tas-v7-work"
    PHASE = "auto"             # auto / prepare / run / score / check
    STAGE = "flow"             # flow / audit / smoke / screen / confirm / controls / diagnostics / final / external
    MAX_SESSION_MINUTES = 180  # planned stop; run again to resume without changing the experiment
    SEEDS = [0, 1, 2]
    REQUIRE_A100_80GB = True
    RUN_CONFIRMATION = True
    RUN_CONTROLS = True
    RUN_DIAGNOSTICS = True
    RUN_FINAL_REMAINDER = False
    RESERVE_POPE_IMAGES = True
    DATASET = "coco"           # AMBER/POPE require STAGE=external and explicit verified manifests below
    EXTERNAL_MANIFEST = None
    EXTERNAL_IMAGE_ROOT = None
    METHODS = []               # empty = exact initial 14; external runs should select frozen finalists explicitly
    BATCH_SIZES = [1, 2, 4, 8]  # measured once, correctness checked, then policy locked
    ATTENTION_BACKENDS = ["eager", "sdpa"]
    LAST_POSITION_LOGITS = True
    SPLIT_SEED = 20261021
    """)
    markdown("""
    ## Connect storage and restore the exact code

    Drive authorization may prompt once per runtime. No credentials are saved
    in configuration or outputs. With private delivery, copying the provided
    Git bundle to `PRIVATE_BUNDLE_PATH` is a one-time setup. The same bundle and
    model/data cache work in CPU and GPU sessions.
    """)
    code("""
    import os, sys, json, subprocess, pathlib, shutil, time
    from IPython.display import display, Markdown, HTML
    from google.colab import drive
    drive.mount("/content/drive")
    persistent = pathlib.Path(PERSISTENT_ROOT)
    persistent.mkdir(parents=True, exist_ok=True)
    checkout = pathlib.Path("/content/tas-v7-checkout")
    if not (checkout / ".git").exists():
        source = PRIVATE_BUNDLE_PATH if pathlib.Path(PRIVATE_BUNDLE_PATH).is_file() else REPO_URL
        subprocess.run(["git", "clone", "--no-checkout", source, str(checkout)], check=True)
    exists = subprocess.run(["git", "cat-file", "-e", REPO_COMMIT + "^{commit}"], cwd=checkout, capture_output=True)
    if exists.returncode:
        # Fetch only code. Missing/private commit fails here before model/setup work.
        fetched = subprocess.run(["git", "fetch", "origin", REPO_BRANCH], cwd=checkout, capture_output=True, text=True)
        if fetched.returncode:
            raise RuntimeError("The pinned experiment code is unavailable. Restore the private Git bundle at PRIVATE_BUNDLE_PATH, or configure its authorized remote. No models were loaded.")
    subprocess.run(["git", "checkout", "--detach", REPO_COMMIT], cwd=checkout, check=True)
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    assert actual_commit == REPO_COMMIT, (actual_commit, REPO_COMMIT)
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    print("Research source:", actual_commit)
    """)
    markdown("""
    ## Isolated, pinned environment

    Python 3.10.16 and matched PyTorch 2.4.1 / torchvision 0.19.1 CPU or CUDA
    12.1 wheels run in subprocesses. The legacy Transformers 4.35.2 stack is
    preserved. No notebook-kernel restart, system Torch replacement or CUDA
    extension compilation is needed. GPU mode validates its actual CUDA path.
    """)
    code("""
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    has_gpu = query is not None and query.returncode == 0 and bool(query.stdout.strip())
    effective_phase = ("run" if has_gpu else "prepare") if PHASE == "auto" else PHASE
    mode = "gpu" if effective_phase == "run" else "cpu"
    print("Hardware:", query.stdout.strip() if has_gpu else "CPU")
    print("Phase:", effective_phase, "| environment:", mode)
    if effective_phase == "run" and has_gpu and REQUIRE_A100_80GB and ("A100" not in query.stdout or int(query.stdout.strip().splitlines()[0].split(",")[-1].strip().split()[0]) < 70000):
        raise RuntimeError("This configuration requires A100 80GB; hardware check stopped before environment/model setup.")
    if effective_phase == "run" and not has_gpu:
        raise RuntimeError("Select a GPU runtime, or use PHASE=prepare/score on CPU.")
    if effective_phase == "run" and not (persistent / "preparation.json").is_file():
        raise RuntimeError("Run this notebook once in a CPU runtime first. That downloads/persists models and data before A100 allocation.")
    from tas_runtime.bootstrap import ensure_environment
    python = ensure_environment(mode, "/content/tas-v7-envs", checkout)
    if DATASET == "amber":
        from tas_runtime.bootstrap import ensure_amber
        ensure_amber(python, "/content/tas-v7-envs", checkout)
    env = dict(os.environ, PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg",
               NLTK_DATA=str(persistent / "nlp"), YOLO_CONFIG_DIR=str(pathlib.Path(LOCAL_ROOT) / "ultralytics"))
    configuration = {
        "persistent_root":PERSISTENT_ROOT, "local_root":LOCAL_ROOT, "stage":STAGE,
        "seeds":SEEDS, "methods":METHODS, "batch_sizes":BATCH_SIZES,
        "attention_backends":ATTENTION_BACKENDS, "last_position_logits":LAST_POSITION_LOGITS,
        "max_session_minutes":MAX_SESSION_MINUTES, "require_a100":REQUIRE_A100_80GB,
        "run_confirmation":RUN_CONFIRMATION, "run_controls":RUN_CONTROLS,
        "run_diagnostics":RUN_DIAGNOSTICS, "run_final":RUN_FINAL_REMAINDER,
        "reserve_pope":RESERVE_POPE_IMAGES, "split_seed":SPLIT_SEED,
        "dataset":DATASET, "external_manifest":EXTERNAL_MANIFEST, "external_image_root":EXTERNAL_IMAGE_ROOT,
        "auto_continue_after_smoke":True, "source_revision":actual_commit,
        "generation":{"max_new_tokens":180,"temperature":1.0,"top_p":1.0,"top_k":None,
                      "precision":"fp16","do_sample":True,"conv_mode":"llava_v1","prompt":"Describe this image."}
    }
    config_path = pathlib.Path("/content/tas-v7-config.json")
    config_path.write_text(json.dumps(configuration, indent=2))
    def run_phase(phase):
        subprocess.run([str(python), "-m", "experiments.tas_v7.cli", "--config",str(config_path),"--phase",phase], cwd=checkout,env=env,check=True)
    if DATASET == "amber":
        subprocess.run([str(python), "-m", "tas_runtime.bootstrap", "--root", "/content/tas-v7-envs", "--nlp-root", str(persistent / "nlp"), "--amber"], cwd=checkout, env=env, check=True)
    run_phase("plan")
    """)
    markdown("""
    ## Prepare or run the study

    First CPU session: approximately 17–18 GB of checkpoints plus annotations,
    selected images and NLP resources; existing verified files are reused.
    Only 30 train images are fetched, not all COCO train2017.

    GPU sessions: lightweight preflight → local asset restoration → one model
    load → baseline/view audit → correctness/throughput pilot → configured
    stages. Per-image RNG makes retries/resume independent of job order. A
    rejected optimization is recorded; baseline failures block larger runs.
    Results use immutable chunks and perception caches use shards on Drive.
    """)
    code("""
    phase_started = time.perf_counter()
    if effective_phase == "prepare":
        run_phase("prepare")
        run_phase("check")
        run_phase("score")
        display(Markdown("**CPU preparation complete.** Files are persistent. Switch to the A100 runtime and Run all again."))
    elif effective_phase == "run":
        run_phase("check")
        run_phase("run")
    elif effective_phase == "score":
        run_phase("score")
    elif effective_phase == "check":
        run_phase("check")
    else:
        raise ValueError("Unknown phase")
    print("Session phase seconds:", round(time.perf_counter() - phase_started, 1))
    """)
    markdown("""
    ## Inspect saved reports

    LEGACY and HARMONIZED are separate protocols; do not infer a matched
    algorithm gain across them. CHAIR and annotation category recall do not
    establish caption quality or preserved relations. Inspect ambiguous images,
    captions, failures, coverage, length, timings and uncertainty together.
    CPU scoring/report regeneration is available with PHASE="score".
    """)
    code("""
    import csv, html
    from IPython.display import Image as DisplayImage
    runs_root = persistent / "runs"
    for run_dir in sorted(runs_root.glob("*"))[-3:]:
        display(Markdown("### Run `" + run_dir.name + "`"))
        session = run_dir / "session.json"
        if session.exists():
            display(Markdown("```json\\n" + session.read_text() + "\\n```"))
        gallery = run_dir / "views.png"
        if gallery.exists():
            display(DisplayImage(filename=str(gallery)))
        for summary in sorted(run_dir.glob("*/summary.csv")):
            rows = list(csv.reader(summary.open()))
            table = "<table>" + "".join("<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in row) + "</tr>" for row in rows) + "</table>"
            display(Markdown("**" + summary.parent.name + "**"))
            display(HTML(table))
    print("Persistent artifacts:", persistent)
    print("Reuse this folder for later sessions/new idea notebooks. Do not delete frozen manifests or rewrite completed runs.")
    """)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "colab": {"name": "TAS_v7_Colab.ipynb", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--output", default="notebooks/TAS_v7_Colab.ipynb")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = (
        args.commit
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    )
    notebook = build(commit)
    path = root / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(path)


if __name__ == "__main__":
    main()
