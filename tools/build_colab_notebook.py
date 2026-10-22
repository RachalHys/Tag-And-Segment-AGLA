"""Generate the repository-backed notebook; source remains in Python modules."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import textwrap
from pathlib import Path

if __package__:
    from .notebook_delivery import (
        BASE_COMMIT,
        package_from_commit,
        restore_source_package,
    )
else:
    from notebook_delivery import (
        BASE_COMMIT,
        package_from_commit,
        restore_source_package,
    )


def run_logged(command, *, cwd=None, env=None, label="process", heartbeat_seconds=30):
    """Forward child output through notebook stdout, including quiet-phase heartbeats."""
    import codecs
    import os
    import selectors
    import signal
    import subprocess
    import time

    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    command = list(map(str, command))
    started = time.monotonic()
    print(f"[{label}] started", flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    next_heartbeat = started + heartbeat_seconds
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                timeout = max(0, next_heartbeat - time.monotonic())
                if selector.select(timeout):
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        print(decoder.decode(b"", final=True), end="", flush=True)
                        break
                    print(decoder.decode(chunk), end="", flush=True)
                    next_heartbeat = time.monotonic() + heartbeat_seconds
                now = time.monotonic()
                if now >= next_heartbeat:
                    print(
                        f"\n[{label}] still running; elapsed {now - started:.0f}s",
                        flush=True,
                    )
                    next_heartbeat = now + heartbeat_seconds
        returncode = process.wait()
    except BaseException as exc:
        if process.poll() is None:
            interrupted = isinstance(exc, KeyboardInterrupt)
            if interrupted:
                process.send_signal(signal.SIGINT)
            else:
                process.terminate()
            try:
                process.wait(timeout=10 if interrupted else 5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    print(f"\n[{label}] finished in {time.monotonic() - started:.1f}s", flush=True)


def build(
    commit,
    repo_url="https://github.com/RachalHys/Tag-And-Segment-AGLA.git",
    *,
    source_package,
    stage="flow",
):
    if stage not in {"flow", "audit", "batch_compare"}:
        raise ValueError("Notebook default stage must be flow, audit or batch_compare.")
    if (
        source_package["source_commit"] != commit
        or source_package["base_commit"] != BASE_COMMIT
    ):
        raise ValueError("Notebook and packaged source identities differ.")
    cells = []

    def markdown(source):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n",
            }
        )

    def code(source, *, hidden=False, release_on_error=False):
        source = textwrap.dedent(source).strip() + "\n"
        if release_on_error:
            source = (
                "try:\n"
                + textwrap.indent(source, "    ")
                + "except BaseException:\n"
                + "    import traceback\n"
                + "    traceback.print_exc()\n"
                + "    release_runtime()\n"
                + "    raise\n"
            )
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"cellView": "form"} if hidden else {},
                "outputs": [],
                "source": source,
            }
        )

    if stage == "audit":
        markdown("""
        **Performance check:** this copy uses `STAGE="audit"`. It restores the
        saved assets, runs P0 and a short operator profile, displays the report,
        then releases the runtime. It does not start the 420-caption smoke run.
        Use `STAGE="flow"` for the complete study after choosing the execution policy.
        """)
    if stage == "batch_compare":
        markdown("""
        **Batched execution comparison:** this notebook defaults to
        `STAGE="batch_compare"`. It keeps strict P0 for a serial B=1 reference,
        measures serial and packed B=4, then freezes the faster sampled F4 policy
        before evaluating complete captions. Default: the first 4 frozen
        development images, E0/E1/F4, one shared seed, 24 captions total.
        It reports token drift, paired CHAIR/coverage/length, uncertainty, speed
        and sampled operator profiles, then saves to Drive and releases the runtime.
        It does not run smoke or select a winning TAS hypothesis. This small
        comparison does not establish quality equivalence or approve the batch
        policy for the complete study. The regular notebook still defaults to flow.
        Existing CPU-prepared assets are reused; choose A100 and Run all.
        The study has a 20-minute cooperative budget, checked between calls;
        unfinished work is saved. Environment installation is additional and
        a fresh runtime must still restore and load the model. This is a quick
        bottleneck/drift investigation, not evidence of quality equivalence.
        """)
    markdown("""
    # TAS v7 — CPU preparation / A100 experiments

    Upload this notebook to Colab. Use **CPU → Run all** first, then select your
    **A100 40 or 80 GB → Run all**. `PHASE="auto"` selects preparation on CPU and the
    study on GPU. Models/data/caches/results live in the same persistent Drive
    folder across sessions and future notebooks. CPU/GPU environments are
    recreated locally; the notebook kernel is left intact.

    Bring **only this notebook**. It clones the fixed public baseline, verifies
    and restores its packaged experiment modules, then imports them normally.
    The editable implementation remains in the experiment branch; the
    compressed package is generated from its exact commit. No separate source
    upload is required. The code snapshot is also saved to Drive with its hashes.

    Automatic flow: blocking P0 audit → 30-image smoke → 240-image screen →
    predeclared shortlist → confirmation (3 seeds + greedy) → controls → P0-D.
    Final remainder and P3/P4/P5 remain disabled by default. A session
    budget saves progress; the next Run all continues completed stages.
    Shortlisting requires the predeclared coverage guard and a nondominated
    observed hallucination improvement. If none qualifies, the study saves its
    comparison and runs baseline perception diagnostics, then stops dependent
    stages without forcing a finalist.

    Automatic P0-D covers D-PRED, D-MASK, D-BOX and D-MISS. D-FALSE and
    correctness-based phrase probes require independently verified examples
    and remain manual research steps. Caption quality and ambiguous mentions
    also require scientific review.

    Full A100 weights have not been tested by the local CPU suite. P0 rejects
    baseline mismatches before bulk generation. No scientific gains are assumed.
    """)
    code(f"""
    # Edit these settings once. Reuse PERSISTENT_ROOT in future idea notebooks.
    REPO_URL = {repo_url!r}
    BASELINE_COMMIT = {BASE_COMMIT!r}
    RESEARCH_COMMIT = {commit!r}
    PERSISTENT_ROOT = "/content/drive/MyDrive/TAS-Shared"
    LOCAL_ROOT = "/content/tas-v7-work"
    PHASE = "auto"             # auto / prepare / run / score / check
    STAGE = {stage!r}          # flow = complete study; batch_compare = paired execution check; audit = P0 only
    COMPARISON_IMAGES = {4 if stage == "batch_compare" else 12}    # fixed A prefix; used only by batch_compare
    MAX_SESSION_MINUTES = {20 if stage == "batch_compare" else 180}  # cooperative study budget; environment setup is additional
    SEEDS = [0, 1, 2]
    REQUIRE_A100_80GB = False   # supports your A100 40 GB; True enforces at least 80 GB
    AUTO_SHUTDOWN = True       # flush Drive, then release runtime after this session's work
    RUN_CONFIRMATION = True
    RUN_CONTROLS = True
    RUN_DIAGNOSTICS = True
    RUN_FINAL_REMAINDER = False
    RESERVE_POPE_IMAGES = True
    DATASET = "coco"           # AMBER/POPE require STAGE=external and explicit verified manifests below
    EXTERNAL_MANIFEST = None
    EXTERNAL_IMAGE_ROOT = None
    METHODS = {["E0", "E1", "F4"] if stage == "batch_compare" else []!r}  # comparison subset; empty = initial 14 for flow
    BATCH_SIZES = [1, 2, 4]     # image batches, including serial views; measured and checked before use
    ATTENTION_BACKENDS = ["eager", "sdpa"]
    LAST_POSITION_LOGITS = True
    SPLIT_SEED = 20261021
    COVERAGE_TOLERANCE_PP = 2.0  # approved absolute percentage-point loss vs harmonized TAG; freeze before outputs

    def release_runtime():
        if AUTO_SHUTDOWN:
            from google.colab import drive, runtime
            print("Saving pending Drive writes, then releasing the Colab runtime.")
            drive.flush_and_unmount()
            runtime.unassign()
    """)
    code(
        "# @title Versioned experiment source package — generated, do not edit\n"
        + "SOURCE_PACKAGE = "
        + repr(source_package)
        + "\n\n"
        + inspect.getsource(restore_source_package)
        + "\n\n"
        + inspect.getsource(run_logged),
        hidden=True,
    )
    markdown("""
    ## Connect storage and restore the exact code

    Drive authorization may prompt once per runtime. No credentials are saved
    in configuration or outputs. Source restoration is automatic and verifies
    both the archive and every file before installation. It cannot replace any
    file from the frozen public baseline. The same caches work in CPU/GPU sessions.
    """)
    code(
        """
    import os, sys, json, subprocess, pathlib, shutil, time
    from IPython.display import display, Markdown, HTML
    from google.colab import drive
    drive.mount("/content/drive")
    persistent = pathlib.Path(PERSISTENT_ROOT)
    persistent.mkdir(parents=True, exist_ok=True)
    checkout = pathlib.Path("/content/tas-v7-checkout")
    if (SOURCE_PACKAGE["base_commit"], SOURCE_PACKAGE["source_commit"]) != (BASELINE_COMMIT, RESEARCH_COMMIT):
        raise ValueError("Regenerate the notebook to change its fixed code versions.")
    source_receipt = restore_source_package(SOURCE_PACKAGE, checkout, persistent, REPO_URL)
    actual_commit = source_receipt["source_commit"]
    os.chdir(checkout)
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    # A new idea notebook may reuse this VM with different experiment modules.
    for module_name in list(sys.modules):
        if module_name == "tas_runtime" or module_name.startswith("tas_runtime."):
            del sys.modules[module_name]
    import importlib
    importlib.invalidate_caches()
    print("Public baseline:", BASELINE_COMMIT)
    print("Experiment Python source:", actual_commit, "| verified files:", len(source_receipt["files"]))
    """,
        release_on_error=True,
    )
    markdown("""
    ## Isolated, pinned environment

    Python 3.10.16 and matched PyTorch 2.4.1 / torchvision 0.19.1 CPU or CUDA
    12.1 wheels run in subprocesses. The legacy Transformers 4.35.2 stack is
    preserved. No notebook-kernel restart, system Torch replacement or CUDA
    extension compilation is needed. GPU mode validates its actual CUDA path.
    """)
    code(
        """
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
    from tas_runtime.bootstrap import ensure_environment, subprocess_environment
    python = ensure_environment(mode, "/content/tas-v7-envs", checkout,
                                diagnostics_dir=persistent / "diagnostics/environment")
    if DATASET == "amber":
        from tas_runtime.bootstrap import ensure_amber
        ensure_amber(python, "/content/tas-v7-envs", checkout)
    env = dict(subprocess_environment(), PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false",
               NLTK_DATA=str(persistent / "nlp"), YOLO_CONFIG_DIR=str(pathlib.Path(LOCAL_ROOT) / "ultralytics"))
    configuration = {
        "persistent_root":PERSISTENT_ROOT, "local_root":LOCAL_ROOT, "stage":STAGE,
        "seeds":SEEDS, "methods":METHODS, "batch_sizes":BATCH_SIZES,
        "attention_backends":ATTENTION_BACKENDS, "last_position_logits":LAST_POSITION_LOGITS,
        "comparison_images":COMPARISON_IMAGES,
        "max_session_minutes":MAX_SESSION_MINUTES, "require_a100":REQUIRE_A100_80GB,
        "run_confirmation":RUN_CONFIRMATION, "run_controls":RUN_CONTROLS,
        "run_diagnostics":RUN_DIAGNOSTICS, "run_final":RUN_FINAL_REMAINDER,
        "reserve_pope":RESERVE_POPE_IMAGES, "split_seed":SPLIT_SEED,
        "coverage_tolerance":COVERAGE_TOLERANCE_PP / 100.0,
        "dataset":DATASET, "external_manifest":EXTERNAL_MANIFEST, "external_image_root":EXTERNAL_IMAGE_ROOT,
        "auto_continue_after_smoke":True, "source_revision":actual_commit,
        "generation":{"max_new_tokens":180,"temperature":1.0,"top_p":1.0,"top_k":None,
                      "precision":"fp16","do_sample":True,"conv_mode":"llava_v1","prompt":"Describe this image."}
    }
    config_path = pathlib.Path("/content/tas-v7-config.json")
    config_path.write_text(json.dumps(configuration, indent=2))
    def run_phase(phase):
        run_logged([str(python), "-u", "-m", "experiments.tas_v7.cli", "--config",str(config_path),"--phase",phase], cwd=checkout,env=env,label=phase)
    if DATASET == "amber":
        run_logged([str(python), "-u", "-m", "tas_runtime.bootstrap", "--root", "/content/tas-v7-envs", "--nlp-root", str(persistent / "nlp"), "--amber"], cwd=checkout, env=env, label="NLP preparation")
    run_phase("plan")
    """,
        release_on_error=True,
    )
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
    Child stdout/stderr is forwarded live to this cell. Quiet phases print an
    elapsed-time heartbeat every 30 seconds; it indicates a running process,
    not a measured percentage complete. Pilot checks and caption counts follow.
    """)
    code(
        """
    phase_started = time.perf_counter()
    if effective_phase == "prepare":
        # Small NLP assets enable the offline scientific fixtures first.
        run_logged([str(python), "-u", "-m", "tas_runtime.bootstrap", "--root", "/content/tas-v7-envs", "--nlp-root", str(persistent / "nlp")], cwd=checkout, env=env, label="NLP preparation")
        run_phase("check")
        run_phase("prepare")
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
    """,
        release_on_error=True,
    )
    markdown("""
    ## Inspect saved reports

    LEGACY and HARMONIZED are separate protocols; do not infer a matched
    algorithm gain across them. CHAIR and annotation category recall do not
    establish caption quality or preserved relations. Inspect ambiguous images,
    captions, failures, coverage, length, timings and uncertainty together.
    Reports include descriptive comparisons for paired captions with exactly
    equal word counts. Batch latency and amortized seconds/image are separate.
    CPU scoring/report regeneration is available with PHASE="score".
    """)
    code(
        """
    import csv, html
    from IPython.display import Image as DisplayImage
    def show_table(rows):
        markup = "".join("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in row) + "</tr>" for row in rows)
        display(HTML("<table>" + markup + "</table>"))
    runs_root = persistent / "runs"
    run_dirs = [path for path in runs_root.glob("*") if path.is_dir()]
    run_dirs.sort(key=lambda path: (path / "resolved-config.json").stat().st_mtime if (path / "resolved-config.json").exists() else path.stat().st_mtime, reverse=True)
    for run_dir in run_dirs[:3]:
        display(Markdown("### Run `" + run_dir.name + "`"))
        session = run_dir / "session.json"
        if session.exists():
            display(Markdown("```json\\n" + session.read_text() + "\\n```"))
        gallery = run_dir / "views.png"
        if gallery.exists():
            display(DisplayImage(filename=str(gallery)))
        audit = run_dir / "audit/audit.json"
        if audit.exists():
            value = json.loads(audit.read_text())
            display(Markdown("**P0 selected policy:** `" + json.dumps(value.get("selected_policy")) + "`"))
            cells = [("Attention", "Views", "Images/batch", "Last-position head", "Status", "Outputs/h (pilot)", "Peak GiB")]
            for item in value.get("pilot", []):
                cells.append((item["backend"], item["execution"], item["batch_images"], item.get("last_position_logits", value.get("last_position_logits", "")), item["status"], round(item["outputs_per_hour"]) if "outputs_per_hour" in item else "", round(item["peak_memory_bytes"] / 1024**3, 1) if "peak_memory_bytes" in item else ""))
            show_table(cells)
            operators = run_dir / "audit/operators.txt"
            if operators.exists():
                display(HTML("<details><summary>Short operator profile (diagnostic timings)</summary><pre>" + html.escape(operators.read_text()) + "</pre></details>"))
        for comparison in sorted(run_dir.glob("batch-comparison-*/comparison.json")):
            value = json.loads(comparison.read_text())
            display(Markdown("**Paired execution comparison — descriptive only**"))
            display(Markdown("Reference: `" + json.dumps(value["reference_policy"]) + "`  Candidate: `" + json.dumps(value["candidate_policy"]) + "`"))
            cells = [("Method", "Mode", "Pairs", "Exact tokens", "Speedup", "CHAIRi delta", "CHAIRs delta", "Coverage delta", "Word length delta")]
            for item in value["summary"]:
                cells.append((item["method"], item["generation_mode"], item["n_pairs"], item["exact_token_match_rate"], item["speedup"], item["CHAIRi_delta"], item["CHAIRs_delta"], item["category_coverage_delta"], item["length_words_delta"]))
            show_table(cells)
            display(Markdown(value["interpretation"] + " Full intervals and diagnostics: `" + str(comparison) + "`"))
            changed = comparison.parent / "changed-captions.jsonl"
            if changed.exists():
                examples = [json.loads(line) for line in changed.read_text().splitlines() if line][:12]
                show_table([("Image", "Method", "Reference caption", "Batched caption")] + [(item["image_id"], item["method"], item["reference_caption"], item["candidate_caption"]) for item in examples])
            for operators in sorted(comparison.parent.glob("profile-*/operators.txt")):
                display(HTML("<details><summary>Sampled " + html.escape(operators.parent.name) + " operator profile</summary><pre>" + html.escape(operators.read_text()) + "</pre></details>"))
        diagnostic_pairs = run_dir / "diagnostics/paired-interventions.json"
        if diagnostic_pairs.exists():
            value = json.loads(diagnostic_pairs.read_text())
            display(Markdown("**Paired perception interventions:** " + value["scope"]))
            cells = [("Intervention", "Method", "Applied images", "Mask changed", "CHAIRi delta (pp), 95% CI", "Coverage delta (pp), 95% CI")]
            def interval(result):
                if result.get("status") == "undefined":
                    return "undefined"
                return "{:+.2f} [{:+.2f}, {:+.2f}]".format(*(100 * result[key] for key in ("delta_candidate_minus_baseline", "ci_low", "ci_high")))
            for item in value["comparisons"]:
                metrics = {result["metric"]: result for result in item["paired"]}
                cells.append((item["intervention"], item["method"], len(item["applied_image_ids"]), len(item["mask_changed_image_ids"]), interval(metrics["CHAIRi"]), interval(metrics["category_coverage"])))
            show_table(cells)
        for summary in sorted(run_dir.glob("*/summary.csv")):
            rows = list(csv.reader(summary.open()))
            display(Markdown("**" + summary.parent.name + "**"))
            show_table(rows)
    print("Persistent artifacts:", persistent)
    print("Reuse this folder for later sessions/new idea notebooks. Do not delete frozen manifests or rewrite completed runs.")
    """,
        release_on_error=True,
    )
    markdown("""
    ## Release the session

    With `AUTO_SHUTDOWN=True`, the final cell flushes Drive writes and releases
    the runtime after preparation, evaluation or a planned budget stop. Errors
    also invoke this cleanup before propagating. Reports remain on Drive.
    Set it to False only when keeping the runtime for inspection.
    """)
    code("""
    release_runtime()
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
            "colab": {"name": "TAS_v7_Batch_Compare.ipynb" if stage == "batch_compare" else "TAS_v7_Colab.ipynb", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--output")
    parser.add_argument("--stage", choices=("flow", "audit", "batch_compare"), default="flow")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", (args.commit or "HEAD") + "^{commit}"],
        cwd=root,
        text=True,
    ).strip()
    notebook = build(commit, source_package=package_from_commit(root, commit), stage=args.stage)
    path = root / (args.output or (
        "notebooks/TAS_v7_Batch_Compare.ipynb" if args.stage == "batch_compare"
        else "notebooks/TAS_v7_Colab.ipynb"
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(path)


if __name__ == "__main__":
    main()
