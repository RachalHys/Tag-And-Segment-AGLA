# TAS v7 research notebook

The editable experiment code is on `experiment/tas-v7-colab`. The notebook is a
launcher with a generated, compressed copy of those Python modules. It clones
the fixed public baseline and restores the experiment modules automatically.
Only the notebook needs to be uploaded to Colab.

The current delivery starts with `STAGE="audit"` for the requested performance
investigation. It runs P0, profiles one warmed 24-token F4 call, displays the
execution table and stops before smoke. Use `STAGE="flow"` for the full study.
The profile is diagnostic, adds overhead and never supplies policy-selection
timings. CUDA-profiler unavailability is reported without changing P0.
Generation errors during profiling still stop the run. Completed P0 evidence
is saved first, and each profile clears older operator tables.

## Run in Colab

1. Upload `notebooks/TAS_v7_Colab.ipynb` and select a **CPU** runtime.
2. Press **Run all** and authorize Drive. Preflight runs before the large
   downloads. Models, selected data and NLP resources are saved to
   `MyDrive/TAS-Shared`.
3. Select **A100 40 or 80 GB**, then **Run all** again. The notebook installs the GPU
   environment, restores files locally, loads one LLaVA runtime and runs P0.
4. Run again after a session ends. Completed results and perception caches are
   reused. Keep the same persistent root in future idea notebooks.

Defaults use `REQUIRE_A100_80GB=False` and `BATCH_SIZES=[1, 2, 4]` for the observed
A100 40 GB runtime. Serial views and packed views both test image batches;
rejected last-position heads also test their full-head alternative. Larger
batch candidates can be tested separately on 80 GB;
the blocking P0 audit still selects and freezes the execution policy.
Final partial batches repeat the last input to preserve that shape and discard
the padded outputs. Reports distinguish real images from the execution batch.

Allow at least **25 GB free Drive space** for the roughly 17–18 GB checkpoints
plus data and results. Larger runs need more. The default study budget is three
hours per subprocess; unfinished work resumes next session. A hard VM stop may
require regenerating the latest unpublished output chunk.

`AUTO_SHUTDOWN=True` flushes Drive and releases the runtime at the final cell,
including after a planned budget stop. This uses Colab's official
[`runtime.unassign()`](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py).
Setup and execution errors also flush Drive and request release before propagating.
Environment preflight output, including subprocess errors and labelled checks,
is saved under `TAS-Shared/diagnostics/environment/` before release.

The study cell forwards child stdout/stderr live, announces each phase and
prints an elapsed-time heartbeat every 30 seconds during quiet work. Pilot
checks and caption counts appear as they complete. A heartbeat means the child
process is still running; it is not a percentage-complete or GPU-use claim.
Named stages and text bars show model bytes, prepared images, P0 references,
execution options and captions. Caption bars include rate, ETA and the last
measured generation time. Updates are outside token-level model execution.
Rejected-caption comparisons include the first differing token and measured
prefill/decode time; a passing numerical snapshot alone cannot make them eligible.

## Experiment flow

Blocking P0 -> smoke (30 images, 14 methods) -> screen (240, one seed) -> at most
two finalists -> confirmation (300, three seeds plus greedy) -> development
controls -> P0-D diagnostics (75 images). Final remainder is opt-in. P3/P4/P5, FG-G,
hidden interventions and a second backbone are deferred.

The approved shortlist rule permits **at most two percentage points of measured
coverage loss** versus harmonized TAG (`COVERAGE_TOLERANCE_PP=2.0`, absolute
fraction 0.02). It requires an observed CHAIRi or
CHAIRs improvement and excludes dominated policies. The tolerance is frozen in
the run identity before outputs are seen. If no candidate qualifies, reports and
baseline perception diagnostics are saved; confirmation and finalist controls
stop without forcing a finalist.

Native E0/E1 use the LEGACY protocol. New policies and E0-H/E1-H use HARMONIZED;
compare within protocols. FG-U blurs foreground only and preserves background.
New probability math is FP32; weights remain FP16. P0 checks native baselines,
geometry, filtered support and serial/batched behavior before selecting an
execution policy. Rejected options are recorded; the chosen policy stays fixed.

Cached visual tensors are encoded individually with the native image batch size
of one, then reused across methods and batched language decoding. FP16 results
can depend on computation shape, as described in
[PyTorch's numerical accuracy notes](https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/numerical_accuracy.rst).
If every requested optimization is rejected, P0 also tests eager, serial,
single-image decoding with the full vocabulary head. This option must pass the
same checks; tolerances and token/support requirements are unchanged. The head
choice is saved in the locked execution policy and restored on reconnect.

Reports retain failures, CHAIR, category coverage, length, timing scopes and
paired image-bootstrap intervals and a descriptive exact-word-count paired
comparison. Actual batch-return latency is separate from amortized seconds per
image; older records without latency measurements report it as unavailable.
Diagnostics use provisional annotation matches; they do not establish causal
explanations or guaranteed oracle bounds.
Caption quality and ambiguous object mentions still need scientific review.

## Code and persistence

- `tas_runtime/`: model/data preparation, inference, attention, sampling and caches.
- `experiments/tas_v7/`: views, policies, stages, evaluation and diagnostics.
- `tools/build_colab_notebook.py`: package a committed experiment into the notebook.
- Drive `code/`, `assets/`, `data/`, `nlp/`, `caches/`, `runs/`: source snapshots,
  reusable files, results and reports. One writer per study/cache is supported.

An isolated Python 3.10.16 subprocess uses Torch 2.4.1 / torchvision 0.19.1 and
Transformers 4.35.2. Colab's kernel is left intact. Profiles are in `requirements/`;
the pinned protobuf dependency supports the checkpoint's `legacy=False` tokenizer
without changing its tokenization settings.
CLIP builds with pinned setuptools and no build isolation. No external
FlashAttention compilation is needed. Bootstrap checks and research workers use
Matplotlib's file-only `Agg` backend so they do not inherit Colab's kernel-only
inline backend; CPU preflight checks this before checkpoint downloads. See the
official
[PyTorch matrix](https://pytorch.org/get-started/previous-versions/),
[uv Python guide](https://docs.astral.sh/uv/concepts/python-versions/) and
[Colab storage/runtime guidance](https://research.google.com/colaboratory/faq.html).

For optional AMBER/POPE, select `STAGE="external"`, the dataset and explicit
JSONL/image paths. Select frozen methods there; do not tune on those benchmarks.
The default notebook prepares COCO. AMBER's optional scorer uses the original
evaluator and its separate `colab-amber.txt` profile.

## Local checks

```bash
NLTK_DATA=/path/to/prepared/nlp python -m pytest tests -q
python -m compileall -q tas_runtime experiments tools tests
python tools/build_colab_notebook.py --commit <tested-implementation-commit>
# Short performance check instead of the full study:
python tools/build_colab_notebook.py --commit <tested-implementation-commit> --stage audit
```

The tests passed from source restored by the notebook itself over a fresh public
baseline clone. Notebook schema/cell syntax, compilation and lint passed.
Full-checkpoint Colab/A100 execution and the optional full AMBER scorer remain
unrun locally. P0 must pass on the actual GPU before bulk generation; no measured
speedup or scientific improvement is claimed by the local fixtures.
