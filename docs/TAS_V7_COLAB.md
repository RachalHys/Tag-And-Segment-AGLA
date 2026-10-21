# Running the TAS v7 experiment

This implementation is on `experiment/tas-v7-colab`. The original CLI runners,
model definitions, legacy sampler, checkpoint precision and perception questions
remain available. The notebook imports the new Python modules; it does not hold
a second implementation of the model or fusion rules.

## Start in Colab

1. Make the pinned experiment commit available through your chosen Git remote,
   or copy the provided private `tas-v7-source.bundle` to
   `MyDrive/TAS-Shared/code/tas-v7-source.bundle`. A GitHub branch in a public
   repository is public too; a local branch/bundle does not publish the research.
2. Upload `notebooks/TAS_v7_Colab.ipynb`. Keep the same persistent root in this
   and subsequent notebooks. Check that Drive has at least **25 GB free** for
   the roughly 17–18 GB checkpoints plus data, masks and results. Large studies
   need additional room. Colab compute subscriptions do not imply extra Drive
   storage.
3. Select a **CPU** runtime and **Run all**. Approve the normal Drive mount.
   This creates the isolated environment, downloads/verifies model files and
   data, freezes the cohorts, downloads NLP resources, and runs preflight tests.
   No inference model is loaded to download its weights.
4. Select **A100 80 GB**, then **Run all** again. GPU setup validates CUDA,
   SDPA and torchvision NMS before copying/loading model weights. The notebook
   restores local files and starts one LLaVA instance for the study.
5. If the session ends or its configured budget is reached, use **Run all** in
   the next GPU session. Completed image/method/seed records are reused. A CPU
   session can regenerate reports with `PHASE="score"`.

The default run budget is three hours per study subprocess; change it to match
the session you intend to use. A budget stop is not a failed image. Hard runtime
termination may lose the most recent, not-yet-published group of outputs; their
per-record RNG streams make them reproducible on retry under the same kernels.
No claim is made that the full study fits into one session.

Colab's CPU and GPU allocations are treated as different VMs. Python environments
are rebuilt locally; model/data/result files are restored from Drive. Working
inference reads local disk rather than many small Drive reads. These choices
follow the [Colab runtime/storage guidance](https://research.google.com/colaboratory/faq.html).

## Environment compatibility

Do not install `local_requirements.txt` or `kaggle_requirements.txt` into the
notebook kernel for this workflow. `tas_runtime.bootstrap` creates a managed
**Python 3.10.16** subprocess environment, leaving Colab's changing kernel alone.
It installs matched **Torch 2.4.1 / torchvision 0.19.1** CPU or CUDA 12.1 wheels,
Transformers 4.35.2, NumPy 1.26.4 and the declared profiles in `requirements/`.
The exact resolved package versions are recorded. OpenAI CLIP/MobileSAM source
revisions are fixed; CLIP builds with pinned setuptools, not a newer incompatible
`pkg_resources`-free build environment.

The [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/)
documents the matched versions. [uv managed Python](https://docs.astral.sh/uv/concepts/python-versions/)
avoids depending on the Colab kernel's Python version. CUDA 12.x requires a
compatible driver; [newer NVIDIA drivers retain backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The GPU bootstrap exercises actual FP16 SDPA and CUDA NMS, so a mismatched CUDA
stack fails before a large model is loaded. No CUDA extension is compiled.

Actual Colab/A100 execution is a separate verification step from local CPU
integration tests. Network failures, quota exhaustion and future runtime-image
changes cannot be excluded by source review; caches, resumable downloads and
explicit preflight checks reduce lost work.

## Scientific execution

P0 compares sampled E0 and E1 against the actual pinned native/legacy sampler.
It checks actual model input geometry, displays the four RGB views, and measures
serial versus packed branches and image batches 1/2/4/8. Backend candidates must
pass forced-prefix logit checks and match greedy pilot outputs. Rejected options
include any candidate that changes the pilot's original-view plausibility support.
The selected backend/batch policy is locked for subsequent sessions;
there is no silent OOM batch reduction or kernel switch mid-study.

The initial fourteen are exactly E0, C0, C1, C2, E1, E2–E5 and F0–F4. The screen
also records E0-H/E1-H as separate protocol controls. Native E1 retains raw-image
masking before CLIP resize/crop and its original no-detection fallback. The new
views use a locked processor canvas, RGB black, and nearest-neighbor mask geometry.
FG-U blurs the **full image**, then replaces pixels **only inside foreground**;
outside pixels remain unchanged. Probability/log math is FP32 for new policies;
legacy E1 retains its native arithmetic. Temperature is applied once.

The two protocols are explicit:

| Protocol | Purpose |
|---|---|
| LEGACY | Native E0 and original TAG E1 reproduction |
| HARMONIZED | Matched canvas/FP32 support and new policies, with E0-H/E1-H controls |

Do not interpret a cross-protocol table as a matched algorithm comparison.
Within a method every view sees the same selected generated prefix, with separate
KV rows. Across methods, prefixes diverge. Immutable visual features are reused
only for exact input tensors within a minibatch; another method's generated logits
or mutable KV cache are never replayed as free-running outputs.

The owner explicitly changed the brief's original smoke-review pause. The default
flow now continues automatically after blocking correctness checks:

1. Smoke: 30 train images × fourteen methods.
2. Screen: 240 locked val images, one seed, plus harmonized baseline controls.
3. Shortlist: at most two P2 candidates ranked by CHAIRi, CHAIRs, coverage and
   true mentions. Pareto tradeoffs are reported; no unapproved coverage tolerance
   or automatic scientific winner is invented.
4. Confirmation: 300 disjoint images, at least three seeds plus greedy, selected
   candidates and baseline controls.
5. Controls: fixed filter-off, duplicate-context, black-context, area-matched
   random-region, context-only and temperature checks on the development screen
   cohort. Their one-seed/greedy results are labeled development controls.
   Context-only replaces object evidence with ORIG, preserving context weights
   and the parent policy's scale/selective penalty.
6. P0-D: 50–100 images from the locked diagnostic cohort, E0/E1/one finalist,
   predicted evidence and D-MASK/D-BOX/D-MISS. Automatic category/box matching is
   explicitly provisional and retained for review; ground-truth masks are not a
   guaranteed upper bound. Fixed-prefix influence is a diagnostic, not accuracy
   or a gate. D-FALSE needs independent absence verification; it is not fabricated.

Final remainder opening is opt-in. FG-G/P2-A/adaptive/P3/P4/P5, learned controllers
and a second backbone are later work, not automatically claimed implemented or
validated. No COCO split is described as unseen during model training. Official
POPE image IDs are reserved from development by default; exclusions and exact
manifests are recorded before outputs exist.

COCO CHAIR uses the original lexical mapping, instance plus reference-caption
ground truth, and real COCO2017 IDs. The adapter was checked against the original
CHAIR parser. Category coverage is separate annotation-scoped recall, not AMBER
COVER. Reports include true/false mentions, length, empty-object rate, failures,
warm generation cost, shared feature/perception cost and paired image-cluster
bootstrap intervals. Caption quality and relation preservation still require
independent review; the code does not invent a quality score.

## Persistent files and reuse

```text
TAS-Shared/
  assets/                 # fixed model files, revisions, hashes and receipts
  data/coco/              # image bytes, source annotations, locked cohorts
  data/pope/              # official manifests used for overlap exclusion
  nlp/                    # fixed NLP resources
  caches/perception/      # immutable NPZ shards + verified index
  preparations/           # complete preparation identities
  runs/<identity>/        # config, P0/pilot, gallery, stage results and reports
```

Model/data files are shared across new ideas. Perception cache keys include image
bytes, auxiliary model/source versions, precision and settings. Fusion changes do
not recompute perception. Changing masks, view definitions, models, prompts,
generation settings, execution policy or scientific source creates a distinct run.
Resume identity includes image, query, method, seed and full protocol dependencies.
Completed outputs cannot be overwritten; failures remain visible and retries do
not silently exclude difficult examples. One writer per cache/run is supported.

Results are fsynced locally and published as immutable incremental chunks, keeping
Drive traffic proportional to new outputs. Perception is also bundled into shards.
Peak VRAM, generated tokens and stage timing scopes are recorded. Warm feature-cache
generation time is not presented as cold end-to-end image latency.

## Optional AMBER and POPE

Use `STAGE="external"`, choose `DATASET`, and provide a verified JSONL query
manifest and image root in `EXTERNAL_MANIFEST` / `EXTERNAL_IMAGE_ROOT`. Select
frozen finalist/baseline IDs in `METHODS`; do not tune there. The default COCO flow
does not automatically download another full benchmark or open final data.

The generic external manifest accepts official `question_id`/`text`/`image`
fields (or `id`/`query`), preserving question identity and labels. Store the files
under the same persistent root. Official AMBER images are linked in `AMBER/README.md`;
normalize its JSON query array to JSONL without changing text or IDs. Missing files
are rejected before captioning. POPE variants are evaluated separately with the
official binary parser plus separately labeled invalid/Yes-rate diagnostics.

AMBER additionally installs `requirements/colab-amber.txt` and its NLP resources.
It calls the existing `AMBER/inference.py` with frozen resources and memoizes the
unchanged NLP operations. The existing global-args dependency is supplied explicitly.
Its evaluator/NLP model revisions are recorded. This optional full scorer profile
has not been executed with real AMBER outputs locally.

## Local checks and delivery

```bash
NLTK_DATA=/path/to/nltk_data python -m pytest tests -q
python -m compileall -q tas_runtime experiments tools tests
python -m experiments.tas_v7.cli --config config.json --phase plan
python tools/build_colab_notebook.py --commit <implementation-commit>
```

The Python module sources are the editable implementation. Regenerate the thin
notebook after selecting the implementation commit. To deliver privately, create
a full Git bundle containing that commit and its history and put it at the notebook's
configured Drive path. To deliver through GitHub, publish the branch only to the
remote/visibility you intend; nothing in the notebook pushes code automatically.
