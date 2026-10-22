# TAS v7 research notebook

The editable experiment code is on `experiment/tas-v7-colab`. The notebook is a
launcher with a generated, compressed copy of those Python modules. It clones
the fixed public baseline and restores the experiment modules automatically.
Only the notebook needs to be uploaded to Colab.

The notebook defaults to `STAGE="flow"` and runs the full implemented first-round
study with `EXECUTION_PROTOCOL="tas-batched-study-v1"`: **SDPA, packed visual
branches, B=4 and last-position logits**. This explicit research protocol accepts
reported token drift from batched FP16 computation. Native E0/E1 B=1 reproduction,
geometry, fixed-prefix numerical and filtered-support checks remain blocking.
P0 checks fixed-prefix logits/support for all 14 methods and matched E0-H/E1-H
baselines against immutable raw-per-image, eager/full-head references. Short
greedy/sample drift probes cover E0/E1/F4; P0 does not generate complete serial
captions for all 16 configurations. It then runs smoke, screening and the eligible
stages below, checking one fixed policy without the old 24-option sweep. A session
budget saves progress for the next Run all. There is no silent batch or precision
fallback.

The user's A100 40 GB comparison measured **6.6× TAG and 7.9× F4 warm generation
speedups** for packed B=4, with peak allocation 22.3 GiB. FlashAttention kernels
were already active through SDPA. These are four-image, one-seed observations;
they establish neither all-method quality equivalence nor whole-study speed.
The new protocol is for quick hypothesis testing, not official native evaluation.
Weights remain FP16; LEGACY keeps native arithmetic and HARMONIZED fusion math
is FP32. Generation/cohort settings are unchanged.
Every study method, baseline and control uses the same frozen execution policy.
Changed identities keep this study separate from earlier strict outputs.

Set `EXECUTION_PROTOCOL="strict"` in the settings cell before outputs to restore
exact-token policy selection. Its settings derive the original B=1/2/4,
eager/SDPA and serial/packed search. The API also retains strict defaults.

`STAGE="audit"` is an optional short performance investigation, built with strict
defaults. It runs P0,
profiles one warmed 24-token F4 call and stops before smoke. Its profile never
supplies policy-selection timings. CUDA-profiler unavailability is reported
without changing P0; generation errors still stop the run. Completed P0 evidence
is saved first, and each profile clears older operator tables.

`notebooks/TAS_v7_Batch_Compare.ipynb` is a separate, bounded execution comparison
authorized after the A100 rejected every batched strict-token candidate. Use
this notebook on A100 with **Run all** and the existing `TAS-Shared` folder;
existing CPU preparation is reused. The regular notebook still defaults to flow.

The comparison keeps native E0/E1, geometry, support and strict execution gates
for a serial B=1 reference. It then tests the largest configured image batch
(default B=4), with serial and packed views, holding attention/head/precision
fixed. Initial forced-prefix numerical/support failures still reject a candidate;
free greedy token differences are recorded under a separate execution protocol.
At their first divergence, a natural-prefix replay records branch logit errors,
support differences, top-two margins, sampling-distribution total variation and
whether replay reproduced the observed choices. A failure to reproduce them
remains unresolved; the report does not infer harmless rounding.

Before looking at quality scores, it freezes the candidate with the higher
sampled F4 token throughput in a warmed, up-to-48-token probe. It compares
**4 frozen A development images × E0/E1/F4 × one sampled seed × two policies
= 24 complete captions**, with the normal EOS/180-token limit. Separate ledgers
retain the original method IDs/RNG keys. Results include paired CHAIR, coverage,
length and image-bootstrap intervals, exact-token/caption rates, changed-caption
examples, EOS/truncation rates, full-caption throughput and peak memory. CHAIRi
uses pooled mention counts. Warm generation speed excludes shared setup,
perception, feature preparation and profiling; the short probe does not promise
full-stage speed. Sampled E0/F4 B=1 profiles save operator tables and explicit
scalar-read, synchronization, transfer, matrix-multiply and attention totals.

The notebook uses a **20-minute cooperative study budget**, including local
asset restoration and model loading. Environment installation is additional.
The budget is checked between calls and saves completed work; it cannot interrupt
an active model call or file operation. A fresh runtime still copies the cached
weights from Drive, so no few-minute total-runtime promise is made.

This is a **descriptive pilot**, not a test establishing equivalence. One seed
and four already-inspected development images cannot validate all 14 methods or
held-out quality. A confidence interval containing zero is not equivalence.
The two-point shortlist coverage rule supplies no execution-equivalence margin.
The comparison never promotes a rejected policy to strict P0 or automatically
changes full flow. The named batched study above is a separate, explicitly
authorized research protocol; its descriptive token drift is not strict parity.
The comparison saves `batch-comparison-*/comparison.json`, `discovery.json`,
`changed-captions.jsonl` and per-policy results in its own run directory, then
releases the runtime. Completed captions resume without regeneration.

## Run in Colab

1. Upload `notebooks/TAS_v7_Colab.ipynb` and select a **CPU** runtime.
2. Press **Run all** and authorize Drive. Preflight runs before the large
   downloads. Models, selected data and NLP resources are saved to
   `MyDrive/TAS-Shared`.
3. Select **A100 40 or 80 GB**, then **Run all** again. The notebook installs the GPU
   environment, restores files locally, loads one LLaVA runtime and runs P0.
   With the default `STAGE="flow"`, a passing P0 continues into the study automatically.
4. Run again after a session ends. Completed results and perception caches are
   reused. Keep the same persistent root in future idea notebooks.

Defaults use `REQUIRE_A100_80GB=False`, `BATCH_SIZES=[4]`, `EXECUTION="packed"`,
`ATTENTION_BACKENDS=["sdpa"]` and `PILOT_IMAGES=4` for the measured A100 40 GB
policy. Keep them fixed for the batched study. Insufficient memory or a failed
blocking check stops with its evidence; it does not silently downgrade the run.
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
Caption drift reports include differing tokens and measured prefill/decode time.
Strict mode rejects caption mismatches; the named batched study records them
without claiming exact equivalence. Both modes block numerical/support failures.
P0 generates raw reference captions only for images a requested or locked batch
can consume; additional preview images do not trigger unused generation.

## Experiment flow

Blocking P0 -> smoke (30 images, 14 methods) -> screen (240, one seed) -> at most
two finalists -> confirmation (300, three seeds plus greedy) -> development
controls -> P0-D diagnostics (75 images). Final remainder is opt-in. P3/P4/P5, FG-G,
hidden interventions and a second backbone are deferred.

Full flow reuses matching sampled baseline/finalist captions from screening in
controls. It requires identical ordered images, methods, seeds, generation and
execution identities. Greedy and modified controls run independently. Reports
mark reused captions, retain their original measured costs and separate new
decoder time in stage metadata.

Perception diagnostics also reuse completed D-PRED captions when every named
visual tensor, prompt, seed, method and execution setting matches. The entire
batch, including companions and final padding, must be unchanged. This avoids
225 duplicate E0 captions in the default 75-image, three-intervention audit;
unchanged no-match cases can reuse other methods as well. Destination records
retain the intervention metadata and source provenance. New generation costs
are separate from reused measurements.
Stage metadata also records current-session wall time through report construction
and counts imported, previously completed and newly generated outputs. This
includes CPU/scoring/retry/storage overhead; global model restore/load time is
included in the session elapsed time instead.

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
geometry, filtered support and execution behavior. Strict mode selects a passing
policy; the batched study validates its fixed policy with reported token drift.
The execution protocol and policy are visible in audit/report displays.

Cached visual tensors are encoded individually with the native image batch size
of one, then reused across methods and batched language decoding. FP16 results
can depend on computation shape, as described in
[PyTorch's numerical accuracy notes](https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/numerical_accuracy.rst).
In strict mode, if every requested optimization is rejected, P0 also tests eager, serial,
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
Automatic P0-D runs D-PRED, D-MASK, D-BOX and D-MISS. Annotation-supported
D-MISS targets remain provisional; they are not marked independently verified.
No-match interventions record unchanged counts. The audit records matched mask
IoU and areas; a separate CPU report displays within-method paired intervention
deltas and image-bootstrap intervals, retaining no-op images and undefined
denominators. These supervised perturbations do not prove causal headroom.
D-FALSE and phrase-utility
checks require independently verified examples and remain manual research steps;
the notebook does not infer verified absence from missing COCO annotations.

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
The integrated full batched study and optional full AMBER scorer remain unrun
locally. P0 must pass on the actual GPU before bulk generation; local fixtures
do not establish measured speed or scientific quality. The small A100 comparison
above is separate evidence, with the stated sample and timing limitations.
