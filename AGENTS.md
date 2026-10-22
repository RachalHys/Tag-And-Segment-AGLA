# Project guidance

The TAS v7 experiment is documented in [docs/TAS_V7_COLAB.md](docs/TAS_V7_COLAB.md).
Its Python implementation lives in `tas_runtime/` and `experiments/tas_v7/`;
the Colab notebook imports a fixed implementation commit. Keep the original
`eval_tas/`, `eval_baseline/` and model definitions reproducible.

- Use the isolated Python 3.10 profile in `tas_runtime.bootstrap` for this
  experiment. Do not install the legacy requirements into Colab's kernel.
  Set `MPLBACKEND=Agg` for managed subprocesses without changing the kernel.
  Preserve the checkpoint's `legacy=False` tokenizer and its protobuf dependency.
- Validate with `NLTK_DATA=/path/to/prepared/nlp python -m pytest tests -q` and
  `python -m compileall -q tas_runtime experiments tools tests`. Tests use a
  miniature real LLaVA model and require no full checkpoints or GPU.
- Generate the notebook with
  `python tools/build_colab_notebook.py --commit <implementation-commit>`.
  The generated notebook packages that commit's experiment Python modules and
  restores them over the pinned public baseline; no separate upload is needed.
- Keep the P0 baseline, geometry, support and execution checks blocking. CPU
  fixtures do not establish full-checkpoint parity or A100 performance.
  Encode cached visual tensors individually to preserve the native FP16 vision
  batch shape. Freeze the vocabulary-head choice with the execution policy;
  the eager/serial/full-head option must pass the same P0 checks.
  Serial views can still batch multiple independent images. Audit those image
  batches and failed-head alternatives; keep rejected options out of generation.
  Keep the selected batch shape on partial batches; padded rows cannot become
  scored results or change a real image's RNG stream.
- Deliver the notebook with the default `STAGE="flow"` for the full implemented
  study. `--stage audit` is an optional short diagnostic, not the default delivery.
  Its optional operator profile runs only after P0 and never supplies selection timings.
  Profiler unavailability must not suppress generation errors or display stale tables.
- `TAS_v7_Batch_Compare.ipynb` is a separate descriptive execution comparison.
  It keeps strict B=1 P0, evaluates batched token drift under its own run/ledger
  identities, and never promotes a rejected policy into the full study.
  Preserve method IDs and per-image RNG keys across both execution policies.
  Compare complete captions on the predeclared A prefix; keep quality scores
  out of execution-policy selection. No equivalence margin is approved.
  Keep its quick notebook defaults at 4 images, E0/E1/F4, 24 complete captions
  and a 20-minute cooperative study budget; full-study defaults stay unchanged.
- Preserve frozen dataset/run identities. Use a new identity for changed
  methods, inputs or execution semantics; never rewrite successful outputs.
  Reuse screen samples in controls only with identical ordered inputs, methods,
  seeds and execution settings. Retain source provenance and measured costs;
  report reused outputs separately from new generation.
  Diagnostic reuse also requires matching every visual tensor for the whole
  batch, including unchanged companions/padding; retain destination metadata.
- Keep provisional annotation correspondences distinct from independent
  verification. Report no-match interventions honestly; paired perception
  deltas/mask IoU are supervised diagnostics, not causal bounds or a gate.
- Freeze the shortlist coverage rule before outputs: the approved default is
  at most two absolute percentage points of loss versus E1-H; never force a finalist.
- Keep models, datasets, caches, results and delivery bundles out of Git.
  Research publication and promotion to main are separate from local validation.
