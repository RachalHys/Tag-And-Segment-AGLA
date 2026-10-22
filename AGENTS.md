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
- Preserve frozen dataset/run identities. Use a new identity for changed
  methods, inputs or execution semantics; never rewrite successful outputs.
- Freeze the shortlist coverage rule before outputs: the approved default is
  at most two absolute percentage points of loss versus E1-H; never force a finalist.
- Keep models, datasets, caches, results and delivery bundles out of Git.
  Research publication and promotion to main are separate from local validation.
