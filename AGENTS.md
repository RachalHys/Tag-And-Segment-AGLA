# Project guidance

The TAS v7 experiment is documented in [docs/TAS_V7_COLAB.md](docs/TAS_V7_COLAB.md).
Its Python implementation lives in `tas_runtime/` and `experiments/tas_v7/`;
the Colab notebook imports a fixed implementation commit. Keep the original
`eval_tas/`, `eval_baseline/` and model definitions reproducible.

- Use the isolated Python 3.10 profile in `tas_runtime.bootstrap` for this
  experiment. Do not install the legacy requirements into Colab's kernel.
- Validate with `NLTK_DATA=/path/to/prepared/nlp python -m pytest tests -q` and
  `python -m compileall -q tas_runtime experiments tools tests`. Tests use a
  miniature real LLaVA model and require no full checkpoints or GPU.
- Generate the notebook with
  `python tools/build_colab_notebook.py --commit <implementation-commit>`.
  Ensure that commit is present in the selected remote or private Git bundle.
- Keep the P0 baseline, geometry, support and execution checks blocking. CPU
  fixtures do not establish full-checkpoint parity or A100 performance.
- Preserve frozen dataset/run identities. Use a new identity for changed
  methods, inputs or execution semantics; never rewrite successful outputs.
- Keep models, datasets, caches, results and delivery bundles out of Git.
  Research publication and promotion to main are separate from local validation.
