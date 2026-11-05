# Reproduction

Run commands from the repository checkout. Data and model downloads use pinned revisions with SHA256 checks. No machine-specific drive letter or server directory is required by the code.

## CPU verification

```bash
python -m venv .venv
# Activate this environment using your shell's normal command.
python -m pip install -e .
python -m egus verify --statistics --output runs/verification.json
python -m unittest discover -s tests -v
```

`verify` is independent of GPU inference and confirms the released prediction summaries. It recalculates the primary confidence intervals and adjusted tests when `--statistics` is supplied. It does not validate inference by rerunning the models.

## Main experiment

Install PyTorch with CUDA support appropriate to the machine, then the dependencies in `requirements/paper.txt`. Paper captures used PyTorch 2.14.1+cu132 on the local GPU and 2.6.0+cu124 on the server, with Transformers 5.18.0, scikit-learn 1.9.1 and NumPy 2.5.3. Floating-point kernels and hardware can affect individual predictions. New runs record their environment and checkpoint hashes.

```bash
python -m pip install -r requirements/paper.txt
python -m pip install -e . --no-deps
python -m egus prepare-data
python -m egus download-models --models qwen3 --models-root models
python -m egus run --config configs/paper.toml --models qwen3 --models-root models
```

Supported IDs: `qwen3`, `minicpm5`, `qwen25`, `qwen3_8b`, `qwen35_4b`, `qwen35_9b`. For an existing model store, pass its parent with `--models-root`. Directory names must match each checkpoint name in the configuration. A run verifies required checkpoint files before inference. Both Windows and Linux use the same experiment entry.

By default, all selected models complete fitting and validation selection before any selected model is evaluated. The phases can also be invoked explicitly:

```bash
python -m egus run --config configs/paper.toml --models qwen3 --stage fit
python -m egus run --config configs/paper.toml --models qwen3 --stage evaluate
```

Use identical arguments in both phases. An output directory is bound to its configuration and source hashes; changing methods, seeds or model selection requires a new `--output` directory. Interrupted captures, probe fits and counterfactual scores can resume. Do not edit source files while a run is active.

The complete all-layer run stores substantial activation caches, especially for the 8B/9B models. Reserve disk space for checkpoints and all seed captures. The default batch size is eight; use a new configuration/output if memory requires a smaller batch. This is a new numerical execution, and results should carry that configuration.

## FairSteer and KnowBias

Use a separate Linux Python 3.12 environment. Install a compatible CUDA PyTorch build (the baseline execution used PyTorch 2.6.0 with CUDA 12.4), then:

```bash
python -m pip install -r requirements/author-linux.txt
python -m pip install -e . --no-deps
python -m egus fetch-sources
python -m egus prepare-data
python -m egus download-models --models-root models
python -m egus run --config configs/author_baselines.toml --models-root models
```

This profile requires cuML; it will not substitute a CPU solver. KnowBias sources are fetched at the pinned revision and verified before use. GPU workloads run sequentially across checkpoints. The same `--models`, `--stage` and `--output` options apply to this profile.

## Outputs

Every run contains `plan.json` and a snapshot of the executing package. Per-model folders contain clean captures, event logs, fitted probes/directions, validation candidates, a selection lock, test predictions and metrics. Main aggregates are written to `summary.json`. Author-source KnowBias has one deterministic fit per model with its own selection and prediction files.

The main runner can execute from restored inputs without historical research modules or old caches. `benchmarks/` contains frozen reference outputs for comparison. Running a new configuration never overwrites these reference artifacts.
