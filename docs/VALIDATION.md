# Release validation

This release contains the current paper's selected-layer experiments and the completed native author baselines.

- All 276 numeric cells in the six result/configuration tables were checked against the current manuscript. `tables.json` records these cells; full-precision metrics remain in `summary.json`.
- CPU verification recalculates 152,064 saved predictions, including six single-fit KnowBias outputs. It checks per-seed scores, aggregates, matched-gate results and all 12 clustered statistical contrasts.
- The five 2,200-question training samples, 352 validation questions and 704 test questions were restored from pinned upstream files and verified by hash. Family separation and the 110 direction questions (ten per category) passed the data audit.
- The original six-model author-baseline run completed and passed its independent audit. `benchmarks/main/provenance.json` records its audit status and source-artifact hashes. The published FairSteer predictions use cuML.
- Seventeen CPU tests check threshold boundaries, unknown recognition, training order, author-probe train/validation separation, all-layer selection, matched-gate anchoring, cache identity, selection locks, source integrity and published tables. Lint checks cover the maintained package, tests and packaging script.
- The packaged release is tested in an isolated directory without the private research archive, downloaded models or historical modules.

This cleanup does not rerun GPU inference on all six checkpoints. Released predictions verify the reported results; a fresh end-to-end numerical reproduction can differ with hardware and dependency versions. The two environment profiles and their base-prediction differences are documented in `PROTOCOL.md` and `REPRODUCING.md`.
