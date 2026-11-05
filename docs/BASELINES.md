# Baseline implementations and author code

## Paper baselines

`configs/author_baselines.toml` runs the FairSteer and KnowBias implementations used for the paper's baseline rows. `configs/paper.toml` runs EGUS and evidence-gated steering. Both use the same question manifests, validation set and test set. The final baseline run completed all six checkpoints and passed its independent audit; its predictions are included in `benchmarks/main/`.

FairSteer receives the common BBQ training pool without additional MMLU questions. All blocks and the declared strength grid are searched on validation accuracy. These settings adapt the authors' method to our evaluation protocol and do not reproduce their original published benchmark scores.

Evidence steering reuses EGUS's evidence target and threshold rule, replacing direct selection with activation addition. Independently selected methods and shared-gate controls are distinguished in the released summaries.

## Pinned author sources

| Method | Upstream | Revision | Used code |
|---|---|---|---|
| FairSteer | https://github.com/LiYichen99/FairSteer | `0545cdbaa71b3301030accf1b98130820e08026f` | `train_probes`, `mean_difference`, `lt_modulated_vector_add` |
| KnowBias | https://github.com/JP-25/KnowBias | `ccfd3daea0067707cbda6238839206c194794881` | `knowledge_neurons.py`, `patch.py`, concept-question CSV |

FairSteer's MIT-licensed files are included unchanged in `third_party/FairSteer/`. The adapter extracts the named functions without executing model-loading scripts or modifying their source. Native `train_probes` calls cuML logistic regression with the author's defaults and `max_iter=10000`. The edit callback is applied per activated batch row because the original last-token callback checks only the first row. Activation masks are read from clean states before editing.

KnowBias source is downloaded on demand. No explicit upstream repository license was found in the pinned snapshot, so it is not bundled or relicensed as part of EGUS. Review upstream terms for your intended use.

```bash
python -m egus fetch-sources
python -m egus run --config configs/author_baselines.toml --models qwen3 --models-root models
```

Use a separate Linux Python 3.12 environment with `requirements/author-linux.txt` and a compatible CUDA PyTorch build. The main paper environment and the native cuML environment have different NumPy/scikit-learn constraints. The released FairSteer rows use cuML; there is no scikit-learn fallback for FairSteer. EGUS uses scikit-learn logistic regression.

## KnowBias adaptation details

The adapter calls the original integrated-gradients and neuron-enhancement functions at `gate_proj`, using the last prompt position. The five concepts are expanded across gender/race/religion into 45 questions. Identification uses 20 integration endpoints, integration batch four, threshold 0.1 and consistency values 0.05/0.1. Neuron sets are united across dimensions. Enhancement strengths are 1, 1.5, 2, 3.5 and 5; one is the explicit no-op candidate. Consistency and strength are selected using validation accuracy with sufficient-context tie breaking. This deterministic identification is performed once per model, not counted as five independent fits.

The referenced upstream evaluation prompt file `evaluate_know_new.txt` is absent. The adapter declares its replacement: `Answer the following question with only yes or no.\nQuestion: {question}`. Its chat prompt disables model thinking, while BBQ evaluation uses the common raw prompt. Attribution is computed in BF16 to avoid FP16 underflow for small target probabilities; BBQ evaluation uses FP16. The Qwen3.5 wrapper's language-model paths are adapted to the architecture. These are declared adaptations, not a claim of bitwise reproduction of the original published scores.
