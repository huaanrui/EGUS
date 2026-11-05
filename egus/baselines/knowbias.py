"""Native KnowBias identification and enhancement under the shared BBQ protocol."""

import csv
from pathlib import Path
import numpy as np
import torch

from .author import knowbias_module
from ..config import ROOT
from ..io import read, write, rows, sha
from ..metrics import summarize_prediction


def questions(concepts):
    path = ROOT / "third_party/KnowBias/data/know_bias_data_all_combine_extend.csv"
    with path.open(encoding="utf-8-sig") as f:
        source = list(csv.DictReader(f))
    result = []
    for dimension in ("gender", "race", "religion"):
        for row in source:
            if row["concept"] not in concepts:
                continue
            text = row["question"].replace(
                "gender, race, or  religious affiliation",
                {"gender": "gender", "race": "race", "religion": "religious affiliation"}[
                    dimension
                ],
            )
            text = text.replace(
                "gender/race/religious group",
                {"gender": "gender", "race": "race", "religion": "religious group"}[dimension],
            )
            text = text.replace(
                "gender/race/religious bias",
                {"gender": "gender bias", "race": "racial bias", "religion": "religious bias"}[
                    dimension
                ],
            )
            result.append(
                {
                    "dimension": dimension,
                    "concept": row["concept"],
                    "question": text,
                    "target": row["gold"],
                }
            )
    assert len(result) == len(concepts) * 9
    return result


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class KnowBias:
    def __init__(self, engine, config):
        self.engine, self.c, self.out = engine, config, engine.out
        self.native = knowbias_module()

    def adapter(self):
        kn = self.native.KnowledgeNeurons(
            self.engine.model, Tokenizer(self.engine.tokenizer), model_type="Qwen"
        )
        if hasattr(self.engine.model.model, "language_model"):
            kn.transformer_layers_attr = "model.language_model.layers"
            kn.word_embeddings_attr = "model.language_model.embed_tokens.weight"
        return kn

    def fit(self):
        destination = self.out / "knowledge_sets.json"
        if destination.exists():
            return read(destination)
        self.engine.load(torch.bfloat16)
        kn = self.adapter()
        knowledge = questions(self.c["kb_concepts"])
        arrays = []
        for i, question in enumerate(knowledge):
            path = self.out / "knowledge_attributions" / f"{i:03d}.npz"
            if not path.exists():
                text = (
                    "Answer the following question with only yes or no.\nQuestion: "
                    + question["question"]
                )
                scores = []
                for layer in range(len(self.engine.layers)):
                    score = (
                        kn.get_scores_for_layer(
                            text,
                            question["target"],
                            layer,
                            steps=self.c["ig_steps"],
                            batch_size=self.c["ig_batch"],
                        )
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    assert np.isfinite(score).all() and np.any(score != 0), (i, layer)
                    scores.append(score)
                path.parent.mkdir(exist_ok=True)
                with path.with_suffix(".tmp").open("wb") as f:
                    np.savez(f, scores=np.stack(scores))
                path.with_suffix(".tmp").replace(path)
            with np.load(path) as saved:
                arrays.append(saved["scores"])
            self.engine.log("knowledge_attribution", done=i + 1, total=len(knowledge))
        sets = {}
        for beta in self.c["kb_consistency"]:
            union = set()
            for dimension in ("gender", "race", "religion"):
                group = torch.from_numpy(
                    np.stack([a for a, q in zip(arrays, knowledge) if q["dimension"] == dimension])
                )
                union.update(
                    map(
                        tuple,
                        self.native.fix_p_neuron(group, threshold=self.c["kb_threshold"], p=beta),
                    )
                )
            sets[str(beta)] = [list(node) for node in sorted(union)]
        write(destination, sets)
        del kn
        return sets

    def score(self, values, nodes, strength):
        kn = self.adapter()
        modified = kn.enhance(nodes, val=strength) if nodes else []
        try:
            scores = np.concatenate(
                [
                    self.engine.forward(values[i : i + self.c["batch_size"]])[0]
                    for i in range(0, len(values), self.c["batch_size"])
                ]
            )
        finally:
            self.native.unpatch_ff_layers(
                self.engine.model,
                modified,
                transformer_layers_attr=kn.transformer_layers_attr,
                ff_attrs=kn.input_ff_attr,
            )
        return scores

    def select(self):
        path = self.out / "knowbias_selection.json"
        if path.exists():
            return
        sets = self.fit()
        self.engine.load(torch.float16)
        validation = rows(Path(self.c["data_dir"]) / "validation.jsonl")
        candidates = []
        for beta, nodes, strength in [(1.0, [], 1.0)] + [
            (float(b), nodes, strength)
            for b, nodes in sets.items()
            for strength in self.c["kb_strengths"]
            if strength != 1.0
        ]:
            scores = self.score(validation, nodes, strength)
            candidates.append(
                {
                    "consistency": beta,
                    "strength": strength,
                    "nodes": nodes,
                    "metrics": summarize_prediction(
                        validation, scores.argmax(1), np.zeros(len(validation), bool)
                    ),
                }
            )
        selected = max(
            candidates,
            key=lambda v: (
                v["metrics"]["correct"],
                v["metrics"]["clear_correct"],
                -abs(v["strength"] - 1),
                v["consistency"],
            ),
        )
        write(
            path,
            {
                "selected": selected,
                "candidates": candidates,
                "knowledge_sets_sha256": sha(self.out / "knowledge_sets.json"),
            },
        )

    def evaluate(self):
        path = self.out / "knowbias_selection.json"
        fingerprint = sha(path)
        selected = read(path)["selected"]
        self.engine.load(torch.float16)
        test = rows(Path(self.c["data_dir"]) / "test.jsonl")
        scores = self.score(test, selected["nodes"], selected["strength"])
        assert sha(path) == fingerprint
        np.savez_compressed(
            self.out / "knowbias_predictions.npz",
            scores=scores,
            pred=scores.argmax(1),
            uids=np.array([r["uid"] for r in test]),
        )
        write(
            self.out / "knowbias_metrics.json",
            summarize_prediction(test, scores.argmax(1), np.zeros(len(test), bool)),
        )
