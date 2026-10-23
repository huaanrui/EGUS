from __future__ import annotations

import contextlib
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import POOLS, PromptBuilder


class Runner:
    def __init__(self, path: str, max_tokens=512, device="cuda", rotation_batch_size=3):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this local run")
        self.device = device
        self.rotation_batch_size = rotation_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.builder = PromptBuilder(self.tokenizer, max_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32, attn_implementation="sdpa").to(device).eval()
        self.model.requires_grad_(False)
        self.layers = self.model.model.layers
        self.current = None
        self.intervention = None
        self.capture = None
        self.handles = []

    def install(self, layers: list[int]):
        self.remove_hooks()
        for layer in layers:
            block = self.layers[layer]
            self.handles.append(block.register_forward_hook(self._block_hook(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self._mlp_hook(layer)))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture(self, kind, layer, tensor):
        if self.capture is not None:
            self.capture.setdefault(kind, {})[layer] = tensor

    def _block_hook(self, layer):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self._capture("residual", layer, h)
            delta = h - args[0]
            self._capture("update", layer, delta)
            if self.intervention is not None:
                changed = self._apply("residual", layer, h, update=delta)
                if changed is not h:
                    return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return hook

    def _mlp_hook(self, layer):
        def hook(module, args):
            a = args[0]
            self._capture("mlp", layer, a)
            if self.intervention is not None:
                changed = self._apply("mlp", layer, a)
                if changed is not a:
                    return (changed, *args[1:])
        return hook

    def _apply(self, kind, layer, h, update=None):
        if layer not in self.intervention.spec["layers"]:
            return h
        if isinstance(self.current, list):
            outputs = []
            for i, prompt in enumerate(self.current):
                outputs.append(self.intervention.apply(kind, layer, h[i:i+1], prompt, update=None if update is None else update[i:i+1]))
            return torch.cat(outputs, dim=0)
        return self.intervention.apply(kind, layer, h, self.current, update=update)

    def forward_batch(self, prompts):
        """Batch only equal-length rotations of one question, with no padding."""
        if len({len(p["ids"]) for p in prompts}) != 1:
            raise ValueError("Rotation token lengths differ; padding is intentionally unsupported")
        self.current = prompts
        inputs = torch.tensor([p["ids"] for p in prompts], device=self.device)
        with torch.inference_mode():
            return self.model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[:, -1]

    def forward(self, prompt, grad=False):
        self.current = prompt
        inputs = torch.tensor([prompt["ids"]], device=self.device)
        with torch.enable_grad() if grad else torch.inference_mode():
            if grad:
                embedded = self.model.get_input_embeddings()(inputs).detach().requires_grad_(True)
                out = self.model(inputs_embeds=embedded, use_cache=False, logits_to_keep=1)
            else:
                out = self.model(input_ids=inputs, use_cache=False, logits_to_keep=1)
            return out.logits[0, -1]

    def evaluate(self, rows, intervention=None, progress=None):
        self.intervention = intervention
        results = []
        start = time.perf_counter()
        try:
            for j, row in enumerate(rows):
                rotations = []
                prompts = [self.builder.build(row, rot) for rot in range(3)]
                if self.rotation_batch_size == 3:
                    all_logits = self.forward_batch(prompts)
                elif self.rotation_batch_size == 1:
                    all_logits = [self.forward(p) for p in prompts]
                else:
                    raise ValueError("rotation_batch_size must be 1 or 3")
                for p, raw_logits in zip(prompts, all_logits):
                    logits = raw_logits.float()
                    selected = logits[self.builder.label_ids]
                    logprobs = selected.log_softmax(0).cpu().numpy()
                    semantic = [0.0] * 3
                    for label, source in enumerate(p["permutation"]):
                        semantic[source] = float(logprobs[label])
                    rotations.append(dict(prediction=int(p["permutation"][int(selected.argmax())]), semantic_logprobs=semantic, candidate_mass=float(torch.softmax(logits, 0)[self.builder.label_ids].sum())))
                results.append(dict(uid=row["uid"], rotations=rotations))
                if progress and (j + 1) % 32 == 0:
                    progress(j + 1, len(rows), time.perf_counter() - start)
        finally:
            self.intervention = None
        return dict(predictions=results, seconds=time.perf_counter() - start, rotation_batch_size=self.rotation_batch_size, perturbation=intervention.summary() if intervention else None)

    def collect(self, rows, layers, progress=None):
        """Fit-only features. Token reduction precedes CPU transfer and row averaging."""
        self.install(layers)
        feature = {kind: {pool: {l: [] for l in layers} for pool in POOLS} for kind in ("residual", "mlp", "mlp_abs", "update")}
        # Role means are averaged within each example first, preventing long inputs dominating.
        role_sum = {kind: {l: np.zeros((16, self.model.config.intermediate_size if kind == "mlp" else self.model.config.hidden_size), dtype=np.float64) for l in layers} for kind in ("residual", "mlp", "update")}
        role_count = np.zeros(16)
        from .data import example_weights
        weights = example_weights(rows)
        predictions = []
        start = time.perf_counter()
        try:
            for j, row in enumerate(rows):
                rowfeat = {kind: {pool: {l: [] for l in layers} for pool in POOLS} for kind in feature}
                rotations = []
                for rot in range(3):
                    p = self.builder.build(row, rot)
                    self.capture = {}
                    logits = self.forward(p).float()
                    scores = logits[self.builder.label_ids]
                    logprobs = scores.log_softmax(0).cpu().tolist()
                    semantic = [0.0] * 3
                    for label, source in enumerate(p["permutation"]):
                        semantic[source] = logprobs[label]
                    rotations.append(dict(prediction=int(p["permutation"][int(scores.argmax())]), semantic_logprobs=semantic, candidate_mass=float(torch.softmax(logits, 0)[self.builder.label_ids].sum())))
                    for kind in ("residual", "mlp", "update"):
                        for l in layers:
                            h = self.capture[kind][l][0].float()
                            for pool in POOLS:
                                positions = p["masks"][pool]
                                rowfeat[kind][pool][l].append(h[positions].mean(0).cpu().numpy())
                                if kind == "mlp":
                                    rowfeat["mlp_abs"][pool][l].append(h[positions].abs().mean(0).cpu().numpy())
                            for role in np.unique(p["roles"]):
                                values = h[np.flatnonzero(p["roles"] == role)].mean(0).cpu().numpy()
                                role_sum[kind][l][role] += weights[j] * values / 3
                    for role in np.unique(p["roles"]):
                        role_count[role] += weights[j] / 3
                    self.capture = None
                for kind in feature:
                    for pool in POOLS:
                        for l in layers:
                            feature[kind][pool][l].append(np.mean(rowfeat[kind][pool][l], axis=0))
                predictions.append(dict(uid=row["uid"], rotations=rotations))
                if progress and (j + 1) % 32 == 0:
                    progress(j + 1, len(rows), time.perf_counter() - start)
        finally:
            self.capture = None
        for kind in feature:
            for pool in POOLS:
                for l in layers:
                    feature[kind][pool][l] = np.asarray(feature[kind][pool][l], dtype=np.float32)
        means = {kind: {l: (v / np.maximum(role_count[:, None], 1e-12)).astype(np.float32) for l, v in layer_map.items()} for kind, layer_map in role_sum.items()}
        return dict(features=feature, role_means=means, weights=weights, predictions=predictions, seconds=time.perf_counter() - start)
