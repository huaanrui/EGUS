from __future__ import annotations

import numpy as np
import torch


class Intervention:
    """Prefill-only operators; all means and vectors must come from fit."""
    def __init__(self, spec: dict, artifacts: dict, device="cuda"):
        self.spec = spec
        self.artifacts = artifacts
        self.device = device
        self.cache = {}
        self.energy = 0.0
        self.original_energy = 0.0
        self.positions = 0
        self.calls = 0

    def tensor(self, key, value):
        if key not in self.cache:
            self.cache[key] = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        return self.cache[key]

    def apply(self, kind, layer, h, prompt, update=None):
        s = self.spec
        if layer not in s["layers"] or float(s.get("strength", 1)) == 0:
            return h
        is_neuron = s["method"] in {"sn_consistency", "neuron_contrast", "activation_magnitude", "random_neurons", "relp_nodes", "relp_nodes_global", "gradient_nodes"}
        if (kind == "mlp") != is_neuron:
            return h
        positions = prompt["masks"][s["scope"]]
        if not positions:
            return h
        idx = torch.tensor(positions, device=h.device)
        value = h[0, idx].float()
        strength = float(s.get("strength", 1))
        if is_neuron:
            if s.get("global_budget", False):
                selected = np.flatnonzero(self.artifacts["global_ranks"][s["ranking_key"]][layer] < s["k"])
            else:
                selected = self.artifacts["rankings"][s["ranking_key"]][layer][:s["k"]]
            if len(selected) == 0 and s.get("operation") != "retain":
                return h
            selected = torch.as_tensor(selected, device=h.device)
            cols = torch.ones(value.shape[-1], device=h.device, dtype=torch.bool) if s.get("operation") == "retain" else torch.zeros(value.shape[-1], device=h.device, dtype=torch.bool)
            cols[selected] = s.get("operation") != "retain"
            mean = self.tensor(("mlp", layer), self.artifacts["role_means"]["mlp"][layer])[torch.as_tensor(prompt["roles"][positions], device=h.device)]
            target = torch.zeros_like(value) if s.get("reference") == "zero" else mean
            delta = strength * (target - value) * cols
        elif s["method"] == "layer_span":
            old_update = update[0, idx].float()
            mean = self.tensor(("update", layer), self.artifacts["role_means"]["update"][layer])[torch.as_tensor(prompt["roles"][positions], device=h.device)]
            target = torch.zeros_like(old_update) if s.get("reference") == "zero" else mean
            delta = strength * (target - old_update)
        else:
            fitted = self.artifacts["directions"][s["direction_key"]][layer]
            v = self.tensor((s["direction_key"], layer, "vector"), fitted["vector"])
            if s["operation"] == "add":
                scale = strength * fitted["sigma"]
                if s.get("energy_match", False):
                    scale /= np.sqrt(len(positions))
                delta = torch.ones((len(positions), 1), device=h.device) * scale * v[None]
            else:
                mean = self.tensor(("residual", layer), self.artifacts["role_means"]["residual"][layer])[torch.as_tensor(prompt["roles"][positions], device=h.device)]
                center = torch.zeros_like(value) if s.get("reference") == "zero" else mean
                delta = -strength * ((value - center) @ v)[:, None] * v[None]
        out = h.clone()
        changed = (value + delta).to(h.dtype)
        out[0, idx] = changed
        # Report the actual BF16 perturbation, including rounding, in the hook's own space.
        actual = changed.float() - value
        self.energy += float(actual.square().sum())
        self.original_energy += float(value.square().sum())
        self.positions += len(positions)
        self.calls += 1
        return out

    def summary(self):
        return dict(l2_total=float(np.sqrt(self.energy)), relative_l2=float(np.sqrt(self.energy / max(self.original_energy, 1e-12))), modified_positions=self.positions, hook_calls=self.calls, space="pre_down_mlp" if "ranking_key" in self.spec else "residual_stream", note="Neuron-space norms are not directly comparable to residual-space norms")
