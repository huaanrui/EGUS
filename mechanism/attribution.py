"""RelP local replacement backward rules, adapted to the installed Qwen3.

Forward values are preserved. Freeze attention Q/K paths (therefore attention
weights), RMSNorm denominators and SiLU sigmoids; halve both product gradients.
Reference: https://arxiv.org/html/2601.22594v2 (Table 2 / Appendix B).
This implements node rankings, not neuron-to-neuron edges or a full circuit.
"""
from __future__ import annotations

import contextlib
import types

import numpy as np
import torch
from torch.nn import functional as F

from .data import POOLS, example_weights


class FrozenSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(torch.sigmoid(x))
        return F.silu(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.saved_tensors[0]


class HalfProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad * b * 0.5, grad * a * 0.5


@contextlib.contextmanager
def replacement_backward(model):
    originals = []

    def patch(module, fn):
        originals.append((module, module.forward))
        module.forward = types.MethodType(fn, module)

    for module in model.modules():
        if type(module).__name__ == "Qwen3RMSNorm":
            original = module.forward

            def norm(self, x, _original=original):
                actual = _original(x)
                scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon).detach()
                linear = self.weight * (x.float() * scale).to(x.dtype)
                # Exact original forward; gradients follow the local linear replacement.
                return linear + (actual - linear).detach()
            patch(module, norm)
        elif type(module).__name__ == "Qwen3MLP":
            def mlp(self, x):
                return self.down_proj(HalfProduct.apply(FrozenSiLU.apply(self.gate_proj(x)), self.up_proj(x)))
            patch(module, mlp)
        elif type(module).__name__ == "Qwen3Attention":
            for projection in (module.q_proj, module.k_proj):
                original = projection.forward

                def frozen(self, x, _original=original):
                    return _original(x).detach()
                patch(projection, frozen)
    try:
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original


def node_scores(runner, rows, layers, role_means, mode="relp", progress=None, sample_weights=None):
    """Signed (activation - fit reference) * local gradient, then token/row mean.

    The target is stereotype-logit minus unknown-logit. This is an unpaired,
    mean-reference adaptation; source-paper paired SVA circuit claims do not apply.
    """
    result = {pool: {l: np.zeros(runner.model.config.intermediate_size, dtype=np.float64) for l in layers} for pool in POOLS}
    weights = example_weights(rows) if sample_weights is None else np.asarray(sample_weights)
    original_logits = runner.forward(runner.builder.build(rows[0], 0)).detach().float()
    context = replacement_backward(runner.model) if mode == "relp" else contextlib.nullcontext()
    with context:
        replacement_logits = runner.forward(runner.builder.build(rows[0], 0)).detach().float()
        max_error = float((original_logits - replacement_logits).abs().max())
        if max_error != 0:
            raise AssertionError(f"Replacement forward differs by {max_error}")
        try:
            for j, row in enumerate(rows):
                for rot in range(3):
                    p = runner.builder.build(row, rot)
                    runner.capture = {}
                    logits = runner.forward(p, grad=True)
                    perm = p["permutation"]
                    stereotype = runner.builder.label_ids[perm.index(row["stereotype_index"])]
                    unknown = runner.builder.label_ids[perm.index(row["unknown_index"])]
                    metric = logits[stereotype].float() - logits[unknown].float()
                    acts = [runner.capture["mlp"][l] for l in layers]
                    grads = torch.autograd.grad(metric, acts)
                    for layer, act, grad in zip(layers, acts, grads):
                        reference = torch.as_tensor(role_means["mlp"][layer][p["roles"]], device=act.device)
                        importance = (act[0].detach().float() - reference) * grad[0].detach().float()
                        for pool in POOLS:
                            result[pool][layer] += weights[j] * importance[p["masks"][pool]].mean(0).cpu().numpy() / 3
                    runner.capture = None
                    del logits, metric, acts, grads
                if progress and (j + 1) % 8 == 0:
                    progress(j + 1, len(rows))
        finally:
            runner.capture = None
    return dict(scores=result, replacement_forward_max_error=max_error, mode=mode, fit_uids=[r["uid"] for r in rows])
