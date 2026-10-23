from __future__ import annotations

import torch

from .attribution import replacement_backward
from .interventions import Intervention


def check_runner(runner, row, artifacts, anchor):
    p = runner.builder.build(row, 0)
    base = runner.forward(p).detach().float()
    probe = dict(method="dim", direction_key="dim:last1", layers=[anchor], pool="last1", scope="last1", operation="project", reference="mean", strength=0.)
    runner.intervention = Intervention(probe, artifacts)
    zero = runner.forward(p).detach().float()
    runner.intervention = None
    restored = runner.forward(p).detach().float()
    assert torch.equal(base, zero), "Zero intervention changes logits"
    assert torch.equal(base, restored), "Removing intervention fails to restore logits"
    probe = dict(method="random_neurons", ranking_key="random_neurons:last1", layers=[27], pool="last1", scope="nonlast", k=256, operation="remove", strength=1., reference="zero")
    runner.intervention = Intervention(probe, artifacts)
    nonlast = runner.forward(p).detach().float()
    nonlast_stats = runner.intervention.summary()
    runner.intervention = None
    assert torch.equal(base, nonlast), "Non-last final-layer MLP intervention changes first answer logits"
    with replacement_backward(runner.model):
        replaced = runner.forward(p).detach().float()
    assert torch.equal(base, replaced), "RelP replacement changes forward values"
    # Cache parity for prefill. Main scoring always uses fresh cache-free forwards.
    inputs = torch.tensor([p["ids"]], device=runner.device)
    with torch.inference_mode():
        cached = runner.model(input_ids=inputs, use_cache=True, logits_to_keep=1).logits[0, -1].float()
    assert torch.equal(base, cached), "Cache/no-cache prefill mismatch"
    # Test a nonzero last-position intervention actually reaches the logits.
    active = dict(method="dim", direction_key="dim:last1", layers=[anchor], pool="last1", scope="last1", operation="add", reference="mean", strength=1.)
    runner.intervention = Intervention(active, artifacts)
    changed = runner.forward(p).detach().float()
    runner.intervention = None
    difference = float((base - changed).abs().max())
    assert difference > 0, "Positive intervention control has no effect"
    prompts = [runner.builder.build(row, rot) for rot in range(3)]
    batched_base = runner.forward_batch(prompts).detach().float()
    runner.intervention = Intervention(dict(active, strength=0.), artifacts)
    batched_zero = runner.forward_batch(prompts).detach().float()
    runner.intervention = None
    assert torch.equal(batched_base, batched_zero), "Zero intervention changes batched logits"
    runner.intervention = Intervention(probe, artifacts)
    batched_nonlast = runner.forward_batch(prompts).detach().float()
    runner.intervention = None
    assert torch.equal(batched_base, batched_nonlast), "Batched token mask mixes examples"
    return dict(zero_strength_exact=True, hook_restore_exact=True, final_layer_nonlast_exact=True, nonlast_modified_positions=nonlast_stats["modified_positions"], replacement_forward_exact=True, prefill_cache_parity_exact=True, nonzero_intervention_max_logit_change=difference, batched_zero_exact=True, batched_nonlast_exact=True, fit_batch_size=1, evaluation_rotation_batch_size=runner.rotation_batch_size, padding="none", generation="not implemented; this runner scores the first answer token only")
