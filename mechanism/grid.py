"""One-factor-at-a-time branches. Every configuration is saved before screening."""
from __future__ import annotations


def make_grid(config):
    anchor = config["anchor_layer"]
    result = []

    def add(method, axis, **kw):
        s = dict(method=method, axis=axis, layers=[anchor], pool="last1", scope="last1", operation="remove", strength=1.0, reference="mean")
        if method in {"dim", "cav", "random_direction"}:
            s["operation"] = "project"
        elif method != "layer_span":
            s["k"] = 64
        s.update(kw)
        if method in {"dim", "cav", "random_direction"}:
            s["direction_key"] = f'{method}:{s["pool"]}'
        elif method != "layer_span":
            s["ranking_key"] = f'{method}:{s["pool"]}'
        s["id"] = f'{method}__{len([x for x in result if x["method"] == method]):02d}'
        result.append(s)

    for method in ("dim", "cav"):
        for layer in config["search_layers"]:
            add(method, "layer", layers=[layer])
        for pool in ("last2", "last4", "last8", "user"):
            add(method, "extraction", pool=pool)
        for scope in ("last2", "last4", "last8", "all"):
            add(method, "intervention_scope", scope=scope)
        for strength in (-1.0, 1.0):
            add(method, "operator", operation="add", strength=strength)
        add(method, "strength", strength=.5)
    for method in ("sn_consistency", "neuron_contrast", "relp_nodes"):
        for k in (16, 64, 256):
            add(method, "component_budget", k=k)
        for pool in ("last4", "user"):
            add(method, "extraction", pool=pool)
        for scope in ("last4", "all"):
            add(method, "intervention_scope", scope=scope)
        add(method, "strength", strength=.5)
        add(method, "retain_diagnostic", operation="retain")
    for method in ("activation_magnitude", "random_neurons"):
        add(method, "ranking_control")
    add("random_direction", "ranking_control")
    for width in (1, 2, 4):
        add("layer_span", "component_budget", layers=list(range(anchor, anchor + width)))
    for layer in (7, 23):
        add("layer_span", "layer", layers=[layer])
    for scope in ("last4", "all"):
        add("layer_span", "intervention_scope", scope=scope)
    # Explicit universe control for the keep-only neuron diagnostic.
    add("random_neurons", "retain_diagnostic", operation="retain")
    return result
