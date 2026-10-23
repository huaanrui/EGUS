"""Matched OFAT deltas from the frozen second-round screening panel."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.data import write_json


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = tomllib.loads((ROOT / 'configs/experiment_round2.toml').read_text(encoding='utf-8'))
    out = ROOT / c['output_dir']
    rows = json.loads((out / 'screen_results.json').read_text(encoding='utf-8'))
    grid = json.loads((out / 'fine_grid_frozen.json').read_text(encoding='utf-8'))
    records = []
    anchors = {}
    for spec in grid:
        method, layer = spec['method'], spec['layers'][0]
        anchor = next(r for r in rows if r['spec']['method'] == method and r['spec']['axis'] == 'anchor' and r['spec']['layers'] == [layer])
        anchors[(method, layer)] = anchor
        changed = next(r for r in rows if r['spec']['id'] == spec['id'])
        records.append(dict(method=method, axis=spec['axis'], layer=layer, layers=','.join(map(str, spec['layers'])), pool=spec['pool'], scope=spec['scope'], operation=spec['operation'], strength=spec['strength'], k=spec['k'], anchor_id=anchor['spec']['id'], candidate_id=spec['id'], feasible=changed['feasible'], delta_macro_stereotype_pp=100*(changed['bbq']['macro_stereotype']-anchor['bbq']['macro_stereotype']), delta_macro_counterstereotype_pp=100*(changed['bbq']['macro_counterstereotype']-anchor['bbq']['macro_counterstereotype']), delta_macro_ambiguous_accuracy_pp=100*(changed['bbq']['macro_ambiguous_accuracy']-anchor['bbq']['macro_ambiguous_accuracy']), delta_disambiguated_accuracy_pp=100*(changed['bbq']['disambiguated_accuracy']-anchor['bbq']['disambiguated_accuracy']), delta_stereotype_counter_margin=changed['bbq']['macro_stereotype_counter_margin']-anchor['bbq']['macro_stereotype_counter_margin']))
    with (out / 'ablation_deltas.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    methods = ('paired_dim', 'paired_cav', 'paired_relp')
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, method in enumerate(methods):
        for j, (axis, field, label) in enumerate((('extraction', 'delta_macro_stereotype_pp', 'Fit pooling: stereotype change (pp)'), ('scope', 'delta_macro_stereotype_pp', 'Intervention scope: stereotype change (pp)'), ('scope', 'delta_stereotype_counter_margin', 'Intervention scope: S - CS logit margin'))):
            group = [r for r in records if r['method'] == method and r['axis'] == axis]
            key = 'pool' if axis == 'extraction' else 'scope'
            order = ['last1', 'last2', 'last4', 'last8', 'question_end', 'identity'] + (['all'] if key == 'scope' else [])
            values = {r[key]: r[field] for r in group}
            values['last1'] = 0.
            x = [k for k in order if k in values]
            y = [values[k] for k in x]
            ax = axes[i, j]
            ax.plot(range(len(x)), y, marker='o', linewidth=1.5)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set_xticks(range(len(x)), x, rotation=25, ha='right', fontsize=8)
            ax.set_title(method + '\n' + label, fontsize=10)
            ax.grid(alpha=.2)
    fig.suptitle('Screening OFAT deltas relative to each method anchor; lower is better\nExploratory development results, not held-out generalization', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / 'ablation_factors.png', dpi=180)
    plt.close(fig)
    write_json(out / 'ablation_summary.json', dict(matched_comparisons=len(records), note='Each fine configuration differs from the same-method same-layer anchor along its declared axis. Operators and layer sets are whole-family comparisons, not equal-energy interventions. Screening only; no test reselection.', by_method={m: dict(comparisons=sum(r['method'] == m for r in records), largest_macro_stereotype_reduction_pp=max(-r['delta_macro_stereotype_pp'] for r in records if r['method'] == m)) for m in {r['method'] for r in records}}))
    print(f'Matched ablation comparisons: {len(records)}', flush=True)


if __name__ == '__main__':
    main()
