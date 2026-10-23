from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json
import numpy as np
from mechanism.identity_combo import Experiment,metrics
from mechanism.fairsteer_author import read,write,rows,sha,ROOT


def relative_trace(e):
    values=e.values['rank'];groups=defaultdict(dict)
    for i,r in enumerate(values):groups[r['pair_id']][r['gender']]=i
    male=np.array([r['male'] for r in groups.values()]);female=np.array([r['female'] for r in groups.values()]);out={}
    for mode in ('raw','instruction'):
        c=np.load(e.out/f'rank_{mode}.npz');lines=[]
        for l in range(28):
            d={'layer':l}
            for name,a in [('before',c['before'][:,l]),('after',c['after'][:,l]),('half',c['before'][:,l]+.5*c['mlp'][:,l])]:
                denominator=(np.linalg.norm(a[male],axis=1)+np.linalg.norm(a[female],axis=1))/2
                d[name]=float(np.mean(np.linalg.norm(a[male]-a[female],axis=1)/np.maximum(denominator,1e-8)))
            lines.append(d)
        out[mode]=lines
    write(e.out/'relative_identity_trace.json',out);return out


def main():
    e=Experiment();e.prepare();r=read(e.out/'results.json');g=read(e.out/'chat_results.json');trace=relative_trace(e)
    plan=read(e.out/'plan_frozen.json');selection=read(e.out/'selection_frozen.json');neutral=read(e.out/'neutral_diagnostics.json')
    lines=['# Bounded combination and identity-mechanism results','',
      'Historical WinoBias type2 / Winogender have been inspected previously. Their results are exploratory retests. New native WinoBias TYPE1 development160 is an unseen-output transfer check with native contextual/commonsense labels; it is not uniformly unambiguous explicit evidence.',
      '',f"Counts: {plan['counts']}",f"Selected configurations: {json.dumps({k:v['spec'] if v else None for k,v in selection['selected'].items()},ensure_ascii=False)}",'',
      '| Dataset | Method | Accuracy | Both genders correct | Identity flip | Pro-anti gap |',
      '|---|---|---:|---:|---:|---:|']
    for split,taskdata in r['metrics'].items():
        for name,m in taskdata['coref'].items():
            lines.append('| '+split+' | '+name+' | '+' | '.join(f'{100*m[k]:.2f}%' for k in ('accuracy','both_correct','flip','gap'))+' |')
    lines+=['','Native chat short generation (historical, frozen configs):',
            '| Method | Strict accuracy | Both correct | Flip | Unparsed |','|---|---:|---:|---:|---:|']
    for name,d in g['metrics'].items():
        m=d['strict'];lines.append(f"| {name} | {100*m['accuracy']:.2f}% | {100*m['both_correct']:.2f}% | {100*m['flip']:.2f}% | {100*d['unparsed']:.2f}% |")
    lines+=['','Identity/control accuracies:', '| Method | Literal reading | Gender classification | Identity binding | Role exchange |', '|---|---:|---:|---:|---:|']
    for name in r['metrics']['historical']['coref']:
        lines.append('| '+name+' | '+' | '.join(f"{100*r['metrics']['historical'][t][name]['accuracy']:.2f}%" for t in ('literal','reading','binding','role_swap'))+' |')
    lines+=['','Layerwise normalized male-female differences (rank only):',
            '| Layer | Before MLP | After MLP | MLP half-scaled |', '|---|---:|---:|---:|']
    for x in trace['raw']:
        if x['layer'] in (0,5,6,7,8,9):lines.append(f"| {x['layer']} | {x['before']:.4f} | {x['after']:.4f} | {x['half']:.4f} |")
    lines+=['','Norm growth and identity differences are not semantic bias attribution. The old suffix-neutral donor was causally inaccessible to the pronoun. Corrected prefix-neutral diagnostics:',json.dumps(neutral,indent=2),
            '', 'Comparisons against previous layer7 single:',json.dumps(r['comparisons_vs_old_l7'],indent=2),
            '', 'Native chat differences:',json.dumps(g['comparisons_vs_old_l7'],indent=2)]
    if (e.out/'interaction.json').exists():lines+=['','Constituent interaction:',json.dumps(read(e.out/'interaction.json'),indent=2)]
    if (e.out/'mechanism_panel.json').exists():lines+=['','Fixed descriptive mechanism contrasts:',json.dumps(read(e.out/'mechanism_panel.json'),indent=2)]
    if (e.out/'occupation_interaction_diagnostics.json').exists():lines+=['','Occupation-context diagnostics (post-selection exploratory):',json.dumps(read(e.out/'occupation_interaction_diagnostics.json'),indent=2)]
    if (e.out/'auxiliary_comparisons.json').exists():lines+=['','Secondary comparisons:',json.dumps(read(e.out/'auxiliary_comparisons.json'),indent=2)]
    if (e.out/'mechanism_panel_comparisons.json').exists():lines+=['','Mechanistic descriptive contrast uncertainty:',json.dumps(read(e.out/'mechanism_panel_comparisons.json'),indent=2)]
    neuron=read(e.out/'mechanism_trace.json')
    lines+=['',f"Neuron-control RMS: top128={neuron['top128_rms_mean']:.4f}, matched128={neuron['matched128_rms_mean']:.4f}. Nearest-neighbor control is only approximate; this mismatch limits claims of channel specificity.",
            '', 'Limits: one model; repeated old selection splits; historical retests not independent; agent data audit; no verified circuit edges or broad free-generation claim.']
    (ROOT/'research/2026-10-16_identity_combo_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:35]))


if __name__=='__main__':main()
