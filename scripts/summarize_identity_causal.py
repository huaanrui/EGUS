from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json
import numpy as np
from mechanism.identity_causal import Experiment,metrics,variants,comparison
from mechanism.fairsteer_author import read,rows,write,sha,ROOT


def main():
    e=Experiment();e.prepare();r=read(e.out/'results.json');g=read(e.out/'generation_results.json')
    pred=rows(e.out/'predictions.jsonl');lookup=defaultdict(dict)
    for x in pred:lookup[(x['split'],x['task'],x['method'])][x['uid']]=x
    checked=0
    for (split,task,method),saved in lookup.items():
        vs=e.task_values(split,task);p=np.array([saved[x['uid']]['prediction'] for x in vs]);m=metrics(vs,predictions=p)
        for field in ('accuracy','both_correct','flip','gap'):
            assert abs(m[field]-r['metrics'][split][task][method][field])<1e-12
        checked+=len(vs)
    # Joint native/role-swapped correctness: four contexts per gender pair/order.
    joint={}
    for method in r['metrics']['confirmation']['coref']:
        task_ok=[]
        for task in ('coref','role_swap'):
            vs=e.task_values('confirmation',task);ps=lookup[('confirmation',task,method)]
            task_ok.append({(x['pair_id'],x['gender'],x['permutation']):ps[x['uid']]['prediction']==x['label'] for x in vs})
        groups=defaultdict(list)
        for key,ok in task_ok[0].items():groups[(key[0],key[2])].extend([ok,task_ok[1][key]])
        joint[method]=float(np.mean([all(v) for v in groups.values()]))
    # Identity facts cannot be solved simply by always selecting the last fact.
    binding_vs=e.task_values('confirmation','binding')
    fact_order_counts=defaultdict(int)
    for x in binding_vs:
        fact_order_counts['woman_last' if x['context'].endswith('is a woman.') else 'woman_first']+=1
    generated=rows(e.out/'generation_predictions.jsonl');gvs=[variants(x) for x in e.values['confirmation']]
    gp={method:{x['uid']:x for x in generated if x['method']==method} for method in ('baseline','selected')}
    generation_comparisons={}
    for parse,field in [('strict','strict_prediction'),('unique_entity','unique_entity_prediction')]:
        for method in gp:
            p=[gp[method][x['uid']][field] for x in gvs];gm=metrics(gvs,predictions=p)
            for k in ('accuracy','both_correct','flip','gap'):assert abs(gm[k]-g[method][parse][k])<1e-12
        generation_comparisons[parse]={k:comparison(gvs,[gp['selected'][x['uid']][field] for x in gvs],
            [gp['baseline'][x['uid']][field] for x in gvs],k,4000,e.c['seed']) for k in ('accuracy','both_correct')}
    write(e.out/'generation_comparisons.json',generation_comparisons)
    plan=read(e.out/'plan_frozen.json')
    assert plan['identity']['code']==sha(ROOT/'mechanism/identity_causal.py')
    for path,h in plan['input_sha256'].items():assert sha(Path(path))==h
    write(e.out/'verification.json',{'recomputed_predictions':checked,'metrics_match':True,'generation_metric_records_checked':len(generated),
        'source_inputs_unchanged':True,'code_unchanged':True,'binding_fact_order_counts':dict(fact_order_counts),
        'joint_native_and_role_correct':joint})
    lines=['# Identity interference pilot results','',
      'Agent text review; not independent human annotation. Native WinoBias development families are newly evaluated; Winogender is previously inspected exploratory validation.',
      '',f"Data: {plan['counts']}",f"Selected: {r['selection']['name']} {r['selection']['spec']}",'',
      '| Method | Native accuracy | Both genders correct | Identity flip | Reading | Explicit identity binding | Role-swap accuracy | Both native and swapped evidence correct |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    main=r['metrics']['confirmation']
    for method,m in main['coref'].items():
        cells=[m['accuracy'],m['both_correct'],m['flip'],main['reading'][method]['accuracy'],main['binding'][method]['accuracy'],main['role_swap'][method]['accuracy'],joint[method]]
        lines.append('| '+method+' | '+' | '.join(f'{100*x:.2f}%' for x in cells)+' |')
    lines+=['','Selected versus baseline, family cluster bootstrap:',json.dumps(r['comparisons']['confirmation'],ensure_ascii=False,indent=2),
      '', 'Short actual generation:',json.dumps(g,ensure_ascii=False,indent=2),
      '', 'Short generation differences:',json.dumps(generation_comparisons,ensure_ascii=False,indent=2),'',
      f'Independent recomputation checked {checked} item predictions. Identity-fact clause order: {dict(fact_order_counts)}.',
      '', 'Limits: one model, agent annotation, simple explicit fact controls; native coreference role exchange does not establish universal evidence following. Oracle donor recovery cannot establish a deployable bias correction.']
    line_path=e.out/'answer_line_generation_results.json'
    if line_path.exists():
        line=read(line_path)
        lines+=['','## Answer-line format correction (post-hoc, exploratory)',
          'The original whole-completion strict scores are all zero because generations continue after the first answer line; they are NOT evidence of zero semantic task accuracy. Preserve original metrics, and separately apply exact first-nonempty-line occupation matching or an anchored pronoun-refers-to sentence. The parser never scans subsequent lines or uses gold.',
          '', '| Method | Answer-line accuracy | Both genders correct | Unparsed |', '|---|---:|---:|---:|']
        for method,d in line['metrics'].items():
            lines.append(f"| {method} | {100*d['metrics']['accuracy']:.2f}% | {100*d['metrics']['both_correct']:.2f}% | {100*d['unparsed']:.2f}% |")
        lines+=['',json.dumps(line['comparisons'],ensure_ascii=False,indent=2)]
    literal_path=e.out/'literal_reading_results.json'
    if literal_path.exists():
        literal=read(literal_path);lines+=['','Literal pronoun reading: secondary control added after selection-set metalinguistic performance inspection, with no reselection.',
                                       json.dumps(literal,ensure_ascii=False,indent=2)]
    chat_path=e.out/'native_chat_generation/results.json'
    if chat_path.exists():
        chat=read(chat_path);chat_rows=rows(e.out/'native_chat_generation/predictions.jsonl')
        cvs=[variants(x) for x in e.values['confirmation']]
        for method in ('baseline','selected','instruction'):
            byuid={x['uid']:x for x in chat_rows if x['method']==method}
            p=[byuid[x['uid']]['strict_prediction'] for x in cvs]
            cm=metrics(cvs,predictions=p)
            for k in ('accuracy','both_correct','flip','gap'):assert abs(cm[k]-chat['metrics'][method]['strict'][k])<1e-12
        lines+=['','## Native-chat short generation (exploratory)',
                'Native tokenizer chat template, enable_thinking=False, greedy max32, frozen component/strength, prefill-only intervention. Primary parsing is whole-completion exact occupation. This adapter condition was added after raw-generation format problems; it is not a newly untouched confirmation dataset.',
                '', '| Method | Strict accuracy | Both genders correct | Identity flip | Unparsed |', '|---|---:|---:|---:|---:|']
        for method,d in chat['metrics'].items():
            m=d['strict'];lines.append(f"| {method} | {100*m['accuracy']:.2f}% | {100*m['both_correct']:.2f}% | {100*m['flip']:.2f}% | {100*d['strict_unparsed']:.2f}% |")
        lines+=['',json.dumps(chat['comparisons'],ensure_ascii=False,indent=2)]
    if (e.out/'confirmation_patch_results.json').exists():
        patch=read(e.out/'confirmation_patch_results.json');lines+=['','Oracle patch confirmation:',json.dumps(patch,ensure_ascii=False,indent=2)]
    dest=ROOT/'research/2026-10-15_identity_causal_findings.md';dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:18]));print('comparisons',json.dumps(r['comparisons']['confirmation'],ensure_ascii=False))


if __name__=='__main__':main()
