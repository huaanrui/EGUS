"""Read-only paired comparisons beyond the predeclared combo-vs-old contrast."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from mechanism.identity_combo import Experiment,task_row
from mechanism.identity_causal import comparison,variants,metrics
from mechanism.fairsteer_author import rows,write,read


def main():
    e=Experiment();e.prepare();saved=rows(e.out/'predictions.jsonl');lookup=defaultdict(dict)
    for r in saved:lookup[(r['split'],r['task'],r['method'])][r['uid']]=r['prediction']
    result={}
    for split in ('historical','new_type1','winogender'):
        vs=e.task_values(split,'coref');r={}
        for a,b in [('old_l7','baseline'),('best_combo','baseline'),('combo_instruction','instruction')]:
            if (split,'coref',a) not in lookup:continue
            aa=[lookup[(split,'coref',a)][v['uid']] for v in vs];bb=[lookup[(split,'coref',b)][v['uid']] for v in vs]
            r[a+'_vs_'+b]={k:comparison(vs,aa,bb,k,4000,e.c['seed']) for k in ('accuracy','both_correct','flip')}
        result[split]=r
    gen=rows(e.out/'chat_predictions.jsonl');gg=defaultdict(dict)
    for r in gen:gg[r['method']][r['uid']]=r['strict_prediction']
    vs=[variants(r) for r in e.values['historical']];result['chat']={}
    for a,b in [('old_l7','baseline'),('best_combo','baseline'),('combo_instruction','instruction')]:
        if a not in gg:continue
        result['chat'][a+'_vs_'+b]={k:comparison(vs,[gg[a][v['uid']] for v in vs],[gg[b][v['uid']] for v in vs],k,4000,e.c['seed']) for k in ('accuracy','both_correct','flip')}
    write(e.out/'auxiliary_comparisons.json',{'status':'secondary descriptive comparisons, not extra model selection','comparisons':result})
    panel=e.out/'mechanism_panel_predictions.jsonl'
    if panel.exists():
        pp=defaultdict(dict)
        for r in rows(panel):pp[(r['task'],r['method'])][r['uid']]=r['prediction']
        mechanistic={}
        for task in ('coref','literal'):
            vs=e.task_values('historical',task);mechanistic[task]={}
            for a,b in [('subspace_rank1','complement_rank1'),('neurons_top128_0.5','neurons_matched128_0.5')]:
                aa=[pp[(task,a)][v['uid']] for v in vs];bb=[pp[(task,b)][v['uid']] for v in vs]
                mechanistic[task][a+'_vs_'+b]={k:comparison(vs,aa,bb,k,4000,e.c['seed']) for k in ('accuracy','both_correct')}
        write(e.out/'mechanism_panel_comparisons.json',{'status':'descriptive, multiple contrasts unadjusted; RMS matching is approximate','comparisons':mechanistic})
    chat=read(e.out/'chat_results.json');checks=0
    for method,saved in gg.items():
        m=metrics(vs:=[variants(r) for r in e.values['historical']],predictions=[saved[v['uid']] for v in vs])
        for field in ('accuracy','both_correct','flip','gap'):assert abs(m[field]-chat['metrics'][method]['strict'][field])<1e-12
        checks+=len(vs)
    verify=read(e.out/'verification.json');verify['chat_prediction_records_recomputed']=checks
    verify['chat_metrics_match']=True;write(e.out/'verification.json',verify)
    print(result['new_type1']);print(result['chat'].get('combo_instruction_vs_instruction'))


if __name__=='__main__':main()
