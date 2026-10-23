"""Post-hoc format correction. Original whole-completion metrics remain intact."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
from mechanism.identity_causal import Experiment,variants,metrics,comparison
from mechanism.identity_answer_parser import parse_answer_line
from mechanism.fairsteer_author import read,rows,write,sha


def main():
    e=Experiment();e.prepare();vs=[variants(r) for r in e.values['confirmation']]
    write(e.out/'answer_line_rescore_declaration.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
       'parser_sha256':sha(Path('mechanism/identity_answer_parser.py')),
       'status':'post-hoc format correction after reviewing raw samples; no model/parameter reselection',
       'rule':'first nonempty line only; exact occupation or anchored pronoun-refers-to sentence; never search subsequent lines or use gold'})
    data=rows(e.out/'generation_predictions.jsonl');methods={}
    for method in ('baseline','selected'):methods[method]={r['uid']:r['raw'] for r in data if r['method']==method}
    instruction=e.out/'instruction_generation_secondary.json'
    if instruction.exists():methods['instruction']={r['uid']:r['raw'] for r in read(instruction)['raw_predictions']}
    result={};preds={};records=[]
    for method,saved in methods.items():
        p=[parse_answer_line(saved[r['uid']],r['choices']) for r in vs];preds[method]=p
        result[method]={'metrics':metrics(vs,predictions=p),'unparsed':float(np.mean(np.array(p)<0))}
        for r,q in zip(vs,p):records.append({'method':method,'uid':r['uid'],'prediction':q,'label':r['label'],'raw':saved[r['uid']]})
    comps={}
    for method in preds:
        if method=='baseline':continue
        comps[method]={field:comparison(vs,preds[method],preds['baseline'],field,4000,e.c['seed']) for field in ('accuracy','both_correct','flip')}
    write(e.out/'answer_line_generation_results.json',{'status':'post-hoc conservative answer-line scoring','metrics':result,'comparisons':comps})
    (e.out/'answer_line_generation_predictions.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in records),encoding='utf-8')
    print(result);print(comps)


if __name__=='__main__':main()
