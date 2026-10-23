"""Exploratory frozen-component transfer using the model's native nonthinking chat.

Added after raw-completion format problems; neither selection nor components change.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import re
import time
import numpy as np
import torch
from mechanism.identity_causal import Experiment,variants,metrics,comparison,INSTRUCTION,PRONOUN
from mechanism.identity_answer_parser import parse_answer_line
from mechanism.fairsteer_author import write,read,sha


def strict_parse(text,choices):
    t=re.sub(r'^the\s+','',text.strip().strip(' \t\r\n.,:;!?\"\'').lower())
    return next((i for i,x in enumerate(choices) if t==x.lower()),-1)


def main():
    e=Experiment();e.prepare();out=e.out/'native_chat_generation';out.mkdir(exist_ok=True)
    declaration=out/'plan_frozen.json'
    identity={'script':sha(Path(__file__)),'parser':sha(Path('mechanism/identity_answer_parser.py')),
              'selection':sha(e.out/'selection_frozen.json')}
    if declaration.exists():assert read(declaration)['identity']==identity
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,
          'status':'exploratory adapter correction after raw completion outputs and native closed confirmation inspected',
          'template':'tokenizer native chat template, enable_thinking=False, one user message',
          'generation':'greedy max32 new tokens, EOS stopping, batch8, prompt-prefill edit only',
          'primary_scoring':'whole-completion exact occupation; first-answer-line secondary; no reselection'})
    e.setup();selected=e.selected_spec();vs=[variants(r) for r in e.values['confirmation']]
    results={};preds={};records=[]
    for method,spec in [('baseline',None),('selected',selected),('instruction',None)]:
        folder=out/method;folder.mkdir(exist_ok=True);generated=[]
        for start in range(0,len(vs),e.c['batch_size']):
            batch=vs[start:start+e.c['batch_size']];path=folder/f'{start:06d}.json'
            if path.exists():texts=read(path)
            else:
                prompts=[]
                for r in batch:
                    prefix=INSTRUCTION.replace('Output only the answer letter.','Output only the occupation name.') if method=='instruction' else ''
                    content=prefix+r['context']+' '+r['question']+'\nAnswer with only the occupation name.'
                    prompts.append(e.tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                   add_generation_prompt=True,enable_thinking=False))
                enc=e.tok(prompts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
                offsets=enc.pop('offset_mapping');positions=[]
                for i,r in enumerate(batch):
                    m=PRONOUN.search(r['context']);st=prompts[i].index(r['context'])+m.start();en=st+len(m.group())
                    token=[j for j,(a,b) in enumerate(offsets[i].tolist()) if b>a and a<en and b>st]
                    assert token;positions.append([token[-1],enc.input_ids.shape[1]-1])
                pos=torch.tensor(positions,device='cuda');inputs=enc.to('cuda');handles=[]
                if spec and spec['operation']!='none':
                    edit=e.edit_hook(spec['site'],pos,spec['operation'],spec['strength'],spec.get('mean'),spec.get('direction'))
                    state={'prefill':True}
                    def once(module,args,output):
                        if state['prefill']:state['prefill']=False;return edit(module,args,output)
                        return output
                    handles.append(e.module(spec['site']).register_forward_hook(once))
                try:
                    with torch.inference_mode():
                        output=e.model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                                               pad_token_id=e.tok.pad_token_id,eos_token_id=e.tok.eos_token_id)
                    texts=e.tok.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
                finally:
                    for h in handles:h.remove()
                write(path,texts)
            generated.extend(texts)
            if start%(e.c['batch_size']*10)==0:e.log('chat_generation_progress',method=method,done=start+len(batch),total=len(vs))
        strict=[strict_parse(t,r['choices']) for r,t in zip(vs,generated)]
        answerline=[parse_answer_line(t,r['choices']) for r,t in zip(vs,generated)]
        preds[method]={'strict':strict,'answer_line':answerline}
        results[method]={'strict':metrics(vs,predictions=strict),'answer_line':metrics(vs,predictions=answerline),
                         'strict_unparsed':float(np.mean(np.array(strict)<0)),
                         'line_unparsed':float(np.mean(np.array(answerline)<0))}
        for r,t,p,q in zip(vs,generated,strict,answerline):
            records.append({'method':method,'uid':r['uid'],'raw':t,'label':r['label'],'strict_prediction':p,'answer_line_prediction':q})
    contrasts={}
    for method in ('selected','instruction'):
        contrasts[method]={parse:{field:comparison(vs,preds[method][parse],preds['baseline'][parse],field,4000,e.c['seed'])
                                 for field in ('accuracy','both_correct','flip')} for parse in ('strict','answer_line')}
    write(out/'results.json',{'metrics':results,'comparisons':contrasts,'status':'exploratory native-chat adapter transfer; no retraining'})
    (out/'predictions.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in records),encoding='utf-8')
    e.log('chat_generation_complete',results=results)
    del e.model;gc.collect();torch.cuda.empty_cache()
    write(out/'complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'gpu_allocated':torch.cuda.memory_allocated()})


if __name__=='__main__':main()
