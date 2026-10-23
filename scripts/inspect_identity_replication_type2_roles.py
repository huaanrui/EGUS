"""Read native mention-order support before designing a balanced probe."""
from pathlib import Path
import sys,re,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
from mechanism.identity_causal import RAW
from mechanism.closed_transfer import PRONOUN
from mechanism.fairsteer_author import ROOT,write,sha

if __name__=='__main__':
    occs=sorted({x.strip() for g in ('male','female') for x in (RAW/f'{g}_occupations.txt').read_text().splitlines() if x.strip()},key=len,reverse=True);result={}
    for split in ('dev','test'):
        count=Counter();first=[]
        for group in ('pro','anti'):
            for line in (RAW/f'{group}_stereotyped_type2.txt.{split}').read_text().splitlines():
                marked=re.sub(r'^\d+\s+','',line)
                spans=[x.strip() for x in re.findall(r'\[([^]]+)\]',marked)]
                context=marked.replace('[','').replace(']','')
                entities=[o for o in occs if re.search(r'\b'+re.escape(o)+r'\b',context,re.I)]
                goldspan=[x for x in spans if not PRONOUN.fullmatch(x)]
                gold=[o for o in entities if len(goldspan)==1 and re.search(r'\b'+re.escape(o)+r'\b',goldspan[0],re.I)]
                if len(entities)!=2 or len(gold)!=1:count['ambiguous_entity_or_gold']+=1;continue
                order=sorted(entities,key=lambda o:re.search(r'\b'+re.escape(o)+r'\b',context,re.I).start())
                ps=PRONOUN.findall(context);key='second' if gold[0]==order[1] else 'first'
                count['gold_'+key]+=1
                if len(ps)==1:count['single_pronoun_gold_'+key]+=1
                if key=='first':first.append({'group':group,'line':line})
        result[split]={'counts':dict(count),'gold_first_examples':first[:20]}
    write(ROOT/'results/identity_replication_20261015_v1/native_type2_mention_order_audit.json',{
        'distribution':result,'script_sha256':sha(Path(__file__)),
        'input_sha256':{str(p):sha(p) for p in [RAW/f'{g}_stereotyped_type2.txt.{s}' for g in ('pro','anti') for s in ('dev','test')]},
        'interpretation':'descriptive native bracket-label audit; only uniquely extracted two-occupation rows; no claim every gold label manually reviewed'})
    print(json.dumps(result,indent=2))
