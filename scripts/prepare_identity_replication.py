"""Audit previous prepared inputs and reserve unused native WinoBias families."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict,Counter
import json,re
from mechanism.identity_causal import RAW,family
from mechanism.closed_transfer import PRONOUN,canonical,digest,gender,rank
from mechanism.fairsteer_author import ROOT,rows,write,sha

OUT=ROOT/'data/prepared/identity_replication_20261015_v1'

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'manifest.json').exists():
        print('already prepared');return
    occs=sorted({s.strip() for g in ('male','female') for s in (RAW/f'{g}_occupations.txt').read_text().splitlines() if s.strip()},key=len,reverse=True)
    known=set();sources={};records=0
    for path in sorted((ROOT/'data/prepared').rglob('*.jsonl')):
        if OUT in path.parents:continue
        used=0
        for r in rows(path):
            if not isinstance(r,dict):continue
            context=r.get('context','')
            if context and (str(r.get('source_id','')).startswith('winobias') or r.get('dataset')=='winobias'):
                known.add(family(context,occs));used+=1
        if used:sources[str(path)]={'sha256':sha(path),'records':used};records+=used
    groups=defaultdict(list);excluded=Counter();native={}
    for typ in (1,2):
        for split in ('dev','test'):
            for g in ('pro','anti'):
                path=RAW/f'{g}_stereotyped_type{typ}.txt.{split}';native[str(path)]=sha(path)
                for line in path.read_text().splitlines():
                    num=int(line.split()[0]);marked=re.sub(r'^\d+\s+','',line)
                    spans=[s.strip() for s in re.findall(r'\[([^]]+)\]',marked)]
                    ps=[p for p in spans if PRONOUN.fullmatch(p)];ctx=marked.replace('[','').replace(']','')
                    ents=[o for o in occs if re.search(r'\b'+re.escape(o)+r'\b',ctx,re.I)]
                    f=family(ctx,occs)
                    if f in known:excluded['previously_materialized_family']+=1;continue
                    if len(ps)!=1 or len(PRONOUN.findall(ctx))!=1 or len(ents)!=2 or ps[0].lower() not in ('he','she','him','her'):
                        excluded['structure']+=1;continue
                    es=[s for s in spans if not PRONOUN.fullmatch(s)]
                    gold=[o for o in ents if len(es)==1 and re.search(r'\b'+re.escape(o)+r'\b',es[0],re.I)]
                    if len(gold)!=1:excluded['gold']+=1;continue
                    pid=f'fresh:type{typ}:{split}:'+digest(canonical(ctx))[:20]
                    groups[pid].append({'source_id':f'winobias:{split}:type{typ}:{g}:{num:04d}',
                        'context':ctx,'entities':sorted(ents),'gold':gold[0],'pronoun':ps[0],
                        'gender':gender(ps[0]),'stereotype':g=='pro','pair_id':pid,
                        'family_id':'fresh:'+digest(f)[:20],'occupation_pair':'|'.join(sorted(ents)),
                        'native_type':typ,'native_split':split,'template':f})
    usable=defaultdict(list)
    for pid,vs in groups.items():
        if len(vs)!=2 or {r['gender'] for r in vs}!={'male','female'} or len({r['gold'] for r in vs})!=1:
            excluded['incomplete_or_gold_mismatch']+=1;continue
        usable[vs[0]['family_id']].extend(vs)
    chosen=[]
    for f in sorted(usable,key=lambda f:rank(20261015,f)):
        if len(chosen)//2>=128:break
        chosen.extend(usable[f])
    assert not {r['template'] for r in chosen}&known
    dest=OUT/'unused_native.jsonl';dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen),encoding='utf-8')
    write(OUT/'unused_pool.json',{'families':len(usable),'pairs':sum(len(v) for v in usable.values())//2,
                                 'selected_families':len({r['family_id'] for r in chosen}),'selected_pairs':len(chosen)//2,
                                 'types':dict(Counter(r['native_type'] for r in chosen)),'exclusions':dict(excluded)})
    write(OUT/'manifest.json',{'created':__import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'audit_scope':'all previous prepared JSONL with WinoBias contexts, all native dev/test types; exclude entire delexicalized families',
        'known_families':len(known),'previous_materialized_records':records,'previous_prepared_sources':sources,
        'native_sources':native,'script_sha256':sha(Path(__file__)),'data_sha256':sha(dest),
        'status':'model-output-unseen native remainder, pending agent text review; not independent human annotation'})
    print(json.dumps(__import__('json').loads((OUT/'unused_pool.json').read_text()),indent=2))

if __name__=='__main__':main()
