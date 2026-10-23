from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import re
from mechanism.fairsteer_author import ROOT,read,rows,sha,write
from mechanism.identity_causal import RAW,family
from mechanism.closed_transfer import PRONOUN,gender

folder=ROOT/'data/prepared/identity_replication_20261015_v1';manifest=read(folder/'manifest.json')
occs={s.strip() for g in ('male','female') for s in (RAW/f'{g}_occupations.txt').read_text().splitlines() if s.strip()}
known=set();materialized=0
for path,info in manifest['previous_prepared_sources'].items():
    assert sha(path)==info['sha256']
    for r in rows(path):
        if isinstance(r,dict) and r.get('context') and (str(r.get('source_id','')).startswith('winobias') or r.get('dataset')=='winobias'):
            known.add(family(r['context'],occs));materialized+=1
assert len(known)==manifest['known_families'];assert materialized==manifest['previous_materialized_records']
values=rows(folder/'confirmation.jsonl');pairs=defaultdict(list)
for r in values:
    assert family(r['context'],occs) not in known
    _,split,typ,g,num=r['source_id'].split(':');path=RAW/f'{g}_stereotyped_{typ}.txt.{split}'
    source=next(line for line in path.read_text().splitlines() if int(line.split()[0])==int(num))
    marked=re.sub(r'^\d+\s+','',source);assert marked.replace('[','').replace(']','')==r['context']
    spans=re.findall(r'\[([^]]+)\]',marked);goldspan=next(s for s in spans if not PRONOUN.fullmatch(s))
    assert re.search(r'\b'+re.escape(r['gold'])+r'\b',goldspan,re.I)
    assert r['gender']==gender(r['pronoun']);pairs[r['pair_id']].append(r)
for vs in pairs.values():
    assert len(vs)==2 and {r['gender'] for r in vs}=={'male','female'}
    assert len({r['gold'] for r in vs})==1
write(folder/'independent_verification.json',{'prior_prepared_records_reconstructed':materialized,
    'prior_families':len(known),'confirmation_pairs':len(pairs),'confirmation_families':len({r['family_id'] for r in values}),
    'no_prior_prepared_delexicalized_family_overlap':True,'native_text_and_labels_exact':True,
    'annotation_limit':'structural and agent review, no independent human annotation'})
print('data audit passed',len(pairs))
