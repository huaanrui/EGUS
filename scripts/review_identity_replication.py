"""Agent text review before any model outcomes; preserve native text and labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.fairsteer_author import rows,write,sha
from scripts.prepare_identity_replication import OUT
import json

REJECT={0:'trustful vs trustworthy changes intended explanation',5:'subject verb agreement: often forget',
        6:'missing for: pay the vodka',7:'stressful person vs stressed; contextual interpretation unclear',
        54:'subject verb agreement: she need',76:'a bad breath',86:'subject verb agreement: he want'}

if __name__=='__main__':
    values=rows(OUT/'unused_native.jsonl');excluded=[];kept=[]
    badfamilies={values[2*i]['family_id'] for i in REJECT}
    for i in range(0,len(values),2):
        pair=values[i:i+2];reason='Type2 remainder contains previously reviewed malformed or out-of-scope items; exclude entire type uniformly' if pair[0]['native_type']==2 else REJECT.get(i//2)
        if not reason and pair[0]['family_id'] in badfamilies:reason='same family as rejected text'
        if reason:excluded.append({'pair_id':pair[0]['pair_id'],'reason':reason})
        else:kept.extend(pair)
    p=OUT/'confirmation.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept),encoding='utf-8')
    write(OUT/'text_review.json',{'reviewer':'agent text review, no independent human annotation',
        'pairs':len(kept)//2,'families':len({r['family_id'] for r in kept}),
        'native_type':1,'status':'native contextual/commonsense labels, not uniformly explicit evidence',
        'excluded':excluded,'confirmation_sha256':sha(p),'review_script_sha256':sha(Path(__file__))})
    print(len(kept)//2,len({r['family_id'] for r in kept}))
