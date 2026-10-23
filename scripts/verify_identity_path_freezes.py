"""Final source/supplement fingerprint and numerical-validation audit."""
from pathlib import Path
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/identity_path_20261015_v3'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    plan=read(OUT/'plan_frozen.json')
    for name,value in {**plan['implementation'],**plan['source_inputs']}.items():assert sha(Path(name))==value,name
    supplements={}
    for name in ('active_controls','token_routes','receiver_edges','qkv_validation'):
        declaration=read(OUT/(name+'_declaration.json'));identity=declaration['identity']
        script=ROOT/'scripts'/('identity_path_'+name+'.py')
        assert sha(script)==identity['script'],name
        assert sha(OUT/'selection_frozen.json')==identity['selection'],name
        if 'protocol' in identity:
            protocol=ROOT/'research'/('2026-10-15_identity_path_'+name+'.md')
            assert sha(protocol)==identity['protocol'],name
        if (OUT/(name+'_plan.json')).exists():
            early=read(OUT/(name+'_plan.json'))
            assert early['script_sha256']==identity['script']
            assert early['protocol_sha256']==identity['protocol']
        supplements[name]=True
    q=read(OUT/'qkv_validation.json');assert q['passed'] and q['max_error']<q['candidate_logit_tolerance']
    v=read(OUT/'verification.json');v.update({'supplement_source_hashes_match':supplements,
         'manual_QKV_node_patch_equivalent':True,'manual_QKV_max_logit_error':q['max_error'],
         'manual_QKV_validated_rows_times_heads':q['rows']*len(q['heads'])})
    write(OUT/'verification.json',v)
    note='\n\nFinal fingerprint/numerical validation:\n\n'+json.dumps(v,indent=2)+'\n'
    report=ROOT/'research/2026-10-15_identity_path_findings.md'
    with report.open('a',encoding='utf-8') as f:f.write(note)
    write(OUT/'completed_followup.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),
       'main_and_supplements_complete':True,'verification_sha256':sha(OUT/'verification.json'),
       'report_sha256':sha(report),'all_behavior_sets_historically_inspected':True})
    print(json.dumps(v,indent=2))


if __name__=='__main__':main()
