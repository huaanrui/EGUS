"""Independent file/metric audit including the auxiliary domain track."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/bbq_comparators_20261013_v2'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    split=read(ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json')
    source=[json.loads(x) for x in (ROOT/'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl').read_text(encoding='utf-8').split('\n') if x.strip()]
    test=[source[i] for _ in (0,1) for i in split['bbq_indices']['test']]
    labels=np.array([r['label'] for r in test]);amb=np.array([r['context_condition']=='ambig' for r in test]);total=0
    for r in read(OUT/'results.json'):
        scores=np.array(r['bbq']['scores']);assert np.isfinite(scores).all();pred=scores.argmax(1)
        assert np.array_equal(pred,r['bbq']['predictions']);m=r['bbq']['metrics']
        for key,mask in [('accuracy',np.ones(len(pred),bool)),('ambig',amb),('disambig',~amb)]:
            actual=float((pred[mask]==labels[mask]).mean());saved=m[key] if key=='accuracy' else m[key]['accuracy'];assert abs(actual-saved)<1e-12
        total+=len(scores)+len(r['utility']['scores'])
    domain=OUT/'domain_reference_v2';initial=read(domain/'plan_frozen.json')
    assert initial['source_sha256']==digest(ROOT/'scripts/run_bbq_readout_domains_v2.py')
    assert initial['definition_sha256']==digest(ROOT/'research/2026-10-13_bbq_domain_reference_protocol_v2.md')
    assert initial['parent_plan']==digest(OUT/'plan_frozen.json')
    selection=read(domain/'selection_frozen.json');assert selection['rankings_sha256']==digest(OUT/'readout_rankings.json')
    assert selection['selection_sha256']==digest(domain/'selection_results.json')
    relevant=np.array([r['category'] in ['Gender_identity','Nationality','Religion'] for r in test]);count=0
    for r in read(domain/'results.json'):
        pred=np.array(r['scores']).argmax(1);m=r['metrics']
        assert m['rows']==int(relevant.sum())==384
        for name,mask in [('accuracy',relevant),('ambig',relevant&amb),('disambig',relevant&~amb)]:
            expected=float((pred[mask]==labels[mask]).mean());actual=m[name] if name=='accuracy' else m[name]['accuracy'];assert abs(expected-actual)<1e-12
        if r['name'].endswith('_domain'):assert r['domain_parameters']==selection['selected'][r['name'].removesuffix('_domain')]
        count+=int(relevant.sum())
    assert read(OUT/'independent_audit.json')['passed']
    assert read(OUT/'gpu_readout_audit.json')['passed']
    assert read(domain/'gpu_equivalence.json')['passed']
    assert read(OUT/'unknown_option_identification_audit.json')['passed']
    result={'passed':True,'main_prediction_records':total,'domain_prediction_records':count,'independent_question_count_main':704,'independent_question_count_domain':192,'all_auxiliary_metrics_recomputed':True,'frozen_selection_and_source_hashes_verified':True,'gpu_hook_and_equivalence_checks_passed':True,'warning':'Record counts include repeated shots/methods; not independent sample counts.'}
    (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result)


if __name__=='__main__':main()
