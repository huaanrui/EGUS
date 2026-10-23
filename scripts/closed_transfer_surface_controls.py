"""Predeclared source-only surface controls; no target fitting/selection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/closed_transfer_20261014_v2'
DATA=ROOT/'data/prepared/closed_transfer_20261014_v2'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').split('\n') if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def surface(x):
    c,q=x['context'],x['question']
    return [len(c),len(q),len(c.split()),len(q.split()),c.count('.'),c.count(','),c.count(';'),
            c.count('!'),sum(z.isdigit() for z in c),sum(z.isupper() for z in c),q.count('?')]


def main():
    out=OUT/'surface_controls';out.mkdir(exist_ok=True)
    source=ROOT/'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl'
    split=ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json'
    identity={'source_sha256':sha(source),'split_sha256':sha(split),'data_manifest_sha256':sha(DATA/'manifest.json'),
              'script_sha256':sha(Path(__file__)),'supervision':'Only BBQ fit questions; one row/question; MMLU excluded. This differs from head training budget, explicitly diagnostic.',
              'fit_parameters':'L2 lbfgs C1 max_iter10000 tol1e-4; surface standardized; question TFIDF 1-2grams max_features3000 min_df2; no target hyperparameter selection.'}
    path=out/'plan_frozen.json'
    if path.exists():assert read(path)==identity
    else:path.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    allrows=rows(source);ids=read(split)['bbq_indices']['fit'];train=[allrows[i] for i in ids]
    y=np.array([x['context_condition']=='ambig' for x in train])
    models={'surface':make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4)),
            'question_tfidf':make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=3000,min_df=2),LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4))}
    result={'source_rows':len(train),'source_classes':dict(Counter(map(str,y))), 'evaluation':'AUROC/AUPRC only; no target threshold selection.'}
    with threadpool_limits(limits=4):
        for name,model in models.items():
            start=time.time();model.fit(np.array([surface(x) for x in train]) if name=='surface' else [x['question'] for x in train],y)
            joblib.dump(model,out/f'{name}.joblib');scores={}
            for key in ('synth_test','synth_calibration'):
                target=[x for x in rows(DATA/f'{key}.jsonl') if x['permutation']==0];labels=np.array([x['condition']=='ambig' for x in target])
                inputs=np.array([surface(x) for x in target]) if name=='surface' else [x['question'] for x in target]
                margin=model.decision_function(inputs);prob=model.predict_proba(inputs)[:,1]
                scores[key]={'rows':len(target),'auroc':float(roc_auc_score(labels,margin)),'auprc':float(average_precision_score(labels,margin)),
                             'margin_quantiles':np.quantile(margin,[0,.1,.5,.9,1]).tolist()}
                np.savez(out/f'{name}_{key}.npz',uids=np.array([x['uid'] for x in target]),margin=margin,probability=prob)
            result[name]={'seconds':time.time()-start,'scores':scores}
    result['identity']=identity
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
