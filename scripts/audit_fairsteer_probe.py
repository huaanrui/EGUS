"""Describe frozen detector behavior by shot/condition, without modifying selection."""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/fairsteer_20261012_v1"
DATA=ROOT/"data/prepared/bbq_fairsteer_20261012_v1"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trusted_pickle(name):
    path=OUT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==load_json(path.with_suffix(path.suffix+".sha256.json"))["sha256"]
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    fitted=trusted_pickle("fitted.pkl")
    v=trusted_pickle("probe_features_probe_validation.pkl")
    lookup={r["uid"]:r for r in (json.loads(line) for line in (DATA/"probe_validation.jsonl").read_text(encoding="utf-8").splitlines())}
    strata=collections.defaultdict(list)
    for i,m in enumerate(v["metadata"]):
        if m["source"]=="mmlu":
            strata["mmlu"].append(i)
        else:
            r=lookup[m["uid"]]
            shot="fewshot" if m["shots"] else "zeroshot"
            for name in ("bbq",shot,r["context_condition"],shot+"_"+r["context_condition"],r["category"]):
                strata[name].append(i)
    results=[]
    with threadpool_limits(limits=4):
        for layer,p in enumerate(fitted["probes"]):
            pred=(v["x"][:,layer]@p["coef"]+p["intercept"]>=0).astype(int)
            detail={}
            for name,indices in strata.items():
                y=v["y"][indices];pp=pred[indices]
                neg=y==0;pos=y==1
                recall=float(np.mean(pp[neg]==0)) if neg.any() else None
                fpr=float(np.mean(pp[pos]==0)) if pos.any() else None
                detail[name]=dict(n=len(indices),biased=int(neg.sum()),unbiased=int(pos.sum()),accuracy=float(np.mean(pp==y)),biased_recall=recall,false_positive_rate=fpr,balanced_accuracy=(recall+1-fpr)/2 if recall is not None and fpr is not None else None)
            results.append(dict(layer=layer,strata=detail))
    report=dict(scope="Frozen family-held-out detector only. Excludes other wrong BBQ answers, includes author-defined MMLU unbiased controls; native order only. Descriptive, no reselection or refit.",layers=results)
    (OUT/"probe_stratified_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    for layer in (0,7,13,20,27):
        print(json.dumps(results[layer],ensure_ascii=False))


if __name__=="__main__":main()
