"""Separate, versioned author-style per-domain FBA/BBA reference track.

Primary global-component experiment is untouched. Parameters selected on dev.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import types
import numpy as np
import torch
from mechanism.bbq_comparators import Comparators,DOMAINS,CATEGORY
from mechanism.fairsteer_author import ROOT,metrics,read,write,sha
from mechanism.fairsteer_improve_report import cluster_intervals


def subset_metrics(pilot,split,scores,domain=None):
    examples=pilot.examples(split)
    categories=[CATEGORY[domain]] if domain else list(CATEGORY.values())
    ix=[j for j,(r,_,_) in enumerate(examples) if r['category'] in categories]
    return metrics([examples[j][0] for j in ix],scores[ix].argmax(1))


def domain_scores(pilot,split,spec,domain,utility=False):
    """Identical scalar readout, only evaluate rows used in this domain."""
    pilot.setup();r=pilot.runner;cache=pilot.cache_readout(split,utility)
    ordering=read(pilot.out/'readout_rankings.json')[domain][spec['method']]
    indices=ordering[:int(len(ordering)*spec['ratio'])]
    rows=range(len(cache['scores'])) if utility else [j for j,(x,_,_) in enumerate(pilot.examples(split)) if x['category']==CATEGORY[domain]]
    result=cache['scores'].copy();letters='ABCD' if utility else 'ABC';ids=[r.tokenizer('Answer: '+a).input_ids[-1] for a in letters]
    with torch.inference_mode():
        for j in rows:
            h=torch.tensor(cache[spec['hook']][j],device='cuda').reshape(1,1,-1)
            h[:,:,indices]=spec['constant']
            if spec['hook']=='pre':h=r.model.model.norm(h)
            result[j]=r.model.lm_head(h)[0,0,ids].float().cpu().numpy()
    return result


def failures(m,u,base,ref,neutral):
    fail=[]
    if m['disambig']['accuracy']<max(base['disambig']['accuracy']-.01,ref['disambig']['accuracy']-.005)-1e-12:fail.append('disambig')
    if u['accuracy']<neutral-.02-1e-12:fail.append('utility')
    if m['ambig']['bias_score'] is None or ref['ambig']['bias_score'] is None or abs(m['ambig']['bias_score'])>abs(ref['ambig']['bias_score'])+.01+1e-12:fail.append('bias')
    return fail


def main():
    pilot=Comparators();out=pilot.out/'domain_reference_v2';out.mkdir(exist_ok=True)
    source=Path(__file__).resolve();definition=ROOT/'research/2026-10-13_bbq_domain_reference_protocol_v2.md'
    initial={'source_sha256':sha(source),'definition_sha256':sha(definition),'parent_plan':sha(pilot.out/'plan_frozen.json')}
    path=out/'plan_frozen.json'
    if path.exists():assert read(path)==initial
    else:write(path,initial)
    for p in (source,definition):
        target=out/'source_snapshot'/p.name
        if not target.exists():target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())
    pilot.setup()
    if not (out/'gpu_equivalence.json').exists():
        checks=[]
        for method,hook,constant in [('fba','post',0.),('bba','pre',-1.)]:
            spec={'method':method,'hook':hook,'ratio':.1,'constant':constant}
            for utility in (False,True):
                original=pilot.evaluate_readout('dev',spec,utility,domain='gender')
                optimized=domain_scores(pilot,'dev',spec,'gender',utility)
                ix=list(range(len(original))) if utility else [j for j,(r,_,_) in enumerate(pilot.examples('dev')) if r['category']==CATEGORY['gender']]
                assert np.array_equal(original[ix],optimized[ix])
                checks.append({'method':method,'hook':hook,'utility':utility,'rows':len(ix),'exact':True})
        write(out/'gpu_equivalence.json',{'passed':True,'checks':checks})
    dev=pilot.baseline('dev');reference=next(r for r in read(pilot.parent/'results.json') if r['name']=='fairsteer')
    # Reconstruct only dev reference outputs from frozen parent counterfactuals.
    refspec=reference['bbq']['spec'];features=np.asarray(pilot.features[[i for _,_,i in pilot.examples('dev')],18],np.float32);probe=np.load(pilot.parent/'probes.npz')
    gate=features@probe['original_coef']+probe['original_intercept']<0
    cf=np.load(pilot.parent/'counterfactuals/dev_dim_1.npz')['scores']
    refdev=dev.copy();refdev[gate]=cf[gate]
    neutral=pilot.score('dev',pilot.baseline('dev',True),True)['accuracy']
    selection_path=out/'selection_frozen.json'
    if selection_path.exists():selection=read(selection_path)
    else:
        selection={};all_candidates=[]
        for mode in ('fba','bba'):
            for hook in ('post','pre'):
                group=mode+'_'+hook;selection[group]={}
                for domain in DOMAINS:
                    candidates=[];base=subset_metrics(pilot,'dev',dev,domain);ref=subset_metrics(pilot,'dev',refdev,domain)
                    for ratio in pilot.c['readout_ratios']:
                        for constant in pilot.c['readout_constants']:
                            spec={'method':mode,'hook':hook,'ratio':ratio,'constant':constant}
                            b=domain_scores(pilot,'dev',spec,domain);u=domain_scores(pilot,'dev',spec,domain,True)
                            bm=subset_metrics(pilot,'dev',b,domain);um=pilot.score('dev',u,True);fail=failures(bm,um,base,ref,neutral)
                            item={'domain':domain,'spec':spec,'bbq':bm,'utility':um,'failures':fail};candidates.append(item);all_candidates.append(item)
                    valid=[r for r in candidates if not r['failures']]
                    chosen=max(valid or candidates,key=lambda r:(r['bbq']['accuracy'],r['bbq']['disambig']['accuracy'],-r['spec']['ratio']))
                    selection[group][domain]=chosen
                    pilot.log('domain_reference_selected',group=group,domain=domain,spec=chosen['spec'],failures=chosen['failures'])
        write(out/'selection_results.json',all_candidates)
        write(selection_path,{'selected':selection,'initial_sha256':sha(path),'selection_sha256':sha(out/'selection_results.json'),'rankings_sha256':sha(pilot.out/'readout_rankings.json')})
        selection=read(selection_path)
    assert selection['rankings_sha256']==sha(pilot.out/'readout_rankings.json')
    assert selection['selection_sha256']==sha(out/'selection_results.json')
    records=[]
    for group,domains in selection['selected'].items():
        scores=pilot.baseline('test').copy();utility=[]
        for domain,item in domains.items():
            counterfactual=domain_scores(pilot,'test',item['spec'],domain)
            ix=[j for j,(r,_,_) in enumerate(pilot.examples('test')) if r['category']==CATEGORY[domain]]
            scores[ix]=counterfactual[ix]
            utility.append(pilot.score('test',domain_scores(pilot,'test',item['spec'],domain,True),True)['accuracy'])
        rec={'name':group+'_domain','scores':scores.tolist(),'metrics':subset_metrics(pilot,'test',scores),'domain_parameters':domains,'utility_by_domain':dict(zip(DOMAINS,utility))}
        records.append(rec)
    parent=read(pilot.parent/'results.json')
    for name in ('evidence','fairsteer','baseline'):
        rec=next(r for r in parent if r['name']==name);scores=np.asarray(rec['bbq']['scores'])
        records.insert(0,{'name':name,'scores':scores.tolist(),'metrics':subset_metrics(pilot,'test',scores),'utility_by_domain':{'same_method':rec['utility']['metrics']['accuracy']}})
    examples=pilot.examples('test');ix=[j for j,(r,_,_) in enumerate(examples) if r['category'] in CATEGORY.values()]
    proxy=types.SimpleNamespace(c=pilot.c,examples=lambda split:[examples[j] for j in ix]);ev=next(r for r in records if r['name']=='evidence');ep=np.asarray(ev['scores'])[ix].argmax(1)
    lines=['# BBA/FBA 按类别参考轨道','','仅 Gender_identity / Nationality / Religion，384 条 zero/few-shot 输出，使用各自类别归因与 dev 选参。类别信息用于路由，是额外信息；不能当作任意未知输入的部署策略。','','|方法|总体/%|歧义/%|消歧/%|相对证据门控/pp|95%家族区间/pp|dev失败类别|','|---|---:|---:|---:|---:|---|---|']
    for rec in records:
        m=rec['metrics'];pred=np.asarray(rec['scores'])[ix].argmax(1);ci=cluster_intervals(proxy,pred,ep)['overall'];delta=100*(m['accuracy']-ev['metrics']['accuracy']);rec['delta_pp_vs_evidence']=delta;rec['ci_pp_vs_evidence']=ci
        fail={d:r['failures'] for d,r in rec.get('domain_parameters',{}).items() if r['failures']}
        lines.append(f"|{rec['name']}|{100*m['accuracy']:.2f}|{100*m['ambig']['accuracy']:.2f}|{100*m['disambig']['accuracy']:.2f}|{delta:+.2f}|[{ci[0]:+.2f}, {ci[1]:+.2f}]|{fail or '通过/固定参考'}|")
    lines+=['','各域对应 MMLU 配置成绩见 results.json，不把不同域设置的均值当作一个实际路由系统的能力成绩。历史测试已观察，结果为探索性，区间未作多重比较校正。']
    write(out/'results.json',records);(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');pilot.log('domain_reference_complete',path=str(out/'report.md'))


if __name__=='__main__':main()
