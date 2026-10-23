"""Exploratory matched relative-clause probe; every intervention remains frozen."""
from pathlib import Path
import sys,re,time,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.closed_transfer import rank,PRONOUN
from mechanism.identity_causal import variants,comparison,RAW
from mechanism.identity_replication import Experiment,extended_metrics
from scripts.analyze_identity_replication_mention_order import mention_order

OUT=ROOT/'results/identity_replication_20261015_v1/grammar_balance'
SOURCE=ROOT/'data/prepared/identity_combo_20261016_v1/historical.jsonl'
MODELS=('qwen3','minicpm5','qwen25')
OCCUPATION_GENDER={o.strip():g for g in ('male','female') for o in (RAW/f'{g}_occupations.txt').read_text().splitlines() if o.strip()}


def relative_row(r):
    """Same tokens except and -> , who; object now refers to first entity."""
    ctx=r['context'];a,b=mention_order(r)
    if r['pronoun'].lower() not in ('him','her') or r['gold']!=b:return None
    if len(PRONOUN.findall(ctx))!=1 or ctx.count(' and ')!=1:return None
    left,right=ctx.split(' and ')
    if not re.search(r'\b'+re.escape(b)+r'$',left,re.I):return None
    if not re.search(r'\b'+re.escape(a)+r'\b',left,re.I):return None
    if any(len(re.findall(r'\b'+re.escape(e)+r'\b',ctx,re.I))!=1 for e in (a,b)):return None
    if not re.search(r'\b'+re.escape(r['pronoun'])+r'\b',right,re.I):return None
    return {**r,'context':left+', who '+right,'gold':a,'stereotype':OCCUPATION_GENDER[a]==r['gender'],
        'source_id':r['source_id']+':relative','pair_id':r['pair_id']+':relative',
        'derivation':'single coordinator and replaced with relative subject who',
        'original_pair_id':r['pair_id'],'probe_case':'relative'}


def candidates():
    by=defaultdict(list)
    for r in rows(SOURCE):by[r['pair_id']].append(r)
    accepted=[]
    for pid,rr in sorted(by.items(),key=lambda z:rank(20261016,z[0])):
        zz=[relative_row(r) for r in rr]
        if len(rr)!=2 or any(z is None for z in zz):continue
        if {r['gender'] for r in rr}!={'male','female'}:continue
        accepted.append({'pair_id':pid,'family_id':rr[0]['family_id'],
            'original':rr,'relative':zz})
    return accepted


def declare():
    review=read(OUT/'text_review.json');pool={r['pair_id']:r for r in candidates()}
    assert review['source_sha256']==sha(SOURCE)
    selected=[];used=set()
    for pid in review['accepted_pair_ids']:
        z=pool[pid]
        if z['family_id'] in used:continue
        selected.append(z);used.add(z['family_id'])
        if len(selected)==64:break
    assert len(selected)>=24
    training_families={r['family_id'] for split in ('rank','screen','dev') for r in rows(SOURCE.parent/(split+'.jsonl'))}
    assert not used&training_families
    values=[]
    for z in selected:
        values += [{**r,'probe_case':'native','original_pair_id':r['pair_id']} for r in z['original']]
        values += z['relative']
    (OUT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values),encoding='utf-8')
    locks={}
    for n in MODELS:
        folder=OUT.parent/n
        locks[n]={k:sha(folder/k) for k in ('selection_frozen.json','norm_controls_frozen.json','parameters.npz','parameter_meta.json','plan_frozen.json')}
    write(OUT/'declaration.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
        'status':'exploratory protocol robustness probe proposed after main outcomes, not new confirmation',
        'script_sha256':sha(Path(__file__)),'source_sha256':sha(SOURCE),
        'review_sha256':sha(OUT/'text_review.json'),'rows_sha256':sha(OUT/'rows.jsonl'),
        'pairs_each_case':len(selected),'families':len(used),'method_locks':locks,
        'rank_screen_dev_family_overlap':0,
        'design':'same occupations/order, one coordinator and -> relative subject who; grammar switches gold from second mention to first; male/female and AB/BA fully crossed',
        'selection':'text and structural rules only; unique family; deterministic hash order; no model-score filtering or refitting',
        'limits':'derived grammar and answer-role shift, not native balanced Type2, not matched syntactic difficulty; agent text review, no independent human annotation',
        'metrics':['accuracy','both_correct','flip','both_wrong','second_mention_prediction_rate'],
        'inference':'frozen models/methods; same main FP16 batch4; all exploratory family intervals, no new winner'})
    print(json.dumps({'pairs_each_case':len(selected),'families':len(used)}))


def frozen_methods(e):
    s=read(e.out/'selection_frozen.json');e.validate_parameters(s)
    m={k:s['methods'][k] for k in ('baseline','selected','fine_combo','source_half')}
    n=read(e.out/'norm_controls_frozen.json')['methods']
    m['norm_whole_combo']=n['fine_norm_whole_combo']
    if e.name=='qwen3':
        m.update({k:s['methods'][k] for k in ('old_l7','old_combo')})
        m['old_norm_whole_combo']=n['old_norm_whole_combo']
    return m


def check_declaration():
    d=read(OUT/'declaration.json');assert d['script_sha256']==sha(Path(__file__))
    for name,file in [('source',SOURCE),('review',OUT/'text_review.json'),('rows',OUT/'rows.jsonl')]:
        assert d[name+'_sha256']==sha(file)
    for model,locks in d['method_locks'].items():
        for p,h in locks.items():assert sha(OUT.parent/model/p)==h
    return d


def evaluate():
    check_declaration();base=rows(OUT/'rows.jsonl');vs=[variants(r,permutation=p) for r in base for p in (0,1)]
    for name in MODELS:
        e=Experiment(name);e.prepare();methods=frozen_methods(e);scores={};saved=[]
        for k,s in methods.items():
            scores[k]=e.infer('grammar_balance_v1_'+k,vs,s)['scores']
            for r,z in zip(vs,scores[k]):saved.append({**r,'method':k,'scores':z.tolist(),'prediction':int(z.argmax())})
            e.log('grammar_balance_method',method=k)
        (OUT/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in saved),encoding='utf-8')
        e.close()
    summarize()


def summarize():
    d=check_declaration();data=rows(OUT/'rows.jsonl');vs=[variants(r,permutation=p) for r in data for p in (0,1)]
    lookup={r['uid']:r for r in vs};result={};count=0
    report=['# 答案角色平衡：冻结干预的探索性检验','',
        '在主结果后提出。保留职业名称和提及顺序，将唯一的and换为逗号加who，使后续非反身宾格代词的正确指代由第二实体转为第一实体。',
        f"{d['pairs_each_case']}个家族，每个都有原句/关系从句、男性/女性、AB/BA。原句来自已观察的历史Type2；派生句不是新的原生确认集。",
        '文本由agent审阅，无独立人工标注；关系从句和并列句难度可能不同。干预、均值、选层、选组、强度全部保持冻结。','',
        '|模型/方法|原句均对|关系从句均对|平衡均对|平衡准确率|平衡翻转|平衡选择第二实体|均对−原模型95%家族区间pp|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name in MODELS:
        pred=rows(OUT/(name+'_predictions.jsonl'));by=defaultdict(dict)
        for r in pred:
            count+=1;assert r['uid'] in lookup
            original=lookup[r['uid']]
            for key,value in original.items():assert r[key]==value,(name,key)
            assert r['prediction']==int(np.argmax(r['scores']))
            assert r['uid'] not in by[r['method']]
            by[r['method']][r['uid']]=r
        expected=set(frozen_methods_for_report(name));assert set(by)==expected
        metrics={};comparisons={}
        for method,z in by.items():
            assert set(z)==set(lookup)
            ordered=[z[r['uid']] for r in vs];ss=np.array([r['scores'] for r in ordered]);pp=ss.argmax(1)
            cases={c:np.array([r['probe_case']==c for r in vs]) for c in ('native','relative')};cases['balanced']=np.ones(len(vs),dtype=bool)
            metrics[method]={}
            basep=np.array([by['baseline'][r['uid']]['prediction'] for r in vs])
            comparisons[method]={}
            for case,mask in cases.items():
                vv=[r for r,keep in zip(vs,mask) if keep];zz=ss[mask]
                v=extended_metrics(vv,zz)
                v['second_mention_prediction_rate']=float(np.mean([r['choices'][p]==mention_order(r)[1] for r,p in zip(vv,pp[mask])]))
                metrics[method][case]=v
                comparisons[method][case]={field:comparison(vv,pp[mask],basep[mask],field,4000,20261016) for field in ('accuracy','both_correct','flip')}
            a=metrics[method];ci=comparisons[method]['balanced']['both_correct']['ci95'];b=a['balanced']
            report.append(f"|{name}/{method}|{a['native']['both_correct']:.2%}|{a['relative']['both_correct']:.2%}|{b['both_correct']:.2%}|{b['accuracy']:.2%}|{b['flip']:.2%}|{b['second_mention_prediction_rate']:.2%}|[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]|")
        result[name]={'metrics':metrics,'vs_baseline':comparisons}
    write(OUT/'analysis.json',{'declaration_sha256':sha(OUT/'declaration.json'),'models':result,
        'records_recomputed':count,'frozen_parameters_and_inputs_unchanged':True,
        'heuristics':{'always_first_balanced_accuracy':.5,'always_second_balanced_accuracy':.5},
        'status':'exploratory; family bootstrap4000; descriptive unadjusted intervals'})
    report+=['','始终选第一/第二实体在平衡集合的准确率均为50%。AB排列对每个版本都交叉，答案提及位置与选项字母位置分开控制。',
        '如果原句提升伴随关系从句下降，应限制为答案角色/句法条件相关收益；不能仅据这个对照确定内部原因，也不能把差异全归因于性别。','']
    p=ROOT/'research/2026-10-16_identity_replication_grammar_balance.md';p.write_text('\n'.join(report),encoding='utf-8')
    write(OUT/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'records':count,'analysis_sha256':sha(OUT/'analysis.json')})
    print(p)


def frozen_methods_for_report(name):
    s=read(OUT.parent/name/'selection_frozen.json')['methods']
    return ['baseline','selected','fine_combo','source_half','norm_whole_combo']+(['old_l7','old_combo','old_norm_whole_combo'] if name=='qwen3' else [])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('inspect','declare','evaluate','summarize'),required=True);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='inspect':
        values=candidates();write(OUT/'candidates.json',values)
        for i,z in enumerate(values):print(json.dumps({'index':i,'pair_id':z['pair_id'],'family_id':z['family_id'],'native':z['original'][0]['context'],'relative':z['relative'][0]['context']}))
    elif a.stage=='declare':declare()
    elif a.stage=='evaluate':evaluate()
    else:summarize()
