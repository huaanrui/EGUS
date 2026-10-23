"""Independent prediction reconstruction and family-level planned comparisons."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json,tomllib
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.identity_replication import extended_metrics,matched_groups,Experiment
from mechanism.identity_parts import signed_scores,signed_top
from mechanism.identity_causal import variants
from scripts.audit_identity_replication_records_v3 import restore_metadata


def bootstrap_delta(vs,aa,bb,reps=4000,seed=20261015):
    groups=defaultdict(list)
    for r,a,b in zip(vs,aa,bb):groups[r['family_id']].append(a-b)
    s=np.array([sum(z) for z in groups.values()]);n=np.array([len(z) for z in groups.values()]);rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(s),(reps,len(s)));est=s[ix].sum(1)/n[ix].sum(1)
    delta=float(s.sum()/n.sum());null=abs(rng.choice([-1,1],size=(reps,len(s)))@s/n.sum())
    return {'delta':delta,'ci95':np.quantile(est,[.025,.975]).tolist(),'p_sign':float((1+(null>=abs(delta)-1e-12).sum())/(reps+1)),
            'clusters':len(s),'cluster':'family_id'}


def pair_success(rr):
    by=defaultdict(list)
    for r in rr:by[(r['pair_id'],r['permutation'])].append(r)
    keys=sorted(by);representatives=[by[k][0] for k in keys]
    success=np.array([float(all(r['prediction']==r['label'] for r in by[k])) for k in keys])
    return keys,representatives,success


def load_shards(folder):
    result=defaultdict(list)
    for p in sorted(folder.glob('*.npz')):
        with np.load(p) as z:
            for k in z.files:
                if k!='identity':result[k].append(z[k])
    return {k:np.concatenate(v) for k,v in result.items()}


def holm(items):
    order=sorted(range(len(items)),key=lambda i:items[i]['p_sign']);current=0
    for j,i in enumerate(order):
        current=max(current,min(1,items[i]['p_sign']*(len(items)-j)));items[i]['p_holm']=current


def main():
    c=tomllib.loads((ROOT/'configs/experiment_identity_replication_v1.toml').read_text());folder=ROOT/c['output_dir']
    lock=read(folder/'all_models_frozen.json');primary=[];result={};total=0
    rankrows=[variants(r) for r in rows(ROOT/c['source_data']/'rank.jsonl')]
    for model in c['models']:
        name=model['name'];out=folder/name;selection=read(out/'selection_frozen.json')
        assert sha(out/'selection_frozen.json')==lock['selection_sha256'][name]
        assert sha(out/'parameters.npz')==selection['parameter_sha256']
        assert sha(out/'parameter_meta.json')==selection['meta_sha256']
        meta=read(out/'parameter_meta.json')
        e=Experiment(name);dev=read(out/'dev_results.json')
        for r in dev:
            feasible=all(r['metrics'][t]['accuracy']>=dev[0]['metrics'][t]['accuracy']-c['selection_tolerance']-1e-12 for t in dev[0]['metrics'])
            assert feasible==r['feasible']
        assert max(dev,key=e.order)['spec']==selection['selected']
        with np.load(out/'parameters.npz') as z:params={k:z[k] for k in z.files}
        a=load_shards(out/'shards/rank_capture_A');b=load_shards(out/'shards/rank_capture_B')
        for key,s in meta['sites'].items():
            if key.startswith('old_') or key not in a:continue
            ga=load_shards(out/'shards'/('grad_'+key+'_A'));gb=load_shards(out/'shards'/('grad_'+key+'_B'))
            mu,ts,gs,_,_=signed_scores(rankrows,a[key].astype(float),b[key].astype(float),ga['gradient'],gb['gradient'],a['scores'],b['scores'])
            assert np.array_equal(mu,params[key+'_mean']);assert np.array_equal(ts,params[key+'_transfer_score']);assert np.array_equal(gs,params[key+'_gap_score'])
            budget=meta['groups'][key]['budget']
            for g,scores,pos in [('gap_pos',gs,True),('gap_neg',gs,False),('transfer_pos',ts,True),('transfer_neg',ts,False)]:
                assert np.array_equal(signed_top(scores,budget,pos),params[key+'_'+g])
            for seed in c['random_seeds']:
                expected=matched_groups({g:params[key+'_'+g] for g in ('gap_pos','gap_neg')},params[key+'_rms'],params[key+'_wn'],seed)
                for g,v in expected.items():assert np.array_equal(v,params[key+f'_random{seed}_'+g])
        evaluation=read(out/'evaluation.json');modelresult={'metrics':evaluation,'comparisons':{},'selected':selection['selected']['name']}
        for p in sorted((out/'predictions').glob('*.jsonl')):
            rr=rows(p);total+=len(rr);by=defaultdict(list)
            for r in rr:assert r['prediction']==int(np.argmax(r['scores']));by[r['method']].append(restore_metadata(r))
            if p.stem in evaluation:
                for method,z in by.items():
                    recalc=extended_metrics(z,np.array([r['scores'] for r in z]))
                    for f in ('accuracy','both_correct','flip','both_wrong'):assert abs(recalc[f]-evaluation[p.stem][method][f])<1e-12
                keys,vs,combo=pair_success(by['fine_combo']);_,_,whole=pair_success(by['whole_combo']);_,_,source=pair_success(by['fine_source']);_,_,ws=pair_success(by['whole_source'])
                random=np.mean([pair_success(by[f'random_combo_{seed}'])[2] for seed in c['random_seeds']],axis=0)
                comps={'fine_combo_vs_whole':bootstrap_delta(vs,combo,whole),
                       'fine_combo_vs_matched_random_mean':bootstrap_delta(vs,combo,random),
                       'fine_source_vs_whole':bootstrap_delta(vs,source,ws)}
                _,_,base=pair_success(by['baseline']);_,_,terminal=pair_success(by['fine_terminal'])
                comps['source_terminal_interaction']=bootstrap_delta(vs,combo-source-terminal+base,np.zeros(len(base)))
                modelresult['comparisons'][p.stem]=comps
                if p.stem=='confirmation':
                    for k in ('fine_combo_vs_whole','fine_combo_vs_matched_random_mean','fine_source_vs_whole'):
                        primary.append({'model':name,'comparison':k,**comps[k]})
                if name=='qwen3':
                    _,_,old=pair_success(by['old_combo']);_,_,om=pair_success(by['old_mean_combo'])
                    random=np.mean([pair_success(by[f'old_random_combo_{seed}'])[2] for seed in c['random_seeds']],axis=0)
                    comps['old_combo_vs_whole']=bootstrap_delta(vs,old,om);comps['old_combo_vs_matched_random_mean']=bootstrap_delta(vs,old,random)
        verification=read(out/'verification.json');verification.update({'independent_fit_scores_recomputed':True,'independent_group_selection_recomputed':True,
            'independent_random_indices_recomputed':True,'independent_behavior_metrics_recomputed':True,'dev_feasibility_and_winner_recomputed':True})
        write(out/'verification.json',verification);result[name]=modelresult
    holm(primary);write(folder/'planned_comparisons.json',{'primary_confirmation_holm9':primary,'models':result})
    write(folder/'verification.json',{'prediction_records_recomputed':total,'models':len(result),'all_parameters_and_selections_unchanged':True,
          'fit_scores_and_groups_recomputed':True,'confirmation_selected_before_model_outputs':True,'primary_tests':len(primary)})
    report=['# 三模型复验与强对照结果','',
        '层号从0开始。历史数据为探索性复验；确认集为96个此前未进入准备数据的原生Type1家族，agent文本复核，无独立人工标注。',
        '主要组合是统一协议的诊断性细粒度组合；overall允许不干预，二者分开报告。旧Qwen3组合为额外冻结锚点。','',
        '|模型/数据|原模型均对|选参赢家均对|细粒度组合均对|整均值组合均对|匹配随机组合均对均值|',
        '|---|---:|---:|---:|---:|---:|']
    for name,z in result.items():
        for split,m in z['metrics'].items():
            rand=np.mean([m[f'random_combo_{seed}']['both_correct'] for seed in c['random_seeds']])
            report.append(f"|{name}/{split}|{m['baseline']['both_correct']:.2%}|{m['selected']['both_correct']:.2%}|{m['fine_combo']['both_correct']:.2%}|{m['whole_combo']['both_correct']:.2%}|{rand:.2%}|")
    report += ['','## 确认集预设主要比较（9项Holm校正）','',
        '|模型|比较|均对差pp|95%家族区间pp|校正p|','|---|---|---:|---|---:|']
    for r in primary:report.append(f"|{r['model']}|{r['comparison']}|{100*r['delta']:+.2f}|[{100*r['ci95'][0]:+.2f},{100*r['ci95'][1]:+.2f}]|{r['p_holm']:.4f}|")
    report += ['','## 细节与边界','',
        '身份读取、匿名职业、固定them/交换档案、明确事件事实与中性档案成绩见各模型utilities.json；短答案生成见chat_results.json。',
        '源效应正负分组真实恢复/移植见diagnostics.json；该符号不等于永久偏见好坏。',
        'Type1原生标签依赖上下文/常识，不统一属于明确证据；派生事件记录问答独立报告。',
        '全部主要结果含两种AB排列；随机种子平均以家族为抽样单位，不增加独立样本数。',
        '本轮未运行CEB自由续写、新模型QKV完整路径或CoT对照。','',f'独立复核预测记录：{total:,}（不是独立问题数）。']
    p=ROOT/'research/2026-10-15_identity_replication_findings.md';p.write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'primary':primary,'records':total},indent=2))


if __name__=='__main__':main()
