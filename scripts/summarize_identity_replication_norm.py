from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.closed_transfer import digest
from mechanism.identity_replication import extended_metrics
from scripts.summarize_identity_replication import pair_success,bootstrap_delta,holm,load_shards
from scripts.audit_identity_replication_records_v3 import restore_metadata

folder=ROOT/'results/identity_replication_20261015_v1';comparisons=[];counts=0;result={}
for name in ('qwen3','minicpm5','qwen25'):
    out=folder/name;saved=read(out/'norm_controls_frozen.json')
    assert sha(out/'norm_controls_frozen.json')==read(folder/'all_norm_controls_frozen.json')['sha256'][name]
    for tag,c in saved['calibrations'].items():
        for step in c['history']:
            z=load_shards(out/'shards'/('normcal_'+step['spec']['name']))
            assert float(z['edit_energy'].mean())==step['energy']
        assert c['relative_rms_error']==min(h['relative_rms_error'] for h in c['history'])
    model={}
    for split in ('historical','new_type1','winogender','confirmation'):
        main=defaultdict(list);norm=defaultdict(list)
        for r in rows(out/'predictions'/(split+'.jsonl')):main[r['method']].append(r)
        for r in rows(out/'predictions'/('norm_'+split+'.jsonl')):
            assert r['prediction']==int(np.argmax(r['scores']));norm[r['method']].append(restore_metadata(r));counts+=1
        m={k:extended_metrics(z,np.array([r['scores'] for r in z])) for k,z in norm.items()}
        model[split]={'metrics':m,'comparisons':{}}
        for label,method in [('fine','fine_combo')]+([('old','old_combo')] if name=='qwen3' else []):
            keys,vs,fine=pair_success(main[method]);_,_,whole=pair_success(norm[label+'_norm_whole_combo'])
            random=np.mean([pair_success(norm[label+f'_norm_random{seed}'])[2] for seed in range(1701,1706)],axis=0)
            cs={'vs_norm_whole':bootstrap_delta(vs,fine,whole),'vs_norm_random_mean':bootstrap_delta(vs,fine,random)}
            model[split]['comparisons'][label]=cs
            originals=read(out/'selection_frozen.json')['methods'];signature=digest(originals[method]['edits'])
            refpaths=[out/'shards'/('eval_'+split+'_'+s['name']) for s in originals.values() if digest(s['edits'])==signature]
            reference=load_shards(next(p for p in refpaths if p.exists()))['edit_energy'].mean()
            ratios={}
            for key in [label+'_norm_whole_combo']+[label+f'_norm_random{s}' for s in range(1701,1706)]:
                value=load_shards(out/'shards'/('normeval_'+split+'_'+key))['edit_energy'].mean()
                ratios[key]=float(np.sqrt(value/max(reference,1e-12)))
            model[split].setdefault('target_edit_rms_ratio',{})[label]=ratios
            if split=='confirmation':
                for k,z in cs.items():comparisons.append({'model':name,'method':label,'comparison':k,**z})
    result[name]=model
holm(comparisons);write(folder/'norm_comparisons.json',{'confirmation_holm8':comparisons,'models':result})
p=ROOT/'research/2026-10-15_identity_replication_findings.md'
lines=['','## 开发集实际扰动RMS校准的补充强对照','',
       '在任何主目标推理前追加声明、拟合并冻结；原9项主要比较保持不变，以下8项补充比较另作Holm校正。',
       '整组件比较包含源/末端四种整体符号组合，按开发集任务正确性与能力约束选择；所有强度只按开发集扰动校准。','',
       '|模型/数据|细粒度组合均对|范数匹配整均值组合均对|范数匹配随机组合均对均值|','|---|---:|---:|---:|']
for name,z in result.items():
    main=read(folder/name/'evaluation.json')
    for split,values in z.items():
        m=values['metrics'];r=np.mean([m[f'fine_norm_random{s}']['both_correct'] for s in range(1701,1706)])
        lines.append(f"|{name}/{split}|{main[split]['fine_combo']['both_correct']:.2%}|{m['fine_norm_whole_combo']['both_correct']:.2%}|{r:.2%}|")
lines+=['','|模型/方案|比较|确认集差pp|95%家族区间pp|校正p|','|---|---|---:|---|---:|']
for r in comparisons:lines.append(f"|{r['model']}/{r['method']}|{r['comparison']}|{100*r['delta']:+.2f}|[{100*r['ci95'][0]:+.2f},{100*r['ci95'][1]:+.2f}]|{r['p_holm']:.4f}|")
lines+=['','校准后的随机组合也检查性别读取与身份绑定；整均值对照另检查记录、明确事实及身份必要任务。',
        '范数匹配在DEV完成，目标域实际扰动仍可能不同；这不是逐条目标样本的在线等范数干预。',
        '固定them的指代控制可能引入单数/复数解释差异，应与明确事件记录对照分开解读。','']
with p.open('a',encoding='utf-8') as f:f.write('\n'.join(lines))
v=read(folder/'verification.json');v.update({'norm_calibration_records_recomputed':True,'norm_prediction_records_recomputed':counts,
    'norm_parameters_froze_before_main_target_lock':read(folder/'all_norm_controls_frozen.json')['created']<=read(folder/'all_models_frozen.json')['created']})
assert v['norm_parameters_froze_before_main_target_lock'];write(folder/'verification.json',v)
print(__import__('json').dumps(comparisons,indent=2))
