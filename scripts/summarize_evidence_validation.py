"""Read-only fitted assets; report repeated fits and the frozen confirmation."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from mechanism.evidence_validation import Campaign,Engine,policies
from mechanism.evidence_validation_stats import comparison,holm,mean_sd
from mechanism.evidence_story import signals,apply
from mechanism.fairsteer_author import read,write,metrics,sha


def summary_metrics(vs,p,mask):
    if vs[0].get('utility'):
        return {'accuracy':float(np.mean(p==np.array([r['label'] for r in vs]))),'trigger':float(mask.mean())}
    m=metrics(vs,p);clear=np.array([r['context_condition']=='disambig' for r in vs]);u=np.array([r['unknown_index'] for r in vs])
    return {'accuracy':m['accuracy'],'ambig':m['ambig']['accuracy'],'clear':m['disambig']['accuracy'],
        'bias':m['ambig']['bias_score'],'clear_unknown':float(np.mean(p[clear]==u[clear])) if clear.any() else None,
        'trigger':float(mask.mean()),'sufficient_trigger':float(mask[clear].mean()) if clear.any() else None}


def pairs_correct(vs,p):
    groups=defaultdict(list)
    for i,r in enumerate(vs):groups[r['pair_id']].append(i)
    assert all(len(ii)==2 for ii in groups.values())
    return float(np.mean([all(p[i]==vs[i]['label'] for i in ii) for ii in groups.values()]))


def main():
    c=Campaign();c.prepare();assert (c.out/'inference_complete.json').exists();result={};primary={};secondary={};count=0;curves={}
    for spec in c.c['models']:
        name=spec['name'];e=Engine(name);allpred={};perseed={};recognition={};curves[name]={}
        for key in ('test','confirmation','test_utility','stress','counterfactual_'+name):
            selectedseeds=c.c['seeds'] if key in ('test','confirmation','test_utility') else [c.c['seed']]
            perseeds={};method_arrays=defaultdict(list)
            for seed in selectedseeds:
                vs,cache,pred,masks,sig=policies(e,key,seed);folder=e.seed_folder(seed);saved={}
                for method,p in pred.items():
                    saved['pred_'+method]=p.astype(np.int16);saved['trigger_'+method]=masks[method];count+=len(p)
                saved['uids']=np.array([r['uid'] for r in vs]);np.savez_compressed(folder/f'predictions_{key}.npz',**saved)
                main=np.array([r.get('permutation',0)==0 for r in vs]) if key=='confirmation' else np.ones(len(vs),bool)
                values=[r for r,keep in zip(vs,main) if keep]
                mm={method:summary_metrics(values,p[main],masks[method][main]) for method,p in pred.items()}
                if key=='confirmation':
                    for method,p in pred.items():mm[method]['gender_pair_both_correct']=pairs_correct(values,p[main])
                    mm['_permutations']={str(pos):{method:summary_metrics([r for r in vs if r['permutation']==pos],p[np.array([r['permutation']==pos for r in vs])],masks[method][np.array([r['permutation']==pos for r in vs])]) for method,p in pred.items()} for pos in range(3)}
                perseeds[str(seed)]=mm
                for method,p in pred.items():method_arrays[method].append(p[main])
                if key in ('test','confirmation'):
                    y=np.array([r['context_condition']=='ambig' for r in values]);probes=dict(np.load(folder/'probes.npz'))
                    x=cache['features'][main].astype(float);margin=x@probes['2200_evidence_coef']+probes['2200_evidence_intercept']
                    recognition.setdefault(key,{})[str(seed)]={'evidence_auroc':float(roc_auc_score(y,margin)),
                        'evidence_probability_auroc':float(roc_auc_score(y,sig['evidence'][main]))}
                    if key=='test':
                        curve={};unknown=np.array([r['unknown_index'] for r in vs]);selection=read(folder/'selection_frozen.json')
                        for budget in c.c['fit_budgets']:
                            ss=signals(cache['features'],probes,budget);temp={}
                            temp['evidence_auroc']=float(roc_auc_score(y,cache['features'].astype(float)@probes[f'{budget}_evidence_coef']+probes[f'{budget}_evidence_intercept']))
                            for gate in ('evidence','product'):
                                mask=ss[gate]>selection['thresholds'][f'{budget}_{gate}']['threshold']
                                for action in ('direct','steer'):
                                    p,_=apply(cache['base'],np.load(folder/'cf_test_1.5.npz')['scores'],unknown,mask,action)
                                    temp[gate+'_'+action]=summary_metrics(vs,p,mask)
                            curve[str(budget)]=temp
                        curves[name][str(seed)]=curve
            aggregate={}
            for method in method_arrays:
                keys=[k for k in perseeds[str(selectedseeds[0])][method] if all(perseeds[str(s)][method][k] is not None for s in selectedseeds)]
                aggregate[method]={metric:mean_sd([perseeds[str(s)][method][metric] for s in selectedseeds]) for metric in keys}
            perseed[key]=perseeds;allpred[key]=(values,{m:np.array(pp) for m,pp in method_arrays.items()})
            result.setdefault(name,{})[key]={'aggregate':aggregate,'per_seed':perseeds,'questions_primary':len(values),'seeds':selectedseeds}
            if key in ('stress','counterfactual_'+name):
                groups=defaultdict(list)
                for i,r in enumerate(vs):groups[r.get('variant',f'p{r.get("permutation",0)}_{r.get("wording","original")}')].append(i)
                result[name][key]['groups']={g:{method:summary_metrics([vs[i] for i in ii],p[ii],masks[method][ii]) for method,p in pred.items()} for g,ii in groups.items()}
                if key.startswith('counterfactual'):
                    use=np.array([r['variant'] in ('neutral_matched','sufficient') for r in vs]);yy=np.array([r['context_condition']=='ambig' for r in vs])
                    probe=dict(np.load(e.seed_folder(c.c['seed'])/'probes.npz'));margin=cache['features'].astype(float)@probe['2200_evidence_coef']+probe['2200_evidence_intercept']
                    result[name][key]['equal_length_evidence_auroc']=float(roc_auc_score(yy[use],margin[use]))
        for dataset in ('test','confirmation'):
            values,pp=allpred[dataset]
            for reference,condition in [('surface_direct',None),('base','disambig')]:
                tag=f'{name}/{dataset}/evidence_direct-{reference}'+('/clear' if condition else '')
                primary[tag]=comparison(values,pp['evidence_direct'],pp[reference],c.c['bootstrap_replicates'],condition=condition)
                if dataset=='confirmation':secondary[tag+'/occupation_cluster']=comparison(values,pp['evidence_direct'],pp[reference],c.c['bootstrap_replicates'],condition=condition,cluster='occupation_pair')
            for reference in ('base','local_fs','product_direct','evidence_steer'):
                secondary[f'{name}/{dataset}/evidence_direct-{reference}']=comparison(values,pp['evidence_direct'],pp[reference],c.c['bootstrap_replicates'])
        result[name]['recognition']=recognition;write(e.out/'results.json',result[name])
    assert len(primary)==12;holm(primary)
    if (c.out/'primary_direct_statistics.json').exists():assert primary==read(c.out/'primary_direct_statistics.json')
    write(c.out/'results.json',{'models':result,'primary':primary,'secondary':secondary,'head_budget_curves':curves,'prediction_records':count,
        'exposure_audit':read(c.out/'plan_frozen.json')['confirmation'],'selection_lock_sha256':sha(c.out/'all_selections_frozen.json')})
    lines=['# 三模型、五训练抽样种子与独立确认','',
        '同一BBQ家族划分；每种子2200训练题，352开发题，历史测试704题。五次重新抽取训练样本并重新估计方向，不是五个独立测试集。层固定为Qwen3 L18、MiniCPM5 L28、Qwen2.5 L24；raw zero-shot。',
        '新确认640道SynthBias原标注问题、三种选项排列，排除了所有旧准备数据中的样本ID、去词汇化模板家族和职业对。主表排列0；其余排列完整报告。选择题派生协议不等于原作者协议，也不保证预训练未见。所有15组源域选择在确认推理前冻结。','',
        '数值为五训练种子均值±样本标准差。证据与表面头使用相同训练题及证据条件标签；FairSteer风格正确性头按原规则过滤部分标签。',
        '结果需要区分任务内收益与迁移代价：三模型BBQ均涨分，但Qwen2.5相对表面基线额外收益很小；新确认集的充分题保留明显失败。不能只引用确认集总体准确率上升宣称泛化成功。','',
        '|模型/方法|BBQ总体/%|BBQ不足/%|BBQ充分/%|确认总体/%|确认不足/%|确认充分/%|',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels=[('base','原模型'),('local_fs','本地FS风格'),('product_steer','乘积门控+方向'),('product_direct','同乘积门控direct'),
        ('evidence_steer','单证据门控+方向'),('evidence_direct','单证据头direct'),('surface_direct','表面特征direct'),('question_direct','问题词袋direct'),('output_direct','输出分数direct'),('always_unknown','总选unknown')]
    def fmt(m,key):return f'{m[key]["mean"]*100:.2f}±{m[key]["sd"]*100:.2f}'
    for name,data in result.items():
        for method,title in labels:
            a=data['test']['aggregate'][method];b=data['confirmation']['aggregate'][method]
            lines.append(f'|{name}/{title}|{fmt(a,"accuracy")}|{fmt(a,"ambig")}|{fmt(a,"clear")}|{fmt(b,"accuracy")}|{fmt(b,"ambig")}|{fmt(b,"clear")}|')
    lines+=['','## 预先声明的12项比较','',
        '95%区间交叉重采样训练种子和题目家族；它们是逐比较区间，不是同时置信区间。p为五种子平均效果的家族符号随机化，Holm校正12项。确认集另提供职业对聚类区间。充分题的零假设检验不是非劣检验。','',
        '|比较|差值pp|95%区间pp|Holm p|','|---|---:|---|---:|']
    for k,v in primary.items():lines.append(f'|{k}|{v["difference_pp"]:+.2f}|[{v["ci95_pp"][0]:+.2f},{v["ci95_pp"][1]:+.2f}]|{v["holm_p"]:.4f}|')
    lines+=['','## 逐种子关键结果','', '|模型/种子|BBQ证据direct/%|BBQ表面direct/%|确认证据direct/%|确认充分损失pp|','|---|---:|---:|---:|---:|']
    for name,data in result.items():
        for seed in c.c['seeds']:
            a=data['test']['per_seed'][str(seed)];b=data['confirmation']['per_seed'][str(seed)]
            lines.append(f'|{name}/{seed}|{a["evidence_direct"]["accuracy"]*100:.2f}|{a["surface_direct"]["accuracy"]*100:.2f}|{b["evidence_direct"]["accuracy"]*100:.2f}|{(b["evidence_direct"]["clear"]-b["base"]["clear"])*100:+.2f}|')
    lines+=['','## 确认选项排列','', '|模型/排列|证据direct/%|表面direct/%|原模型/%|','|---|---:|---:|---:|']
    for name,data in result.items():
        for pos in range(3):
            mm={method:np.mean([data['confirmation']['per_seed'][str(seed)]['_permutations'][str(pos)][method]['accuracy'] for seed in c.c['seeds']])*100 for method in ('evidence_direct','surface_direct','base')}
            lines.append(f'|{name}/{pos}|{mm["evidence_direct"]:.2f}|{mm["surface_direct"]:.2f}|{mm["base"]:.2f}|')
    lines+=['','## 原种子的等长压力集','', '|模型|等长证据AUROC|原不足检出/%|等长不足检出/%|','|---|---:|---:|---:|']
    for name,data in result.items():
        d=data['counterfactual_'+name];g=d['groups'];lines.append(f'|{name}|{d["equal_length_evidence_auroc"]:.4f}|{g["insufficient"]["evidence_direct"]["trigger"]*100:.2f}|{g["neutral_matched"]["evidence_direct"]["trigger"]*100:.2f}|')
    lines+=['','## 范围与解释','',
        '- 历史BBQ均值的稳定提升不能代替新确认的迁移结论；新确认失败/误伤也完整报告。',
        '- 所有阈值只读源域dev，2%误触发仅是源域点估计限制；确认集没有校准或择优种子。',
        '- 偏置匹配方向的unknown频率，不是按目标准确率优化；单层单方向不能否定全部表示干预。',
        '- 直接unknown在无unknown选项的MMLU上不动作；有限能力控制不构成通用能力无损。',
        '- head预算曲线的方向110条固定；不把不同种子、排列、方法的预测当作独立问题。',
        '- 新确认沿用作者经人类标注验证的type1/type2标签；本轮没有独立人工裁决。模型推理前文本抽查发现至少一个type2样本可能存在竞争指代，仍保留原标签和全部样本；充分率/FPR按给定标签解释。',
        '- 本轮研究识别与决策，不宣称已定位因果偏见电路。','']
    text='\n'.join(lines)
    (c.out/'report.md').write_text(text,encoding='utf-8');(Path('research')/'2026-10-17_evidence_validation_findings.md').write_text(text,encoding='utf-8')
    print(json.dumps({'report_complete':True,'prediction_records':count,'primary':primary}),flush=True)


if __name__=='__main__':main()
