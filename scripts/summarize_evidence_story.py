"""Report-only analysis of frozen matched-budget evidence experiments."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import joblib
import numpy as np
from scipy.special import logsumexp,expit
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
from mechanism.evidence_story import Study,ModelStudy,signals,surface
from mechanism.fairsteer_author import read,write,sha,metrics


def contrast(vs,a,b,reps=2000,seed=20261017,condition=None):
    groups=defaultdict(list)
    for r,x,y in zip(vs,a,b):
        if condition is None or r['context_condition']==condition:
            groups[r['family']].append(int(x==r['label'])-int(y==r['label']))
    g=list(groups);s=np.array([sum(groups[k]) for k in g],float);n=np.array([len(groups[k]) for k in g],float)
    rng=np.random.default_rng(seed);ii=rng.integers(len(g),size=(reps,len(g)))
    samples=s[ii].sum(1)/n[ii].sum(1);observed=s.sum()/n.sum()
    null=(rng.choice([-1.,1.],size=(max(4000,reps),len(g)))*s).sum(1)/n.sum()
    p=(1+np.sum(abs(null)>=abs(observed)-1e-12))/(len(null)+1)
    return {'difference_pp':float(observed*100),'ci95_pp':(np.quantile(samples,[.025,.975])*100).tolist(),
            'p':float(p),'families':len(g),'rows':int(n.sum()),'condition':condition}


def holm(items):
    order=sorted(items,key=lambda k:items[k]['p']);v=0.
    for i,k in enumerate(order):v=max(v,min(1.,items[k]['p']*(len(order)-i)));items[k]['holm_p']=v


def continuous_ci(values,families,reps=2000):
    grouped=defaultdict(list)
    for v,f in zip(values,families):grouped[f].append(v)
    sums=np.array([sum(v) for v in grouped.values()]);counts=np.array([len(v) for v in grouped.values()])
    rng=np.random.default_rng(20261017);ii=rng.integers(len(sums),size=(reps,len(sums)))
    means=sums[ii].sum(1)/counts[ii].sum(1)
    return {'mean':float(np.mean(values)),'ci95':np.quantile(means,[.025,.975]).tolist(),'families':len(sums)}


def probe_report(vs,x,probes,budgets):
    y=np.array([r['context_condition']=='ambig' for r in vs]);result={}
    for b in budgets:
        sig=signals(x,probes,b)
        for gate,s in sig.items():
            result[f'{b}_{gate}']={'auroc':float(roc_auc_score(y,s)),
                                   'auprc':float(average_precision_score(y,s))}
    return result


def diagnostic(e,key):
    vs,cache,pred,masks,scores=e.policies(key);b=max(e.c['fit_budgets']);sig=signals(cache['features'],dict(np.load(e.out/'probes.npz')),b)
    selection=read(e.out/'selection_frozen.json')['selections'];methods=['base',f'{b}_product_steer1.5',f'{b}_product_direct',f'{b}_evidence_direct','fixed_local_fs']
    out={'rows':len(vs),'methods':{},'head':{},'aggregate':{method:metrics(vs,pred[method]) for method in methods}}
    groups=defaultdict(list)
    for i,r in enumerate(vs):groups[r.get('variant',f'p{r.get("permutation",0)}_{r.get("wording","native")}')].append(i)
    for name,ii in groups.items():
        yy=np.array([vs[i]['context_condition']=='ambig' for i in ii]);unknown=np.array([vs[i]['unknown_index'] for i in ii])
        ss=cache['base'][ii,:3].copy();uu=ss[np.arange(len(ii)),unknown];ss[np.arange(len(ii)),unknown]=-np.inf
        mm=uu-logsumexp(ss,axis=1)
        out['head'][name]={'rows':len(ii),'mean_unknown_log_odds':float(mm.mean())}
        for gate in ('evidence','product'):
            p=sig[gate][ii];trigger=p>selection[f'{b}_{gate}']['gate']['threshold']
            out['head'][name][gate]={'mean_probability':float(p.mean()),'trigger_fraction':float(trigger.mean()),
                'insufficient_recall':float(trigger[yy].mean()) if yy.any() else None,
                'sufficient_fpr':float(trigger[~yy].mean()) if (~yy).any() else None,
                'auroc':float(roc_auc_score(yy,p)) if len(set(yy))==2 else None}
        for method in methods:
            out['methods'].setdefault(method,{})[name]=metrics([vs[i] for i in ii],pred[method][ii])
    if key.startswith('counterfactual'):
        pairs=defaultdict(dict)
        for i,r in enumerate(vs):pairs[r['pair_id']][r['variant']]=i
        out['paired']={}
        matched=[i for i,r in enumerate(vs) if r['variant'] in ('neutral_matched','sufficient')]
        matched_y=np.array([vs[i]['context_condition']=='ambig' for i in matched])
        out['equal_length_recognition']={gate:float(roc_auc_score(matched_y,sig[gate][matched])) for gate in ('evidence','product')}
        for gate in ('evidence','product'):
            real=[];neutral=[];ordering=[];relative=[];ff=[]
            for pair in pairs.values():
                a=pair['insufficient'];c=pair['sufficient'];n=pair['neutral_matched'];cn=pair['sufficient_neutral']
                real.append(sig[gate][a]-sig[gate][c]);neutral.append(sig[gate][a]-sig[gate][n])
                ordering.append(sig[gate][n]>sig[gate][c]);relative.append(sig[gate][n]-sig[gate][c])
                ff.append(vs[a]['family'])
            out['paired'][gate]={'pairs':len(pairs),'real_evidence_mean_reduction':float(np.mean(real)),
                'neutral_padding_mean_reduction':float(np.mean(neutral)),
                'equal_length_insufficient_above_sufficient_fraction':float(np.mean(ordering)),
                'equal_length_mean_probability_difference':float(np.mean(relative)),
                'equal_length_probability_difference_cluster_ci':continuous_ci(relative,ff),
                'equal_length_ordering_cluster_ci':continuous_ci(ordering,ff)}
    else:
        families=defaultdict(list)
        for i,r in enumerate(vs):families[r['source_uid']].append(i)
        out['consistency']={}
        for method in methods:
            different=[]
            for ii in families.values():
                semantic={vs[i]['ans'+str(pred[method][i])] if pred[method][i]!=vs[i]['unknown_index'] else '__UNKNOWN__' for i in ii}
                different.append(len(semantic)>1)
            out['consistency'][method]={'any_semantic_prediction_change':float(np.mean(different)),'questions':len(families)}
        for gate in ('evidence','product'):
            ranges=[np.ptp(sig[gate][ii]) for ii in families.values()]
            out['consistency'][gate+'_head']={'mean_probability_range':float(np.mean(ranges))}
    return out


def surface_controls(s):
    folder=s.out/'surface_controls';folder.mkdir(exist_ok=True)
    fit=[json.loads(l) for l in (s.data/'fit.jsonl').read_text(encoding='utf-8').splitlines()]
    y=np.array([r['context_condition']=='ambig' for r in fit]);results={}
    models={'surface':make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=10000,tol=1e-4)),
            'question_only':make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=3000,min_df=2),LogisticRegression(C=1.,max_iter=10000,tol=1e-4))}
    for name,model in models.items():
        path=folder/f'{name}.joblib'
        if path.exists():model=joblib.load(path)
        else:
            xx=np.array([surface(r) for r in fit]) if name=='surface' else [r['question'] for r in fit]
            with threadpool_limits(limits=s.c['cpu_threads']):model.fit(xx,y)
            joblib.dump(model,path)
        result={}
        for key in ('test','stress',*[f'counterfactual_{m["name"]}' for m in s.c['models']]):
            vs=[json.loads(l) for l in (s.data/f'{key}.jsonl').read_text(encoding='utf-8').splitlines()]
            inp=np.array([surface(r) for r in vs]) if name=='surface' else [r['question'] for r in vs]
            margin=model.decision_function(inp);yy=np.array([r['context_condition']=='ambig' for r in vs])
            result[key]={'auroc':float(roc_auc_score(yy,margin)),'auprc':float(average_precision_score(yy,margin))}
            if key.startswith('counterfactual'):
                use=np.array([r['variant'] in ('neutral_matched','sufficient') for r in vs])
                result[key]['equal_length_auroc']=float(roc_auc_score(yy[use],margin[use]))
            np.savez(folder/f'{name}_{key}.npz',margin=margin,uids=np.array([r['uid'] for r in vs]))
        results[name]=result
    write(folder/'results.json',{'training_rows':len(fit),'same_fit_questions_as_hidden_heads':True,'results':results})
    return results


def main():
    s=Study();s.prepare();assert (s.out/'inference_complete.json').exists()
    result={};primary={};prediction_count=0
    for model in s.c['models']:
        e=ModelStudy(model['name']);vs,cache,pred,masks,score=e.policies('test');methods={}
        uv,uc,up,um,us=e.policies('test_utility')
        with (e.out/'predictions.jsonl').open('w',encoding='utf-8') as f:
            for name,p in pred.items():
                mm=metrics(vs,p);u=np.array([r['unknown_index'] for r in vs]);yy=np.array([r['context_condition']=='ambig' for r in vs])
                mm.update({'sufficient_unknown_rate':float(np.mean(p[~yy]==u[~yy])),
                    'trigger_fraction':float(masks[name].mean()),'insufficient_trigger':float(masks[name][yy].mean()),
                    'sufficient_trigger':float(masks[name][~yy].mean()),
                    'utility_accuracy':float(np.mean(up[name]==np.array([r['label'] for r in uv]))),
                    'utility_trigger':float(um[name].mean())})
                if name in score:
                    bs=cache['base'][:,:3];es=score[name][:,:3];uu=np.array([r['unknown_index'] for r in vs]);rows=np.arange(len(vs))
                    bb=bs.copy();ee=es.copy();bb[rows,uu]=-np.inf;ee[rows,uu]=-np.inf
                    delta=(es[rows,uu]-logsumexp(ee,axis=1))-(bs[rows,uu]-logsumexp(bb,axis=1))
                    entities=np.array([[j for j in range(3) if j!=u] for u in uu])
                    mm['unknown_log_odds_mean_change_ambig']=float(delta[yy].mean())
                    mm['entity_preference_change_ambig']=float(np.mean((bb.argmax(1)!=ee.argmax(1))[yy]))
                methods[name]=mm
                for r,pp,t in zip(vs,p,masks[name]):
                    f.write(json.dumps({'uid':r['uid'],'method':name,'prediction':int(pp),'trigger':bool(t),'label':r['label']})+'\n');prediction_count+=1
        b=max(s.c['fit_budgets']);pairs=[(f'{b}_product_steer1.5','fixed_local_fs',None),
            (f'{b}_product_direct',f'{b}_product_steer1.5',None),
            (f'{b}_product_steer1.5',f'{b}_expanded_steer1.5',None),
            (f'{b}_product_steer1.5','base','disambig')]
        for a,bb,cond in pairs:
            primary[e.name+':'+a+'-'+bb+(':'+cond if cond else '')]=contrast(vs,pred[a],pred[bb],s.c['bootstrap_replicates'],condition=cond)
        result[e.name]={'methods':methods,'recognition':probe_report(vs,cache['features'],dict(np.load(e.out/'probes.npz')),s.c['fit_budgets']),
            'counterfactual':diagnostic(e,'counterfactual_'+e.name),'stress':diagnostic(e,'stress'),
            'probe_fit':read(e.out/'probe_fit.json'),'selection':read(e.out/'selection_frozen.json')}
        write(e.out/'results.json',result[e.name])
    holm(primary);surface=surface_controls(s)
    output_controls=read(s.out/'output_controls/results.json') if (s.out/'output_controls/results.json').exists() else None
    route_path=s.out/'surface_routes_v2/results.json'
    if not route_path.exists():route_path=s.out/'surface_routes/results.json'
    surface_routes=read(route_path) if route_path.exists() else None
    write(s.out/'results.json',{'models':result,'primary_comparisons':primary,'surface_controls':surface,
                              'output_controls':output_controls,'surface_routes':surface_routes,'prediction_records':prediction_count})
    lines=['# 证据感知：同预算消融与第二模型复验','',
        '两模型使用相同2200道fit、352道dev、704道test，raw zero-shot；与旧zero/few-shot合并成绩不可直接比较。历史测试已观察，全部为探索性复验。',
        '固定深度：Qwen3 L18、MiniCPM5 L28，最后提示token。证据标签是额外监督；相同样本数不等于相同监督信息。原始正确性头仅使用可标为正确/刻板错误的子集，实际行数见probe_fit。',
        '主要发现：Qwen3的乘积门控方向与同门控direct均为73.15%；MiniCPM5则为43.04%与83.66%。单证据头direct分别79.40%与84.38%。这支持分离识别与动作，但不能据固定层/方向结果否定所有表示干预。',
        '等长压力集上，证据头AUROC为0.9690/0.9520，表面特征为0.4703/0.4773；同时填充使证据头召回下降，显示排序信息与阈值稳健性并不相同。','',
        '|模型/方法|总体/%|不足/%|充分/%|BS(a)×100|充分弃答/%|MMLU子集/%|',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels=[('base','原模型'),('fixed_local_fs','固定本地FairSteer风格'),('2200_original_steer1.5','原错误门控+方向（统一阈值规则）'),('2200_expanded_steer1.5','完整错误门控+方向'),
        ('2200_evidence_steer1.5','单证据门控+方向'),('2200_product_steer1.5','乘积门控+方向'),
        ('2200_product_direct','同乘积门控直接unknown'),('2200_product_offset1.5','同乘积门控unknown偏置'),('2200_evidence_direct','单证据门控直接unknown')]
    for model,item in result.items():
        for key,title in labels:
            m=item['methods'][key];bs=m['ambig']['bias_score']
            lines.append(f'|{model}/{title}|{m["accuracy"]*100:.2f}|{m["ambig"]["accuracy"]*100:.2f}|{m["disambig"]["accuracy"]*100:.2f}|{bs*100 if bs is not None else float("nan"):.2f}|{m["sufficient_unknown_rate"]*100:.2f}|{m["utility_accuracy"]*100:.2f}|')
    lines+=['','## 预先声明的八项比较','', '|比较|差值pp|95%家族区间pp|Holm p|','|---|---:|---|---:|']
    for k,v in primary.items():lines.append(f'|{k}|{v["difference_pp"]:+.2f}|[{v["ci95_pp"][0]:+.2f},{v["ci95_pp"][1]:+.2f}]|{v["holm_p"]:.4f}|')
    lines+=['','## 训练预算与识别','',
        '曲线只改变线性头的训练题数；方向始终由固定110条大fit子集样本估计，直接unknown不需要该方向。',
        '|模型/头训练预算|证据AUROC|乘积门控方向准确率/%|同门控direct/%|','|---|---:|---:|---:|']
    for model,item in result.items():
        for b in s.c['fit_budgets']:
            lines.append(f'|{model}/{b}|{item["recognition"][str(b)+"_evidence"]["auroc"]:.4f}|{item["methods"][str(b)+"_product_steer1.5"]["accuracy"]*100:.2f}|{item["methods"][str(b)+"_product_direct"]["accuracy"]*100:.2f}|')
    lines+=['','## 同预算表面与输出校准对照','',
        f'原生BBQ测试上，长度/标点等表面特征的AUROC为 **{surface["surface"]["test"]["auroc"]:.4f}**，question-only为 **{surface["question_only"]["test"]["auroc"]:.4f}**。表面线索已经很强，不能仅凭隐藏头高AUROC声称识别到纯语义证据。',
        '输出校准是在主推理启动后、主结果汇总前单独冻结的探索性补充；只用同一fit/dev训练选阈值。低误触发约束可能使它几乎不动作，并不表示所有输出校准都无效。','',
        '|模型/只用输出的分类器|测试AUROC|直接unknown准确率/%|充分准确率/%|','|---|---:|---:|---:|']
    if output_controls:
        for name,item in output_controls['results'].items():
            m=item['test'];dm=m['groups']['all']['methods']['direct'];lines.append(f'|{name}|{m["auroc"]:.4f}|{dm["accuracy"]*100:.2f}|{dm["disambig"]["accuracy"]*100:.2f}|')
    if surface_routes:
        lines+=['','表面对照的直接路由在看到主结果与表面AUROC后单独冻结，属于事后探索性诊断；训练和阈值仍只用fit/dev。','',
            '|模型/直接unknown路由|总体/%|不足/%|充分/%|等长不足触发/%|','|---|---:|---:|---:|---:|']
        for model,data in surface_routes['results'].items():
            for kind,ds in data.items():
                m=ds['test']['all']['metrics'];recall=ds['counterfactual_'+model]['neutral_matched']['insufficient_recall']
                lines.append(f'|{model}/{kind}|{m["accuracy"]*100:.2f}|{m["ambig"]["accuracy"]*100:.2f}|{m["disambig"]["accuracy"]*100:.2f}|{recall*100:.2f}|')
    lines+=['','## 证据与等长中性对照','',
        '每模型88对原生不足/充分上下文；中性填充使不足版本与充分版本的完整提示token数逐对完全相同。它是人工压力诊断，不能代表自然分布泛化。','',
        '|模型|原不足证据概率|原充分概率|等长不足概率|充分+中性概率|等长不足>充分比例|','|---|---:|---:|---:|---:|---:|']
    for model,item in result.items():
        h=item['counterfactual']['head'];pp=item['counterfactual']['paired']['evidence']
        vals=[h[k]['evidence']['mean_probability'] for k in ('insufficient','sufficient','neutral_matched','sufficient_neutral')]
        lines.append('|'+model+'|'+'|'.join(f'{v:.4f}' for v in vals)+f'|{pp["equal_length_insufficient_above_sufficient_fraction"]*100:.2f}%|')
    lines+=['','|模型的等长子集|隐藏证据头AUROC|表面特征AUROC|','|---|---:|---:|']
    for model,item in result.items():
        lines.append(f'|{model}|{item["counterfactual"]["equal_length_recognition"]["evidence"]:.4f}|{surface["surface"]["counterfactual_"+model]["equal_length_auroc"]:.4f}|')
    lines+=['','等长填充仍明显降低不足题的证据概率和阈值召回；排序区分与固定阈值鲁棒性必须分开解释。该构造仍保留措辞、字符长度、标点等差异，不能排除所有表面捷径。']
    lines+=['','## 选项与unknown措辞压力检验','',
        '|模型|证据头平均概率范围|原模型语义预测改变/%|乘积方向/%|同门控direct/%|','|---|---:|---:|---:|---:|']
    for model,item in result.items():
        cc=item['stress']['consistency'];lines.append(f'|{model}|{cc["evidence_head"]["mean_probability_range"]:.4f}|{cc["base"]["any_semantic_prediction_change"]*100:.2f}|{cc["2200_product_steer1.5"]["any_semantic_prediction_change"]*100:.2f}|{cc["2200_product_direct"]["any_semantic_prediction_change"]*100:.2f}|')
    lines+=['','|模型/压力集版本|原模型准确率/%|乘积方向/%|同门控direct/%|','|---|---:|---:|---:|']
    for model,item in result.items():
        for version,m in item['stress']['methods']['base'].items():
            ds=item['stress']['methods'];lines.append(f'|{model}/{version}|{m["accuracy"]*100:.2f}|{ds["2200_product_steer1.5"][version]["accuracy"]*100:.2f}|{ds["2200_product_direct"][version]["accuracy"]*100:.2f}|')
    lines+=['','## 解释边界','',
        '- 所有门控阈值只在dev选择，以充分题误触发≤2%的点估计约束最大化不足召回；并非统计非劣或测试分布保证。',
        '- 主方向强度1.5预先固定，全部预算/门控结果保留，不按test挑选方法。偏置在dev匹配方向的unknown频率。',
        '- direct与方向使用逐题完全相同的掩码；直接unknown不提供完整概率分布。',
        '- 表面/问题词袋基线与隐藏头使用完全相同2200题，固定参数、无目标训练。',
        '- 充分题准确率和必要能力需要单独报告；direct在无unknown选项的MMLU上不动作，不能据此宣称一般能力无损。',
        '- 本轮验证识别与决策的功能分解，不以探针可读性宣称发现模型原有的因果证据电路。',
        '- 原始冻结结果保持不变；本轮训练预算、提示与阈值协议改变，不以新数字替换旧结果。','']
    if (s.out/'surface_routes_v2/previous_version_comparison.json').exists():
        comparison=read(s.out/'surface_routes_v2/previous_version_comparison.json')
        diffs=sum(v['prediction_differences'] for v in comparison.values());prob=max(v['max_probability_difference'] for v in comparison.values())
        lines+=['## 报告修复记录','',
            '独立审计发现早期报告重绘重新保存了表面基线joblib，改变了该辅助对照的文件指纹。现已改为复用拟合资产，并在surface_routes_v2单独固定修复后的来源；旧计划、输出和失败日志均保留。',
            f'表面对照v1/v2最大概率差为{prob:g}，预测变化{diffs}条。主模型的头、方向、开发选择和所有主结果未改变。','']
    (s.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (Path('research')/'2026-10-17_evidence_story_findings.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'reported':True,'predictions':prediction_count,'primary_comparisons':primary}),flush=True)


if __name__=='__main__':main()
