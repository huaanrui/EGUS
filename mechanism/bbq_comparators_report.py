"""Read-only summaries/audits for the frozen external comparator pilot."""
from __future__ import annotations
import csv
import numpy as np
import torch
from .fairsteer_author import ROOT,metrics,read,sha,write,prompt
from .fairsteer_improve_report import cluster_intervals

SUBSETS={'knowbias':['Gender_identity','Race_ethnicity','Religion'],'crispr':['Age','SES','Disability_status'],'bba':['Gender_identity','Nationality','Religion']}


def audit(pilot,records):
    frozen=read(pilot.out/'plan_frozen.json')
    for p,h in frozen['fingerprint']['sources'].items():
        assert sha(ROOT/p)==h and sha(pilot.out/'source_snapshot'/p)==h
    for p,h in frozen['fingerprint']['inputs'].items():assert sha(ROOT/p)==h
    selection=read(pilot.out/'selection_frozen.json')
    assert selection['selection_sha256']==sha(pilot.out/'selection_results.json')
    for p,h in selection['fitted_sha256'].items():assert sha(pilot.out/p)==h
    assert set(read(pilot.out/'crispr_examples.json')['indices'])<=set(pilot.splits['fit'])
    assert {r['name'] for r in records}==set(selection['selected'])
    count=0
    for record in records:
        assert record['spec']==selection['selected'][record['name']]['spec']
        for utility,key in ((False,'bbq'),(True,'utility')):
            saved=record[key];scores=np.asarray(saved['scores']);pred=np.asarray(saved['predictions'])
            assert np.isfinite(scores).all() and np.array_equal(scores.argmax(1),pred)
            expected=pilot.score('test',scores,utility)
            assert expected==saved['metrics']
            count+=len(pred)
    smoke=read(pilot.out/'smoke.json');assert smoke['node_hooks_change_and_restore']
    result={'passed':True,'prediction_rows':count,'source_and_input_hashes_verified':True,'fit_families_only':True,'selection_frozen_before_test':True,'all_metrics_recomputed':True,'historical_exposure':'Exploratory: test already observed in earlier experiments.'}
    write(pilot.out/'independent_audit.json',result);return result


def gpu_readout_audit(pilot,records):
    path=pilot.out/'gpu_readout_audit.json'
    if path.exists():return read(path)
    pilot.setup();r=pilot.runner;checks=[];rankings=read(pilot.out/'readout_rankings.json')
    examples=pilot.examples('dev');cache=pilot.cache_readout('dev')
    for record in records:
        spec=record['spec']
        if spec['method'] not in ('bba','fba'):continue
        ordering=rankings['unified_'+spec['method']];indices=ordering[:int(len(ordering)*spec['ratio'])]
        expected=pilot.evaluate_readout('dev',spec)
        for j in (0,len(examples)//2):
            row,shot,_=examples[j]
            def hook(module,args,output):
                value=output[0] if isinstance(output,tuple) else output
                changed=value.clone();changed[:,-1,indices]=spec['constant']
                return (changed,*output[1:]) if isinstance(output,tuple) else changed
            module=r.layers[-1] if spec['hook']=='pre' else r.model.model.norm
            handle=module.register_forward_hook(hook)
            try:scores=r.forward(prompt(row,shots=shot))[1]
            finally:handle.remove()
            error=float(np.max(np.abs(scores-expected[j])))
            assert error<=.04 and scores.argmax()==expected[j].argmax(),(spec['id'],j,error)
            checks.append({'method':record['name'],'uid':row['uid'],'shot':shot,'max_logit_error':error,'same_prediction':True})
    result={'passed':True,'checks':checks,'interpretation':'Cached final-state readout modification checked against actual whole-model hooks.'}
    write(path,result);return result


def knowledge_behavior_check(pilot,records):
    """Diagnostic on the 45 localization questions, not held-out accuracy."""
    path=pilot.out/'knowledge_behavior_diagnostic_v2.json'
    if path.exists():return read(path)
    pilot.setup();r=pilot.runner;spec=next(x['spec'] for x in records if x['name']=='knowbias')
    questions=read(pilot.out/'knowledge_questions.json');results=[]
    ids=[r.tokenizer.encode(x,add_special_tokens=False)[0] for x in ('no','yes')]
    for question in questions:
        text=pilot.knowledge_prompt(question['question']);target=int(question['target']=='yes');values=[]
        for edited in (False,True):
            handle=pilot.install_nodes('knowbias',spec) if edited else None
            try:
                with torch.inference_mode():output=r.model(input_ids=pilot.ids(text),use_cache=False,logits_to_keep=1)
                logits=output.logits[0,-1].float();prob=logits.softmax(-1)
                values.append({'target_probability':float(prob[ids[target]]),'lowercase_yes_no_choice_correct':bool(logits[ids].argmax().item()==target)})
            finally:
                if handle:handle.remove()
        results.append({'question':question,'baseline':values[0],'enhanced':values[1]})
    summary={key:{'mean_target_probability':float(np.mean([x[key]['target_probability'] for x in results])),'lowercase_yes_no_choice_accuracy':float(np.mean([x[key]['lowercase_yes_no_choice_correct'] for x in results]))} for key in ('baseline','enhanced')}
    result={'scope':'Post-hoc diagnostic on the same 45 localization questions. Forced choice between lowercase no/yes tokens, not free-generation semantic accuracy or held-out generalization; never used in selection.','summary':summary,'rows':results}
    write(path,result);return result


def report(pilot):
    if not (pilot.out/'results.json').exists():return
    records=read(pilot.out/'results.json');checked=audit(pilot,records);gpu=gpu_readout_audit(pilot,records);knowledge=knowledge_behavior_check(pilot,records)
    previous=read(pilot.parent/'results.json');ref=next(r for r in previous if r['name']=='fairsteer');evidence=next(r for r in previous if r['name']=='evidence')
    combined=[]
    for old in previous:
        if old['name'] in ('baseline','fairsteer','tuned','evidence'):
            rec={'name':old['name'],'spec':old['bbq']['spec'],'bbq':old['bbq'],'utility':old['utility'],'dev_failures':read(pilot.parent/'test_plan_frozen.json')['selected'].get(old['name'],{}).get('failures',[])}
            combined.append(rec)
    combined+=records;summary=[];examples=pilot.examples('test');base=combined[0]
    for rec in combined:
        m=rec['bbq']['metrics'];pred=np.array(rec['bbq']['predictions']);labels=np.array([r['label'] for r,_,_ in examples]);bp=np.array(base['bbq']['predictions']);um=rec['utility']['metrics']
        ci_ref=cluster_intervals(pilot,pred,ref['bbq']['predictions']);ci_ev=cluster_intervals(pilot,pred,evidence['bbq']['predictions'])
        dis=np.array([r['context_condition']=='disambig' for r,_,_ in examples])
        row={'name':rec['name'],'accuracy':m['accuracy'],'ambig_accuracy':m['ambig']['accuracy'],'disambig_accuracy':m['disambig']['accuracy'],'bias_ambig':m['ambig']['bias_score'],'bias_disambig':m['disambig']['bias_score'],'disambig_unknown':m['native_disambig_unknown_rate'],'utility_accuracy':um['accuracy'],'delta_pp_vs_fairsteer':100*(m['accuracy']-ref['bbq']['metrics']['accuracy']),'delta_pp_vs_evidence':100*(m['accuracy']-evidence['bbq']['metrics']['accuracy']),'ci_pp_vs_fairsteer':ci_ref,'ci_pp_vs_evidence':ci_ev,'dev_failures':rec['dev_failures'],'disambig_correct_to_wrong':int(((bp==labels)&(pred!=labels)&dis).sum()),'disambig_wrong_to_correct':int(((bp!=labels)&(pred==labels)&dis).sum()),'parameters':{k:v for k,v in rec['spec'].items() if k!='nodes'},'node_count':len(rec['spec'].get('nodes',[])),'shots':{},'subsets':{}}
        for shot in (False,True):
            ix=[j for j,(_,s,_) in enumerate(examples) if s==shot];row['shots']['fewshot' if shot else 'zeroshot']=metrics([examples[j][0] for j in ix],pred[ix])
        for name,categories in SUBSETS.items():
            ix=[j for j,(r,_,_) in enumerate(examples) if r['category'] in categories]
            row['subsets'][name]=metrics([examples[j][0] for j in ix],pred[ix])
        # Different test reference constraints are descriptive, not reselection.
        failures=[]
        if m['disambig']['accuracy']<max(base['bbq']['metrics']['disambig']['accuracy']-.01,ref['bbq']['metrics']['disambig']['accuracy']-.005)-1e-12:failures.append('disambig')
        if um['accuracy']<base['utility']['metrics']['accuracy']-.02-1e-12:failures.append('utility')
        if m['ambig']['bias_score'] is None or abs(m['ambig']['bias_score'])>abs(ref['bbq']['metrics']['ambig']['bias_score'])+.01+1e-12:failures.append('bias')
        row['test_failures']=failures;summary.append(row)
    write(pilot.out/'summary.json',summary)
    fields=['name','accuracy','ambig_accuracy','disambig_accuracy','bias_ambig','bias_disambig','disambig_unknown','utility_accuracy','delta_pp_vs_fairsteer','delta_pp_vs_evidence','dev_failures','test_failures','disambig_correct_to_wrong','disambig_wrong_to_correct','node_count','parameters']
    with (pilot.out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in summary)
    lines=['# BBQ 外部方法比较：探索性结果','','Qwen3-1.7B / FP16；共同 test 704 道题，每题 zero/few-shot 两次，共 1408 条输出；dev 352 道题。原始 A/B/C 提示，单题推理；历史 test 已被观察。','',
           '|方法|总体/%|歧义/%|消歧/%|BS(a)×100|MMLU 子集/%|相对证据门控/pp|dev 约束失败|',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summary:
        bias='NA' if r['bias_ambig'] is None else f"{100*r['bias_ambig']:.2f}"
        lines.append(f"|{r['name']}|{100*r['accuracy']:.2f}|{100*r['ambig_accuracy']:.2f}|{100*r['disambig_accuracy']:.2f}|{bias}|{100*r['utility_accuracy']:.2f}|{r['delta_pp_vs_evidence']:+.2f}|{','.join(r['dev_failures']) or '通过'}|")
    lines+=['','## 结果解释边界','','KnowBias 为公开知识题重建；CRISPR 为 Qwen MLP/字母答案/两种 shot 适配；FBA/BBA 为合成提示重建及统一跨维度节点适配。不能称这些方法原配置完全复现。外部知识/候选词监督、定位量与候选预算不同。完整差异见 protocol_frozen.md。',
            '', 'FBA/BBA post=最终 norm 后的读出输入干预；pre=作者示例代码的最后 block 输出、norm 前干预。两种结果并列，未按 test 选版本。',
            '', '某方法没有 dev 可行候选时，仍按冻结规则显示整体准确率最好的探索候选；不能把失败候选描述为满足公平—效用约束。MMLU 仅 128 道作者筛选问题，不代表全面能力。',
            '', 'unknown_force/unknown_offset 使用相同证据头，分别直接选择 unknown 或动态修改其输出分数；不使用测试答案/真实歧义条件。它们是检验证据门控机制主张的强对照。',
            '', '## 配对区间（相对证据门控）','','类别内家族配对 bootstrap 2000 次，zero/few-shot 绑定；多方法未经多重比较校正。','']
    for r in summary:
        interval=r['ci_pp_vs_evidence']['overall'];lines.append(f"- {r['name']}: {r['delta_pp_vs_evidence']:+.2f}pp，95% [{interval[0]:+.2f}, {interval[1]:+.2f}]。")
    lines+=['','## 论文覆盖类别子集','','使用全类别统一组件，以下不是作者逐类别组件复现。','']
    for subset,categories in SUBSETS.items():
        lines+=['### '+subset+'：'+', '.join(categories),'','|方法|总体/%|歧义/%|消歧/%|','|---|---:|---:|---:|']
        for r in summary:
            m=r['subsets'][subset];lines.append(f"|{r['name']}|{100*m['accuracy']:.2f}|{100*m['ambig']['accuracy']:.2f}|{100*m['disambig']['accuracy']:.2f}|")
        lines+=['']
    lines+=['## 参数与审计','']
    for r in summary:lines.append(f"- {r['name']}: {r['parameters']}；节点数 {r['node_count']}；消歧正确→错误 {r['disambig_correct_to_wrong']} / 错误→正确 {r['disambig_wrong_to_correct']}。")
    lines+=['',f"独立数值审计：{checked['prediction_rows']} 条预测通过；读出 GPU 实际 hook 核验 {len(gpu['checks'])} 条通过。",'',f"KnowBias 定位题行为诊断（45 条定位题；仅在小写 no/yes token 间强制选择，不是自由生成语义准确率或持出成绩）：{knowledge['summary']}。该诊断未参与选参，也不能单凭目标 token 概率上升断言偏见知识增强。",'', '本轮没有完成跨算子门控组合、多种子、多模型及因果电路验证，不能据此主张跨干预通用性。']
    domain_path=pilot.out/'domain_reference_v2/results.json'
    if domain_path.exists():
        lines+=['','## 已完成的按类别参考轨道','','性别/国籍/宗教分别定位、分别在各自 dev 选参；额外类别路由信息已披露。共同192道题、384条输出，与上述全11类结果不能直接横比。完整参数、失败标记与区间见 [按域报告](domain_reference_v2/report.md)。','','|方法|三个类别总体/%|歧义/%|消歧/%|','|---|---:|---:|---:|']
        for rec in read(domain_path):
            m=rec['metrics'];lines.append(f"|{rec['name']}|{100*m['accuracy']:.2f}|{100*m['ambig']['accuracy']:.2f}|{100*m['disambig']['accuracy']:.2f}|")
    (pilot.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(pilot.out,summary)
    pilot.log('report_complete',path=str(pilot.out/'report.md'))


def plot(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    positions={'baseline':(70,49.5),'bba_post':(70,41),'bba_pre':(70,44),'fba_post':(70,47),'fba_pre':(70,53),'knowbias':(62,57),'crispr':(50,35),'tuned':(63,66),'fairsteer':(61,76),'evidence':(69,90),'unknown_force':(69,100),'unknown_offset':(69,83)}
    for r in summary:
        color='#c44e52' if r['dev_failures'] else '#4c72b0'
        if r['name']=='evidence':color='#2a9d8f'
        axes[0].scatter(100*r['disambig_accuracy'],100*r['ambig_accuracy'],color=color,s=60)
        axes[0].annotate(r['name'],(100*r['disambig_accuracy'],100*r['ambig_accuracy']),xytext=positions[r['name']],textcoords='data',fontsize=8,arrowprops={'arrowstyle':'-','color':'#aaa','lw':.7})
    axes[0].set(xlabel='Disambiguated accuracy (%)',ylabel='Ambiguous accuracy (%)',title='BBQ accuracy trade-off',xlim=(47,78),ylim=(30,104))
    axes[0].plot([],[],'o',color='#c44e52',label='New candidate failed dev constraints');axes[0].legend(loc='upper left',fontsize=8,frameon=False)
    selected=[r for r in summary if r['name'] not in ('baseline','fairsteer','tuned','evidence')]
    for j,r in enumerate(selected):
        lo,hi=r['ci_pp_vs_evidence']['overall'];axes[1].plot([lo,hi],[j,j],color='#4c72b0');axes[1].scatter(r['delta_pp_vs_evidence'],j,color='#4c72b0')
    axes[1].set_yticks(range(len(selected)),[r['name'] for r in selected]);axes[1].axvline(0,color='#888',ls='--');axes[1].set(xlabel='Accuracy difference vs evidence gate (pp)',title='Paired family bootstrap 95% intervals')
    fig.suptitle('Qwen3-1.7B | exploratory BBQ comparison on a shared test split')
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
