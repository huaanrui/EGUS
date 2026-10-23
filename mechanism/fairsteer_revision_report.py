"""Display paired protocol correction and unseen-category confirmation separately."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def pct(x):return f"{100*x:.2f}%"


def correction_contrasts(out,prior,policy,data,confirmation):
    """Descriptive fixed-layer contrasts; no model calls or configuration selection."""
    import numpy as np
    def load_rows(path):return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records=[]
    for scope,source in (("paired",load_rows(data/"test.jsonl")),("confirmation",load_rows(confirmation/"confirmation.jsonl"))):
        categories=sorted({r["category"] for r in source})
        for spec in policy["previous_specs"]:
            identifier=spec["id"]
            a=(prior/"predictions"/f"test_{identifier}.json") if scope=="paired" else (out/"predictions"/f"confirmation_historical_{identifier}.json")
            candidates=[out/"predictions"/f"test_{identifier}.json",out/"predictions"/f"paired_{identifier}.json"]
            c=next(p for p in candidates if p.exists()) if scope=="paired" else out/"predictions"/f"confirmation_corrected_{identifier}.json"
            old={p["uid"]:p for p in read(a)["predictions"]};new={p["uid"]:p for p in read(c)["predictions"]}
            keys=("macro_stereotype","macro_counterstereotype","macro_ambiguous_accuracy","disambiguated_accuracy","disambiguated_overabstention")
            delta=[]
            for r in source:
                indices=(r["stereotype_index"],r["counterstereotype_index"],r["correct_index"],r["correct_index"],r["unknown_index"])
                delta.append([sum(rot["prediction"]==index for rot in new[r["uid"]]["rotations"])/6-sum(rot["prediction"]==index for rot in old[r["uid"]]["rotations"])/6 for index in indices])
            delta=np.asarray(delta)
            rng=np.random.default_rng(20261012);samples=np.zeros((2000,5));point=np.zeros(5)
            dis_count=sum(r["context_condition"]=="disambig" for r in source)
            dis_sum=np.zeros((2000,2));dis_n=np.zeros(2000)
            for cat in categories:
                groups={r["template_group_id"] for r in source if r["category"]==cat}
                sums=[];counts=[]
                for group in sorted(groups):
                    amb=[i for i,r in enumerate(source) if r["template_group_id"]==group and r["context_condition"]=="ambig"]
                    dis=[i for i,r in enumerate(source) if r["template_group_id"]==group and r["context_condition"]=="disambig"]
                    sums.append(np.r_[delta[amb,:3].sum(0),delta[dis,3:].sum(0)])
                    counts.append((len(amb),len(dis)))
                sums,counts=np.asarray(sums),np.asarray(counts)
                picks=rng.integers(len(groups),size=(2000,len(groups)))
                totals=sums[picks].sum(1);denoms=counts[picks].sum(1)
                samples[:,:3]+=totals[:,:3]/denoms[:,0,None]/len(categories)
                point[:3]+=sums[:,:3].sum(0)/counts[:,0].sum()/len(categories)
                dis_sum+=totals[:,3:];dis_n+=denoms[:,1]
                point[3:]+=sums[:,3:].sum(0)/dis_count
            samples[:,3:]=dis_sum/dis_n[:,None]
            records.append(dict(scope=scope,id=identifier,contrast="corrected ambiguous-only minus historical mixed evidence; fixed layer, alpha and probe",metrics={key:dict(delta=float(point[j]),low=float(np.quantile(samples[:,j],.025)),high=float(np.quantile(samples[:,j],.975))) for j,key in enumerate(keys)}))
    (out/"protocol_correction_contrasts.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return records


def direction_geometry(out,prior,policy):
    import numpy as np
    import hashlib
    fitted=[]
    for folder in (prior,out):
        path=folder/"fitted.pkl"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==read(folder/"fitted.pkl.sha256.json")["sha256"]
        with path.open("rb") as handle:fitted.append(pickle.load(handle))
    values=[]
    for spec in policy["previous_specs"]:
        a,c=(item["directions"]["native"][spec["layer"]].astype(np.float64) for item in fitted)
        na,nc=float(np.linalg.norm(a)),float(np.linalg.norm(c))
        values.append(dict(layer=spec["layer"],historical_norm=na,corrected_norm=nc,norm_ratio=nc/na,cosine=float(a@c/na/nc)))
    (out/"direction_correction_geometry.json").write_text(json.dumps(values,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return values


def render_revision(out,data,confirmation):
    from .fairsteer_report import render
    render(out,data)
    prior=out.parent/"fairsteer_20261012_v1"
    policy=read(out/"revision_policy_frozen.json")["policy"]
    plan=read(out/"test_plan_frozen.json")
    old=read(prior/"test_results.json")
    paired=read(out/"test_results.json")+read(out/"paired_extra_results.json")
    fresh=read(out/"confirmation_results.json")
    b=read(out/"predictions/test_baseline.json")["metrics"]
    f=read(out/"predictions/confirmation_baseline.json")["metrics"]
    calibration=read(out/"confirmation_calibration.json")
    contrasts=correction_contrasts(out,prior,policy,data,confirmation)
    geometry=direction_geometry(out,prior,policy)
    selected=[plan["primary"]]+([plan["guarded"]["spec"]] if plan["guarded"] else [])
    lines=["**FairSteer 方向条件修正：完整结果**","",
           "本轮仅将方向训练问题从混合证据条件改为110条模糊题，正确端全部为unknown。模型、提示、检测器、训练/开发/原320题、干预算子、搜索和能力约束保持不变。检测器系数精确复用，经输入哈希与特征代码AST校验。", "",
           f"开发集冻结主参考：`{plan['primary']['id']}`；严格筛选配置：`{plan['guarded']['spec']['id'] if plan['guarded'] else '无可行候选'}`，其开发集至少1pp改善条件{'达到' if plan['guarded'] and plan['guarded']['accepted'] else '未达到'}。两者角色不同，不能据测试更换主要方案。", "",
           "原320题已在上一轮开封，以下仅为配对诊断；固定层17/18同时保留修正前后结果，以避免把重新选层的收益当作方向条件本身的作用。", "",
           "|原320题配对诊断|刻板率|反刻板率|模糊题正确率|明确题正确率|明确题弃答|能力约束|",
           "|---|---:|---:|---:|---:|---:|---|",
           row("原模型",b,None)]
    if plan["guarded"]:
        guarded=next(r for r in fresh if r["spec"]["id"]==plan["guarded"]["spec"]["id"] and "corrected" in r["prediction_name"])
        gain=f['macro_stereotype']-guarded['metrics']['macro_stereotype']
        qualifies=bool(guarded['feasible'] and gain>=.01)
        finding=f"开发集提前冻结的严格候选 `{guarded['spec']['id']}` 在独立新增类别上{'达到' if qualifies else '未达到'}原定至少1pp改善及其余约束：刻板率{pct(f['macro_stereotype'])}→{pct(guarded['metrics']['macro_stereotype'])}，明确题正确率{pct(f['disambiguated_accuracy'])}→{pct(guarded['metrics']['disambiguated_accuracy'])}。这与α=1主参考的结果分别报告，不能据新增确认替换主参考或扩展选参。"
        lines[2:2]=[finding,""]
    for s in policy["previous_specs"]:
        a=next(r for r in old if r["spec"]["id"]==s["id"])
        c=next(r for r in paired if r["spec"]["id"]==s["id"])
        lines += [row("混合方向 "+s["id"],a["metrics"],a["feasible"]),row("仅模糊题 "+s["id"],c["metrics"],c["feasible"])]
    shown={s["id"] for s in policy["previous_specs"]}
    for s in selected:
        if s["id"] not in shown:
            c=next(r for r in paired if r["spec"]["id"]==s["id"])
            lines.append(row("本轮冻结 "+s["id"],c["metrics"],c["feasible"]));shown.add(s["id"])
    m=read(confirmation/"manifest.json")
    lines += ["",f"独立新增确认集：{m['rows']}题、{m['families']}个家族，Sexual_orientation、Race_x_gender、Race_x_SES 各128题，模糊/明确各半。类别未进入本轮方向和检测器训练，故这是跨类别新家族确认；共享示例涉及的家族已排除。能力MMLU另有128题，与之前四份问题集合隔离。", "",
              "|独立新增类别确认|刻板率|变化/pp（95%区间）|反刻板率|模糊题正确率|明确题正确率|MMLU|能力约束|",
              "|---|---:|---|---:|---:|---:|---:|---|",
              f"|原模型|{pct(f['macro_stereotype'])}|—|{pct(f['macro_counterstereotype'])}|{pct(f['macro_ambiguous_accuracy'])}|{pct(f['disambiguated_accuracy'])}|{pct(read(out/'predictions/confirmation_baseline_mmlu.json')['accuracy'])}|—|"]
    for r in fresh:
        ci=r["confidence"]["macro_stereotype"];v=r["metrics"]
        label=("混合方向 " if "historical" in r["prediction_name"] else "匹配随机方向 " if r['spec']['variant']=='random_gated' else "仅模糊题 ")+r["spec"]["id"]
        lines.append(f"|{label}|{pct(v['macro_stereotype'])}|{100*ci['delta']:+.2f} [{100*ci['low']:+.2f}, {100*ci['high']:+.2f}]|{pct(v['macro_counterstereotype'])}|{pct(v['macro_ambiguous_accuracy'])}|{pct(v['disambiguated_accuracy'])}|{pct(r['mmlu_accuracy'])}|{'通过' if r['feasible'] else ';'.join(r['violations'])}|")
    lines += ["","固定层修正差值（仅模糊题方向减去历史混合方向）：以下为既定比较的描述性家族bootstrap，不用于选层或选强度。刻板率变化越负表示下降更多，明确题正确率变化越负表示额外能力损失。","",
              "|集合/固定层|刻板率差/pp（95%区间）|明确题正确率差/pp（95%区间）|","|---|---|---|"]
    for r in contrasts:
        a,c=(r['metrics'][key] for key in ('macro_stereotype','disambiguated_accuracy'))
        label=('原320题配对' if r['scope']=='paired' else '新增类别确认')+' / '+r['id']
        lines.append(f"|{label}|{100*a['delta']:+.2f} [{100*a['low']:+.2f}, {100*a['high']:+.2f}]|{100*c['delta']:+.2f} [{100*c['low']:+.2f}, {100*c['high']:+.2f}]|")
    lines += ["","方向几何（原始均值差，未归一化）：","","|层|历史方向长度|修正方向长度|长度比|余弦相似度|","|---|---:|---:|---:|---:|"]
    for g in geometry:lines.append(f"|{g['layer']}|{g['historical_norm']:.3f}|{g['corrected_norm']:.3f}|{g['norm_ratio']:.3f}|{g['cosine']:.3f}|")
    lines += ["","α=1相同不意味着注入向量长度相同。方向证据条件修正同时改变方向内容和原始长度；本轮不能分离两者的贡献。正负号和新方向内部的平衡/等长对照已按原规则运行，不能据此声称已经控制历史方向与修正方向之间的尺度。","",
              f"unknown校准在开发集冻结的偏置为{calibration['offset']:g}；新增确认刻板率{pct(calibration['metrics']['macro_stereotype'])}，明确题正确率{pct(calibration['metrics']['disambiguated_accuracy'])}，能力约束{'通过' if calibration['feasible'] else '未通过'}。该对照没有在新增确认上重新选强度。", "",
              "所有区间按类别内模板家族配对bootstrap2000次；同题六种排列绑定。主效果表的区间是各配置相对该集合原模型的变化；固定层修正差值表的区间才是两方向版本的直接配对比较。新增类别结果未用于重选方法；全部原始效果和未通过原因保存。", "",
              "**解释范围**", "",
              "应先比较相同模型、相同层和强度、相同评价问题的混合/仅模糊题方向，再看本轮重新选层的冻结配置。独立新增类别同时评价历史拟合产物与修正拟合产物，不能把更换测试类别当作修正收益。明确题能力下降或弃答增加应计入方法局限，不能以模型或划分变化自动解释掉。", "",
              "三个随机方向严格匹配主参考的层、长度和强度；如果严格候选与主参考的层或强度不同，这三个随机对照不能证明该严格候选优于其自身匹配的随机方向。严格候选通过预设效果/能力阈值，也不等于确认了社会偏见的选择性机制。", "",
              f"新增确认原模型A/B/C候选在完整词表中的平均概率质量为{pct(f['candidate_mass'])}。本轮保留裸文本与BF16以控制变量，仍只支持强制选项评价；不据此推断自由生成偏见改善。批次稳定性另见主报告。", "",
              "方向协议已对齐论文附录D.1；六类方向样本、更小检测器训练预算、sklearn、Qwen3-1.7B、粗筛预算等本地差异仍保留，不能称为全论文原配置复现。", "",
              "来源：[FairSteer论文，附录D.1](https://aclanthology.org/2025.findings-acl.589.pdf)。本轮运行前协议：research/2026-10-12_fairsteer_ambiguous_protocol.md。原始预测、冻结规则、复用检查、数据和源码快照保存在本目录。"]
    if (out/"independent_audit.json").exists():
        audit=read(out/"independent_audit.json")
        lines += ["",f"独立审计：{audit['status']}；复核{audit['prediction_files_checked']}份预测文件，指标最大误差{audit['max_metric_error']:.2g}。"]
    (out/"revision_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    records=[]
    for scope,items in (("paired_corrected",paired),("confirmation",fresh)):
        for r in items:
            v=r["metrics"]
            records.append(dict(scope=scope,id=r["spec"]["id"],source=r.get("prediction_name","current"),stereotype=v["macro_stereotype"],counterstereotype=v["macro_counterstereotype"],ambiguous_accuracy=v["macro_ambiguous_accuracy"],disambiguated_accuracy=v["disambiguated_accuracy"],overabstention=v["disambiguated_overabstention"],mmlu_accuracy=r["mmlu_accuracy"],feasible=r["feasible"],violations=";".join(r["violations"])))
    with (out/"revision_comparison.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    plot(out,policy,old,paired,fresh,b,f)


def row(label,m,ok):
    return f"|{label}|{pct(m['macro_stereotype'])}|{pct(m['macro_counterstereotype'])}|{pct(m['macro_ambiguous_accuracy'])}|{pct(m['disambiguated_accuracy'])}|{pct(m['disambiguated_overabstention'])}|{'—' if ok is None else '通过' if ok else '未通过'}|"


def plot(out,policy,old,paired,fresh,b,f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    specs=policy["previous_specs"]
    groups=[]
    groups.append(([next(r for r in old if r["spec"]["id"]==s["id"]) for s in specs],[next(r for r in paired if r["spec"]["id"]==s["id"]) for s in specs],b,"Opened-family paired diagnostic"))
    groups.append(([next(r for r in fresh if r["spec"]["id"]==s["id"] and "historical" in r["prediction_name"]) for s in specs],[next(r for r in fresh if r["spec"]["id"]==s["id"] and "corrected" in r["prediction_name"]) for s in specs],f,"Unseen-category confirmation"))
    x=np.arange(len(specs))
    for col,(a,c,base,title) in enumerate(groups):
        for y,key,label in ((0,"macro_stereotype","Stereotype rate (%)"),(1,"disambiguated_accuracy","Disambiguated accuracy (%)")):
            axis=axes[y,col]
            axis.bar(x-.18,[100*r["metrics"][key] for r in a],.36,label="mixed evidence DSV")
            axis.bar(x+.18,[100*r["metrics"][key] for r in c],.36,label="ambiguous-only DSV")
            axis.axhline(100*base[key],ls="--",color="black",label="unchanged model")
            axis.set(xticks=x,xticklabels=[f"Layer {s['layer']}, alpha {s['alpha']:g}" for s in specs],ylabel=label,title=title if y==0 else "Fixed layer, gate and alpha",ylim=(0,32) if y==0 else (0,75))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.05),ncol=3,frameon=False)
    fig.savefig(out/"revision_results.png",dpi=180,bbox_inches="tight");plt.close(fig)
