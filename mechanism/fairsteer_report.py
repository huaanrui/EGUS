"""Render the frozen reference pilot without changing experiment decisions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def pct(value):
    return "—" if value is None else f"{100*value:.2f}%"


def render(out: Path, data: Path):
    manifest=read(data,"manifest.json")
    direction_conditions={}
    for line in (data/"direction_fit.jsonl").read_text(encoding="utf-8").splitlines():
        condition=json.loads(line)["context_condition"]
        direction_conditions[condition]=direction_conditions.get(condition,0)+1
    plan=read(out,"test_plan_frozen.json")
    base=read(out,"predictions/test_baseline.json")
    base_mmlu=read(out,"predictions/test_mmlu_baseline.json")["accuracy"]
    results=read(out,"test_results.json")
    if (out/"matched_control_complete.json").exists():
        results += read(out,"matched_control_results.json")
    selection=read(out,"selection_results.json")
    screen=read(out,"screen_results.json")
    probe=read(out,"probe_diagnostics.json")
    directions=read(out,"direction_audit.json")
    checks=read(out,"hook_checks.json")
    stability=read(out,"stability_audit.json")
    calibration=read(out,"calibration_test.json")
    primary=next(r for r in results if r["spec"]["id"]==plan["primary"]["id"])
    pm=primary["metrics"]
    gain=100*(base["metrics"]["macro_stereotype"]-pm["macro_stereotype"])
    accepted=[r for r in selection if r["accepted"]]
    rows=[]
    for r in results:
        m=r["metrics"]
        ci=r["confidence"]["macro_stereotype"]
        rows.append(dict(id=r["spec"]["id"],variant=r["spec"]["variant"],layer=r["spec"]["layer"],alpha=r["spec"]["alpha"],macro_stereotype=m["macro_stereotype"],stereotype_change_pp=100*(m["macro_stereotype"]-base["metrics"]["macro_stereotype"]),ci_low_pp=100*ci["low"],ci_high_pp=100*ci["high"],macro_counterstereotype=m["macro_counterstereotype"],macro_ambiguous_accuracy=m["macro_ambiguous_accuracy"],disambiguated_accuracy=m["disambiguated_accuracy"],mmlu_accuracy=r["mmlu_accuracy"],native_accuracy=r["native_metrics"]["accuracy"],trigger_fraction=r["perturbation"]["trigger_fraction"],relative_l2=r["perturbation"]["relative_l2"],feasible=r["feasible"],violations=";".join(r["violations"])))
    with (out/"test_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/"selection_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["id","native_accuracy","six_accuracy","macro_stereotype","macro_counterstereotype","macro_ambiguous_accuracy","disambiguated_accuracy","mmlu_accuracy","feasible","accepted","gain_pp"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in selection:
            writer.writerow(dict(id=r["spec"]["id"],native_accuracy=r["native_metrics"]["accuracy"],six_accuracy=r["metrics"]["accuracy"],macro_stereotype=r["metrics"]["macro_stereotype"],macro_counterstereotype=r["metrics"]["macro_counterstereotype"],macro_ambiguous_accuracy=r["metrics"]["macro_ambiguous_accuracy"],disambiguated_accuracy=r["metrics"]["disambiguated_accuracy"],mmlu_accuracy=r["mmlu_accuracy"],feasible=r["feasible"],accepted=r["accepted"],gain_pp=100*r["macro_stereotype_reduction"]))
    with (out/"screen_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["id","variant","layer","native_accuracy","six_accuracy","macro_stereotype","macro_counterstereotype","macro_ambiguous_accuracy","disambiguated_accuracy","trigger_fraction","relative_l2","direction_norm"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in screen:
            layer=r["spec"]["layer"]
            mode="balanced" if r["spec"]["variant"]=="balanced_gated" else "native"
            writer.writerow(dict(id=r["spec"]["id"],variant=r["spec"]["variant"],layer=layer,native_accuracy=r["native_metrics"]["accuracy"],six_accuracy=r["metrics"]["accuracy"],macro_stereotype=r["metrics"]["macro_stereotype"],macro_counterstereotype=r["metrics"]["macro_counterstereotype"],macro_ambiguous_accuracy=r["metrics"]["macro_ambiguous_accuracy"],disambiguated_accuracy=r["metrics"]["disambiguated_accuracy"],trigger_fraction=r["perturbation"]["trigger_fraction"],relative_l2=r["perturbation"]["relative_l2"],direction_norm=directions[mode+"_norms"][layer]))
    lines=[
        "**FairSteer 风格参考基线：实际结果与论文设置对齐记录**", "",
        f"模型：Qwen3-1.7B / BF16。主要参考配置：`{plan['primary']['id']}`，按原始答案顺序准确率选层，α=1；所有评价同时保存六种答案排列。新测试宏平均刻板率由 {pct(base['metrics']['macro_stereotype'])} 变为 {pct(pm['macro_stereotype'])}，下降 {gain:.2f} 个百分点。该配置测试能力约束{'通过' if primary['feasible'] else '未通过'}。", "",
        f"selection 满足预设至少 1pp 刻板率改善及能力/反刻板错误限制的候选：{len(accepted)} 个。参考协议最优与严格标准最优分别冻结，不能把主要参考配置自动称为可接受去偏方案。", "",
        "以下置信区间按类别内模板家族配对 bootstrap 2000 次；全部六种排列和同一题绑定。不在测试上选赢家，未对探索性多重比较作显著性修正。", "",
        "|新测试配置|宏平均刻板率|变化/pp（95%区间）|反刻板率|模糊题正确率|明确题正确率|MMLU|约束|",
        "|---|---:|---|---:|---:|---:|---:|---|",
        f"|基线|{pct(base['metrics']['macro_stereotype'])}|—|{pct(base['metrics']['macro_counterstereotype'])}|{pct(base['metrics']['macro_ambiguous_accuracy'])}|{pct(base['metrics']['disambiguated_accuracy'])}|{pct(base_mmlu)}|—|",
    ]
    for r in rows:
        lines.append(f"|{r['id']}|{pct(r['macro_stereotype'])}|{r['stereotype_change_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]|{pct(r['macro_counterstereotype'])}|{pct(r['macro_ambiguous_accuracy'])}|{pct(r['disambiguated_accuracy'])}|{pct(r['mmlu_accuracy'])}|{'通过' if r['feasible'] else r['violations']}|")
    lines += ["",f"unknown 输出校准对照的开发集冻结偏置为 {calibration['offset']:g}，测试宏平均刻板率 {pct(calibration['metrics']['macro_stereotype'])}，测试约束{'通过' if calibration['feasible'] else '未通过'}。随机方向与主参考方向等长度、同层、同门控，但等向量长度不保证相同的实际残差变化量。", "",
              f"主参考的宏平均反刻板错误率同时从 {pct(base['metrics']['macro_counterstereotype'])} 降至 {pct(pm['macro_counterstereotype'])}；明确题弃答率从 {pct(base['metrics']['disambiguated_overabstention'])} 升至 {pct(pm['disambiguated_overabstention'])}。宏平均 S−C 差值从 {100*(base['metrics']['macro_stereotype']-base['metrics']['macro_counterstereotype']):.2f}pp 变为 {100*(pm['macro_stereotype']-pm['macro_counterstereotype']):.2f}pp（此项不是直接套用的官方整体 BBQ score）。这些观察支持一般弃答/不确定性偏移这一候选解释，不能仅凭刻板率下降就把方向解释成选择性的社会偏见机制。", "",
              "**本轮修正的实验问题**", "",
              "1. 方向使用同一道问题的正确/刻板答案 teacher forcing，控制上下文；保留原始均值差长度，采用正向加法。身份-order 对照和背景中心投影不再充当该方法的替代。",
              "2. 检测器从未回答的 prompt 末尾取状态，正确回答标 1、刻板错误标 0，其他错误排除；MMLU 依作者设为无偏控制。方向包含原代码的零样本/少样本双回答。",
              "3. 三种变体各自在全部 28 个原始 block 输出处筛选。每种干预按自己的效果定位，没有使用整残差 donor 恢复或 block 更新量结果预选三层。",
              "4. 全部六种答案排列在同一固定 batch 中计算；方向额外比较六排列平均版本，门控逐个 batch 元素计算。原始排列结果单独保存。",
              "5. 新下载的三类别有各自 fit/selection/test 家族；确认测试来自未实验过的家族，且测试类别在本轮 fit 已出现。共享 few-shot 示例家族从评价中排除。",
              "6. 明确区分 S、C、U、signed bias、连续 margin、明确题能力、弃答与实际干预量。筛选门槛与原始效果分别报告。", "",
              "**数据与可比性**", "", "|用途|BBQ 题数|家族数|", "|---|---:|---:|"]
    for name in ("fit","probe_validation","direction_fit","selection","test"):
        value=manifest["summary"][name];lines.append(f"|{name}|{value['rows']}|{value['families']}|")
    lines += ["", "fit/selection 涵盖 Age、Gender_identity、SES、Nationality、Race_ethnicity、Religion。封存测试仅包含最后三个新增类别，测试平均按这三个类别计算。因此与前两轮不同测试集合的数值不能直接当作方法提升。", "",
              "本地适配保留了 FairSteer 的双回答 DiM、原尺度、线性检测器、条件加法及准确率选层。以下与完整论文不同，不能称为原配置精确复现：", "",
              "|项目|论文/作者代码|本轮|", "|---|---|---|",
              "|模型|Llama/Vicuna/Mistral，7–13B|Qwen3-1.7B|",
              f"|方向样本|附录 D.1：仅模糊题，11 类各 10 条，共 110 条|6 类合计 110 条，限定在 fit 内；实际为 {direction_conditions.get('ambig',0)} 条模糊题、{direction_conditions.get('disambig',0)} 条明确题|",
              "|检测器语料|完整 BBQ + 10,266 MMLU，零/少样本扩增|1536 条 BBQ 的 fit/内部留出合计；MMLU 512 训练、128 内部留出|",
              "|检测器划分|随机 4:1|fit 内按模板家族约 4:1，MMLU 按去重问题隔离|",
              "|检测器实现|cuML 默认 L2|sklearn lbfgs C=1，未标准化、未类别加权|",
              "|隐藏状态|作者 hidden_states[1:]，最终状态经过 final norm；注入 raw block|提取与注入均对齐 raw block，最后一层也如此|",
              "|选层规模|2200 条、每层准确率|共同 screen panel 扫所有层，每变体 top-2 在 336 条完整 selection 复核|",
              "|选项格式|作者 probe/direction 使用 A.，选层/评价使用 (A)|保留此差异，明确记录各阶段|",
              "|主要位置|方向答案末 token，检测/注入 prompt 末 token|对齐；没有混用 last-k|",
              "|排列|原始顺序|保留原始顺序得分，另外计算全部六排列|", "",
              "本轮存在不同阶段格式和 teacher-forcing 状态迁移，源于作者实现；不能假定检测准确率直接证明方向具有选择性因果作用。balanced 方向是预登记诊断；不能用其测试效果回选主方案。MMLU 只覆盖去掉 D 后的作者子集和本地小样本，不是完整通用能力评测。", "",
              f"评价通过强制比较空格加 A/B/C 的单 token 分数完成。新测试基线三个候选在完整词表中的平均概率质量仅为 {pct(base['metrics']['candidate_mass'])}，主参考干预为 {pct(pm['candidate_mass'])}；其余概率未参与选项预测。这表明本轮裸文本提示下的强制选项结论不能直接视为自由生成行为改善，也应优先核查 Qwen 聊天模板下的结果是否稳定。", "",
              "**检测与实现检查**", "",
              f"检测器训练标签计数：{probe['train_labels']}；内部验证标签计数：{probe['validation_labels']}。该验证按家族隔离，报告偏见召回和 balanced accuracy，避免多数类准确率掩盖失效门控。", "",
              "|层|BBQ 内部验证 balanced accuracy|偏见召回|无偏假阳性|native DiM 长度|平衡 DiM 长度|", "|---|---:|---:|---:|---:|---:|"]
    selected_layers=sorted({r["spec"]["layer"] for r in results})
    for layer in selected_layers:
        d=probe["layers"][layer]["bbq"]
        lines.append(f"|{layer}|{pct(d['balanced_accuracy'])}|{pct(d['biased_recall'])}|{pct(d['false_positive_rate'])}|{directions['native_norms'][layer]:.4f}|{directions['balanced_norms'][layer]:.4f}|")
    if (out/"probe_stratified_audit.json").exists():
        detail=read(out,"probe_stratified_audit.json")["layers"][plan["primary"]["layer"]]["strata"]
        lines += ["", "主参考层的检测器分层诊断如下，仅含内部验证的原始答案顺序、作者规则保留的正确/刻板错误样本，其他错误已排除。它不等于全部评价输入的检测表现，也不能证明其识别的是纯粹偏见机制。", "", "|内部验证子集|保留样本数|刻板错误数|刻板错误召回|无偏误触发|", "|---|---:|---:|---:|---:|"]
        for name in ("zeroshot_ambig","zeroshot_disambig","fewshot_ambig","fewshot_disambig","mmlu"):
            d=detail[name]
            lines.append(f"|{name}|{d['n']}|{d['biased']}|{pct(d['biased_recall'])}|{pct(d['false_positive_rate'])}|")
    lines += ["", f"零强度/移除干预后 logits 逐位一致；非零原始方向产生实测变化。实际核对 hidden_states 与 raw hook 的差异仅在最后一层：{checks['hidden_states_vs_raw_mismatches']}。最长已核验输入 {checks['max_tokens']} token，未截断。", "",
              f"固定 12 个开发问题的 BF16 batch6/batch1 检查：{stability['comparisons']} 个预测比较中 {stability['prediction_flips']} 个不同。该检查不能证明完整测试在 FP32 下不变；若效应接近此量级，结论应保持探索性。", "",
              "另外在 fit 阶段、selection/test 之前预登记了同层诊断规则：固定主参考层，改变门控、方向提取、平衡方向长度和正负号。它们用于控制变量，不能替换已经冻结的主要方案。若附加诊断尚未执行完，本报告不会列出其结果。", "",
              "本轮只建立方向干预的参考基线，旧的安全神经元、层更新量、RelP 节点结果仍为历史适配实验。未据此声称其他论文已经精确复现，也未发现完整边电路。", "",
              "来源：[FairSteer 论文](https://aclanthology.org/2025.findings-acl.589.pdf)、[固定版本作者代码](https://github.com/LiYichen99/FairSteer/tree/0545cdbaa71b3301030accf1b98130820e08026f)。逐题预测、选择计划、源码快照、模型/数据哈希和统计表保存在本报告同目录。"]
    document="\n".join(lines)+"\n"
    if manifest.get("test_role")=="opened_family_paired_diagnostic":
        document="**注意：本文件中的320题评价是上一轮已开封家族的配对诊断，不能作为新确认测试。独立新增类别确认结果见 revision_report.md。**\n\n"+document
        document=document.replace("新测试","配对重测").replace("新封存测试","已开封重测").replace("新家族测试","已开封家族配对重测")
        document=document.replace("确认测试来自未实验过的家族，且测试类别在本轮 fit 已出现", "320题沿用上一轮已开封家族作配对诊断，且这些类别在 fit 已出现")
    (out/"report.md").write_text(document,encoding="utf-8")
    plot(out,screen,base,results,directions)


def plot(out,screen,base,results,directions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    for variant in ("native_gated","native_always","balanced_gated"):
        values=sorted([r for r in screen if r["spec"]["variant"]==variant],key=lambda r:r["spec"]["layer"])
        axes[0].plot([r["spec"]["layer"] for r in values],[100*r["native_metrics"]["accuracy"] for r in values],label=variant)
    sb=read(out,"predictions/screen_baseline.json")
    axes[0].axhline(100*sb["native_metrics"]["accuracy"],color="black",ls="--",label="unchanged model")
    axes[0].set(xlabel="Block index (zero based)",ylabel="Native answer accuracy (%)",title="Development layer scan")
    axes[0].legend(fontsize=7)
    axes[1].plot(directions["native_norms"],label="native answer order")
    axes[1].plot(directions["balanced_norms"],label="six-permutation mean")
    axes[1].set(xlabel="Block index",ylabel="Raw vector L2",title="Teacher-forced direction scale")
    axes[1].legend(fontsize=8)
    for i,r in enumerate(results):
        ci=r["confidence"]["macro_stereotype"]
        point=100*ci["delta"]
        axes[2].errorbar(point,i,xerr=[[point-100*ci["low"]],[100*ci["high"]-point]],fmt="o",color="grey" if r["spec"]["variant"]=="random_gated" else "tab:blue")
    axes[2].set_yticks(range(len(results)),[r["spec"]["id"] for r in results],fontsize=7)
    axes[2].axvline(0,color="black",ls="--")
    title="Opened-family paired diagnostic" if (out/"revision_policy_frozen.json").exists() else "Frozen new-family confirmation"
    axes[2].set(xlabel="Stereotype-rate change (pp), 95% CI",title=title)
    fig.savefig(out/"reference_results.png",dpi=180)
    plt.close(fig)
