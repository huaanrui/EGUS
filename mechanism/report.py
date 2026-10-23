from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_report(out: Path):
    def read(name):
        return json.loads((out / name).read_text(encoding="utf-8"))
    manifest = read("run_manifest.json")
    table = read("screen_results.json")
    base = read("predictions/validation_baseline.json")["metrics"] if (out / "predictions/validation_baseline.json").exists() else None
    validation = read("validation_results.json") if (out / "validation_results.json").exists() else []
    fields = ["id", "method", "axis", "layers", "pool", "scope", "operation", "strength", "k", "ambiguous_accuracy", "stereotype_rate", "counterstereotype_rate", "disambiguated_accuracy", "neutral_accuracy", "feasible", "relative_l2", "seconds"]
    with (out / "screen_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table:
            values = {key: row["spec"].get(key, "") for key in fields}
            values.update({key: row["bbq"][key] for key in ("ambiguous_accuracy", "stereotype_rate", "counterstereotype_rate", "disambiguated_accuracy")})
            values.update(neutral_accuracy=row["neutral"]["accuracy"], feasible=row["feasible"], relative_l2=row["perturbation"]["relative_l2"], seconds=row["seconds"])
            writer.writerow(values)
    pct = lambda x: f"{100*x:.2f}%"
    lines = ["# BBQ 偏见机理预实验：实际运行报告", "", "日期：2026-10-11。以下来自本地真实推理；所有搜索使用训练内部 selection，验证前冻结配置。", "", "## 范围与实现", "", f"- 模型：`{manifest['config']['model_path']}`；BF16，冻结参数，原生非思考模板；fit/归因 batch=1，评价将同题三种等长轮换合成 batch=3；无 padding、无 KV cache 的首答案评分。", "- 每题执行 A/B/C 三种轮换，按原始答案语义计算指标。检查候选拼接分词；当前实现仅支持已核验的单 token 标签。", "- fit 提取 last-1/2/4/8 及 user 全文均值；干预范围独立配置。均值由 fit 冻结，并按 user、前缀和逐后缀位置匹配。", "- DiM/CAV 使用残差方向；SN 为 MLP 贡献一致性适配；神经元对照使用均值差、激活幅值及随机排名；层方法替换 block update，保留残差旁路。", "- RelP 使用冻结 RMSNorm 分母、SiLU sigmoid、attention Q/K 路径及 gated-product half rule；仅发现 MLP 节点，未发现节点间边。采用带 fit 均值参考的归因适配，原论文评估结论不能直接迁移。", "- `retain` 只保留选中层内的 k 个 MLP 坐标，其他坐标替换为均值；这是条件保留诊断，不能称为保留整个模型电路。", "", "## 数据和选参", "", "- BBQ：Age、Gender_identity、SES；fit 1664 / selection 312 / validation 264；模板家族 47 / 16 / 15。fit 在初始880题稳定刻板选择只有27题后，按预定规则在同一批 fit 家族内扩充；selection/validation 未增加或移动题目。", "- 发现类：三次轮换均选择刻板答案的 ambiguous 题，对比三次均选择 unknown 的题。每类至少40题、8家族；最多取64题，按家族轮转抽样。所有监督排名使用相同发现题。", "- 参考均值来自全部 fit，不使用验证集标签；合成 neutral 控制为120个根场景，每个含证据充分/不足两题，selection/validation 各60根、120题。", f"- 初筛：每个 selection 家族/条件取{manifest['config']['screen_rows_per_family_condition']}题；每个配置还执行 {manifest['config']['neutral_screen_roots']} 个 neutral 根场景。", "- 对各方法先按预先声明的约束选择初筛候选，再在完整 selection 确认。没有确认的改善时报告无可行结果；最终验证仍展示已冻结候选，明确标注其状态。", "- 严格能力约束：充分证据准确率下降≤2个百分点，单类别≤5个百分点；neutral 准确率下降和充分证据过度弃答增幅各≤2个百分点。", "- 层、token 提取、干预范围、组件数和强度分别改变。OFAT 只分析局部单变量效应，不能覆盖轴之间的交互。", ""]
    discovery = read("discovery_classes.json")
    lines += [f"实际发现类：{k}：可用 {v['available']}，使用 {v['used']}，覆盖 {v['families']} 个家族。" for k, v in discovery.items()]
    checks = read("hook_checks.json") if (out / "hook_checks.json").exists() else {}
    lines += ["", "## 实现检查", "", "```json", json.dumps(checks, ensure_ascii=False, indent=2), "```", "", "## Selection 消融", "", "完整逐配置结果见 `screen_comparison.csv` 和 `screen_results.json`；不可将初筛最佳值当作独立泛化表现。", "", "|方法|改变的轴|层|提取|干预|操作|k|模糊题准确率|刻板率|充分证据准确率|neutral准确率|约束通过|", "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in table:
        s, b = r["spec"], r["bbq"]
        lines.append(f"|{s['method']}|{s['axis']}|{','.join(map(str,s['layers']))}|{s['pool']}|{s['scope']}|{s['operation']} {s['strength']}|{s.get('k','')}|{pct(b['ambiguous_accuracy'])}|{pct(b['stereotype_rate'])}|{pct(b['disambiguated_accuracy'])}|{pct(r['neutral']['accuracy'])}|{'是' if r['feasible'] else '否'}|")
    if base:
        lines += ["", "## 冻结配置的验证结果", "", "下面的百分点差值使用按类别分层的模板家族配对 bootstrap（2000次）；未做多重比较修正。", "", "|方法|selection状态|模糊题准确率|刻板率|反刻板率|充分证据准确率|neutral准确率|模糊准确率变化/95%区间（百分点）|", "|---|---|---:|---:|---:|---:|---:|---|"]
        neutral_base = read("predictions/validation_neutral_baseline.json")["metrics"]
        lines.append(f"|baseline|基线|{pct(base['ambiguous_accuracy'])}|{pct(base['stereotype_rate'])}|{pct(base['counterstereotype_rate'])}|{pct(base['disambiguated_accuracy'])}|{pct(neutral_base['accuracy'])}|—|")
        for r in validation:
            b = r["bbq"]
            ci = r["paired_cluster_ci"]["ambiguous_accuracy"]
            lines.append(f"|{r['method']}|{r['selection_status']}|{pct(b['ambiguous_accuracy'])}|{pct(b['stereotype_rate'])}|{pct(b['counterstereotype_rate'])}|{pct(b['disambiguated_accuracy'])}|{pct(r['neutral']['accuracy'])}|{ci['delta']*100:+.2f} [{ci['low']*100:+.2f}, {ci['high']*100:+.2f}]|")
        lines += ["", "每个候选的冻结参数、逐类别指标和置信区间见 `validation_results.json`；逐题分数见 `predictions/`。", ""]
    lines += ["## 解释边界", "", "这是单模型、小样本的可行性实验。BF16 的单样本与三轮批次实测预测在48个提示的两种条件下相同，但候选 log probability 最大差约3.75；所有评价统一批次，不混用分数。15个验证家族带来较大的不确定性；同题的答案轮换不能计作三个独立样本。neutral 控制只检查简单信息读取和弃答，不能证明一般能力保持。降低刻板率若同时增加反刻板错误或充分证据弃答，不能算去偏成功。", "", "不同方法的组件数和激活空间不同，不能凭相同 k 或内部扰动范数判断公平。报告的相对扰动用于同一空间内诊断；跨方法需结合充分证据能力、neutral 表现和时间成本比较。", "", "可读出方向、干预有效和自然计算中的因果机制是不同证据。本轮不支持 CoT、RLHF 前后或完整电路的结论；这些需要新增模型/任务/边干预。", "", "方法来源：[Refusal 单方向](https://arxiv.org/abs/2406.11717)、[SCAV](https://arxiv.org/abs/2404.12038)、[Safety Neuron 作者代码](https://github.com/zhaoyiran924/Safety-Neuron)、[Safety Layers](https://arxiv.org/abs/2408.17003)、[Neuron basis / RelP](https://arxiv.org/abs/2601.22594)。实现均有适配，未宣称逐项复现原论文。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_screen = read("predictions/screen_baseline.json")["metrics"]
    for method in dict.fromkeys(r["spec"]["method"] for r in table):
        subset = [r for r in table if r["spec"]["method"] == method and r["spec"]["operation"] != "retain"]
        ax.scatter([100*(r["bbq"]["disambiguated_accuracy"]-baseline_screen["disambiguated_accuracy"]) for r in subset], [100*(r["bbq"]["ambiguous_accuracy"]-baseline_screen["ambiguous_accuracy"]) for r in subset], label=method, alpha=.7)
    ax.axvline(-2, color="grey", linestyle="--", linewidth=.8)
    ax.axhline(0, color="grey", linewidth=.8)
    ax.set(xlabel="Disambiguated accuracy change (percentage points)", ylabel="Ambiguous accuracy change (percentage points)", title="Selection screening: utility vs ambiguity accuracy")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "screen_tradeoff.png", dpi=160)
    plt.close(fig)
