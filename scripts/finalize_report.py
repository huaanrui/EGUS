"""Merge completed diagnostics into a readable report; never select configurations."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechanism.data import read_rows, sha256, write_json
from mechanism.evaluate import metrics


def main():
    out = ROOT / "results/pilot_20261011_v4"
    read = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
    if read("run_manifest.json")["status"] != "complete":
        raise ValueError("Primary experiment is incomplete")
    base = read("predictions/validation_baseline.json")
    validation = read("validation_results.json")
    global_nodes = read("global_nodes_results.json")
    winners = read("validation_plan_frozen.json")["winners"]
    accepted = [m for m, s in winners.items() if s["status"] == "accepted"]
    screen = read("screen_results.json")
    screen_base = read("predictions/screen_baseline.json")["metrics"]
    continuous = []
    for r in screen:
        b = r["bbq"]
        continuous.append(dict(id=r["spec"]["id"], method=r["spec"]["method"], axis=r["spec"]["axis"], pool=r["spec"]["pool"], scope=r["spec"]["scope"], ambiguous_stereotype_minus_unknown_logit_margin=b["ambiguous_margin"], margin_delta_vs_baseline=b["ambiguous_margin"]-screen_base["ambiguous_margin"], ambiguous_accuracy=b["ambiguous_accuracy"], permutation_consistency=b["permutation_consistency"], feasible=r["feasible"]))
    with (out / "screen_continuous_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(continuous[0]))
        writer.writeheader()
        writer.writerows(continuous)
    paths = {"baseline": "validation_baseline"}
    paths.update({r["method"]: f'validation_{r["method"]}' for r in validation})
    paths.update({r["spec"]["id"]: f'validation_{r["spec"]["id"]}' for r in global_nodes["conditions"]})
    rows = read_rows(ROOT / "data/prepared/bbq_pilot_v4/validation.jsonl")
    subgroups = []
    for method, name in paths.items():
        predictions = read(f"predictions/{name}.json")["predictions"]
        by_uid = {p["uid"]: p for p in predictions}
        for field in ("question_polarity", "label_type", "correct_is_stereotype"):
            for value in sorted({str(r[field]) for r in rows}):
                subset = [r for r in rows if str(r[field]) == value]
                if not subset:
                    continue
                m = metrics(subset, [by_uid[r["uid"]] for r in subset])
                subgroups.append(dict(method=method, dimension=field, value=value, rows=len(subset), families=m["families"], ambiguous_accuracy=m["ambiguous_accuracy"], stereotype_rate=m.get("stereotype_rate"), counterstereotype_rate=m.get("counterstereotype_rate"), disambiguated_accuracy=m["disambiguated_accuracy"], disambiguated_overabstention=m["disambiguated_overabstention"]))
    write_json(out / "validation_subgroups.json", subgroups)
    with (out / "validation_subgroups.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroups[0]))
        writer.writeheader()
        writer.writerows(subgroups)
    report = (out / "report.md").read_text(encoding="utf-8").replace("日期：2026-10-11。", "实验启动：2026-10-11；完成：2026-10-12。")
    marker = "\n## 扩展诊断与汇总\n"
    report = report.split(marker)[0]
    outcome = ("完整 selection 确认通过的改善候选：" + ", ".join(accepted) + "。" if accepted else "按预设规则，本轮没有确认同时满足能力约束并提高宏平均模糊题准确率的候选。验证表中的方法均为冻结诊断，不能作为已证明有效的去偏方案。")
    for r in validation:
        if r["method"] in accepted:
            da = 100*(r["bbq"]["ambiguous_accuracy"]-base["metrics"]["ambiguous_accuracy"])
            ds = 100*(r["bbq"]["stereotype_rate"]-base["metrics"]["stereotype_rate"])
            outcome += f" {r['method']} 的验证模糊题准确率变化为 {da:+.2f} 个百分点，刻板选择率变化为 {ds:+.2f} 个百分点。"
            if ds > 0:
                outcome += "刻板选择并未下降，因此不能将该结果解释为已经成功去偏。"
    absent = [m for m, s in winners.items() if s["spec"] is None]
    extra = [marker, "", outcome, "", "没有可行初筛候选、因而未进入验证的方法：" + ", ".join(absent) + "。", "", "局部神经元消融固定在第15层；层搜索只在方向/层方法中展开，另有7层联合节点诊断。本轮不能据此排除其他层上的神经元策略。", "", "### 数值与批次敏感性", "", "扩大核对至相同初筛基线后，batch=1 与 batch=3 在 BBQ 的96次预测中有2次不同，在 neutral 的72次预测中有3次不同，共5/168（2.98%）；候选 log probability 最大差约3.99。该差异来自 BF16 执行形状敏感性，不应将小幅变化解读为精度无关的机制结论。主实验所有基线与干预均统一 batch=3，没有混用旧初筛结果。原始核对见 `batch_sensitivity_audit.json`。", "", "### 跨层 RelP 节点子网络", "", "此固定诊断在主实验验证前登记：在已提取的7层、每层6144个 MLP 坐标、last prompt 位置的显式集合中，按 RelP 归因跨层选总计256个节点。删除仅替换这些节点；保留则替换该显式集合中的其余节点，集合之外的模型计算继续运行。未追踪边，不能称为完整电路；此处的保留结果仅表示条件保留。", "", "|固定条件|模糊题准确率|刻板率|反刻板率|充分证据准确率|neutral准确率|模糊准确率变化/95%区间（百分点）|", "|---|---:|---:|---:|---:|---:|---|"]
    for r in global_nodes["conditions"]:
        b, n = r["bbq"], r["neutral"]
        ci = r["paired_cluster_ci"]["ambiguous_accuracy"]
        extra.append(f"|{r['spec']['operation']} 256|{100*b['ambiguous_accuracy']:.2f}%|{100*b['stereotype_rate']:.2f}%|{100*b['counterstereotype_rate']:.2f}%|{100*b['disambiguated_accuracy']:.2f}%|{100*n['accuracy']:.2f}%|{100*ci['delta']:+.2f} [{100*ci['low']:+.2f}, {100*ci['high']:+.2f}]|")
    extra += ["", "具体跨层节点列表见 `global_nodes_selected.json`；冻结计划和归因文件哈希见 `global_nodes_plan_frozen.json`。", "", "### 可查看的消融与子群结果", "", "- `ablation_factor_deltas.csv`：每个条件相对同方法固定基准的单变量差值。加法与投影是操作族比较，不能只按强度数值解释。", "- `ablation_factors.png`：提取 last-k、干预范围、组件数和强度的初筛曲线。", "- `screen_continuous_metrics.csv`：刻板答案与 unknown 的平均 logit 差及其变化。分数改变而选项不变时，不能称为已经实现行为去偏。", "- `validation_subgroups.csv`：问句极性、label/name、正确答案是否与刻板印象一致的验证指标；这些分析不参与重新选参。", "", "![Selection ablation curves](ablation_factors.png)", "", "### 后续实验应优先控制的因素", "", "1. 用更多 selection 家族样本复核局部效应，并在单样本执行下复核冻结候选的数值稳健性；不根据本轮验证重新选策略。", "2. 增加与社会群体无关、但同样要求证据不足时选 unknown 的困难控制任务，区分一般弃答/证据读取与偏见特异机制。", "3. 对节点子网络建立身份反事实配对，再分析跨层边、必要性和条件充分性；目前节点归因不能支持完整电路结论。", "4. OFAT 之后再单列提取范围×干预范围的交互实验，给每个新实验保留独立留出集。", ""]
    text = report + "\n".join(extra)
    (out / "report.md").write_text(text, encoding="utf-8")
    # A tracked research summary points to the actual local artifacts.
    summary = "# 已完成的 BBQ 机制预实验\n\n实验启动：2026-10-11；完成：2026-10-12。\n\n" + outcome + "\n\n" + f"完整报告：[report.md]({(out/'report.md').as_posix()})。\n\n" + "|条件|selection状态|模糊题准确率|刻板率|充分证据准确率|\n|---|---|---:|---:|---:|\n"
    bm = base["metrics"]
    summary += f"|baseline|基线|{100*bm['ambiguous_accuracy']:.2f}%|{100*bm['stereotype_rate']:.2f}%|{100*bm['disambiguated_accuracy']:.2f}%|\n"
    for r in validation:
        b = r["bbq"]
        summary += f"|{r['method']}|{r['selection_status']}|{100*b['ambiguous_accuracy']:.2f}%|{100*b['stereotype_rate']:.2f}%|{100*b['disambiguated_accuracy']:.2f}%|\n"
    for method in absent:
        summary += f"|{method}|no_feasible_screen_candidate|—|—|—|\n"
    summary += "\n单模型、15个验证家族，置信区间较宽；单样本与三轮批次初筛基线有5/168预测差异。主比较统一 batch=3；本轮不支持全局最优、一般能力保持或完整电路结论。具体局部消融、跨层节点诊断及置信区间均保存在完整报告和 JSON/CSV 产物中。\n"
    (ROOT / "research/2026-10-11_pilot_results.md").write_text(summary, encoding="utf-8")
    write_json(out / "finalization_manifest.json", dict(finalizer_sha256=sha256(Path(__file__)), primary_run_manifest_sha256=sha256(out/"run_manifest.json"), report_sha256=sha256(out/"report.md"), global_results_sha256=sha256(out/"global_nodes_results.json"), matched_ablation_sha256=sha256(out/"ablation_factor_deltas.csv"), validation_plan_sha256=sha256(out/"validation_plan_frozen.json"), subgroup_analysis="descriptive only, no selection"))
    print(f"Final report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
