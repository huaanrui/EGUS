"""Read-only reporting, separate from frozen inference code."""
from .fairsteer_author import ROOT, CONFIG, read, tomllib


def report():
    c=tomllib.loads(CONFIG.read_text(encoding='utf-8')); out=ROOT/c['output_dir']
    state=read(out/'manifest.json'); data=read(ROOT/c['data_dir']/'manifest.json')
    if 'test' in state['stages']:
        from scripts.audit_fairsteer_author import audit
        audit()
    lines=['# FairSteer 作者协议迁移：Qwen3-1.7B','',
           '作者代码：https://github.com/LiYichen99/FairSteer ，固定提交 '+c['author_commit']+'。','',
           '完成阶段：'+', '.join(state['stages'])+'。','',
           '本实验对齐作者的 BBQ 流程；不包含论文的其他基准或完整通用能力评价。', '',
           '## 预先声明的差异与解释边界','']
    lines += ['- '+x for x in data['limitations']]
    lines += ['', 'sklearn 使用 L2 逻辑回归：C=1，拟合截距（不惩罚截距），LBFGS，max_iter=10000，tol=1e-4，无标准化或类别重加权。目标为总 logistic loss + ||w||²/(2C)；求解器、精度及收敛准则与 cuML 不保证一致。',
              '', '保留作者 hidden_states[1:] 的最后层归一化后特征，同时在原始 block 输出干预；这一原始代码的不对称在 smoke.json 中记录。',
              '', '## 数据与设置','',
              '完整 BBQ 58,492 题，zero/few-shot 各一次，加作者 MMLU 10,266 条用于检测器；作者公开的 110 条 BBQA 用于方向。检测器样本按原代码标签规则过滤后随机 4:1 划分。',
              '', '重建 2,200 条选层样本，每类 200 条；扫描全部 28 层，α=1，按整体准确率选择，平局取先出现的层。最终评价全量 BBQ，保留原始选项顺序。',
              '', '方向/检测器/正式评价用 A.，选层用 (A)。单题 FP16 前向、原始文本提示，无聊天模板、无 CoT；取最后一个 token，方向不归一化，仅检测为偏见时相加。',
              '', '## 结果','']
    if (out/'results.json').exists():
        result=read(out/'results.json')
        lines += ['| 条件 | 方法 | 整体准确率 | 歧义准确率 | 消歧准确率 | BS(a) | BS(d) |', '|---|---|---:|---:|---:|---:|---:|']
        fmt=lambda x:'NA' if x is None else f'{100*x:.3f}'
        for shot,values in result.items():
            for method in ('baseline','fair_steer'):
                m=values[method]
                lines.append('| '+' | '.join([shot,method,fmt(m['accuracy']),fmt(m['ambig']['accuracy']),fmt(m['disambig']['accuracy']),fmt(m['ambig']['bias_score']),fmt(m['disambig']['bias_score'])])+' |')
        lines += ['', '准确率为百分比，BS 为分数乘 100；有符号 BS 应结合距零的大小解读，不能将变负一概称为改善。']
    else: lines += ['尚未完成正式评价，当前不能判断是否复现效果。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out/'report.md')
