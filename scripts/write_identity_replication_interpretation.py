"""Assemble final Chinese interpretation from audited, completed measurements."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mechanism.fairsteer_author import ROOT,read,write,sha


def interval(z):return f"{100*z['delta']:+.2f}pp，95%家族区间[{100*z['ci95'][0]:+.2f},{100*z['ci95'][1]:+.2f}]pp"


if __name__=='__main__':
    folder=ROOT/'results/identity_replication_20261015_v1';complete=read(folder/'complete.json')
    primary=read(folder/'planned_comparisons.json');norm=read(folder/'norm_comparisons.json')
    semantic=read(folder/'semantic_v2_analysis.json');gen=read(folder/'generation_comparisons.json')
    balance=read(folder/'grammar_balance/analysis.json');balancedone=read(folder/'grammar_balance/complete.json')
    assert balancedone['analysis_sha256']==sha(folder/'grammar_balance/analysis.json')
    from scripts.identity_replication_grammar_balance import check_declaration
    check_declaration()
    mention=read(folder/'mention_order_analysis.json');native=read(folder/'native_type2_mention_order_audit.json')
    margins=read(folder/'margin_decomposition.json')
    energy=read(folder/'grammar_balance/target_energy_audit.json')
    for p,h in {**native['input_sha256'],**margins['inputs_sha256']}.items():assert sha(p)==h
    dv=read(folder/'diagnostics_verification.json')
    for p,h in dv['inputs_sha256'].items():assert sha(p)==h
    verification=read(folder/'verification.json');verification.update(dv)
    verification['execution_version']=complete['execution_version'];verification['semantic_v2_review_reported_separately']=True
    verification['grammar_balance_exploratory_records_recomputed']=balance['records_recomputed']
    verification['grammar_balance_frozen_parameters_unchanged']=balance['frozen_parameters_and_inputs_unchanged']
    verification['endpoint_margin_decomposition_full_reconstruction_verified']=margins['all_full_predictions_exactly_reconstructed']
    write(folder/'verification.json',verification)
    for name in ('qwen3','minicpm5','qwen25'):
        v=read(folder/name/'verification.json');v['diagnostic_transmission_and_intervals_recomputed']=True;write(folder/name/'verification.json',v)
    rows=['# 三个本地模型的机制复验与强对照：结果解释','',
        f"完成时间：{complete['completed']}。模型为本地Qwen3-1.7B、MiniCPM5-2B、Qwen2.5-3B-Instruct，FP16/SDPA。层与头从0计数。",'',
        '## 首要结论与数据边界','',
        '三模型均未在96个留出家族上显示细粒度干预的稳定优势。旧Qwen3的历史收益被强整均值对照接近追平；相反贡献可复现，但尚不能归为偏见专属机制。',
        '后续答案角色审计发现当前训练/筛选/开发/历史Type2的金答案100%为第二个提及职业；原始Type2文件也高度不平衡，不能把它单独归因于我们的宾格代词筛选。这个设计无法排除增强答案角色倾向造成涨点。',
        '训练rank64、筛选screen32、开发dev49对共享且按原场景家族隔离。历史Type2 146对、Type1 160对及Winogender240对已在前轮观察，为探索性复验。',
        '确认集为96对/96个此前未进入准备数据的原生Type1家族。源数据审计复核16344条旧准备记录、1177个去职业/代词模板，无模板重叠，原句与标签一致。',
        'Type1标签依赖上下文或常识，不统一属于明确证据；文本复核由agent进行，没有独立人工标注。确认集仍来自同一基准，不称通用领域验证。',
        '三模型定位、方法和全部扰动范数对照均在确认集推理前冻结。统一MLP流程只有四个源深度候选及有限后续位置，并未严格重跑原先完整逐层/注意力路径搜索。旧Qwen3方案另作冻结锚点。','',
        '## 确认集的成对正确率','',
        'fine_combo为固定的细粒度诊断组合，和允许不干预的开发集部署赢家selected分开报告。整均值对照包含不同组件的四种整体增减符号，按开发集正确性及能力约束选定，强度按开发集实际残差扰动RMS校准。','',
        '|模型|原模型|开发集赢家|细粒度诊断组合|范数匹配整均值组合|范数匹配随机组合均值|',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('qwen3','minicpm5','qwen25'):
        m=primary['models'][name]['metrics']['confirmation'];n=norm['models'][name]['confirmation']['metrics']
        random=np.mean([n[f'fine_norm_random{s}']['both_correct'] for s in range(1701,1706)])
        rows.append(f"|{name}|{m['baseline']['both_correct']:.2%}|{m['selected']['both_correct']:.2%}|{m['fine_combo']['both_correct']:.2%}|{n['fine_norm_whole_combo']['both_correct']:.2%}|{random:.2%}|")
    p=primary['primary_confirmation_holm9'];s=norm['confirmation_holm8']
    rows += ['',f"预设主要9项比较：Holm校正后p<.05共{sum(z['p_holm']<.05 for z in p)}项，其中细粒度有利方向{sum(z['p_holm']<.05 and z['delta']>0 for z in p)}项。",
             f"范数匹配补充8项比较单独作Holm校正：p<.05共{sum(z['p_holm']<.05 for z in s)}项，其中细粒度有利方向{sum(z['p_holm']<.05 and z['delta']>0 for z in s)}项。",
             '两组检验族分开声明，不宣称17项联合校正。随机种子先在每个场景对上求平均，再按家族bootstrap；种子不是独立问题。','',
             '## 旧Qwen3方案：局部收益与强基线','']
    m=primary['models']['qwen3']['metrics']['historical'];n=norm['models']['qwen3']['historical']['metrics']
    rows += ['|历史Type2方法|准确率|成对均对|身份翻转|','|---|---:|---:|---:|']
    for label,z in [('原模型',m['baseline']),('旧L7减半',m['old_l7']),('旧细粒度组合',m['old_combo']),('范数匹配整组件均值组合',n['old_norm_whole_combo'])]:
        rows.append(f"|{label}|{z['accuracy']:.2%}|{z['both_correct']:.2%}|{z['flip']:.2%}|")
    comps=norm['models']['qwen3']['historical']['comparisons']['old']
    rows += ['', '旧细粒度组合−强整均值组合：'+interval(comps['vs_norm_whole'])+'。',
        '旧细粒度组合−五个强随机组合均值：'+interval(comps['vs_norm_random_mean'])+'。这些历史区间为探索性比较。',
        '均值保持公式为 u′=μ+(1−λ)(u−μ)。正λ缩小偏离均值，负λ增强；整体抑制与保留均值、分组件改变符号不是同一种操作。',
        '强整均值对照的实际编辑系数：'+str(read(folder/'qwen3/norm_controls_frozen.json')['methods']['old_norm_whole_combo']['edits'])+'。',
        '若两者区间包含0，就不能把收益全部归因于精确选择少数神经元。对比随机有效也不意味着选定子集不可替代。','',
        '实际扰动匹配在开发集完成，并非逐目标样本在线等范数。旧Qwen3历史Type2整均值组合的RMS/细粒度参考约0.989，随机1701约0.991，故该数据上的幅度也接近。','',
        '## 答案位置与语法关系平衡补测','',
        '主结果后进行探索性审计：当前rank64、screen32、dev49及历史Type2 146对金答案均为第二职业；始终选第二职业的公共启发式可达100%。这不依赖答案字母，AB/BA排列本身无法排除。',
        '原始文件唯一抽取两职业及金标的条目中，dev为790/790第二职业，test为788/792；这是标注位置分布，不能据此断言模型实际只用了启发式或数据集没有研究价值。',
        '因此选取64个既有历史家族，保留职业及顺序，把唯一and变为“, who”。关系从句主语转为第二职业，后续非反身宾格代词金答案切换到第一职业；原句/派生句、男女、AB/BA全部交叉。始终选第一/第二在平衡集都只有50%。',
        '只按结构和文本选择，干预全部冻结，未重新调参；原/派生句共享家族进行bootstrap。文本由agent审阅，无独立人工标注；句法难度发生变化，因此是协议鲁棒性诊断，不是新的原生确认集，也不声称证据语义完全不变。','',
        '|模型/方法|原句成对均对|关系从句成对均对|平衡成对均对|相对基线95%家族区间pp|','|---|---:|---:|---:|---|']
    for name,z in balance['models'].items():
        for method in ('baseline','fine_combo','norm_whole_combo')+(('old_combo','old_norm_whole_combo') if name=='qwen3' else ()):
            m=z['metrics'][method];ci=z['vs_baseline'][method]['balanced']['both_correct']['ci95']
            rows.append(f"|{name}/{method}|{m['native']['both_correct']:.2%}|{m['relative']['both_correct']:.2%}|{m['balanced']['both_correct']:.2%}|[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]|")
    b=balance['models']['qwen3'];bm=b['metrics'];bc=b['vs_baseline']['old_combo']['balanced']['both_correct']
    rows += ['',
        f"旧Qwen3细粒度组合在该子集原句由{bm['baseline']['native']['both_correct']:.2%}升至{bm['old_combo']['native']['both_correct']:.2%}，关系从句由{bm['baseline']['relative']['both_correct']:.2%}降至{bm['old_combo']['relative']['both_correct']:.2%}；平衡集均对仅{interval(bc)}。",
        f"平衡集选择第二实体比例从{bm['baseline']['balanced']['second_mention_prediction_rate']:.2%}升至{bm['old_combo']['balanced']['second_mention_prediction_rate']:.2%}，整体准确率从{bm['baseline']['balanced']['accuracy']:.2%}变为{bm['old_combo']['balanced']['accuracy']:.2%}。原句涨点不应再单独作为通用去偏证据。",
        '统一细粒度组合的三个模型在此平衡集均有小幅正点估计，但相对基线的探索性家族区间均包含0；个别源减半或整均值对照表现较好，不能据此事后选出新赢家或断言所有干预均无效。',
        '语法补测实际整均值/细粒度扰动RMS比：'+str({n:{label:{case:round(v['whole_to_reference_rms_ratio'],4) for case,v in zz.items()} for label,zz in z.items()} for n,z in energy['models'].items()})+'。']
    old_endpoint=margins['models']['qwen3']['historical']['old_combo'];rr=old_endpoint['reconstructed']
    rows += ['',
        '输出端另外作C=(r_m+r_f)/2、D=(r_m−r_f)/2分解，其中r是第二实体相对第一实体的logit差。用干预C/基线D及基线C/干预D重组预测，只用于区分共同答案角色变化与两份身份文本差异变化。',
        f"旧Qwen3历史Type2：基线均对{old_endpoint['baseline']['both_correct']:.2%}，实际干预{rr['actual']['both_correct']:.2%}；仅替换共同项C的输出拼接为{rr['C_only']['both_correct']:.2%}，仅替换差异项D为{rr['D_only']['both_correct']:.2%}。这支持重点排查共同答案角色偏移，而非直接认定内部偏见已被消除。",
        '共同项也含有效语法信息，差异项也不是纯社会偏见；这不是实际内部干预或因果中介比例，不能把重组收益相加。全C/D重组已逐条复原真实预测。','',
        '## 组件内相反贡献','',
        '|模型/MLP位置|正组源效应对齐|负组源效应对齐|','|---|---:|---:|']
    for name in ('qwen3','minicpm5','qwen25'):
        for site,groups in read(folder/name/'diagnostics.json').items():
            values=[(groups[g]['transmission']['restore_alignment']+groups[g]['transmission']['transplant_alignment'])/2 for g in ('transfer_pos','transfer_neg')]
            rows.append(f"|{name}/{site}|{values[0]:+.3f}|{values[1]:+.3f}|")
    rows += ['',
        '这些符号指促进/抵消源MLP减半产生的答案logit变化T，分组经真实恢复/移植检验；不是偏见消除比例，也不能相加为完整电路。',
        '源A/B是同一输入下原模型与源MLP减半，未直接等同于身份交换。因此该分解不单独证明身份特异性或社会偏见电路。',
        '传导正负组与部署候选使用的identity-gap导数组是两套分组。头输出坐标仅用于旧Qwen3锚点，不能称独立语义神经元。','',
        '## 实际无选项短答案生成','',
        '|模型|原模型成对均对|开发集赢家|细粒度组合|细粒度组合−原模型95%区间pp|','|---|---:|---:|---:|---|']
    for name,z in gen['models'].items():
        ci=z['fine_combo']['vs_baseline']['both_correct']['ci95']
        rows.append(f"|{name}|{z['baseline']['metrics']['strict']['both_correct']:.2%}|{z['selected']['metrics']['strict']['both_correct']:.2%}|{z['fine_combo']['metrics']['strict']['both_correct']:.2%}|[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]|")
    rows += ['',
        '每模型32个历史场景对，原生chat、关闭思考、贪心max32，仅prefill干预。主解析为整回答职业名，辅助首行解析及无法解析率均保留。',
        '这是有限短答案格式迁移，不能替代CEB自由续写，也不能据小样本不显著断言全体生成任务无效。','',
        '## 修正后的身份语义控制','',
        '原中性档案把红/蓝队伍与蓝卡事实混合，不能用于身份特异性结论。修正v2交换无关编号17/29，并删除多余代词解析前缀，使用直接问题；全部干预保持固定。',
        '该修正是在观察Qwen3旧语义结果后提出，明确为探索性补测；不并入原生Type1确认检验。编号与性别文字也不是逐token/逐范数匹配。','',
        '|模型|明确事实原模型准确率/性别翻转|细粒度准确率/性别翻转|原模型/细粒度必要身份读取准确率|','|---|---|---|---|']
    for name,z in semantic['models'].items():
        m=z['metrics'];a=m['gender_explicit_direct']['baseline'];b=m['gender_explicit_direct']['fine_combo'];r=m['gender_required_direct']
        rows.append(f"|{name}|{a['accuracy']:.2%}/{a['flip']:.2%}|{b['accuracy']:.2%}/{b['flip']:.2%}|{r['baseline']['accuracy']:.2%}/{r['fine_combo']['accuracy']:.2%}|")
    rows += ['',
        '事实任务金答案在两份档案中保持不变，翻转表示不稳定；必要身份读取任务金答案随档案交换，翻转本身不能称偏见。',
        '结合编号交换对照评估身份特异性，不能只用身份一致性或更小的logit差判定公平。如果明确事实任务本身接近满分、缺少身份影响，就不能用它证明干预消除了偏见。','',
        '## 可复核性与限制','',
        '153项工程测试通过。独立重建拟合符号、随机匹配索引、开发集可行性和赢家、行为指标及norm校准记录；7680条诊断预测重算了传导分数和家族区间。这些记录不是独立问题数量。',
        'v1主数值实现和冻结参数保持；执行v2修正误含旧L7基线的诊断列表，复核v3从冻结源数据在内存恢复漏存的职业对元数据，均保留原预测和版本说明。',
        '旧轮batch8、本轮batch4的FP16数值差异导致极少数零附近margin预测改变；所有方法比较使用本轮统一条件，不混用两轮百分比。',
        '没有运行第二模型完整QKV电路、CoT/RLHF对比、SAE、7B量化机制实验、新FairSteer跨模型训练或新的CEB评估。',
        '当前可支持局部相反贡献与强均值基线的实证分析；通用偏见机制、稳定跨任务去偏、特定子集必要性仍需要独立证据。','']
    path=ROOT/'research/2026-10-16_identity_replication_interpretation.md';path.write_text('\n'.join(rows),encoding='utf-8')
    write(folder/'completed_followup.json',{'completed':__import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'interpretation':str(path),'interpretation_sha256':sha(path),'verification_sha256':sha(folder/'verification.json'),
        'semantic_analysis_sha256':sha(folder/'semantic_v2_analysis.json'),'diagnostics_verification_sha256':sha(folder/'diagnostics_verification.json'),
        'grammar_balance_analysis_sha256':sha(folder/'grammar_balance/analysis.json'),
        'mention_order_analysis_sha256':sha(folder/'mention_order_analysis.json'),
        'native_type2_audit_sha256':sha(folder/'native_type2_mention_order_audit.json'),
        'margin_decomposition_sha256':sha(folder/'margin_decomposition.json'),
        'grammar_target_energy_audit_sha256':sha(folder/'grammar_balance/target_energy_audit.json'),
        'tests_log_sha256':sha(ROOT/'results/identity_replication_20261016_tests_v4.log')})
    print(path)
