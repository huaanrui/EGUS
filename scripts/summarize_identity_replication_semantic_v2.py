"""Descriptive family intervals for repaired, explicitly exploratory controls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.identity_causal import pair_records
from mechanism.identity_replication import extended_metrics
from scripts.audit_identity_replication_records_v3 import restore_metadata
from scripts.summarize_identity_replication import bootstrap_delta


def paired(rr,field):
    record=pair_records([restore_metadata(r) for r in rr],[r['prediction'] for r in rr])
    record.sort(key=lambda r:(r['pair_id'],r['permutation']))
    return record,np.array([r[field] for r in record])


if __name__=='__main__':
    folder=ROOT/'results/identity_replication_20261015_v1';result={};count=0
    report=['# 修正后的身份语义对照：探索性补测','',
        '原中性档案使用红/蓝队伍，与蓝卡事实词重叠，不能作为身份特异性对照。新对照改为交换17/29编号，并使用直接问题。',
        '本补测是在看到Qwen3旧语义任务结果后声明的；全部干预参数保持冻结，不并入原生Type1确认检验。','',
        '|模型/方法|性别无关事实准确率|性别交换翻转率|编号无关事实准确率|编号交换翻转率|必要性别读取准确率|必要编号读取准确率|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('qwen3','minicpm5','qwen25'):
        out=folder/name;saved=read(out/'semantic_controls_v2.json');m=saved['metrics'];raw={}
        for case in m:
            by=defaultdict(list)
            for r in rows(out/'predictions'/('semantic_v2_'+case+'.jsonl')):by[r['method']].append(r);count+=1
            for method,rr in by.items():
                recomputed=extended_metrics([restore_metadata(r) for r in rr],np.array([r['scores'] for r in rr]))
                for field in ('accuracy','both_correct','flip','both_wrong','male_correct_female_wrong','male_wrong_female_correct'):
                    assert abs(recomputed[field]-m[case][method][field])<1e-12,(name,case,method,field)
                assert all(r['prediction']==int(np.argmax(r['scores'])) for r in rr)
            raw[case]=by
        contrasts={}
        for method in m['gender_explicit_direct']:
            vs,g=paired(raw['gender_explicit_direct'][method],'flip');_,n=paired(raw['neutral_explicit_direct'][method],'flip')
            _,g0=paired(raw['gender_explicit_direct']['baseline'],'flip');_,n0=paired(raw['neutral_explicit_direct']['baseline'],'flip')
            contrasts[method]={'gender_minus_neutral_flip':bootstrap_delta(vs,g,n),
                'change_in_gender_minus_neutral_flip_vs_base':bootstrap_delta(vs,g-n,g0-n0)}
            z=[m[c][method] for c in ('gender_explicit_direct','neutral_explicit_direct','gender_required_direct','neutral_required_direct')]
            report.append(f"|{name}/{method}|{z[0]['accuracy']:.2%}|{z[0]['flip']:.2%}|{z[1]['accuracy']:.2%}|{z[1]['flip']:.2%}|{z[2]['accuracy']:.2%}|{z[3]['accuracy']:.2%}|")
        result[name]={'metrics':m,'exploratory_family_contrasts':contrasts}
    report += ['','区间按原场景家族bootstrap4000次；这些是探索性、未作多重检验校正的诊断区间，不用于替代原9项/8项确认结论。',
        '事实任务的两份身份/编号输入金答案相同，翻转是不稳定；必要读取任务的金答案随档案交换，翻转本身不能视为偏见。',
        '编号与性别文本并非逐token或逐范数匹配，差分用于排查简单中性扰动解释，不单独证明社会身份特异性。','']
    write(folder/'semantic_v2_analysis.json',{'models':result,'prediction_records':count,'metrics_independently_recomputed':True,'status':'exploratory after initial-control inspection'})
    p=ROOT/'research/2026-10-16_identity_replication_semantic_v2.md';p.write_text('\n'.join(report),encoding='utf-8')
    print(p)
