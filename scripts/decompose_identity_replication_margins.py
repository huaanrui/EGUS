"""Endpoint score decomposition, explicitly not causal mediation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.identity_causal import comparison
from mechanism.identity_replication import extended_metrics
from scripts.analyze_identity_replication_mention_order import mention_order


def margin_prediction(margin,second_index):
    # The original two-logit argmax selects A on an exact tie, regardless of entity order.
    return int(second_index if margin>0 else 1-second_index if margin<0 else 0)


def split_pair(male,female):return (male+female)/2,(male-female)/2


if __name__=='__main__':
    out=ROOT/'results/identity_replication_20261015_v1';result={};inputs={}
    report=['# 身份共同项与差异项：输出端探索性分解','',
        '看到主结果后，使用冻结预测作代数分解，不重新推理、不选择方法。令r_m/r_f为第二实体对第一实体的logit差，C=(r_m+r_f)/2，D=(r_m−r_f)/2。',
        'C是在两种代词身份下共同的答案角色倾向，也包含有效语义/语法信息；D是这对身份文本的输出差异。C/D不能直接命名为纯位置偏见/纯社会偏见。',
        '将基线记为C0,D0，干预记为C1,D1，构造C1±D0和C0±D1的预测。它们只是输出分数的反事实拼接，不是实际内部干预、因果中介比例，也不可相加为收益分解。','',
        '|模型/数据/方法|基线均对|实际均对|只替换C均对|只替换D均对|平均C变化|平均绝对D变化|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('qwen3','minicpm5','qwen25'):
        result[name]={}
        for split in ('historical','confirmation'):
            data=(ROOT/'data/prepared/identity_combo_20261016_v1/historical.jsonl' if split=='historical' else ROOT/'data/prepared/identity_replication_20261015_v1/confirmation.jsonl')
            source={r['pair_id']:r for r in rows(data)};path=out/name/'predictions'/(split+'.jsonl');inputs[str(path)]=sha(path);inputs[str(data)]=sha(data)
            by=defaultdict(dict)
            for r in rows(path):by[r['method']][r['uid']]={**source[r['pair_id']],**r}
            order=list(by['baseline']);vs=[by['baseline'][u] for u in order];idx={u:i for i,u in enumerate(order)}
            pairs=defaultdict(dict)
            for r in vs:pairs[(r['pair_id'],r['permutation'])][r['gender']]=r['uid']
            basep=np.array([r['prediction'] for r in vs]);m0=extended_metrics(vs,predictions=basep)
            result[name][split]={}
            for method in ('fine_combo','selected')+(('old_l7','old_combo') if name=='qwen3' else ()):
                c_only=np.zeros(len(vs),int);d_only=c_only.copy();full=c_only.copy();changes=[]
                for pair in pairs.values():
                    assert set(pair)=={'male','female'}
                    uv=[pair[g] for g in ('male','female')];second=[];margins=[]
                    for key in ('baseline',method):
                        rr=[by[key][u] for u in uv]
                        si=[r['choices'].index(mention_order(r)[1]) for r in rr]
                        second=si
                        margins.append([r['scores'][i]-r['scores'][1-i] for r,i in zip(rr,si)])
                    c0,d0=split_pair(*margins[0]);c1,d1=split_pair(*margins[1]);changes.append([c1-c0,abs(d1)-abs(d0)])
                    for j,u in enumerate(uv):
                        sign=1 if j==0 else -1;k=idx[u]
                        c_only[k]=margin_prediction(c1+sign*d0,second[j])
                        d_only[k]=margin_prediction(c0+sign*d1,second[j])
                        full[k]=margin_prediction(c1+sign*d1,second[j])
                        assert full[k]==by[method][u]['prediction']
                metrics={k:extended_metrics(vs,predictions=p) for k,p in [('actual',full),('C_only',c_only),('D_only',d_only)]}
                result[name][split][method]={'baseline':m0,'reconstructed':metrics,
                    'mean_C_change':float(np.mean(changes,0)[0]),'mean_abs_D_change':float(np.mean(changes,0)[1]),
                    'actual_vs_base':comparison(vs,full,basep,'both_correct',4000,20261016),
                    'full_reconstruction_matches_all_saved_predictions':True}
                report.append(f"|{name}/{split}/{method}|{m0['both_correct']:.2%}|{metrics['actual']['both_correct']:.2%}|{metrics['C_only']['both_correct']:.2%}|{metrics['D_only']['both_correct']:.2%}|{np.mean(changes,0)[0]:+.3f}|{np.mean(changes,0)[1]:+.3f}|")
    write(out/'margin_decomposition.json',{'models':result,'inputs_sha256':inputs,'status':'post-hoc algebraic endpoint diagnostic, not causal mediation',
        'script_sha256':sha(Path(__file__)),'all_full_predictions_exactly_reconstructed':True})
    p=ROOT/'research/2026-10-16_identity_replication_margin_decomposition.md';p.write_text('\n'.join(report)+'\n',encoding='utf-8');print(p)
