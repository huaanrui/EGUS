"""Recompute actual target RMS ratios from frozen derived-probe shards."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mechanism.fairsteer_author import ROOT,rows,write,sha
from mechanism.identity_causal import variants
from scripts.identity_replication_grammar_balance import OUT,check_declaration
from scripts.summarize_identity_replication import load_shards

if __name__=='__main__':
    check_declaration();values=[variants(r,permutation=p) for r in rows(OUT/'rows.jsonl') for p in (0,1)]
    result={}
    for name in ('qwen3','minicpm5','qwen25'):
        result[name]={}
        for label in ('fine','old') if name=='qwen3' else ('fine',):
            ref='fine_combo' if label=='fine' else 'old_combo'
            control='norm_whole_combo' if label=='fine' else 'old_norm_whole_combo'
            a=load_shards(OUT.parent/name/'shards'/('grammar_balance_v1_'+ref))['edit_energy']
            b=load_shards(OUT.parent/name/'shards'/('grammar_balance_v1_'+control))['edit_energy']
            assert len(a)==len(b)==len(values)
            result[name][label]={}
            for case in ('native','relative','balanced'):
                mask=np.array([case=='balanced' or r['probe_case']==case for r in values])
                ea=float(a[mask].mean());eb=float(b[mask].mean())
                result[name][label][case]={'reference_mean_residual_energy':ea,'whole_mean_residual_energy':eb,'whole_to_reference_rms_ratio':float(np.sqrt(eb/ea))}
    p=OUT/'target_energy_audit.json';write(p,{'models':result,'script_sha256':sha(Path(__file__)),'calibration':'DEV only, not sample-wise target equality'})
    print(p)
