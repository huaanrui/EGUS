"""Plot only the predeclared primary judge; reliability is reported separately."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    output=Path(output);rows=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    rows=[r for r in rows if r['judge']=='minicpm5_2b']
    labels={'baseline':'Base model','fairsteer_prefill':'FairSteer: prefill','evidence_prefill':'Evidence: prefill',
            'fairsteer_every_step':'FairSteer: every step','evidence_every_step':'Evidence: every step'}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained',sharey=True)
    y=np.arange(len(rows));means=[r['mean_bias_score'] for r in rows]
    axes[0].barh(y,means,color=['#8c97a7','#5275ae','#477e78','#5275ae','#477e78'],height=.55)
    for i,value in enumerate(means):axes[0].text(value+.5,i,f'{value:.2f}',va='center',fontsize=9)
    axes[0].set_xlim(0,max(100,max(means)+8));axes[0].set_xlabel('Mean judged bias (0-99; lower is less bias)')
    changes=np.array([r['paired_vs_baseline']['mean_change'] for r in rows]);ci=np.array([r['paired_vs_baseline']['template_ci'] for r in rows])
    errors=np.maximum(0,np.stack([changes-ci[:,0],ci[:,1]-changes]))
    axes[1].axvline(0,color='#687384',lw=1)
    axes[1].errorbar(changes,y,xerr=errors,fmt='o',color='#2856a3',capsize=3)
    axes[1].set_xlabel('Change vs base (template cluster 95% interval)')
    axes[0].set_yticks(y,[labels[r['method']] for r in rows]);axes[0].invert_yaxis()
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.suptitle('CEB continuations — MiniCPM5-2B local judge',fontsize=13)
    fig.supxlabel('400 paired inputs; 13 category/template groups. Exploratory, uncalibrated ratings; quality not scored.',fontsize=8)
    fig.savefig(output/'local_bias_scores.png',dpi=180);fig.savefig(output/'local_bias_scores.pdf');plt.close(fig)


if __name__=='__main__':
    main(Path(__file__).resolve().parents[1]/'results/ceb_local_judge_20261013_v1')
