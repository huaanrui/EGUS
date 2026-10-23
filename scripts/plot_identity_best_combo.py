"""Static audited task-score comparison; no test-based method selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanism.fairsteer_author import ROOT,read

if __name__=='__main__':
    folder=ROOT/'results/identity_best_combo_20261016_v1'
    assert read(folder/'independent_verification.json')['all_checks_passed']
    summary=read(folder/'summary.json')['models'];names=list(summary)
    labels={'qwen3':'Qwen3-1.7B','minicpm5':'MiniCPM5-2B','qwen25':'Qwen2.5-3B'}
    methods=[('baseline','Original','#64748b'),('ref_selected','Previous dev-selected','#9ca3af'),
        ('best_task','New dev accuracy winner','#2563eb'),('best_constrained','New constrained winner','#059669')]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,field,title in zip(axes,('accuracy','both_correct'),('Single-answer accuracy','Both gender variants correct')):
        for j,(method,label,color) in enumerate(methods):
            values=[100*summary[n]['evaluation'][method][field] for n in names]
            bars=ax.bar(np.arange(3)+(j-1.5)*.19,values,.18,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
        ax.set_xticks(np.arange(3),[labels[n] for n in names]);ax.set_ylim(0,105);ax.set_title(title)
        old=100*summary['qwen3']['evaluation']['ref_old_combo'][field]
        ax.scatter([0],[old],s=70,marker='D',color='#d97706',label='Frozen old Qwen3 MLP+head',zorder=5)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylabel('%')
    handles,leg=axes[0].get_legend_handles_labels();fig.legend(handles,leg,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('WinoBias Type2: dev-selected combinations on 146 historical pairs\nExploratory; all choices frozen before this evaluation',fontsize=13)
    fig.savefig(folder/'task_score_comparison.png',dpi=180);fig.savefig(folder/'task_score_comparison.pdf');plt.close(fig)
    print(folder/'task_score_comparison.png')
