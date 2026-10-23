from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanism.fairsteer_author import ROOT,read

folder=ROOT/'results/identity_replication_20261015_v1'
names=['qwen3','minicpm5','qwen25'];labels=['Qwen3-1.7B','MiniCPM5-2B','Qwen2.5-3B']
methods=['baseline','selected','fine_combo','whole_combo','random_mean']
legend=['Original','Dev-selected (no-op allowed)','Fine combination (diagnostic)','Norm-matched whole centered combination','Norm-matched random combinations (5 seeds)']
colors=['#64748b','#7c3aed','#0284c7','#059669','#ea580c']
fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
for ax,split,title in zip(axes,['historical','confirmation'],['Historical Type2: exploratory replication','Reserved Type1: 96 new families']):
    for j,m in enumerate(methods):
        vals=[]
        for name in names:
            data=read(folder/name/'evaluation.json')[split]
            norm=read(folder/name/'norm_control_results.json')['coref'][split]
            if m=='whole_combo':value=norm['fine_norm_whole_combo']['both_correct']
            elif m=='random_mean':value=np.mean([norm[f'fine_norm_random{s}']['both_correct'] for s in range(1701,1706)])
            else:value=data[m]['both_correct']
            vals.append(100*value)
        ax.bar(np.arange(3)+(j-2)*.15,vals,.14,label=legend[j],color=colors[j])
    ax.set_xticks(range(3),labels);ax.set_title(title,fontsize=11);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel('Both members correct (%)')
handles,ll=axes[0].get_legend_handles_labels();fig.legend(handles,ll,loc='lower center',ncol=2,frameon=False,fontsize=9)
fig.suptitle('Mean preservation and fine selection across three local models',fontsize=14)
fig.subplots_adjust(bottom=.24,top=.85,wspace=.08)
fig.savefig(folder/'identity_replication.png',dpi=180,bbox_inches='tight');fig.savefig(folder/'identity_replication.pdf',bbox_inches='tight')
stats=read(folder/'norm_comparisons.json')['models']
fig2,axes2=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
for ax,split,title in zip(axes2,['historical','confirmation'],['Historical Type2','Reserved Type1 (96 families)']):
    for j,(key,label,color) in enumerate([('vs_norm_whole','vs norm-matched whole','#059669'),('vs_norm_random_mean','vs norm-matched random mean','#ea580c')]):
        for i,name in enumerate(names):
            z=stats[name][split]['comparisons']['fine'][key];x=100*z['delta'];lo,hi=np.array(z['ci95'])*100
            ax.errorbar(x,i+(j-.5)*.16,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=3,label=label if i==0 else None)
    ax.axvline(0,color='#64748b',lw=1);ax.set_yticks(range(3),labels);ax.set_title(title);ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Fine combination: paired accuracy difference (pp)')
axes2[0].invert_yaxis();axes2[1].legend(frameon=False,loc='best',fontsize=8)
fig2.suptitle('Fine selection versus stronger controls: 95% family bootstrap intervals',fontsize=12)
fig2.tight_layout();fig2.savefig(folder/'identity_replication_effects.png',dpi=180,bbox_inches='tight');fig2.savefig(folder/'identity_replication_effects.pdf',bbox_inches='tight')
print(folder/'identity_replication.png')
