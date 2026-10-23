from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanism.evidence_validation import Campaign
from mechanism.fairsteer_author import read

if __name__=='__main__':
    c=Campaign();r=read(c.out/'results.json');names=[m['name'] for m in c.c['models']]
    display={'qwen3':'Qwen3-1.7B','minicpm5':'MiniCPM5-2B','qwen25':'Qwen2.5-3B'}
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    methods=['base','local_fs','surface_direct','product_steer','evidence_direct']
    labels=['Base','Local FS-style','Surface + unknown','Product gate + steer','Evidence + unknown']
    colors=['#a4aeb7','#5f86ab','#b2a26b','#d68a4b','#408575']
    for ax,key,title in zip(axes[0],['test','confirmation'],['Historical BBQ: 704 questions','New confirmation: 640 questions']):
        x=np.arange(3);w=.15
        for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
            means=[r['models'][name][key]['aggregate'][method]['accuracy']['mean']*100 for name in names]
            sds=[r['models'][name][key]['aggregate'][method]['accuracy']['sd']*100 for name in names]
            bars=ax.bar(x+(j-2)*w,means,w,yerr=sds,capsize=2,color=color,label=label)
            if method in ('base','evidence_direct'):ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=4)
        ax.set_xticks(x,[display[n] for n in names],fontsize=9);ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_title(title)
        ax.set_ylabel('Accuracy (%)');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    for ax,reference,condition,title in [(axes[1,0],'surface_direct',None,'Evidence direct minus surface direct'),
                                         (axes[1,1],'base','disambig','Sufficient accuracy: evidence direct minus base')]:
        tick=[]
        for i,(name,key) in enumerate((n,k) for n in names for k in ('test','confirmation')):
            tag=f'{name}/{key}/evidence_direct-{reference}'+('/clear' if condition else '')
            v=r['primary'][tag];p=v['difference_pp'];lo,hi=v['ci95_pp'];color='#5f86ab' if key=='test' else '#408575'
            ax.errorbar(p,i,xerr=np.array([[p-lo],[hi-p]]),fmt='o',color=color,capsize=3,markersize=5)
            tick.append(display[name]+(' / BBQ' if key=='test' else ' / new'))
        ax.set_yticks(range(6),tick,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='#777',lw=1)
        if condition:ax.axvline(-2,color='#c47c45',ls='--',lw=1)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Difference (percentage points), crossed 95% interval')
        ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle('Three models, five training resamples; all source selections frozen before confirmation',fontsize=12)
    fig.savefig(c.out/'comparison.png',dpi=180);fig.savefig(c.out/'comparison.pdf');plt.close(fig)
    print('Plots saved')
