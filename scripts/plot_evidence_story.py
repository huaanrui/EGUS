from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mechanism.evidence_story import Study
from mechanism.fairsteer_author import read

if __name__=='__main__':
    s=Study();r=read(s.out/'results.json');models=list(r['models'])
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    keys=['base','fixed_local_fs','2200_product_steer1.5','2200_product_direct']
    labels=['Base','Local FS-style','Evidence + steer','Same gate + unknown'];colors=['#9da7b0','#527da5','#d68b45','#478572']
    for ax,metric,title in zip(axes,['accuracy','ambig','disambig'],['Overall accuracy','Insufficient evidence','Sufficient evidence']):
        x=np.arange(len(models));w=.18
        for j,(key,label,color) in enumerate(zip(keys,labels,colors)):
            values=[]
            for model in models:
                m=r['models'][model]['methods'][key];values.append(100*(m[metric] if metric=='accuracy' else m[metric]['accuracy']))
            rect=ax.bar(x+(j-1.5)*w,values,w,label=label,color=color)
            ax.bar_label(rect,fmt='%.1f',fontsize=8,padding=3)
        ax.set_xticks(x,['Qwen3-1.7B','MiniCPM5-2B']);ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Accuracy (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Matched 2,200-question training budget · raw zero-shot · exploratory BBQ retest',fontsize=12)
    fig.savefig(s.out/'comparison.png',dpi=180);fig.savefig(s.out/'comparison.pdf');plt.close(fig)
    print('Plots saved')
