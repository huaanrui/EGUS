"""Render the pilot's saved paired estimates without changing selections."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fairsteer_improve_20261013_v2'


def main(output=OUT):
    output=Path(output)
    rows=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    reference=next(r for r in rows if r['name']=='fairsteer')
    selected=[r for r in rows if r['name'] not in ('baseline','fairsteer')]
    names={'tuned':'Strength / threshold','expanded':'Expanded labels','evidence':'Evidence gate','adaptive':'Benefit / strength','optimized':'Optimized direction','combined':'Evidence + optimized','unknown_calibration':'Unknown score calibration'}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained',sharey=True)
    pos=np.arange(len(selected))
    for ax,metric,label in zip(axes,['overall','disambig'],['Overall accuracy change (pp)','Disambiguated accuracy change (pp)']):
        key='accuracy' if metric=='overall' else metric
        values=np.array([100*(r[key]-reference[key]) for r in selected])
        bounds=np.array([r['paired_family_ci_pp'][metric] for r in selected])
        errors=np.maximum(0,np.stack([values-bounds[:,0],bounds[:,1]-values]))
        ax.axvline(0,color='#596579',linewidth=1)
        ax.errorbar(values,pos,xerr=errors,fmt='o',color='#2856a3',ecolor='#8aa7d3',capsize=3)
        for i,r in enumerate(selected):
            if r['test_constraint_failures'] or r.get('dev_constraint_failures'):ax.scatter(values[i],i,s=65,facecolors='none',edgecolors='#b84343',linewidths=1.5,zorder=3)
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(pos,[names[r['name']] for r in selected]);axes[0].invert_yaxis()
    fig.suptitle('FairSteer improvement pilot — paired family bootstrap, 95% intervals',fontsize=12)
    fig.supxlabel('Relative to refitted FairSteer; red outline: failed a dev/test constraint. Exploratory BBQ evaluation.',fontsize=8)
    fig.savefig(output/'improvement_results.png',dpi=180);fig.savefig(output/'improvement_results.pdf')
    plt.close(fig)
    print(output/'improvement_results.png')


if __name__=='__main__':main()
