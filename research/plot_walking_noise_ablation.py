"""Render the audited noise calibration using the repository's plotting style."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

import walking_noise_ablation as a


def main():
    a.verify_sources()
    proof=a.s.read(a.OUT/'completion_audit.json');rows=a.s.read(a.OUT/'episodes.json')
    assert proof['audit_passed'] and proof['study_complete'] and len(rows)==15
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    assert font_manager.FontProperties(fname=str(font)).get_name()=='Latin Modern Roman'
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    colors=['#355c7d','#62815c','#9a6544'];markers=['o','s','^'];modes=list(a.MODES)
    values=[]
    for ax,key,title,ylabel,threshold in zip(axes,
            ['final_stable_hold_s','final_goal_error_m'],['Final stable hold','Final position error'],
            ['Continuous stable hold (s)','Goal error (m)'],[2.,.15],strict=True):
        for seed,offset in zip(a.SEEDS,np.linspace(-.045,.045,5),strict=True):
            group=[next(r for r in rows if r['mode']==mode and r['seed']==seed) for mode in modes]
            ys=[r[key] for r in group]
            ax.plot(np.arange(3)+offset,ys,color='#b9b9b9',linewidth=.8,zorder=1)
            for x,y,color,marker,row in zip(np.arange(3)+offset,ys,colors,markers,group,strict=True):
                ax.scatter(x,y,s=32,marker=marker,facecolors=color if row['success'] else 'white',edgecolors=color,zorder=3)
            values.append({'metric':key,'seed':seed,'values':ys})
        for i,mode in enumerate(modes):
            median=float(np.median([r[key] for r in rows if r['mode']==mode]))
            ax.plot([i-.12,i+.12],[median,median],color=colors[i],linewidth=2.5)
        ax.axhline(threshold,color='#222222',linestyle='--',linewidth=1)
        ax.set_title(title);ax.set_ylabel(ylabel)
        ax.set_xlim(-.3,2.3);ax.set_ylim(0,max(threshold,max(r[key] for r in rows))*1.13)
        ax.set_xticks(range(3),['Deterministic','0.2x noise','Current noise'])
        ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
        # Explicit in-range ticks exclude undisplayed automatic ticks from the figure-bounds check.
        ax.set_yticks(np.linspace(0,ax.get_ylim()[1],5))
    fig.suptitle('Paired healthy action-noise calibration',fontsize=20)
    fig.text(.5,.025,'Five matched poses; identical standardized noise for stochastic pairs. Filled markers: mission success; hollow: failure.\n'
        'Short bars: medians. Dashed limits: hold >=2 s and position error <=0.15 m; all other mission checks still apply.',
        ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.11,1,.94])
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert Path(font_manager.findfont(text.get_fontproperties())).resolve()==font.resolve()
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,text.get_text()
    path=a.OUT/'comparison.png';assert not path.exists()
    fig.savefig(path,dpi=160);plt.close(fig)
    a.s.write(a.OUT/'figure_values.json',values)
    a.s.write(a.OUT/'visual_validation.json',{'figure_sha256':a.s.digest(path),'font_file':str(font),'font_sha256':a.s.digest(font),
        'font_family':'Latin Modern Roman','dpi':160,'paired_pose_metrics_verified':10,
        'all_visible_text_inside_figure':True,'source_sha256':a.s.digest(__file__),
        'input_sha256':{name:a.s.digest(a.OUT/name) for name in ['completion_audit.json','episodes.json']}})
    print('Created LM Roman comparison, PNG 160 dpi',flush=True)


if __name__=='__main__':main()
