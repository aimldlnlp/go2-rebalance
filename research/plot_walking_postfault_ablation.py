"""Render audited paired results with the established Latin Modern Roman style."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

import walking_postfault_ablation as a


def main():
    a.verify_sources()
    proof=a.s.read(a.OUT/'completion_audit.json')
    assert proof['audit_passed'] and proof['study_complete']
    rows=a.s.read(a.OUT/'diagnostics.json')
    assert len(rows)==40 and len(proof['paired'])==20
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    family=font_manager.FontProperties(fname=str(font)).get_name()
    assert family=='Latin Modern Roman'
    plt.rcParams.update({'font.family':family,'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    values=[r['post_net_progress_m'] for r in rows]
    low,high=min(0.,min(values)),max(values)
    margin=max(.05,.08*(high-low))
    chart_rows=[]
    for ax,condition,title in zip(axes.flat,a.CONDITIONS,['Front left','Front right','Rear left','Rear right'],strict=True):
        for seed,offset in zip(a.SEEDS,np.linspace(-.045,.045,5),strict=True):
            pair=[next(r for r in rows if (r['mode'],r['condition'],r['seed'])==(mode,condition,seed)) for mode in a.MODES]
            y=[r['post_net_progress_m'] for r in pair]
            expected=next(r for r in proof['paired'] if r['condition']==condition and r['seed']==seed)
            assert abs((y[1]-y[0])-expected['post_net_progress_delta_m'])<1e-12
            ax.plot([offset,1+offset],y,color='#b9b9b9',linewidth=.8,zorder=1)
            for x,row,color,marker in zip([offset,1+offset],pair,['#355c7d','#9a6544'],['o','s'],strict=True):
                ax.scatter(x,row['post_net_progress_m'],s=35,marker=marker,
                    edgecolors=color,facecolors='white' if row['fallen'] else color,zorder=2)
            chart_rows.append({'condition':condition,'seed':seed,'always_m':y[0],'off_m':y[1],
                'always_fall':pair[0]['fallen'],'off_fall':pair[1]['fallen']})
        summary=next(r for r in proof['summary'] if r['condition']==condition)
        for x,mode,color in zip([0,1],a.MODES,['#355c7d','#9a6544'],strict=True):
            median=summary['counts'][mode]['median_post_net_progress_m']
            ax.plot([x-.12,x+.12],[median,median],linewidth=2.5,color=color,zorder=3)
        ax.set_title(title)
        ax.set_xlim(-.3,1.3);ax.set_ylim(low-margin,high+margin)
        ax.set_xticks([0,1],['Residual on','Residual off'])
        ax.axhline(0,color='#777777',linewidth=.6)
        ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    for ax in axes[:,0]:ax.set_ylabel('Net progress after fault (m)')
    fig.suptitle('Post-fault residual ablation',fontsize=20)
    fig.text(.5,.02,'Five matched starts per fault. Lines connect the same seed; short bars show medians; hollow markers indicate falls.\n'
        'Both modes use the same residual before fault. Progress does not replace mission success.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.09,1,.95])
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert Path(font_manager.findfont(text.get_fontproperties())).resolve()==font.resolve()
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,text.get_text()
    path=a.OUT/'comparison.png'
    assert not path.exists()
    fig.savefig(path,dpi=160);plt.close(fig)
    a.s.write(a.OUT/'figure_values.json',chart_rows)
    a.s.write(a.OUT/'visual_validation.json',{'figure_sha256':a.s.digest(path),'font_file':str(font),
        'font_sha256':a.s.digest(font),'font_family':family,'dpi':160,
        'paired_points_verified':20,'font_and_bounds_checks_passed':True,
        'source_sha256':a.s.digest(__file__),
        'inputs_sha256':{name:a.s.digest(a.OUT/name) for name in ['completion_audit.json','diagnostics.json']}})
    print('Created paired comparison, LM Roman 10, PNG 160 dpi',flush=True)


if __name__=='__main__':main()
