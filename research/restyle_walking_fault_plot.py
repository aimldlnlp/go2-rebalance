"""Re-render the audited figure using the repository's established plotting style."""
from project_paths import project_file
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

import diagnose_walking_fault as d


def main():
    out=d.OUT/'visual_revision_lmroman'
    assert not out.exists(), 'Refusing to overwrite visual revision'
    previous={str(p.relative_to(d.ROOT)):d.digest(p) for p in d.OUT.rglob('*') if p.is_file()}
    audit=d.read(d.OUT/'report_validation.json')
    for name,sha in audit['input_sha256'].items():assert d.digest(d.OUT/name)==sha,name
    for name,sha in audit['source_sha256'].items():assert d.digest(project_file(d.ROOT, name))==sha,name
    assert d.digest(d.OUT/'postfault_progress.png')==audit['figure_sha256']
    assert d.digest(d.OUT/'REPORT.md')==audit['report_sha256']
    groups=d.read(d.OUT/'figure_values.json')
    episodes=d.read(d.OUT/'episodes.json')
    assert len(groups)==32
    for group in groups:
        rows=sorted([r for r in episodes if (r['pilot'],r['stage'],r['condition'])==
                     (group['pilot'],group['stage'],group['condition'])],key=lambda r:r['seed'])
        assert len(rows)==5
        np.testing.assert_array_equal(group['episode_values_m'],[r['post_max_progress_gain_m'] for r in rows])
        assert group['median_m']==float(np.median(group['episode_values_m']))

    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    family=font_manager.FontProperties(fname=str(font)).get_name()
    assert family=='Latin Modern Roman'
    plt.rcParams.update({'font.family':family,'font.size':12,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    colors={'height_0270':'#355c7d','height_0338':'#9a6544'}
    labels={'height_0270':'Original height reference: 0.270 m','height_0338':'Revised height reference: 0.338 m'}
    top=1.08*max(max(g['episode_values_m']) for g in groups)
    for ax,condition,title in zip(axes.flat,d.CONDITIONS[1:],['Front left','Front right','Rear left','Rear right'],strict=True):
        for pilot,shift,marker in [('height_0270',-.13,'o'),('height_0338',.13,'s')]:
            for i,stage in enumerate(d.STAGES):
                group=next(g for g in groups if (g['pilot'],g['stage'],g['condition'])==(pilot,stage,condition))
                ax.scatter(i+shift+np.linspace(-.045,.045,5),group['episode_values_m'],s=18,
                           marker=marker,color=colors[pilot],alpha=.55)
                ax.plot([i+shift-.075,i+shift+.075],[group['median_m']]*2,
                        color=colors[pilot],lw=2.8,label=labels[pilot] if i==0 else None)
        ax.set_title(title)
        ax.grid(axis='y',color='#dddddd',linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_xticks(range(4),['Zero residual','32,768','65,536','131,072'])
        ax.set_ylim(0,top)
    for ax in axes[:,0]:ax.set_ylabel('Additional forward progress (m)')
    for ax in axes[1]:ax.set_xlabel('Checkpoint training steps')
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,fontsize=11,frameon=False)
    fig.suptitle('Maximum additional progress after motor fault',fontsize=20)
    fig.text(.5,.02,'Five matched starts per condition and checkpoint. Dots: episodes; short lines: medians.\n'
             'Measured from each onset until deadline or fall. More progress does not establish mission success.',
             ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.085,1,.86])
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert Path(font_manager.findfont(text.get_fontproperties())).resolve()==font.resolve()
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,text.get_text()
    out.mkdir()
    figure=out/'postfault_progress.png'
    fig.savefig(figure,dpi=160)
    plt.close(fig)
    for name,sha in previous.items():assert d.digest(project_file(d.ROOT, name))==sha,name
    d.write(out/'validation.json',{'source_sha256':{Path(__file__).name:d.digest(__file__)},
        'font_file':str(font),'font_sha256':d.digest(font),'font_family':family,
        'font_size_pt':12,'title_size_pt':20,'caption_size_pt':11,'dpi':160,
        'colors':colors,'figure_groups_verified':32,'episode_points_verified':160,
        'all_visible_text_uses_requested_font':True,'all_text_inside_figure':True,
        'figure_sha256':d.digest(figure),'protected_previous_files_sha256':previous,
        'new_simulation_episodes':0,'training':False})
    print(f'Rendered {figure}; all {len(previous)} previous files unchanged',flush=True)


if __name__=='__main__':main()
