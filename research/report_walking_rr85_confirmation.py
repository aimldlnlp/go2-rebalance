"""Report the preregistered development confirmation without pooling old cohorts."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import walking_rr85_confirmation as r


def main():
    proof=r.d.read(r.OUT/'completion_audit.json');assert proof['audit_passed'] and proof['study_complete']
    assert not (r.OUT/'REPORT.md').exists() and not (r.OUT/'comparison.png').exists()
    rows=r.d.read(r.OUT/'episodes.json');groups=proof['groups']
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,5.5));colors=['#355c7d','#9a6544']
    axes[0].bar(range(2),[g['successes'] for g in groups],color=colors,width=.5)
    for i,g in enumerate(groups):axes[0].text(i,g['successes']+.2,f"{g['successes']}/10",ha='center')
    axes[0].set_ylim(0,11.5);axes[0].set_yticks([0,2,4,6,8,10]);axes[0].set_ylabel('Successful missions');axes[0].set_title('Fresh development cohort')
    for seed in r.SEEDS:
        pair=[next(row for row in rows if row['condition']==c and row['seed']==seed) for c in r.CONDITIONS]
        values=[row['final_stable_hold_s'] for row in pair]
        axes[1].plot([0,1],values,color='#cccccc',linewidth=.8)
        for i,value in enumerate(values):axes[1].scatter(i,value,color=colors[i],s=30)
    axes[1].axhline(2,color='#777777',linestyle='--',linewidth=1)
    axes[1].set_ylabel('Final continuous stable hold (s)');axes[1].set_title('Paired pose results');axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks([0,1],['Healthy','RR 85%']);ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    fig.suptitle('RR 85%: independent development poses',fontsize=20)
    fig.text(.5,.025,'Ten new paired poses, seeds 23400–23409; no residual or training. Dashed line: required 2 s hold.\n'
        'All original mission criteria apply. Previous five-pose cohort remains separate.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.12,1,.92])
    for text in fig.findobj(matplotlib.text.Text):text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(fig.canvas.get_renderer());assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1
    fig.savefig(r.OUT/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Konfirmasi pengembangan RR85','',
        f"Gerbang konfirmasi {'LOLOS' if proof['confirmation_passed'] else 'BELUM LOLOS'}: sehat dan RR85 masing-masing wajib 10/10 sukses tanpa jatuh atau episode terputus.",'',
        'Cohort baru: sepuluh pose23400–23409, dipasangkan sehat/RR85. Walker nominal dibekukan, seluruh residual nol. Fault RR85 berarti motor RR_calf menerima85% limited torque setelah onset aktual0,50 m. Target akhir10% tetap belum diselesaikan.','',
        '| Kondisi | Sukses | Jatuh | Median hold (s) | Minimum hold (s) |','|---|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['condition']} | {g['successes']}/10 | {g['falls']}/10 | {np.median(g['holds_s']):.3f} | {min(g['holds_s']):.3f} |")
    lines+=['','Seluruh kriteria tetap:2 m/20 s, command maksimum0,20 m/s, posisi0,15 m, heading15°, cross-track0,20 m, final hold2 s dengan speed≤0,05 m/s, tanpa jatuh/nonfinite. Physics0,002 s; kontrol0,02 s. Tidak ada tuning setelah hasil.','',
        f"Audit memeriksa{proof['episodes_checked']} episode/{proof['physics_states_checked']} state;{proof['exact_healthy_fault_prefix_pairs']} prefix pasangan sehat/fault cocok persis. Nominal inference, reward, torque, onset, misi, dan nol residual diverifikasi. {proof['protected_prior_artifacts_unchanged']} file lama/input tetap cocok hash.",'',
        'Hasil lama5/5 pada pose23000–23004 tidak digabung menjadi15/15. Data konfirmasi ini merupakan data pengembangan setelah kandidat dipilih. Tidak ada seed final23100–23119, training, curriculum, atau promosi checkpoint yang dijalankan. Hasil hanya RR, satu nominal policy, sepuluh pose lantai datar; belum membuktikan strength lain, hardware atau manfaat curriculum.']
    with (r.OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    r.d.write(r.OUT/'delivery_audit.json',dict(report_sha256=r.d.digest(r.OUT/'REPORT.md'),figure_sha256=r.d.digest(r.OUT/'comparison.png'),
        source_sha256=r.d.digest(__file__),font_file=str(font),dpi=160))


if __name__=='__main__':main()
