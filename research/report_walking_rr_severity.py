"""Report every frozen severity level, including the two archived controls."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import walking_rr_severity as r


def main():
    p=r.verify_sources();out=r.OUT
    proof=r.s.read(out/'completion_audit.json');assert proof['audit_passed'] and proof['study_complete']
    assert not any((out/name).exists() for name in ['summary.json','REPORT.md','comparison.png'])
    rows=r.s.read(out/'historical_controls.json')+r.s.read(out/'episodes.json')
    groups=[]
    for strength in [1.,.75,.5,.25,.1]:
        group=sorted([row for row in rows if row['strength_remaining']==strength],key=lambda row:row['seed'])
        assert len(group)==5 and [row['seed'] for row in group]==r.SEEDS
        groups.append(dict(strength_remaining=strength,historical_control=bool(group[0]['historical_control']),
            episodes=5,successes=sum(row['success'] for row in group),falls=sum(row['fallen'] for row in group),
            median_goal_error_m=float(np.median([row['final_goal_error_m'] for row in group])),
            median_final_progress_m=float(np.median([row['final_progress_m'] for row in group])),
            median_stable_hold_s=float(np.median([row['final_stable_hold_s'] for row in group])),
            failed_checks={key:sum(not row['checks'].get(key,True) for row in group) for key in set().union(*(row['checks'] for row in group))}))
    r.s.write(out/'summary.json',groups)
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6));x=np.arange(5)
    for offset,key,color,label in [(-.16,'successes','#355c7d','Mission success'),(.16,'falls','#9a6544','Falls')]:
        axes[0].bar(x+offset,[g[key] for g in groups],width=.29,color=color,label=label)
        for i,g in enumerate(groups):axes[0].text(i+offset,g[key]+.07,str(g[key]),ha='center',fontsize=11)
    axes[0].set_ylim(0,5.7);axes[0].set_yticks(range(6));axes[0].set_ylabel('Episodes out of 5');axes[0].set_title('Mission outcomes');axes[0].legend(frameon=False,fontsize=10)
    for i,g in enumerate(groups):
        sample=[row for row in rows if row['strength_remaining']==g['strength_remaining']]
        for row in sample:
            axes[1].scatter(i,row['final_goal_error_m'],s=30,marker='s' if g['historical_control'] else 'o',
                edgecolors='#355c7d',facecolors='white' if row['fallen'] else '#355c7d')
        axes[1].plot([i-.12,i+.12],[g['median_goal_error_m']]*2,color='#9a6544',linewidth=2.5)
    axes[1].axhline(.15,color='#777777',linestyle='--',linewidth=1,label='Goal tolerance 0.15 m')
    axes[1].set_ylim(bottom=0);axes[1].set_ylabel('Final position error (m)');axes[1].set_title('Final goal error');axes[1].legend(frameon=False,fontsize=10)
    for ax in axes:
        ax.set_xticks(x,['Healthy*','RR 75%','RR 50%','RR 25%','RR 10%*']);ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    fig.suptitle('Nominal walker: RR torque severity diagnosis',fontsize=20)
    fig.text(.5,.025,'Five paired development poses per level; no residual or training. * Archived controls, not new rollouts.\n'
        'Markers: episodes; horizontal marks: medians; hollow markers: falls. Original mission criteria remain unchanged.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.12,1,.92])
    for text in fig.findobj(matplotlib.text.Text):text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(fig.canvas.get_renderer());assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    complete_levels=[f"{g['strength_remaining']:.0%}" for g in groups if not g['historical_control'] and g['successes']==5 and g['falls']==0]
    finding=('Level diagnosis dengan sukses 5/5 dan nol jatuh: '+', '.join(complete_levels)+'. Ini mendukung studi kelayakan curriculum, belum membuktikan manfaat training.') if complete_levels else 'Tidak ada level baru yang mencapai sukses 5/5 dengan nol jatuh. Studi ini belum menetapkan starting level nominal yang konsisten untuk curriculum.'
    lines=['# Diagnosis severity RR pada nominal walker','',finding,'',
        'Seluruh 15 episode baru selesai. Dua kontrol historis masing-masing berisi lima episode; tidak ada rollout kontrol baru. Semua residual nol dan tidak ada policy residual yang dimuat. Target fault 10% tetap belum dibuktikan oleh diagnosis strength yang lebih ringan.','',
        '![RR severity](comparison.png)','',
        '| Kondisi | Sumber | Sukses | Jatuh | Median error akhir (m) | Median hold akhir (s) |',
        '|---|---|---:|---:|---:|---:|']
    for g in groups:
        name='Sehat' if g['strength_remaining']==1 else f"RR {g['strength_remaining']:.0%}"
        lines.append(f"| {name} | {'Arsip' if g['historical_control'] else 'Baru'} | {g['successes']}/5 | {g['falls']}/5 | {g['median_goal_error_m']:.4f} | {g['median_stable_hold_s']:.2f} |")
    outcome=r.s.read(out/'outcome.json')
    lines+=['', 'Lulus berarti seluruh kriteria asli terpenuhi bersama: target 2 m, deadline 20 s, error posisi maksimal 0,15 m, heading maksimal 15 derajat, cross-track maksimal 0,20 m, hold akhir minimal 2 s dengan speed maksimal 0,05 m/s, tanpa jatuh/nonfinite, dan fault terjadi sebelum goal. Strength hanya variabel diagnosis. Fault dilatch pada pre-physics-step pertama progress minimal 0,50 m dan dikalikan sesudah torque limiting hanya pada RR_calf.', '',
        'Nominal policy, timestep 0,002/0,02 s, model terkalibrasi, navigator, joint/torque limits, dan observasi tetap. Variant step diverifikasi sama dengan source parent selain strength parameter dan referensi reward tinggi 0,338 m yang sudah digunakan baseline. Fixture 10% cocok persis dengan parent setelah latch. Semua 15 episode baru cocok dengan dua kontrol arsip sampai onset; keadaan setelah onset dapat berbeda sebagai efek closed-loop.', '',
        f"Audit memeriksa {proof['physics_states_checked']:,} state pada 15 episode baru serta 30 pasangan prefix arsip. Sepuluh kontrol arsip diaudit terpisah untuk physics, misi, nol residual, replay nominal dan reward. Seluruh {proof['protected_prior_artifacts_unchanged']} file input/lama cocok hash. Compute {outcome['compute_wall_s']:.2f} detik dari budget {p['wall_time_cap_s']} detik, tanpa perpanjangan. Trace lengkap dan kegagalan tetap disimpan.", '',
        'Sumber: protocol.json, design.json, historical_controls.json, control_audit.json, episodes.json, diagnostics.json, summary.json, completion_audit.json, dan outcome.json. Audit ulang: `.venv\\Scripts\\python.exe -B walking_rr_severity.py audit`. Tidak ada training, replikasi, seed konfirmasi/final, CUDA, atau preview Data.', '',
        'Batas pemeriksaan: hanya RR dan lima pose pengembangan historis pada lantai datar. Hasil tidak membuktikan generalisasi empat motor, strength lain, terrain, hardware, manfaat curriculum, atau ketidakmungkinan pendekatan residual. Tidak ada training atau eksperimen lanjutan otomatis.']
    with (out/'REPORT.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
    r.s.write(out/'delivery_audit.json',dict(report_sha256=r.s.digest(out/'REPORT.md'),figure_sha256=r.s.digest(out/'comparison.png'),
        summary_sha256=r.s.digest(out/'summary.json'),source_sha256=r.s.digest(__file__),font_file=str(font),dpi=160,groups_reported=5))
    print(groups,flush=True)


if __name__=='__main__':main()
