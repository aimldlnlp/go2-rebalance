"""Report all frozen gain groups and paired poses; never promote a checkpoint."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

import walking_gain_ablation as a
import diagnose_walking_fault as d

FONT = Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
METRICS = ['post_net_progress_m','post_max_progress_gain_m','post_residual_rms_rad',
           'post_residual_bound_fraction','post_joint_clip_fraction',
           'post_torque_limit_fraction','post_weak_torque_limit_fraction',
           'post_weak_residual_torque_delta_rms_Nm','post_mean_tilt_deg']


def summarize(rows):
    assert len(rows) == 80
    assert {(r['gain'],r['condition'],r['seed']) for r in rows} == {
        (gain,condition,seed) for gain in a.GAINS for condition in a.CONDITIONS for seed in a.SEEDS}
    groups, pairs = [], []
    for condition in a.CONDITIONS:
        base = {r['seed']:r for r in rows if r['gain'] == 1 and r['condition'] == condition}
        for gain in a.GAINS:
            group = sorted([r for r in rows if r['gain'] == gain and r['condition'] == condition],key=lambda r:r['seed'])
            success = sum(r['success'] for r in group)
            falls = sum(r['fallen'] for r in group)
            paired = [dict(gain=gain,condition=condition,seed=r['seed'],
                success_delta=int(r['success'])-int(base[r['seed']]['success']),
                fall_delta=int(r['fallen'])-int(base[r['seed']]['fallen']),
                progress_delta_m=r['post_net_progress_m']-base[r['seed']]['post_net_progress_m']) for r in group]
            pairs.extend(paired)
            groups.append(dict(gain=gain,condition=condition,episodes=5,successes=success,falls=falls,
                stalls=sum(r['stall_window_start_s'] is not None for r in group),
                primary_direction=a.primary_direction(success-sum(r['success'] for r in base.values()),
                                                       falls-sum(r['fallen'] for r in base.values())),
                fall_rescues=sum(r['fallen'] is False and base[r['seed']]['fallen'] for r in group),
                fall_harms=sum(r['fallen'] and base[r['seed']]['fallen'] is False for r in group),
                medians={key:d.median([r[key] for r in group]) for key in METRICS},
                median_paired_progress_delta_m=float(np.median([r['progress_delta_m'] for r in paired])),
                fixed_horizon={str(t):dict(available_episodes=sum(r['fixed_horizon_progress_m'][str(t)] is not None for r in group),
                    median_progress_m=d.median([r['fixed_horizon_progress_m'][str(t)] for r in group])) for t in [.5,2.]}))
    return dict(groups=groups,paired=pairs)


def main():
    p = a.verify_sources()
    out = a.OUT
    proof = d.read(out/'completion_audit.json')
    assert proof['audit_passed'] and proof['study_complete']
    assert not any((out/name).exists() for name in ['summary.json','REPORT.md','comparison.png'])
    rows = d.read(out/'diagnostics.json')
    summary = summarize(rows)
    d.write(out/'summary.json',summary)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = ['#355c7d','#9a6544','#76854b','#a86183']
    markers = ['o','s','^','D']
    fig,axes = plt.subplots(2,2,figsize=(12.8,9))
    for ax,title,label in zip(axes.flat,['Mission success','Falls','Net progress after fault','Weak-motor torque limiting'],
            ['Episodes out of 5','Episodes out of 5','Forward gain (m)','Limited physics intervals (%)']):
        ax.set_title(title); ax.set_ylabel(label)
        ax.set_xticks(range(4),['FL','FR','RL','RR'])
        ax.grid(axis='y',color='#dddddd',linewidth=.6); ax.set_axisbelow(True)
    for j,(gain,color,marker) in enumerate(zip(a.GAINS,colors,markers)):
        for i,condition in enumerate(a.CONDITIONS):
            x = i+(j-1.5)*.18
            group = next(g for g in summary['groups'] if (g['gain'],g['condition']) == (gain,condition))
            for ax,key in zip(axes[0],['successes','falls']):
                ax.bar(x,group[key],width=.16,color=color,label=f'Gain {gain}' if i==0 else None)
                ax.text(x,group[key]+.07,str(group[key]),ha='center',va='bottom',fontsize=10)
            episodes = [r for r in rows if (r['gain'],r['condition']) == (gain,condition)]
            for ax,key,scale in [(axes[1,0],'post_net_progress_m',1),(axes[1,1],'post_weak_torque_limit_fraction',100)]:
                for r in episodes:
                    ax.scatter(x,r[key]*scale,s=24,marker=marker,edgecolors=color,
                               facecolors='white' if r['fallen'] else color,alpha=.8)
                value = group['medians'][key]*scale
                ax.plot([x-.065,x+.065],[value,value],color=color,linewidth=2.5)
    for ax in axes[0]:
        ax.set_ylim(0,5.6); ax.set_yticks(range(6))
    axes[1,0].axhline(0,color='#777777',linewidth=.7)
    axes[1,1].set_ylim(0,105); axes[1,1].set_yticks([0,25,50,75,100])
    fig.suptitle('Residual gain after fault: frozen low-noise policy',fontsize=19)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
    fig.text(.5,.025,'Five paired development poses per fault; one training seed. Hollow markers: falls; horizontal marks: medians.\n'
        'Post-onset duration ends at fall or deadline. Gain 1 is the historical replay control.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.085,1,.905])
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(FONT),size=text.get_fontsize()))
    fig.canvas.draw()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(fig.canvas.get_renderer())
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,text.get_text()
    fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    totals = {gain:dict(successes=sum(g['successes'] for g in summary['groups'] if g['gain']==gain),
                      falls=sum(g['falls'] for g in summary['groups'] if g['gain']==gain)) for gain in a.GAINS}
    lines = ['# Ablation gain residual setelah fault','',
        'Seluruh 80 episode selesai dan diaudit. Ringkasan per gain: '+
        '; '.join(f"gain {g}: sukses {totals[g]['successes']}/20, jatuh {totals[g]['falls']}/20" for g in a.GAINS)+'.', '',
        'Mission success dan jatuh adalah hasil utama. Gain 1 adalah kontrol historis, bukan checkpoint yang dipromosikan. Semua hasil per fault dan pose dipertahankan.', '',
        '![Perbandingan gain residual](comparison.png)','',
        '| Fault | Gain | Sukses | Jatuh | Median progress pasca-onset (m) | Median selisih berpasangan vs gain 1 (m) | Arah metrik primer |',
        '|---|---:|---:|---:|---:|---:|---|']
    for g in summary['groups']:
        lines.append(f"| {g['condition']} | {g['gain']} | {g['successes']}/5 | {g['falls']}/5 | {g['medians']['post_net_progress_m']:.3f} | {g['median_paired_progress_delta_m']:+.3f} | {g['primary_direction']} |")
    lines += ['', 'Arah primer mengikuti aturan beku: sukses meningkat tanpa tambahan jatuh, atau sukses tetap dengan jatuh berkurang, berarti lebih baik pada cohort ini. Kebalikannya lebih buruk. Metrik sama atau tradeoff tetap inconclusive. Return tidak dipakai memilih pemenang. Progress meningkat saja tidak memenuhi misi.', '',
        '| Fault | Gain | RMS residual (rad) | Action bound | Joint clipping | Torque limiting semua motor | Torque limiting motor lemah | Stall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        m = g['medians']
        lines.append(f"| {g['condition']} | {g['gain']} | {m['post_residual_rms_rad']:.4f} | {m['post_residual_bound_fraction']:.1%} | {m['post_joint_clip_fraction']:.1%} | {m['post_torque_limit_fraction']:.1%} | {m['post_weak_torque_limit_fraction']:.1%} | {g['stalls']}/5 |")
    outcome = d.read(out/'outcome.json')
    lines += ['', 'Nilai diagnostik adalah median lima metrik episode, bukan statistik gabungan timestep. Progress dihitung dari onset aktual hingga akhir episode. Jatuh memperpendek durasi pengamatan. summary.json menyimpan jumlah episode yang tersedia pada horizon tetap 0,5 dan 2 detik; median dengan data hilang hanya menggambarkan survivor.', '',
        'Intervensi memakai clip(gain × action, -1, 1) pada keputusan kontrol pertama setelah fault dilatch. Sebelumnya residual normal. Scale tetap ±0,10 rad; previous residual memakai action yang diterapkan. Semua dinamika, observasi, navigator, joint/torque limits, dan kriteria misi tetap. Hanya calf terpilih menerima 10% limited torque.', '',
        f"Audit memeriksa {proof['complete_episodes_checked']} episode, {proof['physics_states_checked']:,} state fisika, 20 replay historis persis, dan 60 prefix intervensi persis. Prediksi residual dan nominal, observasi, reward, target, torque, onset, dan misi diperiksa ulang. Seluruh {proof['protected_prior_artifacts_unchanged']} file input/lama cocok hash. Compute {outcome['compute_wall_s']:.2f} detik dari budget {p['wall_time_cap_s']} detik. Tidak ada training atau perpanjangan budget.", '',
        'Hasil mengukur intervensi gain pada policy ini dalam closed-loop. Gain mengubah action berikutnya melalui previous residual, state, history nominal, dan command. Hasil tidak mengisolasi amplitudo pada state tetap, tidak membuktikan identitas motor dikenali, dan tidak menetapkan penyebab tunggal. Torque saturation tidak membuktikan seluruh pendekatan residual mustahil.', '',
        'Fault latch adalah informasi privileged untuk diagnosis. Gain 0 mematikan residual hanya setelah fault, bukan baseline nominal sejak awal. Tidak ada promosi checkpoint atau perubahan default. Tidak ada seed konfirmasi/final, replikasi training, CUDA, atau preview Data yang dijalankan.', '',
        'Batas pemeriksaan: satu checkpoint dari satu seed training, lima pose pengembangan historis per fault, lantai datar, fault 10% dengan onset progress 0,50 m. Generalisasi strength, onset, terrain, dan hardware belum diuji. Gain lebih besar dapat menambah jatuh; satu seed training belum membuktikan generalisasi.', '',
        'Sumber: protocol.json, design.json, execution.json, outcome.json, episodes.json, diagnostics.json, summary.json, dan completion_audit.json. Audit ulang: `.venv\\Scripts\\python.exe -B walking_gain_ablation.py audit`. Runner dan laporan menolak overwrite.','']
    with (out/'REPORT.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    d.write(out/'delivery_audit.json',dict(report_sha256=d.digest(out/'REPORT.md'),figure_sha256=d.digest(out/'comparison.png'),
        summary_sha256=d.digest(out/'summary.json'),source_sha256=d.digest(__file__),font_file=str(FONT),
        dpi=160,visible_text_inside_figure=True,groups_reported=16,paired_records=80))
    print(totals,flush=True)


if __name__ == '__main__':
    main()
