"""Paired saved-trace comparison, with all checkpoints and privileged scope explicit."""
from project_paths import project_file
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import diagnose_walking_fault as d
import walking_residual_privileged_pilot as run


def main():
    proof=d.read(run.OUT/'completion_audit.json');assert proof['audit_passed']
    assert not (run.OUT/'REPORT.md').exists() and not (run.OUT/'comparison.json').exists()
    design=d.read(run.OUT/'design.json')
    for name,sha in design['source_sha256'].items():assert d.digest(project_file(run.ROOT, name))==sha,name
    stages=['baseline',*[f'validation_{n}' for n in run.verify_sources()['checkpoints']]]
    summaries=[];episodes=[];pairs=[];baseline_arrays=0
    for stage in stages:
        path=run.OUT/stage/'episodes.json'
        if not path.exists():continue
        current=d.read(path);old=d.read(run.parent.OUT/stage/'episodes.json')
        assert len({(r['condition'],r['seed']) for r in current})==len(current)
        for label,folder,rows in [('low_noise',run.parent.OUT,old),('privileged',run.OUT,current)]:
            summaries.append(dict(stage=stage,pilot=label,episodes=len(rows),
                successes={c:sum(r['success'] for r in rows if r['condition']==c) for c in run.w.CONDITIONS},
                falls=sum(r['fallen'] for r in rows)))
            healthy={}
            for r in rows:
                if r['condition']=='healthy' and not r['interrupted']:
                    p=folder/stage/f"healthy_{r['seed']}.npz";assert d.digest(p)==r['trace_sha256']
                    with np.load(p,allow_pickle=False) as a:
                        healthy[r['seed']]={k:a[k] for k in ['qpos','qvel','command','applied_torque','contacts','control_residual_actions']}
            for r in rows:
                if r['interrupted']:continue
                p=folder/stage/f"{r['condition']}_{r['seed']}.npz";assert d.digest(p)==r['trace_sha256']
                with np.load(p,allow_pickle=False) as a:
                    diagnosis=d.diagnose(a,r,None if r['condition']=='healthy' else healthy[r['seed']])
                    if stage=='baseline' and label=='privileged':
                        old_path=run.parent.OUT/stage/p.name
                        with np.load(old_path,allow_pickle=False) as ref:
                            for key in ref.files:
                                np.testing.assert_array_equal(a[key][:,:60] if key=='residual_observations' else a[key],ref[key],err_msg=key)
                        baseline_arrays+=1
                episodes.append(dict(pilot=label,stage=stage,condition=r['condition'],seed=r['seed'],
                    trace=str(p.relative_to(run.ROOT)),trace_sha256=r['trace_sha256'],episode_return=r['episode_return'],**diagnosis))
        for row in current:
            key=(row['condition'],row['seed']);reference=next(r for r in old if (r['condition'],r['seed'])==key)
            assert row['initial_state_sha256']==reference['initial_state_sha256']
            if row['interrupted'] or reference['interrupted']:continue
            before=next(e for e in episodes if (e['pilot'],e['stage'],e['condition'],e['seed'])==('low_noise',stage,*key))
            after=next(e for e in episodes if (e['pilot'],e['stage'],e['condition'],e['seed'])==('privileged',stage,*key))
            metrics=['post_net_progress_m','post_residual_rms_rad','post_weak_torque_limit_fraction',
                     'post_weak_residual_torque_delta_rms_Nm']
            pairs.append(dict(stage=stage,condition=key[0],seed=key[1],
                success_rescue=not before['success'] and after['success'],success_harm=before['success'] and not after['success'],
                fall_rescue=before['fallen'] and not after['fallen'],fall_harm=not before['fallen'] and after['fallen'],
                differences={k:after.get(k)-before.get(k) if before.get(k) is not None and after.get(k) is not None else None for k in metrics}))
    d.write(run.OUT/'comparison.json',dict(stages=summaries,paired_episodes=len(pairs),baseline_duplicate_controls=baseline_arrays,
        parameter_increase=1536,causal_limits='Input dimension changes capacity and initialization; one training seed, repeated development cohort. No matched-initialization or constant-feature arm.',
        metric_limits='Episode medians, not pooled timesteps. Falls truncate post-onset observation. Same-state torque delta is not a closed-loop causal effect.'))
    d.write(run.OUT/'comparison_episodes.json',episodes);d.write(run.OUT/'paired_changes.json',pairs)
    latest=next((stage for stage in reversed(stages) if (run.OUT/stage/'episodes.json').exists()),'baseline')
    latest_fault=[r for r in summaries if r['stage']==latest]
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6))
    for i,(label,color) in enumerate([('low_noise','#355c7d'),('privileged','#9a6544')]):
        stage_rows=[r for r in summaries if r['pilot']==label and r['stage']!='baseline']
        xs=[int(r['stage'].split('_')[-1])/1024 for r in stage_rows]
        axes[0].plot(xs,[sum(r['successes'][c] for c in run.w.CONDITIONS[1:]) for r in stage_rows],marker='o' if i==0 else 's',color=color,label=label.replace('_',' '))
        axes[1].plot(xs,[r['falls'] for r in stage_rows],marker='o' if i==0 else 's',color=color,label=label.replace('_',' '))
    axes[0].set_ylabel('Fault missions successful / 20');axes[0].set_ylim(-.5,20.5);axes[0].set_yticks([0,5,10,15,20])
    axes[1].set_ylabel('Falls / 25 missions');axes[1].set_ylim(-.5,25.5);axes[1].set_yticks([0,5,10,15,20,25])
    for ax in axes:
        ax.set_xticks([32,64,128]);ax.set_xlabel('Training steps (×1024)');ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=10)
    fig.suptitle('Privileged motor strength: paired pilot comparison',fontsize=20)
    fig.text(.5,.025,'Same five development poses per condition at each checkpoint. One training seed per pilot.\n'
        'Extra simulator information is privileged. No deployment or final-test claim.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.12,1,.92])
    for text in fig.findobj(matplotlib.text.Text):text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(fig.canvas.get_renderer());assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,text.get_text()
    fig.savefig(run.OUT/'comparison.png',dpi=160);plt.close(fig)
    outcome=d.read(run.OUT/'outcome.json')
    lines=['# Pilot residual dengan strength privileged','',
        f"Audit {'lulus' if proof['audit_passed'] else 'gagal'}. Pilot {'lolos' if outcome['pilot_promoted'] else 'belum lolos'} promosi. Training {outcome['training_steps']} langkah, stop {outcome['stop_reason']}, compute {outcome['compute_wall_s']:.2f} s dari1800 s. Checkpoint terpilih: {outcome['selected_checkpoint']}.",'',
        'Satu variant menambahkan12 strength motor aktual ke60 observasi, menjadi72. Semua strength1 sebelum onset; calf terpilih0,1 setelah fault dilatch. Suffix merefleksikan latch yang sudah terjadi pada physics step sebelumnya, tanpa label motor sebelum fault. Urutan motor mengikuti actuator model. Nominal walker, reward tinggi0,338 m, mixed conditions, residual±0,10 rad, PPO[64,64], sigma awal0,2 dan seluruh kriteria misi tetap.','',
        'Informasi fault ini privileged dari simulator. Policy belum dapat dipakai sebagai controller tanpa informasi simulator. Tidak ada estimator, student, curriculum atau navigator baru. Input72 menambah1536 parameter; initial weights tidak matched terhadap kontrol60. Satu seed tidak mengisolasi efek informasi dari capacity/initialization.','',
        '| Tahap | Pilot | Sehat | FL | FR | RL | RR | Jatuh |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('| '+r['stage']+' | '+r['pilot']+' | '+' | '.join(f"{r['successes'][c]}/5" for c in run.w.CONDITIONS)+f" | {r['falls']} |")
    lines+=['','Semua checkpoint dan kegagalan disimpan. Checkpoint terakhir adalah tahap terakhir, bukan otomatis model terbaik. Seleksi tetap sehat5/5, setiap fault minimal3/5, nol jatuh; promosi memerlukan conditional confirmation yang sama. Confirmation hanya dijalankan jika ada kandidat, dalam budget1800 s yang sama.','',
        f"Perbandingan memiliki{len(pairs)} pasangan episode pada pose/condition/tahap sama;{baseline_arrays} baseline adalah kontrol duplikat, bukan pengujian independen. Seluruh array baseline asli cocok persis; hanya suffix observasi baru yang ditambahkan. Detail progress setelah onset, residual RMS/bounds, torque limiting motor lemah dan selisih torsi pada state sama tersedia di comparison_episodes.json; rescue/harm dan selisih berpasangan tersedia di paired_changes.json.",'',
        f"Audit independen memeriksa{proof['independently_checked_complete_episodes']} episode/{proof['physics_states_checked']} state fisika, rekonstruksi72 observasi termasuk timing strength, prediksi ulang checkpoint dan nominal policy, reward, target, torque, onset dan kriteria misi. Preflight45/45 regression replay cocok semua array historis. {proof['protected_prior_artifacts_unchanged']} artefak lama tetap cocok hash.",'',
        'Kriteria tetap2 m/20 s, command≤0,20 m/s, error posisi≤0,15 m, heading≤15°, cross-track≤0,20 m, hold akhir≥2 s dengan speed≤0,05 m/s, tanpa jatuh/nonfinite. Fault10% limited torque dilatch pada progress≥0,50 m; timestep0,002/0,02 s.','',
        'Training seed0, pose24000–24999; validasi23000–23004. Data ini pengembangan berulang, bukan final test atau replikasi. Final23100–23119 tetap belum digunakan. Confirmation23300–23304 hanya jika gate pengembangan lolos. Tidak ada pengubahan ambang setelah hasil. Tidak ada tambahan budget otomatis.','',
        'Batas: satu training seed, simulator CPU dan lantai datar. Tidak ada generalisasi strength/onset/terrain atau hardware yang diuji. Gagal satu pilot tidak membuktikan residual mustahil; sukses dengan privileged strength belum membuktikan controller tanpa informasi simulator. Tidak ada klaim query graph karena graph tidak tersedia.']
    with (run.OUT/'REPORT.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
    for name,sha in design['prior_artifacts_sha256'].items():assert d.digest(project_file(run.ROOT, name))==sha,name
    d.write(run.OUT/'comparison_audit.json',dict(passed=True,paired_episodes=len(pairs),baseline_all_arrays_exact=baseline_arrays,
        trace_hashes_verified=True,all_initial_states_paired=True,prior_artifacts_unchanged=len(design['prior_artifacts_sha256']),
        source_sha256=d.digest(__file__),report_sha256=d.digest(run.OUT/'REPORT.md'),figure_sha256=d.digest(run.OUT/'comparison.png')))


if __name__=='__main__':main()
