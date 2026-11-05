"""Inspect joint-level saved traces and render the completed offline diagnosis."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import diagnose_walking_fault as d


JOINTS = [f'{leg}_{joint}' for leg in ['FL','FR','RL','RR'] for joint in ['hip','thigh','calf']]


def joint_metrics(a, onset):
    n=len(a['qpos'])-1
    mask=np.arange(0,n,10)>=onset
    actions=a['control_residual_actions'][mask]
    assert len(actions)>0
    bound=np.abs(actions)>=1.-1e-7
    clipping=np.abs(a['requested_torque'][onset:]-a['limited_torque'][onset:])>1e-9
    contacts=a['contacts'][onset:]
    result={'residual_bound_fraction':np.mean(bound,axis=0).tolist(),
        'positive_bound_fraction':np.mean(actions>=1.-1e-7,axis=0).tolist(),
        'negative_bound_fraction':np.mean(actions<=-1.+1e-7,axis=0).tolist(),
        'mean_residual_rad':np.mean(.1*actions,axis=0).tolist(),
        'residual_rms_rad':np.sqrt(np.mean((.1*actions)**2,axis=0)).tolist(),
        'torque_limit_fraction':np.mean(clipping,axis=0).tolist(),
        'contacts_fraction':np.mean(contacts,axis=0).tolist(),
        'all_four_contacts_fraction':float(np.mean(np.all(contacts,axis=1)))}
    np.testing.assert_allclose(np.array(result['positive_bound_fraction'])+result['negative_bound_fraction'],result['residual_bound_fraction'],rtol=0,atol=1e-12)
    return result


def main():
    proof=d.read(d.OUT/'completion_audit.json')
    assert proof['audit_passed'] and proof['episodes']==200
    rows=d.read(d.OUT/'episodes.json')
    summary=d.read(d.OUT/'summary.json')
    joint_rows=[]
    for row in rows:
        if row['condition']=='healthy' or row['stage']=='baseline':continue
        path=d.ROOT/row['trace']
        assert d.digest(path)==row['trace_sha256']
        with np.load(path,allow_pickle=False) as a:
            joint_rows.append({k:row[k] for k in ['pilot','stage','condition','seed']}|
                joint_metrics(a,row['fault_onset_tick']))
    assert len(joint_rows)==120
    d.write(d.OUT/'joint_diagnostics.json',joint_rows)
    lookup={(r['pilot'],r['stage'],r['condition']):r for r in summary}
    trained=[r for r in rows if r['condition']!='healthy' and r['stage']!='baseline']
    latencies=[r['first_action_difference_s'] for r in trained]
    assert len(latencies)==120 and all(v is not None for v in latencies)
    latest=[r for r in rows if r['pilot']=='height_0338' and r['stage']=='validation_131072' and r['condition']!='healthy']
    assert len(latest)==20 and sum(r['success'] for r in latest)==0

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#242424',
        'text.color':'#242424','xtick.color':'#343434','ytick.color':'#343434'})
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex=True,sharey=True)
    colors={'height_0270':'#32669A','height_0338':'#AD671C'}
    labels={'height_0270':'Lama: 0,270 m','height_0338':'Revisi: 0,338 m'}
    offsets=np.linspace(-.045,.045,5)
    figure_values=[]
    top=1.08*max(r['post_max_progress_gain_m'] for r in rows if r['condition']!='healthy')
    for ax,condition in zip(axes.flat,d.CONDITIONS[1:]):
        for pilot,shift,marker in [('height_0270',-.13,'o'),('height_0338',.13,'s')]:
            for i,stage in enumerate(d.STAGES):
                group=sorted([r for r in rows if (r['pilot'],r['stage'],r['condition'])==(pilot,stage,condition)],key=lambda r:r['seed'])
                values=[r['post_max_progress_gain_m'] for r in group]
                expected=lookup[(pilot,stage,condition)]['median']['post_max_progress_gain_m']
                assert abs(float(np.median(values))-expected)<1e-12
                ax.scatter(i+shift+offsets,values,s=18,marker=marker,color=colors[pilot],alpha=.55)
                ax.plot([i+shift-.075,i+shift+.075],[expected]*2,color=colors[pilot],lw=2.8,
                    label=labels[pilot] if i==0 else None)
                figure_values.append({'pilot':pilot,'stage':stage,'condition':condition,'episode_values_m':values,'median_m':expected})
        ax.set_title(condition,loc='left',pad=9)
        ax.grid(axis='y',color='#DDDDDD',lw=.6)
        ax.set_axisbelow(True)
        ax.set_xticks(range(4),['Nol','32.768','65.536','131.072'])
        ax.set_ylim(0.,top)
    for ax in axes[:,0]:ax.set_ylabel('Tambahan kemajuan maksimum (m)')
    for ax in axes[1]:ax.set_xlabel('Langkah training checkpoint')
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper right',bbox_to_anchor=(.97,.93),frameon=False,ncol=2)
    fig.suptitle('Kemajuan setelah fault tetap terbatas',x=.075,y=.985,ha='left',fontsize=15,weight='bold')
    fig.text(.075,.942,'Titik: lima seed berpasangan per kondisi. Garis pendek: median, bukan interval kepercayaan.',fontsize=10)
    fig.text(.075,.025,'Dihitung dari onset masing-masing hingga deadline/jatuh. Tambahan kemajuan tidak menggantikan kriteria sukses misi.',fontsize=9)
    fig.subplots_adjust(top=.865,bottom=.13,left=.075,right=.975,wspace=.15,hspace=.3)
    figure=d.OUT/'postfault_progress.png'
    assert not figure.exists()
    fig.savefig(figure,dpi=160,facecolor='white')
    plt.close(fig)
    d.write(d.OUT/'figure_values.json',figure_values)

    findings={'all_trained_fault_episodes_with_action_difference':len(trained),
        'action_difference_latency_range_s':[min(latencies),max(latencies)],
        'height_0338_last_fault_successes':0,'height_0338_last_fault_episodes':20,
        'height_0338_last_fault_falls':sum(r['fallen'] for r in latest),
        'height_0338_last_rear_stalls':sum(r['stall_window_start_s'] is not None for r in latest if r['condition'] in ['RL_calf','RR_calf']),
        'interpretation':'Nonzero state response is present. Frequent limiting of the weakened motor suppresses direct instantaneous torque corrections; other joints sometimes reach residual bounds. These facts do not identify a unique learning failure or establish that larger residual bounds would solve walking.',
        'next_bounded_study':'Before another training revision, preregister a post-fault residual-off ablation of the current checkpoint on the same development poses. Preserve the learned pre-fault trajectory. Compare always-on vs zero residual from the first scheduled control decision after onset. This tests the post-fault controller contribution, not fault awareness or global controllability.',
        'not_executed':'No new episodes, inference, training, seed release or task/server activation.'}
    d.write(d.OUT/'findings.json',findings)
    lines=['# Diagnosis walking setelah fault motor','',
        '**Ada respons residual, tetapi belum menghasilkan kompensasi walking yang berhasil.** Seluruh 120 episode fault dengan residual terlatih memiliki perbedaan aksi terhadap pasangan sehat dalam 2–24 ms. Riwayat state dan aksi sebelum fault cocok persis pada seluruh 160 pasangan fault-sehat, termasuk baseline residual nol. Perubahan aksi ini menunjukkan respons terhadap perubahan state; bukan deteksi identitas motor atau bukti responsnya benar.', '',
        'Dua pilot tetap menghasilkan nol sukses fault pada semua checkpoint. Pada checkpoint terakhir, revisi tinggi mempertahankan sehat 5/5 dan mengurangi jatuh fault dari 6/20 menjadi 1/20 pada cohort lama yang sama, tetapi fault masih 0/20. Tidak ada promosi. Perubahan jumlah jatuh adalah hasil cohort ini, bukan bukti konsistensi lintas training seed.', '',
        '![Kemajuan setelah fault](postfault_progress.png)','',
        '## Checkpoint terakhir revisi tinggi','',
        'Semua angka kontinu di tabel adalah median lima episode. Frekuensi torque limiting dihitung per motor lemah selama interval setelah onset, lalu diambil median antarepisode. RMS delta torsi adalah perbandingan sesaat pada state yang sama terhadap residual nol; bukan perubahan hasil rollout.', '',
        '| Fault | Sukses | Jatuh | Macet maju | Tambahan maju maksimum (m) | Motor lemah terkena limiting | RMS delta torsi motor lemah (Nm) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition in d.CONDITIONS[1:]:
        g=lookup[('height_0338','validation_131072',condition)];m=g['median']
        lines.append(f"| {condition} | {g['successes']}/5 | {g['falls']} | {g['stalls']}/5 | {m['post_max_progress_gain_m']:.3f} | {100*m['post_weak_torque_limit_fraction']:.1f}% | {m['post_weak_residual_torque_delta_rms_Nm']:.4f} |")
    lines += ['', 'Macet maju berarti ada jendela penuh 1 detik dengan kemajuan <=0,02 m meski mean command planar >=0,10 m/s. Ini indikator diagnostik yang dibekukan sebelum ekstraksi, bukan perubahan kriteria kelulusan. Jendela terakhir episode yang tidak lengkap tidak dihitung.','',
        '## Apa yang membatasi koreksi','',
        '1. **Motor lemah sering sudah terkena torque limiting sebelum skala 10%.** Pada RL/RR checkpoint terakhir revisi, median limiting 98,1%/97,3% interval setelah onset. Delta torsi langsung dari residual di motor itu hanya sekitar 0,0078/0,0032 Nm RMS. Pada state yang tersimpan, menaikkan target di arah yang sama sering tidak menambah torsi karena kedua target sudah dipotong ke batas yang sama. Ini hambatan lokal yang terukur, bukan bukti bahwa seluruh sistem tidak dapat dikendalikan.', '',
        '2. **Batas residual tidak menjadi satu penjelasan untuk semua motor.** Pada RL, sebagian joint penopang lain sering mencapai +/-0,10 rad. Pada RR, fraksi aksi di batas jauh lebih rendah. Karena itu, data tidak mendukung menaikkan seluruh batas residual sebagai solusi umum.', '',
        '| Fault | Joint dengan fraksi residual di batas tertinggi | Median fraksi | Median residual (rad; rerata per episode) |',
        '|---|---|---:|---:|']
    for condition in d.CONDITIONS[1:]:
        group=[r for r in joint_rows if (r['pilot'],r['stage'],r['condition'])==('height_0338','validation_131072',condition)]
        bounds=np.median([r['residual_bound_fraction'] for r in group],axis=0)
        means=np.median([r['mean_residual_rad'] for r in group],axis=0)
        index=int(np.argmax(bounds))
        lines.append(f'| {condition} | {JOINTS[index]} | {100*bounds[index]:.1f}% | {means[index]:+.4f} |')
    lines += ['',
        '3. **Belakang berhenti maju tanpa jatuh.** RL/RR masing-masing 0/5 jatuh dan 5/5 mengalami macet maju. Kontak kaki dicatat sebagai fraksi geom menyentuh ground; ini proxy kontak, bukan gaya kontak, distribusi beban, atau validasi fase gait. Nilai per kaki ada dalam episodes.json dan joint_diagnostics.json. Tidak ada pengukuran gaya kontak untuk menyimpulkan transfer beban.', '',
        '## Satu langkah berikut yang layak','',
        '**Bekukan ablation residual-off setelah onset sebelum training baru.** Gunakan checkpoint terakhir revisi sebagai objek diagnosis yang tetap, bukan model terbaik. Pada pose pengembangan yang sama, pertahankan seluruh controller dan residual sebelum fault. Bandingkan residual biasa dengan residual nol mulai keputusan kontrol terjadwal pertama setelah onset. Nilai onset dapat dipakai sebagai informasi eksperimen; gate ini tidak diusulkan sebagai controller deployable.', '',
        'Ablation tersebut membedakan apakah kontribusi residual setelah fault merugikan, membantu tetapi belum cukup, atau tidak menunjukkan perubahan pada metrik yang dipilih. Ia tidak membuktikan fault awareness, kapasitas kontrol optimal, atau solusi training. Jika hasil masih ambigu, jangan menyebut salah satu perubahan observasi, memory, reward, atau residual scale sebagai akar penyebab. Protokol, endpoint, budget dan aturan interpretasi harus dibekukan sebelum rollout. Eksperimen ini belum dijalankan.', '',
        '## Bukti dan batas','',
        f"Analisis hanya membaca 200 episode tersimpan: dua pilot x empat controller x lima kondisi x lima seed. Ada 100 pasangan antarpilot, termasuk 80 pasangan fault; baseline yang identik tidak dihitung sebagai evidence independen tambahan. Seluruh 25 pasangan baseline antarpilot cocok persis pada array fisika/command/aksi yang diperiksa. Sebanyak {proof['protected_prior_artifacts_unchanged']} file sebelumnya tetap utuh.", '',
        'Perbandingan antar-controller memakai seed/pose sama tetapi trajectory setelah onset berbeda. Onset tiap episode dipakai untuk alignment deskriptif; membandingkan sesudah onset tidak memulihkan state matching. Episode yang jatuh dipertahankan, tidak dipad sampai deadline. Statistik seluruh interval sesudah onset memiliki panjang berbeda; tabel memakai median per episode dan tidak mempool interval. Semua lima episode tetap masuk denominator, termasuk jatuh.', '',
        'Raw return kedua desain reward tidak dibandingkan. Delta torsi pada state tetap bukan simulasi counterfactual. Observasi nominal/residual, fault dan batas misi tidak diubah. Tidak ada training, policy inference, episode baru, final test, replikasi atau hardware. Seed 23100–23119 dan cohort konfirmasi 23300–23304 tidak dibuka. Preview Data tidak diaktifkan.', '',
        'Reproduksi analisis: `.venv\\Scripts\\python.exe diagnose_walking_fault.py prepare` lalu `run` pada folder hasil yang belum ada. Runner menolak overwrite. Uji: `.venv\\Scripts\\python.exe -B -m unittest test_diagnose_walking_fault test_report_walking_fault_diagnosis`.','']
    with (d.OUT/'REPORT.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines))
    d.write(d.OUT/'report_validation.json',{'joint_episode_rows':len(joint_rows),'figure_groups':len(figure_values),
        'figure_sha256':d.digest(figure),'report_sha256':d.digest(d.OUT/'REPORT.md'),
        'source_sha256':{Path(__file__).name:d.digest(__file__),
            'test_report_walking_fault_diagnosis.py':d.digest(d.ROOT/'tests/test_report_walking_fault_diagnosis.py')},
        'input_sha256':{name:d.digest(d.OUT/name) for name in ['episodes.json','summary.json','pairs.json','completion_audit.json']}})
    print('Report, per-joint evidence and figure created; no new simulation.',flush=True)


if __name__=='__main__':main()
