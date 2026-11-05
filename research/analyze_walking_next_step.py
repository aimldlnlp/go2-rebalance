"""Offline decision review: input information, exploration and causal ablation evidence."""
from project_paths import project_file, source_files
import csv
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import diagnose_walking_fault as d


OUT=d.ROOT/'results/walking_next_step_decision'
SENSORS=np.r_[np.arange(9),np.arange(12,36)]
TIMES=[-.1,.02,.1,.25,.5,1.,2.]
STAGES=['baseline','validation_32768','validation_65536','validation_131072']


def centroid_probe(features, labels, seeds):
    predictions=np.full(len(labels),-1,dtype=int)
    for seed in sorted(set(seeds)):
        train=seeds!=seed;test=~train
        assert set(seeds[train]).isdisjoint(set(seeds[test]))
        center=features[train].mean(axis=0)
        scale=np.maximum(features[train].std(axis=0),1e-6)
        x=(features-center)/scale
        centroids=np.asarray([x[train&(labels==k)].mean(axis=0) for k in range(5)])
        predictions[test]=np.argmin(np.sum((x[test,None,:]-centroids[None,:,:])**2,axis=2),axis=1)
    confusion=np.zeros((5,5),dtype=int)
    for actual,predicted in zip(labels,predictions,strict=True):confusion[actual,predicted]+=1
    assert np.all(predictions>=0)
    return {'correct':int(np.sum(predictions==labels)),'episodes':len(labels),
        'accuracy':float(np.mean(predictions==labels)),'confusion':confusion.tolist(),
        'prediction_by_episode':predictions.tolist()}


def cohort_probe(features, labels, seeds):
    if len(labels)!=25:
        return {'status':'not_scored_incomplete_cohort','observed_episodes':len(labels),
                'expected_episodes':25,'correct':None,'episodes':25,'accuracy':None,
                'reason':'A saved episode ends before this snapshot; no survivor-only score or imputation.'}
    assert np.bincount(labels).tolist()==[5]*5
    return {'status':'complete_cohort','observed_episodes':25,'expected_episodes':25,
            **centroid_probe(features,labels,seeds)}


def clipped_normal(mean, sigma):
    mean,sigma=np.broadcast_arrays(np.asarray(mean,dtype=float),np.asarray(sigma,dtype=float))
    assert np.all(sigma>0) and np.isfinite(mean).all()
    alpha,beta=(-1-mean)/sigma,(1-mean)/sigma
    cdf=lambda x:np.asarray([.5*(1+math.erf(v/math.sqrt(2))) for v in x.ravel()]).reshape(x.shape)
    lower,upper=cdf(alpha),cdf(beta)
    pa,pb=np.exp(-alpha**2/2)/math.sqrt(2*math.pi),np.exp(-beta**2/2)/math.sqrt(2*math.pi)
    clip=lower+1-upper
    second=(mean**2+sigma**2)*(upper-lower)+2*mean*sigma*(pa-pb)+sigma**2*(alpha*pa-beta*pb)+clip
    assert np.all((clip>=-1e-12)&(clip<=1+1e-12)) and np.all((second>=-1e-10)&(second<=1+1e-10))
    return np.clip(clip,0,1),np.clip(second,0,1)


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite decision analysis'
    ablation=d.ROOT/'results/walking_postfault_ablation'
    assert d.read(ablation/'completion_audit.json')['study_complete']
    protected=dict(d.read(ablation/'design.json')['prior_artifacts_sha256'])
    for name,sha in protected.items():assert d.digest(project_file(d.ROOT, name))==sha,name
    for path in [*ablation.rglob('*'),*source_files(d.ROOT)]:
        if path.is_file() and path.name not in ['analyze_walking_next_step.py','test_analyze_walking_next_step.py']:
            protected[str(path.relative_to(d.ROOT))]=d.digest(path)
    OUT.mkdir()
    d.write(OUT/'protocol.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'question':'Choose the next bounded experiment from existing evidence; do not assume motor-ID observation is the bottleneck.',
        'input_probe':'Current height pilot, all four stages, five paired seeds and five conditions. One snapshot per episode per elapsed time. Nearest-centroid classifier, train-only standardization, leave one pose seed out. Equal class counts; no timestep-level random split.',
        'snapshot_times_after_onset_s':TIMES,
        'anchor':'Use the actual common fault onset for each stage/seed; healthy shares this virtual time anchor. Snapshot is the first recorded control observation at/after requested time.',
        'features':{'sensors_only':SENSORS.tolist(),'full_observation':list(range(60))},
        'labels':'After onset: healthy or actual weak calf. Before onset: future assigned condition, a deliberately unobservable negative control, not an active fault label.',
        'interpretation':'Above-chance probe accuracy shows accessible information in this narrow cohort, not that PPO uses it or that explicit labels cannot help. Low accuracy from this one probe does not prove missing information. Pre-onset control should be 5/25 for five classes.',
        'exploration':'Read complete/partial training logs and saved Gaussian distributions. Compare per-episode training exposure, saved deterministic healthy actions, and analytically expected clipped Gaussian actions at exactly the same saved healthy inputs. Not a stochastic closed-loop evaluation.',
        'distribution_scales':[0.,.2,1.],
        'decision_rule':'Prioritize a controlled stochastic-vs-deterministic rollout check before new training if the learner has zero completed healthy training successes, deterministic healthy validation succeeds, and Gaussian exploration remains broad. This is a hypothesis-selection rule, not proof that noise causes mission failures. If motor identity is distinguishable in ordinary inputs, do not treat oracle labels as the established necessary next intervention.',
        'scope':'No new rollout, simulator construction, PPO updates, final seeds or preview servers. Offline statistical fits and saved-policy distribution evaluation only.'})
    d.write(OUT/'design.json',{'protocol_sha256':d.digest(OUT/'protocol.json'),
        'source_sha256':{name:d.digest(project_file(d.ROOT, name)) for name in ['analyze_walking_next_step.py','test_analyze_walking_next_step.py']},
        'protected_prior_artifacts_sha256':protected})
    print(f'Frozen offline decision review; {len(protected)} prior files protected',flush=True)


def run():
    amended=(OUT/'design_v2.json').exists()
    design=d.read(OUT/('design_v2.json' if amended else 'design.json'))
    assert d.digest(OUT/('protocol_v2.json' if amended else 'protocol.json'))==design['protocol_sha256']
    for name,sha in design['source_sha256'].items():assert d.digest(project_file(d.ROOT, name))==sha,name
    assert not (OUT/'decision.json').exists(), 'Refusing to overwrite decision'
    torch.set_num_threads(1)
    exposure=[];distributions=[];probes=[];identities=[]
    for pilot in ['walking_residual_pilot','walking_residual_height_pilot']:
        base=d.ROOT/'results'/pilot
        logs=[__import__('json').loads(x) for x in (base/'training_episodes.jsonl').read_text().splitlines()]
        complete=[r for r in logs if r['event']=='episode_end'];partial=[r for r in logs if r['event']=='partial']
        assert sum(r['control_steps'] for r in complete+partial)==131072
        for condition in d.CONDITIONS:
            group=[r for r in complete if r['condition']==condition]
            exposure.append({'pilot':pilot,'condition':condition,'complete_episodes':len(group),
                'successes':sum(r['result']['success'] for r in group),'falls':sum(r['result']['fallen'] for r in group),
                'completed_control_steps':sum(r['control_steps'] for r in group),
                'postfault_control_decisions':sum(max(0,r['control_steps']-r['result']['fault_onset_tick']//10-1) for r in group if r['result']['fault_onset_tick'] is not None),
                'fault_received':sum(r['result']['fault_onset_tick'] is not None for r in group)})
    base=d.ROOT/'results/walking_residual_height_pilot'
    for stage in STAGES:
        metadata=d.read(base/stage/'episodes.json')
        assert len(metadata)==25
        saved=[]
        for row in metadata:
            path=base/stage/f"{row['condition']}_{row['seed']}.npz"
            assert d.digest(path)==row['trace_sha256']
            with np.load(path,allow_pickle=False) as a:
                saved.append((row,a['residual_observations'].copy(),a['residual_policy_actions'].copy()))
        for elapsed in TIMES:
            examples=[]
            for row,inputs,actions in saved:
                peers=[r['fault_onset_tick'] for r,_,_ in saved if r['seed']==row['seed'] and r['condition']!='healthy']
                assert len(set(peers))==1 and peers[0] is not None
                tick=int(math.ceil((peers[0]+elapsed/.002)/10)*10)
                if tick<0 or tick//10>=len(inputs):continue
                examples.append({'condition':row['condition'],'seed':row['seed'],'tick':tick,
                    'label':d.CONDITIONS.index(row['condition']),'input':inputs[tick//10]})
            y=np.asarray([r['label'] for r in examples]);seeds=np.asarray([r['seed'] for r in examples])
            for features,indices in [('sensors_only',SENSORS),('full_observation',np.arange(60))]:
                x=np.asarray([r['input'][indices] for r in examples],dtype=np.float64)
                result=cohort_probe(x,y,seeds)
                if elapsed<0:assert result['correct']==5, 'Pre-onset leakage or mismatch'
                probes.append({'stage':stage,'elapsed_s':elapsed,'features':features,**result})
            identities.append({'stage':stage,'elapsed_s':elapsed,'rows':[{k:v for k,v in r.items() if k!='input'} for r in examples]})
        if stage=='baseline':continue
        n=stage.rsplit('_',1)[1];path=base/f'checkpoint_{n}.zip'
        assert d.digest(path)==d.read(path.with_suffix('.json'))['checkpoint_sha256']
        model=PPO.load(path,device='cpu');model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():parameter.requires_grad_(False)
        sigma=model.policy.log_std.detach().cpu().exp().numpy()
        for row,inputs,actions in saved:
            if row['condition']!='healthy':continue
            with torch.inference_mode():
                dist=model.policy.get_distribution(torch.as_tensor(inputs,dtype=torch.float32)).distribution
                mean=dist.mean.cpu().numpy().astype(float)
            np.testing.assert_allclose(np.clip(mean,-1,1),actions,rtol=0,atol=1e-6)
            values=[]
            for scale in [0.,.2,1.]:
                if scale==0:
                    clip=np.zeros_like(mean);second=np.clip(mean,-1,1)**2
                else:clip,second=clipped_normal(mean,sigma*scale)
                values.append({'sigma_scale':scale,'expected_raw_action_clipping_fraction':float(clip.mean()) if scale else None,
                    'expected_applied_residual_rms_rad':.1*float(np.sqrt(second.mean()))})
            distributions.append({'stage':stage,'seed':row['seed'],'sigma_normalized_by_joint':sigma.tolist(),
                'mean_unclipped_noise_sigma_rad':.1*float(sigma.mean()),'scales':values,
                'scope':'Analytic conditional actions at saved deterministic healthy states only; not new trajectories.'})
        print(f'{stage}: grouped information probes and saved Gaussian distribution checked',flush=True)
    d.write(OUT/'training_exposure.json',exposure)
    d.write(OUT/'input_information_probe.json',probes)
    d.write(OUT/'probe_episode_keys.json',identities)
    d.write(OUT/'exploration_distributions.json',distributions)
    healthy=next(r for r in exposure if r['pilot']=='walking_residual_height_pilot' and r['condition']=='healthy')
    latest=[r for r in distributions if r['stage']=='validation_131072']
    assert healthy['successes']==0 and healthy['complete_episodes']==25
    fixed_healthy=d.read(base/'validation_131072/episodes.json')
    assert sum(r['success'] for r in fixed_healthy if r['condition']=='healthy')==5
    decision={'recommendation':'Defer oracle-label PPO training. Next: preregister a fixed-checkpoint, paired action-noise calibration, deterministic vs current Gaussian noise vs 0.2x Gaussian noise; no PPO updates.',
        'why':'Training healthy missions are 0/25 while deterministic latest healthy validation is 5/5; learned Gaussian standard deviations remain close to 1 in normalized action units. This is a concrete unresolved exploration/evaluation mismatch. Motor-ID labels have not been isolated as the bottleneck.',
        'limits':'Training and validation differ in poses, weights over learning and action sampling. This comparison does not establish noise causality. Information-probe results are leave-one-pose-seed-out on five very similar development poses; not terrain/fault generalization or proof PPO uses that information.',
        'new_training_executed':False,'new_simulation_episodes':0,'final_seeds_used':False,
        'next_protocol':'Fixed checkpoint 131072 at height 0.338m, existing development seeds. Start with healthy missions to isolate the learning/evaluation gap. Same standardized Gaussian sequence per paired stochastic mode; keep learned mean, all observations, reward, clipping, physics, fault logic and mission thresholds unchanged. Report all modes; reduced noise is not a proven repair. Freeze budget and decision before rollout.'}
    d.write(OUT/'decision.json',decision)
    plot(probes,latest)
    report(exposure,probes,latest,decision)
    for name,sha in design['protected_prior_artifacts_sha256'].items():assert d.digest(project_file(d.ROOT, name))==sha,name
    d.write(OUT/'completion_audit.json',{'audit_passed':True,'protected_prior_artifacts_unchanged':len(design['protected_prior_artifacts_sha256']),
        'probe_rows':len(probes),'probe_cohorts':len(identities),'target_observations_per_probe_cohort':25,
        'complete_scored_probe_rows':sum(r['status']=='complete_cohort' for r in probes),
        'incomplete_unscored_probe_rows':sum(r['status']=='not_scored_incomplete_cohort' for r in probes),
        'analysis_amendment':amended,
        'healthy_distribution_episodes':len(distributions),'training_control_steps_per_pilot':131072,
        'new_simulation_episodes':0,'new_PPO_training':False,'offline_classifier_fits':True,
        'tests_source_sha256':d.digest(d.ROOT/'tests/test_analyze_walking_next_step.py')})
    print(decision['recommendation'],flush=True)


def plot(probes,latest):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for features,color,marker,label in [('sensors_only','#355c7d','o','Motion and joint sensors'),('full_observation','#9a6544','s','Full residual observation')]:
        group=[r for r in probes if r['stage']=='validation_131072' and r['features']==features and r['accuracy'] is not None]
        axes[0].plot([r['elapsed_s'] for r in group],[100*r['accuracy'] for r in group],color=color,marker=marker,label=label)
    axes[0].axhline(20,color='#777777',linestyle='--',linewidth=1,label='Five-class chance: 20%')
    axes[0].axvline(0,color='#777777',linewidth=.6)
    axes[0].set_ylim(0,105);axes[0].set_xlabel('Time relative to fault onset (s)')
    axes[0].set_xticks([-.1,0,.1,.25,.5]);axes[0].set_yticks([0,20,40,60,80,100])
    axes[0].set_ylabel('Leave-one-pose-seed-out accuracy (%)')
    axes[0].set_title('Information in existing inputs');axes[0].legend(fontsize=10)
    means=[np.mean([r['scales'][i]['expected_applied_residual_rms_rad'] for r in latest]) for i in range(3)]
    bars=axes[1].bar(range(3),means,color=['#355c7d','#62815c','#9a6544'])
    axes[1].bar_label(bars,fmt='%.3f',padding=4,fontsize=11)
    axes[1].set_xticks(range(3),['Deterministic','0.2x noise','Current noise'])
    axes[1].set_ylabel('Expected applied residual RMS (rad)');axes[1].set_ylim(0,max(means)*1.2)
    axes[1].set_yticks([0,.02,.04,.06,.08])
    axes[1].set_title('Same saved healthy inputs; analytic expectation')
    fig.suptitle('Evidence before choosing another training revision',fontsize=20)
    fig.text(.5,.025,'Fixed latest checkpoint; five development poses. Classifier accuracy is not PPO fault awareness.\n'
        'Noise estimates use saved states only. Later input probes are unscored when a fallen episode has no snapshot.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.1,1,.94])
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert Path(font_manager.findfont(text.get_fontproperties())).resolve()==font.resolve()
            b=text.get_window_extent(renderer)
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,text.get_text()
    fig.savefig(OUT/'decision_evidence.png',dpi=160);plt.close(fig)
    d.write(OUT/'figure_values.json',{'information_probes':[r for r in probes if r['stage']=='validation_131072'],
        'healthy_mean_conditional_residual_rms_rad':means,'font_file':str(font),'font_sha256':d.digest(font)})


def report(exposure,probes,latest,decision):
    lines=['# Keputusan langkah walking berikutnya','',
        '**Tunda pilot one-hot fault. Prioritaskan uji terkontrol terhadap noise aksi pada checkpoint tetap, tanpa training baru.** Usulan one-hot sebelumnya belum memiliki bukti bahwa identitas motor merupakan hambatan utama.','',
        '## Bukti yang mengubah prioritas','',
        'Ablation kausal yang sudah selesai menunjukkan residual aktif lebih aman pada fault depan: FL jatuh 1/5 vs 2/5 bila residual dimatikan; FR 0/5 vs 5/5. Seluruh kondisi tetap 0/5 sukses. Jadi residual berguna untuk sebagian stabilitas; mematikannya bukan solusi umum. RL/RR tetap tanpa jatuh dan tanpa sukses pada kedua mode.','',
        'Pada pilot tinggi 0,338 m, seluruh 25 episode sehat training lengkap gagal misi tetapi tidak jatuh. Validasi deterministik checkpoint terakhir pada lima pose lama berhasil 5/5. Ini perbedaan distribusi sampling, pose dan policy sepanjang training, sehingga belum mengisolasi penyebab. Tidak tepat langsung menyimpulkan noise sebagai akar penyebab.','',
        '| Kondisi training 0,338 m | Episode lengkap | Sukses | Jatuh | Keputusan kontrol setelah fault |','|---|---:|---:|---:|---:|']
    for r in exposure:
        if r['pilot']=='walking_residual_height_pilot':
            lines.append(f"| {r['condition']} | {r['complete_episodes']} | {r['successes']} | {r['falls']} | {r['postfault_control_decisions']} |")
    sigma=latest[0]['sigma_normalized_by_joint']
    lines+=['',f"Gaussian checkpoint terakhir masih memiliki mean sigma {np.mean(sigma):.4f} pada action ternormalisasi; sigma per joint berkisar {min(sigma):.4f}–{max(sigma):.4f}. Sebelum clipping, noise target joint berskala sekitar {1000*.1*np.mean(sigma):.1f} mrad. Nilai ini berasal dari parameter policy, bukan estimasi std torque atau noise sensor.",'',
        'Pada setiap input sehat tersimpan, mean policy dipertahankan dan ekspektasi action setelah clipping dihitung secara analitik untuk sigma saat ini dan sigma x0,2. Hasil hanya menggambarkan distribusi action di state yang sama. Mengurangi noise pada rollout akan mengubah state berikutnya; tidak ada prediksi sukses dari perhitungan ini.','',
        '![Evidence for next decision](decision_evidence.png)','',
        '## Apakah motor bisa dibedakan dari observasi sekarang?','',
        'Probe nearest-centroid memakai 25 episode per waktu: lima kondisi x lima pose. Standardisasi hanya menggunakan empat seed training; satu pose seed ditahan untuk evaluasi, bergiliran lima fold. Feature sensor tidak memasukkan command, aksi sebelumnya, target nominal, label motor, strength atau clock; hanya body velocity, gyro, gravity dan posisi/kecepatan joint. Probe observasi penuh diperiksa terpisah.','',
        '| Waktu relatif onset (s) | Sensor saja | Observasi penuh |','|---|---:|---:|']
    for t in TIMES:
        group=[r for r in probes if r['stage']=='validation_131072' and r['elapsed_s']==t]
        a=next(r for r in group if r['features']=='sensors_only');b=next(r for r in group if r['features']=='full_observation')
        label=lambda r:f"{r['correct']}/{r['episodes']}" if r['accuracy'] is not None else f"Tidak dinilai ({r['observed_episodes']}/25 tersedia)"
        lines.append(f"| {t:+.2f} | {label(a)} | {label(b)} |")
    lines+=['',
        'Sebelum onset, target probe adalah kondisi yang akan terjadi, bukan fault aktif; seluruh kondisi fisik masih sehat. Karena itu akurasi 5/25 merupakan negative control yang diharapkan. Sesudah onset, label adalah motor yang benar-benar melemah atau healthy. Pose reset sangat mirip, sehingga hasil ini tidak membuktikan generalisasi, identifikasi fault di hardware, atau bahwa PPO memanfaatkan informasi tersebut. Akurasi tinggi hanya menunjukkan bahwa informasi dapat dibedakan pada cohort ini; label eksplisit masih mungkin membantu efisiensi belajar, tetapi belum menjadi kebutuhan yang terbukti.','',
        'Satu episode FL_calf pada checkpoint terakhir jatuh 0,814 detik setelah onset. Snapshot detik 1 dan 2 karenanya tidak lengkap; kedua probe tidak diberi skor. Versi awal analisis sengaja berhenti saat menemukan cohort tidak lengkap. Source, protokol dan log kegagalan awal dipertahankan; protocol_v2.json hanya mengubah perlakuan missing menjadi tidak dinilai, tanpa survivor-only scoring, imputation, atau eksperimen baru.','',
        '## Rekomendasi konkret','',
        'Bekukan uji checkpoint terakhir pada lima pose sehat pengembangan yang sama: deterministik, sampling Gaussian checkpoint saat ini, dan sampling dengan sigma x0,2. Mean dipertahankan sebagai fungsi state, tanpa PPO update; gunakan deret epsilon Gaussian standar yang sama antar-mode stokastik untuk masing-masing pasangan. Ini eksperimen berikut yang diusulkan, belum dijalankan.','',
        'Endpoint primer tetap sukses misi dan jatuh; error akhir, kecepatan serta durasi hold menjadi diagnostik. Pertahankan reward, navigator, observasi, residual maksimal, clipping dan seluruh ambang misi. Bekukan jumlah episode, seed sampling, budget dan aturan interpretasi sebelum rollout. Jangan menyebut deret noise sebagai seed pose baru atau membuka seed final.','',
        'Jika sampling saat ini gagal dan sigma rendah menyelamatkan pasangan tanpa menambah jatuh, kandidat studi training berikutnya adalah satu perubahan eksplorasi awal. Jika tidak, hipotesis noise belum mendapat dukungan kausal; jangan otomatis mengubah log_std_init. Label oracle/memory/ruang residual tetap kandidat terpisah, bukan langkah wajib.','',
        '## Batas dan reproduksi','',
        'Tidak ada simulasi atau policy training baru. Ada evaluasi distribusi checkpoint pada input lama dan fit classifier statistik offline. Seluruh kontrol langkah training lengkap+parsial cocok dengan 131.072 pada tiap pilot; episode parsial tidak dimasukkan denominator episode lengkap. Tidak ada perbandingan raw return antara reward berbeda. Outcome training yang berubah sepanjang waktu tidak diperlakukan sebagai perbandingan kausal dengan checkpoint terakhir.','',
        'Seed final 23100–23119 dan konfirmasi 23300–23304 tidak dibuka. Artefak lama diperiksa melalui manifest; hasil analisis disimpan terpisah. Semua angka rinci, confusion matrix, identitas episode/fold, exposure dan distribusi ada dalam JSON pendamping.','']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines))


def finish():
    design=d.read(OUT/'design_v3.json')
    assert d.digest(OUT/'protocol_v2.json')==design['protocol_sha256']
    for name,sha in design['source_sha256'].items():assert d.digest(project_file(d.ROOT, name))==sha,name
    for name,sha in design['data_inputs_sha256'].items():assert d.digest(OUT/name)==sha,name
    exposure=d.read(OUT/'training_exposure.json');probes=d.read(OUT/'input_information_probe.json')
    distributions=d.read(OUT/'exploration_distributions.json');decision=d.read(OUT/'decision.json')
    latest=[r for r in distributions if r['stage']=='validation_131072']
    assert len(latest)==5 and len(probes)==56
    assert not (OUT/'REPORT.md').exists() and not (OUT/'decision_evidence.png').exists()
    plot(probes,latest)
    report(exposure,probes,latest,decision)
    for name,sha in design['protected_prior_artifacts_sha256'].items():assert d.digest(project_file(d.ROOT, name))==sha,name
    d.write(OUT/'completion_audit.json',{'audit_passed':True,
        'protected_prior_artifacts_unchanged':len(design['protected_prior_artifacts_sha256']),
        'probe_rows':len(probes),'probe_cohorts':28,'target_observations_per_probe_cohort':25,
        'complete_scored_probe_rows':sum(r['status']=='complete_cohort' for r in probes),
        'incomplete_unscored_probe_rows':sum(r['status']=='not_scored_incomplete_cohort' for r in probes),
        'healthy_distribution_episodes':len(distributions),'training_control_steps_per_pilot':131072,
        'new_simulation_episodes':0,'new_PPO_training':False,'offline_classifier_fits':True,
        'analysis_amendment':True,'visual_amendment':'Explicit ticks; completed from saved analysis without rerunning probes.',
        'source_sha256':design['source_sha256'],
        'inputs_sha256':design['data_inputs_sha256'],
        'figure_sha256':d.digest(OUT/'decision_evidence.png'),'report_sha256':d.digest(OUT/'REPORT.md')})
    print(decision['recommendation'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','finish'])
    {'prepare':prepare,'run':run,'finish':finish}[parser.parse_args().stage]()
