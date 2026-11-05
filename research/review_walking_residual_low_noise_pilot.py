"""Independent low-initial-noise pilot review using the frozen physical/reward auditors."""
from project_paths import project_file
from collections import Counter
import numpy as np
import torch
from stable_baselines3 import PPO
import sudden as s
import review_walking as independent
from walking_baseline import rotation
import walking_residual_height_env as w
import walking_residual_low_noise_pilot as runner
from review_walking_residual_height_pilot import actual_gate, audit_preflight, audit_reward_and_nominal


def audit():
    torch.set_num_threads(1)
    p = runner.verify_sources()
    out = runner.OUT
    outcome = s.read(out/'outcome.json')
    design = s.read(out/'design.json')
    tests = s.read(out/'execution.json')
    assert tests['tests_sha256'] == s.digest(out/'tests-before.log')
    assert outcome['training_steps'] <= p['max_training_control_steps']
    baseline_rows = s.read(out/'baseline/episodes.json') if (out/'baseline/episodes.json').exists() else []
    baseline_initial = {(r['condition'],r['seed']):r['initial_state_sha256'] for r in baseline_rows}
    stages = [('baseline',baseline_rows,None)]
    for n in p['checkpoints']:
        path = out/f'validation_{n}/episodes.json'
        if path.exists():
            checkpoint = out/f'checkpoint_{n}.zip'
            metadata = s.read(out/f'checkpoint_{n}.json')
            assert metadata['checkpoint_sha256']==s.digest(checkpoint) and metadata['training_steps']==n
            model = PPO.load(checkpoint,device='cpu')
            runner.finite_policy(model)
            assert model.num_timesteps==n
            assert model.seed == p['training_seed'] and model.device.type == 'cpu'
            for key, value in p['hyperparameters'].items():
                actual = model.policy_kwargs['net_arch'] if key=='net_arch' else getattr(model,key)
                assert actual == value, key
            stages.append((f'validation_{n}',s.read(path),model))
    if (out/'confirmation/episodes.json').exists():
        candidate = outcome['development_selected_checkpoint']
        stages.append(('confirmation',s.read(out/'confirmation/episodes.json'),PPO.load(out/f'checkpoint_{candidate}.zip',device='cpu')))
    initial=s.read(out/'initial_policy.json')
    assert initial['training_steps']==initial['ppo_epoch_updates']==0 and initial['sigma_learnable']
    np.testing.assert_allclose(initial['sigma'],.2,rtol=0,atol=2e-8)
    np.testing.assert_allclose(initial['log_std'],np.log(.2),rtol=0,atol=1e-7)
    env=w.WalkingResidualEnv('healthy')
    try:
        reference=PPO('MlpPolicy',env,seed=0,device='cpu',policy_kwargs={'net_arch':[64,64],'log_std_init':0.})
        assert runner.policy_record(reference)['shared_non_log_std_parameters_sha256']==initial['shared_non_log_std_parameters_sha256']
    finally:
        env.close()
    for label,_,model in stages:
        if model is not None:
            assert model.policy_kwargs['log_std_init']==p['ppo_fixed_defaults']['log_std_init']
            if label!='confirmation':
                n=label.rsplit('_',1)[1]
                np.testing.assert_array_equal(s.read(out/f'checkpoint_{n}.json')['sigma'],model.policy.log_std.detach().cpu().exp().numpy())
    states, episodes, eligible, table, stage_results = 0, 0, [], [], {}
    default = np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
    for label, rows, model in stages:
        assert len({(r['condition'],r['seed']) for r in rows})==len(rows)
        seeds = p['confirmation_seeds'] if label == 'confirmation' else p['validation_seeds']
        assert all(r['seed'] in seeds and r['condition'] in w.CONDITIONS for r in rows)
        for row in rows:
            stem = f"{row['condition']}_{row['seed']}"
            path = out/label/(stem+'.npz')
            assert row==s.read(out/label/(stem+'.json'))
            assert s.digest(path)==row['trace_sha256']
            if label != 'confirmation':
                assert row['initial_state_sha256']==baseline_initial[(row['condition'],row['seed'])]
            if row.get('interrupted'):
                assert not row['success']
                continue
            states += independent.verify_episode(path,row,p)
            episodes += 1
            with np.load(path,allow_pickle=False) as a:
                q,v = a['qpos'],a['qvel']
                commands = a['command']
                actions = a['control_residual_actions']
                inputs, predictions = a['residual_observations'],a['residual_policy_actions']
                assert inputs.shape==(len(actions),60) and predictions.shape==actions.shape
                np.testing.assert_array_equal(actions,np.clip(predictions,-1,1))
                nominal = default+.25*a['control_actions']
                target = np.clip(nominal+.1*actions, np.array([[-1.0472,-1.5708,-2.7227],[-1.0472,-1.5708,-2.7227],[-1.0472,-.5236,-2.7227],[-1.0472,-.5236,-2.7227]]).reshape(12),
                    np.array([[1.0472,3.4907,-.83776],[1.0472,3.4907,-.83776],[1.0472,4.5379,-.83776],[1.0472,4.5379,-.83776]]).reshape(12))
                np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:len(q)-1])
                reconstructed=[]
                for i,tick in enumerate(range(0,len(q)-1,10)):
                    r=rotation(q[tick,3:7])
                    previous=np.zeros(12) if i==0 else actions[i-1]
                    reconstructed.append(np.r_[r.T@v[tick,:3],v[tick,3:6],r.T@np.array([0.,0.,-1.]),
                        commands[tick],q[tick,7:]-default,v[tick,6:]*.05,nominal[i]-default,previous])
                np.testing.assert_array_equal(inputs,np.asarray(reconstructed,dtype=np.float32))
                effort=float(np.mean(actions**2))
                assert abs(effort-row['residual_effort'])<1e-12
                if model is None:
                    assert np.all(actions==0)
                else:
                    expected=np.asarray([model.predict(obs,deterministic=True)[0] for obs in inputs])
                    np.testing.assert_allclose(predictions,expected,rtol=0,atol=1e-7)
                    n = outcome['development_selected_checkpoint'] if label == 'confirmation' else label.rsplit('_',1)[1]
                    assert row['residual_checkpoint_sha256']==s.digest(out/f"checkpoint_{n}.zip")
                audit_reward_and_nominal(a, row)
        passed=actual_gate(rows,seeds)
        stage_results[label]={'episodes':len(rows),'promotion_passed':passed,
            'successes':{c:sum(r['success'] for r in rows if r['condition']==c) for c in w.CONDITIONS},
            'falls':sum(r['fallen'] for r in rows)}
        if label not in ['baseline','confirmation'] and int(label.rsplit('_',1)[1]) in outcome['completed_checkpoint_validations']:
            n=int(label.rsplit('_',1)[1])
            acceptance=s.read(out/f'acceptance_{n}.json')
            assert acceptance['passed']==passed
            if passed:
                counts=stage_results[label]['successes']
                key=(min(counts[c] for c in w.CONDITIONS[1:]),sum(counts.values()),-float(np.mean([r['residual_effort'] for r in rows])),-n)
                eligible.append((key,n))
        table.append('| '+label+' | '+' | '.join(f"{stage_results[label]['successes'][c]}/5" for c in w.CONDITIONS)+f" | {stage_results[label]['falls']} | {'PASS' if passed else 'FAIL'} |")
    selected=max(eligible)[1] if eligible else None
    assert outcome['development_selected_checkpoint']==selected
    if selected is not None:
        confirmation = stage_results.get('confirmation')
        passed = confirmation is not None and confirmation['promotion_passed']
        if (out/'confirmation_acceptance.json').exists():
            acceptance = s.read(out/'confirmation_acceptance.json')
            assert acceptance == {'checkpoint_steps':selected,'passed':passed}
        if not passed:
            selected = None
    assert outcome['selected_checkpoint']==selected and outcome['pilot_promoted']==(selected is not None)
    logs=[__import__('json').loads(line) for line in (out/'training_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    resets=[r for r in logs if r['event']=='reset']
    assert all(r['pose_seed'] in p['reset_seed_pool'] and r['condition'] in w.CONDITIONS for r in resets)
    assert all(r['pose_seed']==p['reset_seed_pool'][r['episode_index']%len(p['reset_seed_pool'])] for r in resets)
    assert not set(r['pose_seed'] for r in resets).intersection(p['validation_seeds']+p['confirmation_seeds']+p['reserved_later_test_seeds'])
    for name,sha in design['prior_artifacts_sha256'].items():
        assert s.digest(project_file(runner.ROOT, name))==sha,name
    audit_preflight(p, out)
    complete = [r for r in logs if r['event']=='episode_end']
    fault_episodes = [r for r in complete if r['condition']!='healthy']
    assert not outcome['final_test_run'] and not outcome['replicate_training_run']
    training_complete=[r for r in logs if r['event']=='episode_end']
    partial=[r for r in logs if r['event']=='partial']
    assert sum(r['control_steps'] for r in training_complete+partial)==outcome['training_steps']
    training_groups={c:{'complete_episodes':sum(r['condition']==c for r in training_complete),
        'successes':sum(r['result']['success'] for r in training_complete if r['condition']==c),
        'falls':sum(r['result']['fallen'] for r in training_complete if r['condition']==c)} for c in w.CONDITIONS}
    sigmas={label:model.policy.log_std.detach().cpu().exp().numpy().tolist() for label,_,model in stages if model is not None}
    proof={'initial_log_std':p['ppo_fixed_defaults']['log_std_init'],'initial_sigma_normalized':.2,
        'sigma_learnable':True,'learned_sigma_by_stage':sigmas,'training_complete_groups':training_groups,
        'height_reference_m':w.HEIGHT_REFERENCE,'parity_episodes':45,
        'reward_and_nominal_replayed_episodes':episodes,
        'training_complete_fault_episodes':len(fault_episodes),
        'training_fault_episodes_receiving_fault':sum(r['result']['fault_onset_tick'] is not None for r in fault_episodes),
        'audit_passed':True,'pilot_promoted':selected is not None,'selected_checkpoint':selected,
        'training_executed':outcome['training_executed'],
        'training_steps':outcome['training_steps'],'stop_reason':outcome['stop_reason'],
        'independently_checked_complete_episodes':episodes,'physics_states_checked':states,
        'training_episode_scenarios':dict(Counter(r['condition'] for r in resets)),
        'protected_prior_artifacts_unchanged':len(design['prior_artifacts_sha256']),
        'validation':stage_results,'reserved_final_seeds_unused':True,'replicates_trained':False}
    path=out/'completion_audit.json'
    if path.exists():assert s.read(path)==proof
    else:s.write(path,proof)
    report='\n'.join(['# Pilot walking residual — eksplorasi awal sigma 0,2', '',
        f"**Pilot {'LOLOS gerbang promosi' if selected is not None else 'BELUM LOLOS gerbang promosi'}.** Training {outcome['training_steps']:,} langkah; stop: {outcome['stop_reason']}. Waktu compute {outcome['compute_wall_s']/60:.1f} menit. Checkpoint terpilih: {selected}.", '',
        'Satu perubahan dari pilot tinggi 0,338 m: log_std_init 0 menjadi ln(0,2), sekitar -1,60944. PPO dimulai dari awal dengan sigma aksi awal 0,2; sigma tetap dilatih, tanpa schedule atau clamp tambahan. Mean/critic awal selain parameter log_std identik dengan referensi seed 0. Reward, observasi, residual, controller dan seluruh ambang misi tetap. Tidak ada gate command nol atau label fault.', '',
        '| Controller | Sehat | FL | FR | RL | RR | Jatuh | Gerbang pilot |',
        '|---|---:|---:|---:|---:|---:|---:|---|',*table,'',
        'Gerbang pilot: sehat 5/5, setiap fault minimal 3/5, nol jatuh pada 25 misi. Setiap episode memakai kriteria misi 2 m/20 s yang tetap, command maksimum 0,20 m/s, error posisi <=0,15 m, heading <=15 derajat, cross-track <=0,20 m dan hold >=2 s pada kecepatan <=0,05 m/s. Ini target kelayakan pilot, bukan klaim reliabilitas akhir.', '',
        'Walker nominal, model, PD gains dan torque-speed envelope dibekukan. PPO hanya melatih residual MLP [64,64], seed 0, dengan action efektif maksimal ±0,10 rad sebelum joint/torque clipping dan fault. Identitas motor dan strength tidak ada dalam 60 observasi residual. Training pose memakai pool 24000–24999; validasi pengembangan historis 23000–23004. Konfirmasi baru 23300–23304 hanya dibuka bagi kandidat yang lolos pengembangan, masih dalam budget yang sama. Promosi memerlukan kedua cohort lolos gate; tanpa kandidat, konfirmasi tidak dijalankan. Baseline residual nol diuji pada pose validasi yang sama. State RNG Python/NumPy/Torch training dipulihkan setelah reload dan validasi.', '',
        'Seluruh checkpoint dan kegagalan dipertahankan. Trace validasi memuat observasi residual, aksi residual, nominal policy, qpos/qvel, target, requested/limited/applied torque, force aktual, command dan kontak. Audit independen memeriksa hasil misi, onset first pre-step >=0,50 m, strength 10% pada motor terpilih setelah limiting, batas residual/joint/torque, rekonstruksi observasi biasa, prediksi ulang checkpoint dan policy nominal, serta rekonstruksi independen seluruh komponen reward. Seluruh 45/45 fixture parity residual nol yang sudah ada diperiksa ulang melalui hash dan array sebelum training; environment tidak berubah, tidak ada replay simulasi parity baru.', '',
        f"Training lengkap per kondisi: {training_groups}. Sigma checkpoint per joint tersedia di completion_audit.json. Mengurangi sigma awal tidak menjamin sigma akhir tetap 0,2 atau kompensasi fault berhasil.", '',
        f"Audit memeriksa {episodes} episode lengkap / {states:,} state fisika dan {proof['protected_prior_artifacts_unchanged']} artefak lama yang tetap utuh. Exposure reset training per kondisi: {proof['training_episode_scenarios']}.", '',
        ('Langkah berikutnya yang layak diusulkan adalah replikasi tiga seed independen dengan protokol baru; belum dijalankan.' if selected is not None else 'Tidak melanjutkan tiga seed, memperbesar residual, atau menambah budget otomatis. Bedah kurva training dan kegagalan per kondisi untuk merumuskan satu revisi studi berikutnya. Hasil ini tidak membuktikan residual secara umum mustahil, hanya konfigurasi dan budget pilot ini belum lolos.'), '',
        'Validasi checkpoint berulang adalah data pengembangan. Seed final 23100–23119 tetap tidak disentuh. Batas residual dan reward hanya satu konfigurasi; hasil simulasi datar tidak membuktikan hardware atau generalisasi strength/onset/terrain. Status jatuh mengikuti tinggi <0,16 m atau tilt >45 derajat/nonfinite. Preview Data tidak diaktifkan.', '',
        'Recheck: `.venv\\Scripts\\python.exe walking_residual_low_noise_pilot.py audit`. Run menolak overwrite.'])+'\n'
    path=out/'REPORT.md'
    if path.exists():assert path.read_text(encoding='utf-8')==report
    else:
        with path.open('x',encoding='utf-8') as handle:handle.write(report)
    print(report.split('Gerbang pilot:')[0],flush=True)
    return proof
