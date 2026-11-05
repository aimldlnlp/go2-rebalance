"""Independent mission/physics and checkpoint review for the bounded walking pilot."""
from project_paths import project_file
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import sudden as s
import review_walking as independent
from walking_baseline import rotation, Actor
import walking_residual_height_env as w
import walking_residual_height_pilot as runner


def actual_gate(rows, seeds):
    expected = {(c,seed) for c in w.CONDITIONS for seed in seeds}
    if len(rows)!=25 or {(r['condition'],r['seed']) for r in rows}!=expected:
        return False
    counts = Counter(r['condition'] for r in rows if r['success'] and not r.get('interrupted',False))
    return not any(r['fallen'] or r.get('interrupted',False) for r in rows) and counts['healthy']==5 and all(counts[c]>=3 for c in w.CONDITIONS[1:])


def audit_reward_and_nominal(a, row):
    from walking_residual_ablation import rotations
    q, v = a['qpos'], a['qvel']
    ticks = np.arange(0,len(q)-1,10)
    end = np.minimum(ticks+10,len(q)-1)
    r = rotations(q)
    forward = np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    error = np.linalg.norm(q[:,:2]-(np.asarray(row['initial_xy'])+2*forward),axis=1)
    yaw = np.arctan2(r[:,1,0],r[:,0,0])
    heading = np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
    tilt = np.arccos(np.clip(r[:,2,2],-1,1))
    cross = np.abs((q[:,:2]-row['initial_xy'])@np.array([-forward[1],forward[0]]))
    stable = (error<=.15)&(heading<=np.deg2rad(15))&(np.linalg.norm(v[:,:2],axis=1)<=.05)
    body = np.einsum('nji,nj->ni',r[end],v[end,:3])
    commands, actions = a['command'][ticks], a['control_residual_actions']
    previous = np.vstack([np.zeros(12),actions[:-1]])
    fall, success = np.zeros(len(ticks)), np.zeros(len(ticks))
    fall[-1], success[-1] = float(row['fallen']), float(row['success'])
    raw = {'velocity_tracking':np.exp(-np.sum((body[:,:2]-commands[:,:2])**2,axis=1)/.15**2),
        'yaw_tracking':np.exp(-((v[end,5]-commands[:,2])/.5)**2),
        'goal_progress':error[ticks]-error[end], 'height_error':(q[end,2]-.338)**2,
        'tilt_squared':tilt[end]**2, 'cross_track_excess':np.maximum(0.,cross[end]-.20)**2,
        'stable_goal':stable[end].astype(float),'residual_effort':np.mean(actions**2,axis=1),
        'residual_change':np.mean((actions-previous)**2,axis=1),'fall':fall,'mission_success':success}
    np.testing.assert_array_equal(a['reward_component_names'],list(w.REWARD_WEIGHTS))
    components = np.stack([raw[k]*w.REWARD_WEIGHTS[k] for k in w.REWARD_WEIGHTS],axis=1)
    np.testing.assert_allclose(a['reward_components'],components,rtol=0,atol=1e-10)
    np.testing.assert_allclose(a['control_rewards'],components.sum(axis=1),rtol=0,atol=1e-10)
    assert abs(sum(a['control_rewards'])-row['episode_return'])<1e-8
    actor = Actor('go2_rl_lab')
    for i,tick in enumerate(ticks):
        obs = actor.observation(q[tick],v[tick],commands[i])
        np.testing.assert_array_equal(obs,a['control_observations'][i])
        raw_action, action = (np.zeros(12),np.zeros(12)) if i==0 else actor.predict(obs)
        np.testing.assert_array_equal(raw_action,a['control_raw_actions'][i])
        np.testing.assert_array_equal(action,a['control_actions'][i])


def audit_preflight(p, out):
    from datetime import datetime
    import verify_walking_residual_env as verified
    parity = s.read(out/'parity.json')
    assert parity['all_passed'] and len(parity['episodes'])==45
    assert s.read(out/'execution.json')['parity_sha256']==s.digest(out/'parity.json')
    assert datetime.fromisoformat(p['created_utc']) < datetime.fromisoformat(s.read(out/'execution.json')['started_utc'])
    for row in parity['episodes']:
        path = runner.ROOT/row['actual_trace']
        assert s.digest(path)==row['actual_trace_sha256']
        assert s.digest(runner.ROOT/row['trace'])==row['trace_sha256']
        with np.load(path,allow_pickle=False) as actual, np.load(runner.ROOT/row['trace'],allow_pickle=False) as old:
            for key in old.files:
                np.testing.assert_array_equal(actual[key],old[key])
        reference = s.read(runner.ROOT/row['metadata'])
        assert s.digest(runner.ROOT/row['metadata'])==row['metadata_sha256']
        assert all(verified.metric_differences(row['actual_result'],reference,1e-12).values())


def audit():
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
    proof={'height_reference_m':w.HEIGHT_REFERENCE,'parity_episodes':45,
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
    report='\n'.join(['# Pilot residual walking — referensi tinggi 0,338 m', '',
        f"**Pilot {'LOLOS gerbang promosi' if selected is not None else 'BELUM LOLOS gerbang promosi'}.** Training {outcome['training_steps']:,} langkah; stop: {outcome['stop_reason']}. Waktu compute {outcome['compute_wall_s']/60:.1f} menit. Checkpoint terpilih: {selected}.", '',
        'Satu perubahan reward: referensi tinggi 0,270 menjadi 0,338 m. PPO dimulai dari awal. Seluruh dinamika, bobot reward lain dan ambang misi tetap. Tidak ada gate command nol.', '',
        '| Controller | Sehat | FL | FR | RL | RR | Jatuh | Gerbang pilot |',
        '|---|---:|---:|---:|---:|---:|---:|---|',*table,'',
        'Gerbang pilot: sehat 5/5, setiap fault minimal 3/5, nol jatuh pada 25 misi. Setiap episode memakai kriteria misi 2 m/20 s yang tetap, command maksimum 0,20 m/s, error posisi <=0,15 m, heading <=15 derajat, cross-track <=0,20 m dan hold >=2 s pada kecepatan <=0,05 m/s. Ini target kelayakan pilot, bukan klaim reliabilitas akhir.', '',
        'Walker nominal, model, PD gains dan torque-speed envelope dibekukan. PPO hanya melatih residual MLP [64,64], seed 0, dengan action efektif maksimal ±0,10 rad sebelum joint/torque clipping dan fault. Identitas motor dan strength tidak ada dalam 60 observasi residual. Training pose memakai pool 24000–24999; validasi pengembangan historis 23000–23004. Konfirmasi baru 23300–23304 hanya dibuka bagi kandidat yang lolos pengembangan, masih dalam budget yang sama. Promosi memerlukan kedua cohort lolos gate; tanpa kandidat, konfirmasi tidak dijalankan. Baseline residual nol diuji pada pose validasi yang sama. State RNG Python/NumPy/Torch training dipulihkan setelah reload dan validasi.', '',
        'Seluruh checkpoint dan kegagalan dipertahankan. Trace validasi memuat observasi residual, aksi residual, nominal policy, qpos/qvel, target, requested/limited/applied torque, force aktual, command dan kontak. Audit independen memeriksa hasil misi, onset first pre-step >=0,50 m, strength 10% pada motor terpilih setelah limiting, batas residual/joint/torque, rekonstruksi observasi biasa, prediksi ulang checkpoint dan policy nominal, serta rekonstruksi independen seluruh komponen reward. Parity residual nol 45/45 fixture historis diperiksa sebelum training.', '',
        f"Audit memeriksa {episodes} episode lengkap / {states:,} state fisika dan {proof['protected_prior_artifacts_unchanged']} artefak lama yang tetap utuh. Exposure reset training per kondisi: {proof['training_episode_scenarios']}.", '',
        ('Langkah berikutnya yang layak diusulkan adalah replikasi tiga seed independen dengan protokol baru; belum dijalankan.' if selected is not None else 'Tidak melanjutkan tiga seed, memperbesar residual, atau menambah budget otomatis. Bedah kurva training dan kegagalan per kondisi untuk merumuskan satu revisi studi berikutnya. Hasil ini tidak membuktikan residual secara umum mustahil, hanya konfigurasi dan budget pilot ini belum lolos.'), '',
        'Validasi checkpoint berulang adalah data pengembangan. Seed final 23100–23119 tetap tidak disentuh. Batas residual dan reward hanya satu konfigurasi; hasil simulasi datar tidak membuktikan hardware atau generalisasi strength/onset/terrain. Status jatuh mengikuti tinggi <0,16 m atau tilt >45 derajat/nonfinite. Preview Data tidak diaktifkan.', '',
        'Recheck: `.venv\\Scripts\\python.exe walking_residual_height_pilot.py audit`. Run menolak overwrite.'])+'\n'
    path=out/'REPORT.md'
    if path.exists():assert path.read_text(encoding='utf-8')==report
    else:
        with path.open('x',encoding='utf-8') as handle:handle.write(report)
    print(report.split('Gerbang pilot:')[0],flush=True)
    return proof
