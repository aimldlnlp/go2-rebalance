"""Independent mission/physics and checkpoint review for the bounded walking pilot."""
from project_paths import project_file
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import sudden as s
import review_walking as independent
from walking_baseline import rotation
import walking_residual_env as w
import walking_residual_pilot as runner


def actual_gate(rows, seeds):
    expected = {(c,seed) for c in w.CONDITIONS for seed in seeds}
    if len(rows)!=25 or {(r['condition'],r['seed']) for r in rows}!=expected:
        return False
    counts = Counter(r['condition'] for r in rows if r['success'] and not r.get('interrupted',False))
    return not any(r['fallen'] or r.get('interrupted',False) for r in rows) and counts['healthy']==5 and all(counts[c]>=3 for c in w.CONDITIONS[1:])


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
            stages.append((f'validation_{n}',s.read(path),model))
    states, episodes, eligible, table, stage_results = 0, 0, [], [], {}
    default = np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
    for label, rows, model in stages:
        assert len({(r['condition'],r['seed']) for r in rows})==len(rows)
        assert all(r['seed'] in p['validation_seeds'] and r['condition'] in w.CONDITIONS for r in rows)
        for row in rows:
            stem = f"{row['condition']}_{row['seed']}"
            path = out/label/(stem+'.npz')
            assert row==s.read(out/label/(stem+'.json'))
            assert s.digest(path)==row['trace_sha256']
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
                    assert row['residual_checkpoint_sha256']==s.digest(out/f"checkpoint_{label.rsplit('_',1)[1]}.zip")
        passed=actual_gate(rows,p['validation_seeds'])
        stage_results[label]={'episodes':len(rows),'promotion_passed':passed,
            'successes':{c:sum(r['success'] for r in rows if r['condition']==c) for c in w.CONDITIONS},
            'falls':sum(r['fallen'] for r in rows)}
        if label!='baseline' and int(label.rsplit('_',1)[1]) in outcome['completed_checkpoint_validations']:
            n=int(label.rsplit('_',1)[1])
            acceptance=s.read(out/f'acceptance_{n}.json')
            assert acceptance['passed']==passed
            if passed:
                counts=stage_results[label]['successes']
                key=(min(counts[c] for c in w.CONDITIONS[1:]),sum(counts.values()),-float(np.mean([r['residual_effort'] for r in rows])),-n)
                eligible.append((key,n))
        table.append('| '+label+' | '+' | '.join(f"{stage_results[label]['successes'][c]}/5" for c in w.CONDITIONS)+f" | {stage_results[label]['falls']} | {'PASS' if passed else 'FAIL'} |")
    selected=max(eligible)[1] if eligible else None
    assert outcome['selected_checkpoint']==selected and outcome['pilot_promoted']==(selected is not None)
    logs=[__import__('json').loads(line) for line in (out/'training_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    resets=[r for r in logs if r['event']=='reset']
    assert all(r['pose_seed'] in p['reset_seed_pool'] and r['condition'] in w.CONDITIONS for r in resets)
    assert all(r['pose_seed']==p['reset_seed_pool'][r['episode_index']%len(p['reset_seed_pool'])] for r in resets)
    assert not set(r['pose_seed'] for r in resets).intersection(p['validation_seeds']+p['reserved_later_test_seeds'])
    for name,sha in design['prior_artifacts_sha256'].items():
        assert s.digest(project_file(runner.ROOT, name))==sha,name
    proof={'audit_passed':True,'pilot_promoted':selected is not None,'selected_checkpoint':selected,
        'training_executed':outcome['training_executed'],
        'training_steps':outcome['training_steps'],'stop_reason':outcome['stop_reason'],
        'independently_checked_complete_episodes':episodes,'physics_states_checked':states,
        'training_episode_scenarios':dict(Counter(r['condition'] for r in resets)),
        'protected_prior_artifacts_unchanged':len(design['prior_artifacts_sha256']),
        'validation':stage_results,'reserved_final_seeds_unused':True,'replicates_trained':False}
    path=out/'completion_audit.json'
    if path.exists():assert s.read(path)==proof
    else:s.write(path,proof)
    report='\n'.join(['# Pilot residual walking — satu seed', '',
        f"**Pilot {'LOLOS gerbang promosi' if selected is not None else 'BELUM LOLOS gerbang promosi'}.** Training {outcome['training_steps']:,} langkah; stop: {outcome['stop_reason']}. Waktu compute {outcome['compute_wall_s']/60:.1f} menit. Checkpoint terpilih: {selected}.", '',
        '| Controller | Sehat | FL | FR | RL | RR | Jatuh | Gerbang pilot |',
        '|---|---:|---:|---:|---:|---:|---:|---|',*table,'',
        'Gerbang pilot: sehat 5/5, setiap fault minimal 3/5, nol jatuh pada 25 misi. Setiap episode memakai kriteria misi 2 m/20 s yang tetap, command maksimum 0,20 m/s, error posisi <=0,15 m, heading <=15 derajat, cross-track <=0,20 m dan hold >=2 s pada kecepatan <=0,05 m/s. Ini target kelayakan pilot, bukan klaim reliabilitas akhir.', '',
        'Walker nominal, model, PD gains dan torque-speed envelope dibekukan. PPO hanya melatih residual MLP [64,64], seed 0, dengan action efektif maksimal ±0,10 rad sebelum joint/torque clipping dan fault. Identitas motor dan strength tidak ada dalam 60 observasi residual. Training pose memakai pool 24000–24999; validasi 23000–23004. Baseline residual nol diuji pada pose validasi yang sama. State RNG Python/NumPy/Torch training dipulihkan setelah reload dan validasi.', '',
        'Seluruh checkpoint dan kegagalan dipertahankan. Trace validasi memuat observasi residual, aksi residual, nominal policy, qpos/qvel, target, requested/limited/applied torque, force aktual, command dan kontak. Audit independen memeriksa hasil misi, onset first pre-step >=0,50 m, strength 10% pada motor terpilih setelah limiting, batas residual/joint/torque, rekonstruksi observasi biasa, serta prediksi ulang checkpoint.', '',
        f"Audit memeriksa {episodes} episode lengkap / {states:,} state fisika dan {proof['protected_prior_artifacts_unchanged']} artefak lama yang tetap utuh. Exposure reset training per kondisi: {proof['training_episode_scenarios']}.", '',
        ('Langkah berikutnya yang layak diusulkan adalah replikasi tiga seed independen dengan protokol baru; belum dijalankan.' if selected is not None else 'Tidak melanjutkan tiga seed, memperbesar residual, atau menambah budget otomatis. Bedah kurva training dan kegagalan per kondisi untuk merumuskan satu revisi studi berikutnya. Hasil ini tidak membuktikan residual secara umum mustahil, hanya konfigurasi dan budget pilot ini belum lolos.'), '',
        'Validasi checkpoint berulang adalah data pengembangan. Seed final 23100–23119 tetap tidak disentuh. Batas residual dan reward hanya satu konfigurasi; hasil simulasi datar tidak membuktikan hardware atau generalisasi strength/onset/terrain. Status jatuh mengikuti tinggi <0,16 m atau tilt >45 derajat/nonfinite. Preview Data tidak diaktifkan.', '',
        'Recheck: `.venv\\Scripts\\python.exe walking_residual_pilot.py audit`. Run menolak overwrite.'])+'\n'
    path=out/'REPORT.md'
    if path.exists():assert path.read_text(encoding='utf-8')==report
    else:
        with path.open('x',encoding='utf-8') as handle:handle.write(report)
    print(report.split('Gerbang pilot:')[0],flush=True)
    return proof
