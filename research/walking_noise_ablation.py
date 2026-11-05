"""Bounded paired healthy action-noise calibration; no optimizer or training."""
from project_paths import project_file, source_files
from datetime import datetime, timezone
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO

import sudden as s
import walking_residual_height_env as w
import walking_residual_height_pilot as pilot
import review_walking as independent
from review_walking_residual_height_pilot import audit_reward_and_nominal
from walking_baseline import rotation
import diagnose_walking_fault as d


ROOT=s.ROOT
OUT=ROOT/'results/walking_noise_ablation'
MODES={'deterministic':0.,'low_noise':.2,'current_noise':1.}
SEEDS=list(range(23000,23005))


def sample_action(mean, sigma, epsilon, scale):
    mean,sigma,epsilon=[np.asarray(x,dtype=np.float32) for x in [mean,sigma,epsilon]]
    if (scale not in MODES.values() or any(x.shape!=(12,) or not np.isfinite(x).all() for x in [mean,sigma,epsilon])
            or np.any(sigma<=0)):
        raise ValueError('Invalid Gaussian parameters, standardized noise or scale')
    raw=mean.copy() if scale==0 else mean+(sigma*np.float32(scale))*epsilon
    return raw,np.clip(raw,-1.,1.)


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite noise ablation'
    parent=pilot.verify_sources()
    decision=ROOT/'results/walking_next_step_decision'
    assert s.read(decision/'completion_audit.json')['audit_passed']
    design=s.read(decision/'design_v3.json')
    protected=dict(design['protected_prior_artifacts_sha256'])
    for name,sha in protected.items():assert d.digest(project_file(ROOT, name))==sha,name
    for path in [*decision.rglob('*'),*source_files(ROOT)]:
        if path.is_file() and path.name not in ['walking_noise_ablation.py','test_walking_noise_ablation.py','plot_walking_noise_ablation.py']:
            protected[str(path.relative_to(ROOT))]=d.digest(path)
    checkpoint=pilot.OUT/'checkpoint_131072.zip'
    assert s.digest(checkpoint)==s.read(pilot.OUT/'checkpoint_131072.json')['checkpoint_sha256']
    OUT.mkdir();(OUT/'noise').mkdir()
    noise=[]
    for index,seed in enumerate(SEEDS):
        sampling_seed=33000+index
        epsilon=np.random.default_rng(sampling_seed).standard_normal((1000,12)).astype(np.float32)
        path=OUT/'noise'/f'pose_{seed}.npy'
        with path.open('xb') as handle:np.save(handle,epsilon,allow_pickle=False)
        noise.append({'pose_seed':seed,'sampling_seed':sampling_seed,'generator':'NumPy default_rng PCG64',
            'path':str(path.relative_to(ROOT)),'sha256':s.digest(path)})
    p={'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256':s.digest(checkpoint),'checkpoint_status':'Not promoted; fixed diagnostic checkpoint only.',
        'modes':MODES,'seeds':SEEDS,'condition':'healthy','episodes':15,'max_control_steps':15000,'wall_time_cap_s':300,
        'noise':noise,'action_rule':'raw = learned_mean(current_observation) + scale * learned_sigma * fixed_standard_noise[control_index]; clip to [-1,1] before environment. Deterministic uses the learned mean with no noise.',
        'pairing':'Same initial qpos/qvel per pose across all modes. Stochastic modes share exactly the same 1000x12 standardized Gaussian array, indexed by control step; not a replay of fixed mean actions. Sampling seeds are a separate namespace from pose seeds.',
        'distribution_scope':'Same diagonal Gaussian law as the checkpoint/SB3, manually sampled with frozen NumPy noise for common random numbers; no claim of bitwise training RNG replay.',
        'frozen':'Policy weights/mean function/std parameters, observations, reward, navigator, limits, model, timing and mission criteria unchanged. Only Gaussian standard-deviation scale changes.',
        'mission_acceptance':parent['mission_acceptance'],'height_reference_m':.338,
        'primary_metrics':['mission success','fall'],
        'secondary_metrics':['final goal error','final heading error','final speed','final stable hold duration','raw action clipping fraction'],
        'decision':'Low noise is supported for this healthy checkpoint cohort only if it increases successes versus current noise without increasing falls. Reverse changes oppose it. Equal primary outcomes or tradeoffs remain inconclusive. Deterministic is the historical reference; no fault compensation or checkpoint promotion is tested.',
        'next_training':'Even a rescue at fixed weights does not prove what caused all historical training failures or that changing log_std_init will improve learning. Any later training needs a separate one-change protocol and full fault gates.',
        'budget_scope':'From model/environment construction through all 15 episodes and serialization. Tests/preparation and final read-only audit outside this compute cap. Stop on budget/nonfinite; retain partial evidence; no retries or extension.',
        'training':False,'promotion_test':False,'fault_episodes':0,
        'reserved_final_seeds':list(range(23100,23120)),'reserved_confirmation_seeds':list(range(23300,23305)),
        'preview':'No server or scheduled task activation.'}
    s.write(OUT/'protocol.json',p)
    s.write(OUT/'design.json',{'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in ['walking_noise_ablation.py','test_walking_noise_ablation.py','plot_walking_noise_ablation.py']},
        'prior_artifacts_sha256':protected})
    print(f'Frozen 15 healthy episodes, 300s; {len(protected)} prior files protected',flush=True)


def verify_sources():
    pilot.verify_sources()
    p,design=s.read(OUT/'protocol.json'),s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==design['protocol_sha256']
    for name,sha in design['source_sha256'].items():assert s.digest(project_file(ROOT, name))==sha,name
    assert s.digest(ROOT/p['checkpoint'])==p['checkpoint_sha256']
    assert p['modes']==MODES and p['seeds']==SEEDS and p['episodes']==15
    assert not set(SEEDS).intersection(p['reserved_final_seeds']+p['reserved_confirmation_seeds'])
    for item in p['noise']:assert s.digest(ROOT/item['path'])==item['sha256']
    assert torch.version.cuda is None
    return p


def frozen_policy(path):
    torch.set_num_threads(1)
    model=PPO.load(path,device='cpu');model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():parameter.requires_grad_(False)
    pilot.finite_policy(model)
    assert model.num_timesteps==131072
    return model


def parameters(model, obs):
    with torch.inference_mode():
        distribution=model.policy.get_distribution(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)).distribution
        return distribution.mean.cpu().numpy()[0].copy(),distribution.stddev.cpu().numpy()[0].copy()


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun noise ablation'
    p=verify_sources();tests=(OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'tests_sha256':s.digest(OUT/'tests-before.log'),'training':False,'device':'cpu'})
    started=time.perf_counter();deadline=started+p['wall_time_cap_s']
    rows,steps,reason,error=[],0,'episodes_completed',None
    env=None
    try:
        model=frozen_policy(ROOT/p['checkpoint'])
        env=w.WalkingResidualEnv('healthy',record_trace=True)
        for mode,scale in MODES.items():
            folder=OUT/mode;folder.mkdir()
            for seed in SEEDS:
                pilot.budget_check(deadline)
                item=next(r for r in p['noise'] if r['pose_seed']==seed)
                epsilon=np.load(ROOT/item['path'],allow_pickle=False)
                obs,_=env.reset(seed=seed)
                initial=s.digest_bytes(env.data.qpos.tobytes()+env.data.qvel.tobytes())
                inputs,means,sigmas,raws,actions,rewards,components=[],[],[],[],[],[],[]
                interrupted,pending=False,None
                try:
                    done=False
                    while not done:
                        pilot.budget_check(deadline)
                        assert steps<p['max_control_steps']
                        mean,sigma=parameters(model,obs)
                        raw,action=sample_action(mean,sigma,epsilon[len(actions)],scale)
                        inputs.append(obs.copy());means.append(mean);sigmas.append(sigma);raws.append(raw);actions.append(action)
                        obs,reward,terminated,truncated,info=env.step(action)
                        steps+=1;rewards.append(reward)
                        components.append([info['reward_terms'].get(k,0.) for k in w.REWARD_WEIGHTS])
                        if not np.isfinite(obs).all() or not np.isfinite(reward) or info.get('episode_result',{}).get('reason')=='nonfinite state':
                            raise FloatingPointError('Nonfinite noise-ablation state or reward')
                        done=terminated or truncated
                except Exception as exc:
                    interrupted,pending=True,exc
                arrays=env.trace()
                arrays.update({'residual_observations':np.asarray(inputs,dtype=np.float32),
                    'policy_means':np.asarray(means,dtype=np.float32),'policy_sigmas':np.asarray(sigmas,dtype=np.float32),
                    'standard_noise':epsilon[:len(actions)].copy(),'raw_residual_actions':np.asarray(raws,dtype=np.float32),
                    'residual_policy_actions':np.asarray(actions,dtype=np.float32),'control_rewards':np.asarray(rewards),
                    'reward_components':np.asarray(components),'reward_component_names':np.asarray(list(w.REWARD_WEIGHTS))})
                stem=f'healthy_{seed}';path=folder/(stem+'.npz')
                with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
                result=env.mission_result()
                if interrupted:result['success']=False
                row={'mode':mode,'sigma_scale':scale,'seed':seed,'condition':'healthy','candidate':'go2_rl_lab',
                    'initial_xy':env.origin.tolist(),'initial_yaw':env.initial_yaw,'initial_state_sha256':initial,
                    'trace_sha256':s.digest(path),'checkpoint_sha256':p['checkpoint_sha256'],
                    'noise_sha256':item['sha256'],'sampling_seed':item['sampling_seed'],
                    'interrupted':interrupted,'episode_return':float(sum(rewards)),
                    'raw_action_clip_fraction':float(np.mean(np.abs(raws)>1)) if raws else None,**result}
                s.write(folder/(stem+'.json'),row);rows.append(row)
                print(f"{mode}/{seed}: success={row['success']} fall={row['fallen']} hold={row.get('final_stable_hold_s')} error={row.get('final_goal_error_m')}",flush=True)
                if pending is not None:raise pending
    except pilot.BudgetExpired as exc:
        reason,error='wall_budget_reached',str(exc)
    except Exception as exc:
        reason,error='aborted_error_or_nonfinite',str(exc)
        s.write(OUT/'error.json',{'message':str(exc),'traceback':traceback.format_exc()})
    finally:
        if env is not None:env.close()
    s.write(OUT/'episodes.json',rows)
    s.write(OUT/'outcome.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'stop_reason':reason,'error':error,
        'compute_wall_s':time.perf_counter()-started,'episodes_recorded':len(rows),'control_steps':steps,
        'training':False,'fault_episodes':0,'final_test_run':False})
    audit()
    if reason=='aborted_error_or_nonfinite':raise RuntimeError(error)


def audit():
    p=verify_sources();design=s.read(OUT/'design.json');rows=s.read(OUT/'episodes.json');outcome=s.read(OUT/'outcome.json')
    assert s.digest(OUT/'tests-before.log')==s.read(OUT/'execution.json')['tests_sha256']
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(s.read(OUT/'execution.json')['started_utc'])
    assert len(rows)<=15 and outcome['control_steps']<=15000
    assert len({(r['mode'],r['seed']) for r in rows})==len(rows)
    model=frozen_policy(ROOT/p['checkpoint'])
    states,checked,parity=0,0,0
    initials={};first_observations={}
    for row in rows:
        assert row['mode'] in MODES and row['seed'] in SEEDS and row['condition']=='healthy'
        path=OUT/row['mode']/f"healthy_{row['seed']}.npz"
        assert row==s.read(path.with_suffix('.json')) and s.digest(path)==row['trace_sha256']
        assert row['checkpoint_sha256']==p['checkpoint_sha256'] and row['sigma_scale']==MODES[row['mode']]
        if row['interrupted']:
            assert not row['success'];continue
        states+=independent.verify_episode(path,row,p);checked+=1
        assert row['fault_onset_tick'] is None
        with np.load(path,allow_pickle=False) as a:
            q,v=a['qpos'],a['qvel'];actions=a['control_residual_actions'];inputs=a['residual_observations']
            ticks=np.arange(0,len(q)-1,10)
            item=next(r for r in p['noise'] if r['pose_seed']==row['seed'])
            assert row['noise_sha256']==item['sha256'] and row['sampling_seed']==item['sampling_seed']
            epsilon=np.load(ROOT/item['path'],allow_pickle=False)
            np.testing.assert_array_equal(a['standard_noise'],epsilon[:len(ticks)])
            for i,obs in enumerate(inputs):
                mean,sigma=parameters(model,obs)
                np.testing.assert_array_equal(a['policy_means'][i],mean)
                np.testing.assert_array_equal(a['policy_sigmas'][i],sigma)
                raw,expected=sample_action(mean,sigma,epsilon[i],MODES[row['mode']])
                np.testing.assert_array_equal(a['raw_residual_actions'][i],raw)
                np.testing.assert_array_equal(a['residual_policy_actions'][i],expected)
                np.testing.assert_array_equal(actions[i],expected)
            assert row['raw_action_clip_fraction']==float(np.mean(np.abs(a['raw_residual_actions'])>1))
            nominal=d.DEFAULT+.25*a['control_actions']
            target=np.clip(nominal+.1*actions,d.RANGES[:,0],d.RANGES[:,1])
            np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:len(q)-1])
            reconstructed=[]
            for i,tick in enumerate(ticks):
                r=rotation(q[tick,3:7]);previous=np.zeros(12) if i==0 else actions[i-1]
                reconstructed.append(np.r_[r.T@v[tick,:3],v[tick,3:6],r.T@np.array([0.,0.,-1.]),a['command'][tick],
                    q[tick,7:]-d.DEFAULT,v[tick,6:]*.05,nominal[i]-d.DEFAULT,previous])
            np.testing.assert_array_equal(inputs,np.asarray(reconstructed,dtype=np.float32))
            audit_reward_and_nominal(a,row)
            if row['mode']=='deterministic':
                reference=pilot.OUT/'validation_131072'/path.name
                assert s.digest(reference)==s.read(reference.with_suffix('.json'))['trace_sha256']
                with np.load(reference,allow_pickle=False) as old:
                    for key in old.files:np.testing.assert_array_equal(a[key],old[key],err_msg=key)
                parity+=1;initials[row['seed']]=row['initial_state_sha256'];first_observations[row['seed']]=inputs[0].copy()
            else:
                assert row['initial_state_sha256']==initials[row['seed']]
                np.testing.assert_array_equal(inputs[0],first_observations[row['seed']])
    complete=checked==len(rows)==15 and outcome['stop_reason']=='episodes_completed'
    summary=[];paired=[]
    if complete:
        assert parity==5
        for mode in MODES:
            group=[r for r in rows if r['mode']==mode];assert len(group)==5
            summary.append({'mode':mode,'episodes':5,'successes':sum(r['success'] for r in group),
                'falls':sum(r['fallen'] for r in group),
                'median_final_goal_error_m':float(np.median([r['final_goal_error_m'] for r in group])),
                'median_final_speed_m_s':float(np.median([r['final_speed_m_s'] for r in group])),
                'median_final_stable_hold_s':float(np.median([r['final_stable_hold_s'] for r in group])),
                'median_raw_action_clip_fraction':float(np.median([r['raw_action_clip_fraction'] for r in group])),
                'failed_checks':{k:sum(not r['checks'][k] for r in group) for k in group[0]['checks']}})
        for seed in SEEDS:
            low=next(r for r in rows if r['mode']=='low_noise' and r['seed']==seed)
            current=next(r for r in rows if r['mode']=='current_noise' and r['seed']==seed)
            paired.append({'seed':seed,'low_success':low['success'],'current_success':current['success'],
                'low_fall':low['fallen'],'current_fall':current['fallen'],
                'hold_delta_s':low['final_stable_hold_s']-current['final_stable_hold_s'],
                'goal_error_delta_m':low['final_goal_error_m']-current['final_goal_error_m']})
    for name,sha in design['prior_artifacts_sha256'].items():assert d.digest(project_file(ROOT, name))==sha,name
    support=False
    if complete:
        low=next(r for r in summary if r['mode']=='low_noise');current=next(r for r in summary if r['mode']=='current_noise')
        support=low['successes']>current['successes'] and low['falls']<=current['falls']
    proof={'audit_passed':True,'study_complete':complete,'complete_episodes_checked':checked,'physics_states_checked':states,
        'deterministic_historical_exact_pairs':parity,'common_noise_arrays_verified':5 if complete else None,
        'protected_prior_artifacts_unchanged':len(design['prior_artifacts_sha256']),
        'lower_noise_supported_for_fixed_healthy_cohort':support,'summary':summary,'paired':paired,
        'rescued_low_vs_current':sum(r['low_success'] and not r['current_success'] for r in paired),
        'harmed_low_vs_current':sum(r['current_success'] and not r['low_success'] for r in paired),
        'training':False,'fault_compensation_test':False,'promotion_test':False,'final_test_run':False}
    path=OUT/'completion_audit.json'
    if path.exists():assert s.read(path)==proof
    else:s.write(path,proof)
    report(proof,outcome)
    return proof


def report(proof,outcome):
    lines=['# Kalibrasi noise aksi walking','',
        f"Studi {'selesai' if proof['study_complete'] else 'tidak lengkap'}: {proof['complete_episodes_checked']}/15 episode sehat diaudit. Compute {outcome['compute_wall_s']:.1f} detik dari 300 detik. Tidak ada training.",'',
        '| Mode | Sukses | Jatuh | Median error akhir (m) | Median hold akhir (s) | Median raw-action clipping |',
        '|---|---:|---:|---:|---:|---:|']
    for r in proof['summary']:
        lines.append(f"| {r['mode']} | {r['successes']}/5 | {r['falls']} | {r['median_final_goal_error_m']:.3f} | {r['median_final_stable_hold_s']:.3f} | {100*r['median_raw_action_clip_fraction']:.1f}% |")
    lines+=['','![Paired action-noise calibration](comparison.png)','',
        f"Kriteria dukungan yang dibekukan untuk noise rendah pada cohort sehat ini: {proof['lower_noise_supported_for_fixed_healthy_cohort']}. Rescue low vs current: {proof['rescued_low_vs_current']}; harm: {proof['harmed_low_vs_current']}. Hasil sama atau tradeoff pada metrik primer tidak membuktikan tidak ada efek.",'',
        'Semua mode memakai checkpoint, fungsi mean policy, state awal dan kriteria misi yang sama. Dua mode stokastik memakai deret epsilon Gaussian standar yang sama per pose dan control step. Mean dievaluasi ulang pada state masing-masing, bukan diputar ulang dari trajectory deterministik. Yang berubah hanya skala std: 0, 0,2 atau 1. Aksi raw dipotong ke [-1,1] sebelum residual +/-0,10 rad, joint clipping dan torque limiting. Distribusi Gaussian sama dengan hukum sampling checkpoint/SB3; RNG NumPy bukan replay bitwise RNG training.', '',
        f"Audit independen memeriksa {proof['physics_states_checked']:,} state, batas misi/torque, observasi, distribusi mean/std, noise bersama, raw/applied action, reward dan nominal policy. Seluruh {proof['deterministic_historical_exact_pairs']}/5 trajectory deterministik cocok persis dengan setiap array arsip. Sebanyak {proof['protected_prior_artifacts_unchanged']} file lama tetap utuh.",'',
        'Jika noise rendah menyelamatkan misi sehat pada checkpoint tetap, itu mendukung pengaruh sampling dalam cohort ini. Itu belum membuktikan noise sebagai seluruh penyebab kegagalan training lama: bobot berubah sepanjang learning, pose training berbeda, dan std bisa berubah jika training baru dilakukan. Tidak ada klaim bahwa perubahan log_std_init sudah berhasil atau optimal.', '',
        'Hanya lima pose pengembangan 23000–23004 dan satu deret noise per pose. Tidak ada fault, training, promosi, final test, replikasi, terrain baru atau hardware. Seed final 23100–23119 dan konfirmasi 23300–23304 tetap disimpan. Kriteria misi 2m/20s, posisi <=0,15m, heading <=15 derajat, cross-track <=0,20m, hold >=2s pada speed <=0,05m/s, serta batas jatuh tetap. Preview Data tidak diaktifkan.', '',
        'Audit ulang: `.venv\\Scripts\\python.exe -B walking_noise_ablation.py audit`. Run menolak overwrite.','']
    text='\n'.join(lines);path=OUT/'REPORT.md'
    if path.exists():assert path.read_text(encoding='utf-8')==text
    else:
        with path.open('x',encoding='utf-8') as handle:handle.write(text)
    print('\n'.join(lines[:9]),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
