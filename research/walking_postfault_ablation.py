"""One bounded inference-only ablation: suppress residual after actual fault onset."""
from project_paths import project_file, source_files
from datetime import datetime, timezone
from pathlib import Path
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
import diagnose_walking_fault as diagnosis


ROOT=s.ROOT
OUT=ROOT/'results/walking_postfault_ablation'
MODES=['always','postfault_off']
SEEDS=list(range(23000,23005))
CONDITIONS=w.CONDITIONS[1:]


def choose_action(action, mode, onset):
    action=np.asarray(action)
    if mode not in MODES or action.shape!=(12,) or not np.isfinite(action).all():
        raise ValueError('Invalid mode or residual action')
    off=mode=='postfault_off' and onset is not None
    return np.zeros_like(action) if off else action.copy(),off


def direction(success_delta, fall_delta):
    if success_delta>0 and fall_delta<=0 or success_delta==0 and fall_delta<0:
        return 'off_better_primary'
    if success_delta<0 and fall_delta>=0 or success_delta==0 and fall_delta>0:
        return 'off_worse_primary'
    if success_delta==0 and fall_delta==0:
        return 'primary_unchanged_inconclusive'
    return 'primary_tradeoff_inconclusive'


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite post-fault ablation'
    parent=pilot.verify_sources()
    assert s.read(pilot.OUT/'completion_audit.json')['audit_passed']
    assert s.read(diagnosis.OUT/'completion_audit.json')['audit_passed']
    protected=dict(s.read(diagnosis.OUT/'design.json')['protected_prior_artifacts_sha256'])
    for name,sha in protected.items():assert diagnosis.digest(project_file(ROOT, name))==sha,name
    for path in diagnosis.OUT.rglob('*'):
        if path.is_file():protected[str(path.relative_to(ROOT))]=diagnosis.digest(path)
    for path in [*source_files(ROOT),*pilot.OUT.rglob('*')]:
        if path.is_file() and path.name not in ['walking_postfault_ablation.py','test_walking_postfault_ablation.py','plot_walking_postfault_ablation.py']:
            protected[str(path.relative_to(ROOT))]=diagnosis.digest(path)
    checkpoint=pilot.OUT/'checkpoint_131072.zip'
    expected=s.read(pilot.OUT/'checkpoint_131072.json')['checkpoint_sha256']
    assert s.digest(checkpoint)==expected
    OUT.mkdir()
    p={'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256':expected,'checkpoint_steps':131072,'checkpoint_status':'Not promoted; fixed diagnostic object only.',
        'modes':MODES,'conditions':CONDITIONS,'seeds':SEEDS,'episodes':40,
        'max_control_steps':40000,'wall_time_cap_s':600,
        'budget_scope':'Model/environment construction, all 40 episodes and serialization. Tests/preparation and final read-only audit are outside this compute budget. Stop at budget/nonfinite, preserve partial evidence; never retry or extend.',
        'intervention':'Both modes use the frozen deterministic residual before fault. postfault_off zeros all 12 residual actions from the first scheduled control decision strictly after actual physics fault onset, latched through episode end. Nominal policy, command and yaw remain active.',
        'gate_information':'Privileged diagnostic fault latch only; not an observation feature, fault detector, deployable controller, or proven repair.',
        'parity':'All 20 always episodes must match every archived checkpoint trace array exactly. Paired off episodes must match physics through the first disabled decision state, commands/torques before it, and observations/proposals through it.',
        'mission_acceptance':parent['mission_acceptance'],'height_reference_m':.338,
        'primary_metrics':['mission successes','falls'],
        'secondary_metrics':['net and maximum forward progress after onset','goal error','weak-motor torque limiting','per-joint residual bounds'],
        'decision':'Per condition, off is better on primary metrics if successes rise without more falls, or successes tie and falls decrease. Reverse means off is worse. Equal primary metrics or conflicting directions remain inconclusive. Progress/torque deltas are descriptive, never substitute for mission success or prove no effect.',
        'causal_scope':'Effect of disabling this fixed residual after fault on the full subsequent closed-loop trajectory. Previous residual observation, nominal history, contacts and commands may change afterwards. No claim of motor identification or optimal control authority.',
        'development_only':True,'reserved_final_seeds':list(range(23100,23120)),
        'reserved_confirmation_seeds':list(range(23300,23305)),
        'training':False,'promotion_test':False,'preview':'No preview server or task activation.'}
    s.write(OUT/'protocol.json',p)
    s.write(OUT/'design.json',{'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in ['walking_postfault_ablation.py','test_walking_postfault_ablation.py','plot_walking_postfault_ablation.py']},
        'prior_artifacts_sha256':protected})
    print(f'Frozen 40-episode post-fault ablation, 600s; {len(protected)} old files protected',flush=True)


def verify_sources():
    pilot.verify_sources()
    p,d=s.read(OUT/'protocol.json'),s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==d['protocol_sha256']
    for name,sha in d['source_sha256'].items():assert s.digest(project_file(ROOT, name))==sha,name
    assert s.digest(ROOT/p['checkpoint'])==p['checkpoint_sha256']
    assert p['seeds']==SEEDS and p['conditions']==CONDITIONS and p['modes']==MODES
    assert not set(SEEDS).intersection(p['reserved_final_seeds']+p['reserved_confirmation_seeds'])
    assert torch.version.cuda is None
    return p


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun ablation'
    p=verify_sources()
    tests=(OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'tests_sha256':s.digest(OUT/'tests-before.log'),'training':False,'device':'cpu'})
    started=time.perf_counter();deadline=started+p['wall_time_cap_s']
    rows,steps,reason,error=[],0,'episodes_completed',None
    env=None
    try:
        torch.set_num_threads(1)
        model=PPO.load(ROOT/p['checkpoint'],device='cpu')
        model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():parameter.requires_grad_(False)
        pilot.finite_policy(model)
        assert model.num_timesteps==131072
        env=w.WalkingResidualEnv('healthy',record_trace=True)
        for mode in MODES:
            folder=OUT/mode;folder.mkdir()
            for condition in CONDITIONS:
                for seed in SEEDS:
                    pilot.budget_check(deadline)
                    obs,_=env.reset(seed=seed,options={'condition':condition})
                    initial=s.digest_bytes(env.data.qpos.tobytes()+env.data.qvel.tobytes())
                    inputs,proposals,chosen,masks,ticks,rewards,components=[],[],[],[],[],[],[]
                    interrupted,pending=False,None
                    try:
                        done=False
                        while not done:
                            pilot.budget_check(deadline)
                            assert steps<p['max_control_steps']
                            action=model.predict(obs,deterministic=True)[0]
                            selected,off=choose_action(action,mode,env.onset)
                            inputs.append(obs.copy());proposals.append(action.copy());chosen.append(selected.copy())
                            masks.append(off);ticks.append(env.tick)
                            obs,reward,terminated,truncated,info=env.step(selected)
                            steps+=1
                            rewards.append(reward);components.append([info['reward_terms'].get(k,0.) for k in w.REWARD_WEIGHTS])
                            if not np.isfinite(obs).all() or not np.isfinite(reward) or info.get('episode_result',{}).get('reason')=='nonfinite state':
                                raise FloatingPointError('Nonfinite ablation state or reward')
                            done=terminated or truncated
                    except Exception as exc:
                        interrupted,pending=True,exc
                    arrays=env.trace()
                    arrays.update({'residual_observations':np.asarray(inputs,dtype=np.float32),
                        'residual_policy_actions':np.asarray(proposals,dtype=np.float32),
                        'applied_residual_actions':np.asarray(chosen,dtype=np.float32),
                        'residual_disabled':np.asarray(masks,dtype=bool),'control_ticks':np.asarray(ticks,dtype=np.int64),
                        'control_rewards':np.asarray(rewards),'reward_components':np.asarray(components),
                        'reward_component_names':np.asarray(list(w.REWARD_WEIGHTS))})
                    stem=f'{condition}_{seed}';path=folder/(stem+'.npz')
                    with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
                    result=env.mission_result()
                    if interrupted:result['success']=False
                    row={'mode':mode,'condition':condition,'seed':seed,'candidate':'go2_rl_lab',
                        'initial_xy':env.origin.tolist(),'initial_yaw':env.initial_yaw,
                        'initial_state_sha256':initial,'trace_sha256':s.digest(path),
                        'checkpoint_sha256':p['checkpoint_sha256'],'interrupted':interrupted,
                        'episode_return':float(sum(rewards)),'disabled_controls':int(sum(masks)),
                        'first_disabled_tick':next((t for t,m in zip(ticks,masks) if m),None),**result}
                    s.write(folder/(stem+'.json'),row);rows.append(row)
                    print(f"{mode}/{stem}: success={row['success']} fall={row['fallen']} progress={row.get('final_progress_m')} gate={row['first_disabled_tick']}",flush=True)
                    if pending is not None:raise pending
    except pilot.BudgetExpired as exc:
        reason,error='wall_budget_reached',str(exc)
    except Exception as exc:
        reason,error='aborted_error_or_nonfinite',str(exc)
        s.write(OUT/'error.json',{'message':str(exc),'traceback':traceback.format_exc()})
    finally:
        if env is not None:env.close()
    s.write(OUT/'episodes.json',rows)
    s.write(OUT/'outcome.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),
        'stop_reason':reason,'error':error,'compute_wall_s':time.perf_counter()-started,
        'episodes_recorded':len(rows),'control_steps':steps,'training':False,'final_test_run':False})
    audit()
    if reason=='aborted_error_or_nonfinite':raise RuntimeError(error)


def audit():
    p=verify_sources();design=s.read(OUT/'design.json');rows=s.read(OUT/'episodes.json');outcome=s.read(OUT/'outcome.json')
    assert s.digest(OUT/'tests-before.log')==s.read(OUT/'execution.json')['tests_sha256']
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(s.read(OUT/'execution.json')['started_utc'])
    assert len(rows)<=40 and outcome['control_steps']<=40000
    assert len({(r['mode'],r['condition'],r['seed']) for r in rows})==len(rows)
    model=PPO.load(ROOT/p['checkpoint'],device='cpu');pilot.finite_policy(model)
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():parameter.requires_grad_(False)
    states,checked,reference_pairs,prefix_pairs=0,0,0,0
    diagnostics=[]
    for row in rows:
        assert row['mode'] in MODES and row['condition'] in CONDITIONS and row['seed'] in SEEDS
        path=OUT/row['mode']/f"{row['condition']}_{row['seed']}.npz"
        assert row==s.read(path.with_suffix('.json')) and s.digest(path)==row['trace_sha256']
        if row['interrupted']:
            assert not row['success']
            continue
        states+=independent.verify_episode(path,row,p);checked+=1
        with np.load(path,allow_pickle=False) as a:
            actions=a['control_residual_actions'];inputs=a['residual_observations'];proposals=a['residual_policy_actions']
            ticks=np.arange(0,len(a['qpos'])-1,10);onset=row['fault_onset_tick']
            np.testing.assert_array_equal(a['control_ticks'],ticks)
            mask=(ticks>onset) if row['mode']=='postfault_off' and onset is not None else np.zeros(len(ticks),dtype=bool)
            np.testing.assert_array_equal(a['residual_disabled'],mask)
            assert row['disabled_controls']==int(mask.sum())
            assert row['first_disabled_tick']==(int(ticks[mask][0]) if mask.any() else None)
            expected=proposals.copy();expected[mask]=0
            np.testing.assert_array_equal(actions,np.clip(expected,-1,1))
            np.testing.assert_array_equal(a['applied_residual_actions'],expected)
            np.testing.assert_allclose(proposals,np.asarray([model.predict(obs,deterministic=True)[0] for obs in inputs]),rtol=0,atol=1e-7)
            nominal=diagnosis.DEFAULT+.25*a['control_actions']
            targets=np.clip(nominal+.1*actions,diagnosis.RANGES[:,0],diagnosis.RANGES[:,1])
            np.testing.assert_array_equal(a['target'],np.repeat(targets,10,axis=0)[:len(a['qpos'])-1])
            reconstructed=[]
            for i,tick in enumerate(ticks):
                q,v=a['qpos'][tick],a['qvel'][tick];r=rotation(q[3:7])
                previous=np.zeros(12) if i==0 else actions[i-1]
                reconstructed.append(np.r_[r.T@v[:3],v[3:6],r.T@np.array([0.,0.,-1.]),a['command'][tick],
                    q[7:]-diagnosis.DEFAULT,v[6:]*.05,nominal[i]-diagnosis.DEFAULT,previous])
            np.testing.assert_array_equal(inputs,np.asarray(reconstructed,dtype=np.float32))
            audit_reward_and_nominal(a,row)
            metrics=diagnosis.diagnose(a,row)
            diagnostics.append({'mode':row['mode'],'condition':row['condition'],'seed':row['seed'],**metrics})
            if row['mode']=='always':
                reference=pilot.OUT/'validation_131072'/path.name
                metadata=s.read(reference.with_suffix('.json'))
                assert s.digest(reference)==metadata['trace_sha256']
                with np.load(reference,allow_pickle=False) as original:
                    for key in original.files:np.testing.assert_array_equal(a[key],original[key],err_msg=key)
                assert row['checks']==metadata['checks'] and row['success']==metadata['success'] and row['fallen']==metadata['fallen']
                reference_pairs+=1
            else:
                before=OUT/'always'/path.name
                with np.load(before,allow_pickle=False) as b:
                    assert row['fault_onset_tick']==s.read(before.with_suffix('.json'))['fault_onset_tick']
                    assert mask.any(), 'Intervention never received'
                    first=int(ticks[mask][0])
                    for key in ['qpos','qvel']:np.testing.assert_array_equal(a[key][:first+1],b[key][:first+1])
                    for key in ['requested_torque','limited_torque','applied_torque','actuator_force','command','target','kp','kd','contacts','strength']:
                        np.testing.assert_array_equal(a[key][:first],b[key][:first],err_msg=key)
                    control=first//10
                    for key in ['control_observations','control_raw_actions','control_actions','residual_observations','residual_policy_actions']:
                        np.testing.assert_array_equal(a[key][:control+1],b[key][:control+1],err_msg=key)
                    np.testing.assert_array_equal(actions[:control],b['control_residual_actions'][:control])
                prefix_pairs+=1
    complete=len(rows)==checked==40 and outcome['stop_reason']=='episodes_completed'
    summaries=[];pairs=[]
    if complete:
        for condition in CONDITIONS:
            groups={mode:[r for r in diagnostics if r['mode']==mode and r['condition']==condition] for mode in MODES}
            counts={mode:{'successes':sum(r['success'] for r in group),'falls':sum(r['fallen'] for r in group),
                'median_post_net_progress_m':diagnosis.median([r['post_net_progress_m'] for r in group]),
                'median_post_max_progress_m':diagnosis.median([r['post_max_progress_gain_m'] for r in group]),
                'median_weak_torque_limit_fraction':diagnosis.median([r['post_weak_torque_limit_fraction'] for r in group])}
                for mode,group in groups.items()}
            for seed in SEEDS:
                old=next(r for r in groups['always'] if r['seed']==seed);new=next(r for r in groups['postfault_off'] if r['seed']==seed)
                pairs.append({'condition':condition,'seed':seed,'always_success':old['success'],'off_success':new['success'],
                    'always_fall':old['fallen'],'off_fall':new['fallen'],
                    'post_net_progress_delta_m':new['post_net_progress_m']-old['post_net_progress_m'],
                    'post_max_progress_delta_m':new['post_max_progress_gain_m']-old['post_max_progress_gain_m']})
            success_delta=counts['postfault_off']['successes']-counts['always']['successes']
            fall_delta=counts['postfault_off']['falls']-counts['always']['falls']
            summaries.append({'condition':condition,'episodes_per_mode':5,'counts':counts,
                'primary_direction':direction(success_delta,fall_delta),
                'median_paired_post_net_progress_delta_m':diagnosis.median([r['post_net_progress_delta_m'] for r in pairs if r['condition']==condition])})
    for name,sha in design['prior_artifacts_sha256'].items():assert diagnosis.digest(project_file(ROOT, name))==sha,name
    proof={'audit_passed':True,'study_complete':complete,'complete_episodes_checked':checked,'physics_states_checked':states,
        'historical_always_exact_pairs':reference_pairs,'prefault_and_preswitch_exact_pairs':prefix_pairs,
        'protected_prior_artifacts_unchanged':len(design['prior_artifacts_sha256']),
        'training':False,'final_test_run':False,'promotion_test':False,'summary':summaries,'paired':pairs}
    if complete:assert reference_pairs==prefix_pairs==20
    for name,value in [('diagnostics.json',diagnostics),('completion_audit.json',proof)]:
        path=OUT/name
        if path.exists():assert s.read(path)==value
        else:s.write(path,value)
    report(proof,outcome)
    return proof


def report(proof,outcome):
    lines=['# Ablation residual setelah onset fault','',
        f"Studi {'selesai' if proof['study_complete'] else 'tidak lengkap'}; {proof['complete_episodes_checked']}/40 episode lengkap diaudit. Compute {outcome['compute_wall_s']:.1f} detik dari budget 600 detik. Tidak ada training.",'',
        '| Fault | Sukses biasa/off | Jatuh biasa/off | Median delta kemajuan neto berpasangan off-biasa (m) | Arah metrik primer |',
        '|---|---:|---:|---:|---|']
    for group in proof['summary']:
        a,b=group['counts']['always'],group['counts']['postfault_off']
        lines.append(f"| {group['condition']} | {a['successes']}/5 / {b['successes']}/5 | {a['falls']} / {b['falls']} | {group['median_paired_post_net_progress_delta_m']:+.3f} | {group['primary_direction']} |")
    lines+=['','![Post-fault residual ablation](comparison.png)','',
        'off_better_primary berarti penghapusan residual memperbaiki metrik primer pada cohort ini; off_worse_primary berarti kontribusi residual lebih baik pada metrik primer. Bila keduanya sama, kesimpulan primer tetap inconclusive, bukan bukti tidak ada efek. Kemajuan dan torque limiting hanya ukuran sekunder. Tidak ada perubahan kriteria misi atau promosi checkpoint.', '',
        'Kedua mode memakai residual terlatih yang sama sebelum fault. Mode off meniadakan seluruh residual mulai keputusan kontrol terjadwal pertama setelah fault benar-benar dilatch. Fault tetap first pre-physics-step progress >=0,50 m; motor calf terpilih menerima 10% limited torque. Kontrol nominal, yaw, joint/torque limits, model, dan reward tinggi 0,338 m tetap. Gate memakai latch eksperimen yang tidak tersedia dalam observasi policy; bukan controller deployable.', '',
        f"Audit mencakup {proof['physics_states_checked']:,} state fisika. Seluruh {proof['historical_always_exact_pairs']} episode biasa cocok persis dengan setiap array trace historis. Seluruh {proof['prefault_and_preswitch_exact_pairs']} pasangan off cocok sampai state keputusan pertama yang berbeda, termasuk observasi/proposal pada keputusan itu. Sesudahnya previous residual, history nominal, command, kontak dan trajectory dapat berubah sebagai bagian dari intervensi closed loop.",'',
        f"Sebanyak {proof['protected_prior_artifacts_unchanged']} file sebelumnya tetap utuh. Pose 23000–23004 adalah data pengembangan lama, bukan held-out. Hanya satu checkpoint dari satu training seed. Belum menguji kondisi sehat baru, generalisasi, final test, atau hardware. Seed final 23100–23119 dan konfirmasi 23300–23304 tetap disimpan. Preview Data tidak diaktifkan.",'',
        'Audit ulang: `.venv\\Scripts\\python.exe -B walking_postfault_ablation.py audit`. Run menolak overwrite.','']
    path=OUT/'REPORT.md';text='\n'.join(lines)
    if path.exists():assert path.read_text(encoding='utf-8')==text
    else:
        with path.open('x',encoding='utf-8') as handle:handle.write(text)
    print('\n'.join(lines[:9]),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
