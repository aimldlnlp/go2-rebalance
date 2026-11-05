"""Bounded nominal RR severity diagnosis. No residual policy or training."""
from project_paths import project_file, source_files
from datetime import datetime, timezone
from pathlib import Path
import inspect
import math
import textwrap
import time
import traceback
import numpy as np
import mujoco
import torch
import sudden as s
import walking_baseline as baseline
import walking_residual_env as original
import walking_residual_height_env as w
import walking_residual_low_noise_pilot as pilot
import diagnose_walking_fault as d
from review_walking_residual_height_pilot import audit_reward_and_nominal
from walking_baseline import CONTROL, TICKS, DT, rotation, torque_limit
from walking_residual_env import RESIDUAL_SCALE, REWARD_WEIGHTS

ROOT=s.ROOT
OUT=ROOT/'results/walking_rr_severity'
STRENGTHS=[.75,.5,.25]
SEEDS=list(range(23000,23005))
ARCHIVE=pilot.OUT/'baseline'


class SeverityEnv(w.WalkingResidualEnv):
    def __init__(self, strength_remaining, condition='RR_calf', record_trace=True):
        if strength_remaining not in [*STRENGTHS,.1]:
            raise ValueError('Unsupported frozen RR strength')
        self.strength_remaining=float(strength_remaining)
        super().__init__(condition=condition,record_trace=record_trace)

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset before stepping a finished environment')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Residual action must contain 12 finite values')
        action = np.clip(action, -1., 1.)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self._update_metrics()
            self._done = True
            info = self._info()
            info['reward_terms'] = {'fall': -10.}
            return self._observation(), -10., True, False, info
        error_before = self.metrics['goal_error']
        previous = self.previous.copy()
        target = np.clip(self.nominal_target+RESIDUAL_SCALE*action, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        for key, value in [('control_observations', self.nominal_observation),
                ('control_raw_actions', self.raw_action), ('control_actions', self.nominal_action),
                ('control_residual_actions', action)]:
            self._append(key, value)
        for _ in range(CONTROL):
            if self.weak is not None and self.onset is None and (self.data.qpos[:2]-self.origin)@self.forward >= .5:
                self.onset = self.tick
                self.fault_before_goal = not self.ever_goal_entered
            requested = self.kp*(target-self.data.qpos[7:])-self.kd*self.data.qvel[6:]
            limited = torque_limit(requested, self.data.qvel[6:], 'go2_rl_lab', self.model.actuator_ctrlrange)
            applied = limited.copy()
            if self.onset is not None:
                applied[self.weak] *= self.strength_remaining
            for key, value in [('requested_torque', requested), ('limited_torque', limited),
                    ('applied_torque', applied), ('strength', self.strength_remaining if self.onset is not None else 1.),
                    ('command', self.command), ('target', target), ('kp', self.kp), ('kd', self.kd)]:
                self._append(key, value)
            self.data.ctrl[:] = applied
            mujoco.mj_step(self.model, self.data)
            self.tick += 1
            self._append('actuator_force', self.data.actuator_force)
            self._append('qpos', self.data.qpos)
            self._append('qvel', self.data.qvel)
            if self.record_trace:
                contacts = np.zeros(4, dtype=bool)
                for contact in self.data.contact:
                    for i, geom in enumerate(self.feet):
                        if geom in (contact.geom1, contact.geom2):
                            contacts[i] = True
                self._append('contacts', contacts)
            self._update_metrics()
            if self.fallen or self.tick >= TICKS:
                break
        self.previous = action.copy()
        terminated, truncated = bool(self.fallen), bool(self.tick >= TICKS)
        self._done = terminated or truncated
        if self.metrics['finite']:
            r = rotation(self.data.qpos[3:7])
            body_velocity = r.T@self.data.qvel[:3]
            values = {'velocity_tracking': float(np.exp(-np.sum((body_velocity[:2]-self.command[:2])**2)/.15**2)),
                'yaw_tracking': float(np.exp(-((self.data.qvel[5]-self.command[2])/.5)**2)),
                'goal_progress': error_before-self.metrics['goal_error'],
                'height_error': (self.metrics['height']-.338)**2,
                'tilt_squared': np.deg2rad(self.metrics['tilt_deg'])**2,
                'cross_track_excess': max(0., self.metrics['cross_track']-.20)**2,
                'stable_goal': float(self.metrics['stable']), 'residual_effort': float(np.mean(action**2)),
                'residual_change': float(np.mean((action-previous)**2)),
                'fall': float(terminated), 'mission_success': float(self._done and self.mission_result()['success'])}
            terms = {key: float(REWARD_WEIGHTS[key]*value) for key, value in values.items()}
        else:
            terms = {'fall': -10.}
        reward = float(sum(terms.values()))
        if not np.isfinite(reward):
            raise RuntimeError('Nonfinite reward')
        if not self._done:
            self._prepare_control()
        info = self._info()
        info['reward_terms'] = terms
        return self._observation(), reward, terminated, truncated, info


def verify_episode(path,row,protocol):
    assert baseline.s.digest(path)==row['trace_sha256']
    with np.load(path,allow_pickle=False) as a:
        q,v=a['qpos'],a['qvel'];n=len(q)-1
        assert v.shape==(n+1,18) and q.shape==(n+1,19) and n<=10000
        assert baseline.s.digest_bytes(q[0].tobytes()+v[0].tobytes())==row['initial_state_sha256']
        for key in ['requested_torque','limited_torque','applied_torque','actuator_force','target','kp','kd']:
            assert a[key].shape==(n,12) and np.isfinite(a[key]).all(),key
        assert a['command'].shape==(n,3) and np.isfinite(a['command']).all()
        assert np.all(np.linalg.norm(a['command'][:,:2],axis=1)<=.2+1e-12)
        assert np.all(np.abs(a['command'][:,2])<=.5)
        assert np.all(a['command'][:min(n,1000)]==0)
        assert a['control_observations'].shape[0]==(n+9)//10
        assert np.isfinite(a['control_observations']).all()
        assert np.isfinite(a['control_raw_actions']).all()
        np.testing.assert_array_equal(a['control_actions'],np.clip(a['control_raw_actions'],-100,100))
        expected_obs_dim=45 if row['candidate']=='go2_rl_lab' else 49
        assert a['control_observations'].shape[1]==expected_obs_dim
        expected_torque=a['kp']*(a['target']-q[:-1,7:])-a['kd']*v[:-1,6:]
        np.testing.assert_allclose(a['requested_torque'],expected_torque,rtol=0,atol=1e-12)
        limit=np.array([23.7,23.7,45.43]*4)
        assert np.all(np.abs(a['limited_torque'])<=limit+1e-12)
        joint_ranges=np.array([[-1.0472,1.0472],[-1.5708,3.4907],[-2.7227,-.83776]]*2 + [[-1.0472,1.0472],[-.5236,4.5379],[-2.7227,-.83776]]*2)
        assert np.all(a['target']>=joint_ranges[:,0]-1e-12) and np.all(a['target']<=joint_ranges[:,1]+1e-12)
        if row['candidate']=='go2_rl_lab':
            velocity=v[:-1,6:]
            capacity=np.where(velocity*a['requested_torque']>0,20.2,23.4)
            high=np.abs(velocity)>13.5
            capacity[high]*=np.maximum(0,(30-np.abs(velocity[high]))/16.5)
            expected=np.clip(a['requested_torque'],-capacity,capacity)
            expected=np.clip(expected,-limit,limit)
            np.testing.assert_allclose(a['limited_torque'],expected,rtol=0,atol=1e-12)
        forward=np.array([math.cos(row['initial_yaw']),math.sin(row['initial_yaw'])])
        lateral=np.array([-forward[1],forward[0]])
        origin=np.array(row['initial_xy'])
        progress=(q[:,:2]-origin)@forward
        onset=None
        if row['condition']!='healthy':
            reached=np.flatnonzero(progress[:-1]>=.5)
            onset=int(reached[0]) if len(reached) else None
        assert row['fault_onset_tick']==onset
        applied=a['limited_torque'].copy()
        strength=np.ones(n)
        if onset is not None:
            weak=baseline.JOINTS.index(row['condition'])
            applied[onset:,weak]*=row['strength_remaining']
            strength[onset:]=row['strength_remaining']
        np.testing.assert_array_equal(a['strength'],strength)
        np.testing.assert_array_equal(a['applied_torque'],applied)
        np.testing.assert_allclose(a['actuator_force'],applied,rtol=0,atol=1e-12)
        finite=np.isfinite(q).all() and np.isfinite(v).all()
        if not finite:
            assert not row['success'] and row['fallen'] and row['reason']=='nonfinite state'
            return n+1
        # Quaternion formulas and trailing-run reduction are independent of the runner's rotation helper/loop.
        qw,qx,qy,qz=q[:,3],q[:,4],q[:,5],q[:,6]
        np.testing.assert_allclose(qw*qw+qx*qx+qy*qy+qz*qz,1,atol=1e-10,rtol=0)
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(qx*qx+qy*qy),-1,1)))
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        heading=np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
        error=np.hypot(q[:,0]-(origin+2*forward)[0],q[:,1]-(origin+2*forward)[1])
        cross=np.abs((q[:,:2]-origin)@lateral)
        speed=np.hypot(v[:,0],v[:,1])
        stable=(error<=.15)&(heading<=math.radians(15))&(speed<=.05)
        bad=np.flatnonzero(~stable)
        start=int(bad[-1]+1) if len(bad) else 0
        hold=max(0.,(len(q)-1-start)*.002) if stable[-1] else 0.
        fallen=bool(np.any(q[:,2]<.16) or np.any(tilt>45))
        checks={'full_duration':n==10000,'no_fall':not fallen,'goal_error':bool(error[-1]<=.15),
            'heading_error':bool(heading[-1]<=math.radians(15)),'cross_track':bool(cross.max()<=.20),
            'stable_stop':hold>=2,'fault_received':row['condition']=='healthy' or onset is not None}
        if row['condition']!='healthy' and onset is not None:checks['fault_before_goal']=bool(np.all(error[:onset+1]>.15))
        assert row['checks']==checks and row['success']==all(checks.values()) and row['fallen']==fallen
        independent={'final_goal_error_m':float(error[-1]),'max_cross_track_m':float(cross.max()),
            'final_heading_error_deg':float(math.degrees(heading[-1])),'max_tilt_deg':float(tilt.max()),
            'final_speed_m_s':float(speed[-1]),'final_stable_hold_s':hold,'final_progress_m':float(progress[-1])}
        for field,value in independent.items():np.testing.assert_allclose(row[field],value,rtol=0,atol=1e-9)
        return n+1


def source_contract():
    expected=inspect.getsource(original.WalkingResidualEnv.step)
    expected=expected.replace('applied[self.weak] *= .1','applied[self.weak] *= self.strength_remaining')
    expected=expected.replace('.1 if self.onset is not None else 1.','self.strength_remaining if self.onset is not None else 1.')
    expected=expected.replace("(self.metrics['height']-.27)**2","(self.metrics['height']-.338)**2")
    assert textwrap.dedent(inspect.getsource(SeverityEnv.step))==textwrap.dedent(expected)


def read_arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def check_prefix(a,b,onset):
    for key in ['qpos','qvel']:
        np.testing.assert_array_equal(a[key][:onset+1],b[key][:onset+1],err_msg=key)
    for key in ['requested_torque','limited_torque','applied_torque','actuator_force','strength',
                'command','target','kp','kd','contacts']:
        np.testing.assert_array_equal(a[key][:onset],b[key][:onset],err_msg=key)
    controls=int(np.sum(np.arange(len(a['control_actions']))*10<=onset))
    for key in ['control_observations','control_raw_actions','control_actions','control_residual_actions','residual_observations']:
        np.testing.assert_array_equal(a[key][:controls],b[key][:controls],err_msg=key)


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite RR severity study'
    source_contract(); parent=pilot.verify_sources()
    assert s.read(ROOT/'results/walking_gain_ablation/completion_audit.json')['study_complete']
    protected=dict(s.read(ROOT/'results/walking_gain_ablation/design.json')['prior_artifacts_sha256'])
    for folder in [ROOT/'results/walking_gain_ablation',ROOT/'results/walking_gain_next_step_review',pilot.OUT]:
        for path in folder.rglob('*'):
            if path.is_file():protected[str(path.relative_to(ROOT))]=d.digest(path)
    for path in source_files(ROOT):protected[str(path.relative_to(ROOT))]=d.digest(path)
    for name,sha in protected.items():assert d.digest(project_file(ROOT, name))==sha,name
    nominal=baseline.Actor('go2_rl_lab').path
    order=[dict(strength_remaining=strength,seed=seed) for strength in STRENGTHS for seed in SEEDS]
    OUT.mkdir()
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),question='Which lighter RR torque multipliers permit this nominal walker to finish the unchanged mission?',
        strengths_remaining=STRENGTHS,seeds=SEEDS,new_episodes=15,run_order=order,wall_time_cap_s=180,max_control_steps=15000,
        budget_scope='Construction, 15 new episodes, online prefix checks and serialization; preparation/tests, archived-control audit and final read-only audit/report excluded. No extension or retries.',
        nominal_policy=str(nominal.relative_to(ROOT)),nominal_policy_sha256=s.digest(nominal),
        physics_model=str(baseline.MODEL.relative_to(ROOT)),physics_model_sha256=s.digest(baseline.MODEL),
        condition='RR_calf',intervention='Only selected calf multiplier changes to 0.75, 0.50 or 0.25 after the same torque limiting and actual progress>=0.50m latch. All residual actions remain zero.',
        mission_acceptance=parent['mission_acceptance'],goal_fault_strength=.1,diagnostic_only=True,
        controls='Reuse five archived zero-residual healthy and five RR10 episodes, audit all physics/mission/nominal/reward arrays and hashes. No control rollouts. New episodes must match both controls through actual onset.',
        implementation='Explicit step copy; source equality permits only fault multiplier parameter and the existing 0.338m height reward reference. Synthetic unit fixtures compare 0.10 against the frozen parent.',
        primary_metrics=['mission success','falls'],secondary_metrics=['final goal error','stable hold','post-onset net progress','torque limiting'],
        interpretation='Report all five poses per level. Lighter severity success supports curriculum feasibility only; it does not prove training benefit, an optimal curriculum, four-motor generalization or success at10%. If no mild level completes, no feasible nominal starting level is established by this study.',
        training=False,residual_policy_loaded=False,confirmation_seeds_used=False,final_seeds_used=False,preview=False)
    s.write(OUT/'protocol.json',p)
    s.write(OUT/'design.json',dict(protocol_sha256=s.digest(OUT/'protocol.json'),
        source_sha256={name:s.digest(project_file(ROOT, name)) for name in ['walking_rr_severity.py','test_walking_rr_severity.py','report_walking_rr_severity.py']},
        prior_artifacts_sha256=protected))
    controls=[];states=0
    for condition in ['healthy','RR_calf']:
        for seed in SEEDS:
            path=ARCHIVE/f'{condition}_{seed}.npz';row=s.read(path.with_suffix('.json'))
            row.update(strength_remaining=1. if condition=='healthy' else .1,historical_control=True,trace=str(path.relative_to(ROOT)))
            states+=verify_episode(path,row,p)
            a=read_arrays(path)
            assert not np.any(a['control_residual_actions']) and not np.any(a['residual_observations'][:,48:60])
            audit_reward_and_nominal(a,row)
            controls.append(row)
    s.write(OUT/'historical_controls.json',controls)
    s.write(OUT/'control_audit.json',dict(audit_passed=True,episodes=10,physics_states_checked=states,
        residual_zero=True,nominal_and_reward_replayed=True,no_control_rollouts=True,step_source_contract=True))
    print(f'Frozen 15 new episodes / 180s; 10 archived controls audited; {len(protected)} protected files',flush=True)


def verify_sources():
    source_contract();p=s.read(OUT/'protocol.json');design=s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==design['protocol_sha256']
    for name,sha in design['source_sha256'].items():assert s.digest(project_file(ROOT, name))==sha,name
    for name,sha in design['prior_artifacts_sha256'].items():assert d.digest(project_file(ROOT, name))==sha,name
    for key in ['nominal_policy','physics_model']:assert s.digest(ROOT/p[key])==p[key+'_sha256']
    assert p['strengths_remaining']==STRENGTHS and p['seeds']==SEEDS
    assert p['run_order']==[dict(strength_remaining=g,seed=seed) for g in STRENGTHS for seed in SEEDS]
    assert torch.version.cuda is None
    return p


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun RR severity study'
    p=verify_sources();assert s.read(OUT/'control_audit.json')['audit_passed']
    tests=(OUT/'tests-before.log').read_text(encoding='utf-8-sig');assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        tests_sha256=s.digest(OUT/'tests-before.log'),historical_controls_sha256=s.digest(OUT/'historical_controls.json'),
        control_audit_sha256=s.digest(OUT/'control_audit.json'),training=False,device='cpu',residual_policy_loaded=False))
    started=time.perf_counter();deadline=started+p['wall_time_cap_s']
    rows=[];steps=0;reason='episodes_completed';error=None;env=None
    try:
        for case in p['run_order']:
            pilot.budget_check(deadline)
            env=SeverityEnv(case['strength_remaining'])
            obs,_=env.reset(seed=case['seed'])
            reference=read_arrays(ARCHIVE/f"RR_calf_{case['seed']}.npz")
            healthy=read_arrays(ARCHIVE/f"healthy_{case['seed']}.npz")
            inputs=[];rewards=[];components=[];interrupted=False;pending=None;prefix_checked=False
            try:
                done=False
                while not done:
                    pilot.budget_check(deadline);assert steps<p['max_control_steps']
                    inputs.append(obs.copy())
                    obs,reward,terminated,truncated,info=env.step(np.zeros(12,dtype=np.float32));steps+=1
                    rewards.append(reward);components.append([info['reward_terms'].get(k,0.) for k in w.REWARD_WEIGHTS])
                    if not np.isfinite(obs).all() or not np.isfinite(reward) or not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                        raise FloatingPointError('Nonfinite severity state or reward')
                    if env.onset is not None and not prefix_checked:
                        prefix=env.trace();prefix['residual_observations']=np.asarray(inputs,dtype=np.float32)
                        check_prefix(prefix,reference,env.onset);check_prefix(prefix,healthy,env.onset)
                        prefix_checked=True
                    done=terminated or truncated
            except Exception as exc:interrupted=True;pending=exc
            a=env.trace();a.update(residual_observations=np.asarray(inputs,dtype=np.float32),
                residual_policy_actions=np.zeros((len(inputs),12),dtype=np.float32),control_rewards=np.asarray(rewards),
                reward_components=np.asarray(components),reward_component_names=np.asarray(list(w.REWARD_WEIGHTS)))
            folder=OUT/f"rr_{round(100*case['strength_remaining'])}";folder.mkdir(exist_ok=True)
            path=folder/f"RR_calf_{case['seed']}.npz"
            with path.open('xb') as handle:np.savez_compressed(handle,**a)
            result=env.mission_result()
            if interrupted:result['success']=False
            row=dict(**case,condition='RR_calf',candidate='go2_rl_lab',initial_xy=env.origin.tolist(),initial_yaw=env.initial_yaw,
                initial_state_sha256=s.digest_bytes(a['qpos'][0].tobytes()+a['qvel'][0].tobytes()),
                trace_sha256=s.digest(path),trace=str(path.relative_to(ROOT)),interrupted=interrupted,historical_control=False,
                episode_return=float(sum(rewards)) if np.isfinite(rewards).all() else None,**result)
            s.write(path.with_suffix('.json'),row);rows.append(row);env.close();env=None
            print(f"{len(rows)}/15 RR {case['strength_remaining']:.0%}/{case['seed']}: success={row['success']} fall={row['fallen']} goal_error={row.get('final_goal_error_m')}",flush=True)
            if pending is not None:raise pending
            pilot.budget_check(deadline)
    except pilot.BudgetExpired as exc:reason='wall_budget_reached';error=str(exc)
    except Exception as exc:
        reason='aborted_error_or_nonfinite';error=str(exc)
        s.write(OUT/'error.json',dict(message=error,traceback=traceback.format_exc()))
    finally:
        if env is not None:env.close()
    s.write(OUT/'episodes.json',rows)
    s.write(OUT/'outcome.json',dict(finished_utc=datetime.now(timezone.utc).isoformat(),stop_reason=reason,error=error,
        compute_wall_s=time.perf_counter()-started,episodes_recorded=len(rows),control_steps=steps,training=False))
    audit()
    if error:raise RuntimeError(error)


def audit():
    p=verify_sources();rows=s.read(OUT/'episodes.json');outcome=s.read(OUT/'outcome.json');execution=s.read(OUT/'execution.json')
    assert s.digest(OUT/'tests-before.log')==execution['tests_sha256']
    assert s.digest(OUT/'historical_controls.json')==execution['historical_controls_sha256']
    assert s.digest(OUT/'control_audit.json')==execution['control_audit_sha256']
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(execution['started_utc'])
    assert [dict(strength_remaining=r['strength_remaining'],seed=r['seed']) for r in rows]==p['run_order'][:len(rows)]
    assert len(rows)<=15 and outcome['control_steps']<=15000
    states=0;checked=0;prefixes=0;diagnostics=[]
    for row in rows:
        path=ROOT/row['trace'];assert s.read(path.with_suffix('.json'))==row and s.digest(path)==row['trace_sha256']
        if row['interrupted']:continue
        states+=verify_episode(path,row,p);a=read_arrays(path);checked+=1
        assert np.all(a['control_residual_actions']==0) and np.all(a['residual_policy_actions']==0)
        assert np.all(a['residual_observations'][:,48:60]==0)
        audit_reward_and_nominal(a,row)
        for condition in ['healthy','RR_calf']:
            ref=ARCHIVE/f"{condition}_{row['seed']}.npz";meta=s.read(ref.with_suffix('.json'))
            assert s.digest(ref)==meta['trace_sha256'] and row['initial_state_sha256']==meta['initial_state_sha256']
            check_prefix(a,read_arrays(ref),row['fault_onset_tick']);prefixes+=1
        onset=row['fault_onset_tick'];forward=np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
        progress=(a['qpos'][:,:2]-row['initial_xy'])@forward
        diagnostics.append(dict(strength_remaining=row['strength_remaining'],seed=row['seed'],
            post_net_progress_m=float(progress[-1]-progress[onset]),
            weak_torque_limit_fraction=float(np.mean(np.abs(a['requested_torque'][onset:,11]-a['limited_torque'][onset:,11])>1e-9)),
            fixed_horizon_progress_m={str(t):float(progress[onset+round(t/.002)]-progress[onset]) if onset+round(t/.002)<len(progress) else None for t in [.5,2.]}))
    controls=s.read(OUT/'historical_controls.json')
    for row in controls:
        assert s.digest(ROOT/row['trace'])==row['trace_sha256']
    complete=len(rows)==checked==15 and outcome['stop_reason']=='episodes_completed'
    if complete:assert prefixes==30
    proof=dict(audit_passed=True,study_complete=complete,new_episodes_checked=checked,physics_states_checked=states,
        exact_prefault_control_pairs=prefixes,archived_controls=10,no_control_rollouts=True,
        protected_prior_artifacts_unchanged=len(s.read(OUT/'design.json')['prior_artifacts_sha256']),
        only_declared_strength_changed=True,residual_zero=True,nominal_reward_and_physics_verified=True,
        training=False,confirmation_final_seeds_unused=True)
    for name,value in [('diagnostics.json',diagnostics),('completion_audit.json',proof)]:
        if (OUT/name).exists():assert s.read(OUT/name)==value
        else:s.write(OUT/name,value)
    print(f'Audit passed: {checked} new episodes, {prefixes} archived prefix pairs',flush=True)
    return proof


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
