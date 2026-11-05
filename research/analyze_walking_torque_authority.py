"""Instantaneous residual-box torque envelopes on saved states; no simulation."""
from project_paths import project_file
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from walking_baseline import torque_limit, JOINTS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/walking_torque_authority_review'
PILOTS={'progress10':ROOT/'results/walking_residual_privileged_pilot',
        'progress100':ROOT/'results/walking_progress_reward_pilot'}
DEFAULT=np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
RANGES=np.array([[-1.0472,1.0472],[-1.5708,3.4907],[-2.7227,-.83776]]*2+
    [[-1.0472,1.0472],[-.5236,4.5379],[-2.7227,-.83776]]*2)
LIMITS=np.tile([[-23.7,23.7],[-23.7,23.7],[-45.43,45.43]],(4,1))
EPS=1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def envelope(nominal,q,v,kp,kd,strength):
    assert nominal.shape==q.shape==v.shape==kp.shape==kd.shape==strength.shape
    assert np.all(kp>0) and np.isfinite(nominal).all()
    low_target=np.clip(nominal-.1,RANGES[:,0],RANGES[:,1])
    high_target=np.clip(nominal+.1,RANGES[:,0],RANGES[:,1])
    low=torque_limit(kp*(low_target-q)-kd*v,v,'go2_rl_lab',LIMITS)*strength
    high=torque_limit(kp*(high_target-q)-kd*v,v,'go2_rl_lab',LIMITS)*strength
    assert np.isfinite(low).all() and np.isfinite(high).all() and np.all(high>=low-EPS)
    return low,high


def metrics(low,high,actual,nominal_unfaulted,start,end,weak):
    low,high,actual,reference=[x[start:end] for x in [low,high,actual,nominal_unfaulted]]
    span=high-low
    result=dict(physics_rows=end-start,duration_s=(end-start)*.002,
        zero_span_fraction_per_joint=np.mean(span<=EPS,axis=0).tolist(),
        mean_span_Nm_per_joint=np.mean(span,axis=0).tolist(),
        mean_lower_headroom_Nm_per_joint=np.mean(actual-low,axis=0).tolist(),
        mean_upper_headroom_Nm_per_joint=np.mean(high-actual,axis=0).tolist(),
        mean_max_change_Nm_per_joint=np.mean(np.maximum(actual-low,high-actual),axis=0).tolist(),
        nominal_unfaulted_outside_fraction_per_joint=np.mean((reference<low-EPS)|(reference>high+EPS),axis=0).tolist())
    if weak is not None:
        for key in ['zero_span_fraction','mean_span_Nm','mean_max_change_Nm','nominal_unfaulted_outside_fraction']:
            result['weak_'+key]=result[key+'_per_joint'][weak]
        other=[j for j in range(12) if j//3!=weak//3]
        result['other_legs_mean_span_Nm']=float(span[:,other].mean())
    return result


def main():
    assert not OUT.exists(),'Never overwrite a study'
    assert 'OK' in (ROOT/'work/torque_authority_tests.log').read_text(encoding='utf-8')
    paths=[Path(__file__),ROOT/'tests/test_analyze_walking_torque_authority.py',
        ROOT/'work/torque_authority_tests.log',ROOT/'research/walking_baseline.py',ROOT/'research/walking_residual_env.py',
        ROOT/'research/walking_residual_height_env.py',ROOT/'research/walking_residual_privileged_env.py',
        ROOT/'research/walking_progress_reward_pilot.py',ROOT/'vendor/mujoco_menagerie/unitree_go2/go2.xml',
        ROOT/'vendor/mujoco_menagerie/unitree_go2/scene.xml']
    cohorts={}
    for pilot,folder in PILOTS.items():
        assert read(folder/'final_verification.json')['passed'] if pilot=='progress100' else read(folder/'final_verification.json')['audit_passed']
        paths+=[folder/'final_verification.json',folder/'protocol.json',folder/'checkpoint_131072.json',folder/'checkpoint_131072.zip']
        rows=read(folder/'validation_131072/episodes.json')
        assert len(rows)==25 and len({(r['condition'],r['seed']) for r in rows})==25
        paths.append(folder/'validation_131072/episodes.json')
        for row in rows:
            stem=f"{row['condition']}_{row['seed']}"
            path=folder/'validation_131072'/(stem+'.npz')
            assert digest(path)==row['trace_sha256'] and not row['interrupted']
            assert row['seed'] in range(23000,23005)
            paths+=[path,path.with_suffix('.json')]
            assert read(path.with_suffix('.json'))==row
        cohorts[pilot]=rows
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    OUT.mkdir()
    write(OUT/'protocol.json',dict(question='How much instantaneous torque authority remains inside the frozen residual box?',
        pilots=list(PILOTS),checkpoint_steps=131072,cohort='All25 episodes each; repeated development poses23000–23004',
        residual_box_rad=[-.1,.1],joint_order=JOINTS,joint_ranges=RANGES.tolist(),torque_limits=LIMITS.tolist(),
        frozen_state='qpos/qvel/nominal target/PD gains/fault multiplier are all factual saved values',
        endpoint_definition='Per-joint exact attainable minimum/maximum torque at one state, independent target box; no coupled future trajectory claim',
        windows='First informed control interval; complete first0.5s; complete2–5s after first informed control; entire available post-informed supplemental',
        reference='Unfaulted nominal torque on the SAME fault state, not torque from healthy trajectory or a required successful compensation',
        numeric_tolerance_Nm=EPS,aggregation='Episode means/fractions; equal-weight median/range across five poses per pilot/fault/window; coverage explicit',
        budget_compute_s=120,input_hashes=hashes,training=False,new_rollouts=0,new_policy_inference=0,
        stops=['input hash mismatch','nonfinite','actual outside envelope','factual torque mismatch','budget exhausted'],
        limits='One seed, fixed-state local authority. Unreachable nominal torque does not prove locomotion impossible; healthy-leg headroom does not prove a feasible gait.'))
    (OUT/'arrays').mkdir()
    started=time.perf_counter()
    execution=dict(completed=False,error=None,training=False,new_rollouts=0,new_policy_inference=0)
    records=[]
    try:
        states,controls=0,0
        for pilot,rows in cohorts.items():
            for row in rows:
                assert time.perf_counter()-started<=120,'Budget exhausted'
                stem=f"{row['condition']}_{row['seed']}"
                with np.load(PILOTS[pilot]/'validation_131072'/(stem+'.npz'),allow_pickle=False) as z:
                    a={k:z[k] for k in z.files}
                n=len(a['qpos'])-1
                nc=len(a['control_residual_actions'])
                assert nc==(n+9)//10
                nominal_control=DEFAULT+.25*a['control_actions']
                nominal=np.repeat(nominal_control,10,axis=0)[:n]
                strength=np.ones((n,12))
                weak=None if row['condition']=='healthy' else JOINTS.index(row['condition'])
                if weak is not None:
                    np.testing.assert_array_equal(a['strength'],np.where(np.arange(n)>=row['fault_onset_tick'],.1,1.))
                    strength[:,weak]=a['strength']
                else:
                    np.testing.assert_array_equal(a['strength'],np.ones(n))
                q,v=a['qpos'][:-1,7:],a['qvel'][:-1,6:]
                low,high=envelope(nominal,q,v,a['kp'],a['kd'],strength)
                action=a['control_residual_actions']
                assert np.isfinite(action).all() and np.all(np.abs(action)<=1)
                target=np.clip(nominal_control+.1*action,RANGES[:,0],RANGES[:,1])
                target=np.repeat(target,10,axis=0)[:n]
                np.testing.assert_array_equal(target,a['target'])
                requested=a['kp']*(target-q)-a['kd']*v
                np.testing.assert_allclose(requested,a['requested_torque'],rtol=0,atol=1e-12)
                actual=torque_limit(requested,v,'go2_rl_lab',LIMITS)*strength
                np.testing.assert_array_equal(actual,a['applied_torque'])
                assert np.all(actual>=low-EPS) and np.all(actual<=high+EPS)
                nominal_requested=a['kp']*(np.clip(nominal,RANGES[:,0],RANGES[:,1])-q)-a['kd']*v
                nominal_unfaulted=torque_limit(nominal_requested,v,'go2_rl_lab',LIMITS)
                other=[j for j in range(12) if j!=weak]
                assert np.all(nominal_unfaulted[:,other]>=low[:,other]-EPS) and np.all(nominal_unfaulted[:,other]<=high[:,other]+EPS)
                first=None if weak is None else (row['fault_onset_tick']//10+1)*10
                windows={}
                if first is not None:
                    assert first<n
                    for name,start,end in [('first',first,min(first+10,n)),('0.5s',first,first+250),
                        ('2-5s',first+1000,first+2500),('available',first,n)]:
                        windows[name]=None if end>n else metrics(low,high,actual,nominal_unfaulted,start,end,weak)
                path=OUT/'arrays'/f'{pilot}_{stem}.npz'
                np.savez_compressed(path,low=low,high=high,actual=actual,nominal_unfaulted=nominal_unfaulted)
                records.append(dict(pilot=pilot,condition=row['condition'],seed=row['seed'],
                    weak_index=weak,source_trace_sha256=row['trace_sha256'],first_informed_tick=first,
                    historic_success=row['success'],historic_fallen=row['fallen'],windows=windows,
                    envelope_sha256=digest(path),physics_rows=n,control_rows=nc))
                states+=n+1
                controls+=nc
        write(OUT/'episodes.json',records)
        summary=[]
        for pilot in PILOTS:
            for condition in ['FL_calf','FR_calf','RL_calf','RR_calf']:
                group=[r for r in records if (r['pilot'],r['condition'])==(pilot,condition)]
                for name in ['first','0.5s','2-5s','available']:
                    rows=[r['windows'][name] for r in group if r['windows'][name] is not None]
                    metrics_summary={}
                    for key in ['weak_zero_span_fraction','weak_mean_span_Nm','weak_mean_max_change_Nm',
                                'weak_nominal_unfaulted_outside_fraction','other_legs_mean_span_Nm']:
                        values=[r[key] for r in rows]
                        metrics_summary[key]=None if not values else dict(median=float(np.median(values)),minimum=min(values),maximum=max(values))
                    summary.append(dict(pilot=pilot,condition=condition,window=name,eligible_episodes=len(rows),metrics=metrics_summary))
        write(OUT/'summary.json',summary)
        assert time.perf_counter()-started<=120,'Budget exhausted'
        execution.update(completed=True,episodes=len(records),physics_states=states,control_rows=controls)
    except BaseException:
        execution['error']=traceback.format_exc()
        raise
    finally:
        execution['compute_wall_s']=time.perf_counter()-started
        write(OUT/'execution.json',execution)
    for name,sha in hashes.items():
        assert digest(project_file(ROOT, name))==sha,name
    write(OUT/'integrity.json',dict(passed=True,selected_inputs_unchanged=len(hashes),scope='Selected sources and traces, not full historical archive re-audit'))
    print(json.dumps(execution),flush=True)


if __name__=='__main__':
    main()
