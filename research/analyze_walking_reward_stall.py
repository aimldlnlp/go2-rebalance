"""Offline reward decomposition from frozen closed-loop traces, no policy calls."""
from project_paths import project_file
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/walking_strength_masking_rollout'
OUT = ROOT/'results/walking_reward_stall_review'
WINDOWS = {'early':(0.,.5),'middle':(.5,2.),'late':(2.,5.)}
WEIGHTS = dict(velocity_tracking=1.,yaw_tracking=.25,goal_progress=10.,height_error=-10.,
    tilt_squared=-2.,cross_track_excess=-5.,stable_goal=.5,residual_effort=-.05,
    residual_change=-.01,fall=-10.,mission_success=5.)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def rotation(q):
    w,x,y,z = q[:,3:7].T
    np.testing.assert_allclose(w*w+x*x+y*y+z*z,1.,rtol=0,atol=1e-10)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)


def reconstruct(a,row):
    q,v = a['qpos'],a['qvel']
    ticks = np.arange(0,len(q)-1,10)
    ends = np.minimum(ticks+10,len(q)-1)
    r = rotation(q)
    direction = np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    error = np.linalg.norm(q[:,:2]-(np.asarray(row['initial_xy'])+2*direction),axis=1)
    yaw = np.arctan2(r[:,1,0],r[:,0,0])
    heading = np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
    tilt = np.arccos(np.clip(r[:,2,2],-1,1))
    cross = np.abs((q[:,:2]-row['initial_xy'])@np.array([-direction[1],direction[0]]))
    stable = (error<=.15)&(heading<=np.deg2rad(15))&(np.linalg.norm(v[:,:2],axis=1)<=.05)
    body = np.einsum('nji,nj->ni',r[ends],v[ends,:3])
    command,action = a['command'][ticks],a['control_residual_actions']
    previous = np.vstack([np.zeros((1,12)),action[:-1]])
    fall,success = np.zeros(len(ticks)),np.zeros(len(ticks))
    fall[-1],success[-1] = float(row['fallen']),float(row['success'])
    raw = dict(velocity_tracking=np.exp(-np.sum((body[:,:2]-command[:,:2])**2,axis=1)/.15**2),
        yaw_tracking=np.exp(-((v[ends,5]-command[:,2])/.5)**2),goal_progress=error[ticks]-error[ends],
        height_error=(q[ends,2]-.338)**2,tilt_squared=tilt[ends]**2,
        cross_track_excess=np.maximum(0.,cross[ends]-.20)**2,stable_goal=stable[ends].astype(float),
        residual_effort=np.mean(action**2,axis=1),residual_change=np.mean((action-previous)**2,axis=1),
        fall=fall,mission_success=success)
    terms = np.stack([raw[k]*weight for k,weight in WEIGHTS.items()],axis=1)
    np.testing.assert_array_equal(a['reward_component_names'],list(WEIGHTS))
    np.testing.assert_allclose(terms,a['reward_components'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(terms.sum(axis=1),a['control_rewards'],rtol=0,atol=1e-10)
    assert abs(a['control_rewards'].sum()-row['episode_return'])<1e-8
    return terms,error


def window(a,row,anchor,start,end):
    first,last = anchor+round(start/.002),anchor+round(end/.002)
    n = len(a['qpos'])-1
    if first < 0 or last > n:
        return None
    ticks = np.arange(0,n,10)
    ends = np.minimum(ticks+10,n)
    selected = (ticks>=first)&(ends<=last)
    assert selected.any()
    components = a['reward_components'][selected]
    forward = np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    cmd = np.linalg.norm(a['command'][ticks[selected],:2],axis=1)
    speed = np.linalg.norm(a['qvel'][ends[selected],:2],axis=1)
    stalled = (cmd>=.10)&(speed<=.03)
    means = components.mean(axis=0)
    return dict(duration_s=(last-first)*.002,control_rows=int(selected.sum()),
        mean_reward_per_control=float(a['control_rewards'][selected].mean()),
        mean_terms=dict(zip(WEIGHTS,map(float,means))),
        net_forward_progress_m=float((a['qpos'][last,:2]-a['qpos'][first,:2])@forward),
        mean_command_m_s=float(cmd.mean()),mean_planar_speed_m_s=float(speed.mean()),
        commanded_low_speed_fraction=float(stalled.mean()),commanded_low_speed_rows=int(stalled.sum()),
        low_speed_mean_reward=None if not stalled.any() else float(components[stalled].sum(axis=1).mean()),
        low_speed_mean_terms=None if not stalled.any() else dict(zip(WEIGHTS,map(float,components[stalled].mean(axis=0)))))


def main():
    assert not OUT.exists(), 'Never overwrite a review'
    assert read(SOURCE/'audit.json')['passed']
    paths = [Path(__file__),ROOT/'tests/test_analyze_walking_reward_stall.py',ROOT/'work/reward_stall_tests.log',
        ROOT/'research/walking_residual_env.py',ROOT/'research/walking_residual_height_env.py',SOURCE/'audit.json',SOURCE/'protocol.json']
    rows_by_arm = {}
    for arm in ['factual_131072','masked_131072']:
        rows = read(SOURCE/arm/'episodes.json')
        assert len(rows)==25 and len({(r['condition'],r['seed']) for r in rows})==25
        rows_by_arm[arm] = rows
        paths.append(SOURCE/arm/'episodes.json')
        for row in rows:
            stem = f"{row['condition']}_{row['seed']}"
            assert row['seed'] in range(23000,23005) and not row['interrupted']
            assert digest(SOURCE/arm/(stem+'.npz')) == row['trace_sha256']
            assert read(SOURCE/arm/(stem+'.json'))==row
            paths += [SOURCE/arm/(stem+'.npz'),SOURCE/arm/(stem+'.json')]
    assert 'OK' in (ROOT/'work/reward_stall_tests.log').read_text(encoding='utf-8')
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    OUT.mkdir()
    protocol = dict(question='Which recorded reward terms persist during commanded low-speed behavior?',
        windows_after_fault_s=WINDOWS,low_speed_definition='Control-end planar speed <=0.03 m/s and command >=0.10 m/s; descriptive sample classification, not a stall episode detector',
        cohorts='50 existing traces; 25 factual exact historical replays, 25 masked; no independent training replication',
        reference='Healthy same seed/arm, same absolute physics-time window as each fault; repeated references not independent episodes',
        aggregation='Per-episode window means, equal-weight median/range per condition and arm; complete windows only with explicit coverage; no padding after falls',
        cap_compute_s=120,input_hashes=hashes,training=False,new_rollouts=0,new_policy_inference=0,
        limits='Reward accounting is not advantage, policy gradient, or evidence of causal failure. No new reward weights chosen by return.')
    write(OUT/'protocol.json',protocol)
    started = time.perf_counter()
    result = dict(completed=False,error=None,training=False,new_rollouts=0,new_policy_inference=0)
    try:
        data = {}
        states,controls = 0,0
        for arm,rows in rows_by_arm.items():
            for row in rows:
                assert time.perf_counter()-started<=120,'Budget exceeded'
                stem = f"{row['condition']}_{row['seed']}"
                with np.load(SOURCE/arm/(stem+'.npz'),allow_pickle=False) as z:
                    a = {k:z[k] for k in z.files}
                reconstructed,error = reconstruct(a,row)
                assert all(np.isfinite(x).all() for x in [reconstructed,error])
                states+=len(a['qpos'])
                controls+=len(a['control_rewards'])
                data[(arm,row['condition'],row['seed'])]=(a,row)
        episodes = []
        for arm,rows in rows_by_arm.items():
            for row in rows:
                if row['condition']=='healthy':
                    continue
                a,_ = data[(arm,row['condition'],row['seed'])]
                healthy,hr = data[(arm,'healthy',row['seed'])]
                onset = row['fault_onset_tick']
                assert onset is not None
                values = {}
                for name,(start,end) in WINDOWS.items():
                    f = window(a,row,onset,start,end)
                    h = window(healthy,hr,onset,start,end)
                    if f is not None:
                        assert h is not None
                    values[name] = dict(fault=f,healthy_same_time=h)
                episodes.append(dict(arm=arm,condition=row['condition'],seed=row['seed'],
                    historic_fallen=row['fallen'],historic_success=row['success'],
                    onset_tick=onset,windows=values))
        summary = []
        for arm in rows_by_arm:
            for condition in ['FL_calf','FR_calf','RL_calf','RR_calf']:
                group = [e for e in episodes if e['arm']==arm and e['condition']==condition]
                for name in WINDOWS:
                    eligible = [e for e in group if e['windows'][name]['fault'] is not None]
                    item = dict(arm=arm,condition=condition,window=name,eligible=len(eligible),total=5,
                        missing_seeds=[e['seed'] for e in group if e not in eligible])
                    for label in ['fault','healthy_same_time']:
                        rows = [e['windows'][name][label] for e in eligible]
                        fields = ['mean_reward_per_control','net_forward_progress_m','mean_command_m_s',
                                  'mean_planar_speed_m_s','commanded_low_speed_fraction']
                        item[label] = {key:None if not rows else dict(median=float(np.median([r[key] for r in rows])),
                            minimum=min(r[key] for r in rows),maximum=max(r[key] for r in rows)) for key in fields}
                        item[label]['median_terms'] = {key:None if not rows else float(np.median([r['mean_terms'][key] for r in rows])) for key in WEIGHTS}
                        # Median of episode component means need not sum to median reward.
                    summary.append(item)
        write(OUT/'episodes.json',episodes)
        write(OUT/'summary.json',summary)
        write(OUT/'reconstruction.json',dict(passed=True,episodes=50,physics_states=states,controls=controls,
            all_reward_components_reconstructed=True,component_atol=1e-10,return_atol=1e-8))
        assert time.perf_counter()-started<=120,'Budget exceeded'
        result['completed']=True
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        result['compute_wall_s']=time.perf_counter()-started
        write(OUT/'execution.json',result)
    for name,expected in hashes.items():
        assert digest(project_file(ROOT, name))==expected,name
    write(OUT/'integrity.json',dict(passed=True,selected_inputs_unchanged=len(hashes),
        scope='Selected traces, metadata, reward sources, test and analysis implementation',
        new_rollouts=0,training=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
