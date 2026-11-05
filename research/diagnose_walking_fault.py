"""Offline paired fault diagnosis. Never constructs environments or runs policies."""
from project_paths import project_file
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/walking_fault_diagnosis'
PILOTS = {'height_0270': 'walking_residual_pilot', 'height_0338': 'walking_residual_height_pilot'}
STAGES = ['baseline', 'validation_32768', 'validation_65536', 'validation_131072']
CONDITIONS = ['healthy', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
DEFAULT = np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
RANGES = np.array([[-1.0472,1.0472],[-1.5708,3.4907],[-2.7227,-.83776]]*2+
                  [[-1.0472,1.0472],[-.5236,4.5379],[-2.7227,-.83776]]*2)
DT = .002


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def median(values):
    valid = [v for v in values if v is not None]
    return float(np.median(valid)) if valid else None


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def limited(requested, velocity):
    capacity = np.where(velocity*requested > 0, 20.2, 23.4)
    capacity *= np.where(np.abs(velocity)>13.5, np.maximum(0.,(30-np.abs(velocity))/16.5), 1.)
    return np.clip(np.clip(requested,-capacity,capacity),-np.array([23.7,23.7,45.43]*4),np.array([23.7,23.7,45.43]*4))


def stall_start(progress, command_norm, onset, window_ticks, max_gain):
    """First full window with weak forward gain despite mean command >=0.10m/s."""
    starts = np.arange(onset, len(progress)-window_ticks)
    if not len(starts):
        return None
    cumulative = np.r_[0.,np.cumsum(command_norm)]
    gain = progress[starts+window_ticks]-progress[starts]
    commanded = (cumulative[starts+window_ticks]-cumulative[starts])/window_ticks
    hits = starts[(gain <= max_gain)&(commanded >= .1-1e-12)]
    return float((hits[0]-onset)*DT) if len(hits) else None


def rotation_values(q):
    qw,qx,qy,qz = q[:,3:7].T
    tilt = np.rad2deg(np.arccos(np.clip(1-2*(qx*qx+qy*qy),-1,1)))
    return tilt


def diagnose(a, row, healthy=None):
    q,v = a['qpos'],a['qvel']
    n = len(q)-1
    ticks = np.arange(0,n,10)
    assert a['control_residual_actions'].shape == (len(ticks),12)
    forward = np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    progress = (q[:,:2]-row['initial_xy'])@forward
    onset = row['fault_onset_tick']
    if onset is None:
        reached = np.flatnonzero(progress[:-1] >= .5)
        onset = int(reached[0]) if len(reached) else None
    result = {'success':row['success'],'fallen':row['fallen'],
        'simulated_s':n*DT,'fault_onset_tick':row['fault_onset_tick'],
        'anchor_tick':onset,'final_progress_m':float(progress[-1]),
        'failed_checks':[key for key,value in row['checks'].items() if not value]}
    if onset is None:
        return result
    assert 0 <= onset < n
    if row['condition']!='healthy':
        assert row['fault_onset_tick']==onset and np.all(progress[:onset]<.5) and progress[onset]>=.5
    post = np.arange(onset,n)
    control_post = ticks >= onset
    actions = a['control_residual_actions']
    clipped = np.abs(actions)>=1.-1e-7
    raw_target = DEFAULT+.25*a['control_actions']+.1*actions
    target = np.clip(raw_target,RANGES[:,0],RANGES[:,1])
    np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:n])
    joint_clipped = np.abs(raw_target-target)>1e-12
    torque_clipped = np.abs(a['requested_torque']-a['limited_torque'])>1e-9
    np.testing.assert_allclose(limited(a['requested_torque'],v[:-1,6:]),a['limited_torque'],rtol=0,atol=1e-12)
    # Instantaneous same-state comparison only: no alternate closed-loop trajectory is simulated.
    nominal_target = np.repeat(np.clip(DEFAULT+.25*a['control_actions'],RANGES[:,0],RANGES[:,1]),10,axis=0)[:n]
    nominal_requested = a['kp']*(nominal_target-q[:-1,7:])-a['kd']*v[:-1,6:]
    nominal_applied = limited(nominal_requested,v[:-1,6:])
    factors = np.ones((n,12))
    weak = None
    if row['condition']!='healthy':
        weak = 3*CONDITIONS[1:].index(row['condition'])+2
        factors[onset:,weak] = .1
    nominal_applied *= factors
    np.testing.assert_array_equal(a['applied_torque'],a['limited_torque']*factors)
    delta = a['applied_torque']-nominal_applied
    assert np.all(np.abs(delta) <= a['kp']*.1*factors+1e-10)
    result.update({'onset_s':onset*DT,'onset_progress_m':float(progress[onset]),
        'post_duration_s':(n-onset)*DT,'post_net_progress_m':float(progress[-1]-progress[onset]),
        'post_max_progress_gain_m':float(progress[onset:].max()-progress[onset]),
        'post_mean_forward_velocity_m_s':float(np.mean(v[onset:, :2]@forward)),
        'post_command_mean_m_s':float(np.mean(np.linalg.norm(a['command'][post,:2],axis=1))),
        'slow_window_start_s':stall_start(progress,np.linalg.norm(a['command'][:,:2],axis=1),onset,250,.025),
        'stall_window_start_s':stall_start(progress,np.linalg.norm(a['command'][:,:2],axis=1),onset,500,.02),
        'post_min_height_m':float(q[onset:,2].min()),'post_max_tilt_deg':float(rotation_values(q)[onset:].max()),
        'post_residual_rms_rad':rms(.1*actions[control_post]),
        'post_residual_bound_fraction':float(np.mean(clipped[control_post])) if control_post.any() else None,
        'post_joint_clip_fraction':float(np.mean(joint_clipped[control_post])) if control_post.any() else None,
        'post_torque_limit_fraction':float(np.mean(torque_clipped[post])),
        'post_residual_torque_delta_rms_Nm':rms(delta[post]),
        'post_weak_torque_limit_fraction':float(np.mean(torque_clipped[post,weak])) if weak is not None else None,
        'post_weak_applied_torque_rms_Nm':rms(a['applied_torque'][post,weak]) if weak is not None else None,
        'post_weak_residual_torque_delta_rms_Nm':rms(delta[post,weak]) if weak is not None else None,
        'post_weak_residual_torque_delta_max_Nm':float(np.max(np.abs(delta[post,weak]))) if weak is not None else None,
        'post_weak_residual_bound_fraction':float(np.mean(clipped[control_post,weak])) if weak is not None and control_post.any() else None,
        'post_contacts_fraction':[float(x) for x in a['contacts'][post].mean(axis=0)],
        'onset_contacts':a['contacts'][onset].astype(int).tolist(),
        'fall_after_onset_s':(n-onset)*DT if row['fallen'] else None,
        'windows':[]})
    for start,end in [(-.5,0.),(0.,.5),(.5,2.),(2.,5.)]:
        mask = (ticks >= onset+start/DT)&(ticks < onset+end/DT)
        physics = (np.arange(n)>=onset+start/DT)&(np.arange(n)<onset+end/DT)
        result['windows'].append({'start_s':start,'end_s':end,'controls':int(mask.sum()),
            'observed_duration_s':int(physics.sum())*DT,
            'residual_rms_rad':rms(.1*actions[mask]),
            'forward_velocity_m_s':float(np.mean(v[:-1,:2][physics]@forward)) if physics.any() else None,
            'residual_bound_fraction':float(np.mean(clipped[mask])) if mask.any() else None,
            'torque_limit_fraction':float(np.mean(torque_clipped[physics])) if physics.any() else None})
    if healthy is not None:
        # Same physical/observation history before the imposed fault, not a matched history afterwards.
        for key in ['qpos','qvel']:
            np.testing.assert_array_equal(a[key][:onset+1],healthy[key][:onset+1])
        for key in ['command','applied_torque','contacts']:
            np.testing.assert_array_equal(a[key][:onset],healthy[key][:onset])
        before = ticks<=onset
        np.testing.assert_array_equal(actions[before],healthy['control_residual_actions'][:int(before.sum())])
        count = min(len(actions),len(healthy['control_residual_actions']))
        elapsed = (ticks[:count]-onset)*DT
        action_delta = .1*(actions[:count]-healthy['control_residual_actions'][:count])
        hits = np.flatnonzero((elapsed>0)&(np.max(np.abs(action_delta),axis=1)>1e-6))
        result['first_action_difference_s'] = float(elapsed[hits[0]]) if len(hits) else None
        result['action_difference_first_05s_rms_rad'] = rms(action_delta[(elapsed>=0)&(elapsed<.5)])
        result['action_difference_post_rms_rad'] = rms(action_delta[elapsed>=0])
        result['prefault_exact_healthy_parity'] = True
    return result


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite fault diagnosis'
    height = ROOT/'results/walking_residual_height_pilot'
    protected = dict(read(height/'design.json')['prior_artifacts_sha256'])
    for name,sha in protected.items():
        assert digest(project_file(ROOT, name))==sha,name
    for path in height.rglob('*'):
        if path.is_file():protected[str(path.relative_to(ROOT))]=digest(path)
    for pilot in PILOTS.values():
        folder = ROOT/'results'/pilot
        assert read(folder/'completion_audit.json')['audit_passed']
        assert read(folder/'protocol.json')['validation_seeds']==list(range(23000,23005))
    OUT.mkdir()
    write(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Existing 200 saved episodes only; 100 exact old/new pairs, including 80 fault pairs. No new simulation, policy inference, training or final seeds.',
        'stages':STAGES,'conditions':CONDITIONS,'seeds':list(range(23000,23005)),
        'windows_s':[[-.5,0.],[0.,.5],[.5,2.],[2.,5.]],
        'onset':'Actual fault physics tick; healthy uses first progress >=0.50m as a virtual anchor only.',
        'slow':'First full 0.5s window with forward gain <=0.025m and mean planar command >=0.10m/s.',
        'stall':'First full 1s window with forward gain <=0.02m and mean planar command >=0.10m/s.',
        'action_difference':'First post-onset control tick with any residual target difference >1e-6rad against same-policy paired healthy trajectory. This indicates state-response divergence, not fault identification.',
        'clipping':'Action bound: abs(normalized action)>=1-1e-7. Joint clipping: target changed by >1e-12rad. Torque limiting: requested-limited magnitude >1e-9Nm. PPO pre-clipping raw actions are unavailable.',
        'torque_delta':'At saved state only, actual applied torque minus torque with residual target zero, including same clipping/limiting/fault. Not a closed-loop causal effect or proof of insufficient authority.',
        'aggregation':'Medians of episode metrics within each 5-seed condition; no pooling of actuator intervals across durations. Missing windows/latencies remain null; fall is retained, never padded.',
        'limits':'Exploratory diagnostics, not revised mission thresholds. Same training seed only. Raw returns from different reward definitions are not compared.',
        'preview':'No server/task activation.'})
    write(OUT/'design.json',{'protocol_sha256':digest(OUT/'protocol.json'),
        'source_sha256':{p.name:digest(p) for p in [Path(__file__),ROOT/'tests/test_diagnose_walking_fault.py']},
        'protected_prior_artifacts_sha256':protected})
    print(f'Frozen offline diagnosis; {len(protected)} prior files protected',flush=True)


def run():
    d = read(OUT/'design.json')
    assert digest(OUT/'protocol.json')==d['protocol_sha256']
    for name,sha in d['source_sha256'].items():assert digest(project_file(ROOT, name))==sha,name
    assert not (OUT/'episodes.json').exists(), 'Refusing to overwrite diagnosis'
    rows, baseline_hashes = [], {}
    for pilot,folder in PILOTS.items():
        for stage in STAGES:
            base = ROOT/'results'/folder/stage
            metadata = read(base/'episodes.json')
            assert len(metadata)==25 and {(r['condition'],r['seed']) for r in metadata}=={(c,s) for c in CONDITIONS for s in range(23000,23005)}
            healthy = {}
            for row in metadata:
                if row['condition']=='healthy':
                    path=base/f"healthy_{row['seed']}.npz"
                    assert digest(path)==row['trace_sha256']
                    with np.load(path,allow_pickle=False) as a:
                        healthy[row['seed']]={k:a[k] for k in ['qpos','qvel','command','applied_torque','contacts','control_residual_actions']}
            for row in metadata:
                stem=f"{row['condition']}_{row['seed']}"
                path=base/(stem+'.npz')
                assert digest(path)==row['trace_sha256'] and read(base/(stem+'.json'))==row
                assert not row['interrupted']
                with np.load(path,allow_pickle=False) as a:
                    actual_initial=hashlib.sha256(a['qpos'][0].tobytes()+a['qvel'][0].tobytes()).hexdigest()
                    assert actual_initial==row['initial_state_sha256']
                    r=diagnose(a,row,None if row['condition']=='healthy' else healthy[row['seed']])
                    if stage=='baseline':
                        hashes={k:hashlib.sha256(a[k].tobytes()).hexdigest() for k in healthy[row['seed']]}
                        key=(row['condition'],row['seed'])
                        if pilot=='height_0270':baseline_hashes[key]=hashes
                        else:assert baseline_hashes[key]==hashes
                rows.append({'pilot':pilot,'stage':stage,'condition':row['condition'],'seed':row['seed'],
                    'initial_state_sha256':actual_initial,'trace_sha256':row['trace_sha256'],'trace':str(path.relative_to(ROOT)),**r})
            print(f'{pilot}/{stage}: 25 traces diagnosed',flush=True)
    write(OUT/'episodes.json',rows)
    summary=[]
    keys=['post_net_progress_m','post_max_progress_gain_m','slow_window_start_s','stall_window_start_s',
        'post_residual_rms_rad','post_residual_bound_fraction','post_joint_clip_fraction','post_torque_limit_fraction',
        'post_weak_torque_limit_fraction','post_weak_applied_torque_rms_Nm','post_weak_residual_torque_delta_rms_Nm',
        'post_weak_residual_torque_delta_max_Nm','post_weak_residual_bound_fraction',
        'first_action_difference_s','action_difference_first_05s_rms_rad','action_difference_post_rms_rad',
        'post_min_height_m','post_max_tilt_deg']
    for pilot in PILOTS:
        for stage in STAGES:
            for condition in CONDITIONS:
                group=[r for r in rows if (r['pilot'],r['stage'],r['condition'])==(pilot,stage,condition)]
                assert len(group)==5
                summary.append({'pilot':pilot,'stage':stage,'condition':condition,'episodes':5,
                    'successes':sum(r['success'] for r in group),'falls':sum(r['fallen'] for r in group),
                    'stalls':sum(r.get('stall_window_start_s') is not None for r in group),
                    'median':{k:median([r.get(k) for r in group]) for k in keys},
                    'observed_count':{k:sum(r.get(k) is not None for r in group) for k in keys}})
    pairs=[]
    reference={(r['stage'],r['condition'],r['seed']):r for r in rows if r['pilot']=='height_0270'}
    for new in [r for r in rows if r['pilot']=='height_0338']:
        old=reference[(new['stage'],new['condition'],new['seed'])]
        assert old['initial_state_sha256']==new['initial_state_sha256']
        pairs.append({'stage':new['stage'],'condition':new['condition'],'seed':new['seed'],
            'old_fall':old['fallen'],'new_fall':new['fallen'],
            'delta':{k:new[k]-old[k] if new.get(k) is not None and old.get(k) is not None else None for k in keys}})
    write(OUT/'summary.json',summary)
    write(OUT/'pairs.json',pairs)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (OUT/'episodes.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(flat)
    for name,sha in d['protected_prior_artifacts_sha256'].items():assert digest(project_file(ROOT, name))==sha,name
    proof={'audit_passed':True,'episodes':len(rows),'paired_old_new_episodes':len(pairs),
        'fault_episodes':sum(r['condition']!='healthy' for r in rows),
        'prefault_exact_healthy_pairs':sum(r.get('prefault_exact_healthy_parity',False) for r in rows),
        'baseline_cross_pilot_exact_pairs':len(baseline_hashes),
        'protected_prior_artifacts_unchanged':len(d['protected_prior_artifacts_sha256']),
        'new_simulation_episodes':0,'training':False,'policy_inference':False,'final_seeds_used':False}
    assert proof['episodes']==200 and proof['paired_old_new_episodes']==100 and proof['prefault_exact_healthy_pairs']==160
    write(OUT/'completion_audit.json',proof)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.stage]()
