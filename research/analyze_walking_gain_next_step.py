"""Saved-trace decision review after gain ablation; no policy or simulator runs."""
from project_paths import project_file
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import diagnose_walking_fault as d

OUT = d.ROOT/'results/walking_gain_next_step_review'
WINDOWS = {'pre':(-.5,0.),'early':(0.,.5),'middle':(.5,2.),'late':(2.,5.)}


def window_metrics(a, row, onset, start_s, end_s):
    start, end = onset+round(start_s/.002), onset+round(end_s/.002)
    n = len(a['qpos'])-1
    if start < 0 or end > n:
        return None
    q,v = a['qpos'],a['qvel']
    command = a['command'][start:end]
    contact = a['contacts'][start:end].astype(float)
    forward = np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    control_ticks = np.arange(len(a['control_residual_actions']))*10
    controls = (control_ticks>=start)&(control_ticks<end)
    actions = a['control_residual_actions'][controls]
    nominal = np.repeat(d.DEFAULT+.25*a['control_actions'],10,axis=0)[:n]
    nominal = np.clip(nominal,d.RANGES[:,0],d.RANGES[:,1])
    nominal_requested = a['kp'][start:end]*(nominal[start:end]-q[start:end,7:])-a['kd'][start:end]*v[start:end,6:]
    nominal_applied = d.limited(nominal_requested,v[start:end,6:])
    weak = None if row['condition']=='healthy' else 3*d.CONDITIONS[1:].index(row['condition'])+2
    if weak is not None:
        nominal_applied[np.arange(start,end)>=onset,weak] *= .1
    delta = a['applied_torque'][start:end]-nominal_applied
    energy = np.mean(delta**2,axis=0)
    weak_share = None if weak is None or energy.sum()==0 else float(energy[weak]/energy.sum())
    weak_leg_share = None if weak is None or energy.sum()==0 else float(energy.reshape(4,3).sum(axis=1)[weak//3]/energy.sum())
    return dict(duration_s=(end-start)*.002,
        forward_speed_m_s=float(np.mean(v[start+1:end+1,:2]@forward)),
        command_speed_m_s=float(np.linalg.norm(command[:,:2],axis=1).mean()),
        net_progress_m=float((q[end,:2]-q[start,:2])@forward),
        contacts_fraction=contact.mean(axis=0).tolist(),
        contact_transitions_per_s=(np.sum(np.abs(np.diff(contact,axis=0)),axis=0)/((end-start)*.002)).tolist(),
        all_four_contacts_fraction=float(np.mean(contact.sum(axis=1)==4)),
        joint_velocity_rms_rad_s=np.sqrt(np.mean(v[start:end,6:].reshape(-1,4,3)**2,axis=(0,2))).tolist(),
        nominal_target_variation_rms_rad=np.sqrt(np.mean((nominal[start:end].reshape(-1,4,3)-nominal[start:end].reshape(-1,4,3).mean(axis=0))**2,axis=(0,2))).tolist(),
        residual_rms_rad=np.sqrt(np.mean((.1*actions.reshape(-1,4,3))**2,axis=(0,2))).tolist(),
        applied_torque_delta_rms_Nm=np.sqrt(np.mean(delta.reshape(-1,4,3)**2,axis=(0,2))).tolist(),
        weak_calf_share_squared_torque_delta=weak_share,
        weak_leg_share_squared_torque_delta=weak_leg_share)


def median_window(rows, key):
    good = [r['windows'][key] for r in rows if r['windows'][key] is not None]
    return dict(available=len(good),expected=len(rows),
        medians={k:np.median([r[k] for r in good],axis=0).tolist()
                 for k in good[0] if all(r[k] is not None for r in good)} if good else {})


def main():
    assert not OUT.exists(), 'Refusing to overwrite saved-trace decision review'
    base = d.ROOT/'results/walking_gain_ablation'
    assert d.read(base/'completion_audit.json')['study_complete']
    originals = d.read(base/'episodes.json')
    assert len(originals)==80 and len({(r['gain'],r['condition'],r['seed']) for r in originals})==80
    manifest = {str(p.relative_to(d.ROOT)):d.digest(p) for p in base.rglob('*') if p.is_file()}
    items = [(base/f"gain_{r['gain']}"/f"{r['condition']}_{r['seed']}.npz",r) for r in originals]
    healthy = d.ROOT/'results/walking_residual_low_noise_pilot/validation_131072'
    for seed in range(23000,23005):
        path = healthy/f'healthy_{seed}.npz'
        row = d.read(path.with_suffix('.json')); row['gain']=1
        items.append((path,row))
        for p in [path,path.with_suffix('.json')]:manifest[str(p.relative_to(d.ROOT))]=d.digest(p)
    OUT.mkdir()
    d.write(OUT/'protocol.json',dict(question='Does the evidence support more per-leg trace diagnosis, and which unresolved mechanism should be prioritized?',
        scope='Read existing 80 gain fault traces and five matched healthy traces only. No rollout, inference, training, final/confirmation seeds, or preview.',
        windows_after_onset_s=WINDOWS,window_eligibility='Only full windows; missing windows remain null, no padding or survivor-only causal claim.',
        healthy_anchor='First pre-state progress >=0.50m, matched to the common pre-intervention trajectory.',
        observations='Per-leg contact fraction/transitions, joint velocity, nominal target modulation, residual and instantaneous applied-torque delta; forward progress/command.',
        torque_delta='Same saved state, actual torque minus torque with residual target zero, using identical clipping/limits/fault. Squared delta shares measure actuator-level influence, not causal benefit or learned credit.',
        contact_limits='Contact flags show touching, not support force or forward propulsion; transitions are descriptive, not a validated gait classifier.',
        selection='Do not diagnose all failures as one cause. Prioritize missing evidence with a plausible next intervention; retain all four gains and all poses.',
        training_seed_count=1,development_only=True))
    d.write(OUT/'design.json',dict(source_sha256=d.digest(__file__),inputs_sha256=manifest))
    rows=[]
    for path,row in items:
        assert d.digest(path)==row['trace_sha256']
        with np.load(path,allow_pickle=False) as archive:
            a = {k:archive[k] for k in ['qpos','qvel','command','contacts','control_residual_actions',
                 'control_actions','kp','kd','applied_torque']}
        assert all(np.isfinite(v).all() for v in a.values())
        progress=(a['qpos'][:,:2]-row['initial_xy'])@np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
        onset=int(np.flatnonzero(progress[:-1]>=.5)[0])
        if row['condition']!='healthy':assert onset==row['fault_onset_tick']
        rows.append(dict(gain=row['gain'],condition=row['condition'],seed=row['seed'],
            success=row['success'],fallen=row['fallen'],failed_checks=[k for k,v in row['checks'].items() if not v],
            fall_after_onset_s=(len(a['qpos'])-1-onset)*.002 if row['fallen'] else None,
            windows={name:window_metrics(a,row,onset,*bounds) for name,bounds in WINDOWS.items()}))
    groups=[]
    for condition in ['healthy',*d.CONDITIONS[1:]]:
        for gain in ([1] if condition=='healthy' else [0,1,2,4]):
            group=[r for r in rows if (r['gain'],r['condition'])==(gain,condition)]
            assert len(group)==5
            groups.append(dict(gain=gain,condition=condition,episodes=5,successes=sum(r['success'] for r in group),
                falls=sum(r['fallen'] for r in group),failed_checks=dict(Counter(k for r in group for k in r['failed_checks'])),
                windows={name:median_window(group,name) for name in WINDOWS}))
    d.write(OUT/'episodes.json',rows);d.write(OUT/'summary.json',groups)
    # A full pre-fault window is a negative control for every future gain/condition.
    control={r['seed']:r['windows']['pre'] for r in rows if r['condition']=='healthy'}
    for r in rows:
        for key,value in r['windows']['pre'].items():
            if key not in ['weak_calf_share_squared_torque_delta','weak_leg_share_squared_torque_delta']:
                assert value==control[r['seed']][key],key
    for name,sha in manifest.items():assert d.digest(project_file(d.ROOT, name))==sha,name
    d.write(OUT/'completion_audit.json',dict(episodes=85,gain_fault_episodes=80,healthy_episodes=5,
        prefault_negative_controls_exact=80,input_files_unchanged=len(manifest),
        source_unchanged=d.digest(__file__)==d.read(OUT/'design.json')['source_sha256'],
        no_rollout=True,no_training=True,windows_require_complete_followup=True,audit_passed=True))
    print('Checked 85 saved traces; pre-fault controls and input hashes match.')
    for g in groups:
        if g['gain']==1:
            print(g['condition'],g['windows']['late'])


if __name__=='__main__':
    main()
