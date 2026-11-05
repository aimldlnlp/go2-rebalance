"""Independent mission and actuator audit from preserved physics traces."""
from project_paths import project_file
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

import walking_baseline as w


def verify_episode(path,row,protocol):
    assert w.s.digest(path)==row['trace_sha256']
    with np.load(path,allow_pickle=False) as a:
        q,v=a['qpos'],a['qvel'];n=len(q)-1
        assert v.shape==(n+1,18) and q.shape==(n+1,19) and n<=10000
        assert w.s.digest_bytes(q[0].tobytes()+v[0].tobytes())==row['initial_state_sha256']
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
            weak=w.JOINTS.index(row['condition'])
            applied[onset:,weak]*=.1
            strength[onset:]=.1
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


def audit():
    p=w.verify_frozen();design=w.s.read(w.OUT/'design.json')
    execution=w.s.read(w.OUT/'execution.json')
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(execution['started_utc'])
    assert execution['tests_sha256']==w.s.digest(w.OUT/'tests-before.log')
    selection=w.s.read(w.OUT/'selection.json')
    groups={};counts={};first_pass=None
    for candidate in p['candidate_order']:
        folder=w.OUT/f'healthy_{candidate}'
        if not folder.exists():break
        rows=w.s.read(folder/'episodes.json')
        assert len(rows)==20 and [r['seed'] for r in rows]==p['healthy_seeds']
        assert all(r['condition']=='healthy' and r['candidate']==candidate for r in rows)
        for r in rows:
            name=f"{r['condition']}_{r['seed']}"
            assert r==w.s.read(folder/(name+'.json'))
            counts[str((folder/(name+'.npz')).relative_to(w.OUT))]=verify_episode(folder/(name+'.npz'),r,p)
        groups[candidate]=rows
        if all(r['success'] for r in rows):
            first_pass=candidate
            break
    assert selection['candidate']==first_pass and selection['healthy_gate_passed']==(first_pass is not None)
    diagnosis=None
    if first_pass is not None:
        diagnosis=w.s.read(w.OUT/'diagnosis/episodes.json')
        conditions=['healthy',*[leg+'_calf' for leg in w.LEGS]]
        expected={(c,s) for c in conditions for s in p['diagnostic_seeds']}
        assert len(diagnosis)==25 and {(r['condition'],r['seed']) for r in diagnosis}==expected
        assert len({r['initial_state_sha256'] for r in diagnosis})==5
        for seed in p['diagnostic_seeds']:
            assert len({r['initial_state_sha256'] for r in diagnosis if r['seed']==seed})==1
        for r in diagnosis:
            assert r['candidate']==first_pass
            name=f"{r['condition']}_{r['seed']}"
            assert r==w.s.read(w.OUT/'diagnosis'/(name+'.json'))
            counts['diagnosis/'+name+'.npz']=verify_episode(w.OUT/'diagnosis'/(name+'.npz'),r,p)
    else:
        assert not (w.OUT/'diagnosis').exists()
        assert len(groups)==2
    outcome=w.s.read(w.OUT/'outcome.json')
    assert outcome['selected']==first_pass and outcome['diagnosis_run']==(diagnosis is not None)
    for name,sha in design['prior_artifacts_sha256'].items():assert w.s.digest(project_file(w.s.ROOT, name))==sha,name
    proof={'audit_passed':True,'goal_requirements_achieved':first_pass is not None and diagnosis is not None,
        'selected':first_pass,'healthy_episodes':sum(len(v) for v in groups.values()),
        'diagnostic_episodes':0 if diagnosis is None else len(diagnosis),'physics_states_checked':sum(counts.values()),
        'datasets':counts,'prior_artifacts_unchanged':len(design['prior_artifacts_sha256']),
        'independent_mission_checks':True,'fault_latch_and_torque_verified':True,
        'joint_torque_limits_verified':True,'cpu':True,'training':False,'rendering':False}
    target=w.OUT/'completion_audit.json'
    if target.exists():assert w.s.read(target)==proof
    else:w.s.write(target,proof)
    lines=[]
    for candidate,rows in groups.items():
        lines.append(f"| {candidate} | healthy pilot | {sum(r['success'] for r in rows)}/20 | {sum(r['fallen'] for r in rows)} |")
    if diagnosis:
        for condition in ['healthy',*[leg+'_calf' for leg in w.LEGS]]:
            rows=[r for r in diagnosis if r['condition']==condition]
            lines.append(f"| {first_pass} | {condition} diagnosis | {sum(r['success'] for r in rows)}/5 | {sum(r['fallen'] for r in rows)} |")
    report=f"""# Short walking mission baseline

**Healthy gate: {'PASS' if first_pass else 'FAIL'}. Diagnosis: {'25 episodes completed' if diagnosis else 'not run because no baseline passed'}. No locomotion or adaptation training was run.

| Candidate | Stage/condition | Mission successes | Falls |
|---|---|---:|---:|
{chr(10).join(lines)}

## Frozen mission

Move 2 m in the initial heading on the unchanged flat Menagerie Go2 model. Maximum planar command 0.20 m/s, 2 s initial settle, 0.5 s command ramp, 20 s episode. A mission must complete the whole episode without a fall/nonfinite, end within 0.15 m of goal and 15 degrees of heading, never exceed 0.20 m cross-track, and end with at least 2 continuous seconds inside the goal/heading bounds with planar speed <=0.05 m/s. A temporary stop followed by drift is insufficient. Falls retain the old height<0.16 m / tilt>45 degrees bounds.

Faults latch on the first physics interval whose pre-state forward progress is >=0.50 m. Only the named knee actuator's torque is multiplied by 0.10 after all torque limits. Fault mission success also requires the fault actually occurred before entering the goal. Healthy episodes never apply a fault. Initial poses are paired across diagnostic conditions by hash.

Healthy seeds 20000-20019 and diagnostic seeds 20100-20104 were checked against recorded prior seeds and frozen before rollout. The healthy cohort is deliberately paired if candidate 2 is attempted; it is not presented as a second fresh held-out test. Selection uses only the healthy pilot. Every attempted episode, including failures, is retained. Criteria and source gains were not tuned after results.

## Source and calibration

Only two candidates were inspected: Go2 RL Lab at commit {p['candidates']['go2_rl_lab']['commit']} and Genesis/Go2 sim-to-real deployment at commit {p['candidates']['genesis']['commit']}. Source URLs, pinned commits, license paths, checksums and reviewed contract hashes are in protocol.json. Licenses and unmodified source contracts are retained next to the new weights. Lab has a combined MIT/Apache repository notice; Genesis has the repository BSD-3-Clause notice. There is no separate checkpoint-specific license declaration; this audit records repository notices rather than asserting additional grants.

Lab: 45 ordinary observations, 12 actions, internal 10-frame history with reset, local FL/FR/RL/RR order, default angles from its config, Kp25/Kd0.5, 0.25 action scale, clip100, direction-dependent 13.5-30 rad/s torque-speed envelope. Genesis: 49 observations, 16 actions including per-leg stiffness, FR/FL/RR/RL mapping, ELU 512/256/128 actor reconstructed exactly from actor weights with no extra dependency, Kp20..60, Kd0.2sqrt(Kp), 0.25 scale, clip100 and 0.1rad target slew. Runtime gain multipliers remain 1.0. No critic or optimizer is trained or used.

Both adapters preserve local joint and torque limits and use the existing 0.002 s physics / 0.02 s control. Lab's source physics step is 0.005 s, so this is a disclosed sim-to-sim compatibility pilot, not a claim of identical dynamics. Pose defaults and +/-0.01 rad reset noise are reconciled with local feet by translating root z for 0.002 m initial foot clearance; model parameters are not modified. First control interval holds the default target; deterministic policy inference starts at 0.02 s. Stateful history and previous actions reset for every episode. No motor identity enters either policy observation.

## Audit and evidence

Tests ran before execution; tests-before.log is hashed in execution.json. NPZ traces preserve every physics state, requested/limited/applied torque, actual actuator force, target, gains, contacts, command, and each control observation/raw/clipped action. Per-episode JSON records all checks and failure metrics. Independent audit uses quaternion formulas and trailing-window reductions rather than the runner's summary helper, and verifies the first fault tick, selected-only multiplication, local limits, exact initial hashes, command bounds and torque-speed formula.

All {proof['physics_states_checked']:,} saved physics states and {proof['prior_artifacts_unchanged']} protected prior artifacts were checked. Old checkpoints, experiments and videos retain their hashes. No CUDA, training, rendering, hardware SDK or simulator-stack installation was used. Frozen source/model/weight checksums are rechecked. See completion_audit.json and all raw episode files.

## Decision

{('The baseline passes the healthy mission gate. Diagnosis outcomes above measure whether the same frozen walker requires fault compensation for this mission; failed fault episodes are valid diagnostic findings, not a reason to change the healthy selection or criteria.' if first_pass else 'Neither baseline met the healthy gate. Do not run fault diagnosis or claim this goal achieved. Advancing requires a compatible calibrated walking baseline or separately authorized locomotion work; do not expand the candidate search or train automatically.')}

This is a short flat-floor simulation pilot with small reset noise and one knee weakened at a time. Results do not establish hardware behavior, other failure types, rough terrain or population reliability. No adaptation has been trained. Recheck with `.venv\\Scripts\\python.exe review_walking.py`.
"""
    target=w.OUT/'REPORT.md'
    if target.exists():assert target.read_text(encoding='utf-8')==report
    else:
        with target.open('x',encoding='utf-8') as h:h.write(report)
    print(report.split('## Frozen mission')[0],flush=True)
    print('Independent walking audit complete:',proof['goal_requirements_achieved'],flush=True)
    return proof


if __name__=='__main__':audit()
