"""One source-backed passive-actuator calibration repair; prior failed study stays frozen."""
from project_paths import project_file
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

import walking_baseline as original
from walking_baseline import Actor, rotation, torque_limit, mission_command, summarize, JOINTS, LEGS, DT, CONTROL, TICKS, MODEL
import review_walking as independent
import sudden as s

OUT = s.ROOT / 'results/walking_calibrated'
HEALTHY_SEEDS = list(range(20020,20040))
DIAGNOSTIC_SEEDS = list(range(20100,20105))


def calibrated_model():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joints = model.actuator_trnid[:,0]
    # Match the pinned deployment model's passive joint contract; retain local limits and geometry.
    model.dof_damping[model.jnt_dofadr[joints]] = 0.
    model.dof_frictionloss[model.jnt_dofadr[joints]] = .01
    return model


def episode(actor, seed, condition='healthy'):
    if condition not in ['healthy',*[f'{leg}_calf' for leg in LEGS]]:
        raise ValueError('Invalid fault condition')
    model = calibrated_model()
    assert model.opt.timestep==DT and model.nu==12 and model.nq==19 and model.nv==18
    assert [model.actuator(i).name for i in range(12)]==JOINTS
    np.testing.assert_array_equal(model.jnt_qposadr[model.actuator_trnid[:,0]],np.arange(7,19))
    np.testing.assert_array_equal(model.jnt_dofadr[model.actuator_trnid[:,0]],np.arange(6,18))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    actor.reset()
    default_local = np.empty(12)
    default_local[actor.order] = actor.default
    data.qpos[7:] = default_local+np.random.default_rng(seed).uniform(-.01,.01,12)
    mujoco.mj_forward(model,data)
    feet = np.array([model.geom(leg).id for leg in LEGS])
    # Reconcile the new policy pose with the unchanged foot geometry, avoiding reset penetration.
    data.qpos[2] += .002-min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in feet)
    mujoco.mj_forward(model,data)
    initial_xy = data.qpos[:2].copy()
    initial_yaw = float(np.arctan2(rotation(data.qpos[3:7])[1,0],rotation(data.qpos[3:7])[0,0]))
    initial_hash = s.digest_bytes(data.qpos.tobytes()+data.qvel.tobytes())
    arrays = {'qpos':np.empty((TICKS+1,19)),'qvel':np.empty((TICKS+1,18)),
        'requested_torque':np.empty((TICKS,12)),'limited_torque':np.empty((TICKS,12)),
        'applied_torque':np.empty((TICKS,12)),'actuator_force':np.empty((TICKS,12)),
        'strength':np.ones(TICKS),'command':np.empty((TICKS,3)),
        'target':np.empty((TICKS,12)),'kp':np.empty((TICKS,12)),'kd':np.empty((TICKS,12)),
        'contacts':np.zeros((TICKS,4),dtype=bool),
        'control_observations':np.empty((1000,33+actor.n_actions),dtype=np.float32),
        'control_raw_actions':np.empty((1000,actor.n_actions),dtype=np.float32),
        'control_actions':np.empty((1000,actor.n_actions),dtype=np.float32)}
    arrays['qpos'][0]=data.qpos
    arrays['qvel'][0]=data.qvel
    target = default_local.copy()
    kp = np.full(12,25. if actor.candidate=='go2_rl_lab' else 40.)
    kd = np.full(12,.5 if actor.candidate=='go2_rl_lab' else 2.)
    command = np.zeros(3)
    onset = None
    weak = None if condition=='healthy' else model.actuator(condition).id
    rng_joint = model.jnt_range[model.actuator_trnid[:,0]]
    controls = 0
    started = time.perf_counter()
    for tick in range(TICKS):
        if weak is not None and onset is None and (data.qpos[:2]-initial_xy)@np.array([np.cos(initial_yaw),np.sin(initial_yaw)])>=.5:
            onset = tick
        if tick%CONTROL==0:
            command = mission_command(data.qpos,initial_xy,initial_yaw,tick)
            obs = actor.observation(data.qpos,data.qvel,command)
            if tick==0:
                raw = action = np.zeros(actor.n_actions,dtype=np.float32)
            else:
                raw,action = actor.predict(obs)
                desired = np.empty(12)
                desired[actor.order] = actor.default+.25*action[:12]
                if actor.candidate=='genesis':
                    desired = target+np.clip(desired-target,-.1,.1)
                    stiffness = np.repeat(np.clip(40+20*action[12:],20,60),3)
                    kp[actor.order] = stiffness
                    kd[actor.order] = .2*np.sqrt(stiffness)
                target = np.clip(desired,rng_joint[:,0],rng_joint[:,1])
            arrays['control_observations'][controls]=obs
            arrays['control_raw_actions'][controls]=raw
            arrays['control_actions'][controls]=action
            controls+=1
        requested = kp*(target-data.qpos[7:])-kd*data.qvel[6:]
        limited = torque_limit(requested,data.qvel[6:],actor.candidate,model.actuator_ctrlrange)
        applied = limited.copy()
        if onset is not None:
            applied[weak]*=.1
        arrays['requested_torque'][tick]=requested
        arrays['limited_torque'][tick]=limited
        arrays['applied_torque'][tick]=applied
        arrays['strength'][tick]=.1 if onset is not None else 1.
        arrays['command'][tick]=command
        arrays['target'][tick]=target
        arrays['kp'][tick]=kp
        arrays['kd'][tick]=kd
        data.ctrl[:]=applied
        mujoco.mj_step(model,data)
        arrays['actuator_force'][tick]=data.actuator_force
        for contact in data.contact:
            for foot_id,geom_id in enumerate(feet):
                if geom_id in (contact.geom1,contact.geom2):
                    arrays['contacts'][tick,foot_id]=True
        arrays['qpos'][tick+1]=data.qpos
        arrays['qvel'][tick+1]=data.qvel
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2]<.16:
            break
        r=rotation(data.qpos[3:7])
        if np.rad2deg(np.arccos(np.clip(r[2,2],-1,1)))>45:
            break
    count=tick+1
    arrays={k:v[:controls] if k.startswith('control_') else v[:count+1] if k in ('qpos','qvel') else v[:count] for k,v in arrays.items()}
    result={'seed':seed,'candidate':actor.candidate,'condition':condition,'initial_state_sha256':initial_hash,
        'initial_xy':initial_xy.tolist(),'initial_yaw':initial_yaw,'wall_s':time.perf_counter()-started,
        **summarize(arrays,condition,initial_xy,initial_yaw,onset)}
    return arrays,result


def prepare():
    assert not OUT.exists(),'Refusing to overwrite calibration study'
    old=original.verify_frozen()
    assert not s.read(original.OUT/'outcome.json')['healthy_gate_passed']
    assert not s.read(original.OUT/'outcome.json')['diagnosis_run']
    source_model=s.ROOT/'work/walking_candidates/go2_rl_lab/source_model.xml'
    import xml.etree.ElementTree as ET
    joint=ET.parse(source_model).getroot().find('default/joint')
    assert float(joint.attrib['damping'])==0 and float(joint.attrib['frictionloss'])==.01
    assert float(joint.attrib['armature'])==.01
    used=set(old['prior_recorded_seeds'])|set(old['healthy_seeds'])
    assert not set(HEALTHY_SEEDS).intersection(used)
    assert not set(DIAGNOSTIC_SEEDS).intersection(used)
    prior=dict(s.read(original.OUT/'design.json')['prior_artifacts_sha256'])
    for path in original.OUT.rglob('*'):
        if path.is_file():prior[str(path.relative_to(s.ROOT))]=s.digest(path)
    for name,sha in prior.items():assert s.digest(project_file(s.ROOT, name))==sha,name
    OUT.mkdir()
    p=dict(old)
    p.update({'created_utc':datetime.now(timezone.utc).isoformat(),'candidate_order':['go2_rl_lab'],
        'healthy_seeds':HEALTHY_SEEDS,'diagnostic_seeds':DIAGNOSTIC_SEEDS,'prior_recorded_seeds':sorted(used),
        'repair':{'reason':'Initial adapter missed upstream passive joint damping/friction; one direct source-contract repair, no parameter search.',
            'upstream_model_path':str(source_model.relative_to(s.ROOT)),'upstream_model_sha256':s.digest(source_model),
            'original_damping':2.,'corrected_damping':0.,'original_frictionloss':.2,'corrected_frictionloss':.01,
            'armature':.01,'physics_dt':.002,'control_dt':.02,
            'preserved':'Geometry, mass, contact parameters, joint/torque limits, command mapping, source gains, policy weights, mission thresholds and fault multiplication unchanged.'},
        'parent_protocol_sha256':s.digest(original.OUT/'protocol.json'),
        'gate':'One repaired-contract pilot only, 20/20 healthy required; then the previously declared but unused diagnosis seeds. Stop if this repair fails. No extra policy, tuning or training.'})
    s.write(OUT/'protocol.json',p)
    sources=['walking_calibrated.py','test_walking_calibrated.py','walking_baseline.py','review_walking.py','test_walking_baseline.py','go2_env.py','sudden.py','requirements-lock.txt']
    s.write(OUT/'design.json',{'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(s.ROOT, name)) for name in sources},
        'model_sha256':s.read(original.OUT/'design.json')['model_sha256'],'prior_artifacts_sha256':prior})
    print(f'Frozen source-contract repair: healthy 20020-20039, diagnosis 20100-20104; {len(prior)} protected files',flush=True)


def verify_frozen():
    p=s.read(OUT/'protocol.json');d=s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==d['protocol_sha256']
    for name,sha in {**d['source_sha256'],**d['model_sha256']}.items():assert s.digest(project_file(s.ROOT, name))==sha,name
    assert s.digest(s.ROOT/p['repair']['upstream_model_path'])==p['repair']['upstream_model_sha256']
    original.verify_frozen()
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    return p


def evaluate_group(seeds,conditions,folder):
    folder.mkdir();actor=Actor('go2_rl_lab');rows=[]
    for condition in conditions:
        for seed in seeds:
            data,row=episode(actor,seed,condition)
            name=f'{condition}_{seed}'
            with (folder/(name+'.npz')).open('xb') as handle:np.savez_compressed(handle,**data)
            row['trace_sha256']=s.digest(folder/(name+'.npz'))
            s.write(folder/(name+'.json'),row);rows.append(row)
            print(f"repaired/{name}: success={row['success']} fallen={row['fallen']} progress={row.get('final_progress_m')} hold={row.get('final_stable_hold_s')}",flush=True)
    s.write(folder/'episodes.json',rows)
    return rows


def run():
    assert not (OUT/'execution.json').exists(),'Refusing to rerun calibration pilot'
    p=verify_frozen()
    tests=(OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','torch':torch.__version__,'cuda_build':torch.version.cuda,'tests_sha256':s.digest(OUT/'tests-before.log'),'training':False})
    healthy=evaluate_group(p['healthy_seeds'],['healthy'],OUT/'healthy')
    passed=all(r['success'] for r in healthy)
    s.write(OUT/'selection.json',{'candidate':'go2_rl_lab' if passed else None,'healthy_gate_passed':passed})
    if passed:evaluate_group(p['diagnostic_seeds'],['healthy',*[leg+'_calf' for leg in LEGS]],OUT/'diagnosis')
    s.write(OUT/'outcome.json',{'healthy_gate_passed':passed,'diagnosis_run':passed,'training':False,'finished_utc':datetime.now(timezone.utc).isoformat()})
    audit()


def audit():
    p=verify_frozen();design=s.read(OUT/'design.json')
    rows=s.read(OUT/'healthy/episodes.json')
    assert len(rows)==20 and [r['seed'] for r in rows]==p['healthy_seeds']
    counts={}
    for row in rows:
        stem=f"{row['condition']}_{row['seed']}"
        assert row==s.read(OUT/'healthy'/(stem+'.json'))
        counts['healthy/'+stem]=independent.verify_episode(OUT/'healthy'/(stem+'.npz'),row,p)
    passed=all(r['success'] for r in rows)
    selection=s.read(OUT/'selection.json')
    assert selection['healthy_gate_passed']==passed
    assert selection['candidate']==('go2_rl_lab' if passed else None)
    diagnosis=[]
    if passed:
        diagnosis=s.read(OUT/'diagnosis/episodes.json')
        expected={(c,seed) for c in ['healthy',*[leg+'_calf' for leg in LEGS]] for seed in p['diagnostic_seeds']}
        assert len(diagnosis)==25 and {(r['condition'],r['seed']) for r in diagnosis}==expected
        for seed in p['diagnostic_seeds']:
            assert len({r['initial_state_sha256'] for r in diagnosis if r['seed']==seed})==1
        for row in diagnosis:
            stem=f"{row['condition']}_{row['seed']}"
            assert row==s.read(OUT/'diagnosis'/(stem+'.json'))
            counts['diagnosis/'+stem]=independent.verify_episode(OUT/'diagnosis'/(stem+'.npz'),row,p)
    else:assert not (OUT/'diagnosis').exists()
    for name,sha in design['prior_artifacts_sha256'].items():assert s.digest(project_file(s.ROOT, name))==sha,name
    execution=s.read(OUT/'execution.json')
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(execution['started_utc'])
    assert execution['tests_sha256']==s.digest(OUT/'tests-before.log')
    model=calibrated_model()
    np.testing.assert_array_equal(model.dof_damping[6:],np.zeros(12))
    np.testing.assert_array_equal(model.dof_frictionloss[6:],np.full(12,.01))
    proof={'audit_passed':True,'goal_requirements_achieved':passed and len(diagnosis)==25,
        'healthy_successes':sum(r['success'] for r in rows),'healthy_episodes':20,
        'diagnostic_episodes':len(diagnosis),'physics_states_checked':sum(counts.values()),
        'prior_artifacts_unchanged':len(design['prior_artifacts_sha256']),
        'independent_mission_and_fault_checks':True,'calibration_matched_to_pinned_source':True,
        'old_joint_torque_limits_preserved':True,'cpu':True,'training':False,'cuda':False,'rendering':False}
    target=OUT/'completion_audit.json'
    if target.exists():assert s.read(target)==proof
    else:s.write(target,proof)
    table=[f"| healthy pilot | {sum(r['success'] for r in rows)}/20 | {sum(r['fallen'] for r in rows)} |"]
    for condition in ['healthy',*[leg+'_calf' for leg in LEGS]]:
        group=[r for r in diagnosis if r['condition']==condition]
        if group:table.append(f"| {condition} diagnosis | {sum(r['success'] for r in group)}/5 | {sum(r['fallen'] for r in group)} |")
    report=f"""# Source-calibrated Go2 walking baseline

**Healthy gate: {'PASS' if passed else 'FAIL'}. Goal requirements: {'achieved' if proof['goal_requirements_achieved'] else 'not achieved'}. No training or third pretrained candidate was used.

| Stage/condition | Mission successes | Falls |
|---|---:|---:|
{chr(10).join(table)}

## Why this separate repair exists

Both original candidates failed 0/20 healthy missions without falling. Those results remain frozen under ../walking_baseline/. Post-pilot review found a missing passive-actuator contract in the first adapter: pinned Go2 RL Lab deployment XML uses damping 0 and frictionloss 0.01; the standing Menagerie model uses damping 2 and frictionloss 0.2. This separate run makes that one source-backed correction on an in-memory model copy. It does not search gains, alter acceptance thresholds, or replace the simulator. Geometry, masses, contacts, armature, physical joint/torque limits, policy parameters, observation/action scaling, source PD gains, navigation and progress-triggered fault remain unchanged. The physical passive dynamics now differ from the earlier standing experiments; direct performance comparisons must acknowledge this.

Source: {p['candidates']['go2_rl_lab']['repo']} at {p['candidates']['go2_rl_lab']['commit']}, upstream deployment XML SHA-256 {p['repair']['upstream_model_sha256']}. The existing weight checksum, repository license notice and complete contracts are retained in vendor/walking_baselines/go2_rl_lab/. No source deployment/hardware script was executed. CPU inference, 45 inputs/12 outputs and reset of internal 10-frame history were checked. Local physics stays 0.002s and control 0.02s; source deployment physics is 0.005s, so this remains a measured sim-to-sim pilot, not identical dynamics.

## Unchanged mission and gate

Goal is 2 m forward relative to initial heading, command speed <=0.20m/s, 2s settle, 0.5s initial ramp and a 20s deadline. Full duration, no fall/nonfinite, final position error <=0.15m, heading error <=15 degrees, cross-track <=0.20m throughout, and >=2 continuous final seconds within goal/heading bounds at speed <=0.05m/s are all mandatory. Fall bounds remain height<0.16m or tilt>45 degrees. Faults latch at first pre-step progress>=0.50m and multiply only the selected knee's limited torque by 0.10. A fault success must actually receive the fault before reaching goal. No fault identity is an actor input.

New healthy seeds 20020-20039 are disjoint from the failed 20000-20019 pilot and all prior recorded starts. Diagnosis seeds 20100-20104 were declared earlier but were demonstrably never run because both original gates failed; they are retained as unused reserved diagnostic starts. Only 20/20 healthy success opens all 25 diagnostic episodes. All failures are preserved; thresholds and source calibration were frozen before the new pilot. No further calibration/search is permitted in this revision.

## Evidence

Protocol/design hashes, source and model hashes, tests-before.log and execution timestamps were checked. Every saved qpos/qvel, command, target/gain, raw/clipped action, requested/limited/applied torque, actual actuator force and contact trace remains available in NPZ. An independent reviewer computes mission checks from quaternion formulas and exact trailing windows, verifies fault-onset membership, single-actuator multiplication and the torque-speed/physical limits. {proof['physics_states_checked']:,} physics states and {proof['prior_artifacts_unchanged']} protected earlier artifacts passed these checks. Weights, original models and the earlier failed study retain their hashes.

{('Diagnosis results answer whether this unadapted source-calibrated walker completes the frozen fault missions. A failed diagnostic is retained as evidence for a later adaptation proposal; it does not invalidate healthy selection.' if passed else 'The one corrected-contract pilot still fails the healthy gate. Do not run diagnosis or claim goal completion. Further progress needs an explicitly amended scope, a suitable baseline/calibration, or changed requirements; no automatic third candidate, training or tuning is authorized.')}

This is a small reset-noise, flat-floor simulation assessment. It does not establish population reliability, hardware readiness, other fault strengths, other joints, multiple faults or rough terrain. Recheck with `.venv\\Scripts\\python.exe walking_calibrated.py audit`.
"""
    target=OUT/'REPORT.md'
    if target.exists():assert target.read_text(encoding='utf-8')==report
    else:
        with target.open('x',encoding='utf-8') as handle:handle.write(report)
    print(report.split('## Why')[0],flush=True)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
