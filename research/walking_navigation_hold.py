"""Latched-stop correction after verified stop-boundary command chatter."""
from project_paths import project_file
import argparse
from datetime import datetime, timezone
import inspect
import time

import mujoco
import numpy as np
import torch

import sudden as s
import walking_baseline as original
import walking_calibrated as parent
import review_walking as reviewer
from walking_baseline import Actor, rotation, torque_limit, summarize, JOINTS, LEGS, DT, CONTROL, TICKS, MODEL
from walking_calibrated import calibrated_model

OUT = s.ROOT / 'results/walking_navigation_hold'
DEVELOPMENT = [20223, *range(20400, 20404)]
CONFIRMATION = list(range(20420, 20440))
DIAGNOSIS = list(range(20500, 20505))


def mission_command(qpos, initial_xy, initial_yaw, tick):
    if tick * DT < 2:
        return np.zeros(3)
    r = rotation(qpos[3:7])
    goal = initial_xy + 2 * np.array([np.cos(initial_yaw), np.sin(initial_yaw)])
    error = goal - qpos[:2]
    distance = np.linalg.norm(error)
    # Less early slowdown, then stop inside the unchanged 0.15m acceptance area.
    velocity = np.zeros(2) if distance <= .12 else error/distance * min(.2, 1.2*distance)
    velocity *= min(1., (tick*DT-2)/.5)
    body = r[:2, :2].T @ velocity
    yaw = np.arctan2(r[1, 0], r[0, 0])
    heading = np.arctan2(np.sin(initial_yaw-yaw), np.cos(initial_yaw-yaw))
    return np.r_[body, np.clip(1.5*heading, -.5, .5)]


class Navigator:
    def __init__(self):
        self.stopped = False

    def __call__(self, qpos, initial_xy, initial_yaw, tick):
        command = mission_command(qpos, initial_xy, initial_yaw, tick)
        goal = initial_xy + 2*np.array([np.cos(initial_yaw), np.sin(initial_yaw)])
        if tick*DT >= 2 and np.linalg.norm(goal-qpos[:2]) <= .12:
            self.stopped = True
        if self.stopped:
            command[:2] = 0.
        return command


def expected_episode_source():
    return inspect.getsource(parent.episode).replace(
        '    model = calibrated_model()\n', '    navigator = Navigator()\n    model = calibrated_model()\n').replace(
        'command = mission_command(data.qpos,initial_xy,initial_yaw,tick)',
        'command = navigator(data.qpos,initial_xy,initial_yaw,tick)')


def episode(actor, seed, condition='healthy'):
    if condition not in ['healthy',*[f'{leg}_calf' for leg in LEGS]]:
        raise ValueError('Invalid fault condition')
    navigator = Navigator()
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
            command = navigator(data.qpos,initial_xy,initial_yaw,tick)
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
    assert not OUT.exists(), 'Refusing to overwrite navigation study'
    previous = parent.verify_frozen()
    used, protected = s.inventory()
    for name, sha in s.read(parent.OUT/'design.json')['prior_artifacts_sha256'].items():
        assert s.digest(project_file(s.ROOT, name)) == sha, name
        protected[name] = sha
    # Explicitly protect all new walking evidence, including the previous stop analysis.
    for folder in ['walking_baseline', 'walking_calibrated', 'walking_navigation']:
        for path in (s.ROOT/'results'/folder).rglob('*'):
            if path.is_file():
                protected[str(path.relative_to(s.ROOT))] = s.digest(path)
    used |= set(original.HEALTHY_SEEDS + parent.HEALTHY_SEEDS)
    splits = DEVELOPMENT + CONFIRMATION + DIAGNOSIS
    assert len(set(splits)) == 30 and 20223 in used
    assert not set(DEVELOPMENT[1:] + CONFIRMATION + DIAGNOSIS).intersection(used)
    assert inspect.getsource(episode) == expected_episode_source()
    failed_trace = s.ROOT/'results/walking_navigation/confirmation/healthy_20223.npz'
    with np.load(failed_trace, allow_pickle=False) as a:
        active = np.linalg.norm(a['command'][:, :2], axis=1) > 0
        restarts = int(np.count_nonzero(np.diff(active.astype(int)) == 1)-1)
    assert restarts == 4
    OUT.mkdir()
    protocol = dict(previous)
    protocol.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User approved navigation slowdown/stop study, same policy and thresholds, no training, new confirmation seeds.',
        'known_regression_seed': 20223, 'development_seeds': DEVELOPMENT, 'healthy_seeds': CONFIRMATION,
        'repair_evidence': {'trace': str(failed_trace.relative_to(s.ROOT)),
            'trace_sha256': s.digest(failed_trace), 'planar_command_restarts_after_stop': restarts},
        'diagnostic_seeds': DIAGNOSIS, 'prior_recorded_seeds': sorted(used),
        'navigation': {'approach_gain': 1.2, 'zero_command_radius_m': .12, 'latch_planar_stop': True,
            'maximum_planar_command_m_s': .2, 'settle_s': 2., 'ramp_s': .5,
            'heading_gain': 1.5, 'maximum_yaw_command_rad_s': .5},
        'gate': 'One stop-latch repair, frozen before development; seed 20223 is a known regression, not fresh evidence. Require 5/5 development then 20/20 untouched confirmation; only then 25 diagnosis. No further tuning in this revision.',
        'failure_policy': 'Preserve every failure; do not train, add candidates, retune on confirmation, or relax criteria.',
        'parent_protocol_sha256': s.digest(s.ROOT/'results/walking_navigation/protocol.json'),
    })
    s.write(OUT/'protocol.json', protocol)
    sources = ['walking_navigation_hold.py', 'test_walking_navigation_hold.py', 'walking_navigation.py', 'walking_calibrated.py',
        'walking_baseline.py', 'review_walking.py', 'sudden.py', 'go2_env.py', 'requirements-lock.txt']
    s.write(OUT/'design.json', {'protocol_sha256': s.digest(OUT/'protocol.json'),
        'source_sha256': {name: s.digest(project_file(s.ROOT, name)) for name in sources},
        'prior_artifacts_sha256': protected})
    print('Frozen: 5 development, 20 confirmation, 25 conditional diagnostic episodes', flush=True)


def verify_frozen():
    parent.verify_frozen()
    p, d = s.read(OUT/'protocol.json'), s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json') == d['protocol_sha256']
    for name, sha in {**d['source_sha256'], **d['prior_artifacts_sha256']}.items():
        assert s.digest(project_file(s.ROOT, name)) == sha, name
    assert inspect.getsource(episode) == expected_episode_source()
    return p


def evaluate(seeds, conditions, folder):
    folder.mkdir()
    actor, rows = Actor('go2_rl_lab'), []
    for condition in conditions:
        for seed in seeds:
            arrays, row = episode(actor, seed, condition)
            name = f'{condition}_{seed}'
            with (folder/(name+'.npz')).open('xb') as handle:
                np.savez_compressed(handle, **arrays)
            row['trace_sha256'] = s.digest(folder/(name+'.npz'))
            s.write(folder/(name+'.json'), row)
            rows.append(row)
            print(f"{folder.name}/{name}: success={row['success']} fall={row['fallen']} hold={row.get('final_stable_hold_s')}", flush=True)
    s.write(folder/'episodes.json', rows)
    return rows


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun study'
    p = verify_frozen()
    text = (OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in text and 'FAILED' not in text
    s.write(OUT/'execution.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'tests_sha256': s.digest(OUT/'tests-before.log'), 'device': 'cpu', 'training': False})
    dev = evaluate(p['development_seeds'], ['healthy'], OUT/'development')
    dev_passed = all(r['success'] for r in dev)
    healthy, diagnosis = [], []
    if dev_passed:
        healthy = evaluate(p['healthy_seeds'], ['healthy'], OUT/'confirmation')
    passed = len(healthy) == 20 and all(r['success'] for r in healthy)
    if passed:
        diagnosis = evaluate(p['diagnostic_seeds'], ['healthy', *[leg+'_calf' for leg in LEGS]], OUT/'diagnosis')
    s.write(OUT/'outcome.json', {'development_gate_passed': dev_passed,
        'healthy_gate_passed': passed, 'diagnosis_episodes': len(diagnosis),
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'training': False})
    audit()


def audit():
    p = verify_frozen()
    outcome = s.read(OUT/'outcome.json')
    execution = s.read(OUT/'execution.json')
    assert execution['tests_sha256'] == s.digest(OUT/'tests-before.log')
    assert datetime.fromisoformat(p['created_utc']) < datetime.fromisoformat(execution['started_utc'])
    groups, states = {}, 0
    for stage, seeds, conditions in [
        ('development', p['development_seeds'], ['healthy']),
        ('confirmation', p['healthy_seeds'], ['healthy']),
        ('diagnosis', p['diagnostic_seeds'], ['healthy', *[leg+'_calf' for leg in LEGS]])]:
        folder = OUT/stage
        if not folder.exists():
            continue
        rows = s.read(folder/'episodes.json')
        assert len(rows) == len(seeds)*len(conditions)
        assert {(r['seed'], r['condition']) for r in rows} == {(seed, c) for seed in seeds for c in conditions}
        for row in rows:
            assert row['candidate'] == 'go2_rl_lab'
            stem = f"{row['condition']}_{row['seed']}"
            assert row == s.read(folder/(stem+'.json'))
            states += reviewer.verify_episode(folder/(stem+'.npz'), row, p)
            with np.load(folder/(stem+'.npz'), allow_pickle=False) as a:
                # Check navigation separately from the policy and mission-summary audit.
                navigator = Navigator()
                ticks = range(0, len(a['command']), CONTROL)
                expected_commands = np.array([navigator(a['qpos'][tick], np.array(row['initial_xy']), row['initial_yaw'], tick) for tick in ticks])
                np.testing.assert_allclose(a['command'][::CONTROL], expected_commands, rtol=0, atol=1e-12)
        for seed in seeds:
            assert len({r['initial_state_sha256'] for r in rows if r['seed'] == seed}) == 1
        groups[stage] = rows
    assert len(groups['development']) == 5
    dev_passed = all(r['success'] for r in groups['development'])
    assert ('confirmation' in groups) == dev_passed == outcome['development_gate_passed']
    passed = 'confirmation' in groups and all(r['success'] for r in groups['confirmation'])
    assert ('diagnosis' in groups) == passed == outcome['healthy_gate_passed']
    assert len(groups.get('diagnosis', [])) == outcome['diagnosis_episodes']
    proof = {'audit_passed': True, 'goal_requirements_achieved': passed and len(groups.get('diagnosis', [])) == 25,
        'physics_states_checked': states, 'healthy_successes': sum(r['success'] for r in groups.get('confirmation', [])),
        'healthy_episodes': len(groups.get('confirmation', [])), 'diagnostic_episodes': len(groups.get('diagnosis', [])),
        'prior_artifacts_unchanged': len(s.read(OUT/'design.json')['prior_artifacts_sha256']),
        'thresholds_unchanged': p['mission'] == s.read(parent.OUT/'protocol.json')['mission'],
        'training': False, 'cpu': True}
    assert proof['thresholds_unchanged']
    table = ['| Stage / condition | Success | Falls |', '|---|---:|---:|']
    for stage, rows in groups.items():
        for c in dict.fromkeys(r['condition'] for r in rows):
            group = [r for r in rows if r['condition'] == c]
            table.append(f"| {stage} / {c} | {sum(r['success'] for r in group)}/{len(group)} | {sum(r['fallen'] for r in group)} |")
    text = '\n'.join(['# Walking navigation study', '', *table, '',
        f"Goal achieved: {proof['goal_requirements_achieved']}. Independent trace audit passed; {states:,} physics states checked.", '',
        'User-authorized navigation study: first revision achieved 19/20 and remains preserved under walking_navigation. One source-evidenced stop-transition repair now latches planar zero command after entry into the 0.12m stop radius; it never restarts approach in that episode. This prevents observed command chatter from sub-millimeter boundary crossings. If a fault later drifts outside the goal area, the original mission checks still fail. Other navigation changes remain approach gain 0.8 -> 1.2; zero-command radius 0.10 -> 0.12 m. Settle, ramp, speed cap, heading controller, policy, source PD gains, torque envelope, passive calibration, model, fault logic and all mission acceptance thresholds remain unchanged. The target remains 2 m. The stop radius is a controller setting inside the original 0.15 m goal tolerance.', '',
        'Development contains known regression seed 20223 plus four fresh seeds. Confirmation and diagnosis use entirely fresh cohorts frozen before any rollout; no tuning on confirmation. All attempted failures are retained. Healthy confirmation must pass 20/20 before any diagnostic episode. Diagnostic faults receive exactly 10% of limited selected-knee torque at first pre-step progress >=0.50 m and must occur before goal. Policy receives no motor identity. Initial states are paired by hash across diagnostic conditions.', '',
        'Sources, licenses, pinned commits, policy/contract hashes and CPU calibration evidence are inherited unchanged from walking_baseline and walking_calibrated, and checked through their frozen manifests. Full traces, per-episode metrics, tests, timestamps, source hashes and protocol are stored alongside this report. The old failed pilots remain untouched.', '',
        ('Use the condition results to identify where compensation is needed for this frozen short mission; no adaptation was trained.' if passed else 'Healthy gate failed; diagnosis must not run. This revision does not establish fault performance.'), '',
        'Small reset-noise flat-floor simulation only. Passive damping/friction follow source policy calibration and differ from standing experiments. Local 0.002s physics differs from upstream 0.005s. No hardware, rough terrain, multiple faults, or reliability population claims.', '',
        'Recheck: `.venv\\Scripts\\python.exe walking_navigation_hold.py audit`.'])+'\n'
    for name, value in [('completion_audit.json', proof), ('REPORT.md', text)]:
        path = OUT/name
        if path.exists():
            assert (s.read(path) if name.endswith('.json') else path.read_text(encoding='utf-8')) == value
        elif name.endswith('.json'):
            s.write(path, value)
        else:
            with path.open('x', encoding='utf-8') as handle:
                handle.write(value)
    print(text.split('One user-authorized')[0], flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'audit'])
    {'prepare': prepare, 'run': run, 'audit': audit}[parser.parse_args().stage]()
