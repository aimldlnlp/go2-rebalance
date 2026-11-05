"""Frozen CPU-only short walking missions with at most two pretrained Go2 policies."""
from project_paths import project_file
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from torch import nn

from go2_env import MODEL
import sudden as s

OUT = s.ROOT / 'results/walking_baseline'
BASE = s.ROOT / 'vendor/walking_baselines'
CANDIDATES = ['go2_rl_lab', 'genesis']
LEGS = ['FL','FR','RL','RR']
JOINTS = [f'{leg}_{joint}' for leg in LEGS for joint in ['hip','thigh','calf']]
HEALTHY_SEEDS = list(range(20000,20020))
DIAGNOSTIC_SEEDS = list(range(20100,20105))
DT = .002
CONTROL = 10
TICKS = 10000


def rotation(quat):
    value = np.empty(9)
    mujoco.mju_quat2Mat(value, quat)
    return value.reshape(3,3)


def torque_limit(torque, velocity, candidate, limits):
    if candidate == 'go2_rl_lab':
        # Training motor model: directional torque-speed envelope, plus local physical limits.
        maximum = np.where(velocity*torque > 0, 20.2, 23.4)
        maximum *= np.clip((30-np.abs(velocity))/(30-13.5), 0, 1)
        torque = np.clip(torque,-maximum,maximum)
    return np.clip(torque, limits[:,0], limits[:,1])


class Actor:
    def __init__(self, candidate):
        if candidate not in CANDIDATES:
            raise ValueError('Unknown frozen candidate')
        self.candidate = candidate
        if candidate == 'go2_rl_lab':
            self.path = BASE/candidate/'policy.pt'
            self.policy = torch.jit.load(str(self.path),map_location='cpu').eval()
            self.order = np.arange(12)
            self.default = np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
            self.n_actions = 12
            self.command_scale = np.array([.5,1,1])
        else:
            self.path = BASE/candidate/'walk.pt'
            state = torch.load(self.path,map_location='cpu',weights_only=True)['model_state_dict']
            self.policy = nn.Sequential(nn.Linear(49,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,16))
            self.policy.load_state_dict({key.removeprefix('actor.'): value for key,value in state.items() if key.startswith('actor.')},strict=True)
            self.policy.eval()
            self.order = np.array([3,4,5,0,1,2,9,10,11,6,7,8])
            self.default = np.array([0.,.8,-1.5,0.,.8,-1.5,0.,1.,-1.5,0.,1.,-1.5])
            self.n_actions = 16
            self.command_scale = np.array([2.,2.,.25])
        self.reset()

    def reset(self):
        self.previous = np.zeros(self.n_actions,dtype=np.float32)
        if hasattr(self.policy,'reset'):
            self.policy.reset()

    def observation(self, qpos, qvel, command):
        if np.asarray(command).shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Command must be three finite values')
        r = rotation(qpos[3:7])
        return np.concatenate([qvel[3:6]*.25, r.T@np.array([0.,0.,-1.]),
            command*self.command_scale, qpos[7:][self.order]-self.default,
            qvel[6:][self.order]*.05,self.previous]).astype(np.float32)

    def predict(self, obs):
        if obs.shape != (33+self.n_actions,) or not np.isfinite(obs).all():
            raise ValueError('Invalid observation')
        with torch.inference_mode():
            raw = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().reshape(-1)
        if raw.shape != (self.n_actions,) or not np.isfinite(raw).all():
            raise ValueError('Invalid policy output')
        action = np.clip(raw,-100,100)
        self.previous = action.copy()
        return raw, action


def mission_command(qpos, initial_xy, initial_yaw, tick):
    if tick*DT < 2:
        return np.zeros(3)
    r = rotation(qpos[3:7])
    forward = np.array([np.cos(initial_yaw),np.sin(initial_yaw)])
    goal = initial_xy + 2*forward
    error = goal-qpos[:2]
    distance = np.linalg.norm(error)
    velocity = np.zeros(2) if distance <= .10 else error/distance*min(.2,.8*distance)
    velocity *= min(1.,(tick*DT-2)/.5)
    body = r[:2,:2].T@velocity
    yaw = np.arctan2(r[1,0],r[0,0])
    heading = np.arctan2(np.sin(initial_yaw-yaw),np.cos(initial_yaw-yaw))
    return np.r_[body,np.clip(1.5*heading,-.5,.5)]


def summarize(data, condition, initial_xy, initial_yaw, onset):
    q, v = data['qpos'],data['qvel']
    finite = np.isfinite(q).all() and np.isfinite(v).all()
    if not finite:
        return {'success':False,'fallen':True,'reason':'nonfinite state','simulated_s':(len(q)-1)*DT,'fault_onset_tick':onset}
    forward = np.array([np.cos(initial_yaw),np.sin(initial_yaw)])
    lateral = np.array([-forward[1],forward[0]])
    goal = initial_xy+2*forward
    errors = np.linalg.norm(q[:,:2]-goal,axis=1)
    cross = np.abs((q[:,:2]-initial_xy)@lateral)
    speeds = np.linalg.norm(v[:,:2],axis=1)
    mats = np.asarray([rotation(x[3:7]) for x in q])
    yaw = np.arctan2(mats[:,1,0],mats[:,0,0])
    headings = np.abs(np.arctan2(np.sin(yaw-initial_yaw),np.cos(yaw-initial_yaw)))
    tilt = np.rad2deg(np.arccos(np.clip(mats[:,2,2],-1,1)))
    fallen = bool(np.any(q[:,2]<.16) or np.any(tilt>45))
    stable = (errors<=.15)&(headings<=np.deg2rad(15))&(speeds<=.05)
    hold = 0.
    for i in range(len(stable)-1,-1,-1):
        if not stable[i]:
            break
        hold = (len(stable)-1-i)*DT
    checks = {'full_duration':len(q)==TICKS+1,'no_fall':not fallen,'goal_error':bool(errors[-1]<=.15),
        'heading_error':bool(headings[-1]<=np.deg2rad(15)),'cross_track':bool(cross.max()<=.20),
        'stable_stop':hold>=2.,'fault_received':condition=='healthy' or onset is not None}
    if condition != 'healthy' and onset is not None:
        checks['fault_before_goal'] = bool(np.all(errors[:onset+1]>.15))
    return {'success':all(checks.values()),'checks':checks,'fallen':fallen,'simulated_s':(len(q)-1)*DT,
        'final_goal_error_m':float(errors[-1]),'max_cross_track_m':float(cross.max()),
        'final_heading_error_deg':float(np.rad2deg(headings[-1])),'max_tilt_deg':float(tilt.max()),
        'final_speed_m_s':float(speeds[-1]),'final_stable_hold_s':hold,'fault_onset_tick':onset,
        'final_progress_m':float((q[-1,:2]-initial_xy)@forward)}


def episode(actor, seed, condition='healthy'):
    if condition not in ['healthy',*[f'{leg}_calf' for leg in LEGS]]:
        raise ValueError('Invalid fault condition')
    model = mujoco.MjModel.from_xml_path(str(MODEL))
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
    assert not OUT.exists(), 'Refusing to overwrite experiment'
    used,protected=s.inventory()
    used.update(s.read(s.OUT/'protocol.json')['evaluation_seeds'])
    for path in s.OUT.rglob('*'):
        if path.is_file():protected[str(path.relative_to(s.ROOT))]=s.digest(path)
    assert not set(HEALTHY_SEEDS+DIAGNOSTIC_SEEDS).intersection(used)
    artifacts={}
    for candidate in CANDIDATES:
        inv=json.loads((s.ROOT/f'work/walking_candidates/{candidate}_inventory.json').read_text(encoding='utf-8-sig'))
        folder=s.ROOT/f'work/walking_candidates/{candidate}'
        artifacts[candidate]={'repo':inv['repo'],'commit':inv['commit'],'branch':inv['branch'],
            'license_file':str((BASE/candidate/'LICENSE').relative_to(s.ROOT)),
            'license_sha256':s.digest(BASE/candidate/'LICENSE'),
            'policy_path':str((BASE/candidate/('policy.pt' if candidate=='go2_rl_lab' else 'walk.pt')).relative_to(s.ROOT)),
            'policy_sha256':s.digest(BASE/candidate/('policy.pt' if candidate=='go2_rl_lab' else 'walk.pt')),
            'upstream_contract_sha256':{str(p.relative_to(s.ROOT)):s.digest(p) for p in (BASE/candidate).iterdir() if p.is_file()}}
    OUT.mkdir()
    p={'created_utc':datetime.now(timezone.utc).isoformat(),'candidates':artifacts,'candidate_order':CANDIDATES,
        'healthy_seeds':HEALTHY_SEEDS,'diagnostic_seeds':DIAGNOSTIC_SEEDS,'prior_recorded_seeds':sorted(used),
        'mission':{'distance_m':2.,'max_command_planar_m_s':.2,'settle_s':2.,'command_ramp_s':.5,
            'goal_gain':.8,'goal_command_deadband_m':.10,'heading_gain':1.5,'max_yaw_command_rad_s':.5,
            'deadline_s':20.,'goal_radius_m':.15,'max_heading_deg':15.,'max_cross_track_m':.20,
            'max_stop_speed_m_s':.05,'final_stable_hold_s':2.,'min_height_m':.16,'max_tilt_deg':45.,
            'fault_progress_m':.50,'fault_strength':.1},
        'gate':'First candidate with 20/20 healthy successes and zero falls enables diagnosis; retain every failure. Same predeclared healthy seeds may be paired across candidates, never called fresh on the second candidate. Stop after two candidates; no training.',
        'measurement':'Unchanged model and 0.002 s physics; 0.02 s control. Log pre-interval controller/torque/fault and post-interval state. Success requires full 20s, final goal/heading, global cross-track and >=2 continuous seconds of stable stop ending at deadline. A fall/nonfinite always fails. Fault latches on first pre-step progress>=0.5m and is applied after torque limiting only to selected actuator.',
        'calibration':'Use source policy order/default/scales/action clip, lab Kp25/Kd0.5 and torque-speed curve, Genesis per-leg stiffness20..60/Kd0.2sqrt(Kp)/target slew0.1rad. Retain local model joint/torque limits. Reset pose uses policy default+old +/-0.01 noise; set foot clearance to 0.002m by translating root z, not changing model. First 0.02s holds default target then deterministic inference. Reset history between episodes.',
        'sim2sim_difference':'Lab source uses 0.005s physics, 0.02s control, Newton/implicitfast; this pilot preserves the existing Menagerie model/options and 0.002s physics. This compatibility change is disclosed and measured, not assumed equivalent.',
        'execution':{'device':'cpu','torch_threads':1,'training':False,'cuda':False,'rendering':False},
        'failure_policy':'No threshold, source gain or policy tuning after healthy results. Only demonstrated implementation bugs may be repaired with old evidence preserved and amendment recorded. No third candidate, training, or hardware.'}
    s.write(OUT/'protocol.json',p)
    sources=['walking_baseline.py','review_walking.py','test_walking_baseline.py','go2_env.py','sudden.py','requirements-lock.txt']
    model_hash={str(path.relative_to(s.ROOT)):s.digest(path) for path in MODEL.parent.rglob('*') if path.is_file()}
    s.write(OUT/'design.json',{'protocol_sha256':s.digest(OUT/'protocol.json'),'source_sha256':{name:s.digest(project_file(s.ROOT, name)) for name in sources},'model_sha256':model_hash,'prior_artifacts_sha256':protected})
    print(f'Frozen two candidates, healthy 20000-20019, diagnostic 20100-20104; {len(protected)} prior artifacts',flush=True)


def verify_frozen():
    p=s.read(OUT/'protocol.json');d=s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==d['protocol_sha256']
    for name,sha in {**d['source_sha256'],**d['model_sha256']}.items():assert s.digest(project_file(s.ROOT, name))==sha,name
    for candidate,item in p['candidates'].items():
        assert s.digest(s.ROOT/item['policy_path'])==item['policy_sha256']
        assert s.digest(s.ROOT/item['license_file'])==item['license_sha256']
        for name,sha in item['upstream_contract_sha256'].items():assert s.digest(project_file(s.ROOT, name))==sha,name
    assert not set(HEALTHY_SEEDS+DIAGNOSTIC_SEEDS).intersection(p['prior_recorded_seeds'])
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    return p


def evaluate_group(candidate, seeds, conditions, folder):
    folder.mkdir()
    actor=Actor(candidate)
    results=[]
    for condition in conditions:
        for seed in seeds:
            arrays,row=episode(actor,seed,condition)
            name=f'{condition}_{seed}'
            with (folder/(name+'.npz')).open('xb') as handle:np.savez_compressed(handle,**arrays)
            row['trace_sha256']=s.digest(folder/(name+'.npz'))
            s.write(folder/(name+'.json'),row)
            results.append(row)
            print(f"{candidate}/{name}: success={row['success']} fallen={row['fallen']} progress={row.get('final_progress_m')} stop_hold={row.get('final_stable_hold_s')}",flush=True)
    s.write(folder/'episodes.json',results)
    return results


def run():
    assert not (OUT/'execution.json').exists(),'Refusing to rerun experiment'
    p=verify_frozen()
    log=(OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in log and 'FAILED' not in log
    s.write(OUT/'execution.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'device':'cpu','tests_sha256':s.digest(OUT/'tests-before.log'),'training':False})
    selected=None
    for candidate in p['candidate_order']:
        rows=evaluate_group(candidate,p['healthy_seeds'],['healthy'],OUT/f'healthy_{candidate}')
        if len(rows)==20 and all(r['success'] for r in rows):
            selected=candidate
            break
    s.write(OUT/'selection.json',{'candidate':selected,'healthy_gate_passed':selected is not None,'selection_uses_diagnostic':False})
    if selected is not None:
        evaluate_group(selected,p['diagnostic_seeds'],['healthy',*[leg+'_calf' for leg in LEGS]],OUT/'diagnosis')
    s.write(OUT/'outcome.json',{'healthy_gate_passed':selected is not None,'selected':selected,'diagnosis_run':selected is not None,'eligible_for_completion_audit':selected is not None,'training':False,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('Simulation stages complete; independent audit required',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().stage]()
