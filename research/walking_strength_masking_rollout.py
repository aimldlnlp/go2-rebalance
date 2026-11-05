"""One paired CPU rollout study; reuse the frozen privileged evaluator."""
from project_paths import project_file, source_files
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO
import walking_residual_privileged_pilot as parent
import walking_residual_privileged_env as original
import walking_strength_masking as offline
import sudden as s

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/walking_strength_masking_rollout'
SOURCE = ROOT/'results/walking_residual_privileged_pilot'


class MaskedEnv(original.WalkingResidualEnv):
    def _observation(self):
        return offline.mask(super()._observation())


def prepare():
    assert not OUT.exists(), 'Never overwrite a study'
    assert 'OK' in (ROOT/'work/strength_masking_rollout_tests.log').read_text(encoding='utf-8')
    snapshot = s.read(ROOT/'results/walking_strength_masking/protocol.json')
    for name, sha in snapshot['input_hashes'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    paths = [p for p in (ROOT/'results').rglob('*') if p.is_file()]
    paths += list(source_files(ROOT))
    paths += [ROOT/'work/strength_masking_rollout_tests.log']
    paths += list((ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*.xml'))
    paths += [ROOT/'vendor/walking_baselines/go2_rl_lab/policy.pt']
    hashes = {}
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = s.digest(path)
        if len(hashes)%500 == 0:
            print(f'Protected hashes: {len(hashes)}', flush=True)
    checkpoint = SOURCE/'checkpoint_131072.zip'
    sha = s.digest(checkpoint)
    assert sha == s.read(SOURCE/'checkpoint_131072.json')['checkpoint_sha256']
    p = s.read(SOURCE/'protocol.json')
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does current strength information help closed-loop mission performance for this checkpoint?',
        checkpoint_sha256=sha, checkpoint_steps=131072,
        conditions=original.CONDITIONS, validation_seeds=list(range(23000,23005)),
        arm_order=['factual_131072','masked_131072'], condition_order=original.CONDITIONS,
        new_episodes=50, regression_factual_episodes=25, intervention_masked_episodes=25,
        wall_time_cap_s=300, device='cpu', threads=1,
        intervention='Only observation[60:72] set to 1; first 60 from each arm own trajectory/history',
        intervention_timing='First control observation after actual latch; before latch and healthy identical',
        physical_fault='0.1 after torque limiting, immutable',
        previous_residual='Applied action from own arm, as in original environment',
        primary=['mission_success','fallen'], secondary=['net_progress_after_onset','final_error','stable_hold','return_descriptive_only'],
        controls='Factual must exactly replay all historical trace arrays before masked arm starts; healthy exact full parity; fault exact pre-intervention prefix',
        stop=['hash mismatch','historical parity failure','prefix failure','nonfinite','300 second budget'],
        historic_mission_protocol=p, protected_input_hashes=hashes,
        training=False, final_seeds=False, additional_training_seeds=False,
        interpretation='Within-checkpoint information intervention on repeated development poses. No generalization or training-information causal claim.')
    OUT.mkdir()
    s.write(OUT/'protocol.json', protocol)
    shutil.copyfile(checkpoint, OUT/'checkpoint_131072.zip')
    shutil.copyfile(ROOT/'work/strength_masking_rollout_tests.log', OUT/'tests-before.log')
    print(f'Frozen {len(hashes)} protected inputs; 50 episodes, 300 s cap', flush=True)


def verify():
    p = s.read(OUT/'protocol.json')
    for name, sha in p['protected_input_hashes'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    assert s.digest(OUT/'checkpoint_131072.zip') == p['checkpoint_sha256']
    return p


def parity(folder, reference, rows, prefix=False):
    checked = []
    for row in rows:
        stem = f"{row['condition']}_{row['seed']}"
        oldrow = s.read(reference/(stem+'.json'))
        assert row['initial_state_sha256'] == oldrow['initial_state_sha256']
        with np.load(folder/(stem+'.npz'), allow_pickle=False) as a, np.load(reference/(stem+'.npz'), allow_pickle=False) as b:
            aa, bb = {k:a[k] for k in a.files}, {k:b[k] for k in b.files}
        assert aa.keys() == bb.keys()
        onset = oldrow['fault_onset_tick']
        intervention = None if onset is None else (onset//10+1)*10
        if prefix and intervention is not None:
            assert row['fault_onset_tick'] == onset
            for key in bb:
                if key == 'reward_component_names':
                    np.testing.assert_array_equal(aa[key], bb[key])
                elif key in ['qpos','qvel']:
                    np.testing.assert_array_equal(aa[key][:intervention+1], bb[key][:intervention+1], err_msg=key)
                elif key.startswith('control_') or key in ['residual_observations','residual_policy_actions','reward_components']:
                    np.testing.assert_array_equal(aa[key][:intervention//10], bb[key][:intervention//10], err_msg=key)
                else:
                    np.testing.assert_array_equal(aa[key][:intervention], bb[key][:intervention], err_msg=key)
        else:
            for key in bb:
                np.testing.assert_array_equal(aa[key], bb[key], err_msg=key)
            for key in ['success','fallen','fault_onset_tick','final_progress_m','final_stable_hold_s','episode_return']:
                assert row[key] == oldrow[key], key
        checked.append(dict(condition=row['condition'],seed=row['seed'],intervention_tick=intervention,
                            full_exact=not prefix or intervention is None))
    return checked


def run():
    assert not (OUT/'execution.json').exists() and not (OUT/'factual_131072').exists()
    p = verify()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    deadline = started+p['wall_time_cap_s']
    result = dict(completed=False,training=False,device='cpu',error=None,
                  factual_historical_parity=None,paired_prefix_parity=None)
    parent.OUT = OUT
    try:
        model = PPO.load(OUT/'checkpoint_131072.zip', device='cpu')
        parent.finite_policy(model)
        assert model.num_timesteps == 131072
        factual = parent.evaluate(model,'factual_131072',p,deadline)
        result['factual_historical_parity'] = parity(OUT/'factual_131072',SOURCE/'validation_131072',factual)
        s.write(OUT/'historical_parity.json', result['factual_historical_parity'])
        parent.budget_check(deadline)
        parent.w = SimpleNamespace(WalkingResidualEnv=MaskedEnv, CONDITIONS=original.CONDITIONS,
                                   REWARD_WEIGHTS=original.REWARD_WEIGHTS)
        masked = parent.evaluate(model,'masked_131072',p,deadline)
        result['paired_prefix_parity'] = parity(OUT/'masked_131072',OUT/'factual_131072',masked,True)
        parent.budget_check(deadline)
        result['completed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['compute_wall_s'] = time.perf_counter()-started
        s.write(OUT/'execution.json',result)


def audit():
    import review_walking as physics
    from review_walking_residual_height_pilot import audit_reward_and_nominal
    from walking_baseline import rotation, JOINTS
    started = time.perf_counter()
    p = verify()
    execution = s.read(OUT/'execution.json')
    assert execution['completed'] and execution['error'] is None and execution['compute_wall_s'] <= 300
    torch.set_num_threads(1)
    model = PPO.load(OUT/'checkpoint_131072.zip', device='cpu')
    states, observations = 0, 0
    rows_by_arm = {}
    # The physical auditor accesses each array repeatedly: cache within this open archive only.
    with np.load(OUT/'factual_131072/healthy_23000.npz', allow_pickle=False) as z:
        kind = type(z)
    getitem, close = kind.__getitem__, kind.close
    def cached(self,key):
        if not hasattr(self,'_mask_cache'):
            self._mask_cache = {}
        if key not in self._mask_cache:
            value = getitem(self,key)
            value.setflags(write=False)
            self._mask_cache[key] = value
        return self._mask_cache[key]
    def closed(self):
        self._mask_cache = {}
        return close(self)
    kind.__getitem__, kind.close = cached, closed
    try:
        for arm in p['arm_order']:
            rows = s.read(OUT/arm/'episodes.json')
            assert len(rows) == 25 and len({(r['condition'],r['seed']) for r in rows}) == 25
            rows_by_arm[arm] = rows
            for row in rows:
                assert not row['interrupted'] and row['seed'] in p['validation_seeds']
                stem = f"{row['condition']}_{row['seed']}"
                path = OUT/arm/(stem+'.npz')
                assert s.digest(path) == row['trace_sha256']
                assert row == s.read(OUT/arm/(stem+'.json'))
                assert row['residual_checkpoint_sha256'] == p['checkpoint_sha256']
                states += physics.verify_episode(path,row,p['historic_mission_protocol'])
                with np.load(path,allow_pickle=False) as a:
                    assert a['qpos'] is a['qpos'] and not a['qpos'].flags.writeable
                    q,v = a['qpos'],a['qvel']
                    actions, inputs = a['control_residual_actions'],a['residual_observations']
                    nominal = offline.DEFAULT+.25*a['control_actions']
                    reconstructed = []
                    for i,tick in enumerate(range(0,len(q)-1,10)):
                        r = rotation(q[tick,3:7])
                        previous = np.zeros(12) if i == 0 else actions[i-1]
                        strength = np.ones(12)
                        if arm == 'factual_131072' and row['fault_onset_tick'] is not None and tick > row['fault_onset_tick']:
                            strength[JOINTS.index(row['condition'])] = .1
                        reconstructed.append(np.r_[r.T@v[tick,:3],v[tick,3:6],r.T@np.array([0.,0.,-1.]),
                            a['command'][tick],q[tick,7:]-offline.DEFAULT,v[tick,6:]*.05,nominal[i]-offline.DEFAULT,previous,strength])
                    np.testing.assert_array_equal(inputs,np.asarray(reconstructed,dtype=np.float32))
                    predicted = np.asarray([model.predict(o,deterministic=True)[0] for o in inputs])
                    np.testing.assert_array_equal(predicted,a['residual_policy_actions'])
                    np.testing.assert_array_equal(actions,np.clip(predicted,-1,1))
                    np.testing.assert_allclose(row['residual_effort'],np.mean(actions**2),rtol=0,atol=1e-12)
                    ranges = np.array([[-1.0472,1.0472],[-1.5708,3.4907],[-2.7227,-.83776]]*2+
                                      [[-1.0472,1.0472],[-.5236,4.5379],[-2.7227,-.83776]]*2)
                    target = np.clip(nominal+.1*actions,ranges[:,0],ranges[:,1])
                    np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:len(q)-1])
                    audit_reward_and_nominal(a,row)
                    observations += len(inputs)
                print(f'Audited {arm}/{stem}',flush=True)
    finally:
        kind.__getitem__,kind.close = getitem,close
    factual,masked = rows_by_arm.values()
    parity(OUT/'factual_131072',SOURCE/'validation_131072',factual)
    parity(OUT/'masked_131072',OUT/'factual_131072',masked,True)
    verify()
    pairs = []
    for f,m in zip(factual,masked,strict=True):
        assert (f['condition'],f['seed']) == (m['condition'],m['seed'])
        pairs.append(dict(condition=f['condition'],seed=f['seed'],factual_success=f['success'],masked_success=m['success'],
            factual_fallen=f['fallen'],masked_fallen=m['fallen'],
            progress_delta_m=m['final_progress_m']-f['final_progress_m'],
            hold_delta_s=m['final_stable_hold_s']-f['final_stable_hold_s']))
    summary = []
    for condition in p['conditions']:
        item = dict(condition=condition)
        for arm,rows in rows_by_arm.items():
            group = [r for r in rows if r['condition'] == condition]
            net = []
            for row in group:
                with np.load(OUT/arm/f"{condition}_{row['seed']}.npz",allow_pickle=False) as a:
                    onset = row['fault_onset_tick']
                    if onset is not None:
                        direction = np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
                        net.append(float((a['qpos'][-1,:2]-a['qpos'][onset,:2])@direction))
            item[arm] = dict(success=sum(r['success'] for r in group),falls=sum(r['fallen'] for r in group),
                median_net_postfault_progress_m=None if not net else float(np.median(net)),
                median_final_error_m=float(np.median([r['final_goal_error_m'] for r in group])),
                median_hold_s=float(np.median([r['final_stable_hold_s'] for r in group])))
        group = [r for r in pairs if r['condition']==condition]
        item['masked_minus_factual'] = dict(success_rescue=sum(not r['factual_success'] and r['masked_success'] for r in group),
            success_harm=sum(r['factual_success'] and not r['masked_success'] for r in group),
            fall_rescue=sum(r['factual_fallen'] and not r['masked_fallen'] for r in group),
            fall_harm=sum(not r['factual_fallen'] and r['masked_fallen'] for r in group),
            median_paired_progress_delta_m=float(np.median([r['progress_delta_m'] for r in group])))
        summary.append(item)
    s.write(OUT/'pairs.json',pairs)
    s.write(OUT/'summary.json',summary)
    s.write(OUT/'audit.json',dict(passed=True,episodes=50,physics_states=states,policy_observations=observations,
        historical_exact_episodes=25,paired_exact_prefixes=20,healthy_exact_pairs=5,
        protected_inputs_unchanged=len(p['protected_input_hashes']),audit_wall_s=time.perf_counter()-started,
        readonly_audit=True,additional_rollouts=0,training=False))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','audit'])
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'audit':audit}[args.command]()
