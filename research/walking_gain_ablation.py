"""Frozen inference-only gain intervention over the low-noise walking checkpoint."""
from project_paths import project_file, source_files
from datetime import datetime, timezone
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO

import sudden as s
import walking_residual_height_env as w
import walking_residual_low_noise_pilot as pilot
import diagnose_walking_fault as d
import review_walking as independent
from review_walking_residual_height_pilot import audit_reward_and_nominal
from walking_baseline import rotation, Actor, MODEL

ROOT = s.ROOT
OUT = ROOT/'results/walking_gain_ablation'
GAINS = [0, 1, 2, 4]
SEEDS = list(range(23000, 23005))
CONDITIONS = w.CONDITIONS[1:]


def choose_action(action, gain, onset):
    action = np.asarray(action, dtype=np.float64)
    if gain not in GAINS or action.shape != (12,) or not np.isfinite(action).all():
        raise ValueError('Invalid gain or action')
    return np.clip(action * (gain if onset is not None else 1), -1, 1)


def primary_direction(success_delta, fall_delta):
    if success_delta > 0 and fall_delta <= 0 or success_delta == 0 and fall_delta < 0:
        return 'better_on_primary'
    if success_delta < 0 and fall_delta >= 0 or success_delta == 0 and fall_delta > 0:
        return 'worse_on_primary'
    return 'unchanged_inconclusive' if success_delta == fall_delta == 0 else 'tradeoff_inconclusive'


def verify_prefix(a, b, first):
    control = first//10
    for key in ['qpos', 'qvel']:
        np.testing.assert_array_equal(a[key][:first+1], b[key][:first+1], err_msg=key)
    for key in ['requested_torque', 'limited_torque', 'applied_torque', 'actuator_force',
                'command', 'target', 'kp', 'kd', 'contacts', 'strength']:
        np.testing.assert_array_equal(a[key][:first], b[key][:first], err_msg=key)
    for key in ['control_observations', 'control_raw_actions', 'control_actions',
                'residual_observations', 'residual_policy_actions']:
        np.testing.assert_array_equal(a[key][:control+1], b[key][:control+1], err_msg=key)
    for key in ['control_residual_actions', 'control_rewards', 'reward_components']:
        np.testing.assert_array_equal(a[key][:control], b[key][:control], err_msg=key)


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite gain ablation'
    parent = pilot.verify_sources()
    comparison = ROOT/'results/walking_low_noise_comparison'
    assert s.read(comparison/'completion_audit.json')['audit_passed']
    assert s.read(pilot.OUT/'completion_audit.json')['audit_passed']
    protected = dict(s.read(comparison/'design.json')['protected_prior_artifacts_sha256'])
    for name, sha in protected.items():
        assert d.digest(project_file(ROOT, name)) == sha, name
    for folder in [comparison, pilot.OUT]:
        for path in folder.rglob('*'):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = d.digest(path)
    for path in source_files(ROOT):
        protected[str(path.relative_to(ROOT))] = d.digest(path)
    checkpoint = pilot.OUT/'checkpoint_131072.zip'
    expected = s.read(pilot.OUT/'checkpoint_131072.json')['checkpoint_sha256']
    assert s.digest(checkpoint) == expected
    nominal = Actor('go2_rl_lab').path
    order = [{'gain': g, 'condition': c, 'seed': seed}
             for g in [1, 0, 2, 4] for c in CONDITIONS for seed in SEEDS]
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=expected,
        checkpoint_steps=131072, gains=GAINS, conditions=CONDITIONS, seeds=SEEDS,
        nominal_policy=str(nominal.relative_to(ROOT)), nominal_policy_sha256=s.digest(nominal),
        physics_model=str(MODEL.relative_to(ROOT)), physics_model_sha256=s.digest(MODEL),
        episodes=80, run_order=order, wall_time_cap_s=600, max_control_steps=80000,
        budget_scope='Environment/model construction, 80 episodes and raw serialization. Preparation, unit fixtures and read-only post-run audit excluded. Stop on budget/nonfinite/error, retain partial evidence; no retries or budget extensions.',
        intervention='At the first scheduled decision after actual physics fault latch, apply clip(gain * deterministic residual, -1, 1) to all 12 joints through episode end. Before latch use clip(residual, -1, 1). The environment records the applied action as previous residual.',
        residual_scale_rad=.10, height_reference_m=.338,
        unchanged='Model, nominal actor, observations, navigator, PD gains, reward, joint/torque limits, fault schedule and mission checks remain frozen. Selected calf receives 10% of limited torque at first pre-physics-step progress >=0.50m.',
        mission_acceptance=parent['mission_acceptance'],
        parity='Gain 1 must match all historical trace arrays exactly. Every other gain must match gain 1 through the first intervention state and all preceding applied actions, rewards, torques, targets, contacts and commands. Inputs and proposals at that decision must also match.',
        primary_metrics=['mission success', 'falls'],
        diagnostic_metrics=['post-onset net/max progress', 'stall', 'tilt', 'residual RMS/bounds', 'torque limiting'],
        interpretation='Compare each gain with gain 1 per condition and paired pose. Better means successes increase without more falls, or tied successes with fewer falls; reverse is worse. Conflicting directions or unchanged primary metrics remain inconclusive. Report all 16 groups and all pairs, never rank by return, never promote or change defaults from diagnostics.',
        aggregation='Five episodes per gain/condition. Episode medians and median paired differences, never pooled timesteps. Whole-post-onset windows end at fall or horizon; fixed 0.5/2s progress requires full cohort availability, otherwise report missing and survivor-only caveat.',
        stall_definition='Existing diagnose_walking_fault: a full 1s window after onset with forward gain <=0.02m and mean command norm >=0.10m/s; diagnostic only.',
        privileged_information='Simulator fault latch is used only for diagnostic intervention timing, not a policy observation or deployable fault detector.',
        reserved_confirmation_seeds=list(range(23300,23305)),
        reserved_final_seeds=list(range(23100,23120)),
        graph_available=(ROOT/'graphify-out/graph.json').exists(),
        training=False, promotion_test=False, rendering=False, cuda=False)
    OUT.mkdir()
    s.write(OUT/'protocol.json', p)
    s.write(OUT/'design.json', dict(protocol_sha256=s.digest(OUT/'protocol.json'),
        source_sha256={name:s.digest(project_file(ROOT, name)) for name in [
            'walking_gain_ablation.py', 'test_walking_gain_ablation.py',
            'report_walking_gain_ablation.py']}, prior_artifacts_sha256=protected))
    print(f'Frozen 80 episodes / 600s; checkpoint {expected}; {len(protected)} protected files', flush=True)


def verify_sources():
    pilot.verify_sources()
    p, design = s.read(OUT/'protocol.json'), s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json') == design['protocol_sha256']
    for name, sha in design['source_sha256'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    assert s.digest(ROOT/p['checkpoint']) == p['checkpoint_sha256']
    for key in ['nominal_policy', 'physics_model']:
        assert s.digest(ROOT/p[key]) == p[key+'_sha256']
    for name, sha in design['prior_artifacts_sha256'].items():
        assert d.digest(project_file(ROOT, name)) == sha, name
    assert p['gains'] == GAINS and p['seeds'] == SEEDS and p['conditions'] == CONDITIONS
    assert p['run_order'] == [{'gain':g,'condition':c,'seed':seed}
                             for g in [1,0,2,4] for c in CONDITIONS for seed in SEEDS]
    assert not set(SEEDS).intersection(p['reserved_confirmation_seeds']+p['reserved_final_seeds'])
    assert torch.version.cuda is None
    return p


def load_policy(p):
    torch.set_num_threads(1)
    model = PPO.load(ROOT/p['checkpoint'], device='cpu')
    assert model.num_timesteps == 131072 and model.device.type == 'cpu'
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    pilot.finite_policy(model)
    return model


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun ablation'
    p = verify_sources()
    tests = (OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        tests_sha256=s.digest(OUT/'tests-before.log'), training=False, device='cpu'))
    started = time.perf_counter()
    deadline = started+p['wall_time_cap_s']
    rows, steps, reason, error, env = [], 0, 'episodes_completed', None, None
    try:
        model = load_policy(p)
        env = w.WalkingResidualEnv('healthy', record_trace=True)
        for case in p['run_order']:
            gain, condition, seed = case['gain'], case['condition'], case['seed']
            pilot.budget_check(deadline)
            folder = OUT/f'gain_{gain}'
            folder.mkdir(exist_ok=True)
            obs, _ = env.reset(seed=seed, options={'condition':condition})
            initial = s.digest_bytes(env.data.qpos.tobytes()+env.data.qvel.tobytes())
            inputs, proposals, chosen, latched, ticks, rewards, components = [], [], [], [], [], [], []
            interrupted, pending, done = False, None, False
            prefix_checked = False
            try:
                while not done:
                    pilot.budget_check(deadline)
                    assert steps < p['max_control_steps']
                    proposal = model.predict(obs, deterministic=True)[0]
                    action = choose_action(proposal, gain, env.onset)
                    inputs.append(obs.copy()); proposals.append(proposal.copy()); chosen.append(action.copy())
                    latched.append(env.onset is not None); ticks.append(env.tick)
                    if gain != 1 and env.onset is not None and not prefix_checked:
                        prefix = env.trace()
                        for key, value in [('control_observations',env.nominal_observation),
                                           ('control_raw_actions',env.raw_action), ('control_actions',env.nominal_action)]:
                            prefix[key] = np.vstack([prefix[key],value])
                        prefix.update(residual_observations=np.asarray(inputs,dtype=np.float32),
                            residual_policy_actions=np.asarray(proposals,dtype=np.float32),
                            control_rewards=np.asarray(rewards), reward_components=np.asarray(components))
                        with np.load(OUT/'gain_1'/f'{condition}_{seed}.npz',allow_pickle=False) as reference:
                            verify_prefix(prefix, reference, env.tick)
                        prefix_checked = True
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    rewards.append(reward)
                    components.append([info['reward_terms'].get(k,0.) for k in w.REWARD_WEIGHTS])
                    if not np.isfinite(obs).all() or not np.isfinite(reward) or info.get('episode_result',{}).get('reason') == 'nonfinite state':
                        raise FloatingPointError('Nonfinite state or reward')
                    done = terminated or truncated
            except Exception as exc:
                interrupted, pending = True, exc
            arrays = env.trace()
            arrays.update(residual_observations=np.asarray(inputs,dtype=np.float32),
                residual_policy_actions=np.asarray(proposals,dtype=np.float32),
                applied_residual_actions=np.asarray(chosen,dtype=np.float64),
                fault_latched_at_decision=np.asarray(latched,dtype=bool),
                control_ticks=np.asarray(ticks,dtype=np.int64), control_rewards=np.asarray(rewards),
                reward_components=np.asarray(components), reward_component_names=np.asarray(list(w.REWARD_WEIGHTS)))
            path = folder/f'{condition}_{seed}.npz'
            with path.open('xb') as handle:
                np.savez_compressed(handle, **arrays)
            result = env.mission_result()
            if interrupted:
                result['success'] = False
            row = dict(**case, candidate='go2_rl_lab', initial_xy=env.origin.tolist(),
                initial_yaw=env.initial_yaw, initial_state_sha256=initial, trace_sha256=s.digest(path),
                checkpoint_sha256=p['checkpoint_sha256'], interrupted=interrupted,
                episode_return=float(sum(rewards)) if np.isfinite(rewards).all() else None,
                first_intervention_tick=next((t for t,l in zip(ticks,latched) if l),None), **result)
            s.write(path.with_suffix('.json'), row)
            rows.append(row)
            print(f"{len(rows)}/80 gain={gain} {condition}/{seed}: success={row['success']} fall={row['fallen']} progress={row.get('final_progress_m')}",flush=True)
            # Stop if the replay control differs before running any intervention cohort.
            if gain == 1 and not interrupted:
                with np.load(pilot.OUT/'validation_131072'/path.name,allow_pickle=False) as reference:
                    for key in reference.files:
                        np.testing.assert_array_equal(arrays[key],reference[key],err_msg=key)
            if pending is not None:
                raise pending
            pilot.budget_check(deadline)
    except pilot.BudgetExpired as exc:
        reason, error = 'wall_budget_reached', str(exc)
    except Exception as exc:
        reason, error = 'aborted_error_or_nonfinite', str(exc)
        s.write(OUT/'error.json', dict(message=str(exc),traceback=traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
    s.write(OUT/'episodes.json', rows)
    s.write(OUT/'outcome.json', dict(finished_utc=datetime.now(timezone.utc).isoformat(),
        stop_reason=reason,error=error,compute_wall_s=time.perf_counter()-started,
        episodes_recorded=len(rows),control_steps=steps,training=False,final_test_run=False))
    audit()
    if error:
        raise RuntimeError(error)


def audit():
    p = verify_sources()
    design, rows = s.read(OUT/'design.json'), s.read(OUT/'episodes.json')
    execution, outcome = s.read(OUT/'execution.json'), s.read(OUT/'outcome.json')
    assert execution['tests_sha256'] == s.digest(OUT/'tests-before.log')
    assert datetime.fromisoformat(p['created_utc']) < datetime.fromisoformat(execution['started_utc'])
    assert [dict(gain=r['gain'],condition=r['condition'],seed=r['seed']) for r in rows] == p['run_order'][:len(rows)]
    assert len(rows) <= 80 and outcome['control_steps'] <= 80000
    model = load_policy(p)
    states, replay, prefixes, checked, diagnostics = 0, 0, 0, 0, []
    for row in rows:
        path = OUT/f"gain_{row['gain']}"/f"{row['condition']}_{row['seed']}.npz"
        assert s.read(path.with_suffix('.json')) == row
        assert s.digest(path) == row['trace_sha256']
        if row['interrupted']:
            continue
        states += independent.verify_episode(path,row,p)
        checked += 1
        with np.load(path,allow_pickle=False) as a:
            actions, inputs, proposals = a['control_residual_actions'], a['residual_observations'], a['residual_policy_actions']
            ticks = np.arange(0,len(a['qpos'])-1,10)
            onset = row['fault_onset_tick']
            assert onset is not None
            mask = ticks > onset
            assert mask.any()
            np.testing.assert_array_equal(a['control_ticks'],ticks)
            np.testing.assert_array_equal(a['fault_latched_at_decision'],mask)
            assert row['first_intervention_tick'] == int(ticks[mask][0])
            expected = proposals.astype(np.float64)
            expected[mask] *= row['gain']
            expected = np.clip(expected,-1,1)
            np.testing.assert_array_equal(actions,expected)
            np.testing.assert_array_equal(a['applied_residual_actions'],expected)
            assert np.max(np.abs(.1*actions)) <= .1
            np.testing.assert_allclose(proposals,np.asarray([model.predict(obs,deterministic=True)[0] for obs in inputs]),rtol=0,atol=1e-7)
            nominal = d.DEFAULT+.25*a['control_actions']
            target = np.clip(nominal+.1*actions,d.RANGES[:,0],d.RANGES[:,1])
            np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:len(a['qpos'])-1])
            reconstructed = []
            for i,tick in enumerate(ticks):
                q,v = a['qpos'][tick],a['qvel'][tick]
                r = rotation(q[3:7])
                previous = np.zeros(12) if i == 0 else actions[i-1]
                reconstructed.append(np.r_[r.T@v[:3],v[3:6],r.T@np.array([0.,0.,-1.]),a['command'][tick],
                    q[7:]-d.DEFAULT,v[6:]*.05,nominal[i]-d.DEFAULT,previous])
            np.testing.assert_array_equal(inputs,np.asarray(reconstructed,dtype=np.float32))
            audit_reward_and_nominal(a,row)
            metrics = d.diagnose(a,row)
            progress = (a['qpos'][:,:2]-row['initial_xy'])@np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
            metrics['fixed_horizon_progress_m'] = {str(seconds):float(progress[onset+round(seconds/d.DT)]-progress[onset])
                if onset+round(seconds/d.DT) < len(progress) else None for seconds in [.5,2.]}
            metrics['post_mean_tilt_deg'] = float(np.mean(d.rotation_values(a['qpos'])[onset+1:]))
            diagnostics.append(dict(gain=row['gain'],condition=row['condition'],seed=row['seed'],**metrics))
            reference = (pilot.OUT/'validation_131072' if row['gain'] == 1 else OUT/'gain_1')/path.name
            metadata = s.read(reference.with_suffix('.json'))
            assert s.digest(reference) == metadata['trace_sha256']
            assert row['initial_state_sha256'] == metadata['initial_state_sha256']
            assert onset == metadata['fault_onset_tick']
            with np.load(reference,allow_pickle=False) as b:
                if row['gain'] == 1:
                    for key in b.files:
                        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                    assert row['checks'] == metadata['checks']
                    replay += 1
                else:
                    first = row['first_intervention_tick']
                    verify_prefix(a, b, first)
                    np.testing.assert_array_equal(a['applied_residual_actions'][:first//10],b['applied_residual_actions'][:first//10])
                    prefixes += 1
    for name,sha in design['prior_artifacts_sha256'].items():
        assert d.digest(project_file(ROOT, name)) == sha, name
    complete = len(rows) == checked == 80 and outcome['stop_reason'] == 'episodes_completed'
    if complete:
        assert replay == 20 and prefixes == 60
    proof = dict(audit_passed=True,study_complete=complete,complete_episodes_checked=checked,
        physics_states_checked=states,historical_exact_replay_pairs=replay,
        exact_preintervention_pairs=prefixes,protected_prior_artifacts_unchanged=len(design['prior_artifacts_sha256']),
        protocol_frozen_before_run=True,checkpoint_unchanged=True,previous_applied_actions_verified=True,
        mission_reward_nominal_torque_independently_verified=True,final_confirmation_seeds_unused=True,
        training=False,rendering=False,cuda=False)
    for name,value in [('diagnostics.json',diagnostics),('completion_audit.json',proof)]:
        path = OUT/name
        if path.exists():
            assert s.read(path) == value
        else:
            s.write(path,value)
    print(f'Audit: {checked} episodes, {replay} historical pairs, {prefixes} intervention prefixes',flush=True)
    return proof


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
