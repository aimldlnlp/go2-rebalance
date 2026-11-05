"""One bounded CPU PPO pilot over the already verified residual environment."""
from project_paths import project_file, source_files
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import traceback

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

import sudden as s
import verify_walking_residual_env as verified
import walking_residual_height_env as w
import walking_residual_height_pilot as parent


ROOT = s.ROOT
OUT = ROOT/'results/walking_residual_low_noise_pilot'
LOG_STD_INIT = float(np.log(.2))
CONFIRMATION_SEEDS = list(range(23300, 23305))


class BudgetExpired(RuntimeError):
    pass


def budget_check(deadline):
    if time.perf_counter() >= deadline:
        raise BudgetExpired('Pilot wall-time budget reached')


def finite_policy(model):
    if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
        raise FloatingPointError('Nonfinite residual policy parameters')


@contextmanager
def preserved_rng():
    python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.random.set_rng_state(torch_state)


class TrainingPool(gym.Wrapper):
    """SB3 seed controls PPO; this wrapper uses only the declared pose seed pool."""
    def __init__(self, env, seeds, log_path):
        super().__init__(env)
        self.seeds, self.log_path = list(seeds), Path(log_path)
        self.cursor, self.total_steps, self.active = 0, 0, False
        with self.log_path.open('x', encoding='utf-8'):
            pass

    def log(self, event, **values):
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'event':event, **values})+'\n')

    def reset(self, *, seed=None, options=None):
        if self.active:
            self.log('partial', **self.episode, control_steps=self.episode_steps, episode_return=self.episode_return)
        pose_seed = self.seeds[self.cursor % len(self.seeds)]
        obs, info = self.env.reset(seed=pose_seed, options={'condition':'mixed'})
        self.episode = {'episode_index':self.cursor, 'pose_seed':pose_seed,
            'condition':self.unwrapped.condition, 'start_training_step':self.total_steps,
            'initial_state_sha256':s.digest_bytes(self.unwrapped.data.qpos.tobytes()+self.unwrapped.data.qvel.tobytes())}
        self.cursor += 1
        self.episode_steps, self.episode_return, self.active = 0, 0., True
        self.log('reset', **self.episode)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_return += reward
        if terminated or truncated:
            self.log('episode_end', **self.episode, control_steps=self.episode_steps,
                episode_return=self.episode_return, result=info['episode_result'])
            self.active = False
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.active:
            self.log('partial', **self.episode, control_steps=self.episode_steps, episode_return=self.episode_return)
            self.active = False
        self.env.close()


class Guard(BaseCallback):
    def __init__(self, deadline, max_steps):
        super().__init__()
        self.deadline, self.max_steps = deadline, max_steps

    def _on_step(self):
        budget_check(self.deadline)
        if self.num_timesteps > self.max_steps:
            raise RuntimeError('Training step budget exceeded')
        for key in ['new_obs', 'rewards', 'actions']:
            if not np.isfinite(self.locals[key]).all():
                raise FloatingPointError('Nonfinite training '+key)
        for info in self.locals['infos']:
            if info.get('episode_result', {}).get('reason') == 'nonfinite state':
                raise FloatingPointError('Nonfinite simulator state')
        return True


def promotion(rows):
    if len(rows) != 25 or {(r['condition'],r['seed']) for r in rows} != {(c,seed) for c in w.CONDITIONS for seed in range(23000,23005)}:
        return False
    if any(r.get('interrupted') or r['fallen'] for r in rows):
        return False
    counts = {c:sum(r['success'] for r in rows if r['condition']==c) for c in w.CONDITIONS}
    return counts['healthy'] == 5 and all(counts[c] >= 3 for c in w.CONDITIONS[1:])


def selection_key(steps, rows):
    counts = [sum(r['success'] for r in rows if r['condition']==c) for c in w.CONDITIONS[1:]]
    return min(counts), sum(r['success'] for r in rows), -float(np.mean([r['residual_effort'] for r in rows])), -steps


def check_protocol(p, previous):
    metadata = {'created_utc','status','parent_protocol_sha256','diagnosis_sha256',
                'only_controller_change','initialization','parity','scope'}
    for key, value in previous.items():
        if key in metadata:
            continue
        expected = {**value, 'log_std_init':LOG_STD_INIT} if key=='ppo_fixed_defaults' else value
        assert p[key] == expected, key
    assert p['ppo_fixed_defaults']['log_std_init'] == LOG_STD_INIT
    assert p['height_reference_m'] == .338


def policy_record(model):
    shared = b''.join(name.encode()+b'\0'+value.detach().cpu().numpy().tobytes()
        for name,value in sorted(model.policy.state_dict().items()) if name!='log_std')
    return {'training_steps':model.num_timesteps,'ppo_epoch_updates':model._n_updates,
        'shared_non_log_std_parameters_sha256':s.digest_bytes(shared),
        'log_std':model.policy.log_std.detach().cpu().numpy().tolist(),
        'sigma':model.policy.log_std.detach().cpu().exp().numpy().tolist(),
        'sigma_learnable':bool(model.policy.log_std.requires_grad)}


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite low-noise pilot'
    previous = parent.verify_sources()
    from diagnose_walking_fault import digest
    noise = ROOT/'results/walking_noise_ablation'
    assert s.read(noise/'completion_audit.json')['lower_noise_supported_for_fixed_healthy_cohort']
    protected = dict(s.read(noise/'design.json')['prior_artifacts_sha256'])
    for name,sha in protected.items():
        assert digest(project_file(ROOT, name))==sha,name
    for path in [*noise.rglob('*'),*source_files(ROOT)]:
        if path.is_file() and path.name not in ['walking_residual_low_noise_pilot.py',
                'review_walking_residual_low_noise_pilot.py','test_walking_residual_low_noise_pilot.py']:
            protected[str(path.relative_to(ROOT))] = digest(path)
    for seed in CONFIRMATION_SEEDS:
        assert not any((ROOT/'results').rglob(f'*_{seed}.npz')), f'Confirmation seed already used: {seed}'
    protocol = {**previous,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'One-change low-initial-exploration pilot, frozen before training',
        'parent_protocol_sha256':s.digest(parent.OUT/'protocol.json'),
        'calibration_audit_sha256':s.digest(noise/'completion_audit.json'),
        'ppo_fixed_defaults':{**previous['ppo_fixed_defaults'],'log_std_init':LOG_STD_INIT},
        'only_controller_change':'Only PPO log_std_init changes: 0 to log(0.2). Environment/controller and all other learning parameters unchanged from height-0.338 pilot.',
        'initialization':'Fresh PPO seed 0. Initial normalized sigma 0.2 (joint target noise sigma 0.02rad before clipping). Sigma remains trainable; no fixed noise temperature, schedule or std clamp.',
        'parity':'Reuse all 45 exact historical zero-residual regressions after rechecking archived hashes and arrays. Environment implementation is unchanged; no new physics replay needed for a PPO initialization-only change.',
        'scope':'One fresh training seed, 131072 steps or 1800s including evaluations. Same development split and conditional confirmation as parent; no repeats or final test.',
        'initial_sigma_normalized':.2,'initial_target_noise_sigma_rad':.02}
    check_protocol(protocol,previous)
    OUT.mkdir()
    s.write(OUT/'protocol.json',protocol)
    names=[*s.read(parent.OUT/'design.json')['source_sha256'],
           'walking_residual_low_noise_pilot.py','review_walking_residual_low_noise_pilot.py',
           'test_walking_residual_low_noise_pilot.py']
    s.write(OUT/'design.json',{'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in dict.fromkeys(names)},
        'prior_artifacts_sha256':protected})
    print(f'Frozen seed-0 sigma-0.2 pilot: 131072 steps/1800s; {len(protected)} files protected',flush=True)


def preflight():
    verify_sources()
    assert not (OUT/'execution.json').exists()
    assert not (OUT/'parity.json').exists()
    from review_walking_residual_height_pilot import audit_preflight
    audit_preflight(s.read(parent.OUT/'protocol.json'),parent.OUT)
    evidence = s.read(parent.OUT/'parity.json')
    s.write(OUT/'parity.json',{**evidence,'reused_from':str((parent.OUT/'parity.json').relative_to(ROOT)),
        'source_sha256':s.digest(parent.OUT/'parity.json'),
        'new_simulation_episodes':0,'reason':'Unchanged environment; only PPO initialization changes.'})
    for name,sha in s.read(OUT/'design.json')['prior_artifacts_sha256'].items():
        assert s.digest(project_file(ROOT, name))==sha,name
    print('Preflight: all 45 archived regressions rechecked; no new parity rollouts',flush=True)


def verify_sources():
    previous = parent.verify_sources()
    p,d = s.read(OUT/'protocol.json'),s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==d['protocol_sha256']
    for name,sha in d['source_sha256'].items():
        assert s.digest(project_file(ROOT, name))==sha,name
    check_protocol(p,previous)
    assert p['parent_protocol_sha256']==s.digest(parent.OUT/'protocol.json')
    assert p['calibration_audit_sha256']==s.digest(ROOT/'results/walking_noise_ablation/completion_audit.json')
    return p


def evaluate(model, label, p, deadline):
    folder = OUT/label
    folder.mkdir()
    env = w.WalkingResidualEnv('healthy', record_trace=True)
    checkpoint_steps = p['checkpoint_steps'] if label == 'confirmation' else label.rsplit('_',1)[-1]
    checkpoint_sha = None if model is None else s.digest(OUT/f"checkpoint_{checkpoint_steps}.zip")
    rows = []
    try:
        for condition in w.CONDITIONS:
            for seed in p['validation_seeds']:
                budget_check(deadline)
                obs, _ = env.reset(seed=seed, options={'condition':condition})
                initial_hash = s.digest_bytes(env.data.qpos.tobytes()+env.data.qvel.tobytes())
                observations, actions, rewards, components = [], [], [], []
                interrupted, pending_error = False, None
                try:
                    done = False
                    while not done:
                        budget_check(deadline)
                        action = np.zeros(12, dtype=np.float32) if model is None else model.predict(obs, deterministic=True)[0]
                        if not np.isfinite(action).all():
                            raise FloatingPointError('Nonfinite validation action')
                        observations.append(obs.copy())
                        actions.append(action.copy())
                        obs, reward, terminated, truncated, info = env.step(action)
                        rewards.append(reward)
                        components.append([info['reward_terms'].get(key,0.) for key in w.REWARD_WEIGHTS])
                        if not np.isfinite(obs).all() or not np.isfinite(reward) or info.get('episode_result',{}).get('reason')=='nonfinite state':
                            raise FloatingPointError('Nonfinite validation state/reward')
                        done = terminated or truncated
                except Exception as exc:
                    interrupted, pending_error = True, exc
                arrays = env.trace()
                arrays.update({'residual_observations':np.asarray(observations, dtype=np.float32),
                    'residual_policy_actions':np.asarray(actions, dtype=np.float32), 'control_rewards':np.asarray(rewards),
                    'reward_components':np.asarray(components), 'reward_component_names':np.asarray(list(w.REWARD_WEIGHTS))})
                stem = f'{condition}_{seed}'
                path = folder/(stem+'.npz')
                with path.open('xb') as handle:
                    np.savez_compressed(handle, **arrays)
                result = env.mission_result()
                if interrupted:
                    result['success'] = False
                row = {'seed':seed,'condition':condition,'candidate':'go2_rl_lab',
                    'initial_xy':env.origin.tolist(),'initial_yaw':env.initial_yaw,
                    'initial_state_sha256':initial_hash, 'trace_sha256':s.digest(path),
                    'controller':label,'residual_checkpoint_sha256':checkpoint_sha,
                    'interrupted':interrupted, 'residual_effort':float(np.mean(np.square(arrays['control_residual_actions']))) if rewards else 0.,
                    'episode_return':float(sum(rewards)), **result}
                s.write(folder/(stem+'.json'), row)
                rows.append(row)
                print(f"{label}/{stem}: success={row['success']} fall={row['fallen']} progress={row.get('final_progress_m')} interrupted={interrupted}", flush=True)
                if interrupted:
                    s.write(folder/'episodes.json', rows)
                    raise pending_error
    except BaseException:
        if not (folder/'episodes.json').exists():
            s.write(folder/'episodes.json', rows)
        raise
    finally:
        env.close()
    s.write(folder/'episodes.json', rows)
    return rows


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun pilot'
    p = verify_sources()
    parity = s.read(OUT/'parity.json')
    assert parity['all_passed'] and len(parity['episodes']) == 45
    tests = (OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json', {'started_utc':datetime.now(timezone.utc).isoformat(),
        'tests_sha256':s.digest(OUT/'tests-before.log'), 'parity_sha256':s.digest(OUT/'parity.json'), 'device':'cpu','torch':torch.__version__,'sb3':__import__('stable_baselines3').__version__})
    started = time.perf_counter()
    deadline = started+p['wall_time_cap_s']
    model, training, pool = None, None, None
    completed, reason, error = [], 'step_budget_completed', None
    try:
        pool = TrainingPool(w.WalkingResidualEnv('mixed'), p['reset_seed_pool'], OUT/'training_episodes.jsonl')
        training = Monitor(pool, str(OUT/'training.monitor.csv'))
        hp = p['hyperparameters']
        model = PPO('MlpPolicy', training, device='cpu', seed=p['training_seed'],
            n_steps=hp['n_steps'], batch_size=hp['batch_size'], n_epochs=hp['n_epochs'],
            learning_rate=hp['learning_rate'], gamma=hp['gamma'], gae_lambda=hp['gae_lambda'],
            policy_kwargs={'net_arch':hp['net_arch'], 'log_std_init':p['ppo_fixed_defaults']['log_std_init']},
            clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=.5,
            normalize_advantage=True, use_sde=False, verbose=1)
        initial = policy_record(model)
        assert initial['training_steps']==0 and initial['ppo_epoch_updates']==0 and initial['sigma_learnable']
        np.testing.assert_allclose(initial['sigma'],.2,rtol=0,atol=2e-8)
        s.write(OUT/'initial_policy.json',initial)
        model.set_logger(configure(str(OUT), ['stdout','csv']))
        def before_update(optimizer, args, kwargs):
            budget_check(deadline)
            for parameter in model.policy.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite PPO gradient')
        def after_update(optimizer, args, kwargs):
            finite_policy(model)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite PPO optimizer state')
            budget_check(deadline)
        model.policy.optimizer.register_step_pre_hook(before_update)
        model.policy.optimizer.register_step_post_hook(after_update)
        with preserved_rng():
            evaluate(None, 'baseline', p, deadline)
        for checkpoint in p['checkpoints']:
            budget_check(deadline)
            print(f'Training seed 0 to {checkpoint} steps; elapsed {time.perf_counter()-started:.1f}s', flush=True)
            model.learn(total_timesteps=checkpoint-model.num_timesteps, reset_num_timesteps=False,
                callback=Guard(deadline, p['max_training_control_steps']), log_interval=8)
            finite_policy(model)
            assert model.num_timesteps == checkpoint
            path = OUT/f'checkpoint_{checkpoint}.zip'
            model.save(path)
            s.write(OUT/f'checkpoint_{checkpoint}.json', {'training_steps':checkpoint,
                'checkpoint_sha256':s.digest(path),'ppo_epoch_updates':model._n_updates,
                'elapsed_s':time.perf_counter()-started,'finite_parameters':True,
                'sigma':model.policy.log_std.detach().cpu().exp().numpy().tolist()})
            with preserved_rng():
                loaded = PPO.load(path, device='cpu')
                probe_env = w.WalkingResidualEnv('healthy')
                try:
                    probe, _ = probe_env.reset(seed=p['validation_seeds'][0])
                    np.testing.assert_array_equal(model.predict(probe,deterministic=True)[0],loaded.predict(probe,deterministic=True)[0])
                finally:
                    probe_env.close()
                rows = evaluate(loaded, f'validation_{checkpoint}', p, deadline)
            completed.append(checkpoint)
            s.write(OUT/f'acceptance_{checkpoint}.json', {'training_steps':checkpoint,'passed':promotion(rows),
                'condition_successes':{c:sum(r['success'] for r in rows if r['condition']==c) for c in w.CONDITIONS},
                'falls':sum(r['fallen'] for r in rows),'selection_key':list(selection_key(checkpoint,rows))})
        eligible = [n for n in completed if s.read(OUT/f'acceptance_{n}.json')['passed']]
        if eligible:
            candidate = max(eligible,key=lambda n:tuple(s.read(OUT/f'acceptance_{n}.json')['selection_key']))
            with preserved_rng():
                loaded = PPO.load(OUT/f'checkpoint_{candidate}.zip', device='cpu')
                rows = evaluate(loaded, 'confirmation', {**p, 'validation_seeds':p['confirmation_seeds'], 'checkpoint_steps':candidate}, deadline)
            from review_walking_residual_height_pilot import actual_gate
            s.write(OUT/'confirmation_acceptance.json', {'checkpoint_steps':candidate,
                'passed':actual_gate(rows,p['confirmation_seeds'])})
    except BudgetExpired as exc:
        reason, error = 'wall_budget_reached', str(exc)
    except Exception as exc:
        reason, error = 'aborted_error_or_nonfinite', str(exc)
        s.write(OUT/'error.json', {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
    finally:
        if reason != 'step_budget_completed' and model is not None:
            model.save(OUT/'partial_checkpoint.zip')
            data = pool.unwrapped.data
            with (OUT/'diagnostic_state.npz').open('xb') as handle:
                np.savez_compressed(handle,qpos=data.qpos,qvel=data.qvel,previous_residual=pool.unwrapped.previous)
        if training is not None:
            training.close()
    eligible = [n for n in completed if s.read(OUT/f'acceptance_{n}.json')['passed']]
    selected = max(eligible,key=lambda n:tuple(s.read(OUT/f'acceptance_{n}.json')['selection_key'])) if eligible else None
    development_selected = selected
    confirmation = s.read(OUT/'confirmation_acceptance.json') if (OUT/'confirmation_acceptance.json').exists() else None
    if confirmation is None or not confirmation['passed']:
        selected = None
    outcome = {'development_selected_checkpoint':development_selected, 'confirmation_run':(OUT/'confirmation').exists(),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'stop_reason':reason,'error':error,
        'training_executed':model is not None and model.num_timesteps>0,
        'training_steps':model.num_timesteps if model is not None else 0,
        'compute_wall_s':time.perf_counter()-started,'completed_checkpoint_validations':completed,
        'selected_checkpoint':selected,'pilot_promoted':selected is not None,'final_test_run':False,'replicate_training_run':False}
    s.write(OUT/'outcome.json',outcome)
    import review_walking_residual_low_noise_pilot
    review_walking_residual_low_noise_pilot.audit()
    if reason == 'aborted_error_or_nonfinite':
        raise RuntimeError('Pilot aborted; preserved diagnostics and report: '+str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preflight','run','audit'])
    args = parser.parse_args()
    if args.stage == 'audit':
        import review_walking_residual_low_noise_pilot
        review_walking_residual_low_noise_pilot.audit()
    else:
        {'prepare':prepare,'preflight':preflight,'run':run}[args.stage]()
