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


ROOT = s.ROOT
OUT = ROOT/'results/walking_residual_height_pilot'
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


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite height pilot'
    import walking_residual_ablation as ablation
    ablation.verify_sources()
    assert s.read(ablation.OUT/'completion_audit.json')['audit_passed']
    old = ROOT/'results/walking_residual_pilot'
    protocol = s.read(old/'protocol.json')
    used = set(s.read(ablation.OUT/'protocol.json')['prior_seed_membership'])
    used.update(s.read(ablation.OUT/'protocol.json')['seeds'])
    used.update(protocol['reset_seed_pool'] + protocol['validation_seeds'] + protocol['reserved_later_test_seeds'])
    assert not set(CONFIRMATION_SEEDS).intersection(used)
    for seed in CONFIRMATION_SEEDS:
        assert not any((ROOT/'results').rglob(f'*_{seed}.npz')), f'Seed {seed} already has a trace'
    protected = dict(s.read(ablation.OUT/'design.json')['prior_artifacts_sha256'])
    for name, sha in protected.items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    for path in (ROOT/'results').rglob('*'):
        if path.is_file():
            protected[str(path.relative_to(ROOT))] = s.digest(path)
    old_sources = s.read(ablation.OUT/'design.json')['source_sha256']
    for name in old_sources:
        protected[name] = s.digest(project_file(ROOT, name))
    for path in [*source_files(ROOT), ROOT/'README.md', ROOT/'requirements.txt', ROOT/'requirements-lock.txt']:
        if 'residual_height' not in path.name:
            protected[path.name] = s.digest(path)
    protocol.update({
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'Height-reference-only pilot; frozen before parity and training',
        'parent_protocol_sha256':s.digest(old/'protocol.json'),
        'diagnosis_sha256':s.digest(ablation.OUT/'diagnosis.json'),
        'height_reference_m':w.HEIGHT_REFERENCE,
        'only_controller_change':'Reward height reference 0.270 to 0.338 m; no command gate or other change.',
        'initialization':'Fresh PPO seed 0; never load a residual checkpoint for training.',
        'development_split':'Reuse 23000-23004 at all three checkpoints for paired comparison with historical pilot; development data only.',
        'confirmation_seeds':CONFIRMATION_SEEDS,
        'confirmation_rule':'Only if a checkpoint passes development, evaluate the selected checkpoint once on all five conditions at fresh 23300-23304. Both development and confirmation must pass the same gate. No fallback candidate, retries or tuning.',
        'maximum_confirmation_control_steps':25000,
        'confirmation_budget':'Included in the same 1800s wall cap. Interrupted confirmation cannot promote.',
        'parity':'Replay all 45 historical zero-residual fixtures before training. Exact arrays/flags/onset; scalar metrics atol 1e-12. Historical regression, not held-out evidence.',
        'scope':'One seed, same PPO/configuration/training pose pool; three development checkpoints and conditional confirmation. Final 23100-23119 untouched.',
        'preview_task':'Never activate preview task or server.'})
    OUT.mkdir()
    s.write(OUT/'protocol.json', protocol)
    names = [*old_sources, 'walking_residual_height_env.py', 'walking_residual_height_pilot.py',
             'review_walking_residual_height_pilot.py', 'test_walking_residual_height_pilot.py',
             'explain_walking_residual_ablation.py']
    s.write(OUT/'design.json', {'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in dict.fromkeys(names)},
        'prior_artifacts_sha256':protected})
    print(f'Frozen height pilot: .338m, seed 0, 131072 steps/1800s; {len(protected)} old files protected', flush=True)


def preflight():
    p = verify_sources()
    assert not (OUT/'execution.json').exists(), 'Parity must precede training'
    assert not (OUT/'parity.json').exists(), 'Refusing to overwrite parity'
    # Reuse the frozen regression implementation with this variant and a new output directory.
    old_out, old_env = verified.OUT, verified.residual
    try:
        verified.OUT, verified.residual = OUT, w
        verified.parity(s.read(old_out/'protocol.json'))
    finally:
        verified.OUT, verified.residual = old_out, old_env
    assert s.read(OUT/'parity.json')['all_passed']
    for name, sha in s.read(OUT/'design.json')['prior_artifacts_sha256'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    print('Preflight complete: 45 exact zero-residual regressions; prior files unchanged', flush=True)


def verify_sources():
    verified.verify_sources()
    p, d = s.read(OUT/'protocol.json'), s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json') == d['protocol_sha256']
    for name, sha in d['source_sha256'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    assert p['height_reference_m'] == w.HEIGHT_REFERENCE == .338
    assert p['reward_weights'] == w.REWARD_WEIGHTS and p['residual_scale_rad'] == w.RESIDUAL_SCALE
    old = s.read(ROOT/'results/walking_residual_pilot/protocol.json')
    for key in ['training_seed', 'reset_seed_pool', 'hyperparameters', 'ppo_fixed_defaults',
                'max_training_control_steps', 'wall_time_cap_s', 'checkpoints', 'validation_seeds',
                'reserved_later_test_seeds', 'mission_acceptance', 'reward_weights', 'residual_scale_rad']:
        assert p[key] == old[key], key
    assert p['confirmation_seeds'] == CONFIRMATION_SEEDS
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
            policy_kwargs={'net_arch':hp['net_arch'], 'log_std_init':0.},
            clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=.5,
            normalize_advantage=True, use_sde=False, verbose=1)
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
                'elapsed_s':time.perf_counter()-started,'finite_parameters':True})
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
    import review_walking_residual_height_pilot
    review_walking_residual_height_pilot.audit()
    if reason == 'aborted_error_or_nonfinite':
        raise RuntimeError('Pilot aborted; preserved diagnostics and report: '+str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preflight','run','audit'])
    args = parser.parse_args()
    if args.stage == 'audit':
        import review_walking_residual_height_pilot
        review_walking_residual_height_pilot.audit()
    else:
        {'prepare':prepare,'preflight':preflight,'run':run}[args.stage]()
