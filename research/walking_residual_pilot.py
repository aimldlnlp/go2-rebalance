"""One bounded CPU PPO pilot over the already verified residual environment."""
from project_paths import project_file
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
import walking_residual_env as w


ROOT = s.ROOT
OUT = ROOT/'results/walking_residual_pilot'


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
    assert not OUT.exists(), 'Refusing to overwrite pilot'
    verified.verify_sources()
    assert s.read(verified.OUT/'completion_audit.json')['goal_requirements_achieved']
    proposal = s.read(verified.OUT/'pilot_proposal.json')
    assert proposal['max_training_control_steps'] == 131072 and proposal['wall_time_cap_s'] == 1800
    protected = dict(s.read(verified.OUT/'design.json')['prior_artifacts_sha256'])
    for path in verified.OUT.rglob('*'):
        if path.is_file():
            protected[str(path.relative_to(ROOT))] = s.digest(path)
    for name, sha in protected.items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    previous = s.read(verified.OUT/'protocol.json')
    used = set(previous['prior_recorded_seeds']+previous['benchmark']['seeds']+previous['benchmark']['warmup_seeds'])
    assert not set(proposal['validation_seeds']+proposal['reset_seed_pool']+proposal['reserved_later_test_seeds']).intersection(used)
    protocol = {**proposal, 'status':'User-authorized execution protocol, frozen before pilot',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'proposal_sha256':s.digest(verified.OUT/'pilot_proposal.json'),
        'training_executed_at_freeze':False, 'scope':'One training seed only; no subsequent replicates or final test.',
        'baseline_validation':'25 zero-residual missions on the same validation poses, once before training. Additional to the maximum 75000 learned-checkpoint validation steps; included in the same 1800s wall cap.',
        'wall_budget_scope':'Monotonic clock from PPO/environment construction through baseline validation, training, checkpoint validation and serialization. Preparation/tests and final read-only audit are outside this compute budget. Guard each control step and optimizer update.',
        'ppo_fixed_defaults':{'clip_range':.2,'ent_coef':0.,'vf_coef':.5,'max_grad_norm':.5,
            'normalize_advantage':True,'use_sde':False,'log_std_init':0.,'n_envs':1},
        'reset_pool':'Cycle the declared 24000-24999 pose seeds, uniform scenario draw via mixed reset. PPO seed remains 0. Training wrapper overrides the initial VecEnv reset seed so pose seed 0 is not used.',
        'validation_rng':'Save/restore Python, NumPy and Torch RNG around checkpoint reload and deterministic validation; training env and optimizer continue without resetting.',
        'effort_definition':'Selection effort is the equally weighted mean of per-episode mean squared clipped normalized residual actions.',
        'partial_checkpoint':'If wall budget or nonfinite aborts collection/optimization, preserve a separate partial checkpoint and diagnostics; never select it as a scheduled completed checkpoint.'}
    protocol.pop('training_executed',None)
    OUT.mkdir()
    s.write(OUT/'protocol.json', protocol)
    names = [*s.read(verified.OUT/'design.json')['source_sha256'], 'walking_residual_pilot.py', 'test_walking_residual_pilot.py', 'review_walking_residual_pilot.py']
    s.write(OUT/'design.json', {'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in dict.fromkeys(names)}, 'prior_artifacts_sha256':protected})
    print(f'Frozen seed-0 pilot: 131072 steps/1800s; {len(protected)} old artifacts protected', flush=True)


def verify_sources():
    verified.verify_sources()
    p, d = s.read(OUT/'protocol.json'), s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json') == d['protocol_sha256']
    for name, sha in d['source_sha256'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    assert p['reward_weights'] == w.REWARD_WEIGHTS and p['residual_scale_rad'] == w.RESIDUAL_SCALE
    return p


def evaluate(model, label, p, deadline):
    folder = OUT/label
    folder.mkdir()
    env = w.WalkingResidualEnv('healthy', record_trace=True)
    checkpoint_sha = None if model is None else s.digest(OUT/f"checkpoint_{label.rsplit('_',1)[1]}.zip")
    rows = []
    try:
        for condition in w.CONDITIONS:
            for seed in p['validation_seeds']:
                budget_check(deadline)
                obs, _ = env.reset(seed=seed, options={'condition':condition})
                initial_hash = s.digest_bytes(env.data.qpos.tobytes()+env.data.qvel.tobytes())
                observations, actions, rewards = [], [], []
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
                        if not np.isfinite(obs).all() or not np.isfinite(reward) or info.get('episode_result',{}).get('reason')=='nonfinite state':
                            raise FloatingPointError('Nonfinite validation state/reward')
                        done = terminated or truncated
                except Exception as exc:
                    interrupted, pending_error = True, exc
                arrays = env.trace()
                arrays.update({'residual_observations':np.asarray(observations, dtype=np.float32),
                    'residual_policy_actions':np.asarray(actions, dtype=np.float32), 'control_rewards':np.asarray(rewards)})
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
    tests = (OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json', {'started_utc':datetime.now(timezone.utc).isoformat(),
        'tests_sha256':s.digest(OUT/'tests-before.log'), 'device':'cpu','torch':torch.__version__,'sb3':__import__('stable_baselines3').__version__})
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
    outcome = {'finished_utc':datetime.now(timezone.utc).isoformat(),'stop_reason':reason,'error':error,
        'training_executed':model is not None and model.num_timesteps>0,
        'training_steps':model.num_timesteps if model is not None else 0,
        'compute_wall_s':time.perf_counter()-started,'completed_checkpoint_validations':completed,
        'selected_checkpoint':selected,'pilot_promoted':selected is not None,'final_test_run':False,'replicate_training_run':False}
    s.write(OUT/'outcome.json',outcome)
    import review_walking_residual_pilot
    review_walking_residual_pilot.audit()
    if reason == 'aborted_error_or_nonfinite':
        raise RuntimeError('Pilot aborted; preserved diagnostics and report: '+str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    args = parser.parse_args()
    if args.stage == 'audit':
        import review_walking_residual_pilot
        review_walking_residual_pilot.audit()
    else:
        {'prepare':prepare,'run':run}[args.stage]()
