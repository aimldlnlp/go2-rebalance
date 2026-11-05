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
import walking_residual_privileged_env as w
import walking_residual_low_noise_pilot as parent


ROOT = s.ROOT
OUT = ROOT/'results/walking_residual_privileged_pilot'
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
    changed={'created_utc','status','parent_protocol_sha256','only_controller_change','initialization','parity','scope',
             'observation_shape','observation_fields','fault'}
    for key,value in previous.items():
        if key not in changed:assert p[key]==value,key
    assert p['observation_shape']==[72] and p['observation_fields']==w.OBSERVATION_FIELDS
    assert p['ppo_fixed_defaults']['log_std_init']==LOG_STD_INIT
    assert p['height_reference_m']==.338


def policy_record(model):
    shared = b''.join(name.encode()+b'\0'+value.detach().cpu().numpy().tobytes()
        for name,value in sorted(model.policy.state_dict().items()) if name!='log_std')
    return {'training_steps':model.num_timesteps,'ppo_epoch_updates':model._n_updates,
        'shared_non_log_std_parameters_sha256':s.digest_bytes(shared),
        'log_std':model.policy.log_std.detach().cpu().numpy().tolist(),
        'sigma':model.policy.log_std.detach().cpu().exp().numpy().tolist(),
        'sigma_learnable':bool(model.policy.log_std.requires_grad)}


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite privileged pilot'
    from diagnose_walking_fault import digest
    previous=parent.verify_sources()
    assert s.read(parent.OUT/'completion_audit.json')['audit_passed']
    latest=ROOT/'results/walking_rr85_confirmation'
    assert s.read(latest/'completion_audit.json')['study_complete']
    protected=dict(s.read(latest/'design.json')['prior_artifacts_sha256'])
    for folder in [latest,parent.OUT]:
        for path in folder.rglob('*'):
            if path.is_file():protected[str(path.relative_to(ROOT))]=digest(path)
    for path in source_files(ROOT):protected[str(path.relative_to(ROOT))]=digest(path)
    for name,sha in protected.items():assert digest(project_file(ROOT, name))==sha,name
    for seed in CONFIRMATION_SEEDS:
        assert not any((ROOT/'results').rglob(f'*_{seed}.npz')), f'Confirmation seed already used: {seed}'
    protocol={**previous,'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'One privileged-strength observation pilot; frozen before training',
        'parent_protocol_sha256':s.digest(parent.OUT/'protocol.json'),
        'observation_shape':[72],'observation_fields':w.OBSERVATION_FIELDS,
        'fault':'Actual latched strength is appended to the original 60 observations in actuator order. All ones before latch and in healthy; selected calf0.1 only after latch. No future fault label.',
        'only_controller_change':'Append 12 simulator-known motor strengths. Dynamics, reward, navigator, residual bound, PPO settings, mixed reset, initial sigma0.2 and mission criteria unchanged.',
        'initialization':'Fresh PPO seed0 with standard initialization, MLP widths[64,64]. Input72 instead of60 adds1536 actor/critic weights. Initial weights are not matched to the60-input control; this single-seed comparison cannot isolate information from capacity/initialization effects.',
        'parity':'45 new zero-residual regression replays against the same historical fixtures, all original arrays exact; additionally verify the72-input suffix timing. Not new generalization episodes.',
        'scope':'One privileged simulation pilot,131072 training steps or1800s including baseline, checkpoint/conditional confirmation evaluation and serialization. No automatic extension, curriculum, student or final test.',
        'privileged':True,'deployable_without_simulator_strength':False,
        'parameter_increase_from_input':1536,
        'comparison':'Paired by condition, pose and equal checkpoint steps against the low-noise pilot. Baselines are duplicate controls. Compare all checkpoints; do not call the last checkpoint best.',
        'evidence_review':'Low-noise: healthy5/5 at all checkpoints, faults0/20. Post-fault gain escalation did not solve missions. RR85 development confirmation8/10 versus healthy10/10. These motivate testing explicit fault information, not a claim that missing identity is the proven cause.'}
    check_protocol(protocol,previous)
    OUT.mkdir();s.write(OUT/'protocol.json',protocol)
    names=[*s.read(parent.OUT/'design.json')['source_sha256'],
        'walking_residual_privileged_env.py','walking_residual_privileged_pilot.py',
        'review_walking_residual_privileged_pilot.py','test_walking_residual_privileged.py',
        'report_walking_residual_privileged_pilot.py']
    s.write(OUT/'design.json',dict(protocol_sha256=s.digest(OUT/'protocol.json'),
        source_sha256={name:digest(project_file(ROOT, name)) for name in dict.fromkeys(names)},prior_artifacts_sha256=protected))
    with (OUT/'EVIDENCE_REVIEW.md').open('x',encoding='utf-8') as handle:
        handle.write('# Keputusan sebelum training\n\nSatu pilot menguji tambahan informasi strength aktual. Sumber: audit low-noise, gain ablation dan konfirmasi RR85. Tidak ada hasil yang membuktikan identitas motor adalah akar penyebab. Observasi baru72 dimensi, termasuk12 strength simulator; policy belum deployable tanpa informasi itu.\n\nTetapkan seed0,131072 langkah/1800s, checkpoint32768/65536/131072, training pose24000–24999, development23000–23004 dan conditional confirmation23300–23304. Gate dan seleksi sama dengan kontrol. Final23100–23119 tetap disimpan. Semua reward, batas dan mixed conditions tetap. Jangan tambahkan curriculum, navigator baru atau budget otomatis.\n\nMLP tetap[64,64], tetapi input baru menambah1536 parameter. Initial weights tidak dipasangkan dengan model60-input; klaim efek kausal informasi membutuhkan studi tambahan. Baseline zero-residual diperiksa exact terhadap arsip. Perubahan kekuatan baru terlihat sesudah latch, tanpa label fault di awal episode.\n')
    print(f'Frozen privileged seed0 pilot131072/1800s; {len(protected)} protected files',flush=True)


def verify_sources():
    from diagnose_walking_fault import digest
    previous=parent.verify_sources()
    p,design=s.read(OUT/'protocol.json'),s.read(OUT/'design.json')
    assert digest(OUT/'protocol.json')==design['protocol_sha256']
    for name,sha in design['source_sha256'].items():assert digest(project_file(ROOT, name))==sha,name
    check_protocol(p,previous)
    assert p['parent_protocol_sha256']==digest(parent.OUT/'protocol.json')
    return p


def preflight():
    from diagnose_walking_fault import digest
    p=verify_sources()
    assert not (OUT/'execution.json').exists() and not (OUT/'parity.json').exists()
    from verify_walking_residual_env import metric_differences
    fixtures=parent.s.read(parent.OUT/'parity.json')['episodes']
    result=[];env=w.WalkingResidualEnv('healthy',record_trace=True)
    try:
        for index,row in enumerate(fixtures):
            assert digest(ROOT/row['trace'])==row['trace_sha256']
            metadata=s.read(ROOT/row['metadata'])
            obs,_=env.reset(seed=row['seed'],options={'condition':row['condition']})
            observations=[];done=False
            while not done:
                observations.append(obs.copy())
                obs,_,terminated,truncated,_=env.step(np.zeros(12,dtype=np.float32))
                done=terminated or truncated
            arrays=env.trace()
            with np.load(ROOT/row['trace'],allow_pickle=False) as old:
                for key in old.files:np.testing.assert_array_equal(arrays[key],old[key],err_msg=key)
            actual=env.mission_result()
            assert all(metric_differences(actual,metadata,1e-12).values())
            inputs=np.asarray(observations,dtype=np.float32);expected=np.ones((len(inputs),12),dtype=np.float32)
            if env.onset is not None:expected[np.arange(len(inputs))*10>env.onset,env.weak]=.1
            np.testing.assert_array_equal(inputs[:,60:],expected)
            path=OUT/f'parity_{index:02}.npz'
            with path.open('xb') as handle:np.savez_compressed(handle,**arrays,residual_observations=inputs)
            result.append({**row,'actual_trace':str(path.relative_to(ROOT)),
                'actual_trace_sha256':digest(path),'actual_result':actual})
    finally:env.close()
    assert len(result)==45
    s.write(OUT/'parity.json',dict(all_passed=True,episodes=result,new_simulation_episodes=45,
        scope='Historical regression replays only; all original arrays exact; strength timing independently checked.'))
    for name,sha in s.read(OUT/'design.json')['prior_artifacts_sha256'].items():assert digest(project_file(ROOT, name))==sha,name
    print('45/45 exact historical zero-residual replays; privileged suffix timing checked',flush=True)


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
    import review_walking_residual_privileged_pilot
    review_walking_residual_privileged_pilot.audit()
    if reason == 'aborted_error_or_nonfinite':
        raise RuntimeError('Pilot aborted; preserved diagnostics and report: '+str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preflight','run','audit'])
    args = parser.parse_args()
    if args.stage == 'audit':
        import review_walking_residual_privileged_pilot
        review_walking_residual_privileged_pilot.audit()
    else:
        {'prepare':prepare,'preflight':preflight,'run':run}[args.stage]()
