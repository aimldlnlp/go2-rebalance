"""Frozen-state strength sensitivity. No environment, rollout, or optimization."""
from project_paths import project_file, source_files
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
import stable_baselines3
from walking_baseline import JOINTS, torque_limit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/walking_residual_privileged_pilot'
PREP = ROOT/'results/walking_strength_masking_preparation'
OUT = ROOT/'results/walking_strength_masking'
DEFAULT = np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
STAGES = ['raw', 'action', 'offset', 'target', 'requested', 'limited', 'applied']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mask(obs):
    assert obs.shape == (72,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    result = obs.copy()
    result[60:] = 1
    np.testing.assert_array_equal(obs[:60], result[:60])
    return result


def eligibility(ticks, onset):
    return np.zeros(len(ticks), bool) if onset is None else ticks > onset


def action_path(raw, nominal, ranges):
    action = np.clip(raw.astype(np.float64), -1, 1)
    offset = .1*action
    return action, offset, np.clip(nominal+offset, ranges[:, 0], ranges[:, 1])


def physical_path(target, a, limits, weak):
    n = len(a['qpos'])-1
    repeated = np.repeat(target, 10, axis=0)[:n]
    requested = a['kp']*(repeated-a['qpos'][:-1, 7:])-a['kd']*a['qvel'][:-1, 6:]
    limited = torque_limit(requested, a['qvel'][:-1, 6:], 'go2_rl_lab', limits)
    applied = limited.copy()
    if weak is not None:
        applied[:, weak] *= a['strength']
    return requested, limited, applied


def bounds(first, end, name):
    if name == 'first':
        return first, min(first+10, end)
    if name == 'available':
        return first, end
    length = {'0.5s': 250, '2s': 1000}[name]
    return (first, first+length) if first+length <= end else None


def metrics(a, start, end, weak):
    controls = (a['ticks'] >= start) & (a['ticks'] < end)
    result = dict(physics_steps=end-start, control_rows=int(controls.sum()),
                  duration_s=(end-start)*.002)
    for stage in STAGES:
        sl = controls if stage in STAGES[:4] else slice(start, end)
        delta = a[stage+'_m'][sl].astype(np.float64)-a[stage+'_f'][sl].astype(np.float64)
        result[stage] = dict(rms=float(np.sqrt(np.mean(delta**2))),
            maximum=float(np.max(np.abs(delta))),
            per_joint_rms=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
            per_joint_maximum=np.max(np.abs(delta), axis=0).tolist())
    delta = a['applied_m'][start:end]-a['applied_f'][start:end]
    leg = weak//3
    for name, joints in [('weak_knee', [weak]), ('weak_leg', list(range(leg*3, leg*3+3))),
                         ('other_legs', [j for j in range(12) if j//3 != leg])]:
        result[name+'_torque_rms'] = float(np.sqrt(np.mean(delta[:, joints]**2)))
    for label, before, after, sl in [('action_clip', 'raw', 'action', controls),
            ('joint_clip', 'unclipped_target', 'target', controls),
            ('torque_saturation', 'requested', 'limited', slice(start, end))]:
        values = []
        for arm in ['f', 'm']:
            value = np.mean(a[before+'_'+arm][sl] != a[after+'_'+arm][sl], axis=0)
            values.append(value)
        result[label] = dict(factual=float(values[0].mean()), masked=float(values[1].mean()),
            difference=float((values[1]-values[0]).mean()),
            factual_per_joint=values[0].tolist(), masked_per_joint=values[1].tolist())
    for name, numerator, denominator in [('action_retention', 'action', 'raw'),
            ('joint_retention', 'target', 'offset'), ('limiter_retention', 'limited', 'requested'),
            ('physical_strength_retention', 'applied', 'limited')]:
        den = result[denominator]['rms']
        result[name] = dict(value=None if den <= 1e-7 else result[numerator]['rms']/den,
            reason='denominator at or below numerical tolerance' if den <= 1e-7 else None)
    return result


def verify_inputs(plan):
    for name, sha in plan['input_hashes'].items():
        assert digest(project_file(ROOT, name)) == sha, name


def prepare():
    assert not OUT.exists(), 'Never overwrite an existing study'
    prep = read(PREP/'source_inventory.json')
    for name, sha in prep['verified_input_hashes'].items():
        assert digest(project_file(ROOT, name)) == sha, name
    assert 'OK' in (ROOT/'work/strength_masking_tests.log').read_text(encoding='utf-8')
    from walking_calibrated import calibrated_model
    model = calibrated_model()
    assert [model.actuator(i).name for i in range(12)] == JOINTS
    files = list((ROOT/'results').rglob('*')) + list(source_files(ROOT))
    files += list((ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*.xml'))
    files += [ROOT/'work/audit_strength_masking.py', ROOT/'work/strength_masking_tests.log',
              ROOT/'vendor/walking_baselines/go2_rl_lab/policy.pt']
    hashes = {}
    for p in files:
        if p.is_file():
            hashes[str(p.relative_to(ROOT))] = digest(p)
            if len(hashes) % 500 == 0:
                print(f'Hashing protected inputs: {len(hashes)}', flush=True)
    rows = read(SOURCE/'validation_131072/episodes.json')
    rows.sort(key=lambda r: (r['condition'], r['seed']))
    plan = dict(status='Frozen before inference', compute_budget_s=300,
        checkpoint=str((SOURCE/'checkpoint_131072.zip').relative_to(ROOT)),
        checkpoint_sha256=prep['checkpoint_sha256'], rows=rows, input_hashes=hashes,
        joint_order=JOINTS, joint_ranges=model.jnt_range[model.actuator_trnid[:, 0]].tolist(),
        torque_limits=model.actuator_ctrlrange.tolist(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     torch=torch.__version__, sb3=stable_baselines3.__version__),
        atol=1e-7, rtol=0, device='cpu', threads=1,
        arm_order=['factual', 'masked'], training=False, new_rollouts=0)
    OUT.mkdir()
    write(OUT/'protocol.json', plan)
    (OUT/'PLAN.md').write_bytes((PREP/'PLAN.md').read_bytes())
    print(f'Frozen {len(hashes)} protected inputs and {len(rows)} episodes', flush=True)


def run():
    plan = read(OUT/'protocol.json')
    assert not (OUT/'execution.json').exists() and not (OUT/'arrays').exists()
    verify_inputs(plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import mujoco
    def forbidden(*args, **kwargs):
        raise RuntimeError('Physics calls are forbidden in this study')
    mujoco.mj_step = mujoco.mj_forward = forbidden
    started = time.perf_counter()
    result = dict(completed=False, budget_s=300, training=False, new_rollouts=0,
                  episodes_completed=0, parity_maxima={}, error=None)
    def deadline():
        assert time.perf_counter()-started <= plan['compute_budget_s'], 'Compute budget exhausted'
    def parity(actual, expected, label):
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
        maximum = float(np.max(np.abs(actual-expected)))
        result['parity_maxima'][label] = max(maximum, result['parity_maxima'].get(label, 0))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7, err_msg=label)
    (OUT/'arrays').mkdir()
    episodes = []
    try:
        policy = PPO.load(ROOT/plan['checkpoint'], device='cpu')
        policy.policy.set_training_mode(False)
        assert policy.num_timesteps == 131072 and not policy.policy.squash_output
        assert policy.observation_space.shape == (72,)
        ranges, limits = np.array(plan['joint_ranges']), np.array(plan['torque_limits'])
        for row in plan['rows']:
            deadline()
            stem = f"{row['condition']}_{row['seed']}"
            with np.load(SOURCE/'validation_131072'/f'{stem}.npz', allow_pickle=False) as z:
                source = {k:z[k] for k in z.files}
            obs = source['residual_observations']
            n = len(source['qpos'])-1
            ticks = np.arange(len(obs))*10
            assert len(obs) == (n+9)//10 and obs.dtype == np.float32
            post = eligibility(ticks, row['fault_onset_tick'])
            weak = None if row['condition'] == 'healthy' else JOINTS.index(row['condition'])
            expected = np.ones((len(obs), 12), dtype=np.float32)
            if weak is not None:
                expected[post, weak] = .1
                np.testing.assert_array_equal(source['strength'], np.where(np.arange(n) >= row['fault_onset_tick'], .1, 1.))
            else:
                np.testing.assert_array_equal(source['strength'], np.ones(n))
            np.testing.assert_array_equal(obs[:, 60:], expected)
            previous = np.vstack([np.zeros((1, 12)), source['control_residual_actions'][:-1]]).astype(np.float32)
            np.testing.assert_array_equal(obs[:, 48:60], previous)
            raw_f, raw_m = [], []
            for i, original in enumerate(obs):
                deadline()
                masked = mask(original)
                with torch.inference_mode():
                    f = policy.policy.get_distribution(torch.from_numpy(original[None])).mode().numpy()[0].copy()
                    predicted = policy.predict(original, deterministic=True)[0]
                    parity(np.clip(f, -1, 1), predicted, 'raw_to_predict')
                    parity(predicted, source['residual_policy_actions'][i], 'policy_replay')
                    parity(np.clip(predicted.astype(float), -1, 1), source['control_residual_actions'][i], 'action_replay')
                    m = policy.policy.get_distribution(torch.from_numpy(masked[None])).mode().numpy()[0].copy()
                assert np.isfinite(m).all()
                if not post[i]:
                    np.testing.assert_array_equal(f, m)
                raw_f.append(f)
                raw_m.append(m)
            nominal = DEFAULT+.25*source['control_actions']
            a = dict(ticks=ticks, post=post, strength=source['strength'])
            for arm, values in [('f', raw_f), ('m', raw_m)]:
                raw = np.asarray(values)
                action, offset, target = action_path(raw, nominal, ranges)
                requested, limited, applied = physical_path(target, source, limits, weak)
                for key, value in dict(raw=raw, action=action, offset=offset, target=target,
                        unclipped_target=nominal+offset, requested=requested,
                        limited=limited, applied=applied).items():
                    assert np.isfinite(value).all(), (stem, arm, key)
                    a[key+'_'+arm] = value
            parity(np.repeat(a['target_f'], 10, axis=0)[:n], source['target'], 'target_replay')
            for stage in STAGES[4:]:
                parity(a[stage+'_f'], source[stage+'_torque'], stage+'_replay')
            first = int(ticks[post][0]) if post.any() else None
            windows = {}
            if first is not None:
                for window in ['first', '0.5s', '2s', 'available']:
                    interval = bounds(first, n, window)
                    windows[window] = None if interval is None else metrics(a, *interval, weak)
            else:
                for stage in STAGES:
                    np.testing.assert_array_equal(a[stage+'_f'], a[stage+'_m'])
            np.savez_compressed(OUT/'arrays'/f'{stem}.npz', **a)
            episodes.append(dict(condition=row['condition'], seed=row['seed'],
                source_trace_sha256=row['trace_sha256'], historic_success=row['success'],
                historic_fallen=row['fallen'], control_rows=len(obs), post_rows=int(post.sum()),
                first_informed_tick=first, windows=windows,
                output_sha256=digest(OUT/'arrays'/f'{stem}.npz')))
            result['episodes_completed'] += 1
            deadline()
            print(f'{stem}: parity passed, {int(post.sum())} informed rows', flush=True)
        write(OUT/'episodes.json', episodes)
        deadline()
        result['completed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['compute_wall_s'] = time.perf_counter()-started
        write(OUT/'execution.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.command]()
