"""Single reward-weight intervention, matched privileged initialization, CPU only."""
from project_paths import project_file, source_files
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import walking_residual_privileged_pilot as parent
import walking_residual_privileged_env as original
import sudden as s

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/walking_residual_privileged_pilot'
OUT=ROOT/'results/walking_progress_reward_pilot'
REWARD_WEIGHTS={**original.REWARD_WEIGHTS,'goal_progress':100.}


class ProgressEnv(original.WalkingResidualEnv):
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        terms=info['reward_terms']
        if 'goal_progress' in terms:
            terms['goal_progress']*=10.
            reward=float(sum(terms.values()))
        if not np.isfinite(reward):
            raise FloatingPointError('Nonfinite progress reward')
        return obs,reward,terminated,truncated,info


def wire():
    parent.OUT=OUT
    parent.w=SimpleNamespace(WalkingResidualEnv=ProgressEnv,CONDITIONS=original.CONDITIONS,
                             REWARD_WEIGHTS=REWARD_WEIGHTS)


def prepare():
    assert not OUT.exists(),'Never overwrite a pilot'
    assert 'OK' in (ROOT/'work/progress_reward_tests.log').read_text(encoding='utf-8')
    assert s.read(ROOT/'results/walking_reward_stall_review/final_verification.json')['passed']
    old=s.read(SOURCE/'protocol.json')
    paths=[p for p in (ROOT/'results').rglob('*') if p.is_file()]+list(source_files(ROOT))
    paths+=list((ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*.xml'))
    paths+=[ROOT/'vendor/walking_baselines/go2_rl_lab/policy.pt',ROOT/'work/progress_reward_tests.log']
    hashes={}
    for path in paths:
        hashes[str(path.relative_to(ROOT))]=s.digest(path)
        if len(hashes)%500==0:
            print(f'Protected hashes: {len(hashes)}',flush=True)
    p={**old,'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'One goal-progress weight pilot, frozen before training',
        'reward_weights':REWARD_WEIGHTS,'only_change':'goal_progress weight 10 to100; all else matched to privileged pilot',
        'rationale':'Tenfold sensitivity probe, not optimized weight. At .20m/s and .02s, ideal .004m progress earns .4 rather than .04 per control.',
        'control':'Archived privileged pilot seed0, matched initial policy weights and observation72',
        'wall_time_cap_s':1800,'max_training_control_steps':131072,'checkpoints':[32768,65536,131072],
        'validation_seeds':list(range(23000,23005)),
        'confirmation_seeds':[], 'confirmation_policy':'Never run; separate authorization required even if development gate passes',
        'scope':'Fresh seed0 matched initialization; one weight change, no fine tuning, no training replication, no curriculum/student/final test',
        'primary_endpoints':['mission_success','falls'],
        'parity':'Before training, 25 zero-residual baseline episodes match all historical arrays except reward components and sums, which must match exact tenfold progress transformation',
        'gate':'Healthy5/5, each fault at least3/5, zero falls/interruption. No promotion without unopened confirmation.',
        'comparison':'Same training steps and poses at every checkpoint. Returns not comparable across reward definitions.'}
    for key in ['hyperparameters','ppo_fixed_defaults','reset_seed_pool','training_seed','observation_shape','residual_scale_rad']:
        if key in old:
            assert p[key]==old[key],key
    OUT.mkdir()
    s.write(OUT/'protocol.json',p)
    s.write(OUT/'design.json',dict(protocol_sha256=s.digest(OUT/'protocol.json'),protected_inputs=hashes))
    shutil.copyfile(ROOT/'work/progress_reward_tests.log',OUT/'tests-before.log')
    print(f'Frozen goal-progress100 pilot:131072/1800s; {len(hashes)} protected files',flush=True)


def verify(full=False):
    p,design=s.read(OUT/'protocol.json'),s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==design['protocol_sha256']
    for name,sha in design['protected_inputs'].items():
        if full or name.endswith('.py'):
            assert s.digest(project_file(ROOT, name))==sha,name
    return p


def baseline_parity(rows):
    for row in rows:
        stem=f"{row['condition']}_{row['seed']}"
        old=s.read(SOURCE/'baseline'/(stem+'.json'))
        assert row['initial_state_sha256']==old['initial_state_sha256']
        with np.load(OUT/'baseline'/(stem+'.npz'),allow_pickle=False) as a,np.load(SOURCE/'baseline'/(stem+'.npz'),allow_pickle=False) as b:
            assert a.files==b.files
            for key in b.files:
                if key not in ['control_rewards','reward_components']:
                    np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            terms=b['reward_components'].copy()
            index=list(b['reward_component_names']).index('goal_progress')
            terms[:,index]*=10
            np.testing.assert_array_equal(a['reward_components'],terms)
            np.testing.assert_allclose(a['control_rewards'],terms.sum(axis=1),rtol=0,atol=1e-12)
        for key in ['success','fallen','fault_onset_tick','final_progress_m','final_stable_hold_s']:
            assert row[key]==old[key],key
    s.write(OUT/'baseline_parity.json',dict(passed=True,episodes=25,physical_observation_action_arrays_exact=True,
        reward_only_expected_change=True))


def run():
    assert not (OUT/'execution.json').exists(),'No rerun'
    p=verify(True)
    wire()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    s.write(OUT/'execution.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        tests_sha256=s.digest(OUT/'tests-before.log'),device='cpu',torch=torch.__version__,
        sb3=__import__('stable_baselines3').__version__))
    started=time.perf_counter()
    deadline=started+1800
    model,pool,training=None,None,None
    completed=[]
    reason,error='step_budget_completed',None
    try:
        pool=parent.TrainingPool(ProgressEnv('mixed'),p['reset_seed_pool'],OUT/'training_episodes.jsonl')
        training=Monitor(pool,str(OUT/'training.monitor.csv'))
        hp=p['hyperparameters']
        model=PPO('MlpPolicy',training,device='cpu',seed=p['training_seed'],
            n_steps=hp['n_steps'],batch_size=hp['batch_size'],n_epochs=hp['n_epochs'],
            learning_rate=hp['learning_rate'],gamma=hp['gamma'],gae_lambda=hp['gae_lambda'],
            policy_kwargs={'net_arch':hp['net_arch'],'log_std_init':p['ppo_fixed_defaults']['log_std_init']},
            clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,normalize_advantage=True,use_sde=False,verbose=1)
        initial=parent.policy_record(model)
        assert initial==s.read(SOURCE/'initial_policy.json'),'Initial weights/sigma/update counters must exactly match control'
        s.write(OUT/'initial_policy.json',initial)
        model.set_logger(configure(str(OUT),['stdout','csv']))
        def before(optimizer,args,kwargs):
            parent.budget_check(deadline)
            for parameter in model.policy.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
        def after(optimizer,args,kwargs):
            parent.finite_policy(model)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite optimizer state')
            parent.budget_check(deadline)
        model.policy.optimizer.register_step_pre_hook(before)
        model.policy.optimizer.register_step_post_hook(after)
        with parent.preserved_rng():
            rows=parent.evaluate(None,'baseline',p,deadline)
            baseline_parity(rows)
        for n in p['checkpoints']:
            parent.budget_check(deadline)
            model.learn(total_timesteps=n-model.num_timesteps,reset_num_timesteps=False,
                callback=parent.Guard(deadline,p['max_training_control_steps']),log_interval=8)
            parent.finite_policy(model)
            assert model.num_timesteps==n
            path=OUT/f'checkpoint_{n}.zip'
            model.save(path)
            s.write(OUT/f'checkpoint_{n}.json',dict(training_steps=n,checkpoint_sha256=s.digest(path),
                elapsed_s=time.perf_counter()-started,sigma=model.policy.log_std.detach().exp().numpy().tolist()))
            with parent.preserved_rng():
                loaded=PPO.load(path,device='cpu')
                probe=np.zeros(72,dtype=np.float32)
                probe[60:]=1
                np.testing.assert_array_equal(model.predict(probe,deterministic=True)[0],loaded.predict(probe,deterministic=True)[0])
                rows=parent.evaluate(loaded,f'validation_{n}',p,deadline)
            completed.append(n)
            s.write(OUT/f'acceptance_{n}.json',dict(passed=parent.promotion(rows),
                successes={c:sum(r['success'] for r in rows if r['condition']==c) for c in original.CONDITIONS},
                falls=sum(r['fallen'] for r in rows)))
        parent.budget_check(deadline)
    except parent.BudgetExpired as exc:
        reason,error='wall_budget_reached',str(exc)
    except BaseException:
        reason,error='aborted_error_or_nonfinite',traceback.format_exc()
    finally:
        if reason!='step_budget_completed' and model is not None:
            model.save(OUT/'partial_checkpoint.zip')
            if pool is not None:
                np.savez_compressed(OUT/'diagnostic_state.npz',qpos=pool.unwrapped.data.qpos,
                    qvel=pool.unwrapped.data.qvel,previous_residual=pool.unwrapped.previous)
        if training is not None:
            training.close()
        s.write(OUT/'outcome.json',dict(stop_reason=reason,error=error,training_steps=0 if model is None else model.num_timesteps,
            compute_wall_s=time.perf_counter()-started,completed_checkpoint_validations=completed,
            development_gate_candidates=[n for n in completed if s.read(OUT/f'acceptance_{n}.json')['passed']],
            pilot_promoted=False,selected_checkpoint=None,confirmation_run=False,final_test_run=False,
            replicate_training_run=False))
    if error is not None:
        raise RuntimeError(error)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.command]()
