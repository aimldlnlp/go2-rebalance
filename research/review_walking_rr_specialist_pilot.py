"""Independent mission/reward audit and RR-only allocation checks."""
import time
import numpy as np
import torch
from stable_baselines3 import PPO
import walking_rr_specialist_pilot as study
import review_walking as physical
from review_walking_residual_height_pilot import audit_reward_and_nominal
from walking_baseline import rotation,JOINTS
from walking_strength_masking import DEFAULT


def audit():
    started=time.perf_counter()
    p=study.verify(True)
    out,s=study.OUT,study.s
    outcome=s.read(out/'outcome.json')
    assert outcome['training_steps']<=131072 and outcome['compute_wall_s']<=1800
    assert not outcome['confirmation_run'] and not outcome['final_test_run'] and not outcome['replicate_training_run']
    assert not outcome['pilot_promoted'] and outcome['selected_checkpoint'] is None
    assert not (out/'confirmation').exists()
    assert s.read(out/'initial_policy.json')==s.read(study.SOURCE/'initial_policy.json')
    assert s.read(out/'baseline_parity.json')['passed']
    torch.set_num_threads(1)
    stages=['baseline']+[f'validation_{n}' for n in outcome['completed_checkpoint_validations']]
    with np.load(out/'baseline/healthy_23000.npz',allow_pickle=False) as z:
        kind=type(z)
    getitem,close=kind.__getitem__,kind.close
    def cached(self,key):
        if not hasattr(self,'_progress_audit_cache'):
            self._progress_audit_cache={}
        if key not in self._progress_audit_cache:
            value=getitem(self,key)
            value.setflags(write=False)
            self._progress_audit_cache[key]=value
        return self._progress_audit_cache[key]
    def closed(self):
        self._progress_audit_cache={}
        return close(self)
    kind.__getitem__,kind.close=cached,closed
    states,observations,episodes=0,0,0
    summary=[]
    ranges=np.array([[-1.0472,1.0472],[-1.5708,3.4907],[-2.7227,-.83776]]*2+
        [[-1.0472,1.0472],[-.5236,4.5379],[-2.7227,-.83776]]*2)
    try:
        for stage in stages:
            rows=s.read(out/stage/'episodes.json')
            assert len(rows)==25 and {(r['condition'],r['seed']) for r in rows}=={(c,seed) for c in study.original.CONDITIONS for seed in range(23000,23005)}
            model=None
            if stage!='baseline':
                n=int(stage.rsplit('_',1)[1])
                checkpoint=out/f'checkpoint_{n}.zip'
                assert s.digest(checkpoint)==s.read(out/f'checkpoint_{n}.json')['checkpoint_sha256']
                model=PPO.load(checkpoint,device='cpu')
                assert model.num_timesteps==n and model.seed==0
                for key,val in p['hyperparameters'].items():
                    assert (model.policy_kwargs['net_arch'] if key=='net_arch' else getattr(model,key))==val,key
                study.parent.finite_policy(model)
            old=s.read(study.SOURCE/stage/'episodes.json')
            for row in rows:
                assert not row['interrupted']
                stem=f"{row['condition']}_{row['seed']}"
                path=out/stage/(stem+'.npz')
                assert row==s.read(out/stage/(stem+'.json'))
                assert s.digest(path)==row['trace_sha256']
                prior=next(r for r in old if (r['condition'],r['seed'])==(row['condition'],row['seed']))
                assert row['initial_state_sha256']==prior['initial_state_sha256']
                if model is not None:
                    assert row['residual_checkpoint_sha256']==s.digest(checkpoint)
                states+=physical.verify_episode(path,row,p)
                with np.load(path,allow_pickle=False) as a:
                    q,v=a['qpos'],a['qvel']
                    actions=a['control_residual_actions']
                    nominal=DEFAULT+.25*a['control_actions']
                    obs=[]
                    for i,tick in enumerate(range(0,len(q)-1,10)):
                        r=rotation(q[tick,3:7])
                        previous=np.zeros(12) if i==0 else actions[i-1]
                        strength=np.ones(12)
                        if row['fault_onset_tick'] is not None and tick>row['fault_onset_tick']:
                            strength[JOINTS.index(row['condition'])]=.1
                        obs.append(np.r_[r.T@v[tick,:3],v[tick,3:6],r.T@np.array([0.,0.,-1.]),a['command'][tick],
                            q[tick,7:]-DEFAULT,v[tick,6:]*.05,nominal[i]-DEFAULT,previous,strength])
                    obs=np.asarray(obs,dtype=np.float32)
                    np.testing.assert_array_equal(obs,a['residual_observations'])
                    expected=np.zeros_like(actions) if model is None else np.asarray([model.predict(o,deterministic=True)[0] for o in obs])
                    np.testing.assert_array_equal(a['residual_policy_actions'],expected)
                    np.testing.assert_array_equal(actions,np.clip(expected,-1,1))
                    target=np.clip(nominal+.1*actions,ranges[:,0],ranges[:,1])
                    np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:len(q)-1])
                    names=list(a['reward_component_names'])
                    index=names.index('goal_progress')
                    ticks=np.arange(0,len(q)-1,10)
                    ends=np.minimum(ticks+10,len(q)-1)
                    forward=np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
                    error=np.linalg.norm(q[:,:2]-(np.asarray(row['initial_xy'])+2*forward),axis=1)
                    expected_progress=10.*(error[ticks]-error[ends])
                    np.testing.assert_allclose(a['reward_components'][:,index],expected_progress,rtol=0,atol=1e-10)
                    audit_reward_and_nominal(a,row)
                    if stage=='baseline':
                        with np.load(study.SOURCE/stage/(stem+'.npz'),allow_pickle=False) as b:
                            assert a.files==b.files
                            for key in b.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                    np.testing.assert_allclose(a['control_rewards'],a['reward_components'].sum(axis=1),rtol=0,atol=1e-10)
                    assert abs(a['control_rewards'].sum()-row['episode_return'])<1e-8
                    np.testing.assert_allclose(np.mean(actions**2),row['residual_effort'],rtol=0,atol=1e-12)
                    observations+=len(obs)
                episodes+=1
                print(f'Audited {stage}/{stem}',flush=True)
            gate=study.parent.promotion(rows)
            target=[r for r in rows if r['condition'] in ['healthy','RR_calf']]
            specialist_gate=(sum(r['success'] for r in target if r['condition']=='healthy')==5 and
                sum(r['success'] for r in target if r['condition']=='RR_calf')>=3 and
                not any(r['fallen'] or r['interrupted'] for r in target))
            if stage!='baseline':
                assert s.read(out/f'acceptance_{n}.json')['passed']==gate
            for condition in study.original.CONDITIONS:
                group=[r for r in rows if r['condition']==condition]
                control=[r for r in old if r['condition']==condition]
                summary.append(dict(stage=stage,condition=condition,gate_passed=gate,specialist_gate_passed=specialist_gate,
                    control_success=sum(r['success'] for r in control),variant_success=sum(r['success'] for r in group),
                    control_falls=sum(r['fallen'] for r in control),variant_falls=sum(r['fallen'] for r in group),
                    median_control_progress_m=float(np.median([r['final_progress_m'] for r in control])),
                    median_variant_progress_m=float(np.median([r['final_progress_m'] for r in group]))))
    finally:
        kind.__getitem__,kind.close=getitem,close
    logs=[__import__('json').loads(line) for line in (out/'training_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    resets=[r for r in logs if r['event']=='reset']
    assert all(r['condition']=='RR_calf' for r in resets)
    assert all(r['pose_seed']==p['reset_seed_pool'][r['episode_index']%len(p['reset_seed_pool'])] for r in resets)
    assert all(r['pose_seed'] not in p['validation_seeds']+p['reserved_later_test_seeds'] for r in resets)
    assert sum(r['control_steps'] for r in logs if r['event'] in ['episode_end','partial'])==outcome['training_steps']
    # Reset-only audit checks the source RNG contract, without stepping any new episode.
    reference=study.original.WalkingResidualEnv('mixed')
    try:
        for row in resets:
            reference.reset(seed=row['pose_seed'],options={'condition':'mixed'})
            expected=s.digest_bytes(reference.data.qpos.tobytes()+reference.data.qvel.tobytes())
            assert expected==row['initial_state_sha256']
    finally:reference.close()
    exposure={}
    for name,source in [('mixed_control',study.SOURCE),('RR_specialist',out)]:
        history=[__import__('json').loads(line) for line in (source/'training_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        complete=[r for r in history if r['event']=='episode_end']
        allocated=[r for r in history if r['event'] in ['episode_end','partial']]
        exposure[name]=dict(total_control_steps=sum(r['control_steps'] for r in allocated),
            RR_allocated_control_steps=sum(r['control_steps'] for r in allocated if r['condition']=='RR_calf'),
            RR_complete_episodes=sum(r['condition']=='RR_calf' for r in complete),
            RR_complete_successes=sum(r['condition']=='RR_calf' and r['result']['success'] for r in complete),
            RR_complete_receiving_fault=sum(r['condition']=='RR_calf' and r['result']['fault_onset_tick'] is not None for r in complete),
            definition='Control allocation includes pre-fault controls; complete training episodes are separate from validation.')
        assert exposure[name]['total_control_steps']==131072
    study.verify(True)
    s.write(out/'training_exposure.json',exposure)
    s.write(out/'summary.json',summary)
    s.write(out/'audit.json',dict(passed=True,episodes=episodes,physics_states=states,policy_observations=observations,
        matched_initial_policy=True,original_reward_independently_reconstructed=True,
        training_only_RR=True,training_initial_states_matched=len(resets),
        protected_inputs_unchanged=len(s.read(out/'design.json')['protected_inputs']),
        audit_wall_s=time.perf_counter()-started,confirmation_not_run=True,final_seeds_unused=True,
        training_steps=outcome['training_steps'],pilot_promoted=False))


if __name__=='__main__':
    audit()
