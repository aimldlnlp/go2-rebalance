"""Fixed 20-episode development confirmation; no training or final-test seeds."""
from project_paths import project_file, source_files
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import walking_rr_mild_severity as parent

ROOT=parent.ROOT
OUT=ROOT/'results/walking_rr85_confirmation'
SEEDS=list(range(23400,23410))
CONDITIONS=['healthy','RR_calf']
d=parent.d


def order():
    return [dict(condition=c,seed=seed,strength_remaining=1. if c=='healthy' else .85)
            for seed in SEEDS for c in CONDITIONS]


def gate(rows):
    return len(rows)==20 and {(r['condition'],r['seed']) for r in rows}=={(c,s) for c in CONDITIONS for s in SEEDS} and all(
        r['success'] and not r['fallen'] and not r['interrupted'] for r in rows)


def prepare():
    assert not OUT.exists(),'Refusing overwrite'
    parent.source_contract()
    assert d.read(parent.OUT/'completion_audit.json')['study_complete']
    used=set();examined=[]
    for folder in (ROOT/'results').iterdir():
        if not folder.is_dir() or not folder.name.startswith('walking'):continue
        for path in folder.rglob('episodes.json'):
            rows=d.read(path)
            if isinstance(rows,list):
                used.update(r['seed'] for r in rows if isinstance(r,dict) and isinstance(r.get('seed'),int))
                examined.append(str(path.relative_to(ROOT)))
        for path in folder.glob('training_episodes.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row=__import__('json').loads(line)
                if 'pose_seed' in row:used.add(row['pose_seed'])
            examined.append(str(path.relative_to(ROOT)))
    assert not used.intersection(SEEDS)
    assert not set(SEEDS).intersection(range(23100,23120))
    protected=dict(d.read(parent.OUT/'design.json')['prior_artifacts_sha256'])
    for path in parent.OUT.rglob('*'):
        if path.is_file():protected[str(path.relative_to(ROOT))]=d.digest(path)
    for path in source_files(ROOT):protected[str(path.relative_to(ROOT))]=d.digest(path)
    for name,sha in protected.items():assert d.digest(project_file(ROOT, name))==sha,name
    OUT.mkdir()
    d.write(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does nominal RR85 mission success hold on ten unused development poses, with paired healthy controls?',
        development_confirmation=True,seeds=SEEDS,run_order=order(),new_episodes=20,
        wall_time_cap_s=180,max_control_steps=20000,
        budget_scope='Construction, 20 episodes, online prefix checks and serialization only. Preparation/tests/hash checks and offline audit/report excluded. No extension or retries.',
        pass_rule='Both healthy and RR85 must each finish 10/10 missions with zero falls and zero interrupted episodes.',
        mission_acceptance=d.read(parent.OUT/'protocol.json')['mission_acceptance'],
        implementation='Reuse frozen mild-severity step unchanged. Healthy uses condition healthy; RR uses 0.85 after limiting and first progress>=0.50m latch. Residual always zero.',
        provenance='23400–23409 absent from recorded walking episodes/training pose logs inspected before freeze. New development data, not reserved final data.',
        unused_pose_check_files=examined,previous_recorded_pose_count=len(used),
        final_seeds_reserved=list(range(23100,23120)),training=False,residual_policy_loaded=False,
        no_automatic_training_or_followup=True,preview=False))
    names=['walking_rr85_confirmation.py','test_walking_rr85_confirmation.py','report_walking_rr85_confirmation.py']
    d.write(OUT/'design.json',dict(protocol_sha256=d.digest(OUT/'protocol.json'),
        source_sha256={name:d.digest(project_file(ROOT, name)) for name in names},prior_artifacts_sha256=protected))
    print(f'Frozen 20 episodes / 180 s; seeds {SEEDS[0]}–{SEEDS[-1]}; {len(protected)} protected files',flush=True)


def verify_sources():
    parent.source_contract();p=d.read(OUT/'protocol.json');design=d.read(OUT/'design.json')
    assert d.digest(OUT/'protocol.json')==design['protocol_sha256']
    for name,sha in {**design['source_sha256'],**design['prior_artifacts_sha256']}.items():assert d.digest(project_file(ROOT, name))==sha,name
    assert p['seeds']==SEEDS and p['run_order']==order() and p['new_episodes']==20
    assert torch.version.cuda is None
    return p


def run():
    assert not (OUT/'execution.json').exists(),'Refusing rerun'
    p=verify_sources();tests=(OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    d.write(OUT/'execution.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        tests_sha256=d.digest(OUT/'tests-before.log'),torch=torch.__version__,
        mujoco=parent.mujoco.__version__,device='cpu',training=False))
    started=time.perf_counter();deadline=started+p['wall_time_cap_s']
    rows=[];steps=0;error=None;reason='episodes_completed';env=None
    try:
        for case in p['run_order']:
            parent.pilot.budget_check(deadline)
            env=parent.SeverityEnv(.85,condition=case['condition'])
            obs,_=env.reset(seed=case['seed']);inputs=[];rewards=[];components=[]
            done=False;interrupted=False;pending=None;prefix_checked=False
            reference=None if case['condition']=='healthy' else parent.read_arrays(OUT/'healthy'/f"healthy_{case['seed']}.npz")
            try:
                while not done:
                    parent.pilot.budget_check(deadline);assert steps<p['max_control_steps']
                    inputs.append(obs.copy())
                    obs,reward,terminated,truncated,info=env.step(np.zeros(12,dtype=np.float32));steps+=1
                    rewards.append(reward);components.append([info['reward_terms'].get(k,0.) for k in parent.w.REWARD_WEIGHTS])
                    if not np.isfinite(obs).all() or not np.isfinite(reward) or not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                        raise FloatingPointError('Nonfinite confirmation state/reward')
                    if env.onset is not None and not prefix_checked:
                        prefix=env.trace();prefix['residual_observations']=np.asarray(inputs,dtype=np.float32)
                        parent.check_prefix(prefix,reference,env.onset);prefix_checked=True
                    done=terminated or truncated
            except Exception as exc:interrupted=True;pending=exc
            a=env.trace();a.update(residual_observations=np.asarray(inputs,dtype=np.float32),
                residual_policy_actions=np.zeros((len(inputs),12),dtype=np.float32),control_rewards=np.asarray(rewards),
                reward_components=np.asarray(components),reward_component_names=np.asarray(list(parent.w.REWARD_WEIGHTS)))
            folder=OUT/case['condition'];folder.mkdir(exist_ok=True);path=folder/f"{case['condition']}_{case['seed']}.npz"
            with path.open('xb') as handle:np.savez_compressed(handle,**a)
            result=env.mission_result()
            if interrupted:result['success']=False
            row=dict(**case,candidate='go2_rl_lab',initial_xy=env.origin.tolist(),initial_yaw=env.initial_yaw,
                initial_state_sha256=parent.s.digest_bytes(a['qpos'][0].tobytes()+a['qvel'][0].tobytes()),
                trace=str(path.relative_to(ROOT)),trace_sha256=d.digest(path),interrupted=interrupted,
                episode_return=float(sum(rewards)) if np.isfinite(rewards).all() else None,**result)
            d.write(path.with_suffix('.json'),row);rows.append(row);env.close();env=None
            print(f"{len(rows)}/20 {case['condition']}/{case['seed']}: success={row['success']} fall={row['fallen']} hold={row['final_stable_hold_s']:.3f}",flush=True)
            if pending is not None:raise pending
            parent.pilot.budget_check(deadline)
    except parent.pilot.BudgetExpired as exc:reason='wall_budget_reached';error=str(exc)
    except Exception as exc:
        reason='aborted_error_or_nonfinite';error=str(exc)
        d.write(OUT/'error.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        if env is not None:env.close()
    d.write(OUT/'episodes.json',rows)
    d.write(OUT/'outcome.json',dict(stop_reason=reason,error=error,episodes_recorded=len(rows),control_steps=steps,
        compute_wall_s=time.perf_counter()-started,training=False,final_seeds_used=False))
    audit()
    if error:raise RuntimeError(error)


def audit():
    p=verify_sources();rows=d.read(OUT/'episodes.json');outcome=d.read(OUT/'outcome.json')
    assert d.digest(OUT/'tests-before.log')==d.read(OUT/'execution.json')['tests_sha256']
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(d.read(OUT/'execution.json')['started_utc'])
    assert [dict(condition=r['condition'],seed=r['seed'],strength_remaining=r['strength_remaining']) for r in rows]==order()[:len(rows)]
    assert len(rows)<=20 and outcome['control_steps']<=20000
    states=0;complete=[];prefixes=0;healthy={}
    for row in rows:
        path=ROOT/row['trace'];assert d.read(path.with_suffix('.json'))==row
        if row['interrupted']:continue
        states+=parent.verify_episode(path,row,p);a=parent.read_arrays(path)
        assert not np.any(a['control_residual_actions']) and not np.any(a['residual_policy_actions'])
        assert not np.any(a['residual_observations'][:,48:60])
        parent.audit_reward_and_nominal(a,row)
        if row['condition']=='healthy':healthy[row['seed']]=(a,row)
        else:
            ref,meta=healthy[row['seed']]
            assert row['initial_state_sha256']==meta['initial_state_sha256']
            parent.check_prefix(a,ref,row['fault_onset_tick']);prefixes+=1
        complete.append(row)
    groups=[dict(condition=c,episodes=sum(r['condition']==c for r in rows),
        successes=sum(r['condition']==c and r['success'] for r in rows),
        falls=sum(r['condition']==c and r['fallen'] for r in rows),
        holds_s=[r['final_stable_hold_s'] for r in rows if r['condition']==c]) for c in CONDITIONS]
    proof=dict(audit_passed=True,study_complete=len(complete)==20 and outcome['stop_reason']=='episodes_completed',
        confirmation_passed=gate(rows),episodes_checked=len(complete),physics_states_checked=states,
        exact_healthy_fault_prefix_pairs=prefixes,groups=groups,
        protected_prior_artifacts_unchanged=len(d.read(OUT/'design.json')['prior_artifacts_sha256']),
        nominal_reward_physics_replayed=True,residual_zero=True,training=False,final_seeds_unused=True)
    if proof['study_complete']:assert prefixes==10
    path=OUT/'completion_audit.json'
    if path.exists():assert d.read(path)==proof
    else:d.write(path,proof)
    print(f'Audit passed: {len(complete)} episodes; confirmation={proof["confirmation_passed"]}',flush=True)
    return proof


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
