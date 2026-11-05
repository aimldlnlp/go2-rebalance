"""Offline paired comparison; reuses the frozen fault diagnosis definitions."""
from project_paths import project_file
import json
from pathlib import Path
import numpy as np
import diagnose_walking_fault as d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/walking_low_noise_comparison'
PILOTS = {'previous': 'walking_residual_height_pilot', 'low_noise': 'walking_residual_low_noise_pilot'}
METRICS = ['post_net_progress_m', 'post_max_progress_gain_m', 'post_residual_rms_rad',
           'post_residual_bound_fraction', 'post_torque_limit_fraction',
           'post_weak_torque_limit_fraction', 'post_weak_residual_torque_delta_rms_Nm']


def paired_summary(previous, current):
    assert len(previous) == len(current) == 5
    assert [r['seed'] for r in previous] == [r['seed'] for r in current]
    return {'episodes_per_pilot': 5,
            'successes': [sum(r['success'] for r in rows) for rows in [previous, current]],
            'falls': [sum(r['fallen'] for r in rows) for rows in [previous, current]],
            'stalls': [sum(r['stall_window_start_s'] is not None for r in rows) for rows in [previous, current]],
            'fall_rescues': sum(a['fallen'] and not b['fallen'] for a,b in zip(previous,current)),
            'fall_harms': sum(not a['fallen'] and b['fallen'] for a,b in zip(previous,current)),
            'medians': {key: [d.median([r[key] for r in rows]) for rows in [previous,current]] for key in METRICS},
            'paired_median_changes': {key: d.median([b[key]-a[key] for a,b in zip(previous,current)
                                                    if a[key] is not None and b[key] is not None]) for key in METRICS}}


def main():
    assert not OUT.exists(), 'Refusing overwrite'
    for folder in PILOTS.values():
        assert d.read(ROOT/'results'/folder/'completion_audit.json')['audit_passed']
    OUT.mkdir()
    protected = dict(d.read(ROOT/'results/walking_residual_low_noise_pilot/design.json')['prior_artifacts_sha256'])
    for folder in PILOTS.values():
        for path in (ROOT/'results'/folder).rglob('*'):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = d.digest(path)
    d.write(OUT/'protocol.json', {
        'scope': 'Saved traces only, previous height-0.338 pilot versus low-initial-noise pilot; no rollout or training.',
        'seeds': list(range(23000,23005)), 'stages':d.STAGES, 'conditions':d.CONDITIONS,
        'unit':'Matched pose seed and condition at equal checkpoint steps. One training seed per pilot.',
        'metrics':'Frozen diagnose_walking_fault.diagnose: post-onset net/max forward gain, torque limiting, residual RMS/bounds, same-state residual torque delta, mission/fall and stall.',
        'aggregation':'Five episode medians per condition, plus median paired low-minus-previous changes. Never pooled timesteps.',
        'windows':'Frozen onset windows retain observed duration; truncated windows are not padded or treated as full. Whole post-onset summaries have unequal follow-up after falls.',
        'limits':'Exploratory development cohort. Training trajectories differ after initialization; no replicated causal effect of sigma. Same-state torque delta is not a closed-loop counterfactual.',
        'final_seeds_opened':False, 'training_started':False, 'graph_available':False})
    sources = [Path(__file__),ROOT/'tests/test_compare_walking_low_noise.py',ROOT/'research/diagnose_walking_fault.py']
    d.write(OUT/'design.json', {'source_sha256':{str(p.relative_to(ROOT)):d.digest(p) for p in sources},
                              'protocol_sha256':d.digest(OUT/'protocol.json'), 'protected_prior_artifacts_sha256':protected})
    rows=[]; initial={}; baseline={}
    for pilot,folder in PILOTS.items():
        for stage in d.STAGES:
            base=ROOT/'results'/folder/stage
            metadata=d.read(base/'episodes.json')
            assert len(metadata)==25 and {(r['condition'],r['seed']) for r in metadata}=={(c,s) for c in d.CONDITIONS for s in range(23000,23005)}
            healthy={}
            for r in metadata:
                if r['condition']=='healthy':
                    with np.load(base/f"healthy_{r['seed']}.npz",allow_pickle=False) as a:
                        healthy[r['seed']]={k:a[k] for k in ['qpos','qvel','command','applied_torque','contacts','control_residual_actions']}
            for r in metadata:
                path=base/f"{r['condition']}_{r['seed']}.npz"
                assert d.digest(path)==r['trace_sha256'] and not r['interrupted']
                assert d.read(path.with_suffix('.json'))==r
                key=(stage,r['condition'],r['seed'])
                with np.load(path,allow_pickle=False) as a:
                    actual=__import__('hashlib').sha256(a['qpos'][0].tobytes()+a['qvel'][0].tobytes()).hexdigest()
                    assert actual==r['initial_state_sha256']
                    if pilot=='previous':initial[key]=actual
                    else:assert initial[key]==actual
                    if stage=='baseline':
                        if pilot=='previous':baseline[key]=r['trace_sha256']
                        else:assert baseline[key]==r['trace_sha256']
                    result=d.diagnose(a,r,None if r['condition']=='healthy' else healthy[r['seed']])
                    onset=result['anchor_tick']
                    forward=np.array([np.cos(r['initial_yaw']),np.sin(r['initial_yaw'])])
                    progress=(a['qpos'][:,:2]-r['initial_xy'])@forward
                    result['fixed_horizon_progress_m']={str(seconds):
                        float(progress[onset+round(seconds/d.DT)]-progress[onset])
                        if onset is not None and onset+round(seconds/d.DT)<len(progress) else None
                        for seconds in [.5,2.]}
                rows.append({'pilot':pilot,'stage':stage,'condition':r['condition'],'seed':r['seed'],
                             'trace':str(path.relative_to(ROOT)),'trace_sha256':r['trace_sha256'],**result})
            print(f'{pilot}/{stage}: 25 traces',flush=True)
    d.write(OUT/'episodes.json',rows)
    summary=[]
    for stage in d.STAGES:
        for condition in d.CONDITIONS[1:]:
            groups=[[r for r in rows if (r['pilot'],r['stage'],r['condition'])==(p,stage,condition)] for p in PILOTS]
            groups=[sorted(g,key=lambda r:r['seed']) for g in groups]
            summary.append({'stage':stage,'condition':condition,**paired_summary(*groups)})
            summary[-1]['fixed_horizon']={str(seconds): {
                'available_episodes':[sum(r['fixed_horizon_progress_m'][str(seconds)] is not None for r in g) for g in groups],
                'median_progress_m':[d.median([r['fixed_horizon_progress_m'][str(seconds)] for r in g]) for g in groups],
                'warning':'Missing episodes fell before horizon; survivor medians are not full-cohort effects.'}
                for seconds in [.5,2.]}
    d.write(OUT/'summary.json',summary)
    for name,sha in protected.items():assert d.digest(project_file(ROOT, name))==sha,name
    d.write(OUT/'completion_audit.json',{'audit_passed':True,'episodes':len(rows),'fault_pairs':80,
        'learned_fault_pairs':60,'duplicated_baseline_pairs':20,'matched_initial_states':100,
        'prefault_fault_healthy_parity_checks':160,'prior_files_unchanged':len(protected),
        'no_new_rollouts':True,'no_training':True,'final_seeds_unused':True})
    print(json.dumps([r for r in summary if r['stage']=='validation_131072'],indent=2),flush=True)


if __name__=='__main__':main()
