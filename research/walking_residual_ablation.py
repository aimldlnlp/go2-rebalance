"""Data diagnosis and one preregistered 5-pair zero-command residual ablation."""
from project_paths import project_file
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import sudden as s
import walking_residual_env as w
import walking_residual_pilot as pilot
import review_walking as independent
from walking_baseline import rotation


ROOT = s.ROOT
OUT = ROOT/'results/walking_residual_ablation'
SEEDS = list(range(23200,23205))
MODES = ['always', 'zero_command_off']
TERMS = list(w.REWARD_WEIGHTS)


def gated_action(proposed, observation, mode):
    action, obs = np.asarray(proposed), np.asarray(observation)
    if mode not in MODES or action.shape!=(12,) or obs.shape!=(60,) or not np.isfinite(action).all() or not np.isfinite(obs).all():
        raise ValueError('Invalid mode/action/observation')
    off = bool(mode=='zero_command_off' and np.all(obs[9:11]==0))
    return np.zeros_like(action) if off else action.copy(), off


def rotations(q):
    qw,qx,qy,qz=q[:,3:7].T
    return np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]]).transpose(2,0,1)


def decompose(a, row):
    q,v=a['qpos'],a['qvel']
    n=len(q)-1
    ticks=np.arange(0,n,10)
    end=np.minimum(ticks+10,n)
    command=a['command'][ticks]
    actions=a['control_residual_actions']
    assert len(actions)==len(ticks)
    r=rotations(q)
    origin=np.array(row['initial_xy'])
    forward=np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    error=np.linalg.norm(q[:,:2]-(origin+2*forward),axis=1)
    yaw=np.arctan2(r[:,1,0],r[:,0,0])
    heading=np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
    tilt=np.arccos(np.clip(r[:,2,2],-1,1))
    cross=np.abs((q[:,:2]-origin)@np.array([-forward[1],forward[0]]))
    speed=np.linalg.norm(v[:,:2],axis=1)
    stable=(error<=.15)&(heading<=np.deg2rad(15))&(speed<=.05)
    body=np.einsum('nji,nj->ni',r[end],v[end,:3])
    previous=np.vstack([np.zeros(12),actions[:-1]])
    fall=np.zeros(len(ticks));fall[-1]=float(row['fallen'])
    success=np.zeros(len(ticks));success[-1]=float(row['success'])
    raw={'velocity_tracking':np.exp(-np.sum((body[:,:2]-command[:,:2])**2,axis=1)/.15**2),
        'yaw_tracking':np.exp(-((v[end,5]-command[:,2])/.5)**2),
        'goal_progress':error[ticks]-error[end], 'height_error':(q[end,2]-.27)**2,
        'tilt_squared':tilt[end]**2, 'cross_track_excess':np.maximum(0.,cross[end]-.20)**2,
        'stable_goal':stable[end].astype(float), 'residual_effort':np.mean(actions**2,axis=1),
        'residual_change':np.mean((actions-previous)**2,axis=1),'fall':fall,'mission_success':success}
    weighted={key:w.REWARD_WEIGHTS[key]*raw[key] for key in TERMS}
    rewards=sum(weighted.values())
    np.testing.assert_allclose(rewards,a['control_rewards'],rtol=0,atol=1e-10)
    assert abs(float(weighted['goal_progress'].sum())-10*(error[0]-error[-1]))<1e-10
    entered=np.maximum.accumulate((ticks>=1000)&(error[ticks]<=.12))
    phase=np.full(len(ticks),'walking',dtype='<U12')
    phase[ticks<1000]='settle'
    phase[(ticks>=1000)&entered]='stop'
    phase[(ticks>=1000)&~entered&np.all(command[:,:2]==0,axis=1)]='ramp_zero'
    phase_rows=[]
    for name in ['settle','ramp_zero','walking','stop']:
        mask=phase==name
        phase_rows.append({'phase':name,'controls':int(mask.sum()),
            'duration_s':float(np.sum(end[mask]-ticks[mask])*.002),
            'residual_rms_rad':float(.1*np.sqrt(np.mean(actions[mask]**2))) if mask.any() else None,
            'mean_speed_m_s':float(speed[end[mask]].mean()) if mask.any() else None,
            'stable_fraction':float(stable[end[mask]].mean()) if mask.any() else None,
            'reward_contributions':{key:float(value[mask].sum()) for key,value in weighted.items()}})
    return phase_rows, weighted


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite ablation study'
    p=pilot.verify_sources()
    assert s.read(pilot.OUT/'completion_audit.json')['audit_passed']
    assert s.digest(pilot.OUT/'checkpoint_131072.zip')==s.read(pilot.OUT/'checkpoint_131072.json')['checkpoint_sha256']
    protected=dict(s.read(pilot.OUT/'design.json')['prior_artifacts_sha256'])
    for path in pilot.OUT.rglob('*'):
        if path.is_file():protected[str(path.relative_to(ROOT))]=s.digest(path)
    for name,sha in protected.items():assert s.digest(project_file(ROOT, name))==sha,name
    used=set(p['validation_seeds']+p['reserved_later_test_seeds']+p['reset_seed_pool'])
    env_protocol=s.read(ROOT/'results/walking_residual_env/protocol.json')
    used.update(env_protocol['prior_recorded_seeds']+env_protocol['benchmark']['seeds']+env_protocol['benchmark']['warmup_seeds'])
    assert not set(SEEDS).intersection(used)
    for seed in SEEDS:
        assert not any((ROOT/'results').rglob(f'*_{seed}.npz')), f'Seed {seed} already has a trace'
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'question':'Does suppressing residual at zero planar command improve healthy mission completion for the last checkpoint?',
        'checkpoint_steps':131072,'checkpoint':str((pilot.OUT/'checkpoint_131072.zip').relative_to(ROOT)),
        'checkpoint_sha256':s.digest(pilot.OUT/'checkpoint_131072.zip'),
        'seeds':SEEDS,'modes':MODES,'conditions':['healthy'],'episodes':10,
        'intervention':'Set all 12 residual actions to zero iff both observed planar command components [9:11] are exactly zero. Keep yaw command, nominal policy, weights, reward, physics, navigator, bounds and mission criteria unchanged.',
        'causal_scope':'Combined settle/ramp-zero/stop intervention, not a stop-only effect. Closed-loop trajectories and later commands may change; same initial qpos/qvel are paired.',
        'phases':'settle <2s; walking otherwise until first >=2s control-state inside 0.12m stop radius; latched stop thereafter; ramp_zero marks any pre-stop exact-zero planar command at/after 2s.',
        'mission_acceptance':p['mission_acceptance'],'prior_seed_membership':sorted(used),
        'reward_reconstruction_atol':1e-10,'training':False,'final_test_run':False,
        'historical_analysis':'Baseline and all three checkpoints, all 100 saved validation episodes. Recompute reward from physics/control traces. Additive phase/component contributions use sums and equal episode weights; missing-phase RMS stays null.',
        'decision':'If gated successes exceed always successes on the five paired seeds, nominate command-based zero-residual gating as one next revision to verify. Do not declare healthy robustness, fault recovery, or isolate stop versus settle. If no improvement, do not adopt gating; prioritize a different limited reward/observation study.',
        'pilot_promotion':'Not tested; this study is healthy-only and cannot pass the prior fault promotion gate.'}
    OUT.mkdir()
    s.write(OUT/'protocol.json',protocol)
    names=[*s.read(pilot.OUT/'design.json')['source_sha256'],'walking_residual_ablation.py','test_walking_residual_ablation.py']
    s.write(OUT/'design.json',{'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in dict.fromkeys(names)},'prior_artifacts_sha256':protected})
    print(f'Frozen 5 paired healthy seeds / 10 episodes, checkpoint fixed; {len(protected)} old files protected',flush=True)


def verify_sources():
    pilot.verify_sources()
    p,d=s.read(OUT/'protocol.json'),s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==d['protocol_sha256']
    assert s.digest(ROOT/p['checkpoint'])==p['checkpoint_sha256']
    for name,sha in d['source_sha256'].items():assert s.digest(project_file(ROOT, name))==sha,name
    return p


def historical_analysis():
    rows=[]
    for stage in ['baseline','validation_32768','validation_65536','validation_131072']:
        for row in s.read(pilot.OUT/stage/'episodes.json'):
            path=pilot.OUT/stage/f"{row['condition']}_{row['seed']}.npz"
            assert s.digest(path)==row['trace_sha256']
            with np.load(path,allow_pickle=False) as a:
                phases,_=decompose(a,row)
            rows.append({'stage':stage,'condition':row['condition'],'seed':row['seed'],
                'success':row['success'],'fallen':row['fallen'],'episode_return':row['episode_return'],
                'failed_checks':[k for k,v in row['checks'].items() if not v],'phases':phases})
    assert len(rows)==100
    aggregate=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        for condition in w.CONDITIONS:
            group=[r for r in rows if r['stage']==stage and r['condition']==condition]
            assert len(group)==5
            phase_summary=[]
            for phase in ['settle','ramp_zero','walking','stop']:
                items=[next(v for v in r['phases'] if v['phase']==phase) for r in group]
                present=[v for v in items if v['controls']]
                phase_summary.append({'phase':phase,'episodes_present':len(present),
                    'mean_duration_s':float(np.mean([v['duration_s'] for v in items])),
                    'median_residual_rms_rad':float(np.median([v['residual_rms_rad'] for v in present])) if present else None,
                    'mean_reward_contributions':{key:float(sum(v['reward_contributions'][key] for v in items)/5) for key in TERMS}})
            total=sum(sum(v['mean_reward_contributions'].values()) for v in phase_summary)
            expected=float(np.mean([r['episode_return'] for r in group]))
            assert abs(total-expected)<1e-8
            aggregate.append({'stage':stage,'condition':condition,'episodes':5,
                'successes':sum(r['success'] for r in group),'mean_episode_return':expected,
                'failed_checks':dict(Counter(k for r in group for k in r['failed_checks'])), 'phases':phase_summary})
    s.write(OUT/'phase_analysis.json',{'episodes':rows,'groups':aggregate,
        'reward_recomputed_episodes':100,'reward_tolerance':1e-10,'phase_contributions_reconcile':True})
    print('Historical reward reconstruction: 100/100 traces, phase/component totals reconcile',flush=True)


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun ablation'
    p=verify_sources()
    text=(OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in text and 'FAILED' not in text
    s.write(OUT/'execution.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'tests_sha256':s.digest(OUT/'tests-before.log'),'training':False,'device':'cpu'})
    historical_analysis()
    model=PPO.load(ROOT/p['checkpoint'],device='cpu')
    for parameter in model.policy.parameters():parameter.requires_grad_(False)
    model.policy.set_training_mode(False)
    env=w.WalkingResidualEnv('healthy',record_trace=True)
    rows=[]
    try:
        for mode in MODES:
            folder=OUT/mode;folder.mkdir()
            for seed in SEEDS:
                obs,_=env.reset(seed=seed)
                initial_hash=s.digest_bytes(env.data.qpos.tobytes()+env.data.qvel.tobytes())
                inputs,proposed,applied,masks,rewards,components=[],[],[],[],[],[]
                done=False
                while not done:
                    action=model.predict(obs,deterministic=True)[0]
                    chosen,off=gated_action(action,obs,mode)
                    inputs.append(obs.copy());proposed.append(action.copy());applied.append(chosen.copy());masks.append(off)
                    obs,reward,terminated,truncated,info=env.step(chosen)
                    rewards.append(reward);components.append([info['reward_terms'].get(key,0.) for key in TERMS])
                    done=terminated or truncated
                a=env.trace()
                a.update({'residual_observations':np.asarray(inputs),'proposed_residual_actions':np.asarray(proposed),
                    'applied_residual_actions':np.asarray(applied),'residual_disabled':np.asarray(masks),
                    'control_rewards':np.asarray(rewards),'reward_components':np.asarray(components)})
                stem=f'healthy_{seed}';path=folder/(stem+'.npz')
                with path.open('xb') as handle:np.savez_compressed(handle,**a)
                result=env.mission_result()
                row={'seed':seed,'condition':'healthy','candidate':'go2_rl_lab','mode':mode,
                    'initial_xy':env.origin.tolist(),'initial_yaw':env.initial_yaw,'initial_state_sha256':initial_hash,
                    'trace_sha256':s.digest(path),'checkpoint_sha256':p['checkpoint_sha256'],
                    'residual_disabled_controls':int(sum(masks)), 'episode_return':float(sum(rewards)),**result}
                s.write(folder/(stem+'.json'),row);rows.append(row)
                print(f"ablation/{mode}/{seed}: success={row['success']} hold={row.get('final_stable_hold_s')} disabled={sum(masks)}",flush=True)
    finally:env.close()
    s.write(OUT/'episodes.json',rows)
    s.write(OUT/'outcome.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'episodes_completed':len(rows),'training':False})
    audit()


def audit():
    p=verify_sources()
    rows=s.read(OUT/'episodes.json');d=s.read(OUT/'design.json')
    assert len(rows)==10 and {(r['mode'],r['seed']) for r in rows}=={(m,seed) for m in MODES for seed in SEEDS}
    states=0
    model=PPO.load(ROOT/p['checkpoint'],device='cpu')
    default=np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
    for row in rows:
        path=OUT/row['mode']/f"healthy_{row['seed']}.npz"
        assert row==s.read(path.with_suffix('.json'))
        states+=independent.verify_episode(path,row,p)
        with np.load(path,allow_pickle=False) as a:
            q,v=a['qpos'],a['qvel'];inputs=a['residual_observations'];actions=a['control_residual_actions']
            proposals=a['proposed_residual_actions'];mask=a['residual_disabled'];commands=a['command']
            expected_mask=np.all(inputs[:,9:11]==0,axis=1) if row['mode']=='zero_command_off' else np.zeros(len(inputs),dtype=bool)
            np.testing.assert_array_equal(mask,expected_mask)
            expected_actions=proposals.copy();expected_actions[mask]=0
            np.testing.assert_array_equal(actions,expected_actions)
            np.testing.assert_array_equal(a['applied_residual_actions'],expected_actions)
            np.testing.assert_allclose(proposals,np.asarray([model.predict(obs,deterministic=True)[0] for obs in inputs]),rtol=0,atol=1e-7)
            nominal=default+.25*a['control_actions']
            target=np.clip(nominal+.1*actions,np.array([[-1.0472,-1.5708,-2.7227]]*2+[[-1.0472,-.5236,-2.7227]]*2).reshape(12),
                np.array([[1.0472,3.4907,-.83776]]*2+[[1.0472,4.5379,-.83776]]*2).reshape(12))
            np.testing.assert_array_equal(a['target'],np.repeat(target,10,axis=0)[:len(q)-1])
            expected_inputs=[]
            for i,tick in enumerate(range(0,len(q)-1,10)):
                r=rotation(q[tick,3:7]);previous=np.zeros(12) if i==0 else actions[i-1]
                expected_inputs.append(np.r_[r.T@v[tick,:3],v[tick,3:6],r.T@np.array([0.,0.,-1.]),
                    commands[tick],q[tick,7:]-default,v[tick,6:]*.05,nominal[i]-default,previous])
            np.testing.assert_array_equal(inputs,np.asarray(expected_inputs,dtype=np.float32))
            _,weighted=decompose(a,row)
            np.testing.assert_allclose(a['reward_components'],np.column_stack([weighted[key] for key in TERMS]),rtol=0,atol=1e-10)
    paired=[]
    for seed in SEEDS:
        before=next(r for r in rows if r['seed']==seed and r['mode']=='always')
        after=next(r for r in rows if r['seed']==seed and r['mode']=='zero_command_off')
        assert before['initial_state_sha256']==after['initial_state_sha256']
        paired.append({'seed':seed,'always_success':before['success'],'gated_success':after['success'],
            'goal_error_change_m':after['final_goal_error_m']-before['final_goal_error_m'],
            'hold_change_s':after['final_stable_hold_s']-before['final_stable_hold_s']})
    history=s.read(OUT/'phase_analysis.json')
    assert len(history['episodes'])==100 and history['phase_contributions_reconcile']
    assert s.read(OUT/'execution.json')['tests_sha256']==s.digest(OUT/'tests-before.log')
    for name,sha in d['prior_artifacts_sha256'].items():assert s.digest(project_file(ROOT, name))==sha,name
    counts={mode:sum(r['success'] for r in rows if r['mode']==mode) for mode in MODES}
    rescued=sum(not r['always_success'] and r['gated_success'] for r in paired)
    harmed=sum(r['always_success'] and not r['gated_success'] for r in paired)
    result={'audit_passed':True,'historical_reward_reconstructions':100,'ablation_episodes':10,
        'physics_states_checked':states,'successes':counts,'paired':paired,'rescued':rescued,'harmed':harmed,
        'candidate_revision_supported_in_this_cohort':counts['zero_command_off']>counts['always'],
        'protected_prior_artifacts_unchanged':len(d['prior_artifacts_sha256']),'training':False,'final_test_run':False}
    path=OUT/'completion_audit.json'
    if path.exists():assert s.read(path)==result
    else:s.write(path,result)
    write_report(result,history,rows)
    return result


def write_report(result,history,rows):
    lines=['# Diagnosis Data dan ablation residual command nol','',
        f"**Uji sehat berpasangan:** mode biasa {result['successes']['always']}/5, residual nol saat command translasi nol {result['successes']['zero_command_off']}/5. Rescue {result['rescued']} pasangan; harm {result['harmed']} pasangan. Tidak ada training.",'',
        '| Seed | Biasa | Gate command nol | Perubahan error (m) | Perubahan hold (s) |','|---|---|---|---:|---:|']
    for r in result['paired']:
        lines.append(f"| {r['seed']} | {r['always_success']} | {r['gated_success']} | {r['goal_error_change_m']:+.3f} | {r['hold_change_s']:+.3f} |")
    lines+=['','## Diagnosis trace historis','',
        'Komponen reward dihitung ulang dari 100 trace baseline/ketiga checkpoint dan cocok dengan reward tersimpan (atol 1e-10). Fase dan komponen menjumlah tepat ke episode return. Seed validasi lama ini adalah data pengembangan, bukan evaluasi baru.','',
        '| Controller | Sehat | Mean return sehat | Settle RMS residual (rad) | Walking RMS residual (rad) | Stop RMS residual (rad) | Episode dengan fase stop |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in [g for g in history['groups'] if g['condition']=='healthy']:
        phases={r['phase']:r for r in group['phases']}
        rms=lambda phase: f"{phases[phase]['median_residual_rms_rad']:.4f}" if phases[phase]['median_residual_rms_rad'] is not None else 'tidak ada fase'
        lines.append(f"| {group['stage']} | {group['successes']}/5 | {group['mean_episode_return']:.2f} | {rms('settle')} | {rms('walking')} | {rms('stop')} | {phases['stop']['episodes_present']}/5 |")
    lines+=['','Mean return dan kontribusi reward adalah penjumlahan komponen berbobot, dibagi seluruh lima episode setiap cohort. RMS per fase adalah median RMS per episode yang benar-benar memiliki fase tersebut; fase hilang tetap null, tidak dianggap residual nol. Phase ramp_zero di detik 2,0 disimpan terpisah. Breakdown semua kondisi fault dan setiap komponen tersedia di phase_analysis.json.','',
        '## Interpretasi dan satu revisi kandidat','',
        ('Kandidat yang didukung cohort ini adalah gate berbasis command: residual nol saat kedua command translasi nol, tanpa label fault dan tanpa mengubah checkpoint/reward. Sebelum dipakai untuk studi berikutnya, verifikasi gate ini pada konfirmasi sehat baru yang lebih besar dan ulangi diagnosis fault dengan protokol baru; studi ini hanya sehat dan tidak membuktikan pemulihan motor.' if result['candidate_revision_supported_in_this_cohort'] else 'Intervensi gate command nol tidak meningkatkan jumlah sukses pada cohort ini, sehingga belum layak diadopsi sebagai perbaikan. Satu revisi studi yang disarankan adalah desain reward yang lebih selaras dengan penyelesaian misi/tinggi tubuh, dengan protokol baru dan audit skala kontribusi dahulu; jangan otomatis mengulang atau memperpanjang training.'),'',
        'Ablation menguji gabungan settle/ramp-zero/stop, bukan efek stop saja. Mengubah action saat command nol juga mengubah previous residual, gerak tubuh, history nominal dan command selanjutnya melalui closed loop. Semua itu bagian dari intervensi controller ini; data tidak mengisolasi mekanisme satu fase atau membuktikan akar penyebab tunggal.','',
        '## Bukti dan batas','',
        f"Checkpoint 131072 tetap dibekukan; 5 seed baru 23200–23204 berpasangan dengan initial qpos/qvel hash yang sama. Audit independen memeriksa {result['physics_states_checked']:,} state, metrik misi/batas torque, observasi biasa, aksi checkpoint, mask, applied residual dan reward. Sebanyak {result['protected_prior_artifacts_unchanged']} artefak sebelumnya tetap utuh.",'',
        'Semua ambang misi, model, passive calibration, gains, torque-speed envelope dan navigator tetap sama. Tidak ada episode fault baru, training, final test, atau replikasi tiga seed. Seed final 23100–23119 tidak digunakan. Lima pasangan hanya bukti pada cohort simulasi ini, bukan reliability populasi atau hardware. Preview Data tidak diaktifkan.','',
        'Recheck: `.venv\\Scripts\\python.exe walking_residual_ablation.py audit`. Prepare/run menolak overwrite.']
    text='\n'.join(lines)+'\n';path=OUT/'REPORT.md'
    if path.exists():assert path.read_text(encoding='utf-8')==text
    else:
        with path.open('x',encoding='utf-8') as handle:handle.write(text)
    print('\n'.join(lines[:5]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','audit'])
    {'prepare':prepare,'run':run,'audit':audit}[parser.parse_args().stage]()
