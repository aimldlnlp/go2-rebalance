"""Frozen parity, CPU benchmark, and proposal only: never optimizes a policy."""
from project_paths import project_file
import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
import os
import platform
import time

import numpy as np
import torch

import sudden as s
import walking_calibrated as calibrated
import review_walking as independent
import walking_residual_env as residual


ROOT = s.ROOT
BASELINE = ROOT/'results/walking_navigation_hold'
OUT = ROOT/'results/walking_residual_env'
BENCHMARK_SEEDS = list(range(22000, 22015))
WARMUP_SEEDS = [22099, 22100, 22101]
SOURCES = ['walking_residual_env.py', 'test_walking_residual_env.py', 'verify_walking_residual_env.py']


def prepare():
    assert not OUT.exists(), 'Refusing to overwrite environment verification'
    calibrated.verify_frozen()
    old = s.read(BASELINE/'protocol.json')
    old_design = s.read(BASELINE/'design.json')
    protected = dict(old_design['prior_artifacts_sha256'])
    for folder in [BASELINE, ROOT/'results/project_crosscheck_20261101']:
        for path in folder.rglob('*'):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = s.digest(path)
    for name, sha in protected.items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    used = set(old['prior_recorded_seeds']) | set(old['development_seeds']+old['healthy_seeds']+old['diagnostic_seeds'])
    assert not set(BENCHMARK_SEEDS+WARMUP_SEEDS).intersection(used)
    fixtures = []
    for stage in ['confirmation', 'diagnosis']:
        for row in s.read(BASELINE/stage/'episodes.json'):
            stem = f"{row['condition']}_{row['seed']}"
            path = BASELINE/stage/(stem+'.npz')
            assert s.digest(path) == row['trace_sha256']
            fixtures.append({'stage':stage, 'seed':row['seed'], 'condition':row['condition'],
                'trace':str(path.relative_to(ROOT)), 'trace_sha256':row['trace_sha256'],
                'metadata':str((BASELINE/stage/(stem+'.json')).relative_to(ROOT)),
                'metadata_sha256':s.digest(BASELINE/stage/(stem+'.json'))})
    assert len(fixtures) == 45
    acceptance = {k:v for k,v in old['mission'].items() if k not in ['goal_gain', 'goal_command_deadband_m', 'heading_gain', 'max_yaw_command_rad_s']}
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'mission_acceptance':acceptance, 'navigation':old['navigation'],
        'source_protocol_sha256':s.digest(BASELINE/'protocol.json'),
        'residual_scale_rad':residual.RESIDUAL_SCALE, 'action_shape':[12], 'action_range':[-1.,1.],
        'observation_shape':[60], 'observation_fields':residual.OBSERVATION_FIELDS,
        'reward_weights':residual.REWARD_WEIGHTS,
        'reward_definition':'Tracking: exp(-planar_error^2/0.15^2) and exp(-yaw_error^2/0.5^2). Progress: decrease of goal distance per control interval. Height squared error to 0.27m, tilt radians squared, excess cross-track squared. Stable goal indicator uses unchanged mission position/heading/speed thresholds. Mean normalized residual action squared and action-difference squared penalties. Fall penalty and terminal success bonus. No origin-distance penalty or tilt gate.',
        'timing':{'physics_dt_s':.002, 'control_dt_s':.02, 'deadline_physics_ticks':10000,
            'termination':'fall/nonfinite at its physics tick', 'truncation':'20s deadline; no early success termination'},
        'nominal_policy':'Frozen Go2 RL Lab, CPU eval/inference, no gradients; prepare once per control interval. Residual observes cached current nominal target; prediction/history update moves to the preceding observation boundary without a physics step.',
        'fault':'First pre-physics-step progress >=0.50m, selected calf limited torque multiplied by 0.10, latched. Scenario may appear in diagnostics but never observation.',
        'parity':{'fixtures':fixtures, 'kind':'Historical regression only, not new held-out evaluation',
            'arrays':'Exact shape/value agreement for every archived array, including qpos/qvel, all torques, targets, contacts, command and nominal policy inputs/actions.',
            'metrics_atol':1e-12, 'metric_rule':'Flags/onset exact; scalar metrics permit only floating reduction roundoff, no threshold changes.'},
        'benchmark':{'seeds':BENCHMARK_SEEDS, 'warmup_seeds':WARMUP_SEEDS, 'trials':3,
            'conditions':residual.CONDITIONS, 'action':'Zero residual', 'record_trace':False,
            'measures':'Constructor setup, 10-step warmup, reset and stepping separately. Step timing includes nominal inference/history, physics, residual features, reward, metrics and loop overhead; excludes trace serialization and PPO forward/backward optimization.'},
        'prior_recorded_seeds':sorted(used), 'training':False, 'cuda':False,
        'preview_task':'Must remain disabled; this runner never starts a web preview.'}
    OUT.mkdir()
    s.write(OUT/'protocol.json', protocol)
    source_names = list(dict.fromkeys([*SOURCES, *old_design['source_sha256']]))
    s.write(OUT/'design.json', {'protocol_sha256':s.digest(OUT/'protocol.json'),
        'source_sha256':{name:s.digest(project_file(ROOT, name)) for name in source_names},
        'prior_artifacts_sha256':protected})
    print(f'Frozen: 45 historical parity fixtures, 15 benchmark episodes, {len(protected)} protected artifacts', flush=True)


def verify_sources():
    calibrated.verify_frozen()
    p, d = s.read(OUT/'protocol.json'), s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json') == d['protocol_sha256']
    for name, sha in d['source_sha256'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    assert torch.version.cuda is None
    return p


def differences(actual, reference):
    result = {}
    for key in reference.files:
        got, want = actual[key], reference[key]
        same_shape = got.shape == want.shape
        exact = same_shape and np.array_equal(got, want)
        maximum = float(np.max(np.abs(got.astype(np.float64)-want.astype(np.float64)))) if same_shape else None
        result[key] = {'shape_matches':same_shape, 'exact_values':bool(exact), 'max_abs_error':maximum}
    return result


def metric_differences(actual, reference, tolerance):
    result = {}
    for key, value in actual.items():
        expected = reference[key]
        if isinstance(value, float):
            result[key] = bool(abs(value-expected) <= tolerance)
        else:
            result[key] = value == expected
    return result


def parity(p):
    folder = OUT/'parity'
    folder.mkdir()
    env = residual.WalkingResidualEnv('healthy', record_trace=True)
    results = []
    try:
        for fixture in p['parity']['fixtures']:
            env.reset(seed=fixture['seed'], options={'condition':fixture['condition']})
            done = False
            started = time.perf_counter()
            while not done:
                _, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
                assert np.isfinite(reward)
                done = terminated or truncated
            elapsed = time.perf_counter()-started
            arrays = env.trace()
            name = f"{fixture['stage']}_{fixture['condition']}_{fixture['seed']}"
            path = folder/(name+'.npz')
            with path.open('xb') as handle:
                np.savez_compressed(handle, **arrays)
            reference = s.read(ROOT/fixture['metadata'])
            assert s.digest(ROOT/fixture['metadata']) == fixture['metadata_sha256']
            assert s.digest(ROOT/fixture['trace']) == fixture['trace_sha256']
            with np.load(ROOT/fixture['trace'], allow_pickle=False) as old:
                array_checks = differences(arrays, old)
            metrics = metric_differences(info['episode_result'], reference, p['parity']['metrics_atol'])
            passed = all(r['exact_values'] for r in array_checks.values()) and all(metrics.values())
            row = {**fixture, 'passed':passed, 'array_comparison':array_checks,
                'metric_comparison':metrics, 'actual_result':info['episode_result'],
                'actual_trace':str(path.relative_to(ROOT)), 'actual_trace_sha256':s.digest(path),
                'physics_states':len(arrays['qpos']), 'wall_s_with_recording':elapsed}
            s.write(folder/(name+'.json'), row)
            results.append(row)
            print(f"parity/{name}: exact={passed} ticks={env.tick} success={info['episode_result']['success']}", flush=True)
    finally:
        env.close()
    result = {'all_passed':all(r['passed'] for r in results), 'episodes':results,
        'physics_states':sum(r['physics_states'] for r in results)}
    s.write(OUT/'parity.json', result)
    assert result['all_passed'], 'Parity differs: inspect preserved per-fixture reports before proceeding'


def benchmark(p):
    measurements, setup = [], []
    zero = np.zeros(12, dtype=np.float32)
    for trial in range(3):
        started = time.perf_counter()
        env = residual.WalkingResidualEnv('healthy', record_trace=False)
        constructor = time.perf_counter()-started
        started = time.perf_counter()
        env.reset(seed=p['benchmark']['warmup_seeds'][trial])
        for _ in range(10):
            env.step(zero)
        warmup = time.perf_counter()-started
        setup.append({'trial':trial, 'constructor_s':constructor, 'warmup_reset_and_10_steps_s':warmup})
        try:
            for i, condition in enumerate(residual.CONDITIONS):
                seed = p['benchmark']['seeds'][trial*5+i]
                started = time.perf_counter()
                env.reset(seed=seed, options={'condition':condition})
                reset_s = time.perf_counter()-started
                done, steps = False, 0
                started = time.perf_counter()
                while not done:
                    _, reward, terminated, truncated, info = env.step(zero)
                    assert np.isfinite(reward)
                    steps += 1
                    done = terminated or truncated
                elapsed = time.perf_counter()-started
                assert steps == math.ceil(env.tick/10)
                row = {'trial':trial, 'condition':condition, 'seed':seed, 'reset_s':reset_s,
                    'control_steps':steps, 'physics_steps':env.tick, 'stepping_s':elapsed,
                    'control_steps_per_s':steps/elapsed, 'simulated_s':env.tick*.002,
                    'terminated':terminated, 'truncated':truncated}
                measurements.append(row)
                print(f"benchmark/{trial}/{condition}: {steps/elapsed:.1f} control steps/s, reset {reset_s:.4f}s", flush=True)
        finally:
            env.close()
    summary = {'setup':setup, 'episodes':measurements,
        'aggregate_control_steps_per_s':sum(r['control_steps'] for r in measurements)/sum(r['stepping_s'] for r in measurements),
        'min_control_steps_per_s':min(r['control_steps_per_s'] for r in measurements),
        'max_control_steps_per_s':max(r['control_steps_per_s'] for r in measurements),
        'cpu_logical_count':os.cpu_count(), 'processor':platform.processor(), 'torch_threads':torch.get_num_threads(),
        'trace_recording':False, 'nominal_inference_included':True, 'ppo_training_measured':False}
    s.write(OUT/'benchmark.json', summary)


def propose_pilot(p):
    b = s.read(OUT/'benchmark.json')
    rate = b['min_control_steps_per_s']
    steps, validation_steps = 131072, 3*25*1000
    floor = (steps+validation_steps)/rate
    reset_allowance = (math.ceil(steps/1000)+75)*max(r['reset_s'] for r in b['episodes'])
    floor += reset_allowance+max(r['constructor_s'] for r in b['setup'])
    validation, heldout = list(range(23000,23005)), list(range(23100,23120))
    training_resets = list(range(24000,25000))
    used = set(p['prior_recorded_seeds']+p['benchmark']['seeds']+p['benchmark']['warmup_seeds'])
    assert not set(validation+heldout+training_resets).intersection(used)
    proposal = {'status':'Proposal only; not executed or evidence of learning',
        'algorithm':'SB3 PPO MlpPolicy, CPU', 'training_seed':0, 'reset_seed_pool':training_resets,
        'scenario_sampling':'Uniform healthy + four individual calf faults through condition=mixed; log per-condition exposure.',
        'nominal_policy':'Frozen; no optimizer receives nominal parameters', 'residual_scale_rad':residual.RESIDUAL_SCALE,
        'hyperparameters':{'n_steps':1024, 'batch_size':64, 'n_epochs':5, 'learning_rate':.0003,
            'gamma':.99, 'gae_lambda':.95, 'net_arch':[64,64]},
        'max_training_control_steps':steps, 'checkpoints':[32768,65536,131072],
        'validation_seeds':validation, 'validation_conditions':residual.CONDITIONS,
        'maximum_validation_control_steps':validation_steps,
        'conservative_observed_env_control_steps_per_s':rate,
        'environment_only_time_floor_s':floor,
        'assumed_total_time_range_s':[2*floor,4*floor],
        'wall_time_cap_s':min(7200,max(900,math.ceil(4*floor/300)*300)),
        'timing_caveat':'2x-4x is an unmeasured planning allowance for residual inference, PPO gradients, callbacks, serialization and scheduling. Stepping is measured, training is not. Wall cap wins if throughput is worse; no guarantee all steps finish.',
        'promotion_gate':'Proposed: validation healthy 5/5, each calf fault >=3/5, zero falls across 25 episodes, with every original mission threshold unchanged. These are pilot promotion targets, not achieved reliability.',
        'selection':'Among checkpoints satisfying the promotion gate, maximize worst-condition successes then total successes; break ties by lower residual effort then earlier checkpoint. Repeated validation is development evidence, not final generalization.',
        'reserved_later_test_seeds':heldout, 'final_test':'Unused reservation only. Freeze a later acceptance protocol before opening these seeds; do not tune on them.',
        'stop_conditions':['Stop immediately on nonfinite observations/actions/rewards/parameters; save diagnostics.',
            'Stop at 131072 training steps or wall cap, whichever comes first.',
            'Retain failed checkpoints/episodes; if no checkpoint meets gate, report failure without extending budget or tuning automatically.',
            'Three training seeds, other residual bounds, curriculum changes or teacher training belong to a subsequent explicitly scoped study.'],
        'mission_acceptance':p['mission_acceptance'], 'reward_weights':p['reward_weights'], 'training_executed':False}
    s.write(OUT/'pilot_proposal.json', proposal)


def run():
    assert not (OUT/'execution.json').exists(), 'Refusing to rerun verification'
    p = verify_sources()
    tests = (OUT/'tests-before.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    s.write(OUT/'execution.json', {'started_utc':datetime.now(timezone.utc).isoformat(),
        'tests_sha256':s.digest(OUT/'tests-before.log'), 'device':'cpu', 'training':False,
        'versions':{name:importlib.metadata.version(name) for name in ['numpy','mujoco','gymnasium','stable-baselines3','torch']}})
    parity(p)
    benchmark(p)
    propose_pilot(p)
    s.write(OUT/'outcome.json', {'finished_utc':datetime.now(timezone.utc).isoformat(),
        'parity_passed':True, 'benchmark_completed':True, 'pilot_training_executed':False})
    audit()


def audit():
    p = verify_sources()
    d, result = s.read(OUT/'design.json'), s.read(OUT/'parity.json')
    assert result['all_passed'] and len(result['episodes']) == 45
    expected = {(f['stage'], f['condition'], f['seed']) for f in p['parity']['fixtures']}
    assert {(r['stage'],r['condition'],r['seed']) for r in result['episodes']} == expected
    states = 0
    for row in result['episodes']:
        assert s.digest(ROOT/row['actual_trace']) == row['actual_trace_sha256']
        reference = s.read(ROOT/row['metadata'])
        with np.load(ROOT/row['actual_trace'], allow_pickle=False) as got, np.load(ROOT/row['trace'], allow_pickle=False) as want:
            assert all(v['exact_values'] for v in differences(got, want).values())
        assert all(metric_differences(row['actual_result'],reference,p['parity']['metrics_atol']).values())
        independent_row = {**reference, **row['actual_result'], 'trace_sha256':row['actual_trace_sha256']}
        states += independent.verify_episode(ROOT/row['actual_trace'], independent_row, p)
    b, pilot = s.read(OUT/'benchmark.json'), s.read(OUT/'pilot_proposal.json')
    assert len(b['episodes']) == 15 and len(b['setup']) == 3 and not b['ppo_training_measured']
    assert [r['seed'] for r in b['episodes']] == p['benchmark']['seeds']
    for row in b['episodes']:
        assert row['control_steps'] == math.ceil(row['physics_steps']/10) and 0 < row['physics_steps'] <= 10000
        assert row['control_steps_per_s'] == row['control_steps']/row['stepping_s']
    assert not pilot['training_executed'] and pilot['max_training_control_steps'] == 131072
    assert pilot['mission_acceptance'] == p['mission_acceptance']
    assert pilot['environment_only_time_floor_s'] > 0 and 900 <= pilot['wall_time_cap_s'] <= 7200
    execution = s.read(OUT/'execution.json')
    assert execution['tests_sha256'] == s.digest(OUT/'tests-before.log')
    assert datetime.fromisoformat(p['created_utc']) < datetime.fromisoformat(execution['started_utc'])
    for name, sha in d['prior_artifacts_sha256'].items():
        assert s.digest(project_file(ROOT, name)) == sha, name
    proof = {'goal_requirements_achieved':True, 'api_and_tests_passed':True,
        'parity_fixtures':45, 'all_archived_arrays_exact':True, 'independent_mission_and_torque_audit_passed':True,
        'physics_states_checked':states, 'benchmark_episodes':15,
        'protected_prior_artifacts_unchanged':len(d['prior_artifacts_sha256']),
        'pilot_proposed_only':True, 'training_executed':False, 'cuda':False}
    path = OUT/'completion_audit.json'
    if path.exists():
        assert s.read(path) == proof
    else:
        s.write(path, proof)
    report = '\n'.join(['# WalkingResidualEnv — verifikasi tanpa training', '',
        f"**Lulus:** API dan tes, parity 45/45 fixture dengan seluruh array identik, benchmark CPU 15 episode, serta protokol usulan pilot. Audit independen memeriksa {states:,} state fisika. Tidak ada training.", '',
        '## Kontrak environment', '',
        '`WalkingResidualEnv(condition="mixed", record_trace=False)` menyediakan API Gymnasium. Action 12 nilai [-1,1] menghasilkan residual target maksimal ±0,10 rad sebelum joint clipping, torque-speed/physical torque clipping dan fault. Nominal walker tetap CPU eval/inference dengan parameter tanpa gradien. Observasi 60 dimensi berisi body velocity, gyro, gravity, command, posisi/kecepatan joint, target nominal dan residual sebelumnya; tidak memuat label motor, strength, waktu atau posisi goal. Semantiknya berbeda dari student standing 60 dimensi.', '',
        'Physics, passive calibration, sumber gains dan navigator latch sama dengan baseline walking. Awal 0,02 s menahan target default. Nominal prediction berikutnya disiapkan pada batas observasi sebelum residual dipilih, dengan state/history yang sama. Jatuh/nonfinite menghentikan episode pada tick fisika; deadline 20 s menghasilkan truncation. Keberhasilan tidak menghentikan episode lebih awal agar hold sampai deadline dan parity penuh terjaga.', '',
        'Reward mendukung tracking planar/yaw, pengurangan jarak ke goal, tinggi tubuh, tilt, safety, stable goal, serta penalti residual dan perubahannya. Tidak ada penalti kembali ke origin atau gate tilt. Bobot dan definisi dibekukan dalam protocol.json; validasi API/reward belum membuktikan bahwa reward akan menghasilkan policy yang belajar.', '',
        '## Parity', '',
        '20 konfirmasi dan 25 diagnosis historis digunakan hanya sebagai fixture regresi. Seluruh qpos/qvel, requested/limited/applied torque, actual actuator force, target/gains, command, contacts, serta observasi/raw/clipped action nominal dibandingkan tepat. Onset, flags dan hasil misi sama; toleransi 1e-12 hanya untuk pengurangan scalar metric. Trace hasil replay dan laporan tiap fixture tersimpan di parity/. Kegagalan fault lama tetap gagal pada residual nol; tidak ada klaim robustness baru.', '',
        '## Benchmark CPU', '',
        f"Throughput agregat: {b['aggregate_control_steps_per_s']:.1f} control steps/s; rentang episode {b['min_control_steps_per_s']:.1f}–{b['max_control_steps_per_s']:.1f}. Torch satu thread. Constructor median {np.median([r['constructor_s'] for r in b['setup']]):.4f} s; reset median {np.median([r['reset_s'] for r in b['episodes']]):.4f} s. Setup/warmup/reset dipisahkan dari stepping. Stepping mencakup policy nominal, physics, fitur residual dan reward, tanpa trace recording/serialization.", '',
        'Benchmark memakai tiga ulangan kondisi sehat + empat fault, action residual nol, dengan seed kinerja baru. Durasi episode dapat lebih pendek karena fall, sehingga agregat dihitung dari total langkah/total waktu; bukan rata-rata angka throughput episode. Benchmark ini tidak mengukur forward/backward PPO atau inference residual terlatih.', '',
        '## Usulan pilot satu seed — belum dieksekusi', '',
        f"PPO MLP [64,64], seed training 0, n_steps=1024, batch=64, n_epochs=5, residual ±0,10 rad. Maksimum 131.072 langkah training, checkpoint 32.768/65.536/131.072, validasi 25 misi per checkpoint pada seed 23000–23004. Environment-only time floor termasuk maksimum 75.000 langkah validasi: {pilot['environment_only_time_floor_s']/60:.1f} menit. Perkiraan perencanaan dengan asumsi overhead 2–4×: {pilot['assumed_total_time_range_s'][0]/60:.1f}–{pilot['assumed_total_time_range_s'][1]/60:.1f} menit. Batas wall-time yang diusulkan {pilot['wall_time_cap_s']/60:.0f} menit, maksimum langkah atau waktu mana yang lebih dahulu.", '',
        'Asumsi overhead bukan pengukuran training dan tidak menjanjikan runtime. Gate promosi yang diusulkan: sehat 5/5, setiap motor fault minimal 3/5, tanpa jatuh, dengan seluruh ambang misi tetap. Simpan semua checkpoint gagal; stop segera pada nonfinite, stop di budget, dan jangan memperluas otomatis jika tidak lolos. Seed later test 23100–23119 hanya reservasi; belum diuji dan membutuhkan protokol penerimaan berikutnya. Rincian split, hyperparameter, checkpoint selection dan stop conditions ada di pilot_proposal.json.', '',
        '## Batas dan reproduksi', '',
        'Tidak ada optimizer atau training dijalankan. Batas residual dan reward belum divalidasi untuk kemampuan belajar atau stabilitas residual nonzero sepanjang misi. Ini persiapan software pada simulasi datar; bukan bukti kompensasi, hardware atau generalisasi. Kalibrasi pasif berbeda dari standing dan timestep lokal berbeda dari deployment asli. Preview Data tidak diaktifkan oleh runner.', '',
        'Pemakaian: import `WalkingResidualEnv` dari walking_residual_env.py, panggil reset(seed=...), lalu step(action). Gunakan condition="healthy" atau nama calf untuk fixture tetap; condition="mixed" memilih uniform saat reset. record_trace=False untuk training/benchmark, True untuk audit. Body linear velocity berasal dari simulator dan nantinya perlu estimator pada hardware.', '',
        'Recheck: `.venv\\Scripts\\python.exe verify_walking_residual_env.py audit`. Tahap prepare/run menolak overwrite; jangan menjalankan ulang hasil yang sudah selesai.'])+'\n'
    path = OUT/'REPORT.md'
    if path.exists():
        assert path.read_text(encoding='utf-8') == report
    else:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(report)
    print(report.split('## Kontrak')[0], flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run','audit'])
    {'prepare':prepare, 'run':run, 'audit':audit}[parser.parse_args().stage]()
