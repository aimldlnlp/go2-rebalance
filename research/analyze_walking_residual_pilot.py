"""Post-pilot paired readout. Does not run environments or train policies."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np

import sudden as s
import walking_residual_pilot as p
import walking_residual_env as w


def analyze():
    out = p.OUT
    audit = s.read(out/'completion_audit.json')
    assert audit['audit_passed']
    outcome = s.read(out/'outcome.json')
    baseline = s.read(out/'baseline/episodes.json')
    reference = {(r['condition'],r['seed']):r for r in baseline}
    stages = []
    for n in outcome['completed_checkpoint_validations']:
        rows = s.read(out/f'validation_{n}/episodes.json')
        groups = []
        for condition in w.CONDITIONS:
            group = [r for r in rows if r['condition']==condition]
            paired = [reference[(r['condition'],r['seed'])] for r in group]
            assert all(r['initial_state_sha256']==b['initial_state_sha256'] for r,b in zip(group,paired))
            groups.append({'condition':condition,'successes':sum(r['success'] for r in group),
                'falls':sum(r['fallen'] for r in group),
                'failed_checks':dict(Counter(k for r in group for k,v in r['checks'].items() if not v)),
                'median_progress_m':float(np.median([r['final_progress_m'] for r in group])),
                'median_paired_progress_change_m':float(np.median([r['final_progress_m']-b['final_progress_m'] for r,b in zip(group,paired)])),
                'median_goal_error_m':float(np.median([r['final_goal_error_m'] for r in group])),
                'median_residual_rms_rad':float(np.median([.1*np.sqrt(r['residual_effort']) for r in group])),
                'fault_received':sum(r['fault_onset_tick'] is not None for r in group)})
        stages.append({'training_steps':n,'groups':groups,'promotion_passed':s.read(out/f'acceptance_{n}.json')['passed']})
    logs=[__import__('json').loads(line) for line in (out/'training_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    finished=[r for r in logs if r['event']=='episode_end']
    partial=[r for r in logs if r['event']=='partial']
    assert sum(r['control_steps'] for r in finished+partial)==outcome['training_steps']
    exposure=[]
    for condition in w.CONDITIONS:
        group=[r for r in finished if r['condition']==condition]
        exposure.append({'condition':condition,'completed_training_episodes':len(group),
            'training_steps_in_completed_episodes':sum(r['control_steps'] for r in group),
            'falls':sum(r['result']['fallen'] for r in group),
            'mission_successes':sum(r['result']['success'] for r in group),
            'fault_received':sum(r['result']['fault_onset_tick'] is not None for r in group),
            'fault_scenario_ended_before_fault':sum(r['result']['fault_onset_tick'] is None for r in group) if condition!='healthy' else 0})
    with (out/'progress.csv').open(encoding='utf-8-sig',newline='') as handle:
        progress=list(csv.DictReader(handle))
    curve=[{'training_steps':int(float(r['time/total_timesteps'])), 'episode_reward_mean':float(r['rollout/ep_rew_mean']),
        'training_episode_success_rate':float(r['rollout/success_rate'])}
        for r in progress if r.get('rollout/ep_rew_mean') and r.get('rollout/success_rate')]
    data={'source_sha256':s.digest(Path(__file__)),'audit_sha256':s.digest(out/'completion_audit.json'),
        'outcome':outcome,'checkpoint_groups':stages,'training_exposure':exposure,
        'partial_training_episodes':len(partial),'training_learning_curve':curve,
        'interpretation':'All validation comparisons are paired development measurements. Differences in progress alone do not pass the original mission or isolate why the learner failed/succeeded. Training episodes use stochastic actions; validation uses deterministic actions.'}
    s.write(out/'analysis.json',data)
    lines=['# Temuan pilot residual walking','',
        f"Pilot satu seed selesai dengan {outcome['training_steps']:,} langkah dan compute {outcome['compute_wall_s']/60:.1f} menit. Promosi: {outcome['pilot_promoted']}. Tidak ada training tambahan.",'',
        '| Langkah | Kondisi | Berhasil | Jatuh | Median progress (m) | Median selisih berpasangan (m) | RMS residual median (rad) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for stage in stages:
        for g in stage['groups']:
            lines.append(f"| {stage['training_steps']} | {g['condition']} | {g['successes']}/5 | {g['falls']} | {g['median_progress_m']:.3f} | {g['median_paired_progress_change_m']:+.3f} | {g['median_residual_rms_rad']:.4f} |")
    lines += ['',
        'Perubahan progress dibandingkan baseline residual nol pada pose awal yang sama. Median yang lebih baik tidak menggantikan syarat sukses misi atau keselamatan; episode jatuh tetap dihitung gagal.', '',
        '## Exposure training','',
        '| Kondisi | Episode training lengkap | Sukses | Jatuh | Menerima fault | Skenario fault berakhir sebelum onset |',
        '|---|---:|---:|---:|---:|---:|']
    for g in exposure:
        lines.append(f"| {g['condition']} | {g['completed_training_episodes']} | {g['mission_successes']} | {g['falls']} | {g['fault_received']} | {g['fault_scenario_ended_before_fault']} |")
    lines += ['',
        'Episode training menggunakan action stokastik; angka sukses training tidak disamakan dengan validasi deterministik. Episode parsial di akhir budget dicatat terpisah dan tidak dimasukkan denominator episode lengkap. Total langkah log episode lengkap + parsial cocok dengan penghitung PPO.', '',
        ('Pilot lolos gate yang dibekukan. Usulan berikutnya: replikasi tiga seed independen dengan protokol baru sebelum klaim konsistensi; seed akhir tetap belum dibuka.' if outcome['pilot_promoted'] else 'Pilot belum memenuhi gate promosi. Jangan otomatis mereplikasi tiga seed atau menambah langkah. Jadikan hasil ini dasar studi revisi yang terbatas: periksa keterpaparan fault, tracking/tinggi tubuh dan desain reward/observasi sebelum memilih satu perubahan. Data ini belum membuktikan apakah penyebab utamanya budget, representasi observasi, reward, atau ruang residual.'), '',
        'Seluruh checkpoint dan kegagalan dipertahankan. Tidak ada klaim final test, hardware atau reliabilitas populasi. Analisis ini hanya membaca hasil; tidak mengubah konfigurasi atau menjalankan simulasi baru.']
    with (out/'FINDINGS.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:5]))


if __name__=='__main__':
    analyze()
