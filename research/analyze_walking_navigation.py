"""Diagnostic scorecard from frozen navigation traces; no simulation or training."""
from collections import Counter
from pathlib import Path

import numpy as np

import sudden as s
import walking_navigation_hold as w


def analyze():
    proof = s.read(w.OUT/'completion_audit.json')
    assert proof['audit_passed'] and proof['goal_requirements_achieved']
    confirmation = s.read(w.OUT/'confirmation/episodes.json')
    assert len(confirmation) == 20 and all(r['success'] and not r['fallen'] for r in confirmation)
    healthy_ranges = {
        'final_goal_error_m': [min(r['final_goal_error_m'] for r in confirmation), max(r['final_goal_error_m'] for r in confirmation)],
        'final_stable_hold_s': [min(r['final_stable_hold_s'] for r in confirmation), max(r['final_stable_hold_s'] for r in confirmation)],
        'max_cross_track_m': max(r['max_cross_track_m'] for r in confirmation),
        'max_final_heading_error_deg': max(r['final_heading_error_deg'] for r in confirmation),
    }
    rows = s.read(w.OUT/'diagnosis/episodes.json')
    assert len(rows) == 25
    detail = []
    for row in rows:
        path = w.OUT/'diagnosis'/f"{row['condition']}_{row['seed']}.npz"
        assert s.digest(path) == row['trace_sha256']
        with np.load(path, allow_pickle=False) as a:
            onset = row['fault_onset_tick']
            q = a['qpos']
            goal = np.array(row['initial_xy']) + 2*np.array([np.cos(row['initial_yaw']), np.sin(row['initial_yaw'])])
            control_ticks = np.arange(0, len(a['command']), 10)
            entered = control_ticks[(control_ticks >= 1000) & (np.linalg.norm(q[control_ticks, :2]-goal, axis=1) <= .12)]
            if len(entered):
                assert np.all(a['command'][entered[0]:, :2] == 0), 'Stop latch restarted approach'
            power = a['actuator_force'] * .5*(a['qvel'][:-1, 6:]+a['qvel'][1:, 6:])
            detail.append({**row,
                'failed_checks': [key for key, value in row['checks'].items() if not value],
                'fault_onset_s': onset*.002 if onset is not None else None,
                'stop_latch_tick_independently_verified': int(entered[0]) if len(entered) else None,
                'contacts_at_fault_FL_FR_RL_RR': a['contacts'][onset].tolist() if onset is not None else None,
                'positive_joint_work_J': float(np.maximum(power, 0).sum()*.002),
                'torque_limited_fraction': float(np.mean(np.abs(a['requested_torque']-a['limited_torque']) > 1e-9)),
                'stable_goal_start_s': row['simulated_s']-row['final_stable_hold_s'] if row['success'] else None,
            })
    summary = []
    for condition in ['healthy', *[leg+'_calf' for leg in w.LEGS]]:
        group = [r for r in detail if r['condition'] == condition]
        assert len(group) == 5
        summary.append({'condition': condition, 'successes': sum(r['success'] for r in group),
            'episodes': 5, 'falls': sum(r['fallen'] for r in group),
            'failed_checks': dict(Counter(k for r in group for k in r['failed_checks'])),
            'median_final_goal_error_m': float(np.median([r['final_goal_error_m'] for r in group])),
            'max_tilt_deg': max(r['max_tilt_deg'] for r in group),
            'max_cross_track_m': max(r['max_cross_track_m'] for r in group),
            'fault_onset_range_s': [min(r['fault_onset_s'] for r in group), max(r['fault_onset_s'] for r in group)] if condition != 'healthy' else None})
    paired = []
    for seed in w.DIAGNOSIS:
        group = [r for r in detail if r['seed'] == seed]
        assert len({r['initial_state_sha256'] for r in group}) == 1
        healthy = next(r for r in group if r['condition'] == 'healthy')
        for row in group:
            if row['condition'] != 'healthy':
                assert row['checks']['fault_received'] and row['checks']['fault_before_goal']
                paired.append({'seed': seed, 'condition': row['condition'],
                    'healthy_success': healthy['success'], 'fault_success': row['success'],
                    'goal_error_change_m': row['final_goal_error_m']-healthy['final_goal_error_m']})
    failed = [r['condition'] for r in summary if r['condition'] != 'healthy' and r['successes'] < 5]
    faults_reaching_stop = sum(r['stop_latch_tick_independently_verified'] is not None for r in detail if r['condition'] != 'healthy')
    payload = {'source_sha256': s.digest(Path(__file__)),
        'completion_audit_sha256': s.digest(w.OUT/'completion_audit.json'),
        'summary': summary, 'episodes': detail, 'paired_comparisons': paired,
        'healthy_confirmation_ranges': healthy_ranges,
        'conditions_with_observed_mission_failure': failed,
        'fault_episodes_reaching_stop_latch': faults_reaching_stop,
        'scope': 'Five paired simulation seeds per condition, frozen policy, no adaptation; failure shows this baseline needs improvement to complete these cases, not proof that learned compensation is the only remedy.'}
    s.write(w.OUT/'diagnostic_analysis.json', payload)
    lines = ['# Diagnosis berjalan dengan gangguan motor', '',
        'Konfirmasi sehat lolos 20/20. Diagnosis memakai lima seed baru, berpasangan pada kondisi sehat dan empat gangguan calf. Tidak ada training adaptasi.', '',
        '| Kondisi | Berhasil | Jatuh | Median error akhir (m) | Tilt maksimum (°) |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['condition']} | {r['successes']}/5 | {r['falls']} | {r['median_final_goal_error_m']:.3f} | {r['max_tilt_deg']:.2f} |")
    lines += ['', '## Jenis kegagalan', '']
    for r in summary:
        lines.append(f"- {r['condition']}: {r['failed_checks'] or 'seluruh pemeriksaan lolos'}.")
    lines += ['', '## Implikasi', '',
        (f"Baseline tanpa adaptasi belum menyelesaikan seluruh kasus pada: {', '.join(failed)}. Perbaikan kemampuan berjalan saat fault diperlukan untuk kasus-kasus ini bila targetnya seluruh episode lulus. Hasil ini belum memilih antara kompensasi kontrol, pengarah misi yang tanggap fault, atau adaptasi terlatih; training tidak dilakukan." if failed else 'Baseline tanpa adaptasi menyelesaikan semua kasus diagnosis ini. Belum ada kebutuhan kompensasi yang ditunjukkan sampel ini; uji held-out lebih luas sebelum memutuskan training.'), '',
        f'Dari 20 episode fault, {faults_reaching_stop} mencapai radius latch stop 0,12 m. Dengan demikian, bila nilainya nol, seluruh kegagalan fault terjadi sebelum mode stop ini aktif; perbaikan stop saja belum cukup untuk gangguan tersebut.', '',
        'Fault selalu aktif sebelum goal pada progress >=0,50 m dan hanya motor terpilih menerima 10% torsi setelah pembatasan. Trace mencatat torsi aktual dan kontak kaki; audit memeriksa semua episode. Waktu onset, kontak saat interval fault, fraksi torque limiting, kerja mekanik positif, dan selisih error berpasangan ada di diagnostic_analysis.json. Kontak saat onset adalah proxy kondisi gait, bukan validasi seluruh fase gait.', '',
        'Kerja mekanik positif dihitung dari torque aktual × rata-rata joint velocity pada kedua ujung interval, diintegralkan pada 0,002 s; bukan konsumsi energi baterai. Fraksi torque limiting memakai seluruh actuator-interval episode, sehingga episode yang berhenti dini memiliki durasi berbeda.', '',
        'Hanya lima reset dengan noise kecil per kondisi pada lantai datar. Persentase sampel bukan jaminan reliabilitas populasi atau hardware. Kalibrasi pasif dan perbedaan timestep terhadap deployment asal tetap berlaku. Seluruh hasil pilot gagal sebelumnya dipertahankan.', '',
        'Reproduksi: `.venv\\Scripts\\python.exe analyze_walking_navigation.py` (menolak menimpa hasil analisis).']
    with (w.OUT/'DIAGNOSIS.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]))


if __name__ == '__main__':
    analyze()
