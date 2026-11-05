"""Post-pilot stop diagnostics from saved traces only; never runs or tunes a policy."""
import json
from pathlib import Path

import numpy as np

import sudden as s
import walking_calibrated as w


def stable_durations(mask, dt):
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    durations = (ends - starts - 1) * dt
    longest = float(durations.max()) if len(durations) else 0.
    trailing = float(durations[-1]) if len(durations) and mask[-1] else 0.
    return longest, trailing


def analyze():
    # Check inclusive sample endpoints: N stable samples span N-1 physics intervals.
    assert stable_durations(np.array([False, True, True, True]), .002) == (.004, .004)
    assert stable_durations(np.array([True, True, False]), .002) == (.002, 0.)
    assert stable_durations(np.zeros(3, dtype=bool), .002) == (0., 0.)
    w.verify_frozen()
    rows = s.read(w.OUT / 'healthy/episodes.json')
    results = []
    for row in rows:
        path = w.OUT / 'healthy' / f"healthy_{row['seed']}.npz"
        assert s.digest(path) == row['trace_sha256']
        with np.load(path, allow_pickle=False) as trace:
            q, v = trace['qpos'], trace['qvel']
            origin = np.array(row['initial_xy'])
            yaw0 = row['initial_yaw']
            goal = origin + 2 * np.array([np.cos(yaw0), np.sin(yaw0)])
            error = np.linalg.norm(q[:, :2] - goal, axis=1)
            speed = np.linalg.norm(v[:, :2], axis=1)
            qw, qx, qy, qz = q[:, 3:7].T
            yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
            heading = np.abs(np.arctan2(np.sin(yaw-yaw0), np.cos(yaw-yaw0)))
            stable = (error <= .15) & (speed <= .05) & (heading <= np.deg2rad(15))
            longest, trailing = stable_durations(stable, .002)
            assert abs(trailing-row['final_stable_hold_s']) < 1e-12
            assert (longest >= 2) == row['success'], 'An earlier valid hold needs separate interpretation'
            entered = np.flatnonzero(error <= .15)
            last_bad = np.flatnonzero(~stable)
            i = int(last_bad[-1]) if len(last_bad) else None
            results.append({
                'seed': row['seed'], 'success': row['success'],
                'failed_checks': [key for key, value in row['checks'].items() if not value],
                'trace_sha256': row['trace_sha256'],
                'first_goal_entry_s': float(entered[0]*.002) if len(entered) else None,
                'longest_stable_hold_s': longest, 'final_stable_hold_s': trailing,
                'stable_hold_shortfall_s': max(0., 2-trailing),
                'last_unstable_s': i*.002 if i is not None else None,
                'last_unstable_speed_m_s': float(speed[i]) if i is not None else None,
                'last_unstable_goal_error_m': float(error[i]) if i is not None else None,
                'final_goal_error_m': row['final_goal_error_m'],
                'max_cross_track_m': row['max_cross_track_m'],
                'final_heading_error_deg': row['final_heading_error_deg'],
            })
    assert len(results) == 20 and sum(r['success'] for r in results) == 16
    failures = [r for r in results if not r['success']]
    assert all(r['failed_checks'] == ['stable_stop'] for r in failures)
    payload = {'source_sha256': s.digest(Path(__file__)), 'episodes': results,
        'healthy_gate_passed': False, 'diagnosis_permitted': False,
        'interpretation': 'All four failures lack a continuous 2s stable hold anywhere before the 20s deadline. This is not solely a trailing-window convention or endpoint rounding. No new rollout, training, candidate, or parameter tuning was performed.'}
    s.write(w.OUT / 'stop_analysis.json', payload)
    lines = ['# Analisis hambatan berhenti', '',
        'Baseline terkalibrasi lulus 16/20 misi sehat, tanpa jatuh. Keempat kegagalan hanya pada durasi berhenti stabil; posisi akhir, heading, dan cross-track lolos.', '',
        '| Seed | Hold stabil akhir (s) | Kekurangan dari 2 s | Pertama masuk area tujuan (s) |',
        '|---|---:|---:|---:|']
    for r in failures:
        lines.append(f"| {r['seed']} | {r['final_stable_hold_s']:.3f} | {r['stable_hold_shortfall_s']:.3f} | {r['first_goal_entry_s']:.3f} |")
    lines += ['',
        'Durasi dihitung dari rentang waktu antarsampel, bukan jumlah sampel. Pemeriksaan juga memakai interval stabil terpanjang sepanjang episode: tidak ada episode gagal yang pernah memenuhi hold 2 detik sebelum deadline. Karena itu mengganti konvensi trailing hold tidak membuat gerbang lulus.', '',
        'Diagnosis gangguan belum dijalankan. Dampak gangguan dan kebutuhan kompensasi belum dapat disimpulkan.', '',
        'Langkah yang perlu disepakati untuk melanjutkan adalah studi terpisah pada pengarah misi: evaluasi transisi perlambatan dan stop, tanpa mengubah bobot policy maupun ambang misi. Ini usulan setelah melihat hasil, bukan konfigurasi yang sudah terbukti. Gunakan seed pengembangan dan cohort konfirmasi baru yang dibekukan; jangan mengklaim ulang seed pilot sebagai validasi baru. Kandidat ketiga atau training tetap memerlukan perluasan scope tersendiri.', '',
        'Analisis ini membaca trace saja. Tidak ada rollout, perubahan parameter, atau eksperimen baru. Kalibrasi pasif berbeda dari eksperimen berdiri; hasil simulasi tidak membuktikan kesiapan hardware.', '',
        'Reproduksi: `.venv\\Scripts\\python.exe analyze_walking_stop.py`. Angka dan hash trace tersedia di `stop_analysis.json`.']
    (w.OUT / 'STOP_ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'healthy_successes': 16, 'healthy_episodes': 20, 'failures': failures}, indent=2))


if __name__ == '__main__':
    analyze()
