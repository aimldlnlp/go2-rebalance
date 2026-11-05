"""Data interpretation and one offline reward proposal; no simulation/training."""
from pathlib import Path

import numpy as np

import sudden as s
import walking_residual_ablation as a


def explain():
    out=a.OUT
    proof=s.read(out/'completion_audit.json')
    phases=s.read(out/'phase_analysis.json')
    assert proof['audit_passed']
    healthy={g['stage']:g for g in phases['groups'] if g['condition']=='healthy'}
    def components(group):
        return {key:sum(p['mean_reward_contributions'][key] for p in group['phases']) for key in a.TERMS}
    before,after=healthy['validation_65536'],healthy['validation_131072']
    old_terms,new_terms=components(before),components(after)
    drop=before['mean_episode_return']-after['mean_episode_return']
    contributions=[{'component':key,'mean_return_delta':new_terms[key]-old_terms[key],
        'signed_share_of_net_drop_pct':100*(old_terms[key]-new_terms[key])/drop} for key in a.TERMS]
    assert abs(sum(r['mean_return_delta'] for r in contributions)+drop)<1e-8
    assert abs(sum(r['signed_share_of_net_drop_pct'] for r in contributions)-100)<1e-8
    baseline=s.read(a.pilot.OUT/'baseline/episodes.json')
    height_medians=[]
    for row in [r for r in baseline if r['condition']=='healthy']:
        with np.load(a.pilot.OUT/'baseline'/f"healthy_{row['seed']}.npz",allow_pickle=False) as trace:
            q,commands=trace['qpos'],trace['command']
            ticks=np.arange(0,len(commands),10);end=np.minimum(ticks+10,len(q)-1)
            walking=(ticks>=1000)&np.any(commands[ticks,:2]!=0,axis=1)
            height_medians.append(float(np.median(q[end[walking],2])))
    measured=float(np.median(height_medians))
    proposed=float(round(measured,3))
    reweighted=[];fall_count=height_falls=tilt_falls=0
    for stage in ['baseline','validation_32768','validation_65536','validation_131072']:
        values=[]
        for row in s.read(a.pilot.OUT/stage/'episodes.json'):
            path=a.pilot.OUT/stage/f"{row['condition']}_{row['seed']}.npz"
            assert s.digest(path)==row['trace_sha256']
            with np.load(path,allow_pickle=False) as trace:
                q=trace['qpos'];ticks=np.arange(0,len(q)-1,10);end=np.minimum(ticks+10,len(q)-1)
                old=-10*(q[end,2]-.27)**2
                new=-10*(q[end,2]-proposed)**2
                values.append({'condition':row['condition'],'seed':row['seed'],
                    'old_height_component_sum':float(old.sum()),'proposed_height_component_sum':float(new.sum()),
                    'return_change_on_fixed_trace':float((new-old).sum())})
                if row['fallen']:
                    fall_count+=1;height_falls+=int(q[:,2].min()<.16);tilt_falls+=int(row['max_tilt_deg']>45)
        reweighted.append({'stage':stage,'episodes':values})
    result={'source_sha256':s.digest(Path(__file__)),'phase_analysis_sha256':s.digest(out/'phase_analysis.json'),
        'completion_audit_sha256':s.digest(out/'completion_audit.json'),
        'healthy_return_drop_65536_to_131072':drop,'return_drop_components':contributions,
        'height_reference_current_m':.27,'baseline_walking_height_episode_medians_m':height_medians,
        'baseline_walking_height_median_of_medians_m':measured,'proposed_height_reference_m':proposed,
        'falls_checked':fall_count,'height_boundary_falls':height_falls,'tilt_boundary_falls':tilt_falls,
        'offline_height_reward_reweighting':reweighted,
        'proposal':{'status':'Exploratory proposal only, not trained or demonstrated effective',
            'one_change':f'Change height reward reference from 0.27 to {proposed:.3f}m, calibrated only from prior nominal healthy walking traces.',
            'unchanged':'All other reward terms/weights, residual scale, observation, policy architecture, mission/fault criteria, limits and simulator.',
            'gate':'Do not adopt zero-command gate on the basis of a 5/5 vs 5/5 ceiling result.',
            'evaluation':'If a new pilot is authorized, prefreeze new development/confirmation splits. Keep final 23100-23119 untouched until a separate final protocol.',
            'caveat':'Reweighted returns on fixed traces are reward accounting only; they do not simulate a newly learned policy or prove the height target caused failure.'},
        'training':False,'new_episodes_in_this_analysis':0}
    s.write(out/'diagnosis.json',result)
    lines=['# Catatan diagnosis Data','',
        f"**Ablation tidak menunjukkan peningkatan sukses:** biasa {proof['successes']['always']}/5, gate {proof['successes']['zero_command_off']}/5 pada seed 23200–23204. Cohort ini berbeda dari lima pose lama dengan hasil checkpoint terakhir 1/5; jangan mencampurkan denominator atau menyimpulkan perubahan controller dari dua cohort berbeda.",'',
        'Gate belum layak diadopsi berdasarkan metrik utama yang dibekukan. Hasil 5/5 vs 5/5 memiliki ceiling pada sampel ini; bukan bukti bahwa gate tidak pernah berpengaruh. Intervensi juga menggabungkan settle/ramp-zero/stop, sehingga tidak mengisolasi satu fase.','',
        '## Apa yang menjelaskan penurunan return sehat','',
        f"Pada lima pose lama yang sama, checkpoint 65536 -> 131072 menurunkan mean return sehat {before['mean_episode_return']:.2f} -> {after['mean_episode_return']:.2f} (turun {drop:.2f}); sukses 5/5 -> 1/5. Return sehat ikut turun, bukan meningkat berlawanan dengan misi.",'',
        '| Komponen | Perubahan mean return | Kontribusi terhadap penurunan netto |','|---|---:|---:|']
    for row in sorted(contributions,key=lambda r:r['mean_return_delta']):
        lines.append(f"| {row['component']} | {row['mean_return_delta']:+.2f} | {row['signed_share_of_net_drop_pct']:.1f}% |")
    lines+=['','Ini dekomposisi aritmetika, bukan identifikasi penyebab belajar. Semua komponen menjumlah tepat ke penurunan total. Perubahan seed reset dan mix tidak dipakai sebagai penjelasan untuk tabel ini karena pose validasinya berpasangan.','',
        '## Satu kandidat revisi untuk studi berikutnya','',
        f"**Kalibrasi referensi tinggi reward 0,270 -> {proposed:.3f} m**, tanpa mengubah bobot reward lain atau batas misi. Lima median tinggi baseline sehat saat berjalan adalah {[round(x,6) for x in height_medians]}; median antarepisode {measured:.6f} m. Target lama berbeda sekitar {(measured-.27)*100:.1f} cm dari gait nominal yang justru sudah lolos misi.",'',
        f"Pada 100 episode historis terdapat {fall_count} status jatuh; {height_falls} melewati tinggi <0,16 m dan {tilt_falls} melewati tilt >45 derajat. Kandidat ini didukung ketidakselarasan target tinggi terhadap baseline serta pola kegagalan, tetapi belum terbukti memperbaiki kontrol. Menargetkan postur lebih rendah bisa saja merupakan desain yang sah; data ini tidak membuktikan target 0,270 m sebagai akar penyebab.",'',
        'Nilai target usulan diturunkan hanya dari baseline sehat lama, bukan seed ablation baru. diagnosis.json menyimpan reweighting komponen tinggi pada semua trace lama; gerakan, sukses/gagal dan komponen lain tidak berubah dalam perhitungan offline itu. Return hasil reweighting bukan hasil policy baru.','',
        'Belum ada perubahan environment, training ulang, atau tambahan episode setelah 10 ablation. Jika kandidat diuji berikutnya, bekukan satu perubahan scalar, budget pilot dan split baru sebelum training. Jangan memperbesar residual, mengganti observasi/gamma, atau mengaktifkan gate sekaligus; itu akan mengaburkan hasil eksperimen. Seed final tetap disimpan.']
    with (out/'DATA_DIAGNOSIS.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    print(f'Data diagnosis saved: height candidate .270 -> {proposed:.3f}m; healthy reward drop {drop:.2f}; no training')


if __name__=='__main__':explain()
