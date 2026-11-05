"""Summarize the audited comparison with the repository's LM Roman figure style."""
from project_paths import project_file
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import diagnose_walking_fault as d
import compare_walking_low_noise as c


def main():
    out=c.OUT
    assert d.read(out/'completion_audit.json')['audit_passed']
    design=d.read(out/'design.json')
    for name,sha in design['source_sha256'].items():assert d.digest(project_file(c.ROOT, name))==sha
    rows=d.read(out/'episodes.json');summary=d.read(out/'summary.json')
    latest=[r for r in summary if r['stage']=='validation_131072']
    assert len(latest)==4
    assert not (out/'comparison.png').exists() and not (out/'REPORT.md').exists()
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6))
    for ax,key,title,label in zip(axes,['post_net_progress_m','post_residual_rms_rad'],
            ['Net progress after fault','Residual after fault'],['Forward gain (m)','Residual RMS (rad)']):
        for i,r in enumerate(latest):
            for j,(pilot,color,marker) in enumerate(zip(c.PILOTS,['#355c7d','#9a6544'],['o','s'])):
                group=[e for e in rows if (e['pilot'],e['stage'],e['condition'])==(pilot,'validation_131072',r['condition'])]
                assert len(group)==5
                x=i+(-.13 if j==0 else .13)
                for e in group:
                    ax.scatter(x,e[key],s=26,marker=marker,edgecolors=color,
                               facecolors='white' if e['fallen'] else color,alpha=.8)
                median=r['medians'][key][j]
                assert median==float(np.median([e[key] for e in group]))
                ax.plot([x-.07,x+.07],[median,median],color=color,linewidth=2.6,
                        label=('Previous pilot' if j==0 else 'Low-noise pilot') if i==0 else None)
        ax.axhline(0,color='#888888',linewidth=.6)
        ax.set_xticks(range(4),['FL','FR','RL','RR']);ax.set_ylabel(label);ax.set_title(title)
        ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=11)
    axes[0].set_ylim(-.1,.72);axes[0].set_yticks([-.1,0,.2,.4,.6])
    axes[1].set_ylim(0,.09);axes[1].set_yticks([0,.02,.04,.06,.08])
    fig.suptitle('Fault comparison at 131,072 training steps',fontsize=20)
    fig.text(.5,.025,'Five matched development poses per fault; markers: episodes, bars: medians, hollow markers: falls.\n'
             'Post-onset duration ends at fall or deadline. One training seed; no new rollouts.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.12,1,.93])
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(font),size=text.get_fontsize()))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,text.get_text()
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    table=['| Fault | Net progress lama / low (m) | RMS residual lama / low (rad) | Torque limiting motor lemah lama / low | Jatuh lama / low |',
           '|---|---:|---:|---:|---:|']
    for r in latest:
        m=r['medians'];pair=lambda k,fmt:' / '.join(format(v,fmt) for v in m[k])
        table.append(f"| {r['condition']} | {pair('post_net_progress_m','.3f')} | {pair('post_residual_rms_rad','.4f')} | {pair('post_weak_torque_limit_fraction','.1%')} | {r['falls'][0]} / {r['falls'][1]} |")
    text='\n'.join(['# Perbandingan pilot low-noise', '',
        'Audit lengkap runtime asli D lulus: Torch 2.15.0+cpu, SB3 2.7.1; 100 episode, 983.611 state fisika. Prediksi nominal dan residual direplay, reward/observasi/torque/kriteria misi diperiksa, serta 2.328 arsip lama cocok hash. Tidak ada checkpoint yang lolos promosi.', '',
        'Sumber: trace tersimpan pilot tinggi 0,338 m dan pilot low-noise. Masing-masing baseline dan tiga checkpoint memiliki 25 episode pada pose 23000–23004. Ada 200 episode, 80 pasangan fault; 20 pasangan baseline duplikat adalah kontrol, sehingga hanya 60 pasangan fault berasal dari checkpoint terlatih. Baseline identik, initial state seluruh pasangan identik, dan 160 prefix fault cocok dengan episode sehat pasangannya sebelum onset.', '',
        'Pada checkpoint 131.072, kedua pilot sehat 5/5, fault 0/20, jatuh 1/20. Low-noise menghilangkan satu jatuh FL tetapi menambah satu jatuh FR. Seluruh 20 episode fault low-noise mengalami setidaknya satu window stall menurut definisi diagnostik. Stall bukan ambang kelulusan baru.', '',*table,'',
        'Nilai tabel adalah median lima episode per kondisi, bukan median perubahan berpasangan. Median perubahan net progress low-minus-lama: FL -0,092 m; FR +0,012 m; RL -0,165 m; RR +0,031 m. Ringkasan seluruh checkpoint tersedia di summary.json; tidak ada checkpoint dengan keberhasilan fault. Pada tiga checkpoint, jatuh total 5/60 pada pilot lama dan 4/60 pada low-noise. Angka 60 memakai pose berulang sehingga bukan 60 pengujian independen.', '',
        'RMS residual low-noise turun menjadi sekitar 0,006–0,012 rad pada checkpoint terakhir, tanpa aksi menyentuh batas [-1,1]. Pada RL, median torque limiting motor lemah mencapai 99,3%; median delta applied torque dari residual pada motor lemah hanya 0,000977 Nm. Ini perbandingan pada state tersimpan yang sama dengan residual nol, bukan bukti efek closed-loop. Batas torque tetap berlaku sebelum pengali fault 10%.', '',
        'Seluruh 60 episode fault low-noise menunjukkan perubahan residual terhadap pasangan sehat dalam 0,002–0,020 s. Jadi policy merespons perubahan state; data tidak membuktikan policy mengenali identitas motor atau memilih kompensasi yang benar.', '',
        'Satu next step yang disarankan: bekukan lalu jalankan ablation gain residual closed-loop pada checkpoint low-noise 131.072, tanpa training. Bandingkan gain 0, 1, 2, dan 4 pada pose pengembangan yang sama. Terapkan perubahan hanya pada keputusan kontrol pertama setelah fault benar-benar dilatch; prefix sebelumnya identik. Tetap clip aksi ke [-1,1], pertahankan scale ±0,10 rad, joint/torque limits, fault, controller, observasi, dan seluruh kriteria misi. Previous residual harus mencatat aksi yang benar-benar diterapkan. Gunakan gain 1 sebagai replay kontrol historis. Pilihan gain ini rancangan diagnostik, belum terbukti optimal.', '',
        'Alasan: respons residual ada tetapi amplitudonya kecil dan sebagian besar ruang aksi belum dipakai. Uji berpasangan ini bisa memeriksa apakah memperkuat arah koreksi yang sudah dipelajari membantu progress atau justru menambah jatuh. Torque saturation motor lemah dapat membuat gain tidak efektif; hasil negatif tidak membuktikan semua residual mustahil. Belum ada dasar untuk memperbesar batas ±0,10 rad atau menambah label motor.', '',
        'Protokol ablation belum dijalankan atau dibekukan: usulan 80 episode fault (4 gain × 4 fault × 5 pose), tanpa seed baru/final, tanpa training, tanpa memilih pemenang berdasarkan return. Tetapkan budget waktu dan urutan run sebelum eksekusi. Laporkan seluruh hasil, mission success dan jatuh sebagai hasil utama, progress sebagai diagnostik. Penggunaan fault latch bersifat privileged untuk diagnosis, bukan controller deployment.', '',
        'Batas: hanya satu training seed dan cohort pengembangan historis. Perbedaan antar-pilot bukan estimasi kausal yang direplikasi. Durasi pasca-onset berbeda bila jatuh; summary.json juga memuat progress pada 0,5 dan 2 s dengan jumlah episode tersedia. Satu FL pilot lama jatuh sebelum 2 s pada checkpoint terakhir; median 2 s FL tidak memiliki denominator sama. Tidak ada generalisasi terrain/strength/onset atau hardware yang diuji. Tidak ada simulasi baru, training baru, atau seed final dibuka. Graph tidak tersedia; tidak ada klaim query graph.', '',
        'Reproduksi: `.venv\\Scripts\\python.exe -B compare_walking_low_noise.py` menolak menimpa hasil. Audit pembanding: completion_audit.json; data episode: episodes.json; definisi dan input: protocol.json/design.json.'])+'\n'
    with (out/'REPORT.md').open('x',encoding='utf-8') as f:f.write(text)
    d.write(out/'delivery_audit.json',{'figure_sha256':d.digest(out/'comparison.png'),
        'report_sha256':d.digest(out/'REPORT.md'),'source_sha256':d.digest(__file__),
        'font_file':str(font),'dpi':160,'visible_text_inside_figure':True,
        'distinct_episodes_displayed':40,'plotted_points':80,'medians_independently_recomputed':16})


if __name__=='__main__':main()
