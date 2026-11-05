# Go2 Rebalance

## Tujuan

Unitree Go2 berjalan menuju target 2 m dalam 20 detik, lalu berhenti stabil minimal 2 detik. Simulasi berjalan di MuJoCo pada arena datar, dengan CPU saja. Scope terbatas pada misi jalan dan berhenti ini.

## Skenario uji

- **Sehat:** semua motor berfungsi pada 100%.
- **Satu motor melemah:** uji masing-masing dari 12 motor secara terpisah. Setiap kaki memiliki motor panggul (`hip`), paha (`thigh`), dan lutut (`calf`). Pada progres 0,50 m, kekuatan motor yang diuji turun ke 10% dari torsi setelah pembatasan. Sebelas motor lain tetap sehat.
- **Kekuatan menengah:** uji tingkat antara 10% dan 100% bila perlu untuk memahami performa pada misi yang sama.

## Ekspektasi hasil

- Setiap misi yang berhasil memenuhi galat posisi akhir 2D ≤0,15 m, galat heading ≤15°, dan deviasi lateral maksimum ≤0,20 m. Go2 berhenti stabil minimal 2 detik pada kecepatan ≤0,05 m/s dalam durasi misi 20 detik.
- Validasi pengembangan memakai pose 23000–23004. Kontrol sehat berhasil **5/5**, setiap motor pada 10% berhasil **minimal 3/5**, dan **nol jatuh** di seluruh kondisi.
- Laporkan sukses dan jatuh per motor. Evaluasi akhir memakai seed baru setelah kandidat lolos validasi dan dibekukan.
