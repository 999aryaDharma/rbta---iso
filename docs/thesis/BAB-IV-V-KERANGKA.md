# Kerangka Lanjutan Skripsi — Bab IV dan Bab V

Kerangka ini sengaja tidak mengarang angka. Semua nilai numerik diisi dari `research_report.json`, `evaluation.json`, dataset manifest, dan hasil test pada exact commit yang digunakan.

# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Lingkungan dan Reproducibility Penelitian

Jelaskan perangkat ASUS, CPU/RAM/OS, versi Python/Node, versi Wazuh bila relevan, exact Git SHA, model version/training run ID, random seed, timezone UTC/WITA pada tampilan, serta perintah eksekusi. Sertakan tabel berikut.

| Komponen | Nilai | Sumber bukti |
|---|---|---|
| Commit aplikasi | Diisi dari `git rev-parse HEAD` | log verifikasi |
| Model version | Diisi dari model metadata | `evaluation.json` |
| Dataset dan SHA-256 | Diisi dari catalog | dataset manifest |
| Random seed | Diisi dari artifact | `evaluation.json` |
| Jumlah alert valid/tidak valid | Diisi dari catalog | dataset manifest |

## 4.2 Karakteristik dan Persiapan Data

Uraikan sumber data, rentang timestamp, jumlah alert, agen, rule group, distribusi severity, aturan canonicalization, penanganan timestamp timezone-aware, deduplikasi, redaksi data sensitif, serta pemisahan kronologis 60% reference, 20% calibration, dan 20% test. Tegaskan bahwa split dilakukan sebelum agregasi agar informasi masa depan tidak bocor.

Figur yang disarankan: alur raw source → canonical alert → temporal split → agregasi per split.

## 4.3 Hasil Rule-Based Temporal Aggregation

### 4.3.1 Sensitivitas jendela Δt

Tampilkan delapan nilai 1, 5, 10, 15, 20, 30, 45, dan 60 menit. Laporkan `n_raw`, `n_meta`, ARR, waktu eksekusi, dan alasan pemilihan elbow. Jangan hanya menampilkan nilai terpilih.

### 4.3.2 Perbandingan tiga skenario agregasi

| Skenario | Kunci konteks | Window | N meta | ARR | Context purity | Kontaminasi |
|---|---|---|---:|---:|---:|---:|
| Time-only fixed | tidak ada | tetap | dari artifact | dari artifact | dari artifact | dari artifact |
| Contextual fixed | agent + rule group | tetap | dari artifact | dari artifact | dari artifact | dari artifact |
| Contextual adaptive/RBTA | agent + rule group | adaptif per agen | dari artifact | dari artifact | dari artifact | dari artifact |

Bahas trade-off: ARR tinggi tidak otomatis baik bila konteks berbeda tercampur. Context purity menjadi pasangan interpretasi wajib bagi ARR.

### 4.3.3 Robustness terhadap noise

Untuk noise 0%, 5%, 10%, 20%, dan 30%, bandingkan ketiga skenario dengan stream injeksi yang sama pada setiap seed. Laporkan ARR degradation, absorption, context purity, dan runtime. Noise sintetis hanya menguji ketahanan agregasi, bukan mensimulasikan kebenaran serangan.

### 4.3.4 Kompleksitas runtime

Tampilkan delapan ukuran subset. Untuk setiap subset gunakan median lima pengulangan serta IQR; pisahkan waktu persiapan/sorting dari runtime RBTA. Laporkan slope dan R² secara deskriptif, bukan sebagai jaminan skalabilitas produksi.

## 4.4 Hasil Isolation Forest

### 4.4.1 Pelatihan, kalibrasi, dan pengujian temporal

Jelaskan bahwa RobustScaler dan Isolation Forest di-fit pada reference set; kalibrasi skor dan threshold Tukey dibangun pada calibration set; test set hanya di-score tanpa refit. Cantumkan tujuh fitur, 200 trees, random seed, dan provenance model.

### 4.4.2 Distribusi skor dan keputusan

Laporkan min, maksimum, rerata, threshold, jumlah di atas threshold, serta distribusi `CRITICAL`, `SUSPICIOUS`, `NOISE_HIGH`, `NOISE` dan action. Jangan menyebut jumlah anomaly sebagai jumlah serangan. Jelaskan bahwa skor kalibrasi boleh berada di luar 0–1.

### 4.4.3 Evaluasi struktural Silhouette

Laporkan observed Silhouette, mean/std/min/max partisi acak, percentile, z-score, empirical p-value, jumlah permutasi valid, dan seed. Interpretasi yang sah terbatas pada apakah partisi ESCALATE/non-ESCALATE lebih terpisah secara struktural dibanding partisi acak berproporsi sama.

## 4.5 Implementasi Sistem dan Demonstrasi End-to-End

Jelaskan replay isolation, catalog dataset + SHA-256, API authentication, rate limiting, evidence SQLite, traceability raw-to-meta, dashboard Demo, live evaluation, post-replay evaluation, artifact export, Docker non-root/read-only, dan CI. Pisahkan fitur yang benar-benar aktif dari integrasi deferred.

## 4.6 Pengujian Perangkat Lunak

Tampilkan ringkasan test backend, coverage, frontend unit test/lint/typecheck/build, E2E, dependency audit, Docker/health/non-root, serta exact tanggal dan commit. Nilai harus diambil dari run final, bukan run lama.

## 4.7 Pembahasan terhadap Rumusan Masalah

Susun dua subbagian yang masing-masing menjawab satu rumusan masalah. Hubungkan bukti RBTA pada ARR + context purity dan bukti IF pada prioritas + evaluasi struktural. Bedakan “tercapai secara internal pada dataset penelitian” dari “tergeneralisasi pada serangan produksi”.

## 4.8 Keterbatasan Penelitian

Nyatakan tanpa defensif: tidak ada ground truth serangan yang memadai; belum mengukur FPR/FNR/accuracy; validasi eksternal dan drift belum dilakukan; Wazuh live/Shuffle/Telegram eksternal tidak menjadi bukti demo; shared API key ditujukan untuk demo loopback, bukan SOC multi-user; kinerja 1,4 juta alert bukan target demo interaktif.

# BAB V PENUTUP

## 5.1 Kesimpulan

Gunakan dua butir, sejajar satu-per-satu dengan dua rumusan masalah dan dua tujuan penelitian:

1. Kesimpulan RBTA: sebutkan perubahan unit triase, pemeliharaan konteks, traceability, dan hasil perbandingan tiga baseline berdasarkan angka final.
2. Kesimpulan Isolation Forest: sebutkan mekanisme prioritas, penggunaan model beku, hasil struktural test set, serta batas bahwa hasil bukan akurasi deteksi serangan.

Tutup dengan satu kalimat kontribusi gabungan: sistem membantu analis memusatkan investigasi pada meta-alert yang diprioritaskan tanpa menghilangkan akses ke evidence mentah.

## 5.2 Saran

1. Bangun dataset eksternal berlabel melalui adjudikasi analis untuk mengukur precision, recall, FPR, FNR, dan generalisasi.
2. Lakukan evaluasi longitudinal dan monitoring drift pada periode berbeda.
3. Uji skala penuh dengan bounded state, pagination database, dan target SLO yang ditetapkan sebelumnya.
4. Implementasikan identitas multi-user/RBAC dan audit log sebelum penggunaan SOC produksi.
5. Validasi Wazuh live, Shuffle, dan Telegram melalui uji integrasi terpisah.
6. Bandingkan dengan metode agregasi/model lain tanpa mengubah test set setelah hasil terlihat.

## 5.3 Matriks bukti penutup

Sebelum finalisasi naskah, setiap klaim kuantitatif harus memiliki baris yang memetakan klaim → artifact key → tabel/gambar → exact commit. Jika tidak memiliki bukti tersebut, ubah menjadi batasan atau rencana penelitian lanjutan.

