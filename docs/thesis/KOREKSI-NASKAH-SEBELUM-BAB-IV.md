# Koreksi Naskah Sebelum Melanjutkan Bab IV

File sumber laporan yang tersedia hanya PDF, sehingga koreksi berikut harus diterapkan pada dokumen sumber (Word/LaTeX), lalu PDF diekspor ulang.

1. Ubah frasa “tiga permasalahan” dan “tiga tujuan” menjadi dua, atau tambahkan butir ketiga yang benar-benar diteliti.
2. Sinkronkan daftar gambar/tabel dengan nomor dan judul aktual; hapus placeholder `Tabel 3.x`.
3. Hapus klaim FPR atau akurasi dari evaluasi utama tanpa ground truth serangan.
4. Nyatakan timestamp dinormalisasi menjadi timezone-aware UTC; WITA hanya lapisan tampilan bila digunakan.
5. Gunakan istilah fitur `mitre_tactic_count`, bukan `mitre_hit_count`.
6. Jelaskan `contamination="auto"` sebagai konfigurasi offset internal Isolation Forest; parameter itu tidak mengestimasi proporsi serangan dan tidak menentukan eskalasi pipeline.
7. Nyatakan Silhouette sebagai evaluasi pemisahan struktural internal, bukan akurasi deteksi.
8. Ubah evaluasi IF menjadi split kronologis reference/calibration/test dan tegaskan test tidak di-fit ulang.
9. Tambahkan contextual-fixed ablation dan context purity agar baseline time-only tidak dinilai hanya dari ARR.
10. Nyatakan Telegram sebagai deferred file sink dan Wazuh/Shuffle/Telegram eksternal belum tervalidasi live.

