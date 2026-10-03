# Panduan Menjalankan Backend FastAPI & Frontend Dashboard Secara Native (Windows)

Dokumen ini adalah panduan operasional resmi untuk menjalankan sistem riset **RBTA + Isolation Forest** di lingkungan lokal Windows (laptop pengembangan Lenovo) tanpa menggunakan Docker.

---

## 1. Topologi & Arsitektur Lokal

Sistem berjalan sebagai dua proses native terpisah:

```
[Wazuh Export Dataset] 
          │
          ▼
[FastAPI Backend Server] (http://127.0.0.1:8010)
          ▲
          │ (Vite Proxy: /api/*, /health, /ready)
[React / Vite Dashboard] (http://127.0.0.1:5173/dashboard/)
```

* **Backend Entrypoint**: `src/api/server.py`
* **Backend Port**: `8010` (diselaraskan dengan konfigurasi reverse proxy di `dashboard/vite.config.ts`)
* **Frontend Base Path**: `/dashboard/`
* **Model Registry**: `models/rbta-if-v1`
* **Dataset Wazuh**: `D:\KAMPUS\SKRIPSI\wazuh-data-2026\indexer-export` (atau `data/test_datasets` untuk pengujian sintetis)

---

## 2. Penjelasan Variabel Lingkungan (Environment Variables)

### Apakah setiap menjalankan backend harus set variabel environment terlebih dahulu?

**Jawaban:**
Secara bawaan di PowerShell, **YA** jika Anda membuka jendela terminal PowerShell baru dan menjalankannya secara manual dengan perintah `python -m src.api.server`. Hal ini karena perintah `$env:VARIABEL = "..."` hanya tersimpan di memori proses jendela PowerShell yang aktif (*process-scoped*). Ketika terminal ditutup, variabel tersebut hilang.

Karena server FastAPI diatur dalam mode ketat (*strict mode*), server akan gagal mulai (*fail-closed*) jika variabel wajib seperti `RBTA_API_KEY` dan `RBTA_MODEL_VERSION` tidak ditemukan.

### Cara agar tidak perlu set variabel secara manual setiap kali:

Terdapat 2 opsi praktis:

1. **Gunakan Script Otomatis (Sangat Direkomendasikan)**:
   Gunakan script `.\scripts\run-backend.ps1`. Script ini akan otomatis mendeteksi virtual environment, memuat konfigurasi dari file `.env` di root repository (atau memakai nilai default lokal yang valid), lalu langsung menjalankan server.
2. **Set Variabel Permanen di User Environment Windows**:
   Jika ingin variabel selalu tersedia di setiap terminal PowerShell tanpa script tambahan:
   ```powershell
   [System.Environment]::SetEnvironmentVariable('RBTA_API_KEY', '<isi-RBTA_API_KEY-anda>', 'User')
   [System.Environment]::SetEnvironmentVariable('RBTA_MODEL_VERSION', 'rbta-if-v1', 'User')
   [System.Environment]::SetEnvironmentVariable('RBTA_MODEL_REGISTRY_DIR', 'models', 'User')
   [System.Environment]::SetEnvironmentVariable('RBTA_PORT', '8010', 'User')
   [System.Environment]::SetEnvironmentVariable('RBTA_REPLAY_DATA_DIR', 'D:\KAMPUS\SKRIPSI\wazuh-data-2026\indexer-export', 'User')
   ```

---

## 3. Cara Menjalankan Sistem

### Metode A: Menggunakan Script Otomatis (Cepat & Praktis)

Buka dua jendela PowerShell terpisah di root direktori proyek (`D:\KAMPUS\SEMINAR\v2_json\rbta + iso`):

#### Terminal 1 — Menjalankan Backend:
```powershell
.\scripts\run-backend.ps1
```
*Script ini akan otomatis mengaktifkan `.venv-gate`, membaca `.env`, mengisi default port `8010`, dan menyalakan FastAPI.*

#### Terminal 2 — Menjalankan Frontend:
```powershell
.\scripts\run-frontend.ps1
```
*Script ini akan otomatis berpindah ke folder `dashboard`, memeriksa `node_modules`, dan menyalakan server Vite.*

---

### Metode B: Menjalankan Secara Manual Langkah demi Langkah

Jika Anda ingin menjalankan atau memodifikasi parameter secara langsung:

#### Terminal 1 — Backend FastAPI:
```powershell
# 1. Masuk ke root direktori
Set-Location "D:\KAMPUS\SEMINAR\v2_json\rbta + iso"

# 2. Aktifkan virtual environment
.\.venv-gate\Scripts\Activate.ps1

# 3. Definisikan variabel environment wajib
$env:RBTA_API_KEY = "<isi-RBTA_API_KEY-anda>"
$env:RBTA_MODEL_VERSION = "rbta-if-v1"
$env:RBTA_MODEL_REGISTRY_DIR = "models"
$env:RBTA_PORT = "8010"
$env:RBTA_REPLAY_DATA_DIR = "D:\KAMPUS\SKRIPSI\wazuh-data-2026\indexer-export"
$env:RBTA_LOG_LEVEL = "INFO"
$env:RBTA_SOURCE_MODE = "DEFERRED"

# 4. Jalankan server FastAPI
python -m src.api.server
```

#### Terminal 2 — Frontend React/Vite:
```powershell
# 1. Pindah ke direktori dashboard
Set-Location "D:\KAMPUS\SEMINAR\v2_json\rbta + iso\dashboard"

# 2. Pastikan dependensi sudah terpasang (hanya perlu sekali)
npm install

# 3. Jalankan development server
npm run dev -- --host 127.0.0.1
```

---

## 4. Akses & Verifikasi

1. **Verifikasi Backend**:
   * **Health Check**: Buka `http://localhost:8010/health` (respons: `{"status": "ok"}`).
   * **Readiness Check**: Buka `http://localhost:8010/ready` (respons `200 OK` menandakan model bundle `rbta-if-v1` dan scoring pipeline berhasil dimuat).

2. **Akses Dashboard**:
   * Buka browser ke: **`http://127.0.0.1:5173/dashboard/`**
   * Saat prompt autentikasi API Key muncul di antarmuka web, masukkan API Key yang sesuai dengan `$env:RBTA_API_KEY`.
   * Halaman demo replay tersedia di: `http://127.0.0.1:5173/dashboard/demo`

---

## 5. Batasan Akademik & Integritas Data

* **Kerahasiaan Kredensial**: Jangan menyimpan atau melakukan git commit terhadap file yang berisi kunci produksi nyata `RBTA_API_KEY`.
* **Dataset Sidecar `.meta`**: File `.meta` yang ada di direktori ekspor Wazuh adalah metadata ekspor dan secara otomatis diabaikan oleh loader dataset. Hanya `.jsonl` dan `.jsonl.gz` yang diproses.
* **Frozen Model**: Model `rbta-if-v1` adalah artefak model beku (*frozen model*). Replay dan evaluasi tidak boleh melakukan training ulang (*refit*) pada model ini.
* **Prinsip Riset**: Sistem ini berfungsi untuk mereduksi dan memprioritaskan unit triase analis. Skor anomali bukan merupakan label kepastian serangan, dan setiap MetaAlert tetap terhubung dengan bukti mentahnya.
