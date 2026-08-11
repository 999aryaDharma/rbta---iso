# Data Pipeline Specification

## Tujuan

Mendefinisikan satu jalur data yang konsisten dari Wazuh JSON/REST payload menjadi input RBTA tanpa mengubah semantik alert.

## Input Canonical

Pipeline harus menerima alert Wazuh dari dua sumber yang ekuivalen secara semantik:

1. **Batch research mode** — JSONL/CSV historis.
2. **Operational mode** — REST API ingress yang menerima payload alert Wazuh secara langsung.

Kedua mode wajib berakhir pada schema internal yang sama sebelum masuk ke RBTA.

## Canonical Raw Alert Schema

Field minimum:

```text
wazuh_alert_id
occurred_at
timestamp_received (operational only)
agent_id
agent_name
rule_group_primary
rule_level
rule_id
srcip
srcip_type
criticality_score
mitre_tactics[]
has_critical_mitre
```

Field audit/enrichment yang boleh dipertahankan tetapi bukan syarat RBTA:

```text
rule_description
rule_groups_all[]
mitre_techniques[]
mitre_ids[]
agent_ip
decoder_name
location
manager_name
syscheck fields
audit fields
full_log_truncated
```

## Parsing Rules

### Timestamp

- `occurred_at` berasal dari timestamp event Wazuh.
- Simpan sebagai UTC-aware timestamp pada boundary I/O.
- Internal processing boleh menggunakan normalized UTC representation yang konsisten.
- Jangan mengganti event time dengan receive time.

### Deduplication

Primary dedup key:

```text
wazuh_alert_id
```

Operational API wajib idempotent: payload dengan `wazuh_alert_id` yang sudah diproses tidak boleh membuat alert baru kedua kali.

### Primary Rule Group

Jika `rule.groups` memiliki lebih dari satu nilai, pilih `rule_group_primary` menggunakan konfigurasi domain terpusat.

Konfigurasi ini hanya boleh berada pada satu module/config source. Tidak boleh disalin ke parser, RBTA, IF, dan notifier secara terpisah.

### MITRE

Parser harus mempertahankan **daftar taktik unik**, bukan hanya flag `has_mitre`.

Contoh canonical value:

```json
{
  "mitre_tactics": ["Credential Access", "Defense Evasion"]
}
```

Hal ini wajib karena feature final memerlukan:

- `mitre_tactic_count`
- `critical_mitre_tactic_present`

### Agent Criticality

`criticality_score` adalah domain mapping 1–4:

```text
1 = Low
2 = Medium
3 = High
4 = Critical
```

Mapping harus terpusat dan dapat dikonfigurasi.

## Preprocessing

Preprocessing hanya melakukan:

- validation schema;
- timestamp normalization;
- numeric conversion;
- `rule_level` validation 0–15;
- normalization string untuk identifier/kategori;
- deduplication;
- penanganan malformed rows secara eksplisit dan terukur.

Preprocessing **tidak boleh**:

- melakukan feature engineering IF;
- mengubah chronological behavior;
- mengisi label synthetic;
- menentukan anomaly;
- melakukan global sort yang menyebabkan pengujian out-of-order menjadi tidak mungkin tanpa mode pengujian khusus.

## Batch vs Live Equivalence

Satu payload yang sama harus menghasilkan canonical alert yang sama baik dibaca dari file maupun diterima REST API.

Acceptance test:

```text
parse_file(alert_json) == parse_api_payload(alert_json)
```

untuk seluruh field canonical kecuali metadata transport seperti `timestamp_received`.

## Data Quality Report

Batch research mode harus menghasilkan report minimum:

```text
input_rows
valid_rows
invalid_rows
duplicate_rows
missing_agent_id
missing_rule_group
missing_rule_id
invalid_timestamp
rule_group_distribution
agent_distribution
mitre_coverage
critical_mitre_coverage
```

Tidak boleh ada angka hardcoded.