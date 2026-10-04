#!/usr/bin/env bash
# RBTA CT pull-based CD.
#
# Cron memanggil skrip ini tiap N menit. Skrip membandingkan HEAD lokal
# dengan origin/<branch>: bila sama, tidak melakukan apa pun; bila beda,
# pull --ff-only, rebuild image, up, lalu verifikasi /ready.
# Push ke branch = deploy. Tulis "[skip cd]" di pesan commit untuk lewati.
#
# Variabel lingkungan (opsional):
#   RBTA_REPO_DIR  default $HOME/rbta
#   RBTA_CD_BRANCH default prod/final-dashboard-demo
#   RBTA_CD_LOG    default <stack>/cd.log
#   RBTA_HOST_PORT default 8010 (hanya untuk health check)
set -euo pipefail

REPO_DIR="${RBTA_REPO_DIR:-$HOME/rbta}"
BRANCH="${RBTA_CD_BRANCH:-prod/final-dashboard-demo}"
STACK_DIR="$REPO_DIR/deploy/ct"
LOG="${RBTA_CD_LOG:-$STACK_DIR/cd.log}"
LOCK="$STACK_DIR/.cd.lock"
PORT="${RBTA_HOST_PORT:-8010}"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >>"$LOG"; }
fail() { log "ERROR $*"; exit 1; }

exec 9>"$LOCK"
flock -n 9 || exit 0  # run sebelumnya masih jalan; lewati jadwal ini

cd "$REPO_DIR" || fail "repo tidak ada: $REPO_DIR"
git fetch -q origin "$BRANCH" || fail "git fetch gagal"
LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"
[ "$LOCAL" = "$REMOTE" ] && exit 0  # sudah terbaru, tanpa log bising

if git log -1 --format=%B "origin/$BRANCH" | grep -q '\[skip cd\]'; then
  log "SKIP $REMOTE mengandung [skip cd]"
  exit 0
fi

log "DEPLOY $LOCAL -> $REMOTE"
git pull -q --ff-only origin "$BRANCH" || fail "git pull gagal (riwayat menyimpang?)"
cd "$STACK_DIR"
docker compose build >>"$LOG" 2>&1 || fail "docker compose build gagal"
# Volume dibuat milik root; container jalan sebagai UID 10001 (idempoten).
docker run --rm -v rbta-ct-live:/s -v rbta-ct-archive:/a alpine chown -R 10001:10001 /s /a >>"$LOG" 2>&1 || fail "chown volume gagal"
docker compose up -d >>"$LOG" 2>&1 || fail "docker compose up gagal"

for i in $(seq 1 24); do
  if curl -fs -o /dev/null "http://127.0.0.1:$PORT/ready"; then
    log "OK $REMOTE live (ready dalam ~$((i * 5)) dtk)"
    exit 0
  fi
  sleep 5
done
fail "/ready tidak hijau 120 dtk setelah up; periksa docker compose logs"
