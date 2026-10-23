#!/bin/sh
# Extract what nbhost needs from the APK (native libs + assets + classes.dex, ~1.7 GB) into apk_extract/ of this repo.
# Usage: loader/setup_apk.sh [APK] [OUT]    (defaults: apk/nb_68.279_release_7c5c5152.apk, apk_extract/)
# Read-only use of the APK; delete apk_extract/ to reclaim the space (the launchers re-create it when missing).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
APK="${1:-$HERE/apk/nb_68.279_release_7c5c5152.apk}"
OUT="${2:-$HERE/apk_extract}"
[ -f "$APK" ] || { echo "APK not found: $APK" >&2; exit 1; }
mkdir -p "$OUT"
cd "$OUT"
unzip -o -q "$APK" 'lib/arm64-v8a/*' 'assets/*' classes.dex
du -sh "$OUT"
