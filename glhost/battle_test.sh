#!/bin/bash
# Headless end-to-end check: boot -> lobby -> tap PLAY -> native offline battle; prints battle/crash lines.
#   battle_test.sh <apk-extract-dir> <outdir> [seconds-after-play]
APKX=$1; OUT=$2; AFTER=${3:-90}; HERE="$(cd "$(dirname "$0")" && pwd)"; SHM=/dev/shm/nbgl_bt_$$
"$HERE/nbgl" --shm $SHM --size 2400x1080 --hidden --snap $OUT/bt_snap.ppm > $OUT/bt_nbgl.log 2>&1 & GL=$!
sleep 1.5
timeout $((AFTER + 150)) unshare -rn qemu-aarch64 "$HERE/../loader/nbhost" --libs $APKX/lib/arm64-v8a --assets $APKX \
  --payloads "$HERE/../loader/payloads" --stage boot --gl $SHM --dpi 450 --run-seconds $((AFTER + 140)) > $OUT/bt_game.log 2>&1 & G=$!
for i in $(seq 1 120); do grep -q "home=1" $OUT/bt_game.log 2>/dev/null && break; grep -q "\[crash\]" $OUT/bt_game.log && break; sleep 1; done
sleep 12; python3 "$HERE/tap.py" $SHM 2134 970
for i in $(seq 1 $AFTER); do grep -q "\[crash\]" $OUT/bt_game.log && break; sleep 1; done
cp $OUT/bt_snap.ppm $OUT/bt_last.ppm 2>/dev/null
kill $G $GL 2>/dev/null; wait 2>/dev/null; rm -f $SHM
grep -E "\[battle|\[crash\] signal" $OUT/bt_game.log | head -20; tail -1 $OUT/bt_nbgl.log
