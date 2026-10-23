#!/bin/bash
# Build a private qemu-aarch64 (linux-user only, -O2 -g, not installed) from work/qemu/src/qemu-$1.tar.xz.
# usage: bash build_qemu.sh 11.1.2 [extra configure args]   -> work/qemu/bin/qemu-aarch64-$1[-tag]
set -e
V=$1; shift; TAG=${QTAG:+-$QTAG}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
[ -d qemu-$V ] || tar xf src/qemu-$V.tar.xz
[ -n "$QPATCH" ] && (cd qemu-$V && for p in $QPATCH; do patch -p1 -N < "$HERE/$p"; done)
mkdir -p build-$V$TAG bin && cd build-$V$TAG
../qemu-$V/configure --target-list=aarch64-linux-user --disable-system --enable-linux-user --disable-docs --disable-tools \
  --disable-werror "$@" > configure.log 2>&1 || { tail -30 configure.log; exit 1; }
ninja qemu-aarch64 > build.log 2>&1 || { tail -40 build.log; exit 1; }
cp qemu-aarch64 "$HERE/bin/qemu-aarch64-$V$TAG"; echo "built bin/qemu-aarch64-$V$TAG"
