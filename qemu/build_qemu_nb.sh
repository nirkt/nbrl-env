#!/bin/bash
# Build work/qemu/bin/qemu-aarch64-nb: QEMU 8.2.2 (src/qemu-8.2.2.tar.xz from download.qemu.org) + patches/*.patch,
# aarch64-linux-user only, not installed. ~5 min. The patches are switched on at run time by the Python side
# (py/nbpc/instance.py qemu_cmd_env: QEMU_NB_XPAGE=1 QEMU_NB_INLINE=1 QEMU_CPU=max,pauth-impdef=on).
#   0001 jump cache 16 bits + AArch64-friendly hash      0002 cross-page goto_tb chaining in user mode (QEMU_NB_XPAGE)
#   0003 inline jump-cache probe for indirect branches (QEMU_NB_INLINE) + tcg_gen_goto_ptr; QEMU_NB_DEBUG counters
set -e
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
rm -rf nb-src nb-build && mkdir -p nb-src nb-build bin
tar xf src/qemu-8.2.2.tar.xz -C nb-src --strip-components=1
(cd nb-src && for p in "$HERE"/patches/*.patch; do patch -p1 -s < "$p"; done)
cd nb-build
../nb-src/configure --target-list=aarch64-linux-user --disable-system --enable-linux-user --disable-docs --disable-tools \
  --disable-werror --disable-qom-cast-debug --disable-plugins --enable-trace-backends=nop --enable-lto > configure.log 2>&1 || { tail -30 configure.log; exit 1; }
ninja qemu-aarch64 > build.log 2>&1 || { tail -40 build.log; exit 1; }
cp qemu-aarch64 "$HERE/bin/qemu-aarch64-nb.tmp" && mv -f "$HERE/bin/qemu-aarch64-nb.tmp" "$HERE/bin/qemu-aarch64-nb"
cd "$HERE" && rm -rf nb-src nb-build
echo "built bin/qemu-aarch64-nb"
