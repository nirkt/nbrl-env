#!/bin/sh
# Build nbhost (static AArch64 glibc binary; runs under qemu-aarch64 on x86 hosts, natively on ARM64 Linux).
set -e
cd "$(dirname "$0")"
python3 gen_jni.py >/dev/null
[ -f ../glhost/gen_gl.py ] && python3 ../glhost/gen_gl.py >/dev/null  # pctrain: gl_client_gen.c is copied from the build round
aarch64-linux-gnu-gcc -O2 -g -ffp-contract=off -fno-omit-frame-pointer -static -D_GNU_SOURCE -Wall -Wno-unused-function -Wno-incompatible-pointer-types \
  -Wno-int-conversion -fno-strict-aliasing -Iinc -o nbhost.tmp main.c elfload.c bionic_shim.c android_shim.c jni_fake.c boot.c hook.c offline.c battle_fix.c env.c prof.c encode.c heapcheck.c gl_client.c gl_client_gen.c thunk.S -lm -lpthread
mv -f nbhost.tmp nbhost && echo built nbhost
