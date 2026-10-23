#!/usr/bin/env bash
# build nbgl (native GLES replay + capture/observation) — x86-64 host, Mesa EGL/GLESv2, X11 (window mode) / pbuffer (--hidden)
set -e
cd "$(dirname "$0")"
gcc -O2 -Wall -o nbgl.tmp nbgl.c -lEGL -lGLESv2 -lX11 -lm
mv -f nbgl.tmp nbgl && echo built nbgl
