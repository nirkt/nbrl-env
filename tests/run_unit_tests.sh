#!/usr/bin/env bash
# Unit tests (fast, no game process): protocol layout, the in-renderer observation resize vs PIL. The encoder test
# (py/test_encode.py) and the end-to-end smoke test (tests/test_e2e_smoke.py) boot game processes.
set -e
cd "$(dirname "$0")"
gcc -O2 -shared -fPIC -o libpilbox.so pilbox_lib.c -lm
python3 test_proto.py
python3 ../py/test_pilbox.py
