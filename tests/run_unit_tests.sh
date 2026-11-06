#!/usr/bin/env bash
# Unit tests (fast, no game process): protocol layout and the state-observation tests. The end-to-end smoke test
# (tests/test_e2e_smoke.py) boots game processes and takes about 2 minutes.
set -e
cd "$(dirname "$0")"
python3 test_proto.py
python3 test_fair.py
python3 test_grid4.py
echo "unit tests: PASS"
