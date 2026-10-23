#!/usr/bin/env python3
"""Unit test: the Python ctypes mirror (py/nbpc/proto.py) matches loader/nbenv_proto.h (sizes compiled by the C compiler
with the same layout rules as the AArch64 guest: an x86-64 build of a probe program, little-endian, natural alignment)."""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE); sys.path.insert(0, os.path.join(W, "py"))
from nbpc import proto as P
import ctypes as C
src = r'''#include <stdio.h>
#include <stddef.h>
#include "nbenv_proto.h"
int main(void){printf("%zu %zu %zu %zu %d %d %d %d %d %d %d\n", sizeof(NbeCmd), sizeof(NbeResp), sizeof(NbeHdr), sizeof(NbePlayer),
  NBE_OFF_CMD, NBE_OFF_RESP, NBE_OFF_STATE, NBE_OFF_MAP, NBE_OFF_REC, NBE_OFF_PIX, NBE_TOTAL); return 0;}'''
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "p.c"), "w").write(src)
    subprocess.check_call(["gcc", "-I", os.path.join(W, "loader"), "-o", os.path.join(d, "p"), os.path.join(d, "p.c")])
    c = [int(x) for x in subprocess.check_output([os.path.join(d, "p")]).split()]
py = [C.sizeof(P.Cmd), C.sizeof(P.Resp), C.sizeof(P.Hdr), C.sizeof(P.Player), P.OFF_CMD, P.OFF_RESP, P.OFF_STATE, P.OFF_MAP,
      P.OFF_REC, P.OFF_PIX, P.TOTAL]
names = ["Cmd", "Resp", "Hdr", "Player", "OFF_CMD", "OFF_RESP", "OFF_STATE", "OFF_MAP", "OFF_REC", "OFF_PIX", "TOTAL"]
bad = [(n, a, b) for n, a, b in zip(names, c, py) if a != b]
print("C:", dict(zip(names, c))); print("python:", dict(zip(names, py)))
print("PASS" if not bad else f"FAIL {bad}"); sys.exit(1 if bad else 0)
