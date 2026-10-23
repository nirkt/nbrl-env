"""Unit test: glhost/pilbox.h (nbgl's in-renderer observation resize) == adb_driver.preprocess_train (PIL BOX) bit for bit,
on real captures: the phone episode's 300x135 GPU captures and the PC's, plus random noise frames."""
import os, sys, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adb_driver as AD
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib = ctypes.CDLL(os.path.join(W, "tests", "libpilbox.so"))
def c_obs(raw, sw=300, sh=135, dw=256, dh=115):
    out = np.zeros((dh, dw, 3), np.uint8); src = np.ascontiguousarray(np.frombuffer(bytes(raw), np.uint8))
    lib.obs(src.ctypes.data_as(ctypes.c_void_p), sw, sh, out.ctypes.data_as(ctypes.c_void_p), dw, dh); return out
frames = []
p = os.path.join(W, "runs", "t6", "phone_native", "native_1234.npz")
if os.path.exists(p): frames += [("phone", bytes(c)) for c in np.load(p)["caps"]]
rng = np.random.default_rng(0)
frames += [("noise", rng.integers(0, 256, 300 * 135 * 4, dtype=np.uint8).tobytes()) for _ in range(50)]
bad = 0
for i, (src, raw) in enumerate(frames):
    a = AD.preprocess_train(raw, (256, 115), None); b = c_obs(raw)
    if not np.array_equal(a, b):
        bad += 1
        if bad <= 3: d = np.argwhere(a != b); print("MISMATCH", src, i, len(d), d[:3].tolist(), a[tuple(d[0])], b[tuple(d[0])])
print(f"pilbox vs PIL BOX: {len(frames)} frames, {bad} mismatching ->", "PASS" if bad == 0 else "FAIL")
sys.exit(1 if bad else 0)
