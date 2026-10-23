/* ctypes wrapper of glhost/pilbox.h for work/py/test_pilbox.py */
#include "../glhost/pilbox.h"
void obs(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh) { pilbox_obs(src, sw, sh, dst, dw, dh); }
