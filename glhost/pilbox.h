/* Exact port of Pillow's Image.resize(size, Image.BOX) for 8-bit RGB (Pillow 12.1.1 libImaging/Resample.c:
 * precompute_coeffs + normalize_coeffs_8bpc + ImagingResampleHorizontal_8bpc / ImagingResampleVertical_8bpc, two passes,
 * horizontal first over the rows the vertical pass needs, 22-bit fixed point, clip8 after each pass).
 * pilbox_obs(): the phone pipeline's adb_driver.preprocess_train minus the HUD mask, i.e. GPU capture RGBA bottom-left
 * (sw x sh) -> flip to top-left, drop alpha -> BOX resize to (dw x dh) -> RGB top-left. Verified bit-exact against PIL
 * by work/py/test_pilbox.py. */
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PILBOX_PB 22
typedef struct { int in, out, ksize; int *bounds; int32_t *kk; } PilCoeffs;

static int pilbox_coeffs(int inSize, int outSize, PilCoeffs *c) {
    double scale = (double)((float)inSize - 0.0f) / outSize, filterscale = scale < 1.0 ? 1.0 : scale;
    double support = 0.5 * filterscale;                     /* BOX support 0.5 */
    int ksize = (int)ceil(support) * 2 + 1;
    double *pre = calloc((size_t)outSize * ksize, sizeof(double)); int *b = malloc((size_t)outSize * 2 * sizeof(int));
    for (int xx = 0; xx < outSize; xx++) {
        double center = 0.0 + (xx + 0.5) * scale, ww = 0.0, ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double *k = pre + (size_t)xx * ksize;
        for (int x = 0; x < xmax; x++) { double t = (x + xmin - center + 0.5) * ss; double w = (t > -0.5 && t <= 0.5) ? 1.0 : 0.0; k[x] = w; ww += w; }
        for (int x = 0; x < xmax; x++) if (ww != 0.0) k[x] /= ww;
        b[xx * 2] = xmin; b[xx * 2 + 1] = xmax;
    }
    int32_t *kk = malloc((size_t)outSize * ksize * sizeof(int32_t));
    for (int i = 0; i < outSize * ksize; i++)
        kk[i] = pre[i] < 0 ? (int32_t)(-0.5 + pre[i] * (1 << PILBOX_PB)) : (int32_t)(0.5 + pre[i] * (1 << PILBOX_PB));
    free(pre); c->in = inSize; c->out = outSize; c->ksize = ksize; c->bounds = b; c->kk = kk;
    return ksize;
}
static inline uint8_t pilbox_clip8(int32_t v) { v >>= PILBOX_PB; return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v); }

/* src: RGBA, bottom-left origin (glReadPixels), sw x sh; dst: RGB top-left dw x dh */
static void pilbox_obs(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh) {
    static PilCoeffs H, V; static int hw, hh, vw, vh;
    if (!H.kk || hw != sw || hh != dw) { if (H.kk) { free(H.kk); free(H.bounds); } pilbox_coeffs(sw, dw, &H); hw = sw; hh = dw; }
    if (!V.kk || vw != sh || vh != dh) { if (V.kk) { free(V.kk); free(V.bounds); } pilbox_coeffs(sh, dh, &V); vw = sh; vh = dh; }
    int y0 = V.bounds[0], y1 = V.bounds[dh * 2 - 2] + V.bounds[dh * 2 - 1], th = y1 - y0;
    uint8_t *tmp = malloc((size_t)dw * th * 3);
    for (int ty = 0; ty < th; ty++) {                       /* horizontal pass over source rows y0..y1 (top-left rows) */
        const uint8_t *row = src + (size_t)(sh - 1 - (ty + y0)) * sw * 4;   /* flip: top-left row r = GL row sh-1-r */
        for (int xx = 0; xx < dw; xx++) {
            int xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1]; const int32_t *k = H.kk + (size_t)xx * H.ksize;
            int32_t s0 = 1 << (PILBOX_PB - 1), s1 = s0, s2 = s0;
            for (int x = 0; x < xmax; x++) { const uint8_t *p = row + (size_t)(x + xmin) * 4; s0 += p[0] * k[x]; s1 += p[1] * k[x]; s2 += p[2] * k[x]; }
            uint8_t *o = tmp + ((size_t)ty * dw + xx) * 3; o[0] = pilbox_clip8(s0); o[1] = pilbox_clip8(s1); o[2] = pilbox_clip8(s2);
        }
    }
    for (int yy = 0; yy < dh; yy++) {                       /* vertical pass */
        int ymin = V.bounds[yy * 2] - y0, ymax = V.bounds[yy * 2 + 1]; const int32_t *k = V.kk + (size_t)yy * V.ksize;
        for (int xx = 0; xx < dw; xx++) {
            int32_t s0 = 1 << (PILBOX_PB - 1), s1 = s0, s2 = s0;
            for (int y = 0; y < ymax; y++) { const uint8_t *p = tmp + ((size_t)(y + ymin) * dw + xx) * 3; s0 += p[0] * k[y]; s1 += p[1] * k[y]; s2 += p[2] * k[y]; }
            uint8_t *o = dst + ((size_t)yy * dw + xx) * 3; o[0] = pilbox_clip8(s0); o[1] = pilbox_clip8(s1); o[2] = pilbox_clip8(s2);
        }
    }
    free(tmp);
}
