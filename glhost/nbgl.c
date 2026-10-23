/* nbgl: native x86 renderer for nbhost. Opens an X11 window with a GLES 3.2 context on this PC's GPU (Mesa/EGL),
 * replays the GL command stream the game writes into shared memory, and sends mouse input back as touches.
 *   ./nbgl --shm /dev/shm/nbgl --size 1600x720 [--hidden] [--snap out.ppm]
 * Left button = finger 0 (down / move / up); W A S D / arrows = finger 1 on the movement joystick. Close the window to quit. */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include "nbgl_proto.h"
#include <execinfo.h>
#include <signal.h>
static void on_term(int sig) { (void)sig; extern NbglShm *S_ext; if (S_ext) S_ext->host_alive = 0; _exit(0); }
static void on_usr1(int sig) { (void)sig; void *b[32]; int n = backtrace(b, 32); fprintf(stderr, "[nbgl] SIGUSR1 backtrace:\n"); backtrace_symbols_fd(b, n, 2); }

static NbglShm *S; NbglShm *S_ext;
static EGLDisplay g_dpy; static EGLSurface g_surf;
static int g_verbose; static long g_cmds; static int g_shaders, g_shfail, g_progs, g_linkfail;
#define real_getsh glGetShaderiv
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline float nb_u2f(uint64_t u) { uint32_t x = (uint32_t)u; float f; memcpy(&f, &x, 4); return f; }
#define IP(i) (A[i] == NBGL_NULL ? NULL : (void *)(cmd + A[i]))
#define OP(i) (A[i] == NBGL_NULL ? NULL : (void *)(reply + A[i]))
static int g_next; static int g_caps[64][2], g_ncaps, g_noprec;
static int g_opstats; static double g_opt[2048]; static long g_opn[2048], g_opb[2048], g_fmt_n[64], g_fmt_b[64]; static unsigned g_fmt_k[64]; static int g_nfmt;   /* NBGL_OPSTATS=1 */
static void GL_APIENTRY zero_prec(GLenum s, GLenum p, GLint *r, GLint *pr) { (void)s; (void)p; r[0] = r[1] = 0; *pr = 0; }   /* --caps 0xPNAME=value,... */
static void (GL_APIENTRY *real_getiv)(GLenum, GLint *);
/* limits the game sees: the filtered extension count, and phone-like (Adreno 650) values where desktop Mesa reports
 * "unlimited" (the game sizes allocations from GL_MAX_UNIFORM_BLOCK_SIZE = 2^31-1 on radeonsi and then fails) */
static void GL_APIENTRY filt_getiv(GLenum p, GLint *v) {
    real_getiv(p, v);
    if (p == GL_NUM_EXTENSIONS) *v = g_next;
    if (p == GL_MAX_UNIFORM_BLOCK_SIZE && *v > 65536) *v = 65536;
    for (int i = 0; i < g_ncaps; i++) if (g_caps[i][0] == p) *v = g_caps[i][1];
}
static void nb_special(unsigned op, const uint64_t *A, uint8_t *cmd, uint8_t *reply, uint64_t *R);
#include "nbgl_server_gen.c"

/* extensions the guest may see: only ones that add no entry points (we forward core GLES 3.0 only) */
static const char *g_extlist, *g_spoof;   /* --ext a,b,c : expose exactly these (when the driver has them) */
static int ext_ok(const char *e) {
    if (g_extlist) { const char *p = strstr(g_extlist, e); size_t n = strlen(e); return p && (p[n] == ',' || p[n] == 0) && (p == g_extlist || p[-1] == ','); }
    static const char *ok[] = {"texture_compression", "compressed_", "_texture_float", "texture_half_float", "color_buffer_",
        "depth24", "depth32", "packed_depth_stencil", "rgb8_rgba8", "element_index_uint", "texture_npot", "texture_format_BGRA8888",
        "EXT_sRGB", "standard_derivatives", "fragment_precision_high", "vertex_half_float", "texture_rg", "shader_texture_lod", 0};
    for (int i = 0; ok[i]; i++) if (strstr(e, ok[i])) return 1;
    return 0;
}
/* ---- ASTC upload cache. This GPU has no ASTC: Mesa decodes every glCompressedTexImage2D of an ASTC texture on the
 * CPU (~3.7 ms each), and every battle re-uploads the same textures. Linear ASTC uploads (0x93B0..0x93BD) are decoded
 * once in a private GL context of the same driver (upload as ASTC, texelFetch each texel into an RGBA8 target, read
 * back) and cached by content (64-bit hash + full compare); the game's texture then gets those RGBA8 texels with
 * glTexImage2D, on the first upload and on every later one alike (so all levels of a texture share one format).
 * Least-recently-used eviction above NBGL_ASTC_MB (default 256 MB per renderer).
 * NBGL_ASTC_CACHE=0: off; NBGL_ASTC_LOG=1: log hits/misses; NBGL_ASTC_CHECK=1: compare the decoded texels with the driver's own ASTC sampling (float). */
static EGLContext g_ctx_main, g_ctx2; static EGLSurface g_surf2; static int g_astc_on = 1, g_astc_check;
static long g_astc_hit, g_astc_miss, g_astc_bad, g_astc_evicted; static size_t g_astc_bytes;
typedef struct AstcEnt { uint64_t h; uint32_t fmt, w, h_, size; uint8_t *comp, *rgba; struct AstcEnt *next, *older, *newer; } AstcEnt;
static AstcEnt *g_astc_oldest, *g_astc_newest; static size_t g_astc_budget = (size_t)256 << 20;   /* NBGL_ASTC_MB */
static AstcEnt *g_astc_tab[4096];
static uint64_t astc_hash(const uint8_t *p, size_t n, uint64_t seed) {
    uint64_t h = seed ^ (n * 0x9E3779B97F4A7C15ULL); size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t v; memcpy(&v, p + i, 8); h = (h ^ v) * 0xff51afd7ed558ccdULL; h ^= h >> 32; }
    for (; i < n; i++) { h = (h ^ p[i]) * 0x100000001b3ULL; }
    return h ^ (h >> 29);
}
static GLuint g_dec_prog, g_dec_fbo, g_dec_vao, g_dec_prog_f;
static GLuint dec_program(int fl) {
    const char *vs = "#version 300 es\nvoid main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }";
    const char *fs = "#version 300 es\nprecision highp float; uniform highp sampler2D t; out highp vec4 o; void main(){ o = texelFetch(t, ivec2(gl_FragCoord.xy), 0); }";
    (void)fl;
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER), pr = glCreateProgram();
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v); glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(pr, v); glAttachShader(pr, f); glLinkProgram(pr); GLint ok = 0; glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) fprintf(stderr, "[nbgl] ASTC decode program failed to link\n");
    return pr;
}
/* in g_ctx2: texels of (fmt, w, h, data) as the driver decodes them; out = w*h*4 bytes (RGBA8) or w*h*16 (float) */
static int astc_decode(uint32_t fmt, int w, int h, int size, const void *data, void *out, int as_float) {
    if (!eglMakeCurrent(g_dpy, g_surf2, g_surf2, g_ctx2)) return -1;
    if (!g_dec_prog) { g_dec_prog = dec_program(0); glGenFramebuffers(1, &g_dec_fbo); glGenVertexArrays(1, &g_dec_vao); }
    GLuint t, c; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, size, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glGenTextures(1, &c); glBindTexture(GL_TEXTURE_2D, c);
    glTexStorage2D(GL_TEXTURE_2D, 1, as_float ? GL_RGBA32F : GL_RGBA8, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, g_dec_fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, c, 0);
    int ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    if (ok) {
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, t);
        glUseProgram(g_dec_prog); glUniform1i(glGetUniformLocation(g_dec_prog, "t"), 0);
        glBindVertexArray(g_dec_vao); glViewport(0, 0, w, h); glDrawArrays(GL_TRIANGLES, 0, 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadPixels(0, 0, w, h, GL_RGBA, as_float ? GL_FLOAT : GL_UNSIGNED_BYTE, out);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0); glDeleteTextures(1, &t); glDeleteTextures(1, &c);
    GLenum err = glGetError();
    eglMakeCurrent(g_dpy, g_surf, g_surf, g_ctx_main);
    return ok && !err ? 0 : -1;
}
/* op 37 glCompressedTexImage2D(target, level, internalformat, w, h, border, imageSize, data): 1 = handled here */
static int astc_upload(const uint64_t *A, uint8_t *cmd) {
    uint32_t fmt = (uint32_t)A[2]; int w = (int)A[3], h = (int)A[4], size = (int)A[6];
    if (!g_astc_on || fmt < 0x93B0 || fmt > 0x93BD || A[5] != 0 || A[7] == NBGL_NULL || w <= 0 || h <= 0 || size <= 0) return 0;
    const uint8_t *data = cmd + A[7];
    uint64_t hh = astc_hash(data, size, ((uint64_t)fmt << 40) ^ ((uint64_t)w << 20) ^ (uint64_t)h);
    AstcEnt *e = g_astc_tab[hh & 4095];
    while (e && !(e->h == hh && e->fmt == fmt && e->w == (uint32_t)w && e->h_ == (uint32_t)h && e->size == (uint32_t)size && !memcmp(e->comp, data, size))) e = e->next;
    if (!e) {
        uint8_t *rgba = malloc((size_t)w * h * 4);
        if (!rgba || astc_decode(fmt, w, h, size, data, rgba, 0)) { free(rgba); g_astc_bad++; return 0; }   /* fall back to the driver */
        if (g_astc_check) {   /* the 8-bit texels must equal the driver's own sampling exactly */
            float *fl = malloc((size_t)w * h * 16); long diff = 0;
            if (fl && !astc_decode(fmt, w, h, size, data, fl, 1))
                for (long i = 0; i < (long)w * h * 4; i++) { float d = fl[i] * 255.0f - (float)rgba[i]; if (d > 1e-3f || d < -1e-3f) diff++; }
            fprintf(stderr, "[nbgl] ASTC check %#x %dx%d: %ld texel channels differ from 8-bit\n", fmt, w, h, diff); free(fl);
        }
        e = calloc(1, sizeof *e); e->h = hh; e->fmt = fmt; e->w = w; e->h_ = h; e->size = size;
        e->comp = malloc(size); memcpy(e->comp, data, size); e->rgba = rgba;
        size_t need = size + (size_t)w * h * 4;
        while (g_astc_oldest && g_astc_bytes + need > g_astc_budget) {   /* evict the oldest entries (FIFO) */
            AstcEnt *o = g_astc_oldest, **pp = &g_astc_tab[o->h & 4095];
            while (*pp && *pp != o) pp = &(*pp)->next;
            if (*pp) *pp = o->next;
            g_astc_oldest = o->newer; if (g_astc_oldest) g_astc_oldest->older = NULL; else g_astc_newest = NULL;
            g_astc_bytes -= o->size + (size_t)o->w * o->h_ * 4; free(o->comp); free(o->rgba); free(o); g_astc_evicted++;
        }
        if (need <= g_astc_budget) {
            e->next = g_astc_tab[hh & 4095]; g_astc_tab[hh & 4095] = e; g_astc_bytes += need;
            e->older = g_astc_newest; e->newer = NULL; if (g_astc_newest) g_astc_newest->newer = e; else g_astc_oldest = e; g_astc_newest = e;
        } else e->next = (AstcEnt *)1;   /* larger than the whole budget: use once */
        g_astc_miss++; if (getenv("NBGL_ASTC_LOG")) fprintf(stderr, "[astc] miss %#x %dx%d %d B\n", fmt, w, h, size);
    } else {
        g_astc_hit++; if (getenv("NBGL_ASTC_LOG")) fprintf(stderr, "[astc] hit %#x %dx%d %d B\n", fmt, w, h, size);
        if (e != g_astc_newest) {   /* LRU: the entry becomes the newest */
            if (e->older) e->older->newer = e->newer; else g_astc_oldest = e->newer;
            e->newer->older = e->older;
            e->older = g_astc_newest; e->newer = NULL; g_astc_newest->newer = e; g_astc_newest = e;
        }
    }
    /* upload the RGBA8 texels with a neutral unpack state (restored afterwards) */
    GLint ub = 0, al, rl, sr, sp, ih, si; glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &ub);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &al); glGetIntegerv(GL_UNPACK_ROW_LENGTH, &rl); glGetIntegerv(GL_UNPACK_SKIP_ROWS, &sr);
    glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &sp); glGetIntegerv(GL_UNPACK_IMAGE_HEIGHT, &ih); glGetIntegerv(GL_UNPACK_SKIP_IMAGES, &si);
    if (ub) glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4); glPixelStorei(GL_UNPACK_ROW_LENGTH, 0); glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0); glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0); glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
    glTexImage2D((GLenum)A[0], (GLint)A[1], GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, e->rgba);
    glPixelStorei(GL_UNPACK_ALIGNMENT, al); glPixelStorei(GL_UNPACK_ROW_LENGTH, rl); glPixelStorei(GL_UNPACK_SKIP_ROWS, sr);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, sp); glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, ih); glPixelStorei(GL_UNPACK_SKIP_IMAGES, si);
    if (ub) glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ub);
    if (e->next == (AstcEnt *)1) { free(e->comp); free(e->rgba); free(e); }
    return 1;
}
static char g_ext[65536]; static const char *g_exti[512];
static void build_ext(void) {
    GLint n = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (int i = 0; i < n; i++) {
        const char *e = (const char *)glGetStringi(GL_EXTENSIONS, i);
        if (!ext_ok(e)) continue;
        g_exti[g_next++] = e; strcat(g_ext, e); strcat(g_ext, " ");
    }
    fprintf(stderr, "[nbgl] exposing %d of %d extensions to the game: %s\n", g_next, n, g_ext);
}
/* GPU downscale + readback of the frame about to be presented (the phone agent's OBSSRC obs_capture, same chain) */
#define MAXL 8
static GLuint g_cfbo[MAXL], g_crbo[MAXL]; static int g_clw[MAXL], g_clh[MAXL], g_cn, g_csw, g_csh, g_cdw, g_cdh;
static long g_captures; static double g_cap_us, g_box_us;
#include "pilbox.h"
static int cap_build(int sw, int sh, int dw, int dh) {
    for (int i = 0; i < g_cn; i++) { glDeleteFramebuffers(1, &g_cfbo[i]); glDeleteRenderbuffers(1, &g_crbo[i]); }
    g_cn = 0; int cw = sw, ch = sh;
    while (g_cn < MAXL - 1 && cw / 2 >= dw && ch / 2 >= dh) { cw /= 2; ch /= 2; g_clw[g_cn] = cw; g_clh[g_cn] = ch; g_cn++; }
    if (g_cn == 0 || g_clw[g_cn - 1] != dw || g_clh[g_cn - 1] != dh) { g_clw[g_cn] = dw; g_clh[g_cn] = dh; g_cn++; }
    for (int i = 0; i < g_cn; i++) {
        glGenRenderbuffers(1, &g_crbo[i]); glBindRenderbuffer(GL_RENDERBUFFER, g_crbo[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, g_clw[i], g_clh[i]);
        glGenFramebuffers(1, &g_cfbo[i]); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_cfbo[i]);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, g_crbo[i]);
        if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return -1;
    }
    g_csw = sw; g_csh = sh; g_cdw = dw; g_cdh = dh; return 0;
}
static void capture(int dw, int dh, uint8_t *out) {
    double t0 = now();
    GLint pd = 0, pr = 0, prb = 0, pa = 4, prl = 0, psr = 0, psp = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd); glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr); glGetIntegerv(GL_RENDERBUFFER_BINDING, &prb);
    glGetIntegerv(GL_PACK_ALIGNMENT, &pa); glGetIntegerv(GL_PACK_ROW_LENGTH, &prl); glGetIntegerv(GL_PACK_SKIP_ROWS, &psr); glGetIntegerv(GL_PACK_SKIP_PIXELS, &psp);
    GLboolean sc = glIsEnabled(GL_SCISSOR_TEST), rd = glIsEnabled(GL_RASTERIZER_DISCARD);
    if (sc) glDisable(GL_SCISSOR_TEST); if (rd) glDisable(GL_RASTERIZER_DISCARD);
    int sw = S->width, sh = S->height;
    if (!g_cn || g_csw != sw || g_csh != sh || g_cdw != dw || g_cdh != dh) cap_build(sw, sh, dw, dh);
    GLuint src = 0; int cw = sw, ch = sh;   /* the default framebuffer = the frame the game just drew */
    for (int i = 0; i < g_cn; i++) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, src); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_cfbo[i]);
        glBlitFramebuffer(0, 0, cw, ch, 0, 0, g_clw[i], g_clh[i], GL_COLOR_BUFFER_BIT, GL_LINEAR);
        src = g_cfbo[i]; cw = g_clw[i]; ch = g_clh[i];
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glPixelStorei(GL_PACK_ALIGNMENT, 1); glPixelStorei(GL_PACK_ROW_LENGTH, 0); glPixelStorei(GL_PACK_SKIP_ROWS, 0); glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glReadPixels(0, 0, dw, dh, GL_RGBA, GL_UNSIGNED_BYTE, out);
    glPixelStorei(GL_PACK_ALIGNMENT, pa); glPixelStorei(GL_PACK_ROW_LENGTH, prl); glPixelStorei(GL_PACK_SKIP_ROWS, psr); glPixelStorei(GL_PACK_SKIP_PIXELS, psp);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, pr); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pd); glBindRenderbuffer(GL_RENDERBUFFER, prb);
    if (sc) glEnable(GL_SCISSOR_TEST); if (rd) glEnable(GL_RASTERIZER_DISCARD);
    g_captures++; g_cap_us += (now() - t0) * 1e6;
}
static void nb_special(unsigned op, const uint64_t *A, uint8_t *cmd, uint8_t *reply, uint64_t *R) {
    if (op == NBGL_OP_CAPTURE) {
        int w = (int)(int64_t)A[0], h = (int)(int64_t)A[1];
        if (w < 0 && h < 0) {   /* policy observation: the phone's 300x135 GPU capture, then PIL BOX -> (-w x -h) RGB top-left */
            static uint8_t raw[300 * 135 * 4]; double t0 = now();
            capture(300, 135, raw); pilbox_obs(raw, 300, 135, reply + A[2], -w, -h); g_box_us += (now() - t0) * 1e6;
            *R = (uint64_t)(-w) * (-h) * 3; return;
        }
        capture(w, h, reply + A[2]); *R = (uint64_t)w * h * 4; return;
    }
    if (op == NBGL_OP_MAPREAD) {
        void *p = glMapBufferRange((GLenum)A[0], (GLintptr)A[1], (GLsizeiptr)A[2], GL_MAP_READ_BIT);
        if (p && A[3] != NBGL_NULL) memcpy(reply + A[3], p, A[2]);
        glUnmapBuffer((GLenum)A[0]); return;
    }
    const char *n = nb_op_name(op);
    if (!strcmp(n, "glPixelStorei")) glPixelStorei((GLenum)A[0], (GLint)A[1]);
    else if (!strcmp(n, "glGetString") || !strcmp(n, "glGetStringi")) {
        const char *s;
        if (n[11] == 'i') s = A[0] == GL_EXTENSIONS ? (A[1] < (uint64_t)g_next ? g_exti[A[1]] : NULL) : (const char *)glGetStringi(A[0], A[1]);
        else s = A[0] == GL_EXTENSIONS ? g_ext : (const char *)glGetString(A[0]);
        if (g_spoof && !(n[11] == 'i')) {   /* --spoof vendor|renderer|version : the device identity the game sees */
            static char v[3][128]; if (!v[0][0]) sscanf(g_spoof, "%127[^|]|%127[^|]|%127[^|]", v[0], v[1], v[2]);
            if (A[0] == GL_VENDOR) s = v[0]; else if (A[0] == GL_RENDERER) s = v[1]; else if (A[0] == GL_VERSION) s = v[2];
        }
        if (s) { snprintf((char *)OP(2), 65536, "%s", s); *R = strlen(s) + 1; } else *R = 0;
        if (g_verbose) fprintf(stderr, "[nbgl] glGetString%s(%#lx) -> %.80s\n", n[11] == 'i' ? "i" : "", (unsigned long)A[0], s ? s : "(null)");
    } else if (!strcmp(n, "glShaderSource")) { const char *s = IP(2); glShaderSource((GLuint)A[0], 1, &s, NULL); }
    else if (!strcmp(n, "glTransformFeedbackVaryings") || !strcmp(n, "glGetUniformIndices")) {
        const char *v[256]; const char *p = IP(2); int cnt = (int)A[1];
        for (int i = 0; i < cnt && i < 256; i++) { v[i] = p; p += strlen(p) + 1; }
        if (n[2] == 'T') glTransformFeedbackVaryings((GLuint)A[0], cnt, v, (GLenum)A[3]);
        else glGetUniformIndices((GLuint)A[0], cnt, v, OP(3));
    } else fprintf(stderr, "[nbgl] unknown op %u (%s)\n", op, n);
}

/* on-screen FPS counter (window mode): a small child window at the top-left of the game window, text updated with the
 * fps every 2 s. It selects no pointer events (taps pass through to the game window) and touches no GL state. --no-fps: off. */
static Display *fps_dpy; static Window fps_win; static GC fps_gc; static XFontStruct *fps_font; static int fps_on = 1;
static int fps_asc, fps_h;
#define OV_N 64
static double ov_t[OV_N]; static uint64_t ov_f[OV_N]; static int ov_i, ov_cnt;
static void fps_init(Display *x, Window parent) {
    if (!fps_on) return;
    unsigned long black = BlackPixel(x, DefaultScreen(x));
    fps_font = XLoadQueryFont(x, "-*-*-bold-r-*-*-30-*-*-*-*-*-*-*"); if (!fps_font) fps_font = XLoadQueryFont(x, "fixed");
    fps_asc = fps_font ? fps_font->ascent : 16; fps_h = fps_asc + (fps_font ? fps_font->descent : 4) + 4;
    fps_dpy = x; fps_win = XCreateSimpleWindow(x, parent, 8, 8, 160, fps_h, 0, black, black);
    fps_gc = XCreateGC(x, fps_win, 0, NULL); XSetForeground(x, fps_gc, 0x00ff00);
    if (fps_font) XSetFont(x, fps_gc, fps_font->fid);
    XMapWindow(x, fps_win); XFlush(x);
}
static void fps_text(double fps) {
    if (!fps_win) return;
    char s[32]; int n = snprintf(s, sizeof s, "%.1f FPS", fps);
    int w = fps_font ? XTextWidth(fps_font, s, n) + 12 : 160;
    XResizeWindow(fps_dpy, fps_win, w, fps_h);
    XClearWindow(fps_dpy, fps_win); XDrawString(fps_dpy, fps_win, fps_gc, 6, fps_asc + 2, s, n); XFlush(fps_dpy);
}
int main(int argc, char **argv) {
    const char *shm = "/dev/shm/nbgl", *snap = NULL; int W = 1600, H = 720, hidden = 0, present_hz = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--shm") && i + 1 < argc) shm = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &W, &H);
        else if (!strcmp(argv[i], "--hidden")) hidden = 1;
        else if (!strcmp(argv[i], "--no-fps")) fps_on = 0;
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) present_hz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--snap") && i + 1 < argc) snap = argv[++i];
        else if (!strcmp(argv[i], "-v")) g_verbose++;
        else if (!strcmp(argv[i], "--opstats")) g_opstats = 1;
        else if (!strcmp(argv[i], "--ext") && i + 1 < argc) g_extlist = argv[++i];
        else if (!strcmp(argv[i], "--spoof") && i + 1 < argc) g_spoof = argv[++i];
        else if (!strcmp(argv[i], "--noprec")) g_noprec = 1;
        else if (!strcmp(argv[i], "--caps") && i + 1 < argc) {
            for (char *t = strtok(argv[++i], ","); t && g_ncaps < 64; t = strtok(NULL, ","))
                if (sscanf(t, "%i=%i", &g_caps[g_ncaps][0], &g_caps[g_ncaps][1]) == 2) g_ncaps++;
        }
    }
    signal(SIGUSR1, on_usr1); signal(SIGTERM, on_term); signal(SIGINT, on_term);
    int fd = open(shm, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || ftruncate(fd, sizeof(NbglShm))) { perror(shm); return 1; }
    S = mmap(NULL, sizeof(NbglShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); close(fd);
    S->width = W; S->height = H; S->host_alive = 1; S_ext = S;
    __atomic_store_n(&S->magic, NBGL_MAGIC, __ATOMIC_RELEASE);

    Display *x = XOpenDisplay(NULL);
    if (!x) { fprintf(stderr, "[nbgl] no X display\n"); return 1; }
    Window win = XCreateSimpleWindow(x, DefaultRootWindow(x), 0, 0, W, H, 0, 0, 0);
    XSelectInput(x, win, ButtonPressMask | ButtonReleaseMask | Button1MotionMask | StructureNotifyMask | KeyPressMask | KeyReleaseMask);
    XkbSetDetectableAutoRepeat(x, True, NULL);
    XStoreName(x, win, "Null's Brawl on the PC (nbhost + nbgl)");
    XSizeHints sh = {.flags = PMinSize | PMaxSize, .min_width = W, .max_width = W, .min_height = H, .max_height = H};
    XSetWMNormalHints(x, win, &sh);
    Atom del = XInternAtom(x, "WM_DELETE_WINDOW", False); XSetWMProtocols(x, win, &del, 1);
    if (!hidden) { XMapWindow(x, win); fps_init(x, win); }
    XFlush(x);

    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    g_dpy = gpd(EGL_PLATFORM_X11_KHR, x, NULL);
    EGLint ma, mi; eglInitialize(g_dpy, &ma, &mi); eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ca[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
                   EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT | EGL_PBUFFER_BIT, EGL_NONE};
    EGLConfig cfg; EGLint n; eglChooseConfig(g_dpy, ca, &cfg, 1, &n);
    if (hidden) { EGLint pa[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_NONE}; g_surf = eglCreatePbufferSurface(g_dpy, cfg, pa); }   /* headless */
    else g_surf = eglCreateWindowSurface(g_dpy, cfg, (EGLNativeWindowType)win, NULL);
    EGLint xa[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(g_dpy, cfg, EGL_NO_CONTEXT, xa);
    if (!eglMakeCurrent(g_dpy, g_surf, g_surf, ctx)) { fprintf(stderr, "[nbgl] eglMakeCurrent failed\n"); return 1; }
    g_ctx_main = ctx;
    { const char *e = getenv("NBGL_ASTC_CACHE"); if (e && *e == '0') g_astc_on = 0; e = getenv("NBGL_ASTC_CHECK"); g_astc_check = e && *e == '1';
      e = getenv("NBGL_ASTC_MB"); if (e) g_astc_budget = (size_t)atol(e) << 20;
      if (g_astc_on) { EGLint pa[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE}; g_surf2 = eglCreatePbufferSurface(g_dpy, cfg, pa);
          g_ctx2 = eglCreateContext(g_dpy, cfg, EGL_NO_CONTEXT, xa); if (!g_surf2 || !g_ctx2) g_astc_on = 0; } }
    eglSwapInterval(g_dpy, 0);
    nb_load_procs(); build_ext();
    real_getiv = f_glGetIntegerv; f_glGetIntegerv = filt_getiv;
    if (g_noprec) f_glGetShaderPrecisionFormat = zero_prec;   /* the game must see the filtered extension count */
    fprintf(stderr, "[nbgl] %s, window %dx%d, shm %s (%zu MB); waiting for the game\n", glGetString(GL_RENDERER), W, H, shm, sizeof(NbglShm) >> 20);
    glClearColor(0.1f, 0.1f, 0.15f, 1); glClear(GL_COLOR_BUFFER_BIT); eglSwapBuffers(g_dpy, g_surf);

    /* WASD = a second finger on the movement joystick (pointer id 1); the mouse is finger 0 (aim, buttons) */
    int keys[4] = {0}, kfinger = 0, jx = W * 350 / 2400, jy = H * 830 / 1080, jr = H * 150 / 1080;
    int down = 0, run = 1; double t0 = now(), tlast = t0, tev = 0, tover = t0; uint64_t flast = 0, oflast = 0; long idle = 0;
    while (run) {
        /* window events -> touches (an X round trip: every 20 ms, not on every pass) */
        double tq = now(); int poll_x = tq - tev > 0.02; if (poll_x) tev = tq;
        while (poll_x && XPending(x)) {
            XEvent e; XNextEvent(x, &e);
            NbglEvent ev = {-1, 0, 0, 0};
            if (e.type == ButtonPress && e.xbutton.button == 1) { ev.action = 0; ev.x = e.xbutton.x; ev.y = e.xbutton.y; down = 1; }
            else if (e.type == ButtonRelease && e.xbutton.button == 1) { ev.action = 1; ev.x = e.xbutton.x; ev.y = e.xbutton.y; down = 0; }
            else if (e.type == MotionNotify && down) { ev.action = 2; ev.x = e.xmotion.x; ev.y = e.xmotion.y; }
            else if (e.type == ClientMessage && (Atom)e.xclient.data.l[0] == del) run = 0;
            else if (e.type == KeyPress || e.type == KeyRelease) {
                KeySym k = XLookupKeysym(&e.xkey, 0); int v = e.type == KeyPress, idx = -1;
                if (k == XK_w || k == XK_Up) idx = 0; else if (k == XK_a || k == XK_Left) idx = 1; else if (k == XK_s || k == XK_Down) idx = 2; else if (k == XK_d || k == XK_Right) idx = 3;
                if (idx >= 0 && keys[idx] != v) {
                    keys[idx] = v;
                    /* WASD / arrows: the move direction goes to the game host, which moves the local brawler (no touches) */
                    S->key_dx = keys[3] - keys[1]; S->key_dy = keys[2] - keys[0];
                }
            }
            if (ev.action >= 0 && S->ev_w - S->ev_r < NBGL_NEV) {
                S->ev[S->ev_w % NBGL_NEV] = ev; __atomic_store_n(&S->ev_w, S->ev_w + 1, __ATOMIC_RELEASE);
            }
        }
        /* GL commands */
        int did = 0;
        for (int k = 0; k < 20000; k++) {
            uint64_t r = S->r, w = __atomic_load_n(&S->w, __ATOMIC_ACQUIRE);
            if (r >= w) break;
            uint8_t *cmd = S->ring + r % NBGL_RING;
            uint32_t opf = ((uint32_t *)cmd)[0], len = ((uint32_t *)cmd)[1], op = opf & ~NBGL_SYNC;
            uint64_t R = 0;
            if (op == NBGL_OP_WRAP) { __atomic_store_n(&S->r, r + len, __ATOMIC_RELEASE); continue; }
            if (op == NBGL_OP_SWAP) {
                if (snap && (S->frames % 50) == 49) {   /* --snap: dump the frame before presenting it */
                    uint8_t *px = malloc((size_t)W * H * 4); glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
                    FILE *f = fopen(snap, "wb"); fprintf(f, "P6\n%d %d\n255\n", W, H);
                    for (int y = H - 1; y >= 0; y--) for (int xx = 0; xx < W; xx++) fwrite(px + ((size_t)y * W + xx) * 4, 1, 3, f);
                    fclose(f); free(px);
                }
                if (present_hz > 0) {   /* --fps N: present on a 1/N s grid; after a stall, no burst to catch up */
                    static double next_present = 0; double tp = now(), slot = 1.0 / present_hz;
                    if (next_present == 0) next_present = tp;
                    if (next_present > tp) { double d = next_present - tp; struct timespec ts = {(time_t)d, (long)((d - (time_t)d) * 1e9)}; nanosleep(&ts, NULL); tp = now(); }
                    next_present += slot; if (next_present < tp - slot) next_present = tp;
                }
                eglSwapBuffers(g_dpy, g_surf); S->frames++;
            } else {
                if (g_verbose > 1) fprintf(stderr, "[nbgl] %s\n", nb_op_name(op));
                double t_op = g_opstats ? now() : 0;
                if (!(op == 37 && astc_upload((const uint64_t *)(cmd + 8), cmd)))
                    R = nb_dispatch(op, (const uint64_t *)(cmd + 8), cmd, S->reply);
                const char *on = nb_op_name(op);
                if (g_opstats && op < 2048) { g_opn[op]++; g_opb[op] += len; g_opt[op] += now() - t_op;
                    if (!strcmp(on, "glCompressedTexImage2D") || !strcmp(on, "glTexImage2D") || !strcmp(on, "glCompressedTexSubImage2D") || !strcmp(on, "glTexSubImage2D")) {
                        const uint64_t *A = (const uint64_t *)(cmd + 8);
                        unsigned key = (unsigned)(on[2] == 'C' ? (on[15] == 'S' ? A[6] : A[2]) : (on[5] == 'S' ? A[6] * 65536u + A[7] : A[6] * 65536u + A[7])) ^ (on[2] == 'C' ? 0x80000000u : 0);
                        int f; for (f = 0; f < g_nfmt && g_fmt_k[f] != key; f++) {}
                        if (f == g_nfmt && g_nfmt < 64) g_fmt_k[g_nfmt++] = key;
                        if (f < 64) { g_fmt_n[f]++; g_fmt_b[f] += len; } } }
                if (on[2] == 'C' && !strcmp(on, "glCompileShader")) {
                    GLint ok = 0; GLuint sh = (GLuint)((const uint64_t *)(cmd + 8))[0]; real_getsh(sh, GL_COMPILE_STATUS, &ok); g_shaders++;
                    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "[nbgl] shader %u compile FAILED: %.600s\n", sh, log); g_shfail++; }
                } else if (on[2] == 'L' && !strcmp(on, "glLinkProgram")) {
                    GLint ok = 0; GLuint pr = (GLuint)((const uint64_t *)(cmd + 8))[0]; glGetProgramiv(pr, GL_LINK_STATUS, &ok); g_progs++;
                    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(pr, sizeof log, NULL, log); fprintf(stderr, "[nbgl] program %u link FAILED: %.600s\n", pr, log); g_linkfail++; }
                }
                if (g_verbose) { GLenum err = glGetError(); if (err) fprintf(stderr, "[nbgl] GL error %#x after %s\n", err, on); }
            }
            g_cmds++; did = 1;
            if ((opf & NBGL_SYNC) && g_verbose && op != NBGL_OP_MAPREAD && strncmp(nb_op_name(op), "glGetString", 11)) {
                const uint64_t *A = (const uint64_t *)(cmd + 8);
                fprintf(stderr, "[nbgl] sync %s(%#lx, %#lx) -> R=%#lx out0=%d\n", nb_op_name(op), (unsigned long)A[0], (unsigned long)A[1],
                        (unsigned long)R, *(int *)S->reply);
            }
            if (opf & NBGL_SYNC) { S->ret = R; __atomic_store_n(&S->r, r + len, __ATOMIC_RELEASE); __atomic_store_n(&S->rep_seq, S->rep_seq + 1, __ATOMIC_RELEASE);
                nbgl_ring(&S->rep_bell, &S->guest_sleep); }
            else __atomic_store_n(&S->r, r + len, __ATOMIC_RELEASE);
        }
        if (!did) {   /* nothing to do: a short spin, then sleep on the command bell (the game rings it; 5-ms timeout) */
            if (++idle > 32) {
                uint32_t b = __atomic_load_n(&S->w_bell, __ATOMIC_ACQUIRE);
                __atomic_store_n(&S->host_sleep, 1, __ATOMIC_SEQ_CST);
                if (__atomic_load_n(&S->w, __ATOMIC_SEQ_CST) == S->r) nbgl_futex_wait(&S->w_bell, b, 5000000);
                __atomic_store_n(&S->host_sleep, 0, __ATOMIC_RELAXED);
            } else __builtin_ia32_pause();
        }
        else idle = 0;
        double tn = now();
        if (tn - tover >= 0.2) {   /* on-screen counter: refreshed every 200 ms; fps over the last 0.5 s (sliding) */
            ov_t[ov_i] = tn; ov_f[ov_i] = S->frames; ov_i = (ov_i + 1) % OV_N; if (ov_cnt < OV_N) ov_cnt++; tover = tn;
            int best = (ov_i - ov_cnt + OV_N) % OV_N;          /* the oldest sample, unless one is 0.5 s old */
            for (int k = 1; k < ov_cnt; k++) {
                int idx = (ov_i - 1 - k + 2 * OV_N) % OV_N;
                if (tn - ov_t[idx] >= 0.5) { best = idx; break; }
            }
            if (tn - ov_t[best] > 0) fps_text((S->frames - ov_f[best]) / (tn - ov_t[best]));
        }
        if (tn - tlast > 2.0) {
            char title[160]; double fps = (S->frames - flast) / (tn - tlast);
            snprintf(title, sizeof title, "Null's Brawl on the PC (nbhost + nbgl) - %.1f fps, frame %lu", fps, (unsigned long)S->frames);
            XStoreName(x, win, title);
            fprintf(stderr, "[nbgl] t=%.0fs frames=%lu (%.1f fps) gl-commands=%ld ring=%lu MB shaders=%d (failed %d) programs=%d (failed %d)\n",
                    tn - t0, (unsigned long)S->frames, fps, g_cmds, (unsigned long)((S->w - S->r) >> 20), g_shaders, g_shfail, g_progs, g_linkfail);
            if (g_opstats) {   /* top GL ops of the last window by bytes, and the texture formats */
                fprintf(stderr, "[nbgl-ops]");
                for (int k = 0; k < 12; k++) { int best = -1; for (int o = 0; o < 2048; o++) if (g_opn[o] > 0 && (best < 0 || g_opt[o] > g_opt[best])) best = o;
                    if (best < 0) break; fprintf(stderr, " %s=%ld/%ldKB/%.0fms", nb_op_name(best), g_opn[best], g_opb[best] >> 10, g_opt[best] * 1e3); g_opn[best] = 0; }
                fprintf(stderr, "\n[nbgl-tex]");
                for (int f = 0; f < g_nfmt; f++) if (g_fmt_n[f]) fprintf(stderr, " %s:%#x=%ld/%ldKB", g_fmt_k[f] & 0x80000000u ? "comp" : "raw", g_fmt_k[f] & 0x7fffffffu, g_fmt_n[f], g_fmt_b[f] >> 10);
                fprintf(stderr, "\n"); memset(g_opn, 0, sizeof g_opn); memset(g_opb, 0, sizeof g_opb); memset(g_opt, 0, sizeof g_opt); memset(g_fmt_n, 0, sizeof g_fmt_n); memset(g_fmt_b, 0, sizeof g_fmt_b);
            }
            if (g_astc_on && (g_astc_hit || g_astc_miss)) fprintf(stderr, "[nbgl] ASTC cache: %ld hits, %ld decoded, %ld fell back, %ld evicted, %zu MB\n", g_astc_hit, g_astc_miss, g_astc_bad, g_astc_evicted, g_astc_bytes >> 20);
            tlast = tn; flast = S->frames;
        }
    }
    S->host_alive = 0;
    return 0;
}
