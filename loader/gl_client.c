/* Guest side of the GL bridge: ring writer, sync waits, hand-written GL calls, eglGetProcAddress table, touch input.
 * Enabled with  nbhost --gl /dev/shm/nbgl  (nbgl, the native renderer, must have created that file). */
#include "gl_client.h"
#include <stdlib.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>

NbglShm *g_gl;
/* window play: the WASD direction nbgl wrote (glhost/nbgl.c); 0, 0 without a window */
void gl_window_keys(int *dx, int *dy) { if (!g_gl) { *dx = *dy = 0; return; } *dx = g_gl->key_dx; *dy = g_gl->key_dy; }
static pthread_mutex_t g_glock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_w, g_seq;
static int g_unpack_align = 4, g_pack_align = 4;
uint64_t g_gl_cmds, g_gl_syncs, g_gl_bytes;

int nb_gl_attach(const char *path) {
    int fd = open(path, O_RDWR);
    if (fd < 0) { LOG("[gl] cannot open %s (start nbgl first)\n", path); return -1; }
    g_gl = mmap(NULL, sizeof(NbglShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (g_gl == MAP_FAILED || g_gl->magic != NBGL_MAGIC) { LOG("[gl] bad shm\n"); g_gl = NULL; return -1; }
    g_w = g_gl->w; g_seq = g_gl->req_seq; g_gl->guest_alive = 1;
    LOG("[gl] attached to %s: window %ux%u, ring %u MB\n", path, g_gl->width, g_gl->height, NBGL_RING >> 20);
    return 0;
}

static void pause_short(int *spins) {
    if (!g_gl->host_alive) { LOG("[gl] the window was closed; exiting\n"); _exit(0); }
    if (++*spins < 200) { sched_yield(); return; }
    struct timespec t = {0, 50000}; nanosleep(&t, NULL);
}
static int g_dbg_opn[1024];   /* expertbot round: GL command counts per opcode in a dumped frame (NBHOST_GLDBG) */
static int dbg_active(void);
void nb_begin(NbCmd *c, uint32_t op, int nargs, size_t blobs) {
    if ((op & 0x3ff) == op && dbg_active()) g_dbg_opn[op]++;
    size_t len = 8 + 8 * (size_t)nargs + blobs;
    pthread_mutex_lock(&g_glock);
    uint64_t pos = g_w % NBGL_RING;
    if (pos + len > NBGL_RING) {                       /* never straddle the end: wrap marker */
        int s = 0; while (g_w + (NBGL_RING - pos) - __atomic_load_n(&g_gl->r, __ATOMIC_ACQUIRE) > NBGL_RING) pause_short(&s);
        *(uint32_t *)(g_gl->ring + pos) = NBGL_OP_WRAP; *(uint32_t *)(g_gl->ring + pos + 4) = NBGL_RING - pos;
        g_w += NBGL_RING - pos; __atomic_store_n(&g_gl->w, g_w, __ATOMIC_RELEASE); nbgl_ring(&g_gl->w_bell, &g_gl->host_sleep); pos = 0;
    }
    int s = 0; while (g_w + len - __atomic_load_n(&g_gl->r, __ATOMIC_ACQUIRE) > NBGL_RING) pause_short(&s);   /* back-pressure */
    c->base = g_gl->ring + pos; c->a = (uint64_t *)(c->base + 8); c->blob = 8 + 8 * (size_t)nargs; c->len = len;
    c->nout = 0; c->roff = 0;
    ((uint32_t *)c->base)[0] = op; ((uint32_t *)c->base)[1] = (uint32_t)len;
}
void nb_in(NbCmd *c, int i, const void *p, size_t n) {
    if (!p) { c->a[i] = NBGL_NULL; return; }
    memcpy(c->base + c->blob, p, n); c->a[i] = c->blob; c->blob += NB_A8(n);
}
void nb_out(NbCmd *c, int i, void *p, size_t n) {
    if (!p) { c->a[i] = NBGL_NULL; return; }
    c->a[i] = c->roff; c->out[c->nout].dst = p; c->out[c->nout].n = n; c->out[c->nout].off = c->roff; c->nout++; c->roff += NB_A8(n);
}
void nb_end(NbCmd *c) {
    g_w += c->len; g_gl_cmds++; g_gl_bytes += c->len;
    __atomic_store_n(&g_gl->w, g_w, __ATOMIC_RELEASE);
    nbgl_ring(&g_gl->w_bell, &g_gl->host_sleep);
    pthread_mutex_unlock(&g_glock);
}
uint64_t nb_sync(NbCmd *c) {
    uint64_t want = ++g_seq;
    g_w += c->len; g_gl_cmds++; g_gl_syncs++; g_gl_bytes += c->len;
    __atomic_store_n(&g_gl->req_seq, want, __ATOMIC_RELEASE);
    __atomic_store_n(&g_gl->w, g_w, __ATOMIC_RELEASE);
    nbgl_ring(&g_gl->w_bell, &g_gl->host_sleep);
    /* the reply: a short spin, then sleep on the reply bell (nbgl rings it when guest_sleep is set; 2-ms timeout) */
    for (int s = 0; __atomic_load_n(&g_gl->rep_seq, __ATOMIC_ACQUIRE) < want; s++) {
        if (s < 64) continue;
        if (!g_gl->host_alive) { LOG("[gl] the window was closed; exiting\n"); _exit(0); }
        uint32_t b = __atomic_load_n(&g_gl->rep_bell, __ATOMIC_ACQUIRE);
        __atomic_store_n(&g_gl->guest_sleep, 1, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&g_gl->rep_seq, __ATOMIC_SEQ_CST) < want) nbgl_futex_wait(&g_gl->rep_bell, b, 2000000);
        __atomic_store_n(&g_gl->guest_sleep, 0, __ATOMIC_RELAXED);
    }
    for (int i = 0; i < c->nout; i++) memcpy(c->out[i].dst, g_gl->reply + c->out[i].off, c->out[i].n);
    uint64_t r = g_gl->ret;
    pthread_mutex_unlock(&g_glock);
    return r;
}

static int ncomp(GLenum f) {
    switch (f) {
    case 0x1903: case 0x8D94: case 0x1906: case 0x1909: case 0x1902: case 0x1901: return 1;  /* RED RED_INT ALPHA LUM DEPTH STENCIL_INDEX */
    case 0x8227: case 0x8228: case 0x190A: case 0x84F9: return 2;                        /* RG RG_INT LUM_ALPHA DEPTH_STENCIL */
    case 0x1907: case 0x8D98: return 3;                                                  /* RGB RGB_INT */
    }
    return 4;
}
size_t nb_pix(int w, int h, int d, GLenum format, GLenum type, int pack) {
    int bpp;
    switch (type) {
    case 0x8363: case 0x8033: case 0x8034: return (size_t)w * h * d * 2;                  /* packed 16-bit */
    case 0x8368: case 0x8C3B: case 0x8C3E: case 0x84FA: bpp = 4; break;                   /* packed 32-bit */
    case 0x8DAD: bpp = 8; break;
    case 0x1406: case 0x1404: case 0x1405: bpp = 4 * ncomp(format); break;                /* FLOAT INT UINT */
    case 0x140B: case 0x8D61: case 0x1402: case 0x1403: bpp = 2 * ncomp(format); break;   /* HALF SHORT USHORT */
    default: bpp = ncomp(format);
    }
    int al = pack ? g_pack_align : g_unpack_align;
    size_t row = ((size_t)w * bpp + al - 1) / al * al;
    if (w <= 0 || h <= 0 || d <= 0) return 0;
    return row * ((size_t)h * d - 1) + (size_t)w * bpp;
}
size_t nb_idxsz(GLenum type) { return type == 0x1401 ? 1 : type == 0x1403 ? 2 : 4; }
void nb_check_ofs(const char *fn, const void *p) {
    static int warned;
    if ((uintptr_t)p > 0xffffffffULL && warned++ < 5) LOG("[gl] %s with a client-side array (%p): not forwarded\n", fn, p);
}
int nb_get_count(GLenum p) {
    GLint n = 0;
    if (p == 0x86A3) { glGetIntegerv(0x86A2, &n); return n; }   /* COMPRESSED_TEXTURE_FORMATS */
    if (p == 0x87FF) { glGetIntegerv(0x87FE, &n); return n; }   /* PROGRAM_BINARY_FORMATS */
    if (p == 0x8DF8) { glGetIntegerv(0x8DF9, &n); return n; }   /* SHADER_BINARY_FORMATS */
    return nbgl_pname_count(p);
}
int nb_block_count(GLuint prog, GLuint idx, GLenum pname) {
    if (pname != 0x8A43) return 1;   /* UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: count = UNIFORM_BLOCK_ACTIVE_UNIFORMS */
    GLint n = 0; glGetActiveUniformBlockiv(prog, idx, 0x8A42, &n); return n;
}

/* ---- hand-written calls (opcodes = position in nb_procs + 16, the host handles them in nb_special) ---- */
static uint32_t opcode(const char *n) { for (int i = 0; nb_procs[i].name; i++) if (!strcmp(nb_procs[i].name, n)) return 16 + i; abort(); }
GL_APICALL GLenum GL_APIENTRY glGetError(void) { return GL_NO_ERROR; }
GL_APICALL void GL_APIENTRY glPixelStorei(GLenum pname, GLint param) {
    static uint32_t op; if (!op) op = opcode("glPixelStorei");
    if (pname == GL_UNPACK_ALIGNMENT) g_unpack_align = param;
    if (pname == GL_PACK_ALIGNMENT) g_pack_align = param;
    if ((pname == GL_UNPACK_ROW_LENGTH || pname == GL_UNPACK_SKIP_ROWS || pname == GL_UNPACK_SKIP_PIXELS) && param)
        LOG("[gl] glPixelStorei(%#x, %d): unpack sub-rectangles are not sized correctly\n", pname, param);
    NbCmd c; nb_begin(&c, op, 2, 0); c.a[0] = pname; c.a[1] = (uint64_t)(int64_t)param; nb_end(&c);
}
static const char *str_cache(GLenum name, GLuint idx, int indexed) {
    static struct { GLenum n; GLuint i; int ix; char *s; } cache[512]; static int nc;
    for (int k = 0; k < nc; k++) if (cache[k].n == name && cache[k].i == idx && cache[k].ix == indexed) return cache[k].s;
    static uint32_t op1, op2; if (!op1) { op1 = opcode("glGetString"); op2 = opcode("glGetStringi"); }
    char *buf = malloc(65536);
    NbCmd c; nb_begin(&c, (indexed ? op2 : op1) | NBGL_SYNC, 3, 0); c.a[0] = name; c.a[1] = idx; nb_out(&c, 2, buf, 65536);
    uint64_t n = nb_sync(&c);
    if (n == 0) { free(buf); return NULL; }
    if (nc < 512) { cache[nc].n = name; cache[nc].i = idx; cache[nc].ix = indexed; cache[nc].s = buf; nc++; }
    return buf;
}
GL_APICALL const GLubyte *GL_APIENTRY glGetString(GLenum name) { return (const GLubyte *)str_cache(name, 0, 0); }
GL_APICALL const GLubyte *GL_APIENTRY glGetStringi(GLenum name, GLuint index) { return (const GLubyte *)str_cache(name, index, 1); }
GL_APICALL void GL_APIENTRY glShaderSource(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length) {
    size_t tot = 1;
    for (int i = 0; i < count; i++) tot += length && length[i] >= 0 ? (size_t)length[i] : strlen(string[i]);
    char *s = malloc(tot), *p = s;
    for (int i = 0; i < count; i++) { size_t n = length && length[i] >= 0 ? (size_t)length[i] : strlen(string[i]); memcpy(p, string[i], n); p += n; }
    *p = 0;
    static uint32_t op; if (!op) op = opcode("glShaderSource");
    NbCmd c; nb_begin(&c, op, 4, NB_A8(tot)); c.a[0] = shader; c.a[1] = 1; nb_in(&c, 2, s, tot); c.a[3] = NBGL_NULL; nb_end(&c);
    free(s);
}
static size_t pack_strs(GLsizei n, const GLchar *const *v, char **out) {
    size_t tot = 0; for (int i = 0; i < n; i++) tot += strlen(v[i]) + 1;
    char *s = malloc(tot + 1), *p = s; for (int i = 0; i < n; i++) { size_t k = strlen(v[i]) + 1; memcpy(p, v[i], k); p += k; }
    *out = s; return tot;
}
GL_APICALL void GL_APIENTRY glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar *const *varyings, GLenum mode) {
    char *s; size_t n = pack_strs(count, varyings, &s);
    static uint32_t op; if (!op) op = opcode("glTransformFeedbackVaryings");
    NbCmd c; nb_begin(&c, op, 4, NB_A8(n)); c.a[0] = program; c.a[1] = count; nb_in(&c, 2, s, n); c.a[3] = mode; nb_end(&c); free(s);
}
GL_APICALL void GL_APIENTRY glGetUniformIndices(GLuint program, GLsizei count, const GLchar *const *names, GLuint *indices) {
    char *s; size_t n = pack_strs(count, names, &s);
    static uint32_t op; if (!op) op = opcode("glGetUniformIndices");
    NbCmd c; nb_begin(&c, op | NBGL_SYNC, 4, NB_A8(n)); c.a[0] = program; c.a[1] = count; nb_in(&c, 2, s, n); nb_out(&c, 3, indices, count * 4);
    nb_sync(&c); free(s);
}
/* buffer mapping: a guest-side shadow, written back with glBufferSubData on flush/unmap */
static struct { GLenum target; GLintptr off; GLsizeiptr len; GLbitfield access; uint8_t *p; } g_map[16];
static int map_slot(GLenum t) { int k = t & 15; return k; }
GL_APICALL void *GL_APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    int k = map_slot(target);
    uint8_t *p = malloc(length ? length : 1);
    if (access & GL_MAP_READ_BIT) {
        NbCmd c; nb_begin(&c, NBGL_OP_MAPREAD | NBGL_SYNC, 4, 0); c.a[0] = target; c.a[1] = offset; c.a[2] = length; nb_out(&c, 3, p, length);
        nb_sync(&c);
    }
    g_map[k].target = target; g_map[k].off = offset; g_map[k].len = length; g_map[k].access = access; g_map[k].p = p;
    return p;
}
GL_APICALL void GL_APIENTRY glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
    int k = map_slot(target); if (g_map[k].p) glBufferSubData(target, g_map[k].off + offset, length, g_map[k].p + offset);
}
GL_APICALL GLboolean GL_APIENTRY glUnmapBuffer(GLenum target) {
    int k = map_slot(target);
    if (!g_map[k].p) return GL_FALSE;
    if ((g_map[k].access & GL_MAP_WRITE_BIT) && !(g_map[k].access & GL_MAP_FLUSH_EXPLICIT_BIT))
        glBufferSubData(target, g_map[k].off, g_map[k].len, g_map[k].p);
    free(g_map[k].p); g_map[k].p = NULL;
    return GL_TRUE;
}
GL_APICALL void GL_APIENTRY glGetBufferPointerv(GLenum target, GLenum pname, void **params) { (void)pname; *params = g_map[map_slot(target)].p; }

int nb_gl_capture(void *dst, int w, int h) {   /* GPU downscale + readback of the current frame (nbgl NBGL_OP_CAPTURE) */
    if (!g_gl) return -1;
    NbCmd c; nb_begin(&c, NBGL_OP_CAPTURE | NBGL_SYNC, 3, 0); c.a[0] = w; c.a[1] = h; nb_out(&c, 2, dst, (size_t)w * h * 4);
    return (int)nb_sync(&c);
}
int nb_gl_capture_obs(void *dst, int w, int h) {   /* policy observation (nbgl: 300x135 capture + PIL-exact BOX to w x h, RGB top-left) */
    if (!g_gl) return -1;
    NbCmd c; nb_begin(&c, NBGL_OP_CAPTURE | NBGL_SYNC, 3, 0); c.a[0] = (uint64_t)(int64_t)-w; c.a[1] = (uint64_t)(int64_t)-h; nb_out(&c, 2, dst, (size_t)w * h * 3);
    return (int)nb_sync(&c);
}
/* expertbot round: camera-matrix capture. NBHOST_GLDBG=N: every N-th frame, log the distinct 4x4 matrix uniforms set
 * during that frame (program, location, how many draws used the program, the matrix) with the env's current tick. */
int g_dbg_tick = -1;
static int g_dbg_on = -1, g_dbg_every = 30; static unsigned g_dbg_prog; static long g_dbg_frame;
typedef struct { unsigned prog; int loc, tr, n; float m[16]; } DbgMat;
static DbgMat g_dbg_m[256]; static int g_dbg_nm; static int g_dbg_draws[4096];
static int dbg_active(void) {
    if (g_dbg_on < 0) { const char *e = getenv("NBHOST_GLDBG"); g_dbg_on = e ? 1 : 0; if (e && atoi(e) > 0) g_dbg_every = atoi(e); }
    return g_dbg_on && (g_dbg_frame % g_dbg_every) == 0;
}
void nb_dbg_use(unsigned program) { g_dbg_prog = program; }
static unsigned g_dbg_ubo, g_dbg_arr;
void nb_dbg_bind(unsigned target, unsigned buffer) { if (target == 0x8A11) g_dbg_ubo = buffer; else if (target == 0x8892) g_dbg_arr = buffer; }
/* expertbot round: the world camera matrix = a 64-byte upload whose w row depends on Y and Z (perspective; the UI
 * matrix has zeros there); kept for env.c to publish with every response (seqlock: odd = writing) */
volatile float g_cam_mat[16]; volatile int g_cam_mat_seq, g_cam_mat_n;
void nb_dbg_bsd(unsigned target, long offset, long size, const void *data) {
    if (data && size == 64 && offset == 0) {
        const float *m = data;
        if ((m[7] > 1e-3f || m[7] < -1e-3f) && (m[11] > 1e-3f || m[11] < -1e-3f) && m[15] != 0.0f) {
            __atomic_add_fetch(&g_cam_mat_seq, 1, __ATOMIC_ACQ_REL);
            for (int k = 0; k < 16; k++) g_cam_mat[k] = m[k];
            __atomic_add_fetch(&g_cam_mat_seq, 1, __ATOMIC_ACQ_REL); g_cam_mat_n++;
        }
    }
    if (!dbg_active() || !data) return;
    unsigned buf = target == 0x8A11 ? g_dbg_ubo : (target == 0x8892 ? g_dbg_arr : 0);
    LOG("[glbsd] frame %ld tick %d target %#x buf %u off %ld size %ld\n", g_dbg_frame, g_dbg_tick, target, buf, offset, size);
    if (size > 256) return;   /* small uploads: matrices / per-frame constants (any target: the game uploads via ARRAY_BUFFER) */
    const float *f = data; long nf = size / 4;
    for (long k = 0; k < nf; k += 16) {
        char line[512]; int n = snprintf(line, sizeof line, "[glubo] frame %ld buf %u off %ld f%ld", g_dbg_frame, buf, offset, k);
        for (long j = k; j < k + 16 && j < nf; j++) n += snprintf(line + n, sizeof line - n, " %.7g", f[j]);
        LOG("%s\n", line);
    }
}
void nb_dbg_draw(int count) { (void)count; if (dbg_active() && g_dbg_prog < 4096) g_dbg_draws[g_dbg_prog]++; }
void nb_dbg_mat(int location, int count, unsigned char transpose, const float *value) {
    if (!dbg_active() || !value || count < 1) return;
    for (int i = 0; i < g_dbg_nm; i++)
        if (g_dbg_m[i].prog == g_dbg_prog && g_dbg_m[i].loc == location && !memcmp(g_dbg_m[i].m, value, 64)) { g_dbg_m[i].n++; return; }
    if (g_dbg_nm < 256) { DbgMat *d = &g_dbg_m[g_dbg_nm++]; d->prog = g_dbg_prog; d->loc = location; d->tr = transpose; d->n = 1; memcpy(d->m, value, 64); }
}
static void dbg_flush(void) {
    if (dbg_active()) {
        for (int i = 0; i < g_dbg_nm; i++) {
            DbgMat *d = &g_dbg_m[i]; char buf[640]; int n = snprintf(buf, sizeof buf, "[gldbg] frame %ld tick %d prog %u loc %d tr %d n %d draws %d m",
                g_dbg_frame, g_dbg_tick, d->prog, d->loc, d->tr, d->n, d->prog < 4096 ? g_dbg_draws[d->prog] : -1);
            for (int k = 0; k < 16; k++) n += snprintf(buf + n, sizeof buf - n, " %.7g", d->m[k]);
            LOG("%s\n", buf);
        }
    }
    if (dbg_active()) {
        char buf[2048]; int n = snprintf(buf, sizeof buf, "[glops] frame %ld tick %d", g_dbg_frame, g_dbg_tick);
        for (int k = 0; k < 1024 && n < (int)sizeof buf - 20; k++) if (g_dbg_opn[k]) n += snprintf(buf + n, sizeof buf - n, " %d:%d", k, g_dbg_opn[k]);
        LOG("%s\n", buf);
    }
    memset(g_dbg_opn, 0, sizeof g_dbg_opn);
    g_dbg_nm = 0; memset(g_dbg_draws, 0, sizeof g_dbg_draws); g_dbg_frame++;
}
void nb_gl_swap(void) {
    dbg_flush();
    NbCmd c; nb_begin(&c, NBGL_OP_SWAP, 0, 0); nb_end(&c);
}
void *nb_gl_proc(const char *name) {
    for (int i = 0; nb_procs[i].name; i++) if (!strcmp(nb_procs[i].name, name)) return nb_procs[i].fn;
    return NULL;
}
void nb_gl_stats(void) {
    if (g_gl) LOG("[gl] commands %lu (sync %lu), %.1f MB sent, host frames %lu\n", g_gl_cmds, g_gl_syncs, g_gl_bytes / 1e6, g_gl->frames);
}

/* touch input from the window -> GameApp.nOnTouchEvent(action, x, y, pointerId) on our own thread (Java's UI thread role) */
static void *input_thread(void *fn) {
    void (*touch)(void *, void *, int, int, int, int) = fn;
    void *thiz = jni_new_object("com/supercell/titan/GameApp");
    for (;;) {
        while (g_gl->ev_r < __atomic_load_n(&g_gl->ev_w, __ATOMIC_ACQUIRE)) {
            NbglEvent e = g_gl->ev[g_gl->ev_r % NBGL_NEV];
            __atomic_store_n(&g_gl->ev_r, g_gl->ev_r + 1, __ATOMIC_RELEASE);
            if (e.action != 2) LOG("[input] touch action %d at (%d, %d)\n", e.action, e.x, e.y);
            touch(jni_env(), thiz, e.action, e.x, e.y, e.id);
        }
        struct timespec t = {0, 2000000}; nanosleep(&t, NULL);
    }
    return NULL;
}
void nb_gl_start_input(void *touch_fn) {
    pthread_t th; pthread_create(&th, NULL, input_thread, touch_fn); pthread_detach(th);
}
