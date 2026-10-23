/* libandroid (AAsset*, ALooper*, ANativeWindow*, ASensor*), liblog, EGL (+ a null GLES behind eglGetProcAddress),
 * FMOD (silent), Sentry (no-op), ZSTD trace hooks. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

/* ---------------- liblog ---------------- */
int g_log_android = 1;
static int s_log_write(int prio, const char *tag, const char *msg) {
    if (g_log_android) LOG("[alog %d %s] %s\n", prio, tag ? tag : "-", msg ? msg : "");
    return 1;
}
static int s_log_print(int prio, const char *tag, const char *fmt, ...) {
    char buf[2048]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    return s_log_write(prio, tag, buf);
}

/* ---------------- AAssetManager: serves files from <asset_root>/assets/<name> ---------------- */
const char *g_asset_root = ".";
typedef struct { int fd; off_t len; char name[256]; long nread; int trace; } AAsset;
static int g_asset_opens, g_asset_missing;
static void *s_amgr_from_java(void *env, void *obj) { (void)env; LOG("[asset] AAssetManager_fromJava(%s)\n", jni_obj_desc(obj)); return (void *)0xa55e7; }
static AAsset *s_amgr_open(void *mgr, const char *name, int mode) {
    (void)mgr; (void)mode;
    char p[1024]; snprintf(p, sizeof p, "%s/assets/%s", g_asset_root, name);
    int fd = open(p, O_RDONLY);
    if (fd < 0) { g_asset_missing++; LOG("[asset] MISSING %s\n", name); return NULL; }
    AAsset *a = calloc(1, sizeof *a); a->fd = fd; struct stat st; fstat(fd, &st); a->len = st.st_size;
    snprintf(a->name, sizeof a->name, "%s", name);
    g_asset_opens++; VLOG("[asset] open %s (%ld bytes)\n", name, (long)a->len);
    { static const char *tr; static int init; if (!init) { init = 1; tr = getenv("NBHOST_TRACE_ASSET"); }   /* debug: who opens it */
      if (tr && strstr(name, tr)) { a->trace = 1; uint64_t *fp = __builtin_frame_address(0);
        LOG("[asset-trace] open %s tid %d:", name, (int)gettid());
        for (int i = 0; i < 12 && fp && ((uintptr_t)fp & 7) == 0; i++) { LOG(" %s", addr_desc((void *)fp[1])); uint64_t *n = (uint64_t *)fp[0]; if (n <= fp) break; fp = n; }
        LOG("\n"); } }
    return a;
}
static int s_asset_read(AAsset *a, void *b, size_t n) { int r = read(a->fd, b, n); if (r > 0) a->nread += r; return r; }
static off_t s_asset_seek(AAsset *a, off_t o, int w) { return lseek(a->fd, o, w); }
static off_t s_asset_len(AAsset *a) { return a->len; }
static off_t s_asset_remaining(AAsset *a) { return a->len - lseek(a->fd, 0, SEEK_CUR); }
static void s_asset_close(AAsset *a) { if (a) { if (a->trace) LOG("[asset-trace] close %s after %ld bytes read\n", a->name, a->nread); close(a->fd); free(a); } }
static int s_asset_fd(AAsset *a, off_t *start, off_t *len) { *start = 0; *len = a->len; return dup(a->fd); }

/* ---------------- ALooper / ANativeWindow / ASensor ---------------- */
static void *s_looper_prepare(int o) { (void)o; return (void *)0x100e7; }
static void *s_looper_for_thread(void) { return (void *)0x100e7; }
static void *s_window_from_surface(void *env, void *surf) { (void)env; LOG("[window] ANativeWindow_fromSurface(%s)\n", jni_obj_desc(surf)); return (void *)0x3170; }
static void s_window_release(void *w) { (void)w; }
static void *s_null(void) { return NULL; }
static int s_zero(void) { return 0; }

/* ---------------- EGL ---------------- */
#define EGL_TRUE 1
static int egl_attr(int a) {
    switch (a) {
    case 0x3024: case 0x3023: case 0x3022: return 8;   /* RED/GREEN/BLUE */
    case 0x3021: return 8; case 0x3025: return 24; case 0x3026: return 8;  /* ALPHA DEPTH STENCIL */
    case 0x3028: return 1;  /* CONFIG_ID */ case 0x3031: return 0; case 0x3032: return 0; /* SAMPLES BUFFERS */
    case 0x3040: return 0x44; /* RENDERABLE_TYPE: ES2|ES3 */ case 0x3033: return 4; /* SURFACE_TYPE window */
    case 0x302e: return 1; /* NATIVE_VISUAL_ID */
    }
    return 0;
}
static int s_egl_get_config_attrib(void *d, void *c, int a, int *v) { (void)d; (void)c; *v = egl_attr(a); return EGL_TRUE; }
static int s_egl_choose_config(void *d, const int *attr, void **cfg, int n, int *num) {
    (void)d; (void)attr; if (cfg && n > 0) cfg[0] = (void *)0xc0f; *num = 1; return EGL_TRUE;
}
static int s_egl_get_configs(void *d, void **cfg, int n, int *num) { return s_egl_choose_config(d, NULL, cfg, n, num); }
static void *s_egl_get_display(void *n) { (void)n; return (void *)0xd15; }
static int s_egl_initialize(void *d, int *ma, int *mi) { (void)d; if (ma) *ma = 1; if (mi) *mi = 4; LOG("[egl] eglInitialize\n"); return EGL_TRUE; }
static void *g_ctx;
static void *s_egl_create_context(void *d, void *c, void *share, const int *a) { (void)d; (void)c; (void)share; (void)a; LOG("[egl] eglCreateContext\n"); return (void *)0xc7; }
static void *s_egl_create_window_surface(void *d, void *c, void *w, const int *a) { (void)d; (void)c; (void)w; (void)a; LOG("[egl] eglCreateWindowSurface\n"); return (void *)0x5f; }
static int s_egl_make_current(void *d, void *dr, void *rd, void *c) { (void)d; (void)dr; (void)rd; g_ctx = c; return EGL_TRUE; }
static int g_swaps;
int g_swaps_count(void) { return g_swaps; }
extern struct NbglShm *g_gl; extern void nb_gl_swap(void); extern void *nb_gl_proc(const char *);
extern void env_on_swap(void);
static int s_egl_swap(void *d, void *s) { (void)d; (void)s; g_swaps++; if (g_gl) { env_on_swap(); nb_gl_swap(); } return EGL_TRUE; }
static int s_egl_true(void) { return EGL_TRUE; }
static int s_egl_get_error(void) { return 0x3000; }
static void *s_egl_get_current_context(void) { return g_ctx; }

/* null GLES: every gl* name gets a logging thunk returning 0; the ones that hand back data are implemented */
static unsigned g_glname = 1;
static void glGenX(int n, unsigned *ids) { for (int i = 0; i < n; i++) ids[i] = g_glname++; }
static int g_ncreate;
static unsigned glCreateX(void) { if (g_ncreate++ < 3) LOG("[gl-null] glCreateShader/Program #%d\n", g_ncreate); return g_glname++; }
/* the null renderer claims the phone's texture formats so the game uploads compressed textures as they are
 * instead of decoding ASTC in software (expensive under emulation, and pointless when nothing is drawn) */
static const char *g_null_ext[] = {"GL_KHR_texture_compression_astc_ldr", "GL_OES_compressed_ETC1_RGB8_texture", "GL_EXT_texture_format_BGRA8888"};
static const char *glGetStringi_(unsigned e, unsigned i) { return e == 0x1F03 && i < 3 ? g_null_ext[i] : NULL; }
static const char *glGetString_(unsigned e) {
    switch (e) {
    case 0x1F00: return "nbhost"; case 0x1F01: return "null renderer"; case 0x1F02: return "OpenGL ES 3.2 nbhost";
    case 0x8B8C: return "OpenGL ES GLSL ES 3.20";
    case 0x1F03: return "GL_OES_compressed_ETC1_RGB8_texture GL_KHR_texture_compression_astc_ldr GL_EXT_texture_format_BGRA8888";
    }
    return "";
}
static void glGetIntegerv_(unsigned e, int *v) {
    switch (e) {
    case 0x86A3: case 0x87FF: case 0x8DF8: return;     /* *_FORMATS arrays: their NUM_* is 0 here, the buffer may be empty */
    case 0x0BA2: case 0x0C10: v[0] = v[1] = 0; v[2] = 2400; v[3] = 1080; return;  /* VIEWPORT / SCISSOR_BOX: 4 values */
    case 0x0D3A: v[0] = v[1] = 8192; return;          /* MAX_VIEWPORT_DIMS: 2 values */
    case 0x0D33: *v = 8192; return;            /* MAX_TEXTURE_SIZE */
    case 0x8869: *v = 16; return; case 0x8B4D: *v = 32; return; case 0x8872: *v = 16; return;
    case 0x8DFB: *v = 1024; return; case 0x8DFD: *v = 1024; return; case 0x8DFC: *v = 32; return;
    case 0x821B: *v = 3; return; case 0x821C: *v = 2; return;  /* MAJOR/MINOR VERSION */
    case 0x821D: *v = 3; return;                               /* NUM_EXTENSIONS (see g_null_ext) */
    case 0x86A2: *v = 0; return; case 0x8D57: *v = 4; return; case 0x84E8: *v = 8192; return;
    }
    *v = 0;
}
static void glGetShaderiv_(unsigned s, unsigned p, int *v) { (void)s; *v = (p == 0x8B81 /*COMPILE_STATUS*/) ? 1 : 0; }
static void glGetProgramiv_(unsigned s, unsigned p, int *v) { (void)s; *v = (p == 0x8B82 /*LINK_STATUS*/ || p == 0x8B83) ? 1 : 0; }
static unsigned glCheckFramebufferStatus_(unsigned t) { (void)t; return 0x8CD5; }
static int glGetUniformLocation_(unsigned p, const char *n) { (void)p; (void)n; return 1; }
static int glGetAttribLocation_(unsigned p, const char *n) { (void)p; (void)n; return 0; }
/* fences/queries/mapping behave like a driver whose GPU finishes instantly */
static void *glFenceSync_(unsigned c, unsigned f) { (void)c; (void)f; return (void *)0x5ec; }
static unsigned glClientWaitSync_(void *s, unsigned f, uint64_t t) { (void)s; (void)f; (void)t; return 0x911A; }   /* ALREADY_SIGNALED */
static void glGetSynciv_(void *s, unsigned p, int n, int *len, int *v) { (void)s; if (n > 0 && v) v[0] = p == 0x9114 ? 0x9119 : 0; if (len) *len = 1; }
static void glGetQueryObjectuiv_(unsigned id, unsigned p, unsigned *v) { (void)id; *v = p == 0x8867 ? 1 : 0; }   /* RESULT_AVAILABLE */
static void *g_map[16]; static size_t g_mapn[16];
static void *glMapBufferRange_(unsigned t, long off, long len, unsigned acc) {
    (void)off; (void)acc; int k = t & 15; if ((size_t)len > g_mapn[k]) { free(g_map[k]); g_map[k] = calloc(1, len); g_mapn[k] = len; } return g_map[k];
}
static unsigned char glUnmapBuffer_(unsigned t) { (void)t; return 1; }
static unsigned glCheck_(void) { return 0; }
static int g_gl_calls;
typedef struct { char name[60]; int calls; } GlName;
static GlName *gl_name_new(const char *n) { GlName *g = calloc(1, sizeof *g); snprintf(g->name, sizeof g->name, "%s", n); return g; }
static uint64_t gl_null(void *name, uint64_t *x, double *d, void *lr) {
    (void)d; (void)lr; g_gl_calls++;
    GlName *g = name;
    if (g->calls++ == 0) LOG("[gl-first] %s(%#lx, %#lx, %#lx) from %s\n", g->name, x[0], x[1], x[2], addr_desc(lr));
    return 0;
}
static void *s_egl_get_proc(const char *name) {
    if (g_gl) { void *p = nb_gl_proc(name); if (p) return p; }
    static const Shim gl[] = {
        {"glGenTextures", glGenX}, {"glGenBuffers", glGenX}, {"glGenFramebuffers", glGenX}, {"glGenRenderbuffers", glGenX},
        {"glGenVertexArrays", glGenX}, {"glGenVertexArraysOES", glGenX}, {"glGenQueries", glGenX},
        {"glCreateShader", glCreateX}, {"glCreateProgram", glCreateX}, {"glGetString", glGetString_},
        {"glGetIntegerv", glGetIntegerv_}, {"glGetShaderiv", glGetShaderiv_}, {"glGetProgramiv", glGetProgramiv_},
        {"glCheckFramebufferStatus", glCheckFramebufferStatus_}, {"glGetUniformLocation", glGetUniformLocation_},
        {"glGetAttribLocation", glGetAttribLocation_}, {"glGetStringi", glGetStringi_}, {"glFenceSync", glFenceSync_},
        {"glClientWaitSync", glClientWaitSync_}, {"glGetSynciv", glGetSynciv_}, {"glGetQueryObjectuiv", glGetQueryObjectuiv_},
        {"glMapBufferRange", glMapBufferRange_}, {"glUnmapBuffer", glUnmapBuffer_}, {"glGetError", glCheck_}, {0, 0}};
    for (const Shim *s = gl; s->name; s++) if (!strcmp(s->name, name)) return s->addr;
    VLOG("[egl] eglGetProcAddress(%s) -> null thunk\n", name);
    static int nproc; LOG("[egl-proc] #%d %s\n", ++nproc, name);
    return make_thunk(gl_null, gl_name_new(name));
}

/* ---------------- FMOD: everything succeeds silently ----------------
 * Every System/Sound/Channel/ChannelGroup gets its OWN small handle (the game keys maps by these pointers), and
 * set/getUserData round-trips per handle. Handles are never freed (sounds are few). */
typedef struct { uint64_t magic; void *user; uint64_t pad[6]; } FmodObj;
static FmodObj *fobj(void) { FmodObj *o = calloc(1, sizeof *o); o->magic = 0xf30d; return o; }
static int f_create(void **sys, unsigned ver) { (void)ver; *sys = fobj(); return 0; }
static int f_ok(void) { return 0; }
static int f_get_user(FmodObj *self, void **out) { if (out) *out = self ? self->user : NULL; return 0; }
static int f_set_user(FmodObj *self, void *u) { if (self) self->user = u; return 0; }
static int f_bool_out(void *self, char *b) { (void)self; if (b) *b = 0; return 0; }
static int f_uint_out1(void *self, unsigned *v) { (void)self; if (v) *v = 0x00020000; return 0; }               /* getVersion */
static int f_len(void *self, unsigned *v, unsigned u) { (void)self; (void)u; if (v) *v = 1000; return 0; }
static int f_int_out1(void *self, int *v) { (void)self; if (v) *v = 1; return 0; }                               /* getNumDrivers, getIndex */
static int f_float_out1(void *self, float *v) { (void)self; if (v) *v = 44100.f; return 0; }
static int f_create_sound(void *self, const char *n, unsigned m, void *ex, void **snd) { (void)self; (void)n; (void)m; (void)ex; *snd = fobj(); return 0; }
static int f_play(void *self, void *s, void *g, int p, void **ch) { (void)self; (void)s; (void)g; (void)p; if (ch) *ch = fobj(); return 0; }
static int f_get_channel(void *self, int i, void **ch) { (void)self; (void)i; *ch = fobj(); return 0; }
static int f_create_group(void *self, const char *n, void **g) { (void)self; (void)n; *g = fobj(); return 0; }
static int f_sw_format(void *self, int *rate, int *mode, int *raw) { (void)self; if (rate) *rate = 48000; if (mode) *mode = 3; if (raw) *raw = 2; return 0; }
static int f_driver_info(void *self, int id, char *name, int nl, void *guid, int *rate, int *mode, int *ch) {
    (void)self; (void)id; (void)guid; if (name && nl) name[0] = 0; if (rate) *rate = 48000; if (mode) *mode = 3; if (ch) *ch = 2; return 0;
}

/* ---------------- Sentry: no-op (sentry_value_t is a 64-bit union, 0 = null) ---------------- */
static uint64_t sentry0(void) { return 0; }

#define S(n, f) { n, (void *)f }
static const Shim ashims[] = {
    S("__android_log_print", s_log_print), S("__android_log_write", s_log_write),
    S("AAssetManager_fromJava", s_amgr_from_java), S("AAssetManager_open", s_amgr_open), S("AAsset_read", s_asset_read),
    S("AAsset_seek64", s_asset_seek), S("AAsset_getLength", s_asset_len), S("AAsset_getLength64", s_asset_len),
    S("AAsset_getRemainingLength64", s_asset_remaining), S("AAsset_close", s_asset_close),
    S("AAsset_openFileDescriptor64", s_asset_fd),
    S("ALooper_prepare", s_looper_prepare), S("ALooper_forThread", s_looper_for_thread),
    S("ANativeWindow_fromSurface", s_window_from_surface), S("ANativeWindow_release", s_window_release),
    S("ASensorManager_getInstance", s_null), S("ASensorManager_getDefaultSensor", s_null),
    S("ASensorManager_createEventQueue", s_null), S("ASensorManager_destroyEventQueue", s_zero),
    S("ASensorEventQueue_enableSensor", s_zero), S("ASensorEventQueue_disableSensor", s_zero),
    S("ASensorEventQueue_getEvents", s_zero), S("ASensorEventQueue_setEventRate", s_zero),
    S("eglGetDisplay", s_egl_get_display), S("eglInitialize", s_egl_initialize), S("eglChooseConfig", s_egl_choose_config),
    S("eglGetConfigs", s_egl_get_configs), S("eglGetConfigAttrib", s_egl_get_config_attrib),
    S("eglCreateContext", s_egl_create_context), S("eglCreateWindowSurface", s_egl_create_window_surface),
    S("eglDestroySurface", s_egl_true), S("eglMakeCurrent", s_egl_make_current), S("eglSwapBuffers", s_egl_swap),
    S("eglGetError", s_egl_get_error), S("eglGetCurrentContext", s_egl_get_current_context),
    S("eglGetProcAddress", s_egl_get_proc),
    S("FMOD_System_Create", f_create), S("FMOD_Debug_Initialize", f_ok),
    S("_ZN4FMOD12ChannelGroup7releaseEv", f_ok), S("_ZN4FMOD14ChannelControl11getUserDataEPPv", f_get_user),
    S("_ZN4FMOD14ChannelControl11setUserDataEPv", f_set_user), S("_ZN4FMOD14ChannelControl4stopEv", f_ok),
    S("_ZN4FMOD14ChannelControl9isPlayingEPb", f_bool_out), S("_ZN4FMOD14ChannelControl9setPausedEb", f_ok),
    S("_ZN4FMOD14ChannelControl9setVolumeEf", f_ok), S("_ZN4FMOD5Sound7releaseEv", f_ok),
    S("_ZN4FMOD5Sound9getLengthEPjj", f_len), S("_ZN4FMOD6System10getChannelEiPPNS_7ChannelE", f_get_channel),
    S("_ZN4FMOD6System10getVersionEPj", f_uint_out1),
    S("_ZN4FMOD6System11createSoundEPKcjP22FMOD_CREATESOUNDEXINFOPPNS_5SoundE", f_create_sound),
    S("_ZN4FMOD6System11mixerResumeEv", f_ok),
    S("_ZN4FMOD6System12createStreamEPKcjP22FMOD_CREATESOUNDEXINFOPPNS_5SoundE", f_create_sound),
    S("_ZN4FMOD6System12mixerSuspendEv", f_ok),
    S("_ZN4FMOD6System13getDriverInfoEiPciP9FMOD_GUIDPiP16FMOD_SPEAKERMODES4_", f_driver_info),
    S("_ZN4FMOD6System13getNumDriversEPi", f_int_out1),
    S("_ZN4FMOD6System13setFileSystemEPF11FMOD_RESULTPKcPjPPvS5_EPFS1_S5_S5_EPFS1_S5_S5_jS4_S5_EPFS1_S5_jS5_EPFS1_P18FMOD_ASYNCREADINFOS5_ESI_i", f_ok),
    S("_ZN4FMOD6System17getSoftwareFormatEPiP16FMOD_SPEAKERMODES1_", f_sw_format),
    S("_ZN4FMOD6System17setSoftwareFormatEi16FMOD_SPEAKERMODEi", f_ok),
    S("_ZN4FMOD6System18createChannelGroupEPKcPPNS_12ChannelGroupE", f_create_group),
    S("_ZN4FMOD6System19setStreamBufferSizeEjj", f_ok), S("_ZN4FMOD6System4initEijPv", f_ok),
    S("_ZN4FMOD6System5closeEv", f_ok), S("_ZN4FMOD6System6updateEv", f_ok), S("_ZN4FMOD6System7releaseEv", f_ok),
    S("_ZN4FMOD6System9playSoundEPNS_5SoundEPNS_12ChannelGroupEbPPNS_7ChannelE", f_play),
    S("_ZN4FMOD6System9setOutputE15FMOD_OUTPUTTYPE", f_ok), S("_ZN4FMOD7Channel11setPositionEjj", f_ok),
    S("_ZN4FMOD7Channel11setPriorityEi", f_ok), S("_ZN4FMOD7Channel12getFrequencyEPf", f_float_out1),
    S("_ZN4FMOD7Channel12setFrequencyEf", f_ok), S("_ZN4FMOD7Channel12setLoopCountEi", f_ok),
    S("_ZN4FMOD7Channel15setChannelGroupEPNS_12ChannelGroupE", f_ok), S("_ZN4FMOD7Channel8getIndexEPi", f_int_out1),
    S("ZSTD_trace_compress_begin", sentry0), S("ZSTD_trace_compress_end", sentry0),
    S("ZSTD_trace_decompress_begin", sentry0), S("ZSTD_trace_decompress_end", sentry0),
    {0, 0}
};

void *android_shim_lookup(const char *name) {
    for (const Shim *s = ashims; s->name; s++) if (!strcmp(s->name, name)) return s->addr;
    if (!strncmp(name, "sentry_", 7)) return (void *)sentry0;
    return NULL;
}
extern void pthread_handle_stats(void);
void android_stats(void) {
    pthread_handle_stats();
    LOG("[stats] asset opens=%d missing=%d, eglSwapBuffers=%d, null-GL calls=%d\n", g_asset_opens, g_asset_missing, g_swaps, g_gl_calls);
}
