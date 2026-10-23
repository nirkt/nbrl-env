/* boot stage: replay what Titan's Java side does at app start, then let the game's own threads run.
 *   nOnNativeLibrariesLoaded -> nOnApplicationCreate -> createGameMain(...) -> nOnCreate (starts AndroidMain::mainLoop
 *   thread) -> nOnStart -> nOnResume -> nOnSurfaceCreated -> nOnSurfaceChanged(w,h) -> nOnWindowFocusChanged(true)
 * Options: --run-seconds N (default 20). */
#include "nbhost.h"
#define B_LIBG(off) ((uint8_t *)g_libg->base + (off))
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/resource.h>

extern void android_stats(void);
static void *g_thiz, *g_app_cls;

#include <time.h>
extern void *g_sim;
/* --fast: the frame pacer's nanosleep is skipped only while a battle exists (the lobby/boot stays paced at 60 fps,
 * so many instances booting together do not burn CPU on unneeded lobby frames) */
static int noop_sleep(const struct timespec *a, struct timespec *b) { return g_sim ? 0 : nanosleep(a, b); }
/* --fps N: the game's frame pacer sleeps return at once (the game runs uncapped; nbgl --fps paces the presentation) */
static int nosleep_all(const struct timespec *a, struct timespec *b) { (void)a; (void)b; return 0; }
static NativeReg *need(const char *cls, const char *name) {
    NativeReg *r = jni_find_native(cls, name);
    if (!r) { LOG("[boot] native %s.%s not registered\n", cls, name); exit(1); }
    return r;
}
#define GA "com/supercell/titan/GameApp"
static void call0(const char *cls, const char *name) {
    NativeReg *r = need(cls, name);
    LOG("[boot] -> %s.%s\n", cls, name);
    ((void (*)(void *, void *))r->fn)(jni_env(), g_thiz);
    LOG("[boot] <- %s.%s\n", cls, name);
}

int stage_boot(int argc, char **argv) {
    int run_s = 20, W = 2400, H = 1080, dpi = 450, autoplay = -1, fast = 0, target_fps = 0;
    int safe[4] = {0, 0, 0, 0}, safe_set = 0;   /* --safe-margins a,b,c,d: GameApp.setSafeMargins (Java calls it on Android with the
                                                 * display-cutout insets; S20 FE in the battle: 0,0,88,0 read from 0x1242688/84/8c/90) */
    float xdpi = -1, ydpi = -1;   /* createGameMain's physical dpi (default = dpi). The S20 FE (T6): dpi 280 (wm density override),
                                   * xdpi 409.432, ydpi 406.4, read from the game's globals 0x1242678/0x124267c/0x1242680 */
    extern struct NbglShm *g_gl;
    if (g_gl) { W = ((uint32_t *)g_gl)[2]; H = ((uint32_t *)g_gl)[3]; dpi = 300; }   /* NbglShm.width/height */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--run-seconds") && i + 1 < argc) run_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dpi") && i + 1 < argc) dpi = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--xdpi") && i + 1 < argc) xdpi = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--safe-margins") && i + 1 < argc) { safe_set = sscanf(argv[++i], "%d,%d,%d,%d", &safe[0], &safe[1], &safe[2], &safe[3]) == 4; }
        else if (!strcmp(argv[i], "--ydpi") && i + 1 < argc) ydpi = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--autoplay") && i + 1 < argc) autoplay = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast")) fast = 1;
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) target_fps = atoi(argv[++i]);
    }
    LOG("[boot] screen %dx%d dpi %d xdpi %.3f ydpi %.3f%s\n", W, H, dpi, xdpi > 0 ? xdpi : dpi, ydpi > 0 ? ydpi : dpi, fast ? ", fast (frame pacer sleeps removed)" : "");
    if (fast) { extern void hook_bl(Module *, uint64_t, void *); extern Module *g_libg;
        hook_bl(g_libg, 0xea3928, (void *)noop_sleep); hook_bl(g_libg, 0xea3b44, (void *)noop_sleep); }
    if (target_fps > 0 && !fast) {   /* --fps N: frame pacer off; the window's presentation is paced by nbgl --fps N */
        extern Module *g_libg; extern void hook_bl(Module *, uint64_t, void *);
        hook_bl(g_libg, 0xea3928, (void *)nosleep_all); hook_bl(g_libg, 0xea3b44, (void *)nosleep_all);
        LOG("[boot] frame pacer off (uncapped), presentation paced by nbgl at %d fps\n", target_fps);
    }
    extern int g_screen_w, g_screen_h; g_screen_w = W; g_screen_h = H;
    g_thiz = jni_new_object(GA);
    g_app_cls = jni_new_object("com/supercell/titan/TitanApplication");
    NativeReg *r = need("com/supercell/titan/TitanApplication", "nOnNativeLibrariesLoaded");
    LOG("[boot] -> nOnNativeLibrariesLoaded\n"); ((void (*)(void *, void *))r->fn)(jni_env(), g_app_cls);
    call0(GA, "nOnApplicationCreate");
    /* createGameMain(AssetManager, filesDir, cacheDir, versionName, long versionCode, int w, int h, int dpi,
     *                float xdpi, float ydpi, int sdk, String pkg, Activity) -- argument meanings inferred from 0xec6974 */
    r = need(GA, "createGameMain");
    typedef void *(*cgm_t)(void *, void *, void *, void *, void *, void *, long, int, int, int, float, float, int, void *, void *);
    LOG("[boot] -> createGameMain\n");
    void *ret = ((cgm_t)r->fn)(jni_env(), g_thiz, jni_new_object("android/content/res/AssetManager"),
        jni_new_string("/tmp/nbhost_files"), jni_new_string("/tmp/nbhost_files/cache"), jni_new_string("68.279"),
        68279L, W, H, dpi, xdpi > 0 ? xdpi : (float)dpi, ydpi > 0 ? ydpi : (float)dpi, 33, jni_new_string("daniillnull.nulls.brawlstars"),
        jni_new_object("com/supercell/brawlstars/GameApp"));
    LOG("[boot] <- createGameMain returned %s\n", jni_obj_desc(ret));
    call0(GA, "nOnCreate");
    call0(GA, "nOnStart");
    call0(GA, "nOnResume");
    void *surface = jni_new_object("android/view/Surface");
    r = need(GA, "nOnSurfaceCreated");
    LOG("[boot] -> nOnSurfaceCreated\n"); ((void (*)(void *, void *, void *))r->fn)(jni_env(), g_thiz, surface);
    r = need(GA, "nOnSurfaceChanged");
    LOG("[boot] -> nOnSurfaceChanged\n"); ((void (*)(void *, void *, void *, int, int))r->fn)(jni_env(), g_thiz, surface, W, H);
    r = need(GA, "nOnWindowFocusChanged");
    LOG("[boot] -> nOnWindowFocusChanged\n"); ((void (*)(void *, void *, int))r->fn)(jni_env(), g_thiz, 1);
    if (safe_set) {
        NativeReg *sm = need(GA, "setSafeMargins");
        LOG("[boot] -> setSafeMargins(%d, %d, %d, %d)\n", safe[0], safe[1], safe[2], safe[3]);
        ((void (*)(void *, void *, int, int, int, int))sm->fn)(jni_env(), g_thiz, safe[0], safe[1], safe[2], safe[3]);
    }
    if (g_gl) { extern void nb_gl_start_input(void *); nb_gl_start_input(need(GA, "nOnTouchEvent")->fn); }
    LOG("[boot] lifecycle done; letting the game threads run for %d s\n", run_s);
    extern void offline_status(void); extern long net_bytes_out(void);
    extern int g_home_sent, g_battles, g_ticks; extern int g_swaps_count(void);
    int home_at = -1, last_sw = 0; struct rusage ru0; getrusage(RUSAGE_SELF, &ru0); double cpu_last = 0;
    for (int s = 0; s < run_s; s++) {
        if (g_home_sent && home_at < 0) home_at = s;
        if (autoplay >= 0 && home_at >= 0 && s == home_at + autoplay && !g_battles) {
            NativeReg *t = need(GA, "nOnTouchEvent"); int x = W * 2134 / 2400, y = H * 970 / 1080;
            LOG("[boot] autoplay: tapping PLAY at (%d, %d)\n", x, y);
            ((void (*)(void *, void *, int, int, int, int))t->fn)(jni_env(), g_thiz, 0, x, y, 0); usleep(150000);
            ((void (*)(void *, void *, int, int, int, int))t->fn)(jni_env(), g_thiz, 1, x, y, 0);
        }
        { struct rusage ru; getrusage(RUSAGE_SELF, &ru);
          double cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
          int sw = g_swaps_count();
          if (s > 0) LOG("[perf] t=%ds frames/s=%d cpu=%.2f cores battle-ticks=%d\n", s, sw - last_sw, cpu - cpu_last, g_ticks);
          last_sw = sw; cpu_last = cpu; } sleep(1); if (g_gl && !((volatile uint32_t *)g_gl)[14]) { LOG("[boot] window closed\n"); _exit(0); } if (s % 10 == 0) offline_status(); if (s % 5 == 4) { extern void nb_gl_stats(void); nb_gl_stats(); android_stats(); LOG("[net] bytes the client sent to the local sink: %ld\n", net_bytes_out()); } }
    jni_dump_summary(); android_stats();
    LOG("[boot] exiting\n");
    _exit(0);
}
