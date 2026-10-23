/* nbhost driver.  usage (from work/loader):
 *   unshare -rn qemu-aarch64 ./nbhost --libs <dir with libg.so, libc++_shared.so, libscid_sdk.so> --assets <apk extract dir>
 *        [--stage onload|boot|...] [-v]
 * Stages: onload = load + relocate + static constructors + JNI_OnLoad (prints the registered natives). */
#include "nbhost.h"
#include "ret_patches.h"
int g_phone_patches = 0;   /* --phone-patches: RETs alone break the boot (they work together with WildCat hooks we lack) */
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

int g_verbose; FILE *g_log;
extern int g_nmissing;
extern void android_stats(void);
int stage_boot(int argc, char **argv);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void on_crash(int sig, siginfo_t *si, void *uc_) {
    ucontext_t *uc = uc_;
    mcontext_t *mc = &uc->uc_mcontext;
    { extern void env_mark_crash(int); env_mark_crash(sig); }
    fprintf(g_log, "\n[crash] signal %d addr %p pc %s\n", sig, si->si_addr, addr_desc((void *)mc->pc));
    fprintf(g_log, "[crash] lr %s sp %#llx\n", addr_desc((void *)mc->regs[30]), (unsigned long long)mc->sp);
    for (int i = 0; i < 31; i++) fprintf(g_log, "  x%-2d %016llx%s", i, (unsigned long long)mc->regs[i], i % 4 == 3 ? "\n" : "");
    fprintf(g_log, "\n[crash] backtrace (frame chain):\n");
    uint64_t *fp = (uint64_t *)mc->regs[29];
    for (int i = 0; i < 24 && fp && ((uintptr_t)fp & 7) == 0; i++) {
        fprintf(g_log, "   #%d %s\n", i, addr_desc((void *)fp[1]));
        uint64_t *n = (uint64_t *)fp[0]; if (n <= fp) break; fp = n;
    }
    { extern uint64_t g_gl_cmds, g_gl_syncs; fprintf(g_log, "[crash] GL commands sent so far: %lu (sync %lu)\n", g_gl_cmds, g_gl_syncs); }
    fflush(g_log);
    _exit(128 + sig);
}

/* SIGUSR2 (sent by the vector env before it kills a hung instance): every thread logs its pc / lr / frame chain.
 * The first receiver forwards SIGUSR2 to all other threads of the process (tgkill). Debug aid, not async-signal-safe. */
#include <dirent.h>
#include <sys/syscall.h>
static volatile int g_dump_started;
static void on_dump(int sig, siginfo_t *si, void *uc_) {
    (void)si; ucontext_t *uc = uc_; mcontext_t *mc = &uc->uc_mcontext; int me = (int)syscall(SYS_gettid);
    if (__sync_bool_compare_and_swap(&g_dump_started, 0, 1)) {
        fprintf(g_log, "\n[hang-dump] SIGUSR2: thread stacks follow\n");
        { extern void locktrace_dump(void); locktrace_dump(); }
        DIR *d = opendir("/proc/self/task"); struct dirent *e;
        if (d) { while ((e = readdir(d))) { int t = atoi(e->d_name); if (t > 0 && t != me) syscall(SYS_tgkill, getpid(), t, sig); } closedir(d); }
    }
    fprintf(g_log, "[hang-dump] tid %d pc %s lr %s\n", me, addr_desc((void *)mc->pc), addr_desc((void *)mc->regs[30]));
    {   /* a thread blocked in a futex wait: x0 = the futex word; for a glibc mutex that is its start, __owner at +8 */
        uint64_t a = mc->regs[0];
        if (a > 0x10000 && a < 0x0001000000000000ULL && (a & 3) == 0) {
            int32_t *w = (int32_t *)a;
            fprintf(g_log, "[hang-dump]   tid %d x0 %#lx: words %d %d %d %d %d (as a mutex: lock, count, owner, nusers, kind)\n", me,
                    (unsigned long)a, w[0], w[1], w[2], w[3], w[4]);
        }
    }
    uint64_t *fp = (uint64_t *)mc->regs[29];
    for (int i = 0; i < 16 && fp && ((uintptr_t)fp & 7) == 0; i++) {
        fprintf(g_log, "[hang-dump]   tid %d #%d %s\n", me, i, addr_desc((void *)fp[1]));
        uint64_t *n = (uint64_t *)fp[0]; if (n <= fp) break; fp = n;
    }
    fflush(g_log);
}

const char *g_libdir = ".", *g_payloads = "payloads", *g_glpath;
Module *g_libg;
double g_t0;

#include <malloc.h>
int main(int argc, char **argv) {
    g_log = stderr;
    { extern int g_locktrace; g_locktrace = getenv("NBHOST_LOCKTRACE") != NULL; }
    { int arenas = 2; for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--arenas") && i + 1 < argc) arenas = atoi(argv[i + 1]);
      if (arenas > 0) mallopt(M_ARENA_MAX, arenas); }   /* glibc default: 8 x cores 64 MB arenas (most of the RSS) */
    const char *stage = "onload";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--libs") && i + 1 < argc) g_libdir = argv[++i];
        else if (!strcmp(argv[i], "--assets") && i + 1 < argc) g_asset_root = argv[++i];
        else if (!strcmp(argv[i], "--stage") && i + 1 < argc) stage = argv[++i];
        else if (!strcmp(argv[i], "--payloads") && i + 1 < argc) g_payloads = argv[++i];
        else if (!strcmp(argv[i], "--gl") && i + 1 < argc) g_glpath = argv[++i];
        else if (!strcmp(argv[i], "-v")) g_verbose++;
        else if (!strcmp(argv[i], "--phone-patches")) g_phone_patches = 1;
        else if (!strcmp(argv[i], "--alloc") && i + 1 < argc) {   /* raw | safe (default) | check */
            extern int g_heapcheck, g_alloc_check; const char *m = argv[++i];
            g_heapcheck = strcmp(m, "raw") != 0; g_alloc_check = !strcmp(m, "check");
        }
        else if (!strcmp(argv[i], "--zeromalloc")) { extern int g_zeromalloc; g_zeromalloc = 1; }
        else if (!strcmp(argv[i], "--roster") && i + 1 < argc) { extern const char *g_roster; g_roster = argv[++i]; }
        else if (!strcmp(argv[i], "--cube-rules")) { extern int g_window_cubes; g_window_cubes = 1; }
        else if (!strcmp(argv[i], "--storm-rules")) { extern int g_window_storm; g_window_storm = 1; }
        else if (!strcmp(argv[i], "--elimination")) { extern int g_window_elim; g_window_elim = 1; }
        else if (!strncmp(argv[i], "--div-", 6) && i + 1 < argc) { extern int div_window_arg(const char *, const char *); div_window_arg(argv[i] + 6, argv[++i]); }
        else if (!strcmp(argv[i], "--lockstep") && i + 1 < argc) { extern int g_lockstep; g_lockstep = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cores") && i + 1 < argc) { extern int g_cores; g_cores = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--env") && i + 1 < argc) { extern int env_attach(const char *); if (env_attach(argv[++i])) return 1; }
        else if (!strcmp(argv[i], "--env-client")) { extern int g_env_simonly; g_env_simonly = 0; }
        else if (!strcmp(argv[i], "--no-snapshot")) { extern int g_no_snapshot; g_no_snapshot = 1; }
        else if (!strcmp(argv[i], "--snapshot")) { extern int g_no_snapshot; g_no_snapshot = 0; }
        else if (!strcmp(argv[i], "--quarantine-mb") && i + 1 < argc) { extern size_t g_qbudget; g_qbudget = (size_t)atoi(argv[++i]) << 20; }
        else if (!strcmp(argv[i], "--arenas") && i + 1 < argc) i++;
        else if (!strcmp(argv[i], "--gas-lag") && i + 1 < argc) { extern int g_gas_render_lag, g_gas_model; g_gas_render_lag = atoi(argv[++i]); g_gas_model = 0; }
        else if (!strcmp(argv[i], "--memstat")) { extern int g_memstat; g_memstat = 1; }
        else if (!strcmp(argv[i], "--fire-delay") && i + 1 < argc) { extern int g_fire_delay; g_fire_delay = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cl-native-intents") && i + 1 < argc) { extern int g_cl_native_intents; g_cl_native_intents = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--intro-tpf") && i + 1 < argc) { extern int g_intro_tpf; g_intro_tpf = atoi(argv[++i]); if (g_intro_tpf < 1) g_intro_tpf = 1; }
        else if (!strcmp(argv[i], "--cmd-fd") && i + 1 < argc) { extern int g_cmd_fd; g_cmd_fd = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--bell-spin-us") && i + 1 < argc) { extern int g_bell_spin_us; g_bell_spin_us = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--resp-fd") && i + 1 < argc) { extern int g_resp_fd; g_resp_fd = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--no-load-yield")) { extern int g_load_yield; g_load_yield = 0; }
        else if (!strcmp(argv[i], "--shared-copies")) { extern int g_h_alias_fix; g_h_alias_fix = 0; }
        else if (!strcmp(argv[i], "--no-release-defer")) { extern int g_release_defer; g_release_defer = 0; }
        else if (!strcmp(argv[i], "--no-acc694")) { extern int g_acc694; g_acc694 = 0; }
        else if (!strcmp(argv[i], "--acc694-check")) { extern int g_acc694_check; g_acc694_check = 1; }
        else if (!strcmp(argv[i], "--jni-free") && i + 1 < argc) { extern int g_jni_free; g_jni_free = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--intro-render") && i + 1 < argc) { extern int g_intro_render; g_intro_render = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--loading-render") && i + 1 < argc) { extern int g_loading_render; g_loading_render = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--vis-model") && i + 1 < argc) { extern int g_vis_model; g_vis_model = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--count-hits") && i + 1 < argc) { extern const char *g_hits_file; g_hits_file = argv[++i]; }
        else if (!strcmp(argv[i], "--no-item-guard")) { extern int g_item_guard; g_item_guard = 0; }
        else if (!strcmp(argv[i], "--no-weak-skip")) { extern int g_weak_skip; g_weak_skip = 0; }
        else if (!strcmp(argv[i], "--no-cube-counter")) { extern int g_cube_counter; g_cube_counter = 0; }
        else if (!strcmp(argv[i], "--native-charge") && i + 1 < argc) { extern int g_native_charge; g_native_charge = atoi(argv[++i]); }   /* diversity round */
        else if (!strcmp(argv[i], "--area-dot") && i + 1 < argc) { extern int g_area_dot; g_area_dot = atoi(argv[++i]); }   /* diversity round */
        else if (!strcmp(argv[i], "--keep-snap-streams")) { extern int g_free_snap_stream; g_free_snap_stream = 0; }
        else if (!strcmp(argv[i], "--prof")) { extern int g_prof; g_prof = 1; }
        else if (!strcmp(argv[i], "--prof-thread") && i + 1 < argc) { extern int g_prof, g_prof_nth; extern uint64_t g_prof_entry; g_prof = 1;
            char *e; g_prof_entry = strtoull(argv[++i], &e, 16); g_prof_nth = *e == ':' ? atoi(e + 1) : 0; }
        else if (!strcmp(argv[i], "--steps-per-frame") && i + 1 < argc) { extern int g_env_spf; g_env_spf = atoi(argv[++i]); }
    }
    struct sigaction sa = {0}; sa.sa_sigaction = on_crash; sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    static char altstack[1 << 16]; stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack}; sigaltstack(&ss, NULL);
    sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL); sigaction(SIGILL, &sa, NULL); sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL); sigaction(SIGTRAP, &sa, NULL);
    { struct sigaction sd = {0}; sd.sa_sigaction = on_dump; sd.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART; sigaction(SIGUSR2, &sd, NULL); }
    mkdir("/tmp/nbhost_files", 0755);
    g_t0 = now();
    shims_init();
    if (g_glpath) { extern int nb_gl_attach(const char *); if (nb_gl_attach(g_glpath)) return 1; }
    char p[1024];
    snprintf(p, sizeof p, "%s/libc++_shared.so", g_libdir); Module *cxx = load_module(p, "libc++_shared.so");
    snprintf(p, sizeof p, "%s/libscid_sdk.so", g_libdir); Module *scid = load_module(p, "libscid_sdk.so");
    snprintf(p, sizeof p, "%s/libg.so", g_libdir); g_libg = load_module(p, "libg.so");
    relocate_module(cxx); relocate_module(scid); relocate_module(g_libg);
    LOG("[time] loaded+relocated in %.3f s; unresolved imports: %d\n", now() - g_t0, g_nmissing);
    if (g_phone_patches) {   /* reproduce the repack's runtime RET/NOP patches (what WildCat leaves in memory on the phone) */
        extern void patch_u32(Module *, uint64_t, uint32_t);
        for (unsigned i = 0; i < sizeof g_ret_patches / 4; i++) patch_u32(g_libg, g_ret_patches[i], 0xd65f03c0);
        for (int r = 0; r < 2; r++) for (unsigned k = 0; k < g_nop_runs[r][1]; k++) patch_u32(g_libg, g_nop_runs[r][0] + 4 * k, 0xd503201f);
        LOG("[patch] applied %zu RET patches and 2 NOP runs from the phone\n", sizeof g_ret_patches / 4);
    }
    { extern int g_env_on, g_lockstep; if (g_env_on && !g_lockstep) g_lockstep = 1; }   /* env: every frame owes a tick */
    if (strcmp(stage, "onload")) { extern void offline_install(const char *); offline_install(g_payloads); }
    double t = now();
    run_initializers(cxx); run_initializers(scid); run_initializers(g_libg);
    LOG("[time] static constructors in %.3f s\n", now() - t);
    int (*onload)(void *, void *) = module_sym(g_libg, "JNI_OnLoad");
    t = now();
    int v = onload(jni_vm(), NULL);
    LOG("[onload] JNI_OnLoad returned %#x in %.3f s; %d natives registered\n", v, now() - t, g_jni_nnatives);
    if (!strcmp(stage, "onload")) { jni_dump_summary(); android_stats(); return 0; }
    return stage_boot(argc, argv);
}
