/* --prof: sampling profiler for the game's main-loop thread. A helper thread sends SIGPROF to the target thread every
 * PROF_US microseconds; the handler stores pc + the frame-pointer chain (8 return addresses). prof_dump() prints the
 * top libg functions (by the innermost libg address, bucketed to the enclosing function via the frame chain) since
 * the last dump. Used to find where resets and steps spend their time. */
#include "nbhost.h"
#include <signal.h>
#include <ucontext.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>

extern Module *g_libg;
int g_prof; static int g_prof_tid; static volatile int g_prof_on;
#define NS 200000
#define DEPTH 10
static uint64_t g_samp[NS][DEPTH]; static volatile int g_ns;
static void on_prof(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si; ucontext_t *uc = uc_; mcontext_t *mc = &uc->uc_mcontext;
    int i = g_ns; if (i >= NS) return; g_ns = i + 1;
    uint64_t *s = g_samp[i]; s[0] = mc->pc; s[1] = mc->regs[30];
    uint64_t *fp = (uint64_t *)mc->regs[29], lo = mc->sp, hi = mc->sp + (1 << 20);   /* frames stay within 1 MB above sp */
    for (int k = 2; k < DEPTH; k++) {
        if (!fp || ((uintptr_t)fp & 7) || (uint64_t)fp < lo || (uint64_t)fp + 16 > hi) { s[k] = 0; fp = 0; continue; }
        s[k] = fp[1]; uint64_t *n = (uint64_t *)fp[0]; if (n <= fp) { fp = 0; continue; } fp = n;
    }
}
static void *prof_thread(void *a) {
    (void)a; struct timespec ts = {0, 1000000};
    for (;;) { nanosleep(&ts, NULL); if (g_prof_on && g_prof_tid) syscall(SYS_tgkill, getpid(), g_prof_tid, SIGPROF); }
    return NULL;
}
void prof_set_thread(int tid) { if (!g_prof || g_prof_tid) return; g_prof_tid = tid;
    struct sigaction sa = {0}; sa.sa_sigaction = on_prof; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigaction(SIGPROF, &sa, NULL);
    pthread_t t; pthread_create(&t, NULL, prof_thread, NULL); LOG("[prof] sampling thread %d every 1 ms\n", tid); }
/* --prof-thread ENTRY:N: sample the N-th thread (0-based) started at libg+ENTRY instead of the main-loop thread */
uint64_t g_prof_entry; int g_prof_nth;
void prof_thread_started(void *fn, int tid) {
    if (!g_prof || !g_prof_entry || (uint64_t)fn != (uint64_t)g_libg->base + g_prof_entry) return;
    if (g_prof_nth-- == 0) { LOG("[prof] thread %d (libg+%#lx) selected\n", tid, (unsigned long)g_prof_entry); prof_set_thread(tid); }
}
void prof_begin(void) { if (g_prof) { g_ns = 0; g_prof_on = 1; } }
void prof_resume(void) { if (g_prof) g_prof_on = 1; }   /* accumulate over several windows */
void prof_pause(void) { g_prof_on = 0; }
typedef struct { uint64_t key; int n; } Cnt;
static int cmpc(const void *a, const void *b) { return ((const Cnt *)b)->n - ((const Cnt *)a)->n; }
static void top(const char *what, uint64_t (*keyf)(uint64_t *), int n) {
    static Cnt c[8192]; int nc = 0;
    for (int i = 0; i < n; i++) { uint64_t k = keyf(g_samp[i]); int j; for (j = 0; j < nc; j++) if (c[j].key == k) { c[j].n++; break; }
        if (j == nc && nc < 8192) { c[nc].key = k; c[nc++].n = 1; } }
    qsort(c, nc, sizeof *c, cmpc);
    LOG("[prof] %s (%d samples):", what, n);
    for (int j = 0; j < nc && j < 25; j++) LOG(" %s=%d", c[j].key ? addr_desc((void *)c[j].key) : "?", c[j].n);
    LOG("\n");
}
static uint64_t inlibg(uint64_t a) { return a >= (uint64_t)g_libg->base && a < (uint64_t)g_libg->base + g_libg->span; }
static uint64_t k_pc(uint64_t *s) { return s[0] & ~0x3fULL; }
static uint64_t k_firstlibg(uint64_t *s) { for (int k = 0; k < DEPTH; k++) if (inlibg(s[k])) return s[k]; return 0; }
static int g_depth_k;
static uint64_t k_depth(uint64_t *s) { int seen = 0; for (int k = DEPTH - 1; k >= 0; k--) if (inlibg(s[k])) { if (seen++ == g_depth_k) return s[k]; } return 0; }
void prof_dump(const char *label) {
    if (!g_prof) return; g_prof_on = 0; int n = g_ns;
    { const char *raw = getenv("NBHOST_PROF_RAW");   /* raw samples (libg offsets, 0 = not in libg) for offline inclusive analysis */
      if (raw) { char fn[600]; static int k; snprintf(fn, sizeof fn, "%s.%d", raw, k++); FILE *f = fopen(fn, "wb");
        if (f) { fprintf(f, "# %s\n", label); for (int i = 0; i < n; i++) { for (int d = 0; d < DEPTH; d++) {
            uint64_t a = g_samp[i][d]; fprintf(f, "%s%lx", d ? " " : "", inlibg(a) ? (unsigned long)(a - (uint64_t)g_libg->base) : 0UL); }
            fputc('\n', f); } fclose(f); LOG("[prof] raw samples -> %s\n", fn); } } }
    LOG("[prof] ===== %s: %d samples (~%d ms)\n", label, n, n);
    top("pc (64 B buckets)", k_pc, n); top("innermost libg address", k_firstlibg, n);
    for (g_depth_k = 0; g_depth_k < 4; g_depth_k++) { char t[64]; snprintf(t, sizeof t, "outermost libg frame +%d", g_depth_k); top(t, k_depth, n); }
}
