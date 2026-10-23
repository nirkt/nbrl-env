/* Game allocator modes (--alloc raw|safe|check, default safe).
 * safe : what Android's scudo gives the game and glibc does not: zeroed blocks, a 32-byte header so that writes into a
 *        freed block cannot reach glibc's free-list pointers, and a FIFO quarantine of freed blocks (a freed block is
 *        not reused for a while). Needed because the game writes to freed memory during boot (benign on Android,
 *        fatal heap corruption under glibc).
 * check: safe + canary trailer verified at free/realloc, reporting the allocating call site.
 * --heapcheck: canary-guarded malloc for everything the game allocates through our shim table. Header (32 B: magic,
 * size, allocating caller) and a 16-byte trailer; free/realloc verify both and report the allocation site of the
 * victim block. Pointers we did not allocate (from glibc internals) are passed to glibc's free unchanged. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#define HMAGIC 0x4842474e4f4b4f4bULL
#define TMAGIC 0x5452414c4c45524cULL
#define FREED  0x4652454544424b4bULL
typedef struct { uint64_t magic, size, caller, pad; } Hdr;
static long g_checks, g_bad;
static void report(Hdr *h, const char *what, void *by) {
    if (g_bad++ > 20) return;
    LOG("[heapcheck] %s: block %p size %lu allocated by %s", what, (void *)(h + 1), (unsigned long)h->size, addr_desc((void *)h->caller));
    LOG(", freed by %s\n", addr_desc(by));
    uint8_t *t = (uint8_t *)(h + 1) + h->size;
    LOG("[heapcheck]   trailer bytes: "); for (int i = 0; i < 16; i++) LOG("%02x", t[i]); LOG("\n");
}
/* --memstat: live bytes per allocating call site (dumped by heap_memstat_dump) */
int g_memstat;
#pragma GCC diagnostic ignored "-Wframe-address"
static pthread_mutex_t g_msl = PTHREAD_MUTEX_INITIALIZER;
#define MSN 65536
static uint64_t g_ms_key[MSN]; static int64_t g_ms_bytes[MSN], g_ms_n[MSN];
static void memstat(uint64_t caller, int64_t bytes) {
    pthread_mutex_lock(&g_msl);
    unsigned i = (unsigned)((caller * 0x9E3779B97F4A7C15ULL) >> 48) % MSN;
    while (g_ms_key[i] && g_ms_key[i] != caller) i = (i + 1) % MSN;
    g_ms_key[i] = caller; g_ms_bytes[i] += bytes; g_ms_n[i] += bytes > 0 ? 1 : -1;
    pthread_mutex_unlock(&g_msl);
}
void heap_memstat_dump(const char *label) {
    if (!g_memstat) return;
    int idx[40]; int64_t tot = 0;
    for (int k = 0; k < 40; k++) idx[k] = -1;
    for (int i = 0; i < MSN; i++) { if (!g_ms_key[i]) continue; tot += g_ms_bytes[i];
        for (int k = 0; k < 40; k++) if (idx[k] < 0 || g_ms_bytes[i] > g_ms_bytes[idx[k]]) { memmove(idx + k + 1, idx + k, (39 - k) * sizeof(int)); idx[k] = i; break; } }
    LOG("[memstat] %s: live game heap %lld MB by caller:", label, (long long)(tot >> 20));
    for (int k = 0; k < 40 && idx[k] >= 0; k++) LOG(" %s=%lldMB/%lld", addr_desc((void *)g_ms_key[idx[k]]), (long long)(g_ms_bytes[idx[k]] >> 20), (long long)g_ms_n[idx[k]]);
    LOG("\n");
}
static void *hc_alloc(size_t n, void *caller, int zero) {
    Hdr *h = calloc(1, sizeof(Hdr) + n + 16); (void)zero;   /* always zeroed (see --zeromalloc) */
    if (!h) return NULL;
    if (g_memstat) memstat((uint64_t)caller, (int64_t)n);
    h->magic = HMAGIC; h->size = n; h->caller = (uint64_t)caller;
    uint64_t t[2] = {TMAGIC, TMAGIC}; memcpy((uint8_t *)(h + 1) + n, t, 16);
    return h + 1;
}
#include <pthread.h>
static int hc_check(void *p, void *by) {
    Hdr *h = (Hdr *)p - 1; g_checks++;
    if (h->magic == FREED) { report(h, "DOUBLE FREE", by); return 0; }
    if (h->magic != HMAGIC) return -1;                        /* not ours */
    uint64_t t[2]; memcpy(t, (uint8_t *)p + h->size, 16);
    if (t[0] != TMAGIC || t[1] != TMAGIC) report(h, "OVERFLOW past the end", by);
    return 1;
}
/* the game's call site: when malloc is reached through libc++'s operator new, take the caller one frame up */
extern Module *g_libg;
static void *site(void *ra, void *fp) {
    Module *m = module_for_addr(ra);
    if (m && m != g_libg && fp) { uint64_t *f = (uint64_t *)fp; uint64_t *p = (uint64_t *)f[0]; if (p && ((uintptr_t)p & 7) == 0) return (void *)(p[1] & 0xffffffffffffULL); }   /* strip pointer-auth bits */
    return ra;
}
static void *s_malloc(size_t n) { return hc_alloc(n, g_memstat ? site(__builtin_return_address(0), __builtin_frame_address(0)) : __builtin_return_address(0), 0); }
static void *s_calloc(size_t a, size_t b) { return hc_alloc(a * b, __builtin_return_address(0), 1); }
#define QN 16384
static Hdr *g_q[QN]; static unsigned g_qi, g_qo; static pthread_mutex_t g_ql = PTHREAD_MUTEX_INITIALIZER;
static size_t g_qbytes; size_t g_qbudget = (size_t)48 << 20;   /* --quarantine-mb: bytes of freed blocks kept back */
int g_alloc_check;
static void quarantine(Hdr *h) {
    h->magic = FREED;
    if (g_memstat) memstat(h->caller, -(int64_t)h->size);
    Hdr *drop[64]; int nd = 0;
    pthread_mutex_lock(&g_ql);
    if (g_qi - g_qo == QN) { drop[nd++] = g_q[g_qo % QN]; g_qbytes -= drop[nd - 1]->size; g_qo++; }
    g_q[g_qi % QN] = h; g_qi++; g_qbytes += h->size;
    while (g_qbytes > g_qbudget && g_qi - g_qo > 1 && nd < 64) { drop[nd++] = g_q[g_qo % QN]; g_qbytes -= drop[nd - 1]->size; g_qo++; }
    pthread_mutex_unlock(&g_ql);
    for (int i = 0; i < nd; i++) free(drop[i]);
}
size_t heap_quarantine_bytes(void) { return g_qbytes; }
static void s_free(void *p) {
    if (!p) return;
    Hdr *h = (Hdr *)p - 1;
    if (g_alloc_check) {
        int r = hc_check(p, __builtin_return_address(0));
        if (r < 0) { free(p); return; }
        if (r == 0) return;                                   /* double free: leak it, keep running */
    } else {
        if (h->magic == FREED) { g_bad++; return; }           /* double free */
        if (h->magic != HMAGIC) { free(p); return; }          /* not ours */
    }
    quarantine(h);
}
static void *s_realloc(void *p, size_t n) {
    if (!p) return hc_alloc(n, __builtin_return_address(0), 0);
    int r = hc_check(p, __builtin_return_address(0));
    if (r < 0) return realloc(p, n);
    Hdr *h = (Hdr *)p - 1; void *q = hc_alloc(n, (void *)h->caller, 0);
    if (q) { memcpy(q, p, h->size < n ? h->size : n); quarantine(h); }
    return q;
}
static int s_posix_memalign(void **out, size_t al, size_t n) {
    if (al <= 16) { *out = hc_alloc(n, __builtin_return_address(0), 0); return *out ? 0 : ENOMEM; }
    return posix_memalign(out, al, n);                        /* rare: unguarded */
}
void *heapcheck_lookup(const char *name) {
    if (!strcmp(name, "malloc")) return s_malloc;
    if (!strcmp(name, "calloc")) return s_calloc;
    if (!strcmp(name, "free")) return s_free;
    if (!strcmp(name, "realloc")) return s_realloc;
    if (!strcmp(name, "posix_memalign")) return s_posix_memalign;
    return NULL;
}
void heapcheck_stats(void) { LOG("[heapcheck] %ld frees checked, %ld problems\n", g_checks, g_bad); }

/* --zeromalloc: malloc returns zeroed memory (tests whether the game depends on fresh allocations being zero) */
static void *z_malloc(size_t n) { return calloc(1, n ? n : 1); }
static void *z_realloc(void *p, size_t n) { return realloc(p, n); }
void *zeromalloc_lookup(const char *name) {
    if (!strcmp(name, "malloc")) return z_malloc;
    return NULL;
}
