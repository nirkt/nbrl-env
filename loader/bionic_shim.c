/* bionic (Android libc) ABI on top of glibc, for AArch64. Only what libg.so / libc++_shared.so / libscid_sdk.so import.
 * Differences handled: errno accessor, __sF stdio array, FILE* mapping, pthread_mutex size (40 vs 48 bytes),
 * sigset_t size (8 vs 128 bytes), sysconf numbering, *_chk fortify entry points, EAI_* signs, dl_iterate_phdr over
 * our own module list, system properties, the TLS stack-guard slot (tp+0x28). Networking is refused. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <link.h>
#include <locale.h>
#include <setjmp.h>
#include <time.h>
#include <wchar.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netdb.h>
#include <dirent.h>
#include <dlfcn.h>
#include <wctype.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <sys/sendfile.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <poll.h>
#include <syslog.h>
#include <sched.h>
#include <netinet/in.h>
#include <arpa/inet.h>
extern int __cxa_atexit(void (*)(void *), void *, void *);
extern void __cxa_finalize(void *);
extern int __cxa_thread_atexit_impl(void (*)(void *), void *, void *);

/* ---------------- stdio ---------------- */
/* bionic's FILE is 152 bytes on LP64; __sF[0..2] are stdin/stdout/stderr */
static uint8_t bionic_sF[3 * 152];
static FILE *F(FILE *f) {
    uint8_t *p = (uint8_t *)f;
    if (p >= bionic_sF && p < bionic_sF + sizeof bionic_sF) {
        int i = (p - bionic_sF) / 152;
        return i == 0 ? stdin : i == 1 ? stderr /* game stdout -> our stderr */ : stderr;
    }
    return f;
}
static int s_fclose(FILE *f) { return fclose(F(f)); }
static int s_fflush(FILE *f) { return fflush(f ? F(f) : NULL); }
static size_t s_fread(void *b, size_t s, size_t n, FILE *f) { return fread(b, s, n, F(f)); }
static size_t s_fwrite(const void *b, size_t s, size_t n, FILE *f) { return fwrite(b, s, n, F(f)); }
static size_t s_fwrite_chk(const void *b, size_t s, size_t n, FILE *f, size_t bl) { (void)bl; return fwrite(b, s, n, F(f)); }
static int s_fputc(int c, FILE *f) { return fputc(c, F(f)); }
static wint_t s_fputwc(wchar_t c, FILE *f) { return fputwc(c, F(f)); }
static int s_getc(FILE *f) { return getc(F(f)); }
static wint_t s_getwc(FILE *f) { return getwc(F(f)); }
static int s_ungetc(int c, FILE *f) { return ungetc(c, F(f)); }
static wint_t s_ungetwc(wint_t c, FILE *f) { return ungetwc(c, F(f)); }
static int s_fseek(FILE *f, long o, int w) { return fseek(F(f), o, w); }
static int s_fseeko(FILE *f, off_t o, int w) { return fseeko(F(f), o, w); }
static long s_ftell(FILE *f) { return ftell(F(f)); }
static off_t s_ftello(FILE *f) { return ftello(F(f)); }
static int s_ferror(FILE *f) { return ferror(F(f)); }
static int s_fileno(FILE *f) { return fileno(F(f)); }
static char *s_fgets(char *s, int n, FILE *f) { return fgets(s, n, F(f)); }
static int s_vfprintf(FILE *f, const char *fmt, va_list ap) { return vfprintf(F(f), fmt, ap); }
static int s_fprintf(FILE *f, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vfprintf(F(f), fmt, ap); va_end(ap); return r; }
static int s_printf(const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vfprintf(stderr, fmt, ap); va_end(ap); return r; }
static int s_putchar(int c) { return fputc(c, stderr); }
static FILE *s_fopen(const char *p, const char *m) {
    FILE *f = fopen(p, m);
    VLOG("[fopen] %s (%s) -> %p\n", p, m, (void *)f);
    return f;
}
static FILE *s_popen(const char *c, const char *t) { LOG("[popen] refused: %s\n", c); (void)t; errno = EACCES; return NULL; }
static int s_pclose(FILE *f) { (void)f; return -1; }

/* ---------------- errno, assert, abort ---------------- */
static int *s_errno(void) { return &errno; }
static void s_assert2(const char *file, int line, const char *fn, const char *msg) {
    LOG("[assert] %s:%d %s: %s\n", file, line, fn, msg); abort();
}
static void s_abort_message(const char *m) { LOG("[abort-message] %s\n", m); }
static void s_abort(void) {
    LOG("[abort] called from %s\n", addr_desc(__builtin_return_address(0)));
    abort();
}
static void s_stack_chk_fail(void) {
    LOG("[stack_chk_fail] from %s\n", addr_desc(__builtin_return_address(0))); abort();
}

/* ---------------- fortify ---------------- */
static size_t s_strlen_chk(const char *s, size_t n) { (void)n; return strlen(s); }
static char *s_strchr_chk(const char *s, int c, size_t n) { (void)n; return strchr(s, c); }
static char *s_strrchr_chk(const char *s, int c, size_t n) { (void)n; return strrchr(s, c); }
static char *s_strncpy_chk2(char *d, const char *s, size_t n, size_t dl, size_t sl) { (void)dl; (void)sl; return strncpy(d, s, n); }
static char *s_strncpy_chk(char *d, const char *s, size_t n, size_t dl) { (void)dl; return strncpy(d, s, n); }
static char *s_strcpy_chk(char *d, const char *s, size_t dl) { (void)dl; return strcpy(d, s); }
static char *s_strcat_chk(char *d, const char *s, size_t dl) { (void)dl; return strcat(d, s); }
static void *s_memcpy_chk(void *d, const void *s, size_t n, size_t dl) { (void)dl; return memcpy(d, s, n); }
static void *s_memmove_chk(void *d, const void *s, size_t n, size_t dl) { (void)dl; return memmove(d, s, n); }
static ssize_t s_read_chk(int fd, void *b, size_t n, size_t bl) { (void)bl; return read(fd, b, n); }
static int s_vsnprintf_chk(char *b, size_t n, int fl, size_t bl, const char *fmt, va_list ap) { (void)fl; (void)bl; return vsnprintf(b, n, fmt, ap); }
static int s_vsprintf_chk(char *b, int fl, size_t bl, const char *fmt, va_list ap) { (void)fl; return vsnprintf(b, bl, fmt, ap); }
static int s_open_2(const char *p, int fl) { return open(p, fl); }
static int s_open(const char *p, int fl, ...) {
    va_list ap; va_start(ap, fl); int mode = va_arg(ap, int); va_end(ap);
    int fd = open(p, fl, mode); VLOG("[open] %s flags=%#x -> %d\n", p, fl, fd); return fd;
}

/* ---------------- pthread ---------------- */
/* bionic pthread_mutex_t (40 B), pthread_cond_t (48 B) and pthread_rwlock_t (56 B) are int32 arrays: only 4-byte
 * aligned, so glibc objects cannot live in place (8-byte atomics -> SIGBUS) and no pointer fits aligned. Word 1 of the
 * bionic object holds a 32-bit handle into a table of heap-allocated glibc objects, created on first use (CAS). */
#define MAXH (1 << 20)
static void *g_h[MAXH]; static uint32_t g_hn, g_free[MAXH], g_nfree; static pthread_mutex_t g_hlock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_h_alloc, g_h_freed;
static void *g_h_obj[MAXH]; static long g_h_aliases; int g_h_alias_fix = 1;   /* see H() */
static uint32_t h_new(void *o) {
    pthread_mutex_lock(&g_hlock);
    uint32_t i = g_nfree ? g_free[--g_nfree] : ++g_hn;
    g_h_alloc++;
    pthread_mutex_unlock(&g_hlock);
    if (i >= MAXH) { LOG("[pthread] handle table full\n"); abort(); }
    g_h[i] = o; return i;
}
/* destroy: free the glibc object and recycle the handle (bionic code never touches a destroyed object again) */
static void h_free(void *b) {
    uint32_t *w = (uint32_t *)b + 1, i = __atomic_exchange_n(w, 0, __ATOMIC_ACQ_REL);
    if (!i) return;
    if (g_h_obj[i] != b) { __atomic_store_n(w, 0, __ATOMIC_RELEASE); return; }   /* a copy's stale handle: not ours to free */
    void *o = g_h[i]; g_h[i] = NULL; g_h_obj[i] = NULL; free(o);
    pthread_mutex_lock(&g_hlock); g_free[g_nfree++] = i; g_h_freed++; pthread_mutex_unlock(&g_hlock);
}
/* the glibc owner tid of a bionic mutex (0: unlocked or never used); glibc records __owner for every mutex type */
int bionic_mutex_owner(void *bm) {
    uint32_t i = __atomic_load_n((uint32_t *)bm + 1, __ATOMIC_ACQUIRE);
    if (!i || i >= MAXH || !g_h[i] || g_h_obj[i] != bm) return 0;
    return ((pthread_mutex_t *)g_h[i])->__data.__owner;
}
void pthread_handle_stats(void) { LOG("[pthread] sync objects created %lu destroyed %lu live-handles %u copies %ld\n", g_h_alloc, g_h_freed, g_hn - g_nfree, g_h_aliases); }
/* g_h_obj[i]: the bionic object a handle was made for. A bionic mutex/cond copied byte-wise (bionic allows it: the copy
 * is an independent object) carries the source's handle; without this check both would share one glibc object and a
 * thread holding one deadlocks on the other (the rare "loading deadlock": owner == waiter in the hang dumps). A copy
 * gets its own fresh object (unlocked / no waiters) on first use. --shared-copies: the old behaviour (debug). */
static void *H(void *b, void *(*mk)(void *)) {
    uint32_t *w = (uint32_t *)b + 1;
    uint32_t i = __atomic_load_n(w, __ATOMIC_ACQUIRE);
    if (i && (g_h_obj[i] == b || !g_h_alias_fix)) return g_h[i];
    if (i) {   /* a copy of another object: give it its own glibc object */
        void *o = mk(b); uint32_t n = h_new(o); g_h_obj[n] = b;
        if (__atomic_compare_exchange_n(w, &i, n, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            if (g_h_aliases++ < 20) LOG("[pthread] object %p carried handle of %p: gave it its own (copy #%ld)\n", b, g_h_obj[i], g_h_aliases);
            return o;
        }
        return g_h[i];   /* another thread fixed it first */
    }
    void *o = mk(b); uint32_t n = h_new(o), exp = 0; g_h_obj[n] = b;
    if (__atomic_compare_exchange_n(w, &exp, n, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return o;
    return g_h[exp];   /* lost the race: the loser's object leaks (rare, harmless) */
}
static void *mk_mutex(void *b) {
    pthread_mutex_t *n = malloc(sizeof *n); pthread_mutexattr_t a; pthread_mutexattr_init(&a);
    int type = (*(int32_t *)b >> 14) & 3;   /* bionic static initializers: recursive 1<<14, errorcheck 2<<14 */
    pthread_mutexattr_settype(&a, type == 1 ? PTHREAD_MUTEX_RECURSIVE : type == 2 ? PTHREAD_MUTEX_ERRORCHECK : PTHREAD_MUTEX_NORMAL);
    pthread_mutex_init(n, &a); pthread_mutexattr_destroy(&a); return n;
}
static void *mk_cond(void *b) { (void)b; pthread_cond_t *c = malloc(sizeof *c); pthread_cond_init(c, NULL); return c; }
static void *mk_rw(void *b) { (void)b; pthread_rwlock_t *c = malloc(sizeof *c); pthread_rwlock_init(c, NULL); return c; }
#define M(b) ((pthread_mutex_t *)H(b, mk_mutex))
#define C(b) ((pthread_cond_t *)H(b, mk_cond))
#define RW(b) ((pthread_rwlock_t *)H(b, mk_rw))
static int s_mutex_init(void *bm, const pthread_mutexattr_t *a) {
    memset(bm, 0, 40); pthread_mutex_t *n = malloc(sizeof *n); pthread_mutex_init(n, a);
    uint32_t i = h_new(n); g_h_obj[i] = bm; ((uint32_t *)bm)[1] = i; return 0;
}
static int s_mutex_destroy(void *bm) { h_free(bm); return 0; }
/* NBHOST_LOCKTRACE=1 (debug): owner tid of every held mutex and the mutex each thread waits for, printed by the SIGUSR2
 * stack dump (main.c on_dump -> locktrace_dump) to find who holds what in a hang */
int g_locktrace; static int32_t g_lk_owner[MAXH]; static uint32_t g_lk_wait_tid[256], g_lk_wait_h[256]; static void *g_lk_wait_ra[256];
static int lk_slot(int tid) { return (unsigned)tid % 256; }
static int s_mutex_lock(void *bm) {
    if (!g_locktrace) return pthread_mutex_lock(M(bm));
    pthread_mutex_t *m = M(bm); uint32_t h = ((uint32_t *)bm)[1]; int tid = gettid(), k = lk_slot(tid);
    g_lk_wait_tid[k] = tid; g_lk_wait_h[k] = h; g_lk_wait_ra[k] = __builtin_return_address(0);
    int r = pthread_mutex_lock(m);
    g_lk_wait_h[k] = 0; if (!r && h < MAXH) g_lk_owner[h] = tid;
    return r;
}
static int s_mutex_trylock(void *bm) {
    int r = pthread_mutex_trylock(M(bm));
    if (g_locktrace && !r) { uint32_t h = ((uint32_t *)bm)[1]; if (h < MAXH) g_lk_owner[h] = gettid(); }
    return r;
}
static int s_mutex_unlock(void *bm) {
    if (g_locktrace) { uint32_t h = ((uint32_t *)bm)[1]; if (h < MAXH && g_lk_owner[h] == gettid()) g_lk_owner[h] = 0; }
    return pthread_mutex_unlock(M(bm));
}
void locktrace_dump(void) {
    if (!g_locktrace) return;
    for (int k = 0; k < 256; k++) if (g_lk_wait_h[k])
        LOG("[locktrace] tid %u waits for mutex handle %u (owner tid %d) from %s\n", g_lk_wait_tid[k], g_lk_wait_h[k],
            g_lk_wait_h[k] < MAXH ? g_lk_owner[g_lk_wait_h[k]] : -1, addr_desc(g_lk_wait_ra[k]));
    int n = 0; for (uint32_t h = 1; h <= g_hn && h < MAXH; h++) if (g_lk_owner[h] && n++ < 64) LOG("[locktrace] mutex handle %u held by tid %d\n", h, g_lk_owner[h]);
}
static int s_cond_init(void *bc, const pthread_condattr_t *a) {
    memset(bc, 0, 48); pthread_cond_t *c = malloc(sizeof *c); pthread_cond_init(c, a); uint32_t i = h_new(c); g_h_obj[i] = bc; ((uint32_t *)bc)[1] = i; return 0;
}
static int s_cond_destroy(void *bc) { h_free(bc); return 0; }
static int s_cond_signal(void *bc) { return pthread_cond_signal(C(bc)); }
static int s_cond_broadcast(void *bc) { return pthread_cond_broadcast(C(bc)); }
static int s_cond_wait(void *bc, void *bm) {
    uint32_t h = ((uint32_t *)bm)[1]; pthread_mutex_t *m = M(bm);
    if (g_locktrace && h < MAXH) g_lk_owner[h] = 0;
    int r = pthread_cond_wait(C(bc), m);
    if (g_locktrace && h < MAXH) g_lk_owner[h] = gettid();
    return r;
}
static int s_cond_timedwait(void *bc, void *bm, const struct timespec *t) {
    uint32_t h = ((uint32_t *)bm)[1]; pthread_mutex_t *m = M(bm);
    if (g_locktrace && h < MAXH) g_lk_owner[h] = 0;
    int r = pthread_cond_timedwait(C(bc), m, t);
    if (g_locktrace && h < MAXH) g_lk_owner[h] = gettid();
    return r;
}
static int s_rw_rd(void *b) { return pthread_rwlock_rdlock(RW(b)); }
static int s_rw_wr(void *b) { return pthread_rwlock_wrlock(RW(b)); }
static int s_rw_un(void *b) { return pthread_rwlock_unlock(RW(b)); }
/* bionic sigset_t is 8 bytes on LP64 */
static int s_sigfillset(uint64_t *s) { *s = ~0ULL; return 0; }
static int s_pthread_sigmask(int how, const uint64_t *set, uint64_t *old) {
    sigset_t s, o; sigemptyset(&s);
    if (set) for (int i = 1; i < 64; i++) if (*set & (1ULL << (i - 1))) sigaddset(&s, i);
    int r = pthread_sigmask(how, set ? &s : NULL, &o);
    if (old) { *old = 0; for (int i = 1; i < 64; i++) if (sigismember(&o, i)) *old |= 1ULL << (i - 1); }
    return r;
}
struct start_box { void *(*fn)(void *); void *arg; };
static void *thread_tramp(void *p) {
    struct start_box b = *(struct start_box *)p; free(p);
    LOG("[thread] tid %d runs %s\n", (int)gettid(), addr_desc((void *)b.fn));
    { extern void prof_thread_started(void *fn, int tid); prof_thread_started((void *)b.fn, (int)gettid()); }
    return b.fn(b.arg);
}
static int s_pthread_create(pthread_t *t, const void *battr, void *(*fn)(void *), void *arg) {
    pthread_attr_t a; pthread_attr_init(&a);
    if (battr) {   /* bionic attr: u32 flags (1=detached), stack_base, stack_size, guard_size, policy, prio */
        const uint8_t *b = battr;
        if (*(const uint32_t *)b & 1) pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
        size_t ss = *(const size_t *)(b + 16); if (ss >= 65536) pthread_attr_setstacksize(&a, ss);
    } else pthread_attr_setstacksize(&a, 1 << 20);
    struct start_box *box = malloc(sizeof *box); box->fn = fn; box->arg = arg;
    int r = pthread_create(t, &a, thread_tramp, box);
    LOG("[thread] pthread_create entry=%s arg=%p -> %d\n", addr_desc((void *)fn), arg, r);
    pthread_attr_destroy(&a);
    return r;
}

/* ---------------- misc libc ---------------- */
int g_cores;   /* --cores N: CPU count the game sees (sizes its job pool; 0 = host) */
static long s_sysconf(int n) {
    switch (n) {
    case 0x27: case 0x28: return 4096;                 /* _SC_PAGESIZE / _SC_PAGE_SIZE */
    case 0x60: return g_cores ? g_cores : sysconf(_SC_NPROCESSORS_CONF);
    case 0x61: return g_cores ? g_cores : sysconf(_SC_NPROCESSORS_ONLN);
    case 0x62: return sysconf(_SC_PHYS_PAGES);
    case 0x63: return sysconf(_SC_AVPHYS_PAGES);
    case 0x06: return sysconf(_SC_CLK_TCK);
    case 0x0b: return sysconf(_SC_OPEN_MAX);
    }
    LOG("[sysconf] unknown bionic name %#x\n", n); return -1;
}
static int s_system_property_get(const char *name, char *value) {
    const char *v = "";
    if (!strcmp(name, "ro.build.version.sdk")) v = "33";
    else if (!strcmp(name, "ro.product.model")) v = "SM-G780F";
    else if (!strcmp(name, "ro.product.manufacturer")) v = "samsung";
    else if (!strcmp(name, "ro.product.cpu.abi")) v = "arm64-v8a";
    VLOG("[prop] %s -> \"%s\"\n", name, v);
    strcpy(value, v); return strlen(v);
}
static char *s_gnu_strerror_r(int e, char *b, size_t n) { return strerror_r(e, b, n); }
static int s_strerror_r(int e, char *b, size_t n) { snprintf(b, n, "%s", strerror(e)); return 0; }
static locale_t s_newlocale(int mask, const char *name, locale_t base) {
    locale_t l = newlocale(mask & LC_ALL_MASK ? mask : LC_ALL_MASK, name, base);
    if (!l) l = newlocale(LC_ALL_MASK, "C", base);
    return l;
}
static int s_register_atfork(void (*a)(void), void (*b)(void), void (*c)(void), void *d) { (void)a; (void)b; (void)c; (void)d; return 0; }
static int s_mlock(const void *a, size_t n) { (void)a; (void)n; return 0; }

/* ---------------- networking ----------------
 * The PC instance must never reach a server. TCP sockets become one end of an AF_UNIX socketpair whose other end is
 * drained (and discarded) by a local thread, so the client's connection state machine believes it is connected while
 * offline.c injects the server's replies in-process (what sink.js + offline2.js do on the phone). UDP is refused.
 * Run under `unshare -rn` as well: no network namespace at all. */
static int net_refuse(void) { errno = EACCES; return -1; }
static long g_net_out;
static void *drain(void *p) {
    int fd = (int)(intptr_t)p; char b[65536]; ssize_t r;
    while ((r = read(fd, b, sizeof b)) > 0) __atomic_add_fetch(&g_net_out, r, __ATOMIC_RELAXED);
    close(fd); return NULL;
}
static int s_socket(int d, int t, int p) {
    if ((d == AF_INET || d == AF_INET6) && (t & 0xf) == SOCK_STREAM) {
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | (t & (SOCK_NONBLOCK | SOCK_CLOEXEC)), 0, sv) < 0) return -1;
        pthread_t th; pthread_create(&th, NULL, drain, (void *)(intptr_t)sv[1]); pthread_detach(th);
        LOG("[net] socket(%d,%#x,%d) -> local sink fd %d\n", d, t, p, sv[0]);
        return sv[0];
    }
    LOG("[net] socket(%d,%#x,%d) refused\n", d, t, p); return net_refuse();
}
static int s_connect(int fd, const struct sockaddr *a, socklen_t l) {
    char h[64] = "?"; int port = 0;
    if (a && a->sa_family == AF_INET) { const struct sockaddr_in *i = (const void *)a; inet_ntop(AF_INET, &i->sin_addr, h, sizeof h); port = ntohs(i->sin_port); }
    LOG("[net] connect(fd %d, %s:%d) -> pretend connected (local sink)\n", fd, h, port); (void)l; return 0;
}
struct bionic_addrinfo { int flags, family, socktype, protocol; socklen_t addrlen; char *canonname; struct sockaddr *addr; struct bionic_addrinfo *next; };
static int s_getaddrinfo(const char *n, const char *s, const void *h, struct bionic_addrinfo **r) {
    (void)h;
    struct bionic_addrinfo *ai = calloc(1, sizeof *ai); struct sockaddr_in *sa = calloc(1, sizeof *sa);
    sa->sin_family = AF_INET; sa->sin_port = htons(s ? atoi(s) : 0); sa->sin_addr.s_addr = htonl(0x7f000001);
    ai->family = AF_INET; ai->socktype = SOCK_STREAM; ai->protocol = IPPROTO_TCP; ai->addrlen = sizeof *sa; ai->addr = (void *)sa;
    LOG("[net] getaddrinfo(%s, %s) -> 127.0.0.1 (never contacted)\n", n ? n : "-", s ? s : "-");
    *r = ai; return 0;
}
static void s_freeaddrinfo(struct bionic_addrinfo *a) { while (a) { struct bionic_addrinfo *n = a->next; free(a->addr); free(a); a = n; } }
static void s_noop(void *p) { (void)p; }
static int s_getifaddrs(void **p) { *p = NULL; return net_refuse(); }
static int s_sockopt_ok(int fd, int lvl, int opt, void *v, socklen_t *l) {
    (void)fd; (void)lvl; if (opt == SO_ERROR && v) { *(int *)v = 0; if (l) *l = 4; } return 0;
}
static int s_setsockopt(int fd, int lvl, int opt, const void *v, socklen_t l) { (void)fd; (void)lvl; (void)opt; (void)v; (void)l; return 0; }
static int s_getsockname(int fd, struct sockaddr *a, socklen_t *l) {
    (void)fd; struct sockaddr_in *i = (void *)a; memset(i, 0, sizeof *i); i->sin_family = AF_INET; i->sin_addr.s_addr = htonl(0x7f000001);
    if (l) *l = sizeof *i; return 0;
}
static ssize_t s_sendto(int fd, const void *b, size_t n, int fl, const struct sockaddr *a, socklen_t l) { (void)a; (void)l; return send(fd, b, n, fl | MSG_NOSIGNAL); }
static ssize_t s_recvfrom(int fd, void *b, size_t n, int fl, struct sockaddr *a, socklen_t *l) { (void)a; (void)l; return recv(fd, b, n, fl); }
long net_bytes_out(void) { return g_net_out; }

/* ---------------- dl* over our modules ---------------- */
static int s_dl_iterate_phdr(int (*cb)(struct dl_phdr_info *, size_t, void *), void *data) {
    for (int i = 0; i < g_nmods; i++) {
        struct dl_phdr_info info = {0};
        info.dlpi_addr = (ElfW(Addr))g_mods[i].base;
        info.dlpi_name = g_mods[i].name;
        info.dlpi_phdr = g_mods[i].phdr;
        info.dlpi_phnum = g_mods[i].phnum;
        info.dlpi_adds = g_nmods; info.dlpi_subs = 0;
        int r = cb(&info, sizeof info, data);
        if (r) return r;
    }
    return 0;
}
static int s_dladdr(const void *addr, Dl_info *info) {
    Module *m = module_for_addr(addr);
    if (!m) return 0;
    memset(info, 0, sizeof *info);
    info->dli_fname = m->name; info->dli_fbase = m->base;
    const Elf64_Sym *best = NULL;
    for (uint32_t i = 1; i < m->nsyms; i++) {
        const Elf64_Sym *s = &m->symtab[i];
        if (s->st_shndx == SHN_UNDEF) continue;
        const uint8_t *p = m->base + s->st_value;
        if (p <= (const uint8_t *)addr && (!best || s->st_value > best->st_value)) best = s;
    }
    if (best) { info->dli_sname = m->strtab + best->st_name; info->dli_saddr = m->base + best->st_value; }
    return 1;
}
static void *s_dlsym(void *h, const char *name) {
    void *p = find_symbol(name);
    LOG("[dlsym] %p %s -> %s\n", h, name, p ? addr_desc(p) : "NULL");
    return p;
}
static void *s_dlopen(const char *n, int f) { LOG("[dlopen] %s %d -> fake\n", n ? n : "(null)", f); return (void *)0x1; }
static char *s_dlerror(void) { return "nbhost: no dlopen"; }

static FILE *bionic_stdin = (FILE *)(bionic_sF), *bionic_stdout = (FILE *)(bionic_sF + 152), *bionic_stderr = (FILE *)(bionic_sF + 304);
static float s_powisf2(float a, int b) { int r = b < 0; float x = 1; for (unsigned n = r ? -b : b; n; n >>= 1, a *= a) if (n & 1) x *= a; return r ? 1 / x : x; }

/* ---------------- the table ---------------- */
extern void *android_shim_lookup(const char *name);
#define P(n) { #n, (void *)n }
#define S(n, f) { n, (void *)f }
static const Shim shims[] = {
    /* stdio */
    S("__sF", bionic_sF), S("fclose", s_fclose), S("fflush", s_fflush), S("fread", s_fread), S("fwrite", s_fwrite),
    S("__fwrite_chk", s_fwrite_chk), S("fputc", s_fputc), S("fputwc", s_fputwc), S("getc", s_getc), S("getwc", s_getwc),
    S("ungetc", s_ungetc), S("ungetwc", s_ungetwc), S("fseek", s_fseek), S("fseeko", s_fseeko), S("ftell", s_ftell),
    S("ftello", s_ftello), S("ferror", s_ferror), S("fileno", s_fileno), S("fgets", s_fgets), S("vfprintf", s_vfprintf),
    S("fprintf", s_fprintf), S("printf", s_printf), S("putchar", s_putchar), S("fopen", s_fopen), S("popen", s_popen),
    S("pclose", s_pclose), P(snprintf), P(vsnprintf), P(vasprintf), P(sscanf), P(vsscanf), P(swprintf), P(remove), P(rename),
    /* errno / abort */
    S("__errno", s_errno), S("__assert2", s_assert2), S("android_set_abort_message", s_abort_message), S("abort", s_abort),
    S("__stack_chk_fail", s_stack_chk_fail),
    /* fortify */
    S("__strlen_chk", s_strlen_chk), S("__strchr_chk", s_strchr_chk), S("__strrchr_chk", s_strrchr_chk),
    S("__strncpy_chk2", s_strncpy_chk2), S("__strncpy_chk", s_strncpy_chk), S("__strcpy_chk", s_strcpy_chk),
    S("__strcat_chk", s_strcat_chk), S("__memcpy_chk", s_memcpy_chk), S("__memmove_chk", s_memmove_chk),
    S("__read_chk", s_read_chk), S("__vsnprintf_chk", s_vsnprintf_chk), S("__vsprintf_chk", s_vsprintf_chk),
    S("__open_2", s_open_2), S("open", s_open),
    /* memory / strings */
    P(malloc), P(calloc), P(realloc), P(free), P(posix_memalign), P(memchr), P(memcmp), P(memcpy), P(memmove), P(memset),
    P(strlen), P(strnlen), P(strcmp), P(strncmp), P(strcpy), P(strncpy), P(strcat), P(strncat), P(strchr), P(strrchr),
    P(strstr), P(strcasecmp), P(strncasecmp), P(strdup), P(wcslen), P(wmemchr), P(wmemcmp), P(strerror),
    S("__gnu_strerror_r", s_gnu_strerror_r), S("strerror_r", s_strerror_r),
    P(atoi), P(atof), P(atoll), P(strtod), P(strtof), P(strtold), P(strtol), P(strtoll), P(strtoul), P(strtoull),
    P(strtoll_l), P(strtoull_l), P(strtold_l), P(wcstod), P(wcstof), P(wcstold), P(wcstol), P(wcstoll), P(wcstoul), P(wcstoull),
    P(qsort), P(bsearch), P(rand), P(srand), P(getenv),
    /* locale / wide */
    S("newlocale", s_newlocale), P(freelocale), P(uselocale), P(setlocale), P(localeconv), P(__ctype_get_mb_cur_max),
    P(btowc), P(wctob), P(mbrlen), P(mbrtowc), P(mbsnrtowcs), P(mbsrtowcs), P(mbtowc), P(wcrtomb), P(wcsnrtombs),
    P(iswalpha_l), P(iswblank_l), P(iswcntrl_l), P(iswdigit_l), P(iswlower_l), P(iswprint_l), P(iswpunct_l), P(iswspace_l),
    P(iswupper_l), P(iswxdigit_l), P(towlower_l), P(towupper_l), P(strcoll_l), P(strxfrm_l), P(wcscoll_l), P(wcsxfrm_l),
    P(strftime_l), P(strftime),
    /* math */
    P(acosf), P(asinf), P(atan2f), P(atanf), P(cbrtf), P(cos), P(cosf), P(sin), P(sinf), P(sincosf), P(tanf), P(tanhf),
    P(fmod), P(fabs), P(acos), P(asin), P(atan), P(atan2), P(cosh), P(exp), P(log10), P(sinh), P(sqrt), P(tan), P(tanh),
    S("__powisf2", s_powisf2), S("stdin", &bionic_stdin), S("stdout", &bionic_stdout), S("stderr", &bionic_stderr),
    P(exp2), P(expf), P(fmodf), P(hypotf), P(ldexp), P(ldexpf), P(log), P(log2), P(log2f), P(logf), P(pow), P(powf),
    /* pthread */
    S("pthread_mutex_init", s_mutex_init), S("pthread_mutex_destroy", s_mutex_destroy), S("pthread_mutex_lock", s_mutex_lock),
    S("pthread_mutex_trylock", s_mutex_trylock), S("pthread_mutex_unlock", s_mutex_unlock),
    P(pthread_mutexattr_init), P(pthread_mutexattr_destroy), P(pthread_mutexattr_settype),
    S("pthread_cond_init", s_cond_init), S("pthread_cond_destroy", s_cond_destroy), S("pthread_cond_signal", s_cond_signal),
    S("pthread_cond_broadcast", s_cond_broadcast),
    S("pthread_cond_wait", s_cond_wait), S("pthread_cond_timedwait", s_cond_timedwait),
    P(pthread_condattr_init), P(pthread_condattr_destroy), P(pthread_condattr_setclock),
    S("pthread_rwlock_rdlock", s_rw_rd), S("pthread_rwlock_wrlock", s_rw_wr), S("pthread_rwlock_unlock", s_rw_un),
    S("pthread_create", s_pthread_create), P(pthread_detach), P(pthread_join), P(pthread_exit), P(pthread_self),
    P(pthread_key_create), P(pthread_key_delete), P(pthread_getspecific), P(pthread_setspecific), P(pthread_once),
    P(pthread_getschedparam), P(pthread_setschedparam), S("pthread_sigmask", s_pthread_sigmask), S("sigfillset", s_sigfillset),
    P(sched_yield), P(sched_get_priority_min), P(gettid),
    /* time */
    P(clock_gettime), P(gettimeofday), P(nanosleep), P(time), P(difftime), P(gmtime),
    /* files / os */
    P(close), P(read), P(write), S("lseek64", lseek), P(stat), P(lstat), P(fstat), P(mkdir), P(rmdir), P(unlink), P(opendir),
    P(fdopendir), P(readdir), P(closedir), P(getcwd), P(chdir), P(realpath), P(readlink), P(link), P(symlink), P(fchmod),
    P(fchmodat), P(openat), P(unlinkat), P(utimensat), P(truncate), P(ftruncate), P(sendfile), P(statvfs), P(pathconf),
    P(mmap), P(munmap), P(madvise), S("mlock", s_mlock), P(getpid), P(syscall), P(ioctl), P(fcntl), P(pipe), P(poll),
    P(pause), P(eventfd), P(epoll_create), P(epoll_create1), P(epoll_ctl), P(epoll_wait), P(getauxval),
    S("sysconf", s_sysconf), S("__system_property_get", s_system_property_get), P(openlog), P(syslog), P(closelog),
    S("__register_atfork", s_register_atfork),
    S("setjmp", _setjmp), P(longjmp),
    /* C++ runtime hooks that live in bionic */
    P(__cxa_atexit), P(__cxa_finalize), P(__cxa_thread_atexit_impl),
    /* dl */
    S("dl_iterate_phdr", s_dl_iterate_phdr), S("dladdr", s_dladdr), S("dlsym", s_dlsym), S("dlopen", s_dlopen),
    S("dlerror", s_dlerror),
    /* network: refused */
    S("socket", s_socket), S("getaddrinfo", s_getaddrinfo), S("freeaddrinfo", s_freeaddrinfo), S("getifaddrs", s_getifaddrs),
    S("freeifaddrs", s_noop), S("connect", s_connect), S("bind", net_refuse), S("sendto", s_sendto),
    S("recvfrom", s_recvfrom), S("getsockopt", s_sockopt_ok), S("setsockopt", s_setsockopt), S("getsockname", s_getsockname),
    P(shutdown),
    {0, 0}
};

int g_heapcheck = 1, g_zeromalloc;   /* g_heapcheck: game allocator = safe/check wrapper (default) */
extern void *heapcheck_lookup(const char *), *zeromalloc_lookup(const char *);
void *shim_lookup(const char *name) {
    if (g_heapcheck) { void *h = heapcheck_lookup(name); if (h) return h; }
    if (g_zeromalloc) { void *h = zeromalloc_lookup(name); if (h) return h; }
    for (const Shim *s = shims; s->name; s++) if (!strcmp(s->name, name)) return s->addr;
    return android_shim_lookup(name);
}

/* bionic keeps the stack-protector canary in TLS slot 5 (tp + 0x28). In a static glibc AArch64 program the
 * executable's TLS block starts at tp + 16, so we reserve the first 64 bytes of it for bionic's slots. */
/* initialised -> .tdata, which the linker places before glibc's .tbss/.tdata from libc.a (link order) */
__thread uint64_t bionic_tls_slots[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0x5eed5eed5eed5eedULL, 0, 0};
void shims_init(void) {
    uint64_t tp; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));
    long off = (long)((uint64_t)&bionic_tls_slots[0] - tp);
    LOG("[tls] tp=%#lx bionic slots at tp%+ld (stack guard tp+0x28 = slot %ld = %#lx)\n", tp, off, (0x28 - off) / 8, *(uint64_t *)(tp + 0x28));
    if (off != 16) { LOG("[tls] unexpected TLS layout; the stack guard would alias glibc data\n"); exit(1); }
}
