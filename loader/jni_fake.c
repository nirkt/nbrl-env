/* A fake Java VM: just enough JNI for libg.so to believe it runs inside the Titan activity.
 * Every JNI call is logged (that log IS the native->Java surface); unknown Java methods return 0 / "" / a dummy object.
 * Answers for specific Java methods can be added in java_answer(). */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "jni_gen.h"
#include <dirent.h>

enum { K_CLASS = 1, K_OBJ, K_STR, K_ARR, K_METHOD, K_FIELD };
typedef struct JObj {
    uint32_t magic; int kind;
    char cls[160];             /* class name (for K_CLASS, the class itself) */
    char *str; int len;        /* K_STR: UTF-8 */
    uint8_t *data; int elsz;   /* K_ARR */
    char name[96]; char sig[256]; /* K_METHOD / K_FIELD */
    int is_static;
    int perm, refs, pins, queued, ext_data;   /* object lifetime (g_jni_free) */
} JObj;
#define JMAGIC 0x4a4f424a
/* Object lifetime. Before: every FindClass / Get*ID / string / array allocated a JObj that was never freed (~48 KB per
 * client reset). Now (g_jni_free = 1, --jni-free 0: old behaviour): classes and member IDs are cached (a real VM returns
 * the same jclass contents / jmethodID); other objects count their references (the returned local, NewLocalRef /
 * NewGlobalRef / NewWeakGlobalRef, array slots) and pins (Get*Chars / Get*ArrayElements until Release*); the last
 * Delete*Ref of an unpinned object queues it in a FIFO quarantine of JQ objects, freed when it leaves the queue, so a late
 * use after the delete (within the next JQ releases) still sees the intact object. Objects the host creates (jni_new_object / jni_new_string) are permanent. */
int g_jni_free = 1;
static long g_jlive[8], g_jfreed, g_jalloc;
#include <pthread.h>
static pthread_mutex_t g_jmu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;   /* JNI is called from several game threads */
#define JLOCK pthread_mutex_lock(&g_jmu)
#define JUNLOCK pthread_mutex_unlock(&g_jmu)
static struct { char cls[64]; long n; } g_jobj_cls[32]; static int g_njobj_cls;   /* K_OBJ allocations by class (stats) */
static JObj *jnew(int kind, const char *cls) {
    JObj *o = calloc(1, sizeof *o); o->magic = JMAGIC; o->kind = kind; o->refs = 1;
    snprintf(o->cls, sizeof o->cls, "%s", cls ? cls : "?"); JLOCK; g_jlive[kind & 7]++; g_jalloc++;
    if (kind == 2 /* K_OBJ */) { int i; for (i = 0; i < g_njobj_cls; i++) if (!strncmp(g_jobj_cls[i].cls, o->cls, 63)) break;
        if (i == g_njobj_cls && i < 32) snprintf(g_jobj_cls[g_njobj_cls++].cls, 64, "%s", o->cls);
        if (i < 32) g_jobj_cls[i].n++; }
    JUNLOCK; return o;
}
static int isj(const void *p) { return p && ((uintptr_t)p & 7) == 0 && (uintptr_t)p > 0x10000 && ((const JObj *)p)->magic == JMAGIC; }
#define JQ 2048
static JObj *g_jq[JQ]; static unsigned g_jq_n;
static void jrelease(JObj *o);
static void jfree_now(JObj *o) {
    if (o->kind == K_STR) free(o->str);
    if (o->kind == K_ARR && o->data) {
        if (o->cls[0] == '[' && (o->cls[1] == 'L' || o->cls[1] == '[') && o->elsz == 8)   /* object array: its slots' references */
            for (int i = 0; i < o->len; i++) { JObj *e = ((JObj **)o->data)[i]; if (isj(e)) jrelease(e); }
        if (!o->ext_data) free(o->data);
    }
    g_jlive[o->kind & 7]--; g_jfreed++; o->magic = 0; free(o);
}
static void jmaybe_free(JObj *o) {   /* caller holds g_jmu */
    if (!g_jni_free || o->perm || o->refs > 0 || o->pins > 0 || o->queued) return;
    o->queued = 1; JObj *old = g_jq[g_jq_n % JQ]; g_jq[g_jq_n % JQ] = o; g_jq_n++;
    if (old) { if (old->refs > 0 || old->pins > 0) old->queued = 0;   /* referenced again while queued: keep it */
               else jfree_now(old); }
}
static void jref(void *p) { if (!isj(p)) return; JLOCK; ((JObj *)p)->refs++; JUNLOCK; }
static void jrelease(JObj *o) { if (!isj(o) || o->perm) return; JLOCK; if (o->refs > 0) o->refs--; jmaybe_free(o); JUNLOCK; }
static void jpin(void *p) { if (!isj(p)) return; JLOCK; ((JObj *)p)->pins++; JUNLOCK; }
static void junpin(void *p) { if (!isj(p)) return; JObj *o = p; JLOCK; if (o->pins > 0) o->pins--; jmaybe_free(o); JUNLOCK; }
/* classes and member IDs: one permanent object per name / (class, name, signature, static) */
static struct { JObj **v; int n, cap; } g_jcache;
static JObj *jcache_find(int kind, const char *cls, const char *name, const char *sig, int st) {
    for (int i = 0; i < g_jcache.n; i++) { JObj *o = g_jcache.v[i];
        if (o->kind == kind && o->is_static == st && !strcmp(o->cls, cls) && !strcmp(o->name, name) && !strcmp(o->sig, sig)) return o; }
    return NULL;
}
static void jcache_add(JObj *o) {
    if (g_jcache.n == g_jcache.cap) { g_jcache.cap = g_jcache.cap ? g_jcache.cap * 2 : 256; g_jcache.v = realloc(g_jcache.v, g_jcache.cap * sizeof *g_jcache.v); }
    o->perm = 1; g_jcache.v[g_jcache.n++] = o;
}
static JObj *jclass(const char *name) {
    char n[160]; snprintf(n, sizeof n, "%s", name ? name : "?");
    JLOCK; JObj *o = g_jni_free ? jcache_find(K_CLASS, n, "", "", 0) : NULL;
    if (!o) { o = jnew(K_CLASS, n); if (g_jni_free) jcache_add(o); }
    JUNLOCK; return o;
}
void jni_stats_log(void) {
    LOG("[jni-mem] live objects: class %ld obj %ld str %ld arr %ld method %ld field %ld; allocated %ld freed %ld; cached %d (%zu B each)\n",
        g_jlive[K_CLASS], g_jlive[K_OBJ], g_jlive[K_STR], g_jlive[K_ARR], g_jlive[K_METHOD], g_jlive[K_FIELD], g_jalloc, g_jfreed,
        g_jcache.n, sizeof(JObj));
    LOG("[jni-mem] objects allocated by class:"); for (int i = 0; i < g_njobj_cls; i++) LOG(" %s=%ld", g_jobj_cls[i].cls, g_jobj_cls[i].n); LOG("\n");
}
const char *jni_obj_desc(void *p) {
    static __thread char b[256];
    if (!p) return "null";
    if (!isj(p)) { snprintf(b, sizeof b, "<%p>", p); return b; }
    JObj *o = p;
    switch (o->kind) {
    case K_CLASS: snprintf(b, sizeof b, "class %s", o->cls); break;
    case K_STR: snprintf(b, sizeof b, "\"%.200s\"", o->str); break;
    case K_ARR: snprintf(b, sizeof b, "%s[%d]", o->cls, o->len); break;
    case K_METHOD: snprintf(b, sizeof b, "%s.%s%s", o->cls, o->name, o->sig); break;
    default: snprintf(b, sizeof b, "obj %s", o->cls);
    }
    return b;
}
static void *jstr_new(const char *s) { JObj *o = jnew(K_STR, "java/lang/String"); o->str = strdup(s); o->len = strlen(s); return o; }
void *jni_new_string(const char *s) { JObj *o = jstr_new(s); o->perm = 1; return o; }   /* host objects: permanent */
void *jni_new_object(const char *cls) { JObj *o = jnew(K_OBJ, cls); o->perm = 1; return o; }

static void *g_vm_table[8], *g_vm[1];
static void *g_env_table[235], *g_env[1];
void *jni_vm(void) { return g_vm; }
void *jni_env(void) { return g_env; }

/* ---- statistics: which Java members does the native side touch ---- */
typedef struct { char key[300]; int n; } Hit;
static Hit g_hits[4096]; static int g_nhits;
static void hit(const char *what, const char *a, const char *b, const char *c) {
    char k[300]; snprintf(k, sizeof k, "%s %s.%s%s", what, a, b, c);
    for (int i = 0; i < g_nhits; i++) if (!strcmp(g_hits[i].key, k)) { g_hits[i].n++; return; }
    if (g_nhits < 4096) { snprintf(g_hits[g_nhits].key, 300, "%s", k); g_hits[g_nhits++].n = 1; }
    LOG("[jni] %s\n", k);
}
void jni_dump_summary(void) {
    LOG("[jni] ---- %d distinct JNI touches ----\n", g_nhits);
    for (int i = 0; i < g_nhits; i++) LOG("[jni-summary] %5d  %s\n", g_hits[i].n, g_hits[i].key);
}
NativeReg g_jni_natives[512]; int g_jni_nnatives;
NativeReg *jni_find_native(const char *cls, const char *name) {
    for (int i = 0; i < g_jni_nnatives; i++)
        if (!strcmp(g_jni_natives[i].name, name) && (!cls || !strcmp(g_jni_natives[i].cls, cls))) return &g_jni_natives[i];
    return NULL;
}

/* ---- answers for Java methods the game calls (extend as needed) ---- */
/* Object answers (CallObjectMethod / GetObjectField of a non-String class): one permanent object per class (g_jni_free;
 * the answers depend on the method name only, never on the object's identity). Before: a new object per call, e.g. 2
 * android/view/Display$Mode per reset that the game never deletes. */
static void *janswer_obj(const char *cls) {
    if (!g_jni_free) return jnew(K_OBJ, cls);
    JLOCK; JObj *o = jcache_find(K_OBJ, cls, "", "", 0);
    if (!o) { o = jnew(K_OBJ, cls); jcache_add(o); }
    JUNLOCK; return o;
}
static void *java_answer_obj(JObj *m) {
    const char *r = strrchr(m->sig, ')') + 1;
    if (!strcmp(r, "Ljava/lang/String;")) {
        const char *s = "";
        if (strstr(m->name, "Language")) s = "en";
        else if (!strcmp(m->name, "getCountry") || strstr(m->name, "Country")) s = "US";
        else if (strstr(m->name, "Model")) s = "SM-G780F";
        else if (strstr(m->name, "Manufacturer")) s = "samsung";
        else if (strstr(m->name, "OSVersion") || strstr(m->name, "OsVersion")) s = "13";
        else if (strstr(m->name, "Version")) s = "68.279";
        else if (strstr(m->name, "Path") || strstr(m->name, "Dir")) s = "/tmp/nbhost_files";
        return jstr_new(s);
    }
    if (r[0] == '[') { JObj *a = jnew(K_ARR, r); a->len = 0; a->elsz = 8; return a; }
    if (r[0] == 'L') { char c[160]; snprintf(c, sizeof c, "%.*s", (int)strlen(r) - 2, r + 1); return janswer_obj(c); }
    return NULL;
}

/* numeric answers: what a Galaxy S20 FE (the training phone) would report */
int g_screen_w = 2400, g_screen_h = 1080;
static int64_t java_answer_int(JObj *m) {
    const char *n = m->name;
    if (!strcmp(n, "getTotalMemory")) return 7680;            /* MB */
    if (!strcmp(n, "getResidentMemoryInMB")) return 350;
    if (!strcmp(n, "getPhysicalWidth")) return g_screen_w;
    if (!strcmp(n, "getPhysicalHeight")) return g_screen_h;
    if (!strcmp(n, "getModeId")) return 1;
    if (!strcmp(n, "startGetRequest")) return -1;              /* no HTTP */
    return 0;
}
static float java_answer_float(JObj *m) { return strstr(m->name, "RefreshRate") ? 60.0f : 0.0f; }
/* ApplicationUtilBase key-value store (SharedPreferences on the phone), kept in memory */
static struct { char k[128]; char v[512]; } g_kv[256]; static int g_nkv;
static const char *kv_get(const char *k) { for (int i = 0; i < g_nkv; i++) if (!strcmp(g_kv[i].k, k)) return g_kv[i].v; return NULL; }
static void kv_set(const char *k, const char *v) {
    for (int i = 0; i < g_nkv; i++) if (!strcmp(g_kv[i].k, k)) { snprintf(g_kv[i].v, 512, "%s", v); return; }
    if (g_nkv < 256) { snprintf(g_kv[g_nkv].k, 128, "%s", k); snprintf(g_kv[g_nkv].v, 512, "%s", v); g_nkv++; }
}

/* ---- JNIEnv implementations ---- */
static int32_t e_GetVersion(void *env) { (void)env; return 0x10006; }
static void *e_FindClass(void *env, const char *name) { (void)env; hit("FindClass", name, "", ""); return jclass(name); }
static void *e_ret_null(void) { return NULL; }
static int32_t e_ret0(void) { return 0; }
static void *e_ret_arg1(void *env, void *o) { (void)env; return o; }
static void *e_NewRef(void *env, void *o) { (void)env; jref(o); return o; }             /* New{Local,Global,WeakGlobal}Ref */
static void e_DeleteRef(void *env, void *o) { (void)env; jrelease(o); }                 /* Delete{Local,Global,WeakGlobal}Ref */
static int32_t e_IsSame(void *env, void *a, void *b) { (void)env; return a == b; }
static int32_t e_one(void) { return 1; }
static void *e_GetObjectClass(void *env, JObj *o) { (void)env; return jclass(isj(o) ? o->cls : "?"); }
static void *e_GetSuperclass(void *env, JObj *o) { (void)env; (void)o; return jclass("java/lang/Object"); }
static void *getmember(int kind, int st, JObj *c, const char *n, const char *s) {
    const char *cls = isj(c) ? c->cls : "?";
    hit(kind == K_METHOD ? (st ? "GetStaticMethodID" : "GetMethodID") : (st ? "GetStaticFieldID" : "GetFieldID"), cls, n, s);
    char nn[96], ss[256]; snprintf(nn, sizeof nn, "%s", n); snprintf(ss, sizeof ss, "%s", s);
    JLOCK; JObj *m = g_jni_free ? jcache_find(kind, cls, nn, ss, st) : NULL;
    if (!m) { m = jnew(kind, cls); snprintf(m->name, sizeof m->name, "%s", nn); snprintf(m->sig, sizeof m->sig, "%s", ss); m->is_static = st;
        if (g_jni_free) jcache_add(m); }
    JUNLOCK; return m;
}
static void *e_GetMethodID(void *env, JObj *c, const char *n, const char *s) { (void)env; return getmember(K_METHOD, 0, c, n, s); }
static void *e_GetStaticMethodID(void *env, JObj *c, const char *n, const char *s) { (void)env; return getmember(K_METHOD, 1, c, n, s); }
static void *e_GetFieldID(void *env, JObj *c, const char *n, const char *s) { (void)env; return getmember(K_FIELD, 0, c, n, s); }
static void *e_GetStaticFieldID(void *env, JObj *c, const char *n, const char *s) { (void)env; return getmember(K_FIELD, 1, c, n, s); }
static void callhit(JObj *m) { if (isj(m)) hit(m->is_static ? "CallStatic" : "Call", m->cls, m->name, m->sig); }
/* Java arguments, decoded from the signature for the three JNI call forms: (..., ...), (..., va_list), (..., jvalue*) */
typedef struct { uint64_t a[16]; int n; } JArgs;
static void parse_args(JObj *m, va_list *ap, const uint64_t *jv, JArgs *A) {
    A->n = 0; if (!isj(m)) return;
    const char *p = strchr(m->sig, '(');
    if (!p) return;
    for (p++; *p && *p != ')' && A->n < 16; ) {
        char t = *p; uint64_t v = 0;
        if (t == '[') { while (*p == '[') p++; if (*p == 'L') p = strchr(p, ';'); p++; t = 'L'; }
        else if (t == 'L') { p = strchr(p, ';') + 1; }
        else p++;
        if (jv) v = jv[A->n];
        else if (t == 'L') v = (uint64_t)va_arg(*ap, void *);
        else if (t == 'J') v = (uint64_t)va_arg(*ap, long);
        else if (t == 'F' || t == 'D') { double d = va_arg(*ap, double); memcpy(&v, &d, 8); }
        else v = (uint64_t)(int64_t)va_arg(*ap, int);
        A->a[A->n++] = v;
    }
}
static const char *jstr(uint64_t v) { JObj *o = (JObj *)v; return isj(o) && o->kind == K_STR ? o->str : NULL; }
static int cmpstr(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
/* AssetManager.list(path): the game discovers its built-in shader library (sc3d/shaders) and other folders this way */
static void *asset_list(const char *path) {
    char d[1024]; snprintf(d, sizeof d, "%s/assets/%s", g_asset_root, path);
    DIR *dir = opendir(d);
    char *names[8192]; int n = 0;
    if (dir) { struct dirent *e; while ((e = readdir(dir)) && n < 8192) if (e->d_name[0] != '.') names[n++] = strdup(e->d_name); closedir(dir); }
    qsort(names, n, sizeof names[0], cmpstr);
    JObj *a = jnew(K_ARR, "[Ljava/lang/String;"); a->len = n; a->elsz = 8; a->data = calloc(n + 1, 8);
    for (int i = 0; i < n; i++) { ((void **)a->data)[i] = jstr_new(names[i]); free(names[i]); }   /* refs 1: the slot */
    LOG("[jni] AssetManager.list(\"%s\") -> %d entries\n", path, n);
    return a;
}
static void *answer_obj(JObj *m, JArgs *A) {
    if (!isj(m)) return NULL;
    if (!strcmp(m->name, "list") && strstr(m->cls, "AssetManager") && A->n >= 1) return asset_list(jstr(A->a[0]) ? jstr(A->a[0]) : "");
    if (!strcmp(m->name, "getKeyValue") && A->n >= 1) {
        const char *k = jstr(A->a[0]), *v = k ? kv_get(k) : NULL;
        LOG("[jni] getKeyValue(%s) -> %s\n", k ? k : "?", v ? v : "null");
        return v ? jstr_new(v) : NULL;
    }
    return java_answer_obj(m);
}
static int64_t answer_int(JObj *m, JArgs *A) {
    if (!isj(m)) return 0;
    if (!strcmp(m->name, "storeKeyValue") && A->n >= 2 && jstr(A->a[0]) && jstr(A->a[1])) {
        kv_set(jstr(A->a[0]), jstr(A->a[1])); LOG("[jni] storeKeyValue(%s, %.80s)\n", jstr(A->a[0]), jstr(A->a[1])); return 0;
    }
    if (!strcmp(m->name, "removeKeyValue") && A->n >= 1 && jstr(A->a[0])) { kv_set(jstr(A->a[0]), ""); return 0; }
    return java_answer_int(m);
}
#define DEF_CALLS(SUF, RT, ANS)                                                                                         \
static RT c_##SUF(void *env, void *o, JObj *m, ...) { (void)env; (void)o; callhit(m); va_list ap; va_start(ap, m); JArgs A;  \
    parse_args(m, &ap, NULL, &A); va_end(ap); return (RT)ANS; }                                                         \
static RT c_##SUF##V(void *env, void *o, JObj *m, va_list ap) { (void)env; (void)o; callhit(m); va_list cp; va_copy(cp, ap); \
    JArgs A; parse_args(m, &cp, NULL, &A); va_end(cp); return (RT)ANS; }                                               \
static RT c_##SUF##A(void *env, void *o, JObj *m, const uint64_t *jv) { (void)env; (void)o; callhit(m); JArgs A;      \
    parse_args(m, NULL, jv, &A); return (RT)ANS; }                                                                      \
static RT cn_##SUF(void *env, void *o, void *c, JObj *m, ...) { (void)env; (void)o; (void)c; callhit(m); va_list ap;    \
    va_start(ap, m); JArgs A; parse_args(m, &ap, NULL, &A); va_end(ap); return (RT)ANS; }                               \
static RT cn_##SUF##V(void *env, void *o, void *c, JObj *m, va_list ap) { (void)env; (void)o; (void)c; callhit(m);      \
    va_list cp; va_copy(cp, ap); JArgs A; parse_args(m, &cp, NULL, &A); va_end(cp); return (RT)ANS; }                   \
static RT cn_##SUF##A(void *env, void *o, void *c, JObj *m, const uint64_t *jv) { (void)env; (void)o; (void)c;          \
    callhit(m); JArgs A; parse_args(m, NULL, jv, &A); return (RT)ANS; }
DEF_CALLS(obj, void *, answer_obj(m, &A))
DEF_CALLS(int, int64_t, answer_int(m, &A))
DEF_CALLS(float, float, (isj(m) ? java_answer_float(m) : 0))
DEF_CALLS(double, double, 0)
static void *e_NewObject(void *env, JObj *c, JObj *m, ...) { (void)env; callhit(m); return jnew(K_OBJ, isj(c) ? c->cls : "?"); }
static void *e_AllocObject(void *env, JObj *c) { (void)env; return jnew(K_OBJ, isj(c) ? c->cls : "?"); }
static void *f_get_obj(void *env, void *o, JObj *f) { (void)env; (void)o; if (isj(f)) hit("GetField", f->cls, f->name, f->sig);
    if (isj(f) && f->sig[0] == 'L') { if (!strcmp(f->sig, "Ljava/lang/String;")) return jstr_new(""); return janswer_obj(f->sig); } return NULL; }
static int64_t f_get_int(void *env, void *o, JObj *f) { (void)env; (void)o; if (isj(f)) hit("GetField", f->cls, f->name, f->sig); return 0; }
static void f_set(void *env, void *o, JObj *f) { (void)env; (void)o; if (isj(f)) hit("SetField", f->cls, f->name, f->sig); }
static void *e_NewStringUTF(void *env, const char *s) { (void)env; return jstr_new(s ? s : ""); }
static void *e_NewString(void *env, const uint16_t *u, int n) {
    (void)env; char *b = malloc(n * 3 + 1); int k = 0;
    for (int i = 0; i < n; i++) { if (u[i] < 0x80) b[k++] = u[i]; else b[k++] = '?'; }
    b[k] = 0; void *o = jstr_new(b); free(b); return o;
}
static int32_t e_GetStringLength(void *env, JObj *s) { (void)env; return isj(s) ? s->len : 0; }
static const char *e_GetStringUTFChars(void *env, JObj *s, uint8_t *copy) { (void)env; if (copy) *copy = 0; jpin(s); return isj(s) && s->str ? s->str : ""; }
static void e_ReleasePinned(void *env, void *o, void *p) { (void)env; (void)p; junpin(o); }          /* ReleaseStringUTFChars */
static void e_ReleaseArray(void *env, void *a, void *p, int mode) { (void)env; (void)p; if (mode != 1 /* JNI_COMMIT */) junpin(a); }   /* Release*ArrayElements, ReleasePrimitiveArrayCritical */
static void e_ReleaseStringChars(void *env, void *o, void *p) { (void)env; (void)o; if (g_jni_free) free(p); }   /* GetStringChars returned a copy */
static const uint16_t *e_GetStringChars(void *env, JObj *s, uint8_t *copy) {
    (void)env; if (copy) *copy = 1; int n = isj(s) ? s->len : 0; uint16_t *u = calloc(n + 1, 2);
    for (int i = 0; i < n; i++) u[i] = (uint8_t)s->str[i]; return u;
}
static void e_GetStringUTFRegion(void *env, JObj *s, int st, int n, char *buf) { (void)env; memcpy(buf, s->str + st, n); buf[n] = 0; }
static void e_GetStringRegion(void *env, JObj *s, int st, int n, uint16_t *buf) { (void)env; for (int i = 0; i < n; i++) buf[i] = (uint8_t)s->str[st + i]; }
static int32_t e_GetArrayLength(void *env, JObj *a) { (void)env; return isj(a) ? a->len : 0; }
static void *newarr(const char *t, int n, int el) { JObj *a = jnew(K_ARR, t); a->len = n; a->elsz = el; a->data = calloc(n + 1, el); return a; }
static void *e_NewByteArray(void *env, int n) { (void)env; return newarr("[B", n, 1); }
static void *e_NewIntArray(void *env, int n) { (void)env; return newarr("[I", n, 4); }
static void *e_NewLongArray(void *env, int n) { (void)env; return newarr("[J", n, 8); }
static void *e_NewFloatArray(void *env, int n) { (void)env; return newarr("[F", n, 4); }
static void *e_NewObjectArray(void *env, int n, JObj *c, void *init) {
    (void)env; JObj *a = newarr("[L", n, 8); for (int i = 0; i < n; i++) { ((void **)a->data)[i] = init; jref(init); } (void)c; return a;
}
static void *e_GetObjectArrayElement(void *env, JObj *a, int i) {   /* a new local reference to the element */
    (void)env; void *e = isj(a) && a->data && i < a->len ? ((void **)a->data)[i] : NULL; jref(e); return e;
}
static void e_SetObjectArrayElement(void *env, JObj *a, int i, void *v) {
    (void)env; if (!(isj(a) && a->data && i < a->len)) return;
    JLOCK; void *old = ((void **)a->data)[i]; jref(v); ((void **)a->data)[i] = v; if (isj(old)) jrelease(old); JUNLOCK;
}
static void *e_GetArrayElements(void *env, JObj *a, uint8_t *copy) { (void)env; if (copy) *copy = 0; jpin(a); return isj(a) ? a->data : NULL; }
static void e_GetArrayRegion(void *env, JObj *a, int st, int n, void *buf) { (void)env; if (isj(a) && a->data) memcpy(buf, a->data + st * a->elsz, n * a->elsz); }
static void e_SetArrayRegion(void *env, JObj *a, int st, int n, const void *buf) { (void)env; if (isj(a) && a->data) memcpy(a->data + st * a->elsz, buf, n * a->elsz); }
static int32_t e_RegisterNatives(void *env, JObj *c, const struct { const char *n, *s; void *f; } *m, int n) {
    (void)env;
    for (int i = 0; i < n; i++) {
        NativeReg *r = &g_jni_natives[g_jni_nnatives++];
        r->cls = strdup(isj(c) ? c->cls : "?"); r->name = strdup(m[i].n); r->sig = strdup(m[i].s); r->fn = m[i].f;
        LOG("[natives] %s.%s%s -> %s\n", r->cls, r->name, r->sig, addr_desc(r->fn));
    }
    return 0;
}
static int32_t e_GetJavaVM(void *env, void **vm) { (void)env; *vm = g_vm; return 0; }
static int32_t e_Throw(void *env, void *o) { (void)env; LOG("[jni] Throw %s\n", jni_obj_desc(o)); return 0; }
static int32_t e_ThrowNew(void *env, JObj *c, const char *m) { (void)env; LOG("[jni] ThrowNew %s: %s\n", jni_obj_desc(c), m); return 0; }
static void e_FatalError(void *env, const char *m) { (void)env; LOG("[jni] FatalError %s\n", m); abort(); }
static void *e_NewDirectByteBuffer(void *env, void *a, int64_t n) { (void)env; JObj *o = jnew(K_ARR, "java/nio/ByteBuffer"); o->data = a; o->len = n; o->elsz = 1; o->ext_data = 1; return o; }
static void *e_GetDirectBufferAddress(void *env, JObj *o) { (void)env; return isj(o) ? o->data : NULL; }
static int64_t e_GetDirectBufferCapacity(void *env, JObj *o) { (void)env; return isj(o) ? o->len : 0; }

static uint64_t jni_default(void *slot, uint64_t *x, double *d, void *lr) {
    (void)d; int i = (int)(uintptr_t)slot;
    char a[64]; snprintf(a, sizeof a, "%s", jni_obj_desc((void *)x[1]));
    hit("UNIMPLEMENTED", jni_slot_names[i], "", "");
    LOG("[jni] unimplemented slot %d %s(%s) from %s\n", i, jni_slot_names[i], a, addr_desc(lr));
    return 0;
}
static void set(const char *name, void *fn) {
    for (int i = 0; i < 235; i++) if (!strcmp(jni_slot_names[i], name)) { g_env_table[i] = fn; return; }
    LOG("[jni] no slot %s\n", name); exit(1);
}
static void set_calls(const char *prefix, void *const o[3], void *const i[3], void *const f[3], void *const d[3]) {
    static const char *T[] = {"Object", "Boolean", "Byte", "Char", "Short", "Int", "Long", "Float", "Double", "Void"};
    static const char *V[] = {"", "V", "A"};
    for (int t = 0; t < 10; t++) for (int v = 0; v < 3; v++) {
        char n[64]; snprintf(n, sizeof n, "%s%sMethod%s", prefix, T[t], V[v]);
        set(n, t == 0 ? o[v] : t == 7 ? f[v] : t == 8 ? d[v] : i[v]);
    }
}
static void set_fields(const char *get, const char *setp) {
    static const char *T[] = {"Object", "Boolean", "Byte", "Char", "Short", "Int", "Long", "Float", "Double"};
    for (int t = 0; t < 9; t++) {
        char n[64]; snprintf(n, sizeof n, "%s%sField", get, T[t]); set(n, t == 0 ? (void *)f_get_obj : (void *)f_get_int);
        snprintf(n, sizeof n, "%s%sField", setp, T[t]); set(n, (void *)f_set);
    }
}
static void set_arrays(void) {
    static const char *T[] = {"Boolean", "Byte", "Char", "Short", "Int", "Long", "Float", "Double"};
    for (int t = 0; t < 8; t++) {
        char n[64];
        snprintf(n, sizeof n, "Get%sArrayElements", T[t]); set(n, e_GetArrayElements);
        snprintf(n, sizeof n, "Release%sArrayElements", T[t]); set(n, e_ReleaseArray);
        snprintf(n, sizeof n, "Get%sArrayRegion", T[t]); set(n, e_GetArrayRegion);
        snprintf(n, sizeof n, "Set%sArrayRegion", T[t]); set(n, e_SetArrayRegion);
    }
}

/* JavaVM */
static int32_t vm_GetEnv(void *vm, void **env, int v) { (void)vm; (void)v; *env = g_env; return 0; }
static int32_t vm_Attach(void *vm, void **env, void *a) { (void)vm; (void)a; *env = g_env; return 0; }
static int32_t vm_ok(void) { return 0; }

__attribute__((constructor)) static void jni_init(void) {
    for (int i = 0; i < 235; i++) g_env_table[i] = make_thunk(jni_default, (void *)(uintptr_t)i);
    g_env[0] = g_env_table;
    set("GetVersion", e_GetVersion); set("FindClass", e_FindClass); set("GetSuperclass", e_GetSuperclass);
    set("IsAssignableFrom", e_one); set("ExceptionOccurred", e_ret_null); set("ExceptionDescribe", e_ret0);
    set("ExceptionClear", e_ret0); set("ExceptionCheck", e_ret0); set("Throw", e_Throw); set("ThrowNew", e_ThrowNew);
    set("FatalError", e_FatalError); set("PushLocalFrame", e_ret0); set("PopLocalFrame", e_ret_arg1);
    set("NewGlobalRef", e_NewRef); set("DeleteGlobalRef", e_DeleteRef); set("DeleteLocalRef", e_DeleteRef); set("IsSameObject", e_IsSame);
    set("NewLocalRef", e_NewRef); set("EnsureLocalCapacity", e_ret0); set("AllocObject", e_AllocObject);
    set("NewObject", e_NewObject); set("NewObjectV", e_NewObject); set("NewObjectA", e_NewObject);
    set("GetObjectClass", e_GetObjectClass); set("IsInstanceOf", e_one);
    set("GetMethodID", e_GetMethodID); set("GetStaticMethodID", e_GetStaticMethodID);
    set("GetFieldID", e_GetFieldID); set("GetStaticFieldID", e_GetStaticFieldID);
    { void *const O[3] = {c_obj, c_objV, c_objA}, *const I[3] = {c_int, c_intV, c_intA}, *const F[3] = {c_float, c_floatV, c_floatA},
                   *const D[3] = {c_double, c_doubleV, c_doubleA};
      void *const NO[3] = {cn_obj, cn_objV, cn_objA}, *const NI[3] = {cn_int, cn_intV, cn_intA}, *const NF[3] = {cn_float, cn_floatV, cn_floatA},
                   *const ND[3] = {cn_double, cn_doubleV, cn_doubleA};
      set_calls("Call", O, I, F, D); set_calls("CallStatic", O, I, F, D); set_calls("CallNonvirtual", NO, NI, NF, ND); }
    set_fields("Get", "Set"); set_fields("GetStatic", "SetStatic");
    set("NewString", e_NewString); set("GetStringLength", e_GetStringLength); set("GetStringChars", e_GetStringChars);
    set("ReleaseStringChars", e_ReleaseStringChars); set("NewStringUTF", e_NewStringUTF); set("GetStringUTFLength", e_GetStringLength);
    set("GetStringUTFChars", e_GetStringUTFChars); set("ReleaseStringUTFChars", e_ReleasePinned);
    set("GetStringUTFRegion", e_GetStringUTFRegion); set("GetStringRegion", e_GetStringRegion);
    set("GetStringCritical", e_GetStringChars); set("ReleaseStringCritical", e_ReleaseStringChars);
    set("GetArrayLength", e_GetArrayLength); set("NewObjectArray", e_NewObjectArray);
    set("GetObjectArrayElement", e_GetObjectArrayElement); set("SetObjectArrayElement", e_SetObjectArrayElement);
    set("NewBooleanArray", e_NewByteArray); set("NewByteArray", e_NewByteArray); set("NewIntArray", e_NewIntArray);
    set("NewLongArray", e_NewLongArray); set("NewFloatArray", e_NewFloatArray); set_arrays();
    set("GetPrimitiveArrayCritical", e_GetArrayElements); set("ReleasePrimitiveArrayCritical", e_ReleaseArray);
    set("RegisterNatives", e_RegisterNatives); set("UnregisterNatives", e_ret0);
    set("MonitorEnter", e_ret0); set("MonitorExit", e_ret0); set("GetJavaVM", e_GetJavaVM);
    set("NewWeakGlobalRef", e_NewRef); set("DeleteWeakGlobalRef", e_DeleteRef);
    set("NewDirectByteBuffer", e_NewDirectByteBuffer); set("GetDirectBufferAddress", e_GetDirectBufferAddress);
    set("GetDirectBufferCapacity", e_GetDirectBufferCapacity); set("GetObjectRefType", e_one);
    g_vm_table[3] = vm_ok; g_vm_table[4] = vm_Attach; g_vm_table[5] = vm_ok; g_vm_table[6] = vm_GetEnv; g_vm_table[7] = vm_Attach;
    g_vm[0] = g_vm_table;
}
