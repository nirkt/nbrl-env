/* nbenv: the RL environment inside nbhost (one battle per process), driven over shared memory (nbenv_proto.h).
 * C port of what the phone's Frida agent (inputs/agent/v68-rl-candidate.js) does for the RL env:
 *   - snap2 state dump (byte-identical layout, state2.py parses it unchanged) + map dump (snap_map)
 *   - env_step: actions applied as the sim's own ClientInputs through the native apply function 0xc2cef4 (the
 *     function the per-frame input drain 0xc2ce28 calls; it ends in the controller calls move 0xb4790c, fire
 *     0xb487c4 via 0xb465c8, stop 0xb4103c) at the step boundary = right before the step's first tick, i.e. where
 *     the phone's drain applied them; k ticks per step, several steps per client frame; any player index.
 *   - Showdown rules: storm damage (20 %/s of max HP, +5 %/s after 5 s, applied before the snapshot of the tick after
 *     the check, through the damage root 0xb422d0), no regen in gas, elimination tracker / Brawlers left / native
 *     end at <= 1 alive, power cubes (+400 HP, +10 % damage incl. the popup), damage model (P11 x2 + travel curve),
 *     super-charge fix, Piper's leap re-implementation + BoxOfBombs + bomb damage, skill block in the air, client
 *     guard, alive guard, bot 'hold'/native control, local_ai.
 *   - soft reset: StartLoading clone re-injected through 0x816ff8 with the seed (+ the B3 determinism patch).
 * Every agent mechanism keeps the agent's numbers; differences are listed in work/NOTES.md. */
#include "nbhost.h"
#include "nbenv_proto.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <malloc.h>

extern Module *g_libg;
void *hook_replace(Module *m, uint64_t off, void *repl);
void hook_bl(Module *m, uint64_t off, void *fn);
void hook_enter(Module *m, uint64_t off, void (*fn)(uint64_t *ctx));
int hook_null_skip(Module *m, uint64_t off, uint32_t expect, int reg, uint64_t skip_to);
#include <poll.h>
/* bell pipes (--cmd-fd N --resp-fd M, inherited from the Python side): one byte per new command / per published response,
 * so both sides can block instead of spinning (the shared-memory protocol itself is unchanged) */
int g_cmd_fd = -1, g_resp_fd = -1, g_bell_spin_us = 200;
static long g_acc694_calls, g_ax_hits;   /* registry lookups (f_acc694) and the ones answered by the build index */
static void bell_ring(void) { if (g_resp_fd >= 0) { char b = 1; if (write(g_resp_fd, &b, 1) < 0) { } } }
int hook_flag_skip(Module *m, uint64_t off, uint32_t expect, volatile int *flag, uint64_t skip_to);
void patch_u32(Module *m, uint64_t off, uint32_t v);
#define B(off) ((uint8_t *)g_libg->base + (off))
#define I32(p, o) (*(volatile int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(volatile uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void *volatile *)((uint8_t *)(p) + (o)))
#define NIL ((int32_t)0x80000000)
enum { VT_CHAR = 0x1209020, VT_CHAR2 = 0x1208d58, VT_ITEM = 0x1209240, VT_ITEM2 = 0x1208ca8, VT_PROJ = 0x1209318 };
#define VTOF(o) ((int)((uint8_t *)PTR(o, 0) - g_libg->base))
#define BOX_ID 16000120
#define CUBE_ITEM 18000009
#define BOMB_ITEM 18000035

int g_env_on;                 /* --env PATH given */
int g_env_spf = 4;            /* --steps-per-frame (client mode: state-only steps run per client frame) */
int g_env_simonly = 1;        /* default: state-only episodes on a simulation built per reset (no client); --env-client */
int g_build_list[64], g_build_nlist;
static void *g_real_battle; static int g_play_ready, g_in_loop;
static uint64_t g_joy_args[4]; static long g_joy_calls;
void prof_set_thread(int), prof_begin(void), prof_dump(const char *);
static NbeHdr *H; static NbeCmd *CMD; static NbeResp *R; static int32_t *ST, *MAPB, *RECB; static uint8_t *PIXB; static float *OBSB;
static uint32_t g_served;     /* last cmd_seq served */
static double tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
extern double g_t0;
extern void *g_sim; extern int g_battles;

/* ------------------------------------------------------------------ native functions */
typedef uint64_t (*fn_b422d0)(void *, int, int, int, int, void *, void *, int, int, int, void *, int, int, int, int, int, int, int);
typedef void (*fn_cef4)(void *, void *, int, int);
static fn_b422d0 o_root;                          /* trampoline: calls the original damage root (no env bookkeeping) */
#define RAYCAST(fx, fy, tx, ty, map, out, fl) ((int (*)(int, int, int, int, void *, int *, int, int, int, int))B(0xc6a91c))(fx, fy, tx, ty, map, out, fl, 0, 0, 1)
#define ISQRT(v) ((int (*)(int))B(0xd00730))(v)
#define SKILL_ATTACK(ch) ((void *(*)(void *))B(0xb35c00))(ch)
#define SKILL_SUPER(ch) ((void *(*)(void *))B(0xb38fe8))(ch)
#define SKILL_ABS(sd) (((int (*)(void *))B(0xb0dfc8))(sd) & 1)
#define CAN_STOP(ch) (((int (*)(void *))B(0xb50ac8))(ch) & 1)
#define MOVING(ch) (((int (*)(void *))B(0xb393f0))(ch) & 1)
#define STOP(ch) ((void (*)(void *))B(0xb4103c))(ch)
#define LERP(a, b, t) ((int (*)(int, int, int))B(0xc70afc))(a, b, t)
#define CHAR_FLAG31E20(ch) (((int (*)(void *))B(0xb31e20))(ch) & 1)

/* ------------------------------------------------------------------ battle / episode state */
#define MAXOBJ 4096
typedef struct { int id, data, owner, x, y; } ItemRow;
typedef struct { int id; void *ch; int x0, y0, x1, y1, T, t, start; } Leap;
typedef struct { int id, x, y, owner; } Bomb;
typedef struct { int x, y, owner, tick; } PendBox;
typedef struct { int kind; int a, b, c, d; void *p; } TickJob;   /* 1 spawn box, 2 explode */
typedef struct { int id, dmg, pct, secs10, maxhp; int px, py; } GasHit;

static struct {
    void *battle, *sim; int gen;
    int depth, presnap_tick;
    /* episode control */
    int active, intro, act_tick, reset_req, reset_seed, have_reset, gas_on;
    int step_left, cur_flags, publish_frame, steps_in_frame, frame, started_tick;
    NbeCmd cur;                         /* the command being executed */
    double t_step0, t_inject, t_build0, t_build1, t_tick1, t_del;
    /* per-player move state (the agent's IT/CN joystick bookkeeping) */
    int moving[NBE_MAXP], last_dir[NBE_MAXP], sent_valid[NBE_MAXP], sent_x[NBE_MAXP], sent_y[NBE_MAXP];
    int hx[NBE_MAXP][8], hy[NBE_MAXP][8], hid[NBE_MAXP], hn[NBE_MAXP];   /* per-tick positions (client-lag model) */
    /* step accumulators */
    int dealt[NBE_MAXP], taken[NBE_MAXP], gasd[NBE_MAXP], kills[NBE_MAXP], hits[NBE_MAXP], boxd[NBE_MAXP], bombd[NBE_MAXP], gast[NBE_MAXP], ingas[NBE_MAXP], cubes0[NBE_MAXP];
    /* elimination tracker (EliminationTracker) */
    int step_start;   /* tick at the start of the current step (frame) */
    int gas_j, gas_phase, gas_latched, gas_latches;   /* storm clock (phone fog-renderer latch, see gas_test_step) */
    struct { int n; int super[2]; double deg[2], dist[2]; } pf[NBE_MAXP];   /* attack commands of the last step (g_fire_delay) */
    long cl_moves, cl_attacks;
    int seen[NBE_MAXP], elim[NBE_MAXP], erank[NBE_MAXP], etick[NBE_MAXP], egas[NBE_MAXP], eptr_n, total, order_n, end_req, winner, local_rank, alive;
    void *elim_ptr[NBE_MAXP];
    /* cubes: by character id */
    int cube_id[64], cube_n[64], ncube;
    int consumed[1024], nconsumed;
    ItemRow items[512]; int nitems;
    /* gas */
    int gas_count, gas_since_valid[NBE_MAXP], gas_since[NBE_MAXP]; GasHit gq[NBE_MAXP]; int ngq;
    int gas_lethal_id[NBE_MAXP], gas_lethal_tick[NBE_MAXP];
    int gas_args_ok, ga4, ga5, ga7, gs0;   /* renderer's 0xc69094 arguments (constant per frame except the step) */
    long gas_native_checks, gas_formula_mismatch;
    /* leap (SF) */
    Leap jobs[NBE_MAXP]; int njobs; int sf_started, sf_landed, sf_blocked, sf_ignored, sf_boxes, sf_pops;
    int sf_owner_id[NBE_MAXP], sf_owner_dmg[NBE_MAXP];
    PendBox pbox[32]; int npbox; int last_bomb_tick;
    Bomb bombs[64]; int nbombs;
    TickJob tj[64]; int ntj;
    /* damage model / charge */
    int charge_level;   /* >0 while a root call with a level-scaled character attacker is in progress */
    long dm_applied, charge_fixes, guard_hits, alive_forced, root_calls, events;
    int in_own_call;    /* 1 = gas, 2 = bomb (our own calls to the damage root) */
    int seed_applied;
} E;

static int32_t g_cube_tbl_dummy;
static uint32_t g_hold_mask, g_native_mask = 0xffffffffu, g_ctl_mask;
static NbeCmd g_setup; static int g_setup_pending;
static float g_cam_zoom_arg = -1.0f;   /* expertbot round: last zoom argument of the battle camera (published in reserved[7]) */

/* ------------------------------------------------------------------ helpers */
static void **objvec(void *sim, int *n) { void *h = PTR(sim, 0); if (!h) { *n = 0; return NULL; } *n = I32(h, 0xc); return (void **)PTR(h, 0); }
static void *obj_by_id(void *sim, int id) {
    int n; void **v = objvec(sim, &n);
    if (n < 0 || n > MAXOBJ || !v) return NULL;
    for (int i = 0; i < n; i++) if (v[i] && I32(v[i], 8) == id) return v[i];
    return NULL;
}
static void **players(void *sim, int *n) { *n = I32(sim, 0x7c); if (*n < 0 || *n > 64) *n = 0; return (void **)PTR(sim, 0x70); }
static int player_char_id(void *p) { int c = I32(p, 0x1e4); return c ? c : I32(p, 0x10); }
static int local_index(void) {
    if (!E.battle) return -1; void *c = PTR(E.battle, 0x28); return c ? I32(c, 0xe0) : -1;
}
static void *client_mgr(void) {
    if (g_env_simonly || !E.battle) return NULL; void *c = PTR(E.battle, 0x28); return c ? PTR(c, 0x28) : NULL;
}
static int jround(double v) { return (int)floor(v + 0.5); }   /* JavaScript Math.round */
static int cubes_of(int id) { for (int i = 0; i < E.ncube; i++) if (E.cube_id[i] == id) return E.cube_n[i]; return 0; }
static void cubes_add(int id) {
    for (int i = 0; i < E.ncube; i++) if (E.cube_id[i] == id) { E.cube_n[i]++; return; }
    if (E.ncube < 64) { E.cube_id[E.ncube] = id; E.cube_n[E.ncube++] = 1; }
}
static int tile_blocks_move(int id) {  /* SF_BLOCK (tiles.csv BlocksMovement rows) */
    static const int rows[] = {1,2,3,4,6,9,10,11,12,13,14,15,16,18,20,21,39,40,41,42,43,44,45,46,53,54,55,56,57,58,59,60,61,62,63,65,66,69,70,71,72,73};
    for (unsigned i = 0; i < sizeof rows / sizeof *rows; i++) if (id == 27000000 + rows[i]) return 1;
    return 0;
}
static int map_wh(void *sim, int *W, int *Hh) { void *m = PTR(sim, 0x68); if (!m) return 0; *W = I32(m, 0xc4); *Hh = I32(m, 0xc8); return 1; }
static int walkable(void *sim, int x, int y) {
    void *m = PTR(sim, 0x68); if (!m) return 1;
    int W = I32(m, 0xc4), Hh = I32(m, 0xc8), tx = (int)floor(x / 300.0), ty = (int)floor(y / 300.0);
    if (tx < 0 || ty < 0 || tx >= W || ty >= Hh) return 0;
    void *t = ((void **)PTR(m, 0x20))[tx + W * ty]; if (!t) return 1;
    void *d = PTR(t, 8); return !d || !tile_blocks_move(I32(d, 0x20));
}

/* ------------------------------------------------------------------ snap2 (the agent's CModule, verbatim logic) */
#define S2HDR 8
#define P2 8
#define C2 23
#define J2 18
#define I2 12
static int s2_skill_word(uint8_t *ch, int idx, int off) {
    uint8_t *arr = PTR(ch, 0x458); int n = I32(ch, 0x464);
    if (!arr || idx >= n || n > 16) return NIL;
    uint8_t *sk = ((uint8_t **)arr)[idx];
    return sk ? I32(sk, off) : NIL;
}
static void *s2_client_obj(void *mgr, int id) {
    if (!mgr) return NULL; void **a = PTR(mgr, 0); int n = I32(mgr, 0xc);
    if (n < 0 || n > 4096 || (n && !a)) return NULL;
    for (int i = 0; i < n; i++) if (a[i] && I32(a[i], 8) == id) return a[i];
    return NULL;
}
static int s2_in_client(void *mgr, int id) {
    if (!mgr) return -1; void **a = PTR(mgr, 0); int n = I32(mgr, 0xc);
    if (n < 0 || n > 4096 || (n && !a)) return -1;
    for (int i = 0; i < n; i++) if (a[i] && I32(a[i], 8) == id) return 1;
    return 0;
}
static void s2_client_pos(void *mgr, int id, int32_t *r) {
    void *o = s2_client_obj(mgr, id);
    if (!o) { r[0] = r[1] = r[2] = NIL; return; }
    r[0] = I32(o, 0x30); r[1] = I32(o, 0x34); r[2] = I32(o, 0x38);
}
int snap2(void *sim, int32_t *out, int cap, void *cmgr) {
    int n, c, np, k, nch = 0, npj = 0, nit = 0;
    if (!sim) return -1;
    out[0] = 0x4e423253; out[2] = I32(sim, 0x10); out[7] = 0;
    k = S2HDR;
    void **p = PTR(sim, 0x70); c = I32(sim, 0x78); n = I32(sim, 0x7c);
    if (n < 0 || c < n || c > 64 || (n && !p)) return -2;
    np = n;
    if (k + np * P2 > cap) return -3;
    for (int i = 0; i < np; i++) {
        uint8_t *pl = p[i]; int32_t *r = out + k; k += P2;
        for (int j = 0; j < P2; j++) r[j] = NIL;
        if (!pl) continue;
        r[0] = I32(pl, 8); r[1] = I32(pl, 12); r[2] = I32(pl, 0x10); r[3] = I32(pl, 0x1e4);
        r[4] = I32(pl, 0x54); r[5] = I32(pl, 0x58); r[6] = U8(pl, 0x48); r[7] = U8(pl, 0x238);
    }
    void *h = PTR(sim, 0); void **o = h ? PTR(h, 0) : NULL; c = h ? I32(h, 8) : 0; n = h ? I32(h, 12) : 0;
    if (!h || n < 0 || c < n || c > 4096 || (n && !o)) return -4;
    for (int i = 0; i < n; i++) {
        uint8_t *ob = o[i]; if (!ob) continue; int vo = VTOF(ob);
        if (vo != VT_CHAR && vo != VT_CHAR2) continue;
        if (k + C2 > cap) return -5;
        int32_t *r = out + k; k += C2; nch++;
        uint8_t *d = PTR(ob, 0x10);
        r[0] = I32(ob, 8); r[1] = d ? I32(d, 0x20) : NIL; r[2] = I32(ob, 0x80);
        r[3] = I32(ob, 0x38); r[4] = I32(ob, 0x3c); r[5] = I32(ob, 0x40);
        if (vo == VT_CHAR) {
            r[6] = I32(ob, 0x3c4); r[7] = I32(ob, 0x3c8); r[8] = I32(ob, 0x374); r[9] = I32(ob, 0x328);
            r[10] = I32(ob, 0x7d8); r[11] = U8(ob, 0x431); r[12] = U8(ob, 0x469);
            r[13] = s2_skill_word(ob, 0, 0x18); r[14] = s2_skill_word(ob, 0, 0x10);
            r[15] = s2_skill_word(ob, 1, 0x30) == NIL ? NIL : (s2_skill_word(ob, 1, 0x30) & 0xff);
            r[16] = I32(ob, 0xaa0);
        } else for (int j = 6; j < 17; j++) r[j] = NIL;
        r[17] = d ? I32(d, 0x23c) : NIL; r[18] = vo == VT_CHAR ? 0 : 1; r[19] = s2_in_client(cmgr, r[0]);
        s2_client_pos(cmgr, r[0], r + 20);
    }
    for (int i = 0; i < n; i++) {
        uint8_t *ob = o[i]; if (!ob) continue; int vo = VTOF(ob);
        if (vo != VT_PROJ) continue;
        if (k + J2 > cap) return -6;
        int32_t *r = out + k; k += J2; npj++;
        uint8_t *d = PTR(ob, 0x10);
        r[0] = I32(ob, 8); r[1] = d ? I32(d, 0x20) : NIL; r[2] = I32(ob, 0x80);
        r[3] = I32(ob, 0x38); r[4] = I32(ob, 0x3c); r[5] = I32(ob, 0x40);
        r[6] = I32(ob, 0x9c); r[7] = I32(ob, 0xa0);
        void *src = PTR(ob, 0xe0); r[8] = NIL;
        if (src) for (int j = 0; j < n; j++) if (o[j] == src) { r[8] = I32(src, 8); break; }
        r[9] = I32(ob, 0xd4); r[10] = I32(ob, 0x144); r[11] = I32(ob, 0x148); r[12] = I32(ob, 0x90); r[13] = I32(ob, 0x114);
        r[14] = s2_in_client(cmgr, r[0]); s2_client_pos(cmgr, r[0], r + 15);
    }
    for (int i = 0; i < n; i++) {
        uint8_t *ob = o[i]; if (!ob) continue; int vo = VTOF(ob);
        if (vo != VT_ITEM && vo != VT_ITEM2) continue;
        if (k + I2 > cap) return -7;
        int32_t *r = out + k; k += I2; nit++;
        uint8_t *d = PTR(ob, 0x10);
        r[0] = I32(ob, 8); r[1] = d ? I32(d, 0x20) : NIL; r[2] = I32(ob, 0x80);
        r[3] = I32(ob, 0x38); r[4] = I32(ob, 0x3c); r[5] = I32(ob, 0x40);
        r[6] = d ? I32(d, 0x54) : NIL; r[7] = vo == VT_ITEM ? 0 : 1; r[8] = s2_in_client(cmgr, r[0]); s2_client_pos(cmgr, r[0], r + 9);
    }
    out[1] = k; out[3] = np; out[4] = nch; out[5] = npj; out[6] = nit;
    return k;
}
/* map tile data ids (x + width*y) + FNV-1a hash (the agent's snap_map) */
static uint32_t snap_map(void *sim, int32_t *out, int cap, int *wh) {
    uint32_t hash = 2166136261u; wh[0] = wh[1] = 0;
    void *m = sim ? PTR(sim, 0x68) : NULL; if (!m) return 0;
    int w = I32(m, 0xc4), hh = I32(m, 0xc8);
    if (w < 1 || hh < 1 || w * hh > 16384 || w * hh > cap) return 0;
    void **tiles = PTR(m, 0x20); if (!tiles) return 0;
    for (int i = 0; i < w * hh; i++) {
        void *t = tiles[i], *d = t ? PTR(t, 8) : NULL; int id = d ? I32(d, 0x20) : NIL;
        out[i] = id; hash = (hash ^ (uint32_t)id) * 16777619u;
    }
    wh[0] = w; wh[1] = hh;
    return hash ? hash : 1;
}

/* ------------------------------------------------------------------ damage root: damage model, cubes, events */
static int proj_pct(uint8_t *src) {
    if (!src || VTOF(src) != VT_PROJ) return -1;
    uint8_t *d = PTR(src, 0x10); if (!d) return -1;
    int s0 = I32(d, 0x1bc), e0 = I32(d, 0x1c0); if (!s0) s0 = 100; if (!e0) e0 = 100;
    if (s0 == 100 && e0 == 100) return 100;
    int ps = I32(d, 0x1c4), pe = I32(d, 0x1c8), span = pe - ps; if (span < 1) span = 1;
    int tr = I32(src, 0x144) - I32(src, 0x148); if (tr < 0) tr = 0; if (tr > 1000) tr = 1000;
    int t = tr - ps; t = t > 0 ? (int)floor((double)(t < span ? t : span) * 1000 / span) : 0;
    return LERP(s0, e0, t);
}
static void record_event(uint8_t *victim, uint8_t *attacker, int before, int after) {
    int removed = before - after; if (removed <= 0) return;
    int vi = I32(victim, 0x80), known = attacker && VTOF(attacker) == VT_CHAR, ai = known ? I32(attacker, 0x80) : -999;
    int lethal = before > 0 && after <= 0;
    E.events++;
    if (E.in_own_call == 2) {   /* leap bombs: the phone applies them from JS (no HP-store listener): not in dealt/taken */
        if (known && ai >= 0 && ai < NBE_MAXP && vi != ai) E.bombd[ai] += removed;
        return;
    }
    if (E.in_own_call == 1) return;   /* storm: accounted from the storm's own records (gas_damage) like the agent */
    if (known && ai >= 0 && ai < NBE_MAXP && vi != ai) {
        if (vi < 0) E.boxd[ai] += removed;
        else { E.dealt[ai] += removed; E.hits[ai]++; if (lethal) E.kills[ai]++; }
    }
    if (vi >= 0 && vi < NBE_MAXP && known) E.taken[vi] += removed;
}
static uint64_t my_root(void *victim, int a1, int dmg, int a3, int a4, void *attacker, void *src, int a7,
                        int s0, int s1, void *s2, int s3, int s4, int s5, int s6, int s7, int s8, int s9) {
    int mode3 = I32(g_libg->base, 0x12eefb4) == 3;
    E.root_calls++;
    int saved_level = E.charge_level; E.charge_level = 0;
    if (mode3 && !E.in_own_call && attacker) {   /* agent: entryScale (attackerArg 5) + dmgModelApply */
        int raw = dmg, d = raw, aid = I32(attacker, 8), cubes = cubes_of(aid), lvl = 0;
        if (raw > 0) {
            int pct = proj_pct(src);
            if (pct >= 0 && pct != 100) d = (int)trunc((double)d * pct / 100);
            if (VTOF(attacker) == VT_CHAR) { d = jround(d * 2.0); lvl = 2; }
            if (cubes > 0 || d != raw) {
                int scaled = cubes > 0 ? jround(d * (1 + 0.10 * cubes)) : d;
                if (d != raw && cubes == 0) { if (a3 == raw) a3 = scaled; dmg = scaled; }
                else { dmg = scaled; a3 = scaled; }
                E.dm_applied++;
            }
        }
        E.charge_level = lvl;
    }
    int vchar = victim && VTOF(victim) == VT_CHAR, before = vchar ? I32(victim, 0x3c4) : 0;
    uint64_t r = o_root(victim, a1, dmg, a3, a4, attacker, src, a7, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9);
    if (vchar && mode3) record_event(victim, attacker, before, I32(victim, 0x3c4));
    E.charge_level = saved_level;
    return r;
}
/* super charge 0xc420d8(player, damage, ...): the root's calls charge for the level-scaled damage; the real game's
 * charge does not depend on the power level: divide by it (agent: only from the root's call sites) */
typedef uint64_t (*fn6)(void *, int, int, int, void *, int);
static fn6 o_charge;
static uint64_t my_charge(void *p, int d, int a2, int a3, void *a4, int a5) {
    uint64_t ra = (uint64_t)__builtin_return_address(0) - (uint64_t)g_libg->base;
    if (E.charge_level > 0 && (ra == 0xb4328c || ra == 0xb43300 || ra == 0xb43498) && d > 0) { d = jround((double)d / E.charge_level); E.charge_fixes++; }
    return o_charge(p, d, a2, a3, a4, a5);
}

/* ------------------------------------------------------------------ storm */
static int gas_step(int tick) { int t = tick - 440; return t > 0 ? t / 100 : 0; }
typedef int (*fn_gas)(int, int, int, int, int, int, int *, int, int);
static fn_gas o_gastest;
/* The phone agent's storm test used the arguments of the client's fog renderer (0xc69094 called by 0x86aca4), whose
 * storm step comes from the client's render tick: it lags the simulation (the client draws ~1.7 ticks behind and its
 * fog update of a lockstep frame runs after that frame's ticks), i.e. inside a step that started at tick S the phone's
 * storm step was the one of tick S-2. At the ring changes (ticks 540, 640, ..., which are also storm-check ticks) the
 * phone therefore checked the previous ring once more. g_gas_tick_lag reproduces that (0 = exact sim clock). */
int g_gas_render_lag = 2;
/* T6 (phone episodes with the per-step probe, runs/t6/phone_native_probe): the agent's storm clock was 'render' in every
 * harness episode (its sim-clock switch needs validation samples taken at native speed; 0 were taken), i.e. the storm
 * step was the fog renderer's LATCHED argument: the fog update 0x86aca4 -> 0xc69094 runs only every 8th lockstep frame
 * (= every 8 env steps, 32 ticks), with gas_step(client tick) where the client tick is 1 behind the step start; the phase
 * of the 8-frame cycle differs per episode (it continues from the intro's frames). Model (g_gas_model 1): at the start of
 * env step j (j = 1, 2, ... after the reset observation) with (j - phase) % 8 == 0 the latched step becomes
 * gas_step(S - 1), S = the tick before the step's first tick; phase = RESET reserved[0] - 1, else from the seed.
 * A STEP with gas = k + 1 forces latched step k for that step (exact replay of a recorded phone schedule).
 * g_gas_model 0 = the earlier model (step of S - g_gas_render_lag). */
int g_gas_model = 1;
static int gas_test_step(void *sim) {
    if (g_gas_model == 1) return E.gas_latched;
    return gas_step(E.step_start >= 0 ? E.step_start - g_gas_render_lag : I32(sim, 0x10) - g_gas_render_lag);
}
static void gas_latch_reset(int S) {   /* at the reset observation's step */
    int ph = E.cur.reserved[0];
    E.gas_phase = ph > 0 ? (ph - 1) & 7 : (int)(((uint32_t)E.reset_seed * 2654435761u) >> 29);
    E.gas_j = 0; E.gas_latched = gas_step(S - 1); E.gas_latches = 0;
}
static void gas_latch_step(int S, int override) {   /* at the start of every env step */
    E.gas_j++;
    if (override > 0) { E.gas_latched = override - 1; return; }
    if (((E.gas_j - E.gas_phase) & 7) == 0) { E.gas_latched = gas_step(S - 1); E.gas_latches++; }
}
static int in_gas_tile(void *sim, int tx, int ty) {
    int W, Hh; if (!map_wh(sim, &W, &Hh)) return 0;
    int step = gas_test_step(sim);
    int formula = tx < 0 || ty < 0 || tx >= W || ty >= Hh ? 1 : ((tx < ty ? tx : ty) < step || (W - 1 - tx < Hh - 1 - ty ? W - 1 - tx : Hh - 1 - ty) < step);
    if (!E.gas_args_ok) return formula;
    int r = o_gastest(tx, ty, step, step, E.ga4, E.ga5, NULL, E.ga7, E.gs0) & 1;   /* the native per-tile test */
    E.gas_native_checks++; if (r != formula) E.gas_formula_mismatch++;
    return r;
}
/* the fog renderer's call 0x86afe8: bl 0xc69094 -> capture its per-frame arguments once */
static int my_gastest_call(int tx, int ty, int a2, int a3, int a4, int a5, int *a6, int a7, int s0) {
    if (!E.gas_args_ok) { E.ga4 = a4; E.ga5 = a5; E.ga7 = a7; E.gs0 = s0; E.gas_args_ok = 1; LOG("[env] storm test args captured: a4=%d a5=%d a7=%d s0=%d\n", a4, a5, a7, s0); }
    return o_gastest(tx, ty, a2, a3, a4, a5, a6, a7, s0);
}
static void gas_hit(void *sim, uint8_t *victim, GasHit *g) {   /* applyGasDamageNative */
    if (!victim || g->dmg <= 0) return;
    int tick = I32(sim, 0x10), hp0 = I32(victim, 0x3c4);
    if (g->dmg >= hp0 && I32(victim, 0xaa0) != -1) I32(victim, 0xaa0) = -1;          /* gasNoKillCredit */
    E.in_own_call = 1;
    o_root(victim, -1, g->dmg, 0, 0, NULL, NULL, 1, I32(victim, 0x38), I32(victim, 0x3c), NULL, 0, 0, 0, 1, 0, 0, 0);
    E.in_own_call = 0;
    int hp1 = I32(victim, 0x3c4);
    if (I32(victim, 0x7d8) != tick && hp1 > 0) I32(victim, 0x7d8) = tick;                /* gasBlockRegen belt and braces */
    int pi = I32(victim, 0x80);
    if (pi >= 0 && pi < NBE_MAXP) E.gasd[pi] += hp0 - hp1;
    if (hp1 <= 0) for (int i = 0; i < NBE_MAXP; i++) if (E.gas_lethal_id[i] == g->id || E.gas_lethal_id[i] == 0) { E.gas_lethal_id[i] = g->id; E.gas_lethal_tick[i] = tick; break; }
}

/* ------------------------------------------------------------------ Piper's leap (SF) */
static void sf_stop(void *ch) { if (!U8(ch, 0x490) && CAN_STOP(ch) && MOVING(ch)) STOP(ch); }
static void sf_spawn_box(void *sim, void *itemData, int x, int y, int owner) {
    void *it = ((void *(*)(void *))B(0xb5ddd0))(itemData); if (!it) return;
    I32(it, 0x88) = 0; I32(it, 0x80) = owner; I32(it, 0x84) = -1;
    if (E.npbox == 32) { memmove(E.pbox, E.pbox + 1, 31 * sizeof *E.pbox); E.npbox--; }
    E.pbox[E.npbox++] = (PendBox){x, y, owner, I32(sim, 0x10)};
    ((void (*)(void *, int, int, int))B(0xb6a61c))(it, x, y, 0);
    ((void (*)(void *, int, int, int, int, int, int))B(0xb6e310))(it, x, y, 400, x, y, 0);
    ((void (*)(void *, void *))B(0xb63198))(PTR(sim, 0), it);
    E.sf_boxes++; E.last_bomb_tick = I32(sim, 0x10);
}
static void sf_start_hook(uint64_t *ctx) {   /* 0xb475cc (the one-instruction ret stub = charge start) onEnter */
    if (I32(g_libg->base, 0x12eefb4) != 3 || !E.sim) return;
    uint8_t *ch = (uint8_t *)ctx[0]; int tx = (int)ctx[1], ty = (int)ctx[2], dmg = (int)ctx[3], speed = (int)ctx[6];
    uint64_t *sp = (uint64_t *)((uint8_t *)ctx + 160); int type = (int)sp[0]; void *item = (void *)sp[2];
    if (type != 6) { E.sf_ignored++; return; }
    void *sim = E.sim; int W, Hh; if (!map_wh(sim, &W, &Hh)) return; W *= 300; Hh *= 300;
    int id = I32(ch, 8), x0 = I32(ch, 0x38), y0 = I32(ch, 0x3c);
    int cx = tx < 150 ? 150 : tx > W - 150 ? W - 150 : tx, cy = ty < 150 ? 150 : ty > Hh - 150 ? Hh - 150 : ty;
    double L = hypot(cx - x0, cy - y0); int lx = cx, ly = cy; double back = 0;
    while (!walkable(sim, lx, ly) && back < L) { back += 50; double f = (L - back) / (L > 1 ? L : 1); if (f < 0) f = 0;
        lx = jround(x0 + (cx - x0) * f); ly = jround(y0 + (cy - y0) * f); }
    double dist = hypot(lx - x0, ly - y0), per = speed / 20.0 > 1 ? speed / 20.0 : 1; int T = (int)ceil(dist / per); if (T < 1) T = 1;
    int slot = -1; for (int i = 0; i < E.njobs; i++) if (E.jobs[i].id == id) slot = i;
    if (slot < 0) { if (E.njobs >= NBE_MAXP) return; slot = E.njobs++; }
    E.jobs[slot] = (Leap){id, ch, x0, y0, lx, ly, T, 0, I32(sim, 0x10)};
    E.sf_started++; sf_stop(ch);
    if (item) {
        int owner = I32(ch, 0x80);
        if (owner >= 0 && owner < NBE_MAXP) { E.sf_owner_id[owner] = id; E.sf_owner_dmg[owner] = dmg; }
        if (E.ntj < 64) E.tj[E.ntj++] = (TickJob){1, x0, y0, owner, 0, item};
    }
}
static void sf_tick(void *sim) {
    for (int i = 0; i < E.njobs; i++) {
        Leap *j = &E.jobs[i]; uint8_t *ch = obj_by_id(sim, j->id);
        if (!ch || (void *)ch != j->ch || I32(ch, 0x3c4) <= 0) { E.jobs[i--] = E.jobs[--E.njobs]; continue; }
        j->t++; double f = (double)j->t / j->T; if (f > 1) f = 1;
        I32(ch, 0x38) = jround(j->x0 + (j->x1 - j->x0) * f); I32(ch, 0x3c) = jround(j->y0 + (j->y1 - j->y0) * f);
        I32(ch, 0x40) = f < 1 ? jround(700 * 4 * f * (1 - f)) : 0;
        if (f >= 1) { E.sf_landed++; sf_stop(ch); E.jobs[i--] = E.jobs[--E.njobs]; }
    }
}
static int leaping(int id) { for (int i = 0; i < E.njobs; i++) if (E.jobs[i].id == id) return 1; return 0; }
static void sf_explode(void *sim, Bomb *b) {
    int o = b->owner; if (o < 0 || o >= NBE_MAXP || !E.sf_owner_id[o]) return;
    uint8_t *atk = obj_by_id(sim, E.sf_owner_id[o]); int alive = atk && I32(atk, 0x3c4) > 0;
    int np; void **ps = players(sim, &np); uint8_t *pl = o < np ? ps[o] : NULL; int charge0 = pl ? I32(pl, 0x54) : 0;
    int dmg = jround(E.sf_owner_dmg[o] * 2.0 * (1 + 0.10 * cubes_of(E.sf_owner_id[o])));
    int n; void **v = objvec(sim, &n);
    for (int i = 0; i < n; i++) {
        uint8_t *c = v[i]; if (!c || VTOF(c) != VT_CHAR) continue;
        if (I32(c, 0x80) == o || I32(c, 0x3c4) <= 0) continue;
        int dx = I32(c, 0x38) - b->x, dy = I32(c, 0x3c) - b->y; if ((long)dx * dx + (long)dy * dy > 600L * 600) continue;
        E.in_own_call = 2;
        int hp0 = I32(c, 0x3c4);
        o_root(c, o, dmg, dmg, 0, alive ? atk : NULL, NULL, 1, b->x, b->y, NULL, 0, 0, 0, 1, 0, 0, 0);
        record_event(c, alive ? atk : NULL, hp0, I32(c, 0x3c4));
        E.in_own_call = 0;
        v = objvec(sim, &n);
    }
    if (pl) I32(pl, 0x54) = charge0;
    E.sf_pops++;
}
static void sf_bomb_state(void *sim) {   /* afterState: detect popped bombs */
    int tick = I32(sim, 0x10);
    if (tick - E.last_bomb_tick > 200) { E.nbombs = 0; return; }
    Bomb now[64]; int nn = 0, n; void **v = objvec(sim, &n);
    for (int i = 0; i < n && nn < 64; i++) {
        uint8_t *o = v[i]; if (!o) continue; int vo = VTOF(o); if (vo != VT_ITEM) continue;
        uint8_t *d = PTR(o, 0x10); if (!d || I32(d, 0x20) != BOMB_ITEM) continue;
        int id = I32(o, 8), x = I32(o, 0x38), y = I32(o, 0x3c), owner = -1, known = 0;
        for (int k = 0; k < E.nbombs; k++) if (E.bombs[k].id == id) { owner = E.bombs[k].owner; known = 1; }
        if (!known) {
            if (I32(o, 0x80) >= 0) owner = I32(o, 0x80);
            else { long bd = (long)4e18; int best = -1;
                for (int k = 0; k < E.npbox; k++) { if (tick - E.pbox[k].tick > 8) continue;
                    long dd = (long)(E.pbox[k].x - x) * (E.pbox[k].x - x) + (long)(E.pbox[k].y - y) * (E.pbox[k].y - y);
                    if (dd < bd) { bd = dd; best = k; } }
                if (best >= 0 && bd < 400L * 400) owner = E.pbox[best].owner; }
        }
        if (owner >= 0) now[nn++] = (Bomb){id, x, y, owner};
    }
    for (int k = 0; k < E.nbombs; k++) {
        int still = 0; for (int i = 0; i < nn; i++) if (now[i].id == E.bombs[k].id) still = 1;
        if (!still && E.ntj < 64) { TickJob t = {2, E.bombs[k].x, E.bombs[k].y, E.bombs[k].owner, E.bombs[k].id, NULL}; E.tj[E.ntj++] = t; }
    }
    memcpy(E.bombs, now, nn * sizeof *now); E.nbombs = nn;
}
/* NBE_F_REC: the phone agent's action recorder (demo_record; RECSRC): every controller call of every character,
 * move 0xb4790c(char, x, y, .., x6 flag), fire 0xb487c4(char, skillData, x, y, .., x6 isSuper), stop 0xb4103c(char).
 * Record (8 int32): tick, charId, kind (0 move, 1 fire, 2 stop), a, b, c, d, owner; move: a,b = target, c = flag;
 * fire: a,b = target, c = isSuper, d = 1 if blocked (in the air: x1 = 1 set by the skill block before). */
static int g_rec_n, g_rec_lost;
static void rec_put(int kind, uint8_t *ch, int a, int b, int c, int d) {
    if (!(E.cur_flags & NBE_F_REC) || !ch || !E.sim || !RECB) return;
    if (g_rec_n >= NBE_REC_CAP) { g_rec_lost++; return; }
    int32_t *r = RECB + 8 * g_rec_n++;
    r[0] = I32(E.sim, 0x10); r[1] = I32(ch, 8); r[2] = kind; r[3] = a; r[4] = b; r[5] = c; r[6] = d; r[7] = I32(ch, 0x80);
}
static void rec_move(uint64_t *ctx) { rec_put(0, (uint8_t *)ctx[0], (int)ctx[1], (int)ctx[2], (int)ctx[6], 0); }
static void rec_stop(uint64_t *ctx) { rec_put(2, (uint8_t *)ctx[0], 0, 0, 0, 0); }
/* pctrain round: "native + lead" (research B1/B2/D2). The native AI's targeted attack (0xb8d888 -> bl 0xb487c4 at
 * 0xb8dd38, return 0xb8dd3c; w6 = isSuper) passes the target's position + U[-150,150)^2 and never leads. For owners in
 * g_lead_mask the target = the nearest living enemy brawler to the aim point (within 260 units, else a box / nothing:
 * unchanged) and the aim becomes the linear intercept T + v t with |T + v t - S| = speed (t - dt), v = the target's
 * displacement over the last 4 ticks / 4 (record_positions), S = the shooter's position. Supers stay native. Runs before
 * the recorder, so the records (BC labels) carry the corrected target. */
uint32_t g_lead_mask; float g_lead_dt, g_lead_speed = 197.0f, g_lead_frac = 1.0f; static int g_lead_n, g_lead_skip;
static void lead_aim(uint64_t *ctx) {
    uint8_t *ch = (uint8_t *)ctx[0]; void *sim = E.sim;
    if (!sim || !ch || ctx[9] != (uint64_t)(g_libg->base + 0xb8dd3c) || (ctx[6] & 1)) return;
    int owner = I32(ch, 0x80); if (owner < 0 || owner >= NBE_MAXP || !((g_lead_mask >> owner) & 1)) return;
    double ax = (int32_t)ctx[2], ay = (int32_t)ctx[3], bd = 260.0 * 260.0; int best = -1; uint8_t *bc = NULL;
    int np; void **ps = players(sim, &np);
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        if (i == owner || !ps[i]) continue;
        uint8_t *c = obj_by_id(sim, player_char_id(ps[i])); if (!c || VTOF(c) != VT_CHAR || I32(c, 0x3c4) <= 0) continue;
        double dx = I32(c, 0x38) - ax, dy = I32(c, 0x3c) - ay, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; bc = c; }
    }
    if (best < 0) { g_lead_skip++; return; }
    double Sx = I32(ch, 0x38), Sy = I32(ch, 0x3c), Tx = I32(bc, 0x38), Ty = I32(bc, 0x3c), vx = 0, vy = 0;
    int n = E.hn[best];
    if (n >= 5 && E.hid[best] == I32(bc, 8)) { int k0 = (n - 1) & 7, k4 = (n - 5) & 7; vx = (E.hx[best][k0] - E.hx[best][k4]) / 4.0; vy = (E.hy[best][k0] - E.hy[best][k4]) / 4.0; }
    double s = g_lead_speed > 0 ? g_lead_speed : 197.0, dt = g_lead_dt, Dx = Tx - Sx, Dy = Ty - Sy;
    double a = vx * vx + vy * vy - s * s, b = 2 * (Dx * vx + Dy * vy + s * s * dt), c = Dx * Dx + Dy * Dy - s * s * dt * dt;
    double disc = b * b - 4 * a * c, t = 0;
    if (disc >= 0 && a < 0) { double r1 = (-b + sqrt(disc)) / (2 * a), r2 = (-b - sqrt(disc)) / (2 * a); t = r1 > r2 ? r1 : r2; if (t < 0) t = 0; if (t > 40) t = 40; }
    t *= g_lead_frac;
    ctx[2] = (uint64_t)(uint32_t)jround(Tx + vx * t); ctx[3] = (uint64_t)(uint32_t)jround(Ty + vy * t);
    g_lead_n++;
}
static void lead_set(const NbeCmd *c) { g_lead_mask = c->lead_mask; g_lead_dt = c->lead_dt; g_lead_speed = c->lead_speed > 0 ? c->lead_speed : 197.0f; g_lead_frac = c->lead_frac; }
static void my_skill_block(uint64_t *ctx) {   /* 0xb487c4 onEnter: no attack/super while airborne */
    if (g_lead_mask) lead_aim(ctx);
    if (E.njobs && ctx[0]) { int id = I32(ctx[0], 8); if (leaping(id)) { ctx[1] = 1; E.sf_blocked++; } }
    rec_put(1, (uint8_t *)ctx[0], (int)ctx[2], (int)ctx[3], (int)ctx[6], ctx[1] == 1 ? 1 : 0);
}
static uint8_t g_empty_list[64] __attribute__((aligned(16)));
static int my_modlist(void *list, void *out, int type) {   /* 0x50a7a0: bl 0xba8374(list, &out, 0xef): client guard */
    if (list) {
        int n = I32(list, 0xc); void **arr = PTR(list, 0);
        uint64_t hi = (uint64_t)list >> 48;
        #define BADP(q) (!(q) || (uint64_t)(q) < 0x10000 || ((uint64_t)(q) >> 48) != hi)
        if (n > 0) {
            if (n > 512 || BADP(arr)) { list = g_empty_list; E.guard_hits++; }
            else for (int i = 0; i < n; i++) if (BADP(arr[i])) { list = g_empty_list; E.guard_hits++; break; }
        }
    }
    return ((int (*)(void *, void *, int))B(0xba8374))(list, out, type);
}

/* ------------------------------------------------------------------ elimination tracker (Brawlers left, end, rank) */
static void elimination_tick(void *sim) {
    int np; void **ps = players(sim, &np); int tick = I32(sim, 0x10), li = local_index();
    int rows_alive[NBE_MAXP], present[NBE_MAXP], total = 0;
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        present[i] = 0; if (!ps[i]) continue; present[i] = 1; total++;
        uint8_t *c = obj_by_id(sim, player_char_id(ps[i]));
        rows_alive[i] = c && VTOF(c) == VT_CHAR && I32(c, 0x3c4) > 0;
    }
    E.total = total;
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        if (!present[i]) continue;
        if (rows_alive[i]) { E.seen[i] = 1; continue; }
        if (!E.seen[i] || E.elim[i]) continue;
        int id = player_char_id(ps[i]), gas = 0;
        for (int k = 0; k < NBE_MAXP; k++) if (E.gas_lethal_id[k] == id && tick - E.gas_lethal_tick[k] <= 2) gas = 1;
        E.elim[i] = 1; E.erank[i] = total - E.order_n; E.etick[i] = tick; E.egas[i] = gas; E.order_n++;
        if (U8(ps[i], 0x48)) U8(ps[i], 0x48) = 0;     /* sdPlayersLeft */
        E.elim_ptr[E.eptr_n++ % NBE_MAXP] = ps[i];
    }
    E.alive = total - E.order_n;
    if (!E.end_req && E.order_n > 0 && E.alive <= 1) {     /* sdEndOnLastAlive */
        E.end_req = 1; E.winner = -1; int surv = 0;
        for (int i = 0; i < np && i < NBE_MAXP; i++) if (present[i] && !E.elim[i]) { E.winner = i; surv++; }
        if (surv != 1) E.winner = -1;
        E.local_rank = li < 0 ? 0 : (E.elim[li] ? E.erank[li] : 1 + surv - (li >= 0 && !E.elim[li] ? 1 : 0));
        int e4 = U8(sim, 0x4ea); U8(sim, 0x121) = 1; if (e4) U8(sim, 0xa1) = 1;
    }
}
typedef uint64_t (*fn1)(void *);
static fn1 o_alive;
static uint64_t my_alive(void *p) {   /* 0xc42ae8 "dead but still counted" -> 0 for players we eliminated */
    uint64_t r = o_alive(p);
    if ((r & 1) && E.order_n) for (int i = 0; i < NBE_MAXP; i++) if (E.elim_ptr[i] == p) { E.alive_forced++; return 0; }
    return r;
}

/* ------------------------------------------------------------------ per-tick work */
static void item_tick(void *sim) {   /* itemTick: credit a power cube when the cube item is consumed natively */
    int n; void **v = objvec(sim, &n); ItemRow now[512]; int nn = 0;
    for (int i = 0; i < n && nn < 512; i++) {
        uint8_t *o = v[i]; if (!o) continue; int vo = VTOF(o); if (vo != VT_ITEM && vo != VT_ITEM2) continue;
        uint8_t *d = PTR(o, 0x10); now[nn++] = (ItemRow){I32(o, 8), d ? I32(d, 0x20) : NIL, I32(o, 0x80), I32(o, 0x38), I32(o, 0x3c)};
    }
    for (int k = 0; k < E.nitems; k++) {
        int still = 0; for (int i = 0; i < nn; i++) if (now[i].id == E.items[k].id) { still = 1; break; }
        if (still || E.items[k].data != CUBE_ITEM) continue;
        int dup = 0; for (int c = 0; c < E.nconsumed; c++) if (E.consumed[c] == E.items[k].id) dup = 1;
        if (dup) continue;
        if (E.nconsumed < 1024) E.consumed[E.nconsumed++] = E.items[k].id;
        /* nearestPlayerTo: any character of a player (not a box), alive or not */
        long bd = -1; uint8_t *best = NULL;
        for (int i = 0; i < n; i++) { uint8_t *o = v[i]; if (!o) continue; int vo = VTOF(o); if (vo != VT_CHAR && vo != VT_CHAR2) continue;
            if (I32(o, 0x80) < 0) continue; uint8_t *d = PTR(o, 0x10); if (d && I32(d, 0x20) == BOX_ID) continue;
            double dd = hypot(I32(o, 0x38) - E.items[k].x, I32(o, 0x3c) - E.items[k].y); long di = jround(dd);
            if (bd < 0 || di < bd) { bd = di; best = o; } }
        if (!best) continue;
        int pid = I32(best, 8); uint8_t *p = obj_by_id(sim, pid); if (!p) continue;   /* creditCube */
        int mx = I32(p, 0x3c8), hp = I32(p, 0x3c4);
        I32(p, 0x3c8) = mx + 400; I32(p, 0x3c4) = hp + 400;
        cubes_add(pid);
    }
    memcpy(E.items, now, nn * sizeof *now); E.nitems = nn;
}
/* cube counter (cubecounter round 2026-10-07, client mode): the client draws the power-cube counter above a brawler
 * ("player_buffs" / "buff_text" of its sc/ui.sc info badge, badge update 0x526038) from client character +0x2c4, which the
 * snapshot decoder 0xb2de2c reads from tag 0x84; the character encoder 0xb360ac writes there the simulation's cube count
 * +0x874 (<< 1 under status 0xd5). The offline build sets sim+0x11b = (mode 1 or 2) = 0 (0x9bbbe0), so the native pickup
 * 0xb6b8e0 never credits +0x874 and the counter never shows; env credits cubes itself (item_tick). The two `bl`s that write
 * tag 0x84 (0xb36904 -> 0xc75a98 for variation 0x23, 0xb36924 -> 0xc75a0c) send +0x874 + env's count instead: only the
 * serialised snapshot changes, the simulation's state is untouched. x20 = the character, x9 = the 0xd5 shift there. */
int g_cube_counter = 1;   /* --no-cube-counter: off */
__asm__(".pushsection .text\n.p2align 2\ncube84_a: mov x3, x20\n mov x4, x9\n mov w5, #0\n b cube84_w\n"
        "cube84_b: mov x3, x20\n mov x4, x9\n mov w5, #1\n b cube84_w\n.popsection\n");
void cube84_a(void), cube84_b(void);
void cube84_w(void *wr, int v, int tag, uint8_t *ch, uint64_t shift, int which) {
    if (I32(g_libg->base, 0x12eefb4) == 3 && ch) v = (I32(ch, 0x874) + cubes_of(I32(ch, 8))) << (shift & 1);
    ((void (*)(void *, int, int))B(which ? 0xc75a98 : 0xc75a0c))(wr, v, tag);
}
static void cube_counter_install(void) {
    static const uint32_t a[6] = {0xb9487688, 0x92400009, 0x52801082, 0xaa1303e0, 0x1ac92101, 0x9404fc3a},   /* 0xb36910.. */
                          b[6] = {0xb9487688, 0x92400009, 0x52801082, 0xaa1303e0, 0x1ac92101, 0x9404fc65};   /* 0xb368f0.. */
    if (memcmp(B(0xb36910), a, sizeof a) || memcmp(B(0xb368f0), b, sizeof b)) { LOG("[env] cube counter: unexpected bytes, not applied\n"); return; }
    hook_bl(g_libg, 0xb36924, cube84_a); hook_bl(g_libg, 0xb36904, cube84_b);
    LOG("[env] cube counter: snapshot tag 0x84 = env's power cubes (0xb36904, 0xb36924)\n");
}
static void gas_tick(void *sim) {   /* the storm schedule (check every 20 ticks) + regen block + per-tick gas flags */
    int tick = I32(sim, 0x10), np; void **ps = players(sim, &np);
    E.gas_count++;
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        if (!ps[i]) continue; uint8_t *c = obj_by_id(sim, player_char_id(ps[i]));
        int alive = c && VTOF(c) == VT_CHAR && I32(c, 0x3c4) > 0;
        if (!alive) { E.ingas[i] = 0; continue; }
        int g = in_gas_tile(sim, (int)floor(I32(c, 0x38) / 300.0), (int)floor(I32(c, 0x3c) / 300.0));
        E.ingas[i] = g; if (g) E.gast[i]++;
        if (g && E.gas_on) I32(c, 0x7d8) = tick;        /* gasBlockRegen */
        if (E.gas_on && E.gas_count % 20 == 0) {
            if (!g) { E.gas_since_valid[i] = 0; continue; }
            if (!E.gas_since_valid[i]) { E.gas_since_valid[i] = 1; E.gas_since[i] = tick; }
            double secs = (tick - E.gas_since[i]) / 20.0, pct = 20;
            if (secs > 5) pct += (secs - 5) * 5;
            int mx = I32(c, 0x3c8), dmg = jround(mx * pct / 100); if (dmg < 1) dmg = 1;
            if (E.ngq < NBE_MAXP) E.gq[E.ngq++] = (GasHit){I32(c, 8), dmg, (int)pct, (int)(secs * 10), mx, 0, 0};
        }
    }
}
static void apply_setup(void *sim) {   /* the phone test's debug_js_tick: park/freeze/teleport inside a tick */
    int np; void **ps = players(sim, &np);
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        if (!ps[i]) continue; uint8_t *c = obj_by_id(sim, player_char_id(ps[i])); if (!c || VTOF(c) != VT_CHAR) continue;
        if ((g_setup.ai_off_mask >> i) & 1) U8(c, 0x469) = 0;
        if ((g_setup.set_mask >> i) & 1) { sf_stop(c); I32(c, 0x38) = g_setup.set_x[i]; I32(c, 0x3c) = g_setup.set_y[i]; I32(c, 0x3c4) = I32(c, 0x3c8); sf_stop(c); }
    }
}
static void pre_snapshot(void *sim) {   /* 0xb63b0c, once per tick: storm hits, leaps, tick jobs, local AI */
    if (g_setup_pending) { g_setup_pending = 0; apply_setup(sim); }
    for (int i = 0; i < E.ngq; i++) gas_hit(sim, obj_by_id(sim, E.gq[i].id), &E.gq[i]);
    E.ngq = 0;
    if (E.njobs) sf_tick(sim);
    int ntj = E.ntj; TickJob tj[64]; memcpy(tj, E.tj, ntj * sizeof *tj); E.ntj = 0;
    for (int i = 0; i < ntj; i++) {
        if (tj[i].kind == 1) sf_spawn_box(sim, tj[i].p, tj[i].a, tj[i].b, tj[i].c);
        else if (tj[i].kind == 2) { Bomb b = {tj[i].d, tj[i].a, tj[i].b, tj[i].c}; sf_explode(sim, &b); }
    }
    int li = local_index(), np; void **ps = players(sim, &np);
    if (li >= 0 && li < np && ps[li]) {   /* local_ai: the native AI drives the local player while its bit is set */
        /* the phone agent (demo_record local_ai) sets char+0x469 in the pre-snapshot of the first tick after the pause = the
         * reset observation's tick; during the intro the local Piper is the (idle) human player. Same here: flag 0 during
         * the intro, then the native_mask bit from the reset observation's tick on (the AI acts from the next tick). */
        uint8_t *c = obj_by_id(sim, player_char_id(ps[li]));
        /* RESET reserved[1] = 1: the local AI also plays the intro = the phone from the 2nd episode of a session on (the agent's
         * local_ai switch persists across reset_battle; T6 seeds 1235/1236) */
        if (c && VTOF(c) == VT_CHAR) { int want = (E.intro && E.cur.reserved[1] != 1) ? 0 : (g_native_mask >> li) & 1; if (U8(c, 0x469) != want) U8(c, 0x469) = want; }
    }
}
static void presnap_hook(uint64_t *ctx) {
    (void)ctx; void *sim = E.sim;
    if (!sim || I32(g_libg->base, 0x12eefb4) != 3) return;
    int t = I32(sim, 0x10); if (t == E.presnap_tick) return;
    E.presnap_tick = t; pre_snapshot(sim);
}
/* --no-snapshot (sim-only episodes): the per-tick client snapshot serializer 0xb63b0c is skipped (nothing receives
 * it); the pre-snapshot work still runs at its entry. Checked bit-identical with t2_determinism.py (NOTES T1). */
int g_no_snapshot = 1;   /* default on in sim-only mode; --snapshot to keep the serializer */
typedef uint64_t (*fn10)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static fn10 o_snapw;
static uint64_t my_snapw(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7, uint64_t s0, uint64_t s1) {
    uint64_t ctx[1] = {a0}; presnap_hook(ctx);
    if (g_no_snapshot && g_in_loop && E.sim) return 0;
    return o_snapw(a0, a1, a2, a3, a4, a5, a6, a7, s0, s1);
}
void env_tick_enter(void *sim) { if (sim == E.sim) E.depth++; }
static void record_positions(void *sim) {
    int np; void **ps = players(sim, &np);
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        uint8_t *c = ps[i] ? obj_by_id(sim, player_char_id(ps[i])) : NULL;
        if (!c || VTOF(c) != VT_CHAR) { E.hn[i] = 0; continue; }
        int id = I32(c, 8); if (id != E.hid[i]) { E.hid[i] = id; E.hn[i] = 0; }
        int k = E.hn[i]++ & 7; E.hx[i][k] = I32(c, 0x38); E.hy[i][k] = I32(c, 0x3c);
    }
}
/* the client draws (and the finger's joystick uses) the local brawler ~1.73 ticks behind the simulation: demos s1b,
 * client = sim(t) - 0.432 * (sim(t) - sim(t-4)), residual median 6 units (NOTES T1); sim-only episodes have no client */
#define CLIENT_LAG_F 0.432
static void client_pos(int i, uint8_t *ch, int *x, int *y) {
    int X = I32(ch, 0x38), Y = I32(ch, 0x3c), n = E.hn[i];
    if (n < 5 || E.hid[i] != I32(ch, 8)) { *x = X; *y = Y; return; }
    int k = (n - 5) & 7;      /* 4 ticks before the newest entry */
    *x = jround(X - CLIENT_LAG_F * (X - E.hx[i][k])); *y = jround(Y - CLIENT_LAG_F * (Y - E.hy[i][k]));
}
void env_tick_leave(void *sim) {   /* after a completed outer tick: showdownTick + afterState */
    if (sim != E.sim) return;
    if (--E.depth > 0) return;
    record_positions(sim);
    if (I32(sim, 0x104) != 6) return;
    item_tick(sim);
    elimination_tick(sim);
    gas_tick(sim);
    sf_bomb_state(sim);
}

/* ------------------------------------------------------------------ actions (ClientInputs through 0xc2cef4) */
static const uint8_t *KIND_BY_TYPE;   /* libg 0x185710: skill +0x1ac type 2..5 -> input kind */
static void apply_input(void *sim, int pidx, int kind, int x, int y, void *skilldata) {
    uint8_t in[0x48] __attribute__((aligned(16))); memset(in, 0, sizeof in);
    *(int32_t *)(in + 8) = kind; *(int32_t *)(in + 0xc) = x; *(int32_t *)(in + 0x10) = y; *(void **)(in + 0x40) = skilldata;
    ((fn_cef4)B(0xc2cef4))(sim, in, pidx, 1);
}
/* the client's joystick -> world target (0x877448, decompiled re/c/877448.c) on simulation objects:
 * 600 units toward the direction, raycast against the tile map, pulled back 20 units from a hit, wall sliding */
static int g_joy_flag_override = -1; static void *g_joy_map_override;
static void joy_target_at(void *sim, int X, int Y, float dx, float dy, int *ox, int *oy) {
    void *map = g_joy_map_override ? g_joy_map_override : PTR(sim, 0x68);
    /* raycast flag = 1 & ~0xb31e20(client object); 0xb31e20 reads client-object fields (not valid on the sim
     * character); for a walking Piper the client returns 0 (JOYTEST, runs/t1_joytest.txt) -> 1 */
    int flag = g_joy_flag_override >= 0 ? g_joy_flag_override : 1;
    float L = sqrtf(dx * dx + dy * dy), fx = dx * 600.0f / L, fy = dy * 600.0f / L, r14, r18;
    if (fx >= 0.001f || fx <= -0.001f) {
        if (fy >= 0.001f || fy <= -0.001f) { r14 = fabsf(fy / fx); r18 = fabsf(fx / fy); } else { r14 = 0; r18 = 100; }
    } else { r14 = 100; r18 = 0; }
    if (*(uint8_t *)PTR(B(0x1238408), 0)) { fx = -fx; fy = -fy; }
    int x = (int)(fx + (float)X), y = (int)(fy + (float)Y), out[2] = {-1, -1}, hit, ddx, ddy, d;
    hit = RAYCAST(X, Y, x, y, map, out, flag);
    if (!hit) goto final;
    ddx = out[0] - X; ddy = out[1] - Y; d = ISQRT(ddx * ddx + ddy * ddy);
    if (d > 20) { int a = (d - 20) * ddx, b = (d - 20) * ddy; ddx = d ? a / d : 0; ddy = d ? b / d : 0; }
    x = X + ddx; y = Y + ddy;
    if (d > 299) goto final;
    if (hit == 4 || hit == 1) {
        if (!(r14 <= 2.5f) && ddy != 0) {
            int s = ddy > 0 ? 200 : -200;
            int a = RAYCAST(X, Y, x + 100, y + s, map, out, flag), b = RAYCAST(X, Y, x - 100, y + s, map, out, flag);
            if (!((b == 0 && a == 0) || (a != 0 && b != 0))) { if (b == 0) x -= 500; else x += 500; goto slide; }
        }
        if (ddx < 1) x = ddx != 0 ? x - 500 : X; else x += 500;
    } else {
        if (!(r18 <= 2.5f) && ddx != 0) {
            int s = ddx > 0 ? 200 : -200;
            int a = RAYCAST(X, Y, x + s, y + 100, map, out, flag), b = RAYCAST(X, Y, x + s, y - 100, map, out, flag);
            if (!((b == 0 && a == 0) || (a != 0 && b != 0))) { if (b == 0) y -= 500; else y += 500; goto slide; }
        }
        if (ddy > 0) y += 500; else y = ddy != 0 ? y - 500 : Y;
    }
slide:
    hit = RAYCAST(X, Y, x, y, map, out, flag);
    if (hit) {
        ddx = out[0] - X; ddy = out[1] - Y; d = ISQRT(ddx * ddx + ddy * ddy);
        if (d > 20) { int a = (d - 20) * ddx, b = (d - 20) * ddy; ddx = d ? a / d : 0; ddy = d ? b / d : 0; }
        x = X + ddx; y = Y + ddy;
    }
final:;
    int W = I32(map, 0xc4) * 300 - 2, Hh = I32(map, 0xc8) * 300 - 2;
    *ox = x < 2 ? 1 : x >= W ? W : x; *oy = y < 2 ? 1 : y >= Hh ? Hh : y;
}
static void joy_target(void *sim, uint8_t *ch, float dx, float dy, int *ox, int *oy) { joy_target_at(sim, I32(ch, 0x38), I32(ch, 0x3c), dx, dy, ox, oy); }
/* T6 (phone scripted episodes, runs/t6/phone_scripted): the phone agent's client intents queue moves (and a super's
 * prepare input 5) straight into the simulation's input queue, but attacks go through the client's attack function
 * 0x85d094, whose input reaches the simulation only after that frame's ticks: every recorded fire call of the local player
 * happened at the start of the NEXT step (moves: the same step). g_fire_delay 1 reproduces that: the fire input
 * (target from this step's client position) is applied at the start of the next step, before that step's moves. */
int g_fire_delay = 1;
/* T6 (phone fire records vs the scripted actions, runs/t6/phone_scripted): 45 of 48 recorded local attacks have their
 * target on the commanded aim line from the shooter's SIMULATION position at the tick the input is applied (the start of
 * the next step) — the client's attack function 0x85d094 stores a pending attack that the client executes one frame
 * later from the then-current position. (The target is sometimes pulled in along the line: the client clamps the aim
 * point to the visible screen, e.g. dy <= ~2,800; not reproduced, the direction is unchanged.) So the delayed attack keeps
 * the aim and builds the input at application time from the sim position. */
static void fire_input_now(void *sim, int pidx, uint8_t *ch, int super, double deg, double dist, int X, int Y) {
    uint8_t *sd = super ? SKILL_SUPER(ch) : SKILL_ATTACK(ch); if (!sd) return;   /* 0xb35c00 / 0xb38fe8 return the SkillData */
    int type = I32(sd, 0x1ac), kind = (type - 2u < 4u) ? KIND_BY_TYPE[(type - 2) * 4] : 0;
    double r = deg * M_PI / 180, L = super ? 2600 * dist : 3000;
    int tx = jround(X + L * cos(r)), ty = jround(Y + L * sin(r));
    int ix = SKILL_ABS(sd) ? tx : tx - X, iy = SKILL_ABS(sd) ? ty : ty - Y;
    apply_input(sim, pidx, kind, ix, iy, sd);
}
static void fire_input_at(void *sim, int pidx, uint8_t *ch, int super, double deg, double dist, int X, int Y) {
    if (super) apply_input(sim, pidx, 5, 0, 0, NULL);       /* super prepare: straight into the simulation (queueKinds) */
    if (g_fire_delay && pidx < NBE_MAXP) {
        int n = E.pf[pidx].n; if (n < 2) { E.pf[pidx].super[n] = super; E.pf[pidx].deg[n] = deg; E.pf[pidx].dist[n] = dist; E.pf[pidx].n = n + 1; }
        return;
    }
    fire_input_now(sim, pidx, ch, super, deg, dist, X, Y);
}
static void fire_input(void *sim, int pidx, uint8_t *ch, int super, double deg, double dist) {
    int X, Y; client_pos(pidx, ch, &X, &Y); fire_input_at(sim, pidx, ch, super, deg, dist, X, Y);
}
static void apply_pending_fires(void *sim, int i, uint8_t *ch) {   /* origin = the shooter's sim position now */
    for (int k = 0; k < E.pf[i].n; k++) fire_input_now(sim, i, ch, E.pf[i].super[k], E.pf[i].deg[k], E.pf[i].dist[k], I32(ch, 0x38), I32(ch, 0x3c));
    E.pf[i].n = 0;
}
/* Client mode, local player: the phone agent's runClientIntents on the real client objects (the BattleScreen joystick
 * update's arguments 0x863d88: screen, client char, position object, map): position = 0xb5dac4/0xb5dacc(pos), move target
 * = the client's own 0x877448. Attacks: the phone calls the client's attack function 0x85d094 at frame START (before
 * BattleMode::update flushes the client's inputs), so the input is drained at the next frame; here the step runs inside
 * BattleMode::update (after that flush), where the same call would reach the simulation one frame later than on the
 * phone (T6: 2 steps instead of 1). So the attack input is built like the sim-only path (fire_input_at, kind table,
 * absolute/relative target) from the REAL client position and applied at the next step start (g_fire_delay) = the
 * phone's tick. --cl-native-intents 0: the sim-only path (lag-model client position) in client mode too. */
int g_cl_native_intents = 1;
typedef uint64_t (*fn_joy2)(void *, void *, void *, int, float, float);
static int client_intents(void *sim, int i, uint8_t *ch, const NbeAction *a) {
    if (g_env_simonly || !g_cl_native_intents || i != local_index() || !g_joy_args[0] || !g_joy_args[1] || !g_joy_args[2]) return 0;
    void *scr = (void *)g_joy_args[0], *cc = (void *)g_joy_args[1], *pos = (void *)g_joy_args[2], *x3 = (void *)g_joy_args[3];
    int cx = ((int (*)(void *))B(0xb5dac4))(pos), cy = ((int (*)(void *))B(0xb5dacc))(pos);
    if (a->move_dir >= 0 && a->move_dir < 16) {
        double r = a->move_dir * M_PI / 8; int w3 = 1 & ~CHAR_FLAG31E20(pos);
        uint64_t t = ((fn_joy2)B(0x877448))(scr, pos, x3, w3, (float)cos(r), (float)sin(r)); int tx = (int)(uint32_t)t, ty = (int)(t >> 32);
        int far = !E.sent_valid[i] || (long)(tx - E.sent_x[i]) * (tx - E.sent_x[i]) + (long)(ty - E.sent_y[i]) * (ty - E.sent_y[i]) > 225L * 225;
        if (far || E.last_dir[i] != a->move_dir) { apply_input(sim, i, 2, tx, ty, NULL); E.sent_valid[i] = 1; E.sent_x[i] = tx; E.sent_y[i] = ty; E.cl_moves++; }
        E.moving[i] = 1; E.last_dir[i] = a->move_dir;
    } else if (a->move_dir < 0 && E.moving[i]) {
        apply_input(sim, i, 2, cx, cy, NULL); E.moving[i] = 0; E.last_dir[i] = -1; E.sent_valid[i] = 0;
    }
    if (a->fire == 1) { fire_input_at(sim, i, ch, 0, a->aim_deg, 1, cx, cy); E.cl_attacks++; }
    else if (a->fire == 2) { fire_input_at(sim, i, ch, 1, a->aim_deg, a->super_dist > 0 && a->super_dist <= 1 ? a->super_dist : 1, cx, cy); E.cl_attacks++; }
    (void)cc; return 1;
}
static void apply_actions(void *sim, const NbeCmd *c) {
    int np; void **ps = players(sim, &np);
    for (int i = 0; i < np && i < NBE_MAXP; i++) {
        if (E.pf[i].n) {   /* last step's attacks (g_fire_delay) arrive first, like the client's queued inputs */
            uint8_t *ch0 = ps[i] ? obj_by_id(sim, player_char_id(ps[i])) : NULL;
            if (ch0 && VTOF(ch0) == VT_CHAR && I32(ch0, 0x3c4) > 0) apply_pending_fires(sim, i, ch0); else E.pf[i].n = 0;
        }
        if (!((c->ctl_mask >> i) & 1) || !ps[i]) continue;
        uint8_t *ch = obj_by_id(sim, player_char_id(ps[i]));
        if (!ch || VTOF(ch) != VT_CHAR || I32(ch, 0x3c4) <= 0) { E.moving[i] = 0; E.sent_valid[i] = 0; continue; }
        const NbeAction *a = &c->act[i];
        if (client_intents(sim, i, ch, a)) continue;
        if (a->move_dir >= 0 && a->move_dir < 16) {
            double r = a->move_dir * M_PI / 8; int tx, ty, cx, cy; client_pos(i, ch, &cx, &cy);
            joy_target_at(sim, cx, cy, (float)cos(r), (float)sin(r), &tx, &ty);
            int far = !E.sent_valid[i] || (long)(tx - E.sent_x[i]) * (tx - E.sent_x[i]) + (long)(ty - E.sent_y[i]) * (ty - E.sent_y[i]) > 225L * 225;
            if (far || E.last_dir[i] != a->move_dir) { apply_input(sim, i, 2, tx, ty, NULL); E.sent_valid[i] = 1; E.sent_x[i] = tx; E.sent_y[i] = ty; }
            E.moving[i] = 1; E.last_dir[i] = a->move_dir;
        } else if (E.moving[i]) {
            int cx, cy; client_pos(i, ch, &cx, &cy);
            apply_input(sim, i, 2, cx, cy, NULL);
            E.moving[i] = 0; E.last_dir[i] = -1; E.sent_valid[i] = 0;
        }
        if (a->fire == 1) fire_input(sim, i, ch, 0, a->aim_deg, 1);
        else if (a->fire == 2) { double dd = a->super_dist > 0 && a->super_dist <= 1 ? a->super_dist : 1; fire_input(sim, i, ch, 1, a->aim_deg, dd); }
    }
}
/* joystick update 0x863d88(screen, clientChar, posObj, map): keep the client's arguments (JOYTEST compares) */
static void my_joy_enter(uint64_t *ctx) { g_joy_args[0] = ctx[0]; g_joy_args[1] = ctx[1]; g_joy_args[2] = ctx[2]; g_joy_args[3] = ctx[3]; g_joy_calls++; }
typedef uint64_t (*fn_joy)(void *, void *, void *, int, float, float);
/* JOYTEST: for 16 directions at (x, y): the client's own 0x877448 on the client objects (pos temporarily moved) and the
 * port on the simulation character (temporarily moved). out: [native x, y, port x, y] x 16 */
static int joytest(void *sim, int x, int y, int32_t *out) {
    int n = 0; int np; void **ps = players(sim, &np); uint8_t *ch = np > 0 && ps[0] ? obj_by_id(sim, player_char_id(ps[0])) : NULL;
    uint8_t *pos = (uint8_t *)g_joy_args[2];
    for (int k = 0; k < 16; k++) {
        double r = k * M_PI / 8; float dx = (float)cos(r), dy = (float)sin(r);
        int nx = NIL, ny = NIL, px = NIL, py = NIL;
        if (pos) {
            int sx = I32(pos, 0x30), sy = I32(pos, 0x34); I32(pos, 0x30) = x; I32(pos, 0x34) = y;
            int w3 = 1 & ~CHAR_FLAG31E20(pos);
            uint64_t t = ((fn_joy)B(0x877448))((void *)g_joy_args[0], pos, (void *)g_joy_args[3], w3, dx, dy);
            nx = (int)(uint32_t)t; ny = (int)(t >> 32);
            I32(pos, 0x30) = sx; I32(pos, 0x34) = sy;
        }
        int qx = NIL, qy = NIL;
        if (ch) { int sx = I32(ch, 0x38), sy = I32(ch, 0x3c); I32(ch, 0x38) = x; I32(ch, 0x3c) = y; joy_target(sim, ch, dx, dy, &px, &py);
            g_joy_map_override = (void *)g_joy_args[3]; joy_target(sim, ch, dx, dy, &qx, &qy); g_joy_map_override = NULL;
            I32(ch, 0x38) = sx; I32(ch, 0x3c) = sy; }
        out[n++] = nx; out[n++] = ny; out[n++] = px; out[n++] = py; out[n++] = qx; out[n++] = qy;
    }
    if (pos && ch) LOG("[env] joytest flags: client pos object %d, sim char %d; client pos data %p sim data %p\n", CHAR_FLAG31E20(pos), CHAR_FLAG31E20(ch), PTR(pos, 0x10), PTR(ch, 0x10));
    LOG("[env] joytest at (%d,%d): client map %p (vt %s) sim map %p (vt %s), joy calls %ld\n", x, y, (void *)g_joy_args[3],
        g_joy_args[3] ? addr_desc(PTR(g_joy_args[3], 0)) : "-", PTR(sim, 0x68), addr_desc(PTR(PTR(sim, 0x68), 0)), g_joy_calls);
    return n;
}
/* AI dispatcher 0xb47084(char): env-controlled players and 'hold' bots skip the native AI */
static fn1 o_dispatch;
static uint64_t my_dispatch(void *ch) {
    if (E.sim && ch && I32(g_libg->base, 0x12eefb4) == 3) {
        int owner = I32(ch, 0x80);
        if (owner >= 0 && owner < 32) {
            if ((g_hold_mask >> owner) & 1) { if (U8(ch, 0x469) && I32(ch, 0x3c4) > 0) sf_stop(ch); return 0; }
            if (!((g_native_mask >> owner) & 1) && owner != local_index()) return 0;
        }
    }
    return o_dispatch(ch);
}

/* ------------------------------------------------------------------ soft reset (StartLoading re-injection) */
static uint8_t g_sl[4096]; static int g_sl_len, g_sl_seed_off = -1; static void *g_msgmgr;
static void sl_parse(void) {   /* NB68Clock.parseStartLoading: big-endian ints; seed after the modifier list */
    const uint8_t *u = g_sl; int n = g_sl_len;
#define BE(o) ((int32_t)((uint32_t)u[o] << 24 | (uint32_t)u[(o) + 1] << 16 | (uint32_t)u[(o) + 2] << 8 | u[(o) + 3]))
    g_sl_seed_off = -1;
    if (n < 28 || BE(12) != 0 || BE(16) != 0) { LOG("[env] StartLoading layout not parsed (roster %d list2 %d)\n", n >= 16 ? BE(12) : -1, n >= 20 ? BE(16) : -1); return; }
    int mc = BE(20); if (mc < 0 || mc > 64 || 24 + 4 * mc + 4 > n) return;
    g_sl_seed_off = 24 + 4 * mc;
    LOG("[env] StartLoading captured: %d bytes, players %d, index %d, %d modifiers, seed %d (offset %d)\n", n, BE(0), BE(4), mc, BE(g_sl_seed_off), g_sl_seed_off);
}
static void my_schedule_play(void *mgr, void *msg) {   /* startGame 0x963488: bl 0x816ff8(mgr, StartLoading) */
    if (msg && PTR(msg, 0) == (void *)B(0x120f808)) {
        uint8_t *bs = (uint8_t *)msg + 8; int n = I32(bs, 0x14) > I32(bs, 0x18) ? I32(bs, 0x14) : I32(bs, 0x18);
        uint8_t *data = PTR(bs, 0x30); if (!data) data = PTR(bs, 0x38);
        if (n > 0 && n <= (int)sizeof g_sl && data) { memcpy(g_sl, data, n); g_sl_len = n; g_msgmgr = mgr; sl_parse(); }
        else LOG("[env] StartLoading bytes not attached at schedule (len %d)\n", n);
    }
    ((void (*)(void *, void *))B(0x816ff8))(mgr, msg);
}
static void *g_hold_data, *g_hold_el0; static int g_hold_count, g_hold_on;
static int inject_reset(int seed) {
    if (!g_sl_len || !g_msgmgr) { LOG("[env] reset: no StartLoading captured\n"); return -1; }
    if (I32(g_libg->base, 0x12eefb4) != 3) { LOG("[env] reset: offline battle flag not set\n"); return -2; }
    uint8_t *L = B(0x12eefa0); void *d = PTR(L, 0); int n = I32(L, 0xc);
    if (!d || n < 1) { LOG("[env] reset: no local player setup in 0x12eefa0 (count %d)\n", n); return -3; }
    g_hold_data = d; g_hold_count = n; g_hold_el0 = PTR(d, 0); g_hold_on = 1; I32(L, 0xc) = 0;   /* keep it alive */
    uint8_t bytes[4096]; memcpy(bytes, g_sl, g_sl_len);
    if (g_sl_seed_off >= 0) { bytes[g_sl_seed_off] = seed >> 24; bytes[g_sl_seed_off + 1] = seed >> 16; bytes[g_sl_seed_off + 2] = seed >> 8; bytes[g_sl_seed_off + 3] = seed; }
    static uint64_t fac[8]; fac[0] = (uint64_t)B(0x120bb20); fac[1] = 0;
    void *m = ((void *(*)(void *, int))B(0xbc1628))(fac, 20559);
    if (!m) { fac[1] = 1; m = ((void *(*)(void *, int))B(0xbc1628))(fac, 20559); fac[1] = 0; }
    if (!m || PTR(m, 0) != (void *)B(0x120f808)) { I32(L, 0xc) = n; g_hold_on = 0; LOG("[env] reset: createMessage(20559) failed\n"); return -4; }
    void *buf = ((void *(*)(unsigned long))B(0x11a5050))(g_sl_len);
    memcpy(buf, bytes, g_sl_len);
    ((void (*)(void *, void *, int))B(0xcf9528))((uint8_t *)m + 8, buf, g_sl_len);
    ((void (*)(void *, void *))B(0x816ff8))(g_msgmgr, m);
    return 0;
}
void env_before_build(void) {   /* builder 0x9bb710 entry: restore the held setup list */
    E.t_build0 = tnow();
    if (!g_hold_on) return; g_hold_on = 0;
    uint8_t *L = B(0x12eefa0);
    if (PTR(L, 0) != g_hold_data) { LOG("[env] setup list moved\n"); return; }
    if (PTR(g_hold_data, 0) != g_hold_el0) PTR(g_hold_data, 0) = g_hold_el0;
    I32(L, 0xc) = g_hold_count;
}
static void my_setseed(void *sim, int seed) {   /* builder 0x9bb7c4: bl 0xc24a08(sim, seed) */
    if (E.have_reset) seed = E.reset_seed;
    ((void (*)(void *, int))B(0xc24a08))(sim, seed);
    E.seed_applied = I32(sim, 0x14);
}

/* ------------------------------------------------------------------ the step loop */
static void tick1(void *sim, void *x2) { ((void (*)(void *, int, void *))B(0xc2ccd0))(sim, 1, x2); }
static void ep_clear_step(void) {
    memset(E.dealt, 0, sizeof E.dealt); memset(E.taken, 0, sizeof E.taken); memset(E.gasd, 0, sizeof E.gasd);
    memset(E.kills, 0, sizeof E.kills); memset(E.hits, 0, sizeof E.hits); memset(E.boxd, 0, sizeof E.boxd);
    memset(E.bombd, 0, sizeof E.bombd); memset(E.gast, 0, sizeof E.gast);
    int np; void **ps = E.sim ? players(E.sim, &np) : NULL; if (!ps) np = 0;
    for (int i = 0; i < NBE_MAXP; i++) E.cubes0[i] = i < np && ps[i] ? cubes_of(player_char_id(ps[i])) : 0;
}
static void hook_listener(void *sim);
static void hits_mark(void); void hits_dump(const char *why);
static long g_streams_freed; int g_free_snap_stream = 1;   /* --keep-snap-streams: 0 (keeps the game's per-tick leak, see my_deliver) */
void env_on_build(void *battle, void *sim) {   /* a new battle was built: forget the old one */
    g_lead_n = 0; g_lead_skip = 0;
    int gen = E.gen + 1, hr = E.have_reset, seed = E.reset_seed, gas = E.gas_on, act = E.act_tick, sa = E.seed_applied, rr = E.reset_req;
    int ga_ok = E.gas_args_ok, a4 = E.ga4, a5 = E.ga5, a7 = E.ga7, s0 = E.gs0;
    NbeCmd cur = E.cur;
    memset(&E, 0, sizeof E);
    E.battle = battle; E.sim = sim; E.gen = gen; E.presnap_tick = -1; E.gas_on = gas; E.act_tick = act; E.seed_applied = sa;
    E.gas_args_ok = ga_ok; E.ga4 = a4; E.ga5 = a5; E.ga7 = a7; E.gs0 = s0;
    E.have_reset = hr; E.reset_seed = seed; E.cur = cur;
    for (int i = 0; i < NBE_MAXP; i++) E.last_dir[i] = -1;
    if (rr) { E.active = 1; E.intro = 1; E.started_tick = -1; }   /* the battle our reset asked for */
    if (gen <= 3 || gen % 100 == 0)
        LOG("[env] battle %d built (sim %p, seed requested %d, applied %d)%s, snapshot streams freed so far %ld\n", gen, sim, hr ? seed : -1, sa,
            (rr || g_env_simonly) ? "" : " [PLAY battle: waiting for RESET]", g_streams_freed);
    hook_listener(sim);
    memset(g_joy_args, 0, sizeof g_joy_args);   /* the old battle's client objects (the phone: JOY.frame = null on reset) */
}
static void fill_players(void *sim) {
    int np; void **ps = players(sim, &np);
    for (int i = 0; i < NBE_MAXP; i++) {
        NbePlayer *p = &R->pl[i]; memset(p, 0, sizeof *p);
        if (i >= np || !ps[i]) continue;
        uint8_t *c = obj_by_id(sim, player_char_id(ps[i]));
        p->present = 1; p->char_id = player_char_id(ps[i]);
        if (c && VTOF(c) == VT_CHAR) {
            p->hp = I32(c, 0x3c4); p->max_hp = I32(c, 0x3c8); p->alive = p->hp > 0; p->x = I32(c, 0x38); p->y = I32(c, 0x3c);
            p->ammo = s2_skill_word(c, 0, 0x18);
        }
        p->damage_dealt = E.dealt[i]; p->damage_taken = E.taken[i] + E.gasd[i]; p->gas_damage = E.gasd[i]; p->kills = E.kills[i];
        p->hits = E.hits[i]; p->box_damage = E.boxd[i]; p->bomb_damage = E.bombd[i];
        p->cubes_total = cubes_of(p->char_id); p->cubes = p->cubes_total - E.cubes0[i]; p->gas_ticks = E.gast[i]; p->in_gas = E.ingas[i];
        p->eliminated = E.elim[i]; p->elim_rank = E.erank[i]; p->elim_tick = E.etick[i]; p->elim_gas = E.egas[i];
        p->leaping = leaping(p->char_id); p->super_charge = I32(ps[i], 0x54); p->super_max = I32(ps[i], 0x58);
    }
}
/* ---------------- encoded observations (encode.c = state2.encode) */
typedef struct { int W, H; uint8_t *wall, *water, *bush; } EncMap;
typedef struct { int id, x, y; } PrevPos;
typedef struct { const int32_t *pl, *ch, *pj, *it; int npl, nch, npj, nit, tick; const int32_t (*cubes)[2]; int ncubes;
                 const int32_t (*jumps)[5]; int njumps; const PrevPos *prev; int nprev, prev_tick; } EncState;
void enc_map_build(EncMap *m, const int32_t *tiles, int W, int H);
int enc_encode(const EncMap *m, const EncState *s, int pidx, float *vec, float *grid);
static EncMap g_emap; static uint32_t g_emap_hash; static PrevPos g_prev[4096]; static int g_nprev, g_prev_tick = -1, g_prev_valid;
static int32_t g_maptmp[NBE_MAP_CAP];
static void enc_map_update(void *sim) {
    int wh[2]; uint32_t h = snap_map(sim, g_maptmp, NBE_MAP_CAP, wh);
    if (h != g_emap_hash) { enc_map_build(&g_emap, g_maptmp, wh[0], wh[1]); g_emap_hash = h; }
}
/* Sim-only mode has no client, so state2's 'client' flags (presence in the client's object list = what the viewer sees;
 * the encoder's enemy 'visible' feature r[19]) would all be -1 (= not visible: bc3p then ranked 6.84 on the PC vs 5.85 on
 * the phone). Emulation fitted to the phone (T6, runs/t6/_vis_fit.py: 10,632 enemy rows of 7 phone episodes, 98.7 %
 * agreement, open ground 99.4 %, bushes 95.8 %; all-invisible: 89.7 %): camera centre = the local brawler with x clamped
 * to [3000, W*300-3000]; in view iff |dx| < 4500 and -3500 < dy < 2900 (dy = y - camera y; the view reaches further up
 * than down); an enemy brawler on a bush tile is hidden beyond 900 units from the local brawler. Client positions stay
 * NIL. Client mode uses the real client's object list. --vis-model 0: off (all -1). */
int g_vis_model = 1;
int g_item_guard = 1;
int g_weak_skip = 1;   /* client mode: the 0xdc1f64 weak-lock skip patch (--no-weak-skip: off) */
static void emulate_client_flags(void *sim, int32_t *a) {
    int npl = a[3], nch = a[4], npj = a[5], nit = a[6]; int32_t *ch = a + 8 + npl * 8, *pj = ch + nch * 23, *it = pj + npj * 18;
    int li = local_index(); const int32_t *me = NULL;
    for (int k = 0; k < nch; k++) { const int32_t *r = ch + k * 23; if (r[2] == li && r[18] == 0 && r[1] != 16000120) { me = r; break; } }
    enc_map_update(sim);
    int W = g_emap.W, Hh = g_emap.H;
    double mx = me ? me[3] : 0, my = me ? me[4] : 0, cx = mx, cy = my;
    if (cx < 3000) cx = 3000; if (cx > W * 300 - 3000) cx = W * 300 - 3000;
    #define INVIEW(x, y) (me && fabs((x) - cx) < 4500 && ((y) - cy) > -3500 && ((y) - cy) < 2900)
    for (int k = 0; k < nch; k++) {
        int32_t *r = ch + k * 23; int v = INVIEW(r[3], r[4]);
        if (v && r[2] >= 0 && r[2] != li && r[18] == 0 && r[1] != 16000120) {   /* enemy brawler: bush hiding */
            int tx = r[3] / 300, ty = r[4] / 300; if (tx >= W) tx = W - 1; if (ty >= Hh) ty = Hh - 1;
            if (tx >= 0 && ty >= 0 && g_emap.bush && g_emap.bush[ty * W + tx] && hypot(r[3] - mx, r[4] - my) > 900) v = 0;
        }
        if (me == r) v = 1;
        r[19] = v;
    }
    for (int k = 0; k < npj; k++) { int32_t *r = pj + k * 18; r[14] = INVIEW(r[3], r[4]); }
    for (int k = 0; k < nit; k++) { int32_t *r = it + k * 12; r[8] = INVIEW(r[3], r[4]); }
    #undef INVIEW
}
static void write_obs(void *sim, uint32_t mask) {
    enc_map_update(sim);
    const int32_t *a = ST; EncState s; memset(&s, 0, sizeof s);
    s.tick = a[2]; s.npl = a[3]; s.nch = a[4]; s.npj = a[5]; s.nit = a[6];
    s.pl = a + 8; s.ch = s.pl + s.npl * 8; s.pj = s.ch + s.nch * 23; s.it = s.pj + s.npj * 18;
    s.cubes = (const int32_t (*)[2])R->cubes; s.ncubes = R->ncubes; s.jumps = (const int32_t (*)[5])R->jumps; s.njumps = R->njumps;
    if (g_prev_valid) { s.prev = g_prev; s.nprev = g_nprev; s.prev_tick = g_prev_tick; }
    int written = 0;
    for (int p = 0; p < NBE_MAXP; p++) if ((mask >> p) & 1) {
        float *v = OBSB + p * (NBE_OBS_VEC + NBE_OBS_GRID);
        if (enc_encode(&g_emap, &s, p, v, v + NBE_OBS_VEC)) written |= 1 << p;
    }
    R->obs_written = written;
}
static void remember_prev(void) {   /* the next step's velocities come from this dump (state2.StateView prev) */
    const int32_t *a = ST; int npl = a[3], nch = a[4]; const int32_t *ch = a + 8 + npl * 8;
    g_nprev = nch < 4096 ? nch : 4096;
    for (int k = 0; k < g_nprev; k++) { g_prev[k].id = ch[k * 23]; g_prev[k].x = ch[k * 23 + 3]; g_prev[k].y = ch[k * 23 + 4]; }
    g_prev_tick = a[2]; g_prev_valid = 1;
}
static int g_publish_defer;
static void publish(void *sim, uint32_t seq, int status, const char *err) {
    double t0 = tnow();
    R->seq = seq; R->status = status; snprintf(R->err, sizeof R->err, "%s", err ? err : "");
    R->tick = I32(sim, 0x10); R->done = U8(sim, 0xa1) || U8(sim, 0x121) || U8(sim, 0xa3);
    R->result = E.battle && PTR(E.battle, 0x28) ? I32(PTR(E.battle, 0x28), 0x11c) : -1;
    R->local_index = local_index(); R->generation = E.gen; R->alive = E.alive ? E.alive : E.total; R->winner = E.end_req ? E.winner : -1;
    int li = R->local_index; R->local_rank = li >= 0 && li < NBE_MAXP && E.elim[li] ? E.erank[li] : (E.end_req ? E.local_rank : 0);
    void *cl = E.battle ? PTR(E.battle, 0x28) : NULL, *mode = cl ? PTR(cl, 0x28) : NULL;
    R->hud_left = -1; (void)mode;
    R->reserved[5] = g_lead_n; { extern int g_swaps_count(void); R->reserved[6] = g_swaps_count(); }   /* expertbot round: eglSwapBuffers count (aligns nbgl ID-buffer dumps) instead of g_lead_skip */ R->seed_applied = E.seed_applied; R->reserved[0] = E.gas_latched; R->reserved[1] = E.gas_phase; R->reserved[2] = E.gas_j; { void *cb = E.battle ? PTR(E.battle, 0x28) : NULL; R->reserved[3] = (!g_env_simonly && cb) ? I32(cb, 0x4c) : -1; } R->seed_plus1 = I32(B(0x12f40c0), 0) && (uint64_t)(*(int64_t *)B(0x12f40c0) - *(int64_t *)B(0x12f40b8)) > 15;
    R->steps_in_frame = E.steps_in_frame; R->frame = E.frame;
    fill_players(sim);
    R->njumps = 0; for (int i = 0; i < E.njobs && i < NBE_MAXP; i++) { int *j = R->jumps[R->njumps++]; j[0] = E.jobs[i].id; j[1] = E.jobs[i].t; j[2] = E.jobs[i].T; j[3] = E.jobs[i].x1; j[4] = E.jobs[i].y1; }
    R->ncubes = 0; for (int i = 0; i < E.ncube && i < NBE_MAXP; i++) { R->cubes[R->ncubes][0] = E.cube_id[i]; R->cubes[R->ncubes++][1] = E.cube_n[i]; }
    R->cam_valid = 0;
    { void *scr = E.battle ? PTR(E.battle, 0x10) : NULL; if (scr) { R->cam_valid = 1; R->cam_x = *(float *)((uint8_t *)scr + 0xbf8); R->cam_y = *(float *)((uint8_t *)scr + 0xbfc); } }
    { float zf = g_cam_zoom_arg; memcpy(&R->reserved[7], &zf, 4); }   /* expertbot round: camera zoom argument */
    { extern int g_dbg_tick; g_dbg_tick = I32(sim, 0x10); }
    {   /* expertbot round: the world camera matrix of the frame drawn since the last response (gl_client.c capture) */
        extern volatile float g_cam_mat[16]; extern volatile int g_cam_mat_seq, g_cam_mat_n;
        for (int tries = 0; tries < 100; tries++) {
            int s0 = __atomic_load_n(&g_cam_mat_seq, __ATOMIC_ACQUIRE); if (s0 & 1) continue;
            for (int k = 0; k < 16; k++) R->cam_mat[k] = g_cam_mat[k];
            if (__atomic_load_n(&g_cam_mat_seq, __ATOMIC_ACQUIRE) == s0) { R->cam_mat_seq = s0 / 2; break; }
        }
        R->cam_mat_n = g_cam_mat_n; g_cam_mat_n = 0;
    }
    if (getenv("NBHOST_CAMDBG")) {   /* expertbot round: dump the screen object's floats around the camera anchor */
        void *scr = E.battle ? PTR(E.battle, 0x10) : NULL;
        if (scr) { const float *f = (const float *)((uint8_t *)scr + 0xbc0); char buf[1024]; int n = snprintf(buf, sizeof buf, "[camdbg] tick %d zoomarg %.5f scr", I32(sim, 0x10), g_cam_zoom_arg);
            for (int k = 0; k < 40 && n < (int)sizeof buf - 24; k++) n += snprintf(buf + n, sizeof buf - n, " %+.5g", f[k]);
            LOG("%s\n", buf); }
    }
    R->state_ints = 0; R->obs_written = 0;
    if (E.cur_flags & (NBE_F_STATE | NBE_F_OBS)) {
        void *cm = client_mgr();
        int k = snap2(sim, ST, NBE_STATE_CAP, cm); R->state_ints = k > 0 ? k : 0; if (k < 0) R->status = k * 100;
        if (k > 0 && !cm && g_vis_model) emulate_client_flags(sim, ST);
        if (k > 0) {
            if (E.cur.type == NBE_CMD_RESET) g_prev_valid = 0;   /* the reset observation has no previous view */
            if (E.cur_flags & NBE_F_OBS) write_obs(sim, E.cur.obs_mask);
            remember_prev();
            if (!(E.cur_flags & NBE_F_STATE)) R->state_ints = 0;
        }
    }
    R->map_written = 0;
    if ((E.cur_flags & NBE_F_MAP) || E.cur.type == NBE_CMD_RESET) { int wh[2]; R->map_hash = snap_map(sim, MAPB, NBE_MAP_CAP, wh); R->map_w = wh[0]; R->map_h = wh[1]; R->map_written = 1; }
    R->rec_n = (E.cur_flags & NBE_F_REC) ? g_rec_n : 0; R->rec_lost = g_rec_lost;
    R->us_state = (int)((tnow() - t0) * 1e6);
    R->us_step = (int)((tnow() - E.t_step0) * 1e6);
    if (!g_publish_defer) { __atomic_store_n(&H->resp_seq, seq, __ATOMIC_RELEASE); bell_ring(); }
    ep_clear_step(); g_rec_n = 0; g_rec_lost = 0;
}
static int wait_cmd(void) {   /* -> 1 when a new command is in CMD */
    double t0 = tnow(); long spins = 0;
    if (g_cmd_fd >= 0) {   /* bell pipes (--cmd-fd/--resp-fd): spin briefly (--bell-spin-us, default 200), then block until
                            * Python rings instead of spinning on */
        for (;;) {
            uint32_t s = __atomic_load_n(&H->cmd_seq, __ATOMIC_ACQUIRE);
            if (s != g_served) { R->us_wait = (int)((tnow() - t0) * 1e6); return 1; }
            { int sp = CMD->reserved[3] > 0 ? CMD->reserved[3] : g_bell_spin_us;   /* the last command says how long to spin */
              if ((tnow() - t0) * 1e6 < sp) { sched_yield(); continue; } }
            struct pollfd p = {g_cmd_fd, POLLIN, 0};
            int r = poll(&p, 1, 200);
            if (r > 0) { char b[256]; if (read(g_cmd_fd, b, sizeof b) == 0) { LOG("[env] command pipe closed: exiting\n"); _exit(3); } }
            else if (r == 0 && getppid() == 1) { LOG("[env] parent died: exiting\n"); _exit(3); }
        }
    }
    for (;;) {
        uint32_t s = __atomic_load_n(&H->cmd_seq, __ATOMIC_ACQUIRE);
        if (s != g_served) { R->us_wait = (int)((tnow() - t0) * 1e6); return 1; }
        if (++spins < 2000) { sched_yield(); continue; }
        struct timespec ts = {0, 100000}; nanosleep(&ts, NULL);
        if ((spins & 1023) == 0 && getppid() == 1) { LOG("[env] parent died: exiting\n"); _exit(3); }
    }
}
static void (*o_ticks)(void *, int, void *);
/* ======================= client mode (--env-client): the real battle with its client, renderer and the phone's clock
 * One step per client frame (the phone's lockstep frame). Frame f: [frame start hook: root dt := expected ticks x 50 ms]
 * -> the game waits for the render of frame f-1 -> BattleMode::update -> my_ticks: wait for the command, apply the
 * inputs, run the ticks, arm the capture -> the client update runs AFTER the ticks in round(dt*60) 1/60-s substeps
 * (0x9c32a4) -> BattleScreen update (camera: one 1/60 update per 1/60 s of game time) -> draw -> render job f.
 * Frame f+1 start hook: the state part of the response (snap2 after the client update, like the agent's state2());
 * the swap of render job f (render thread): GPU capture. The response is released when both are done. */
int g_intro_tpf = 3;   /* client mode: intro ticks per frame. The phone harness skips the intro at "speed 8" = 8 x 20 ticks/s at
                        * ~60 frames/s = ~2.7 ticks per frame; measured against the phone's frames (runs/opt/frames_tpf*): 3 is
                        * the closest (first 10 steps |diff| 1.4/255 vs 5.5 at 4, 10.7 at 8, 13.7 at 25). --intro-tpf N */
#define INTRO_TICKS_PER_FRAME g_intro_tpf
static long g_frames_undrawn;   /* --intro-render / --loading-render: frames not drawn so far */
static long g_load_frames;      /* frames of reset loading so far (counted when the no-draw patch is on) */
static int intro_skip(void); extern int g_intro_render, g_loading_render; static volatile int g_no_draw;
static int g_swapn, g_trace = 0; int g_trace_n;
static int g_cl_parts, g_cl_cap; static uint32_t g_cl_seq;
static volatile int g_cl_cap_frame = -1;              /* the frame whose swap is captured */
static volatile int g_fifo[64]; static volatile unsigned g_fifo_w, g_fifo_r; static long g_fifo_bad;
static void submit_hook(uint64_t *ctx) {   /* 0xe82924: the frame root enqueues this frame's render job (main thread) */
    (void)ctx; g_fifo[g_fifo_w % 64] = E.frame; __atomic_store_n(&g_fifo_w, g_fifo_w + 1, __ATOMIC_RELEASE);
}
static double g_frame_dt = -1;            /* game time of the current frame (s); -1 = native clock */
static int g_last_ticks = 4, g_cl_ticks_this_frame;
static int g_cl_pending;                  /* a step ended in the previous frame: publish its state at this frame's start */
static void cl_release(void) { if (__atomic_sub_fetch(&g_cl_parts, 1, __ATOMIC_ACQ_REL) == 0) { __atomic_store_n(&H->resp_seq, g_cl_seq, __ATOMIC_RELEASE); bell_ring(); } }
static void cl_step_done(uint32_t seq, int pixels) {   /* end of a step's ticks (main thread, frame f) */
    g_cl_seq = seq; g_cl_pending = 1;
    __atomic_store_n(&g_cl_parts, pixels ? 2 : 1, __ATOMIC_RELEASE);
    if (pixels) { g_cl_cap_frame = E.frame; __atomic_store_n(&g_cl_cap, 1, __ATOMIC_RELEASE); }
}
static void env_frame_ticks(void *sim, void *x2) {
    g_cl_ticks_this_frame = 0;
    if (E.reset_req) return;
    if (!E.active) {   /* the PLAY battle: serve commands until a RESET arrives */
        H->status = NBE_ST_READY;
        for (;;) {
            wait_cmd(); uint32_t seq = __atomic_load_n(&H->cmd_seq, __ATOMIC_ACQUIRE); NbeCmd c = *CMD; g_served = seq;
            if (c.type == NBE_CMD_QUIT) _exit(0);
            if (c.type == NBE_CMD_RESET) { hits_mark(); g_hold_mask = c.hold_mask; g_native_mask = c.native_mask; lead_set(&c);
                E.cur = c; E.reset_req = 1; E.have_reset = 1; E.reset_seed = c.seed & 0x7fffffff;
                E.act_tick = c.act_tick > 0 ? c.act_tick : 200; E.gas_on = c.gas; E.t_step0 = tnow(); return; }
            E.cur = c; E.cur_flags = 0; publish(sim, seq, c.type == NBE_CMD_PING ? 0 : -1, c.type == NBE_CMD_PING ? NULL : "no episode: send RESET first");
        }
    }
    int tick = I32(sim, 0x10);
    if (g_intro_render >= 0 || !g_loading_render) g_no_draw = intro_skip();   /* this frame runs the battle: not a loading frame */
    if (E.intro) {
        if (!E.t_tick1) { E.t_tick1 = tnow(); static int nl; if (nl++ < 3) { prof_dump("client reset: build end -> first intro tick (loading)"); prof_begin(); } }
        if (tick < E.act_tick) { int n = E.act_tick - tick; if (n > INTRO_TICKS_PER_FRAME) n = INTRO_TICKS_PER_FRAME;
            for (int i = 0; i < n; i++) tick1(sim, x2); g_cl_ticks_this_frame = n; return; }
        /* the reset observation: an empty 1-tick step */
        { extern int g_memstat; static int nmem; if (g_memstat && ++nmem % 20 == 0) { char lb[64]; snprintf(lb, sizeof lb, "client reset %d", nmem);
            extern void heap_memstat_dump(const char *), jni_stats_log(void); heap_memstat_dump(lb); jni_stats_log(); struct mallinfo2 mi = mallinfo2();
            LOG("[memstat] glibc arena %zu MB in use %zu MB free %zu MB mmap %zu MB\n", mi.arena >> 20, mi.uordblks >> 20, mi.fordblks >> 20, mi.hblkhd >> 20); } }
        { static int nres; double t = tnow(); if (nres < 3) prof_dump("client reset: first intro tick -> intro end (intro)");
          if (++nres <= 5 || nres % 50 == 0)
            LOG("[env] client reset %d: cmd->inject %.0f ms, inject->build %.0f ms, build->first intro tick %.0f ms, intro %.0f ms (frame %d, undrawn so far %ld, loading frames so far %ld)\n", nres,
                (E.t_inject - E.t_step0) * 1e3, (E.t_build1 - E.t_inject) * 1e3, (E.t_tick1 - E.t_build1) * 1e3, (t - E.t_tick1) * 1e3, E.frame, g_frames_undrawn, g_load_frames); }
        E.intro = 0; E.cur_flags = E.cur.flags | NBE_F_MAP; ep_clear_step(); E.step_start = tick; gas_latch_reset(tick); g_rec_n = 0; g_rec_lost = 0;
        for (int i = 0; i < NBE_MAXP; i++) { E.moving[i] = 0; E.last_dir[i] = -1; E.sent_valid[i] = 0; }
        tick1(sim, x2); g_cl_ticks_this_frame = 1;
        cl_step_done(g_served, (E.cur_flags & NBE_F_PIXELS) != 0);
        return;
    }
    H->status = NBE_ST_RUNNING;
    if (__atomic_load_n(&g_cl_cap, __ATOMIC_ACQUIRE) == 1) {   /* render job f-1 (already submitted) is presented by the render
        thread independently; wait for its swap (the game's own frame wait is not tied to it) */
        double t0 = tnow();
        while (__atomic_load_n(&g_cl_cap, __ATOMIC_ACQUIRE) == 1 && tnow() - t0 < 3.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); }
        if (__atomic_exchange_n(&g_cl_cap, 0, __ATOMIC_ACQ_REL) == 1) { R->pix_bytes = 0; LOG("[env] no swap within 3 s for the captured frame: pixels missing\n"); cl_release(); }
    }
    for (;;) {
        wait_cmd(); uint32_t seq = __atomic_load_n(&H->cmd_seq, __ATOMIC_ACQUIRE); NbeCmd c = *CMD; g_served = seq;
        g_hold_mask = c.hold_mask; g_native_mask = c.native_mask; lead_set(&c);
        if (c.type == NBE_CMD_QUIT) { hits_dump("quit"); LOG("[env] QUIT\n"); _exit(0); }
        if (c.type == NBE_CMD_PING) { E.cur = c; E.cur_flags = c.flags; publish(sim, seq, 0, NULL); continue; }
        if (c.type == NBE_CMD_SETUP) { g_setup = c; g_setup_pending = 1; E.cur = c; E.cur_flags = 0; publish(sim, seq, 0, NULL); continue; }
        if (c.type == NBE_CMD_RESET) {
            E.cur = c; E.reset_req = 1; E.have_reset = 1; E.reset_seed = c.seed & 0x7fffffff; E.act_tick = c.act_tick > 0 ? c.act_tick : 200;
            E.gas_on = c.gas; E.t_step0 = tnow(); return;
        }
        if (c.type == NBE_CMD_STEP) {
            if (U8(sim, 0xa1) || U8(sim, 0x121) || U8(sim, 0xa3)) { E.cur = c; E.cur_flags = c.flags; publish(sim, seq, -2, "episode done: RESET"); continue; }
            E.cur = c; E.cur_flags = c.flags; E.t_step0 = tnow(); g_ctl_mask = c.ctl_mask; E.step_start = tick; gas_latch_step(tick, c.gas);
            int n = c.ticks > 0 && c.ticks <= 400 ? (int)c.ticks : 4; g_last_ticks = n;
            apply_actions(sim, &c);
            for (int i = 0; i < n; i++) tick1(sim, x2);
            g_cl_ticks_this_frame = n; E.steps_in_frame = 1;
            { static int ncs; if (++ncs == 20) prof_begin(); else if (ncs == 150) prof_dump("client mode: 130 steps"); }
            if (g_trace > 0) LOG("[trace] step done in frame %d (swaps so far %d)\n", E.frame, g_swapn);
            cl_step_done(seq, (c.flags & NBE_F_PIXELS) != 0);
            return;
        }
        E.cur = c; publish(sim, seq, -9, "unknown command");
    }
}
/* render thread: eglSwapBuffers of the frame about to be presented (render job f when the capture is armed) */
extern int nb_gl_capture(void *dst, int w, int h), nb_gl_capture_obs(void *dst, int w, int h);
static long g_captures; static double g_cap_us;
void env_on_swap(void) {
    g_swapn++;
    int fr = -1;
    if (g_fifo_r != __atomic_load_n(&g_fifo_w, __ATOMIC_ACQUIRE)) { fr = g_fifo[g_fifo_r % 64]; g_fifo_r++; }
    if (g_trace > 0 && E.active && !E.intro) { g_trace--; LOG("[trace] swap %d of frame %d at main frame %d cap %d\n", g_swapn, fr, E.frame, g_cl_cap); }
    if (!g_env_on || g_env_simonly || __atomic_load_n(&g_cl_cap, __ATOMIC_ACQUIRE) != 1) return;
    if (fr != g_cl_cap_frame) { if (fr > g_cl_cap_frame) g_fifo_bad++; return; }   /* not the step's frame (yet) */
    int w = E.cur.pix_w > 0 ? E.cur.pix_w : 300, h = E.cur.pix_h > 0 ? E.cur.pix_h : 135;
    if (w * h * 4 > NBE_PIX_CAP) { w = 300; h = 135; }
    int obsfmt = E.cur.reserved[2] == 1;   /* cmd reserved[2] = 1: policy observation (RGB top-left, PIL-exact BOX from the 300x135 capture) */
    double t0 = tnow(); int r = obsfmt ? nb_gl_capture_obs(PIXB, w, h) : nb_gl_capture(PIXB, w, h);
    g_cap_us += (tnow() - t0) * 1e6; g_captures++;
    R->pix_w = w; R->pix_h = h; R->pix_bytes = r > 0 ? w * h * (obsfmt ? 3 : 4) : 0; R->reserved[4] = obsfmt;
    if (__atomic_exchange_n(&g_cl_cap, 0, __ATOMIC_ACQ_REL) == 1) cl_release();
}
static void my_ticks(void *sim, int n, void *x2) {   /* BattleMode::update 0x9c31b8: bl 0xc2ccd0(sim, 1, NULL) */
    if (g_env_on && g_env_simonly) {   /* the PLAY battle runs natively until its intro is over (client fully set up) */
        if (g_play_ready || !sim) return;
        ((void (*)(void *, int, void *))B(0xc2ccd0))(sim, n, x2);
        if (I32(sim, 0x10) >= 260 && (g_joy_calls > 0 || I32(sim, 0x10) >= 600)) { g_play_ready = 1; g_real_battle = E.battle;
            LOG("[env] PLAY battle at tick %d (joystick updates %ld, storm args %d): entering the sim-only loop\n", I32(sim, 0x10), g_joy_calls, E.gas_args_ok); }
        return;
    }
    if (!g_env_on || sim != E.sim) { ((void (*)(void *, int, void *))B(0xc2ccd0))(sim, n, x2); return; }
    env_frame_ticks(sim, x2);
}
/* ---------------- sim-only mode: the main loop thread stays in env_loop_simonly() once the PLAY battle runs */
static uint8_t g_fake_battle[0x400] __attribute__((aligned(16))), g_fake_owner[0x100] __attribute__((aligned(16)));
static int32_t g_list_copy[64]; static struct { void *data; int32_t cap, count; } g_list_vec;
static void *g_listener_fn; typedef void (*fn_deliver)(void *, int, void *); static fn_deliver o_deliver;
static long g_deliver_skipped;   /* --keep-snap-streams: 0 (the game's leak) */
static int g_listener_is_ret;
/* Sim listener vtable+0x178 (tick, encoder): the tick (0xc29208..) allocates a ByteStream (0x28 B + a 500-B buffer) per tick,
 * serialises the snapshot into it through an encoder (0xc74b44), calls the listener, then detaches the stream from the
 * encoder (0xc74b60) and deletes only the encoder. In this offline battle the listener is libg+0x87a4ac = a bare `ret`, so
 * the stream is never freed: ~2,300 streams (~1.1 MB) per episode, the game's own leak (the phone has it too). When the
 * listener is that `ret`, we free the stream after it returns: 0xc5d888 (frees the buffer +0x18 if owned) then operator
 * delete 0x11a4d60 (= what the ByteStream's deleting destructor 0xc65780 does, minus the buffer it forgets). */
static void my_deliver(void *l, int tick, void *enc) {
    if (g_env_simonly && g_in_loop) g_deliver_skipped++;
    else o_deliver(l, tick, enc);
    if (g_free_snap_stream && g_listener_is_ret && enc) {
        void *bs = PTR(enc, 8);
        if (bs && VTOF(bs) == 0x12155c8) { ((void (*)(void *))B(0xc5d888))(bs); ((void (*)(void *))B(0x11a4d60))(bs); g_streams_freed++; }
    }
}
static void hook_listener(void *sim) {
    void *lst = sim ? PTR(sim, 0x20) : NULL;
    if (g_listener_fn || !lst) return;
    g_listener_fn = ((void **)PTR(lst, 0))[0x178 / 8];
    g_listener_is_ret = *(uint32_t *)g_listener_fn == 0xd65f03c0u;
    o_deliver = hook_replace(g_libg, (uint64_t)((uint8_t *)g_listener_fn - g_libg->base), my_deliver);
    LOG("[env] snapshot delivery %s hooked (%s; sim-only episodes do not feed the client; per-tick stream freed: %s)\n", addr_desc(g_listener_fn),
        g_listener_is_ret ? "a bare ret" : "NOT a bare ret", g_free_snap_stream && g_listener_is_ret ? "yes" : "no");
}
extern void *offline_build_raw(void *battle, void *list, int a2, int a3);
extern uint64_t g_build_args[4];
static void delete_sim(void *sim) {
    ((void (*)(void *))B(0xc24c4c))(sim); ((void (*)(void *))B(0x9bafe0))(sim); ((void (*)(void *))B(0x11a4d60))(sim);
}
static int sim_reset(int seed) {
    double t0 = tnow();
    void *old = E.sim != PTR(g_real_battle, 0x30) ? E.sim : NULL;
    if (old) { E.sim = NULL; delete_sim(old); }   /* first: the game deletes the old sim before building (global registries) */
    double tdel = tnow();
    E.have_reset = 1; E.reset_seed = seed;
    memset(g_fake_battle, 0, sizeof g_fake_battle); memset(g_fake_owner, 0, sizeof g_fake_owner);
    PTR(g_fake_battle, 0x58) = g_fake_owner; PTR(g_fake_battle, 0x28) = PTR(g_real_battle, 0x28);
    memcpy(g_list_copy, g_build_list, sizeof g_list_copy);
    g_list_vec.data = g_list_copy; g_list_vec.cap = g_build_nlist; g_list_vec.count = g_build_nlist;
    static int prof_build = -1; if (prof_build < 0) prof_build = getenv("NBHOST_PROF_BUILD") != NULL;   /* debug: profile the builds */
    extern void prof_resume(void), prof_pause(void);
    if (prof_build) { static int nb; if (nb++ == 0) prof_begin(); else prof_resume(); }
    void *sim = offline_build_raw(g_fake_battle, g_build_args[0] ? &g_list_vec : NULL, (int)g_build_args[1], (int)g_build_args[2]);
    if (prof_build) { static int nb2; prof_pause(); if (++nb2 == 60) prof_dump("sim-only: 60 builds"); }
    if (!sim) { LOG("[env] sim-only build failed\n"); return -1; }
    hook_listener(sim);
    double t1 = tnow(), t2 = t1;
    extern void *g_sim; g_sim = sim;
    env_on_build(g_real_battle, sim);
    E.active = 1; E.intro = 1;
    E.t_inject = t0; E.t_build0 = tdel; E.t_build1 = t1; E.t_tick1 = t2; E.t_del = tdel - t0;
    return 0;
}
/* BattleMode::update 0x9c30c0 runs, once per client frame before the input drain and the ticks, 0xc2a064(sim): frees
 * the pending list and removes destroyed objects (0xb63798). The phone's lockstep frame = one 4-tick step, so the
 * sim-only loop calls it once per step (and once per 8 intro ticks, the phone's speed-8 frames). */
static void frame_cleanup(void *sim) { ((void (*)(void *))B(0xc2a064))(sim); }
static void env_loop_simonly(void) {
    void *sim = NULL;
    LOG("[env] sim-only loop entered (frame %d)\n", E.frame);
    { extern void heap_memstat_dump(const char *); heap_memstat_dump("sim-only loop entry"); }
    { extern size_t heap_quarantine_bytes(void); size_t q = heap_quarantine_bytes(); malloc_trim(0);
      LOG("[env] malloc_trim done (quarantine holds %zu MB)\n", q >> 20); }
    for (;;) {
        sim = E.sim;
        if (E.active && E.intro) {
            if (I32(sim, 0x10) < E.act_tick) { if (I32(sim, 0x10) % 8 == 0) { frame_cleanup(sim); E.step_start = I32(sim, 0x10); } tick1(sim, NULL); continue; }
            { double t = tnow(); if (E.gen <= 3 || E.gen % 100 == 0) LOG("[env] reset %d: delete old %.0f ms, build %.0f ms, intro %.0f ms (registry lookups %ld, indexed %ld)\n", E.gen, E.t_del * 1e3,
                (E.t_build1 - E.t_build0) * 1e3, (t - E.t_tick1) * 1e3, g_acc694_calls, g_ax_hits); }
            { extern int g_memstat; if (g_memstat && E.gen % 20 == 0) { char lb[64]; snprintf(lb, sizeof lb, "reset %d", E.gen);
                extern void heap_memstat_dump(const char *), jni_stats_log(void); heap_memstat_dump(lb); jni_stats_log(); struct mallinfo2 mi = mallinfo2();
                extern size_t heap_quarantine_bytes(void);
                LOG("[memstat] glibc arena %zu MB in use %zu MB free %zu MB mmap %zu MB quarantine %zu MB\n", mi.arena >> 20, mi.uordblks >> 20,
                    mi.fordblks >> 20, mi.hblkhd >> 20, heap_quarantine_bytes() >> 20); } }
            E.intro = 0; E.cur_flags = E.cur.flags | NBE_F_MAP; E.step_left = 1; ep_clear_step();
            for (int i = 0; i < NBE_MAXP; i++) { E.moving[i] = 0; E.last_dir[i] = -1; E.sent_valid[i] = 0; }
            frame_cleanup(sim); E.step_start = I32(sim, 0x10); gas_latch_reset(E.step_start); g_rec_n = 0; g_rec_lost = 0;
        }
        if (E.active && E.step_left > 0) {
            tick1(sim, NULL);
            if (--E.step_left == 0) { E.frame++; H->heartbeat++; publish(sim, g_served, 0, NULL);
                static int nst; if (!getenv("NBHOST_PROF_BUILD")) { if (++nst == 50) prof_begin(); else if (nst == 3050) prof_dump("sim-only: 3000 steps"); } }
            continue;
        }
        H->status = E.active ? NBE_ST_RUNNING : NBE_ST_READY;
        wait_cmd();
        uint32_t seq = __atomic_load_n(&H->cmd_seq, __ATOMIC_ACQUIRE); NbeCmd c = *CMD; g_served = seq;
        g_hold_mask = c.hold_mask; g_native_mask = c.native_mask; lead_set(&c);
        if (c.type == NBE_CMD_QUIT) { hits_dump("quit"); LOG("[env] QUIT\n"); _exit(0); }
        if (c.type == NBE_CMD_PING) { E.cur = c; E.cur_flags = c.flags; publish(E.sim, seq, 0, NULL); continue; }
        if (c.type == NBE_CMD_JOYTEST) { E.cur = c; E.cur_flags = 0; g_publish_defer = 1; publish(E.sim, seq, 0, NULL); g_publish_defer = 0;
            R->state_ints = joytest(E.sim, c.set_x[0], c.set_y[0], ST); __atomic_store_n(&H->resp_seq, seq, __ATOMIC_RELEASE); bell_ring(); continue; }
        if (c.type == NBE_CMD_SETUP) { g_setup = c; g_setup_pending = 1; E.cur = c; E.cur_flags = 0; publish(E.sim, seq, 0, NULL); continue; }
        if (c.type == NBE_CMD_RESET) {
            hits_mark(); E.cur = c; E.act_tick = c.act_tick > 0 ? c.act_tick : 200; E.gas_on = c.gas; E.t_step0 = tnow();
            if (sim_reset(c.seed & 0x7fffffff)) { publish(E.sim, seq, -20, "sim build failed"); continue; }
            E.cur = c; E.gas_on = c.gas; E.act_tick = c.act_tick > 0 ? c.act_tick : 200;
            continue;
        }
        if (c.type == NBE_CMD_STEP) {
            if (!E.active) { E.cur = c; E.cur_flags = 0; publish(E.sim, seq, -1, "no episode: send RESET first"); continue; }
            if (c.flags & NBE_F_PIXELS) { E.cur = c; E.cur_flags = c.flags; publish(sim, seq, -3, "pixels need --env-client"); continue; }
            if (U8(sim, 0xa1) || U8(sim, 0x121) || U8(sim, 0xa3)) { E.cur = c; E.cur_flags = c.flags; publish(sim, seq, -2, "episode done: RESET"); continue; }
            E.cur = c; E.cur_flags = c.flags; E.t_step0 = tnow();
            g_ctl_mask = c.ctl_mask;
            frame_cleanup(sim);          /* frame start: object cleanup, then the inputs (the drain), then the ticks */
            E.step_start = I32(sim, 0x10); gas_latch_step(E.step_start, c.gas);
            apply_actions(sim, &c);
            E.step_left = c.ticks > 0 && c.ticks <= 400 ? (int)c.ticks : 4;
            continue;
        }
        E.cur = c; publish(E.sim, seq, -9, "unknown command");
    }
}
extern void prof_set_thread(int), prof_begin(void), prof_dump(const char *);
/* ---------------- client mode: the phone agent's 'app' clock (lockstep frames of exactly the step's game time) */
static int64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
static int64_t g_app_tns; static int g_app_tns_valid, g_frame_plan;
static int cl_expected_plan(void) {
    if (!E.active || E.reset_req || !E.sim) return 0;
    if (E.intro) { int tick = I32(E.sim, 0x10); if (tick < E.act_tick) { int n = E.act_tick - tick; return n > INTRO_TICKS_PER_FRAME ? INTRO_TICKS_PER_FRAME : n; } return 1; }
    return g_last_ticks;
}
static void cl_frame_start(uint8_t *app) {
    { extern int g_trace_n; static int once; if (!once) { once = 1; } (void)g_trace_n; }
    if (g_cl_pending && E.sim) {   /* the state part of the last step's response (after its frame's client update) */
        g_cl_pending = 0; g_publish_defer = 1; publish(E.sim, g_cl_seq, 0, NULL); g_publish_defer = 0; cl_release();
    }
    int plan = cl_expected_plan(); g_frame_plan = plan;
    if (plan <= 0 || !app) { g_frame_dt = -1; g_app_tns_valid = 0; return; }
    int64_t now = now_ns(), ns = (int64_t)plan * 50000000LL;
    *(int64_t *)(app + 0x48) = now - ns;                       /* native dt = now - [app+0x48] */
    if (!g_app_tns_valid) { g_app_tns = now - *(int64_t *)(app + 0x40); g_app_tns_valid = 1; } else g_app_tns += ns;
    *(int64_t *)(app + 0x40) = now - g_app_tns;                /* absolute app time advances by exactly the game time */
    g_frame_dt = ns / 1e9;
}
typedef void (*fn_cupd)(void *, float, float);
static fn_cupd o_cupd; static struct { void *c; float t, dt; double fd; int pending; } g_defer;
static long g_cs_calls, g_cs_sub;
static void cs_run(void *c, float t, float dt, double fd) {   /* client_substeps: round(fd*60) updates of 1/60 s */
    /* no cap below the frame's game time: with longer intro frames (--intro-tpf) a 64 cap dropped client time (T4) */
    int n = (int)llround(fd * 60); if (n < 1) n = 1; if (n > 4096) n = 4096;
    double sub = fd / n, t0 = t - fd, mfd = fd < 0.1 ? fd : 0.1, ds = dt > 0 ? (sub < 0.1 ? sub : 0.1) * (dt / mfd) : 0;
    g_cs_calls++; g_cs_sub += n;
    for (int i = 1; i <= n; i++) o_cupd(c, (float)(t0 + i * sub), (float)ds);
}
static void my_cupd(void *c, float t, float dt) {   /* 0xc210b0 client battle update (once per frame by BattleMode) */
    if (g_env_simonly || !(g_frame_dt > 0)) { o_cupd(c, t, dt); return; }
    if (g_frame_plan > 0) { g_defer.c = c; g_defer.t = t; g_defer.dt = dt; g_defer.fd = g_frame_dt; g_defer.pending = 1; return; }   /* client_defer */
    cs_run(c, t, dt, g_frame_dt);
}
static void defer_hook(uint64_t *ctx) {   /* 0x9c32a4: after this frame's ticks, before the BattleScreen update draws */
    (void)ctx; if (!g_defer.pending) return; g_defer.pending = 0; cs_run(g_defer.c, g_defer.t, g_defer.dt, g_defer.fd);
}
typedef void (*fn_cam)(void *, void *, void *, void *, int, int, float, float);
static fn_cam o_cam; static double g_cam_carry; static void *g_screen;
static void my_cam(void *scr, void *a1, void *a2, void *a3, int a4, int a5, float dt, float zoom) {   /* 0x862bfc battle camera */
    g_screen = scr; g_cam_zoom_arg = zoom;
    if (g_env_simonly || g_frame_dt < 0) { g_cam_carry = 0; o_cam(scr, a1, a2, a3, a4, a5, dt, zoom); return; }
    g_cam_carry += g_frame_dt * 60; int n = (int)floor(g_cam_carry + 1e-6); g_cam_carry -= n; if (n > 4096) { n = 4096; g_cam_carry = 0; }
    for (int i = 0; i < n; i++) o_cam(scr, a1, a2, a3, a4, a5, 1.0f / 60, zoom);
}
typedef void (*fn_view)(void *, float); static fn_view o_view;
static void my_view(void *v, float dt) { if (!g_env_simonly && E.active && !(dt > 0)) return; o_view(v, dt); }   /* 0x83b030 dt<=0 guard */
typedef void (*fn_lag)(void *, float, float); static fn_lag o_lag;
static void my_lag(void *io, float a, float b) {   /* 0x7f003c input owner: no lag timer while the env drives the clock */
    o_lag(io, a, b);
    if (!g_env_simonly && E.active && io && *(float *)((uint8_t *)io + 0x3c) != 0) { *(float *)((uint8_t *)io + 0x3c) = 0; *(float *)((uint8_t *)io + 0x40) = 0; }
}
static void my_hud(uint64_t *ctx) {   /* 0x59cb7c HUD update: lag accumulator reset, hide a lag icon that is up */
    if (g_env_simonly || !E.active) return; uint8_t *hud = (uint8_t *)ctx[0]; if (!hud) return;
    if (*(float *)(hud + 0x694) != 0) *(float *)(hud + 0x694) = 0;
    uint8_t *ui = *(uint8_t **)B(0x12edc88); if (!ui) return; uint8_t *icon = PTR(ui, 0x1b8);
    if (icon && U8(icon, 8)) ((void (*)(void *))B(0x5c9588))(ui);
}
/* Client mode: frames that are not drawn. AndroidMain's frame (0xeaa9f0) draws only while app+0x84 (window focus) is set,
 * else it takes the game's own no-draw path (screen vt+0x38, as with an unfocused window); the update (ticks, client
 * substeps, camera) runs either way. Patch at 0xeaab68 (ldrb w8,[x19,#0x84]): g_no_draw != 0 -> 0xeaac58 (no draw).
 * --intro-render N: during a reset only the last N intro frames are drawn (-1: all frames, the patch is not installed);
 * --loading-render 0: the loading frames (inject -> battle built) are not drawn either. */
int g_intro_render = -1, g_loading_render = 1;
static int intro_skip(void) {
    return g_intro_render >= 0 && E.active && E.intro && E.sim && !E.reset_req && I32(E.sim, 0x10) + g_intro_render * INTRO_TICKS_PER_FRAME < E.act_tick;
}
/* frame start: a reset's loading frame (injected, or built but no intro tick yet) or not; a frame that runs ticks
 * overrides the decision in env_frame_ticks (the draw check comes after the frame's update) */
static void no_draw_decide(void) {
    g_frames_undrawn += g_no_draw;   /* the previous frame's final decision */
    int skip = 0;
    if ((g_intro_render >= 0 || !g_loading_render) && g_env_on && !g_env_simonly) {
        int loading = E.reset_req || (E.active && E.intro && E.sim && !E.t_tick1);
        skip = loading ? !g_loading_render : intro_skip();
        if (loading) g_load_frames++;
    }
    g_no_draw = skip;
}
/* f_acc694 (libg 0xacc694, ~100 callers): "the registry entry whose data object is `data`". A cache (libc++
 * unordered_map<int, entry*> at [0x12eff40], key f_ac3a80(data)) is probed first; on a miss the list at [0x12f0190] is
 * scanned (virtual count vt+0x20 / get vt+0x28) for the first entry e with e+0xd8 == data and e+0x100 == 0, which is then
 * inserted (emplace 0xad78a4). Misses are never cached, so each one walks the whole list with 2 virtual + 2 getter calls
 * per element; a sim-only reset's battle build spends nearly all its time here (f_acc4c0 calls it for every entry of
 * the list at [0x12f0158]). Exact C port of the same steps in the same order; the scan reads the list's array directly
 * when its getter is the plain array getter 0xac8bec (else it calls the virtual getter). --no-acc694: the original;
 * --acc694-check: run both and count disagreements (debug). */
int g_acc694 = 1, g_acc694_check; static long (*o_acc694)(long); static long g_acc694_bad;
static long acc694_cache(long data, int id, int *found) {   /* the cache probe: *found = an entry with this key exists */
    uint64_t *map = *(uint64_t **)B(0x12eff40); *found = 0;
    if (map && map[1]) {
        uint64_t bc = map[1], h = (uint64_t)(int64_t)id;
        int pow2 = __builtin_popcountll(bc) < 2;
        uint64_t idx = pow2 ? (h & (bc - 1)) : (bc > h ? h : h % bc);
        uint64_t *nd = ((uint64_t **)map[0])[idx];
        if (nd) for (nd = (uint64_t *)nd[0]; nd; nd = (uint64_t *)nd[0]) {
            uint64_t nh = nd[1];
            if (nh == h) { if ((int32_t)nd[2] == id) { *found = 1; return (long)nd[3]; } }
            else if ((pow2 ? (nh & (bc - 1)) : (nh < bc ? nh : nh % bc)) != idx) break;
        }
    }
    (void)data; return 0;
}
static long acc694_scan(long data, int id, int insert) {   /* the list scan (+ insertion into the cache) */
    void *list = *(void **)B(0x12f0190);
    int n = ((int (*)(void *))(*(void ***)list)[4])(list);
    for (int i = 0; i < n; i++) {
        void **vt = *(void ***)list;
        long e = vt[5] == (void *)B(0xac8bec) ? ((long *)PTR(list, 8))[i] : ((long (*)(void *, int))vt[5])(list, i);
        if (*(long *)(e + 0xd8) != data || *(int32_t *)(e + 0x100) != 0) continue;
        uint64_t *map = *(uint64_t **)B(0x12eff40);
        if (map && insert) { int key = id; int *tup = &key; char empty;
            uint8_t *node = ((uint8_t *(*)(void *, int *, void *, int **, char *))B(0xad78a4))(map, &key, B(0x1f858c), &tup, &empty);
            *(long *)(node + 0x18) = e; }
        return e;
    }
    return 0;
}
/* f_acc4c0 (0xacc4c0, the battle build's weighted list of the [0x12f0158] entries) calls f_acc694 for every entry: N x M
 * scans of the [0x12f0190] list. While it runs (it only reads that list), the list is indexed once: data pointer -> the
 * first entry with e+0xd8 == data and e+0x100 == 0, i.e. exactly what the scan returns. The index is used only on the
 * thread that built it, inside the outermost f_acc4c0 call, and is rebuilt if the list's count or array changes;
 * --acc694-check compares every indexed answer with a full scan. */
static struct { long *key, *val; int cap, on, tid, n; void *arr; } g_ax;
static void ax_build(void) {
    void *list = *(void **)B(0x12f0190); void **vt = *(void ***)list;
    g_ax.on = 0; if (vt[5] != (void *)B(0xac8bec)) return;   /* only for the plain array getter */
    int n = ((int (*)(void *))vt[4])(list); long *a = (long *)PTR(list, 8);
    int cap = 64; while (cap < 2 * n + 2) cap <<= 1;
    if (cap > g_ax.cap) { free(g_ax.key); free(g_ax.val); g_ax.key = malloc(cap * sizeof(long)); g_ax.val = malloc(cap * sizeof(long)); g_ax.cap = cap; }
    memset(g_ax.key, 0, g_ax.cap * sizeof(long));
    for (int i = 0; i < n; i++) {
        long e = a[i], d = *(long *)(e + 0xd8);
        if (!d || *(int32_t *)(e + 0x100) != 0) continue;
        unsigned h = (unsigned)(((uint64_t)d * 0x9E3779B97F4A7C15ULL) >> 40) & (g_ax.cap - 1);
        while (g_ax.key[h] && g_ax.key[h] != d) h = (h + 1) & (g_ax.cap - 1);
        if (!g_ax.key[h]) { g_ax.key[h] = d; g_ax.val[h] = e; }   /* keep the first match */
    }
    g_ax.n = n; g_ax.arr = a; g_ax.tid = gettid(); g_ax.on = 1;
}
static long ax_find(long d) {
    unsigned h = (unsigned)(((uint64_t)d * 0x9E3779B97F4A7C15ULL) >> 40) & (g_ax.cap - 1);
    while (g_ax.key[h]) { if (g_ax.key[h] == d) return g_ax.val[h]; h = (h + 1) & (g_ax.cap - 1); }
    return 0;
}
static __thread int g_ax_depth; static long (*o_acc4c0)(long, long, long);
static long my_acc4c0(long a, long b, long c) {
    int outer = g_ax_depth++ == 0;
    if (outer) ax_build();
    long r = o_acc4c0(a, b, c);
    if (outer) g_ax.on = 0;
    g_ax_depth--;
    return r;
}
static long my_acc694(long data) {
    g_acc694_calls++;
    if (!data) return 0;
    int id = ((int (*)(long))B(0xac3a80))(data), found;
    long c = acc694_cache(data, id, &found);
    if (g_acc694_check) {   /* the original decides (and inserts); compare with what the port would have returned */
        long want = c;
        if (!want) { want = acc694_scan(data, id, 0);
            if (g_ax.on && g_ax.tid == gettid()) { void *list = *(void **)B(0x12f0190);
                if (PTR(list, 8) == g_ax.arr && ((int (*)(void *))(*(void ***)list)[4])(list) == g_ax.n && ax_find(data) != want && g_acc694_bad++ < 20)
                    LOG("[acc694] INDEX MISMATCH data %#lx: index %#lx scan %#lx\n", data, ax_find(data), want); } }
        long o = o_acc694(data);
        if (o != want && g_acc694_bad++ < 20) LOG("[acc694] MISMATCH data %#lx: port %#lx original %#lx (cached %d)\n", data, want, o, found);
        return o;
    }
    if (c) return c;
    if (g_ax.on && g_ax.tid == gettid()) {
        void *list = *(void **)B(0x12f0190);
        if (PTR(list, 8) != g_ax.arr || ((int (*)(void *))(*(void ***)list)[4])(list) != g_ax.n) ax_build();
        if (g_ax.on) {
            long e = ax_find(data); g_ax_hits++;
            if (e) { uint64_t *map = *(uint64_t **)B(0x12eff40);
                if (map) { int key = id; int *tup = &key; char empty;
                    uint8_t *node = ((uint8_t *(*)(void *, int *, void *, int **, char *))B(0xad78a4))(map, &key, B(0x1f858c), &tup, &empty);
                    *(long *)(node + 0x18) = e; } }
            return e;
        }
    }
    return acc694_scan(data, id, 1);
}
/* Loading wait loop f_c84b3c (the main thread waits for a loader task): while (!isDone(task)) { run the main-thread
 * executor's queued jobs }. With no queued job it spins on isDone (f_d3434c: task+0x20 == 3), a quarter of the main
 * thread's loading time, a hardware thread taken from the other instances. The bl at 0xc84b64 calls this instead: the
 * same answer; when not done, 50 us of sleep before the loop polls again. --no-load-yield: off. */
int g_load_yield = 1; static long g_load_waits;
static int my_wait_done(void *task) {
    if (__atomic_load_n((int32_t *)((uint8_t *)task + 0x20), __ATOMIC_ACQUIRE) == 3) return 1;
    g_load_waits++; struct timespec ts = {0, 50000}; nanosleep(&ts, NULL);
    return 0;
}
/* Resource manager deadlock (the rare "loading deadlock", T5): f_dc1ebc (per-frame maintenance of the manager whose mutex
 * is at +0x68) holds that mutex while it promotes weak references (weak_ptr::lock 0x11a51c0); when another thread drops
 * its reference at that moment, the temporary shared_ptr is the last one, its destructor runs inside the critical section
 * and calls f_dbfa1c(manager, id), which locks the same mutex: self-deadlock (hang dumps: the mutex's owner = the waiting
 * thread). A race in the game (tiny on the phone, wide under emulation: 1 per ~1,300 pixel resets). f_dbfa1c called by
 * a thread that already owns manager+0x68 is deferred (in order, same thread) until f_dc1ebc returns — as if the other
 * thread's release had come a moment later; every other call is unchanged. --no-release-defer: off. */
int g_release_defer = 1; static long g_releases_deferred;
static void (*o_dbfa1c)(void *, uint32_t); static void (*o_dc1ebc)(void *);
static __thread struct { void *m; uint32_t id; } t_pend[64]; static __thread int t_npend;
extern int bionic_mutex_owner(void *bm);
static void flush_releases(void) {
    for (int i = 0; i < t_npend; i++) o_dbfa1c(t_pend[i].m, t_pend[i].id);
    t_npend = 0;
}
static void my_dbfa1c(void *m, uint32_t id) {
    if (m && bionic_mutex_owner((uint8_t *)m + 0x68) == gettid() && t_npend < 64) {
        t_pend[t_npend].m = m; t_pend[t_npend].id = id; t_npend++; g_releases_deferred++;
        if (g_releases_deferred <= 20) LOG("[env] release of resource %u deferred (its manager's lock is held by this thread)\n", id);
        return;
    }
    if (t_npend) flush_releases();
    o_dbfa1c(m, id);
}
static void my_dc1ebc(void *m) { o_dc1ebc(m); if (t_npend) flush_releases(); }
/* debug (NBHOST_TRACE_MBANK=1): streaming matrix-bank chunk loads (0xcc0ffc, worker threads): file, offset, sizes */
static void mbank_hook(uint64_t *ctx) {
    uint8_t *o = (uint8_t *)ctx[0]; if (!o || (I32(o, 0x34) & 3) != 1) return;
    const char *fn = (U8(o, 0xd8) & 1) ? (const char *)PTR(o, 0xe8) : (const char *)(o + 0xd9);
    LOG("[mbank] obj %p frame %d file %s off %u csize %u dsize %u\n", (void *)o, E.frame, fn, (unsigned)I32(o, 0xf0), (unsigned)I32(o, 0xf4), (unsigned)I32(o, 0xf8));
}
static void frame_hook(uint64_t *ctx) {   /* AndroidMain frame 0xeaa9f0 entry (main thread, before any game logic) */
    E.frame++; H->heartbeat++;
    if (!g_env_simonly && g_env_on) { cl_frame_start((uint8_t *)ctx[0]); no_draw_decide(); }
    { static int once; if (!once) { once = 1; prof_set_thread(gettid()); } }
    if (g_env_simonly && g_play_ready && !g_in_loop) { g_in_loop = 1; env_loop_simonly(); }
    if (E.reset_req == 1) {
        E.reset_req = 2;   /* injected, waiting for the build */
        prof_begin(); E.t_inject = tnow();
        int r = inject_reset(E.reset_seed);
        if (r) { E.reset_req = 0; if (E.sim) publish(E.sim, g_served, -10 + r, "reset injection failed"); }
    }
}
void env_after_build(void *battle) {   /* builder returned: the new battle's sim */
    void *sim = PTR(battle, 0x30);
    int was = E.reset_req; double ti = E.t_inject, tb = E.t_build0, ts = E.t_step0; env_on_build(battle, sim); E.t_inject = ti; E.t_build0 = tb; E.t_build1 = tnow();
    E.t_step0 = ts;
    if (was) { E.reset_req = 0; E.active = 1; E.intro = 1; }
    if (!H->boot_ms) H->boot_ms = (int)((tnow() - g_t0) * 1000);
    prof_dump("inject -> build end"); prof_begin();
}

/* ------------------------------------------------------------------ setup */
int env_attach(const char *path) {
    int fd = open(path, O_RDWR); if (fd < 0) { LOG("[env] cannot open %s\n", path); return -1; }
    struct stat st; fstat(fd, &st);
    if (st.st_size < NBE_TOTAL) { LOG("[env] %s too small (%ld < %d)\n", path, (long)st.st_size, NBE_TOTAL); return -1; }
    uint8_t *m = mmap(NULL, NBE_TOTAL, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); close(fd);
    if (m == MAP_FAILED) { LOG("[env] mmap failed\n"); return -1; }
    H = (NbeHdr *)m; CMD = (NbeCmd *)(m + NBE_OFF_CMD); R = (NbeResp *)(m + NBE_OFF_RESP);
    ST = (int32_t *)(m + NBE_OFF_STATE); MAPB = (int32_t *)(m + NBE_OFF_MAP); RECB = (int32_t *)(m + NBE_OFF_REC);
    PIXB = m + NBE_OFF_PIX; OBSB = (float *)(m + NBE_OFF_OBS);
    H->magic = NBE_MAGIC; H->version = NBE_VERSION; H->total_size = NBE_TOTAL; H->hdr_size = sizeof(NbeHdr);
    H->off_cmd = NBE_OFF_CMD; H->off_resp = NBE_OFF_RESP; H->off_state = NBE_OFF_STATE; H->off_map = NBE_OFF_MAP;
    H->off_rec = NBE_OFF_REC; H->off_pix = NBE_OFF_PIX; H->off_obs = NBE_OFF_OBS; H->state_cap = NBE_STATE_CAP;
    H->map_cap = NBE_MAP_CAP; H->rec_cap = NBE_REC_CAP; H->pix_cap = NBE_PIX_CAP; H->obs_vec = NBE_OBS_VEC; H->obs_grid = NBE_OBS_GRID; H->maxp = NBE_MAXP;
    H->pid = getpid(); H->status = NBE_ST_BOOT;
    g_served = __atomic_load_n(&H->cmd_seq, __ATOMIC_ACQUIRE);
    g_env_on = 1; E.gas_on = 1; E.act_tick = 200;
    LOG("[env] attached %s (%d bytes, cmd %zu B, resp %zu B)\n", path, NBE_TOTAL, sizeof(NbeCmd), sizeof(NbeResp));
    return 0;
}
void env_mark_crash(int sig) { if (H) H->crashed = sig; }
/* --count-hits FILE (T6a): hit counters on a list of libg addresses (the phone's WildCat-patched ranges), dumped to the
 * log at every reset and on QUIT, to show which patched code runs in a PC battle at all. Debug only. */
#include "hits_gen.h"
const char *g_hits_file;
static void hits_install(void) {
    if (!g_hits_file) return;
    FILE *f = fopen(g_hits_file, "r"); if (!f) { LOG("[hits] cannot open %s\n", g_hits_file); return; }
    char line[64];
    while (fgets(line, sizeof line, f) && g_nhit < 512) {
        uint64_t a = strtoull(line, NULL, 16); if (!a) continue;
        uint32_t insn = *(uint32_t *)B(a);   /* hook_enter cannot relocate these first instructions: not instrumented */
        if ((insn & 0x9f000000) == 0x90000000 || (insn & 0x7c000000) == 0x14000000 || (insn & 0x3b000000) == 0x18000000 ||
            (insn & 0x9f000000) == 0x10000000) {
            uint32_t i2 = *(uint32_t *)B(a + 4);   /* count at the second instruction (same basic block) */
            if ((i2 & 0x9f000000) == 0x90000000 || (i2 & 0x7c000000) == 0x14000000 || (i2 & 0x3b000000) == 0x18000000 ||
                (i2 & 0x9f000000) == 0x10000000 || (i2 & 0x7e000000) == 0x34000000 || (i2 & 0x7e000000) == 0x36000000 || (i2 & 0xff000010) == 0x54000000) {
                LOG("[hits] %lx not instrumented (first two instructions %08x %08x are PC-relative/branches)\n", (unsigned long)a, insn, i2); continue; }
            a += 4;
        }
        g_hit_addr[g_nhit] = a; hook_enter(g_libg, a, g_hc[g_nhit]); g_nhit++;
    }
    fclose(f); LOG("[hits] %d counters installed\n", g_nhit);
}
static long g_hits0[512]; static int g_hits_snap;
static void hits_mark(void) {   /* first RESET: what ran before = boot, lobby, the PLAY battle's start */
    if (!g_nhit || g_hits_snap) return;
    memcpy(g_hits0, g_hits, sizeof g_hits0); g_hits_snap = 1;
    int nz = 0; for (int i = 0; i < g_nhit; i++) nz += g_hits[i] > 0;
    LOG("[hits] before the first reset (boot + lobby + PLAY battle start): %d of %d addresses executed\n", nz, g_nhit);
}
void hits_dump(const char *why) {
    if (!g_nhit) return;
    int nz = 0; for (int i = 0; i < g_nhit; i++) nz += g_hits[i] - g_hits0[i] > 0;
    LOG("[hits] %s: %d of %d addresses executed since the first reset (env episodes):", why, nz, g_nhit);
    for (int i = 0; i < g_nhit; i++) if (g_hits[i] - g_hits0[i]) LOG(" %lx=%ld", (unsigned long)g_hit_addr[i], g_hits[i] - g_hits0[i]);
    LOG("\n");
}
void env_install(void) {
    if (!g_env_on) return;
    KIND_BY_TYPE = B(0x185710);
    /* B3 determinism patch: 0x9bb864 cbnz w8,#0x9bb890 -> b #0x9bb890 (keep the seed; no srand(time)/rand reseed) */
    if (*(uint32_t *)B(0x9bb864) != 0x35000168) { LOG("[env] seed patch: unexpected bytes %08x\n", *(uint32_t *)B(0x9bb864)); exit(1); }
    patch_u32(g_libg, 0x9bb864, 0x1400000b);
    hook_bl(g_libg, 0x9c31b8, my_ticks);
    hook_bl(g_libg, 0x963488, my_schedule_play);
    hook_bl(g_libg, 0x9bb7c4, my_setseed);
    hook_bl(g_libg, 0x50a7a0, my_modlist);
    hook_bl(g_libg, 0x86afe8, my_gastest_call);
    o_gastest = (fn_gas)B(0xc69094);
    hook_enter(g_libg, 0xeaa9f0, frame_hook);
    o_snapw = hook_replace(g_libg, 0xb63b0c, my_snapw);
    hook_enter(g_libg, 0xb475cc, sf_start_hook);
    hook_enter(g_libg, 0xb487c4, my_skill_block);
    hook_enter(g_libg, 0xb4790c, rec_move);
    hook_enter(g_libg, 0xb4103c, rec_stop);
    hook_enter(g_libg, 0x863d88, my_joy_enter);
    o_root = hook_replace(g_libg, 0xb422d0, my_root);
    o_charge = hook_replace(g_libg, 0xc420d8, my_charge);
    o_alive = hook_replace(g_libg, 0xc42ae8, my_alive);
    o_dispatch = hook_replace(g_libg, 0xb47084, my_dispatch);
    if (!g_env_simonly) {
        /* client-mode crash fix (T4/T5): 0xdc1ec0 walks a slot table of weak pointers (x19+0xd0); when lock() 0x11a51c0 fails
         * for an entry that was not yet marked expired (freed concurrently, seen during a reset's loading screen: SIGSEGV
         * at 0xdc1fb0 reading null+0xb6) it dereferences the null result. Patch: 0xdc1f64 cbz x0,#0xdc1f8c -> cbz x0,#0xdc1f10
         * (skip the entry this frame; the expired-entry path frees it later). Byte-checked, reversible, client mode only. */
        if (!g_weak_skip) LOG("[env] patch 0xdc1f64 disabled (--no-weak-skip)\n");
        else if (*(uint32_t *)B(0xdc1f64) == 0xb4000140) { patch_u32(g_libg, 0xdc1f64, 0xb4fffd60); LOG("[env] patch 0xdc1f64 (weak-lock null skip) applied\n"); }
        else LOG("[env] patch 0xdc1f64: unexpected bytes %08x, not applied\n", *(uint32_t *)B(0xdc1f64));
    }
    if (!g_env_simonly) {   /* client mode: the phone agent's lockstep clock for the client and the renderer */
        o_cupd = hook_replace(g_libg, 0xc210b0, my_cupd);
        hook_enter(g_libg, 0x9c32a4, defer_hook);
        o_cam = hook_replace(g_libg, 0x862bfc, my_cam);
        o_view = hook_replace(g_libg, 0x83b030, my_view);
        o_lag = hook_replace(g_libg, 0x7f003c, my_lag);
        hook_enter(g_libg, 0x59cb7c, my_hud);
        hook_enter(g_libg, 0xe82924, submit_hook);
        if (g_cube_counter) cube_counter_install();
        if (g_intro_render >= 0 || !g_loading_render) { int r = hook_flag_skip(g_libg, 0xeaab68, 0x39421268, &g_no_draw, 0xeaac58);
            LOG("[env] no-draw patch 0xeaab68 (intro frames drawn: last %d, loading frames drawn: %d): %s\n", g_intro_render, g_loading_render, r ? "NOT applied" : "applied"); }
    }
    (void)g_cube_tbl_dummy;
    if (getenv("NBHOST_TRACE_MBANK")) hook_enter(g_libg, 0xcc0ffc, mbank_hook);
    if (g_release_defer) { o_dbfa1c = hook_replace(g_libg, 0xdbfa1c, my_dbfa1c); o_dc1ebc = hook_replace(g_libg, 0xdc1ebc, my_dc1ebc); }
    if (g_load_yield) { extern void hook_bl(Module *, uint64_t, void *); uint32_t w = *(uint32_t *)B(0xc84b64);
        if (w == 0x9402bdfa) { hook_bl(g_libg, 0xc84b64, my_wait_done); LOG("[env] loading wait loop 0xc84b64: sleeps when idle\n"); }
        else LOG("[env] loading wait loop: unexpected bytes %08x, not patched\n", w); }
    if (g_acc694) { o_acc694 = hook_replace(g_libg, 0xacc694, my_acc694); o_acc4c0 = hook_replace(g_libg, 0xacc4c0, my_acc4c0); LOG("[env] registry lookup 0xacc694: C port%s\n", g_acc694_check ? " (checked against the original)" : ""); }
    hits_install();
    /* Item update 0xb6b8e0 (vtable 0x1209240 +0x28), arrival of an item flying to a target (+0x90 set, target +0xa8): two
     * virtual calls on the target without a null check. The target pointer was NULL in two soak crashes (SIGSEGV at
     * 0xb6df7c, sim-only and client mode, ~1 per 5k episodes with random actions). Guards: NULL -> the game's own
     * "target not alive" paths (0xb6defc -> 0xb6df68, 0xb6df78 -> 0xb6d154). Byte-checked; --no-item-guard: off. */
    if (g_item_guard) {
        int r1 = hook_null_skip(g_libg, 0xb6defc, 0xf9405660, 0, 0xb6df68), r2 = hook_null_skip(g_libg, 0xb6df78, 0xf9405660, 0, 0xb6d154);
        LOG("[env] item target null guards 0xb6defc/0xb6df78: %s\n", (r1 || r2) ? "NOT applied" : "applied");
    }
    LOG("[env] hooks installed (seed patch on, %d steps per client frame)\n", g_env_spf);
}
