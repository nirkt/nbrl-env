/* C port of the phone's offline2.js (inject LoginOk / OwnHomeData on the connection thread) plus state probes.
 * All offsets are libg file offsets (v68.279, BuildID bd358ea3...), the same ones the Frida agents use. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

extern Module *g_libg;
void *hook_replace(Module *m, uint64_t off, void *repl);
#define B(off) ((uint8_t *)g_libg->base + (off))
#define RD64(p) (*(uint64_t *)(p))
#define RD32(p) (*(int32_t *)(p))

enum { O_recv = 0x80f654, O_connUpd = 0x81ba20, O_mgrSend = 0x80f56c, O_msgSend = 0xe85328, O_homeDecode = 0xc01d34,
       O_installHome = 0x9c4f84, O_loadWatch = 0x9cd824, O_createMsg = 0xbc1628, O_llmfVT = 0x120bb20,
       O_mgrSlot = 0x12ee3b0, O_connHolder = 0x12380a0 };

static uint8_t *pl[32]; static int pl_len[32]; static int pl_id[32]; static int npl;
static int load_payload(const char *dir, int id) {
    char p[1024]; snprintf(p, sizeof p, "%s/%d.bin", dir, id);
    FILE *f = fopen(p, "rb"); if (!f) { LOG("[offline] no payload %s\n", p); return -1; }
    uint8_t *b = malloc(1 << 20); int n = fread(b, 1, 1 << 20, f); fclose(f);
    pl[npl] = b; pl_len[npl] = n; pl_id[npl] = id; return npl++;
}
static int payload(int id, uint8_t **b) { for (int i = 0; i < npl; i++) if (pl_id[i] == id) { *b = pl[i]; return pl_len[i]; } return -1; }

static void *conn(void) { void *h = (void *)RD64(B(O_connHolder)); if (!h) return NULL; return (void *)RD64(h); }
int off_outer(void) { void *c = conn(); return c ? RD32(c) : -1; }
static int vbyte(void) { void *c = conn(); return c ? ((uint8_t *)c)[0xb5] : -1; }
static int trans(void) { void *c = conn(); if (!c) return -1; void *m = (void *)RD64((uint8_t *)c + 8); return m ? RD32((uint8_t *)m + 0x18) : -2; }

typedef void *(*create_t)(void *, int);
typedef int (*recv_t)(void *, void *, int);
static uint64_t fac_obj[8];
static void *create_msg(int id) {
    fac_obj[0] = (uint64_t)B(O_llmfVT); fac_obj[1] = 0;
    void *m = ((create_t)B(O_createMsg))(fac_obj, id);
    if (!m) { fac_obj[1] = 1; m = ((create_t)B(O_createMsg))(fac_obj, id); fac_obj[1] = 0; }
    return m;
}
static int msg_type(void *m) { void **vt = *(void ***)m; return ((int (*)(void *))vt[0x28 / 8])(m); }
static int inject(int id, int ver) {
    uint8_t *bytes; int n = payload(id, &bytes);
    if (n < 0) return -98;
    void *m = create_msg(id);
    if (!m) { LOG("[offline] create %d NULL\n", id); return -99; }
    *(int32_t *)((uint8_t *)m + 0x88) = ver;
    uint8_t *bs = (uint8_t *)m + 8, *buf = malloc(n ? n : 1); memcpy(buf, bytes, n);
    *(int32_t *)(bs + 0x14) = 0; *(int32_t *)(bs + 0x18) = n; *(int32_t *)(bs + 0x1c) = 0; *(void **)(bs + 0x38) = buf;
    void **vt = *(void ***)m; ((void (*)(void *))vt[0x18 / 8])(m);   /* decode */
    int r = ((recv_t)B(O_recv))((void *)RD64(B(O_mgrSlot)), m, 0);
    LOG("[offline] inject %d ver %d len %d -> %d [outer %d trans %d]\n", id, ver, n, r, off_outer(), trans());
    return r;
}

int g_login_done, g_home_sent, g_armed;
static void drive(void) {
    if (!g_armed) return;
    if (!g_login_done && off_outer() == 4 && vbyte() != 0) {
        g_login_done = 1; LOG("[offline] outer==4 -> LoginOk\n"); inject(20104, 1);
        if (off_outer() == 8 && !g_home_sent) { g_home_sent = 1; LOG("[offline] outer==8 -> OwnHomeData\n"); inject(24101, 0); }
        return;
    }
    if (g_login_done && !g_home_sent && off_outer() == 8) { g_home_sent = 1; LOG("[offline] outer==8 -> OwnHomeData\n"); inject(24101, 0); }
}

/* replacements forward x0..x7 unchanged (none of these functions takes float arguments) */
typedef uint64_t (*fn8)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
#define ARGS8 uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7
#define FWD8 a0, a1, a2, a3, a4, a5, a6, a7
static fn8 orig_connupd, orig_mgrsend, orig_msgsend, orig_loadwatch, orig_install;
static uint64_t my_connupd(ARGS8) {
    void *m = (void *)RD64((uint8_t *)a0 + 8);
    if (m) { ((uint8_t *)m)[8] = 0; *(int32_t *)((uint8_t *)m + 0x18) = 5; }
    uint64_t r = orig_connupd(FWD8);
    drive();
    return r;
}
static uint64_t my_mgrsend(ARGS8) {
    int t = msg_type((void *)a1);
    if (t != 10108 && t != 20108) LOG("[offline] OUT mgrSend type=%d [outer %d trans %d val %d]\n", t, off_outer(), trans(), vbyte());
    if (t == 10100 || t == 10101) g_armed = 1;
    return orig_mgrsend(FWD8);
}
static uint64_t my_msgsend(ARGS8) {
    int t = msg_type((void *)a1);
    if (t != 10108 && t != 20108) LOG("[offline] OUT msgSend type=%d [outer %d]\n", t, off_outer());
    if (t == 10100 || t == 10101) g_armed = 1;
    return orig_msgsend(FWD8);
}
int g_load_pct = -1; float g_load_stall;
static uint64_t my_loadwatch(ARGS8) {
    int p = RD32((uint8_t *)a0 + 0x40);
    if (p != g_load_pct) LOG("[offline] loading %d%%\n", p);
    g_load_pct = p; g_load_stall = *(float *)((uint8_t *)a0 + 0x54);
    return orig_loadwatch(FWD8);
}
static uint64_t my_install(ARGS8) {
    LOG("[offline] INSTALL-HOME home=%#lx avatar=%#lx arg=%d marker=%d\n", a1, a2, (int)a3, (int)a4);
    uint64_t r = orig_install(FWD8);
    LOG("[offline] install-home returned [outer %d]\n", off_outer());
    return r;
}

static fn8 orig_resinit;
static uint64_t my_resinit(ARGS8) {   /* 0xd3f5bc: object init that looks a resource up by name (crash site) */
    uint8_t *o = (uint8_t *)a0; int len = *(int32_t *)(o + 0x2c);
    const char *nm = len > 7 ? *(const char **)(o + 0x30) : (const char *)(o + 0x30);
    LOG("[res] 0xd3f5bc this=%p name len %d \"%.*s\"\n", o, len, len > 0 && len < 4000 ? len : 0, nm);
    return orig_resinit(FWD8);
}
static fn8 orig_pipe, orig_ece10c;
static int g_npipe;
static uint64_t my_pipe(ARGS8) {   /* 0xdab8cc(out, material, filler): pipeline descriptor build */
    uint8_t *mat = (uint8_t *)a0; void *o20 = *(void **)(mat + 0x20), *o50 = *(void **)(mat + 0x50);
    if (g_npipe++ < 40) LOG("[pipe] #%d material=%p (vt %s) filler vt %s, mat+0x20=%p (vt %s), mat+0x50=%p (vt %s)\n", g_npipe, mat,
        addr_desc(*(void **)mat), addr_desc(**(void ***)a1), o20, o20 ? addr_desc(*(void **)o20) : "-", o50, o50 ? addr_desc(*(void **)o50) : "-");
    return orig_pipe(FWD8);
}
static uint64_t my_ece10c(ARGS8) {
    uint8_t *d = (uint8_t *)a1;
    if (g_npipe < 41) LOG("[pipe] create: desc+0x18=%p desc+0x28=%p\n", *(void **)(d + 0x18), *(void **)(d + 0x28));
    return orig_ece10c(FWD8);
}
/* v68-native-battle.js in C: the yellow PLAY button calls HomePage::startGame 0x962880 (return address 0x96b8f0);
 * mode 3 makes it build the LOCAL simulation with native bots instead of asking the server for a match */
static fn8 orig_build, orig_tick;
int g_battles, g_ticks; void *g_sim;
static void my_start(uint64_t *ctx) {   /* onEnter: startGame has stack arguments */
    void *ret = (void *)ctx[9];
    int fromPlay = ret == (void *)B(0x96b8f0);
    LOG("[battle] startGame from %s mode=%d%s\n", addr_desc(ret), (int)ctx[3], fromPlay ? " (PLAY) -> mode 3 (native offline battle)" : "");
    if (fromPlay) { ctx[3] = 3; g_battles++; }
}
/* roster pinning (agent: attach(0xc27288)): during the local build every player becomes the brawler g_roster
 * ("Sniper" = Piper, the project's scope) with its native config and default skin (stack argument 9) */
const char *g_roster = "Sniper"; int g_building, g_pinned;
static void *char_data(const char *name) {
    uint8_t *s = ((void *(*)(unsigned long))B(0x11a4d90))(16); char *c = strdup(name);
    ((void (*)(void *, const char *))B(0xe96e10))(s, c);                         /* LogicString ctor */
    void *d = ((void *(*)(void *, void *))B(0xad2b60))(s, NULL);                  /* characters table lookup */
    ((void (*)(void *))B(0xe96d00))(s); free(c);
    return d;
}
/* diversity round (2026-10-09): per player index (sim+0x60 = the index this constructor call creates: 0xc27288 appends the
 * player at players[sim+0x60]) the RESET's character / skin (env.c div_reset -> g_div_char / g_div_skin, global data ids);
 * 0 = the default above. The character must be a playable brawler (characters.csv Type Hero: data+0x23c equal to the
 * roster brawler's), the skin one of that character's (the game's own check 0xc70d2c: the skin's SkinConf lists it). */
extern int32_t g_div_char[16], g_div_skin[16], g_div_gadget[16], g_div_star[16]; extern void *div_data(int id); extern void div_reject(int i);
static void my_player_ctor(uint64_t *ctx) {
    if (!g_building || !g_roster || !*g_roster) return;
    uint8_t *sim = (uint8_t *)ctx[0]; int variation = *(int32_t *)(sim + 0x104), slot = *(int32_t *)(sim + 0x60);
    if (variation != 5 && variation != 6) return;
    void *d = char_data(g_roster); if (!d) { LOG("[battle] roster: no character %s\n", g_roster); return; }
    int ci = slot >= 0 && slot < 16 ? g_div_char[slot] : 0, si = slot >= 0 && slot < 16 ? g_div_skin[slot] : 0;
    if (ci) {
        void *c = div_data(ci);
        if (c && ci / 1000000 == 16 && *(int32_t *)((uint8_t *)c + 0x23c) == *(int32_t *)((uint8_t *)d + 0x23c)) d = c;
        else { div_reject(slot); LOG("[battle] roster: slot %d character %d rejected\n", slot, ci); }
    }
    void *skin = ((void *(*)(void *))B(0xab3414))(d), *cfg = ((void *(*)(void *, void *))B(0xc27d64))(sim, d);
    if (si) {
        void *k = div_data(si);
        if (k && ((void *(*)(void *, void *, int))B(0xc70d2c))(k, d, 0) == k) skin = k;
        else { div_reject(slot); LOG("[battle] roster: slot %d skin %d rejected for character %d\n", slot, si, *(int32_t *)((uint8_t *)d + 0x20)); }
    }
    if (!skin || !cfg) { LOG("[battle] roster: config/skin missing\n"); return; }
    /* diversity round: requested gadget / star power cards into the player's config (0xc33e84 layout: +0x08 star power
     * (card MetaType 4), +0x10 gadget (MetaType 5)); a card must target this character (card+0xd8) with that type (+0x108) */
    int gi = slot >= 0 && slot < 16 ? g_div_gadget[slot] : 0, pi = slot >= 0 && slot < 16 ? g_div_star[slot] : 0;
    for (int k = 0; k < 2; k++) {
        int id = k ? pi : gi; if (!id) continue;
        uint8_t *card = div_data(id);
        if (card && id / 1000000 == 23 && *(void **)(card + 0xd8) == d && *(int32_t *)(card + 0x108) == (k ? 4 : 5))
            *(void **)((uint8_t *)cfg + (k ? 0x08 : 0x10)) = card;
        else { div_reject(slot); LOG("[battle] roster: slot %d %s card %d rejected\n", slot, k ? "star power" : "gadget", id); }
    }
    ctx[2] = (uint64_t)d; ctx[3] = (uint64_t)cfg; *(void **)((uint8_t *)ctx + 160 + 8) = skin;
    if (g_pinned++ < 10 || ((ci || si) && g_pinned < 400)) LOG("[battle] roster: slot %d -> %s (data id %d, skin %d)\n", slot, ci ? "div" : g_roster,
        *(int32_t *)((uint8_t *)d + 0x20), *(int32_t *)((uint8_t *)skin + 0x20));
}
extern int g_env_on; extern void env_before_build(void), env_after_build(void *), env_tick_enter(void *), env_tick_leave(void *);
/* nbenv sim-only resets: run the native builder on a stand-in battle object (only the simulation is built) */
void *offline_build_raw(void *battle, void *list, int a2, int a3) {
    g_building = 1;
    orig_build((uint64_t)battle, (uint64_t)list, a2, a3, 0, 0, 0, 0);
    g_building = 0;
    return *(void **)((uint8_t *)battle + 0x30);
}
uint64_t g_build_args[4];
static uint64_t my_build(ARGS8) {   /* 0x9bb710: builds the local simulation (battle+0x30) */
    g_build_args[0] = a1; g_build_args[1] = a2; g_build_args[2] = a3;
    { extern void div_window_apply(void); div_window_apply(); }   /* window play: the battle setup (location set by now) */
    if (a1 && *(void **)a1) {   /* copy the int list (c26384 ids) of the PLAY build */
        extern int g_build_list[64], g_build_nlist; int n = *(int32_t *)((uint8_t *)a1 + 0xc);
        g_build_nlist = n < 0 ? 0 : n > 64 ? 64 : n; memcpy(g_build_list, *(void **)a1, g_build_nlist * 4);
    }
    g_building = 1;
    if (g_env_on) env_before_build();
    uint64_t r = orig_build(FWD8);
    g_building = 0;
    if (g_env_on) env_after_build((void *)a0);
    g_sim = *(void **)((uint8_t *)a0 + 0x30); g_ticks = 0;
    LOG("[battle] local simulation built: battle=%#lx sim=%p players=%d\n", a0, g_sim, g_sim ? *(int32_t *)((uint8_t *)g_sim + 0x60) : -1);
    return r;
}
static double tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
double g_tick_time; long g_tick_n;
static uint64_t my_tick(ARGS8) {    /* 0xc29098: one simulation tick (timed: the cost of the battle logic under qemu-user) */
    double t0 = tnow();
    if (g_env_on) env_tick_enter((void *)a0);
    else { extern void window_storm_hits(void *); window_storm_hits((void *)a0); }   /* window play: storm hits (--storm-rules) */
    uint64_t r = orig_tick(FWD8);
    if (g_env_on) env_tick_leave((void *)a0);
    else { extern void window_tick(void *); window_tick((void *)a0); }   /* window play: cube pickups (--cube-rules) */
    if ((void *)a0 == g_sim) { g_tick_time += tnow() - t0; g_tick_n++;
        if (g_tick_n % (g_env_on ? 20000 : 200) == 0) LOG("[perf] battle tick: %.3f ms average over %ld ticks (this process, under qemu-user)\n", g_tick_time * 1e3 / g_tick_n, g_tick_n); }
    if ((void *)a0 == g_sim && (++g_ticks <= 3 || g_ticks % (g_env_on ? 20000 : 200) == 0)) {
        LOG("[battle] tick %d (sim tick counter %d)\n", g_ticks, *(int32_t *)((uint8_t *)g_sim + 0x10));
        extern void battle_fix_status(void); if (g_ticks % 200 == 0) battle_fix_status();
    }
    return r;
}
/* --lockstep K: decouple the battle from the wall clock. Every frame owes a tick (0xc2ccb8 "ticks due" forced >= 1)
 * and BattleMode::update's single tick call 0xc2ccd0(sim, 1, NULL) at 0x9c31b8 runs K ticks instead: one frame = one
 * env step of K ticks, as fast as the process can produce frames (the agent's lockstep does the same on the phone). */
int g_lockstep;
static int (*o_due)(void *, float);
static int my_due(void *sim, float dt) { int r = o_due(sim, dt); if (g_lockstep && sim == g_sim && r < 1) r = 1; return r; }
static void my_multi(uint64_t *ctx) {
    if (g_lockstep > 1 && (void *)ctx[0] == g_sim && ctx[9] == (uint64_t)B(0x9c31bc) && ctx[1] == 1) ctx[1] = g_lockstep;
}
void offline_install(const char *payload_dir) {
    if (g_lockstep) {
        extern void hook_enter(Module *, uint64_t, void (*)(uint64_t *));
        o_due = hook_replace(g_libg, 0xc2ccb8, my_due);
        hook_enter(g_libg, 0xc2ccd0, my_multi);
        LOG("[battle] lockstep: %d ticks per frame\n", g_lockstep);
    }
    extern void battle_fix_install(void); battle_fix_install();
    extern void env_install(void); env_install();
    extern void hook_enter(Module *, uint64_t, void (*)(uint64_t *));
    hook_enter(g_libg, 0x962880, my_start);
    hook_enter(g_libg, 0xc27288, my_player_ctor);
    orig_build = hook_replace(g_libg, 0x9bb710, my_build);
    orig_tick = hook_replace(g_libg, 0xc29098, my_tick);


    load_payload(payload_dir, 20104); load_payload(payload_dir, 24101);
    orig_connupd = hook_replace(g_libg, O_connUpd, my_connupd);
    orig_mgrsend = hook_replace(g_libg, O_mgrSend, my_mgrsend);
    orig_msgsend = hook_replace(g_libg, O_msgSend, my_msgsend);
    orig_loadwatch = hook_replace(g_libg, O_loadWatch, my_loadwatch);
    orig_install = hook_replace(g_libg, O_installHome, my_install);
    LOG("[offline] hooks installed; payloads %d\n", npl);
}
void offline_status(void) {
    LOG("[status] outer=%d trans=%d val=%d armed=%d login=%d home=%d loading=%d%%\n", off_outer(), trans(), vbyte(),
        g_armed, g_login_done, g_home_sent, g_load_pct);
}
