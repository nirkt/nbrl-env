/* The offline native battle needs the same repairs the phone's RL agent (v68-rl-candidate.js) applies, otherwise the
 * client desynchronises at tick 2-3 and faults at 0xc40358 (documented in the device sweeps):
 *  - two snapshot-serializer helpers are empty RET stubs in this libg (0xb280a4 ability, 0xb6a998 element): the agent
 *    writes their fields itself at the call sites 0xb64954 / 0xb37448 -> here the `bl`s call C writers instead;
 *  - characters built for the offline battle lack their loadout "elements": ensure() populates them (0xb34e3c, 0xb360ac);
 *  - Showdown null guards (0xb7f310, 0xb4ffec, 0xb48904, 0xb48ad8).
 * Everything is gated on the offline-battle flag (libg+0x12eefb4 == 3), like the agent's mode(). */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>

extern Module *g_libg;
void *hook_replace(Module *m, uint64_t off, void *repl);
void hook_bl(Module *m, uint64_t off, void *fn);
void hook_null_reg(Module *m, uint64_t off, int reg, void *scratch);
#define B(off) ((uint8_t *)g_libg->base + (off))
#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
static int mode(void) { return I32(g_libg->base, 0x12eefb4) == 3; }

static const uint32_t W[] = {0xc74b74,0xc74bfc,0xc74c8c,0xc74d1c,0xc74dac,0xc74e3c,0xc74ecc,0xc74f5c,0xc74fec,0xc7507c,0xc7510c,
    0xc7519c,0xc7522c,0xc752bc,0xc7534c,0xc753dc,0xc7546c,0xc754fc,0xc7558c,0xc7561c,0xc756ac,0xc7573c,0xc757cc,0xc7585c,0xc758ec,
    0xc7597c,0xc75a0c};
static void w(int i, void *wr, int v, int tag) { ((void (*)(void *, int, int))B(W[i]))(wr, v, tag); }
static long g_abil, g_elem, g_populated, g_guard[4], g_range;

/* replaces `bl 0xb280a4` at 0xb64954: x0 = ability, x1 = BitStream writer, x2 bit0 = own character */
static void ability_writer(uint8_t *p, void *wr, uint64_t x2) {
    if (!mode()) return;
    int own = x2 & 1;
    int c8 = I32(p, 0xc8), c4 = I32(p, 0xc4), state = I32(p, 0x20), kind = I32(p, 0x24), f38 = I32(p, 0x38), f88 = I32(p, 0x88);
    int flag = p[0x3c] & 1, f48 = I32(p, 0x48), f4c = I32(p, 0x4c);
    if (own && (c8 < 0 || c8 > 65535 || c4 < 0 || c4 > 65535 || state < 0 || state > 255 || f38 < 0 || f38 > 31 || (kind == 58 && (f88 < 0 || f88 > 65535))))
        { if (g_range++ < 5) LOG("[battle-fix] ability value outside codec range\n"); }
    int ext = own ? state == 1 : flag;
    if (own) { w(0, wr, c8 ? 1 : 0, 0x38); if (c8) w(24, wr, c8, 0x39); w(24, wr, c4, 0x3a); w(26, wr, state, 0x3b); w(5, wr, f38, 0x40);
               if (kind == 58) w(24, wr, f88, 0x3c); }
    else w(0, wr, flag, 0x3d);
    if (ext) { w(14, wr, f48, 0x3e); w(9, wr, f4c, 0x3f); }
    g_abil++;
}
/* replaces `bl 0xb6a998` at 0xb37448: x0 = element, x1 = writer */
static void element_writer(uint8_t *e, void *wr) {
    if (!mode()) return;
    int k = I32(e, 8), flag = e[0x14] & 1, v = I32(e, 0x10);
    w(0, wr, flag, 0x1b2); if (k == 4) w(10, wr, v, 0x1b3);
    g_elem++;
}
/* populate the character's simulation element vector (+0xd68 data, +0xd70 cap, +0xd74 count) from its loadout */
static void ensure(uint8_t *ch) {
    uint8_t *data = PTR(ch, 0x10); if (!data) return;
    int cls = I32(ch, 0x428), kind = cls < 0 ? I32(data, 0x23c) : cls;
    if (kind != 0 && kind != 26) return;
    if (((int (*)(void *))B(0xab423c))(data) & 1) return;
    uint8_t *p = ((uint8_t *(*)(void *))B(0xb6a718))(ch); if (!p || p[0x238]) return;
    uint8_t *cfg = PTR(p, 0x348);
    if (!cfg) {
        int i = I32(p, 0x40), n = I32(p, 0x3c);
        if (i < 0 || n < 0 || n > 128 || i >= n || !PTR(p, 0x30)) return;
        cfg = ((uint8_t *(*)(void *))B(0xc41460))(p);
    }
    if (!cfg) return;
    void *want[2]; int nw = 0;
    if (PTR(cfg, 0x18)) want[nw++] = PTR(cfg, 0x18);
    if (PTR(cfg, 0x20)) want[nw++] = PTR(cfg, 0x20);
    int n = I32(ch, 0xd74), cap = I32(ch, 0xd70);
    if (n || !nw || n < 0 || cap < 0 || cap > 128) return;
    for (int i = 0; i < nw; i++) { int k = ((int (*)(void *))B(0xb6a994))(want[i]); if (k != 0 && k != 1) return; }
    void *fresh[2];
    for (int i = 0; i < nw; i++) {
        fresh[i] = ((void *(*)(unsigned long))B(0x11a4d90))(0x20); memset(fresh[i], 0, 0x20);
        ((void (*)(void *, void *))B(0xb6a8ec))(fresh[i], want[i]);
    }
    if (cap < nw) ((void (*)(void *, int))B(0xb32288))(ch + 0xd68, nw > 5 ? nw : 5);
    void **arr = PTR(ch, 0xd68); if (!arr) return;
    for (int i = 0; i < nw; i++) arr[i] = fresh[i];
    __atomic_store_n((int32_t *)(ch + 0xd74), nw, __ATOMIC_RELEASE);
    g_populated++;
}
typedef uint64_t (*fn8)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
#define ARGS8 uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7
#define FWD8 a0, a1, a2, a3, a4, a5, a6, a7
static fn8 o_init, o_ser, o_pred, o_die;
static uint64_t my_init(ARGS8) { if (mode()) ensure((uint8_t *)a0); return o_init(FWD8); }
static uint64_t my_ser(ARGS8) { if (mode()) ensure((uint8_t *)a0); return o_ser(FWD8); }
static uint8_t g_scratch[0x200] __attribute__((aligned(16)));
static uint64_t my_pred(ARGS8) { if (!a0 && mode()) { a0 = (uint64_t)g_scratch; g_guard[0]++; } return o_pred(FWD8); }
static uint64_t my_die(ARGS8) { uint64_t r = o_die(FWD8); if (!r && a0 && mode()) { r = a0; g_guard[1]++; } return r; }

void battle_fix_install(void) {
    hook_bl(g_libg, 0xb64954, ability_writer);
    hook_bl(g_libg, 0xb37448, element_writer);
    o_init = hook_replace(g_libg, 0xb34e3c, my_init);
    o_ser = hook_replace(g_libg, 0xb360ac, my_ser);
    o_pred = hook_replace(g_libg, 0xb7f310, my_pred);
    o_die = hook_replace(g_libg, 0xb4ffec, my_die);
    hook_null_reg(g_libg, 0xb48904, 28, g_scratch);
    hook_null_reg(g_libg, 0xb48ad8, 28, g_scratch);
    hook_null_reg(g_libg, 0xb7a220, 19, g_scratch);   /* seed-dependent null skill slot (`str w8,[x19,#0x10]`), see device STATUS */
    LOG("[battle-fix] installed: ability/element writers, element repair, 4 null guards\n");
}
void battle_fix_status(void) {
    LOG("[battle-fix] ability fields %ld, elements %ld, characters populated %ld, guards %ld/%ld\n", g_abil, g_elem, g_populated, g_guard[0], g_guard[1]);
}
