/* Inline hooks on libg code from inside the same process (what Frida's Interceptor.replace does on the phone).
 * target:  b <stub>                         (4 bytes; the stub page sits right after libg's mapping, < 128 MB away)
 * stub:    mrs x16, tpidr_el0 ; str x8, [x16, #72] ; ldr x16, #8 ; br x16 ; .quad replacement
 * tramp:   mrs x16, tpidr_el0 ; ldr x8, [x16, #72] ; <displaced first instruction, relocated if PC-relative> ;
 *          ldr x16, #8 ; br x16 ; .quad target+4
 * x8 (AArch64 indirect-result register, used by functions returning structs) is parked in bionic TLS slot 7 of our
 * reserved area (tp+72; libg itself only reads tp+0x28) because the C replacement may clobber it.
 * hook_replace() returns the trampoline, i.e. a pointer that calls the original function. */
#include "nbhost.h"
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

static uint8_t *near_page; static size_t near_used;
static uint32_t *near_alloc(size_t bytes, Module *m) {
    if (!near_page) {   /* within branch range (+-128 MB) of every site in the module: try just after it, then nearby */
        for (int k = 0; k < 64 && (!near_page || near_page == MAP_FAILED); k++) {
            uint8_t *want = k % 2 == 0 ? m->base + m->span + (size_t)(k / 2) * (1 << 20) : m->base - (size_t)(k / 2 + 1) * (1 << 20);
            near_page = mmap(want, 1 << 20, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (near_page != MAP_FAILED && near_page != want) { munmap(near_page, 1 << 20); near_page = MAP_FAILED; }
        }
        if (near_page == MAP_FAILED) { LOG("[hook] cannot map the near stub page\n"); exit(1); }
    }
    uint32_t *p = (uint32_t *)(near_page + near_used); near_used += (bytes + 15) & ~15UL; return p;
}
static void emit_jump(uint32_t *c, uint64_t to) {
    c[0] = 0x58000050;   /* ldr x16, #8 */
    c[1] = 0xd61f0200;   /* br  x16 */
    memcpy(c + 2, &to, 8);
}


/* Relocate a conditional PC-relative branch (cbz/cbnz/tbz/tbnz/b.cond) that sits at `from` into tramp[n..]:
 *   <same condition> -> +8 (to the taken-path far jump) ; b -> after it ; ldr x16,#8 ; br x16 ; .quad taken target
 * The not-taken path falls through to whatever the caller emits next. Returns the number of words written, 0 if
 * `insn` is not such a branch. */
static int reloc_condbr(uint32_t *tramp, uint32_t insn, uint64_t from) {
    int64_t off; uint32_t keep;
    if ((insn & 0x7e000000) == 0x34000000) {            /* cbz/cbnz: imm19 at [23:5] */
        off = ((int64_t)((insn >> 5) & 0x7ffff) << 45) >> 43; keep = insn & ~(0x7ffffu << 5); keep |= 2u << 5;
    } else if ((insn & 0x7e000000) == 0x36000000) {     /* tbz/tbnz: imm14 at [18:5] */
        off = ((int64_t)((insn >> 5) & 0x3fff) << 50) >> 48; keep = insn & ~(0x3fffu << 5); keep |= 2u << 5;
    } else if ((insn & 0xff000010) == 0x54000000) {     /* b.cond: imm19 at [23:5] */
        off = ((int64_t)((insn >> 5) & 0x7ffff) << 45) >> 43; keep = insn & ~(0x7ffffu << 5); keep |= 2u << 5;
    } else return 0;
    uint64_t to = from + off;
    tramp[0] = keep;                 /* condition true -> tramp[2] */
    tramp[1] = 0x14000005;           /* b +5 words -> tramp[6] (fall-through path) */
    tramp[2] = 0x58000050; tramp[3] = 0xd61f0200; memcpy(tramp + 4, &to, 8);
    return 6;
}

void *hook_replace(Module *m, uint64_t off, void *repl) {
    uint32_t *target = (uint32_t *)(m->base + off);
    uint32_t insn = target[0];
    uint32_t *tramp = near_alloc(64, m), *stub = near_alloc(32, m);
    int n = 0;
    tramp[n++] = 0xd53bd050;   /* mrs x16, tpidr_el0 */
    tramp[n++] = 0xf9402608;   /* ldr x8, [x16, #72] */
    if ((insn & 0x9f000000) == 0x90000000) {               /* adrp xd, page: materialise the absolute page */
        int64_t imm = ((int64_t)(((insn >> 5) & 0x7ffff) << 2 | ((insn >> 29) & 3)) << 43) >> 31;
        uint64_t page = ((uint64_t)target & ~0xfffUL) + imm;
        tramp[n++] = 0x58000040 | (insn & 31);             /* ldr xd, #8 */
        tramp[n++] = 0x14000003;                           /* b +12 */
        memcpy(tramp + n, &page, 8); n += 2;
    } else if (reloc_condbr(tramp + n, insn, (uint64_t)target)) {
        n += 6;
    } else if ((insn & 0x7c000000) == 0x14000000 || (insn & 0x3b000000) == 0x18000000 || (insn & 0x9f000000) == 0x10000000) {
        LOG("[hook] %s+%#lx: first instruction %08x is PC-relative; not supported\n", m->name, (unsigned long)off, insn);
        exit(1);
    } else tramp[n++] = insn;
    emit_jump(tramp + n, (uint64_t)(target + 1));
    stub[0] = 0xd53bd050;      /* mrs x16, tpidr_el0 */
    stub[1] = 0xf9002608;      /* str x8, [x16, #72] */
    emit_jump(stub + 2, (uint64_t)repl);
    int64_t d = ((int64_t)stub - (int64_t)target) >> 2;
    if (d >= (1 << 25) || d < -(1 << 25)) { LOG("[hook] stub out of branch range\n"); exit(1); }
    __builtin___clear_cache((char *)tramp, (char *)(tramp + 16));
    __builtin___clear_cache((char *)stub, (char *)(stub + 8));
    __atomic_store_n(target, 0x14000000 | (uint32_t)(d & 0x3ffffff), __ATOMIC_RELEASE);
    __builtin___clear_cache((char *)target, (char *)(target + 1));
    VLOG("[hook] %s+%#lx -> %p (orig insn %08x)\n", m->name, (unsigned long)off, repl, insn);
    return tramp;
}
void patch_u32(Module *m, uint64_t off, uint32_t v) {
    uint32_t *t = (uint32_t *)(m->base + off); *t = v; __builtin___clear_cache((char *)t, (char *)(t + 1));
}

/* Retarget the `bl` at `off` to a C function (used where the shipped libg calls an empty RET stub). */
void hook_bl(Module *m, uint64_t off, void *fn) {
    uint32_t *site = (uint32_t *)(m->base + off);
    if ((*site & 0xfc000000) != 0x94000000) { LOG("[hook] %s+%#lx is not a bl (%08x)\n", m->name, (unsigned long)off, *site); exit(1); }
    uint32_t *stub = near_alloc(16, m); emit_jump(stub, (uint64_t)fn);
    __builtin___clear_cache((char *)stub, (char *)(stub + 4));
    int64_t d = ((int64_t)stub - (int64_t)site) >> 2;
    __atomic_store_n(site, 0x94000000 | (uint32_t)(d & 0x3ffffff), __ATOMIC_RELEASE);
    __builtin___clear_cache((char *)site, (char *)(site + 1));
}
/* Null guard for a single PC-independent instruction that dereferences register `reg`: if reg == 0, point it at a
 * zeroed scratch block first (what the agent does with this.context.xN = scratch). */
void hook_null_reg(Module *m, uint64_t off, int reg, void *scratch) {
    uint32_t *site = (uint32_t *)(m->base + off), *s = near_alloc(48, m);
    uint64_t back = (uint64_t)(site + 1), scr = (uint64_t)scratch;
    s[0] = 0xb5000060 | reg;          /* cbnz xreg, +12 */
    s[1] = 0x580000e0 | reg;          /* ldr  xreg, #28  (-> s[8]) */
    s[2] = 0xd503201f;                /* nop */
    s[3] = *site;                     /* the guarded instruction */
    s[4] = 0x58000050; s[5] = 0xd61f0200; memcpy(s + 6, &back, 8); memcpy(s + 8, &scr, 8);
    __builtin___clear_cache((char *)s, (char *)(s + 12));
    int64_t d = ((int64_t)s - (int64_t)site) >> 2;
    __atomic_store_n(site, 0x14000000 | (uint32_t)(d & 0x3ffffff), __ATOMIC_RELEASE);
    __builtin___clear_cache((char *)site, (char *)(site + 1));
}

/* onEnter-style hook (Frida's Interceptor.attach onEnter): the stub saves x0-x8, lr, d0-d7 on the stack, calls
 * fn(ctx) with ctx[0..8] = x0..x8, ctx[9] = lr, ((double *)ctx)[10..17] = d0..d7 and the caller's stack arguments at
 * (uint8_t *)ctx + 160; fn may edit any of them; the stub restores the registers and enters the original function with
 * the caller's sp unchanged, so stack arguments (functions with > 8 arguments) stay intact. */
static const uint32_t enter_stub[26] = {0xd10283ff, 0xa90007e0, 0xa9010fe2, 0xa90217e4, 0xa9031fe6, 0xa9047be8, 0x6d0507e0,
    0x6d060fe2, 0x6d0717e4, 0x6d081fe6, 0x910003e0, 0x58000230, 0xd63f0200, 0xa94007e0, 0xa9410fe2, 0xa94217e4, 0xa9431fe6,
    0xa9447be8, 0x6d4507e0, 0x6d460fe2, 0x6d4717e4, 0x6d481fe6, 0x910283ff, 0x58000070, 0xd61f0200, 0xd503201f};
void hook_enter(Module *m, uint64_t off, void (*fn)(uint64_t *ctx)) {
    uint32_t *target = (uint32_t *)(m->base + off), insn = target[0];
    if ((insn & 0x9f000000) == 0x90000000 || (insn & 0x7c000000) == 0x14000000 || (insn & 0x3b000000) == 0x18000000 ||
        (insn & 0x9f000000) == 0x10000000) {
        LOG("[hook] enter %s+%#lx: first instruction %08x is PC-relative\n", m->name, (unsigned long)off, insn); exit(1);
    }
    uint32_t *tramp = near_alloc(64, m), *stub = near_alloc(128, m);
    int tn = reloc_condbr(tramp, insn, (uint64_t)target);
    if (!tn) tramp[tn++] = insn;
    emit_jump(tramp + tn, (uint64_t)(target + 1));
    memcpy(stub, enter_stub, sizeof enter_stub);
    uint64_t t = (uint64_t)tramp, f = (uint64_t)fn;
    memcpy(stub + 26, &t, 8); memcpy(stub + 28, &f, 8);
    __builtin___clear_cache((char *)tramp, (char *)(tramp + 16));
    __builtin___clear_cache((char *)stub, (char *)(stub + 32));
    int64_t d = ((int64_t)stub - (int64_t)target) >> 2;
    __atomic_store_n(target, 0x14000000 | (uint32_t)(d & 0x3ffffff), __ATOMIC_RELEASE);
    __builtin___clear_cache((char *)target, (char *)(target + 1));
}

/* Flag-controlled branch at a PC-independent instruction `expect`: while *flag != 0 continue at `skip_to` (a path of the
 * game's own), else run the instruction and continue at off+4. Byte-checked; returns 0 on success. Uses x16 (scratch). */
int hook_flag_skip(Module *m, uint64_t off, uint32_t expect, volatile int *flag, uint64_t skip_to) {
    uint32_t *site = (uint32_t *)(m->base + off);
    if (*site != expect) { LOG("[hook] flag-skip %s+%#lx: unexpected bytes %08x (want %08x)\n", m->name, (unsigned long)off, *site, expect); return -1; }
    uint32_t *t = near_alloc(64, m); uint64_t f = (uint64_t)flag;
    t[0] = 0x58000000 | (12u << 5) | 16;             /* ldr  x16, t[12] (the flag's address) */
    t[1] = 0xb9400210;                               /* ldr  w16, [x16] */
    t[2] = 0x35000000 | (6u << 5) | 16;              /* cbnz w16, t[8] */
    t[3] = expect;                                   /* the original instruction */
    emit_jump(t + 4, (uint64_t)(site + 1));
    emit_jump(t + 8, (uint64_t)(m->base + skip_to));
    memcpy(t + 12, &f, 8);
    __builtin___clear_cache((char *)t, (char *)(t + 16));
    int64_t d = ((int64_t)t - (int64_t)site) >> 2;
    if (d >= (1 << 25) || d < -(1 << 25)) { LOG("[hook] flag-skip out of branch range\n"); return -1; }
    __atomic_store_n(site, 0x14000000 | (uint32_t)(d & 0x3ffffff), __ATOMIC_RELEASE);
    __builtin___clear_cache((char *)site, (char *)(site + 1));
    return 0;
}

/* Null guard after a pointer load: at `off` the instruction `expect` loads a pointer into x<reg>; if it is NULL, continue at
 * `skip_to` (the game's own "no target" exit), else at off+4. Byte-checked; returns 0 on success. */
int hook_null_skip(Module *m, uint64_t off, uint32_t expect, int reg, uint64_t skip_to) {
    uint32_t *site = (uint32_t *)(m->base + off);
    if (*site != expect) { LOG("[hook] null-skip %s+%#lx: unexpected bytes %08x (want %08x)\n", m->name, (unsigned long)off, *site, expect); return -1; }
    uint32_t *t = near_alloc(48, m);
    t[0] = expect;                                   /* the original load */
    t[1] = 0xb4000000 | (5u << 5) | (uint32_t)reg;   /* cbz x<reg>, +20 -> t[6] */
    emit_jump(t + 2, (uint64_t)(site + 1));          /* not null: continue after the load */
    emit_jump(t + 6, (uint64_t)(m->base + skip_to)); /* null: the game's skip path */
    __builtin___clear_cache((char *)t, (char *)(t + 12));
    int64_t d = ((int64_t)t - (int64_t)site) >> 2;
    if (d >= (1 << 25) || d < -(1 << 25)) { LOG("[hook] null-skip out of branch range\n"); return -1; }
    __atomic_store_n(site, 0x14000000 | (uint32_t)(d & 0x3ffffff), __ATOMIC_RELEASE);
    __builtin___clear_cache((char *)site, (char *)(site + 1));
    return 0;
}
