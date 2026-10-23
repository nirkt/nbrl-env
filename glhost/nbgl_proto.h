/* Shared-memory protocol between nbhost (the game, AArch64 under qemu-user) and nbgl (native x86 renderer + window).
 * The guest serialises every GLES 3.0 call into `ring`; nbgl replays it on a real GLES 3.2 context (Mesa, this PC's GPU).
 * Calls that return something are "sync": the guest waits for rep_seq and reads `reply`. Mouse input flows back in `ev`. */
#pragma once
#include <stdint.h>

#define NBGL_MAGIC 0x4e42474c31ULL
#define NBGL_RING (128u << 20)
#define NBGL_REPLY (24u << 20)
#define NBGL_NEV 1024
#define NBGL_SYNC 0x80000000u
#define NBGL_NULL 0xffffffffffffffffULL
enum { NBGL_OP_WRAP = 0, NBGL_OP_SWAP = 1, NBGL_OP_MAPREAD = 2, NBGL_OP_CAPTURE = 3, NBGL_OP_FIRST_GL = 16 };
/* NBGL_OP_CAPTURE (sync): args dw, dh, out-offset. The current default-framebuffer frame (before the swap) is reduced on
 * the GPU by 2:1 GL_LINEAR blits (2400x1080 -> 1200x540 -> 600x270 -> 300x135, the phone agent's chain) then to dw x dh,
 * and read back as RGBA8, bottom-left origin, PACK_ALIGNMENT 1. Game GL state is saved and restored. */

typedef struct { int32_t action, x, y, id; } NbglEvent;

typedef struct NbglShm {
    uint64_t magic;
    uint32_t width, height;                 /* window = game surface size */
    volatile uint64_t w, r;                 /* ring bytes written (guest) / consumed (host) */
    volatile uint64_t req_seq, rep_seq;     /* sync calls issued / answered */
    volatile uint64_t frames;               /* swaps executed by the host */
    volatile uint32_t host_alive, guest_alive;
    volatile uint64_t ev_w, ev_r;
    NbglEvent ev[NBGL_NEV];
    /* futex doorbells (carved from the pad; all other offsets unchanged): a side that found nothing to do sets its
     * *_sleep flag and waits on the bell; the other side bumps the bell and wakes it only when the flag is set */
    volatile uint32_t w_bell, host_sleep;   /* host (nbgl) waits for commands */
    volatile uint32_t rep_bell, guest_sleep;/* guest (game) waits for a sync reply */
    volatile int32_t key_dx, key_dy;        /* WASD direction (-1, 0, 1), written by nbgl; the game host moves the local brawler */
    uint8_t pad[4096 - 80 - 16 - 8];        /* header (80 B) + events (16 KiB) + bells + keys + pad = 5 pages */
    uint64_t ret;                           /* reply: return value */
    uint8_t reply[NBGL_REPLY];              /* reply: out-parameters */
    uint8_t ring[NBGL_RING];
} NbglShm;
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <time.h>
/* shared-memory futex (the mapping is MAP_SHARED in both processes: no FUTEX_PRIVATE_FLAG) */
static inline void nbgl_futex_wait(volatile uint32_t *a, uint32_t v, long ns) {
    struct timespec t = {ns / 1000000000L, ns % 1000000000L}; syscall(SYS_futex, (uint32_t *)a, FUTEX_WAIT, v, &t, NULL, 0); }
static inline void nbgl_futex_wake(volatile uint32_t *a) { syscall(SYS_futex, (uint32_t *)a, FUTEX_WAKE, 1, NULL, NULL, 0); }
/* after publishing (w or rep_seq): ring the other side if it sleeps */
static inline void nbgl_ring(volatile uint32_t *bell, volatile uint32_t *sleep_flag) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(sleep_flag, __ATOMIC_RELAXED)) { __atomic_add_fetch(bell, 1, __ATOMIC_SEQ_CST); nbgl_futex_wake(bell); }
}

/* command layout in the ring: u32 op|SYNC, u32 len (total, multiple of 8), u64 args[n], blobs (8-aligned).
 * pointer args: IN  -> offset of the blob from the command start (NBGL_NULL for NULL)
 *               OUT -> offset into reply[] */

/* number of values glGet{Integer,Float,Boolean,Integer64}v writes for a pname (core GLES 3.0) */
static inline int nbgl_pname_count(uint32_t p) {
    switch (p) {
    case 0x0BA2: case 0x0C10: case 0x0C22: case 0x0C23: case 0x8005: return 4;   /* VIEWPORT SCISSOR_BOX COLOR_CLEAR COLOR_WRITEMASK BLEND_COLOR */
    case 0x0D3A: case 0x846D: case 0x846E: case 0x0B70: return 2;               /* MAX_VIEWPORT_DIMS ALIASED_POINT/LINE_WIDTH_RANGE DEPTH_RANGE */
    }
    return 1;
}
