/* C port of state2.encode (inputs/host_code/state2.py, learn round S0): the egocentric fixed-size observation of any
 * player from a snap2 dump + the extras (cubes per character, active leaps) + the map + the previous dump (velocities).
 * Output: vec[436] (self 22, 9 enemies x 16, 16 projectiles x 12, 6 boxes x 5, 8 items x 6) and grid[6][25][25],
 * float32, written exactly in state2.flatten order. Arithmetic in double, cast to float like numpy's float32 arrays.
 * Verified against the Python encoder on recorded states: work/py/test_encode.py. */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define NIL ((int32_t)0x80000000)
#define P2 8
#define C2 23
#define J2 18
#define I2 12
#define TILE 300
#define SNIPER 16000015
#define BOX_ID 16000120
#define CUBE_ITEM 18000009
#define BOMB_BOX_ITEM 18000011
#define BOMB_ITEM 18000035
#define K_ENEMY 9
#define K_PROJ 16
#define K_BOX 6
#define K_ITEM 8
#define F_SELF 22
#define F_ENEMY 16
#define F_PROJ 12
#define F_BOX 5
#define F_ITEM 6
#define GRID 25
#define R_SCALE 3000.0

typedef struct {   /* map classes (state2.MapInfo) */
    int W, H; uint8_t *wall, *water, *bush;
} EncMap;
static const int WALL_ROWS[] = {1, 2, 3, 4, 6, 10, 12, 13, 14, 15, 16, 20, 21, 39, 40, 41, 42, 43, 44, 45, 46, 53, 54, 55, 56, 57, 58,
                                59, 60, 61, 62, 63, 65, 66, 69, 70, 71, 72, 73};
static const int WATER_ROWS[] = {9, 11, 18};
static const int BUSH_ROWS[] = {7, 8, 17, 68};
static int in_rows(int id, const int *rows, int n) { for (int i = 0; i < n; i++) if (id == 27000000 + rows[i]) return 1; return 0; }
void enc_map_build(EncMap *m, const int32_t *tiles, int W, int H) {
    free(m->wall); free(m->water); free(m->bush);
    m->W = W; m->H = H; m->wall = calloc(W * H, 1); m->water = calloc(W * H, 1); m->bush = calloc(W * H, 1);
    for (int i = 0; i < W * H; i++) {
        int id = tiles[i];
        m->wall[i] = in_rows(id, WALL_ROWS, sizeof WALL_ROWS / 4);
        m->water[i] = in_rows(id, WATER_ROWS, sizeof WATER_ROWS / 4);
        m->bush[i] = in_rows(id, BUSH_ROWS, sizeof BUSH_ROWS / 4);
    }
}
typedef struct { int id, x, y; } PrevPos;
typedef struct {
    const int32_t *pl, *ch, *pj, *it; int npl, nch, npj, nit, tick;
    const int32_t (*cubes)[2]; int ncubes;      /* [char id, n] */
    const int32_t (*jumps)[5]; int njumps;      /* [char id, t, T, x1, y1] */
    const PrevPos *prev; int nprev, prev_tick;  /* previous dump's character positions (velocities) */
} EncState;

static int floordiv(int a, int b) { int q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return q; }

static int is_box(const int32_t *r) { return r[18] == 0 && r[1] == BOX_ID; }
/* diversity round: a brawler = the current character of its owner player (players rows: [0] index, [2] characterId,
 * [3] controlledId); summons (Nita's bear, Jessie's / Penny's turrets) own a player index too but are not brawlers.
 * Piper-only battles have no such characters: unchanged. */
static int is_brawler(const EncState *s, const int32_t *r) {
    if (r[18] != 0 || r[1] == BOX_ID || r[2] < 0) return 0;
    const int32_t *p = r[2] < s->npl ? s->pl + r[2] * P2 : NULL;   /* player rows are in index order (snap2) */
    if (!p || p[0] != r[2]) {
        p = NULL; for (int i = 0; i < s->npl; i++) if (s->pl[i * P2] == r[2]) { p = s->pl + i * P2; break; }
        if (!p) return 1;
    }
    return r[0] == ((p[3] && p[3] != (int32_t)0x80000000) ? p[3] : p[2]);
}
static int cubes_of(const EncState *s, int id) { for (int i = 0; i < s->ncubes; i++) if (s->cubes[i][0] == id) return s->cubes[i][1]; return 0; }
static int jumping(const EncState *s, int id) { for (int i = 0; i < s->njumps; i++) if (s->jumps[i][0] == id) return 1; return 0; }
static double super_frac(const EncState *s, int pidx) {
    for (int i = 0; i < s->npl; i++) { const int32_t *p = s->pl + i * P2; if (p[0] == pidx) return p[5] <= 0 ? 0.0 : (double)p[4] / (double)p[5]; }
    return 0.0;
}
static void vel_of(const EncState *s, int id, double *vx, double *vy) {
    *vx = *vy = 0;
    if (!s->prev || s->prev_tick >= s->tick) return;
    double dt = s->tick - s->prev_tick;
    for (int i = 0; i < s->nprev; i++) if (s->prev[i].id == id) {
        for (int k = 0; k < s->nch; k++) { const int32_t *r = s->ch + k * C2; if (r[0] == id) { *vx = (r[3] - s->prev[i].x) / dt; *vy = (r[4] - s->prev[i].y) / dt; return; } }
    }
}
static int gas_step(int tick) { int t = tick - 440; return t > 0 ? t / 100 : 0; }
static int gas_next(int tick) { int t = tick - 440; return t < 0 ? 440 - tick : 100 - (t % 100); }
static int in_gas(int x, int y, int tick, int W, int H) {
    int st = gas_step(tick), tx = floordiv(x, TILE), ty = floordiv(y, TILE);
    int a = tx < ty ? tx : ty, b = W - 1 - tx < H - 1 - ty ? W - 1 - tx : H - 1 - ty;
    return (a < b ? a : b) < st;
}
static int map_at(const EncMap *m, const uint8_t *arr, int tx, int ty) {   /* m.bush[min(ty, H-1), min(tx, W-1)] (python negative index wraps) */
    if (ty > m->H - 1) ty = m->H - 1; if (tx > m->W - 1) tx = m->W - 1;
    if (ty < 0) ty += m->H; if (tx < 0) tx += m->W;
    if (ty < 0 || tx < 0) return 0;
    return arr[ty * m->W + tx];
}
typedef struct { double key; int seq; float v[16]; } Row;
static int cmprow(const void *a, const void *b) {
    const Row *x = a, *y = b;
    if (x->key < y->key) return -1; if (x->key > y->key) return 1;
    return x->seq - y->seq;   /* python's sort is stable */
}
/* -> 0 if the player has no living brawler (state2.encode returns None; the env then uses zeros) */
int enc_encode(const EncMap *m, const EncState *s, int pidx, float *vec, float *grid) {
    memset(vec, 0, 436 * sizeof(float)); memset(grid, 0, 6 * GRID * GRID * sizeof(float));
    const int32_t *me = NULL;
    for (int k = 0; k < s->nch; k++) { const int32_t *r = s->ch + k * C2; if (r[2] == pidx && is_brawler(s, r)) { me = r; break; } }
    if (!me || me[6] <= 0) return 0;
    int W = m->W, H = m->H, tick = s->tick;
    double mx = me[3], my = me[4]; int mid = me[0];
    double vx, vy; vel_of(s, mid, &vx, &vy);
    int tx = floordiv((int)mx, TILE), ty = floordiv((int)my, TILE);
    int nalive = 0; for (int k = 0; k < s->nch; k++) { const int32_t *r = s->ch + k * C2; if (is_brawler(s, r) && r[6] > 0) nalive++; }
    int ammo = me[13] > 0 ? me[13] : 0, cd = me[14] > 0 ? me[14] : 0, z = me[5] > 0 ? me[5] : 0;
    int cub = cubes_of(s, mid); if (cub > 10) cub = 10;
    double fs[F_SELF] = {
        (double)me[6] / (me[7] > 1 ? me[7] : 1), me[6] / 10000.0, ammo / 3000.0, (ammo % 1000) / 1000.0,
        cd / 600.0, super_frac(s, pidx), cub / 10.0, me[11] > 0 ? 1.0 : 0.0,
        (double)map_at(m, m->bush, tx, ty), (double)in_gas((int)mx, (int)my, tick, W, H), mx / (W * TILE), my / (H * TILE),
        vx / 40.0, vy / 40.0, z / 700.0, jumping(s, mid) ? 1.0 : 0.0,
        (tick < 3000 ? tick : 3000) / 3000.0, nalive / 10.0, gas_step(tick) / 30.0, gas_next(tick) / 440.0,
        (W * TILE / 2.0 - mx) / (W * TILE / 2.0), (H * TILE / 2.0 - my) / (H * TILE / 2.0)};
    for (int i = 0; i < F_SELF; i++) vec[i] = (float)fs[i];
    float *o = vec + F_SELF;
    /* enemies */
    static Row rows[4096]; int n = 0;
    for (int k = 0; k < s->nch && n < 4096; k++) {
        const int32_t *r = s->ch + k * C2;
        if (!is_brawler(s, r) || r[2] == pidx || r[6] <= 0) continue;
        double dx = (r[3] - mx) / R_SCALE, dy = (r[4] - my) / R_SCALE, ex, ey; vel_of(s, r[0], &ex, &ey);
        int etx = floordiv(r[3], TILE), ety = floordiv(r[4], TILE), ea = r[13] > 0 ? r[13] : 0, ec = cubes_of(s, r[0]); if (ec > 10) ec = 10;
        double v[16] = {1.0, dx, dy, ex / 40.0, ey / 40.0, (double)r[6] / (r[7] > 1 ? r[7] : 1), r[6] / 10000.0, ea / 3000.0,
                        super_frac(s, r[2]), ec / 10.0, (double)map_at(m, m->bush, etx, ety), r[19] == 1, r[11] > 0,
                        jumping(s, r[0]), r[16] == pidx, r[1] == SNIPER};
        rows[n].key = dx * dx + dy * dy; rows[n].seq = n; for (int j = 0; j < 16; j++) rows[n].v[j] = (float)v[j]; n++;
    }
    qsort(rows, n, sizeof *rows, cmprow);
    for (int i = 0; i < K_ENEMY && i < n; i++) memcpy(o + i * F_ENEMY, rows[i].v, F_ENEMY * sizeof(float));
    o += K_ENEMY * F_ENEMY;
    /* projectiles */
    n = 0;
    for (int k = 0; k < s->npj && n < 4096; k++) {
        const int32_t *r = s->pj + k * J2;
        double dx = (r[3] - mx) / R_SCALE, dy = (r[4] - my) / R_SCALE;
        double pvx = (double)(int32_t)((uint32_t)r[3] - (uint32_t)r[6]), pvy = (double)(int32_t)((uint32_t)r[4] - (uint32_t)r[7]);   /* numpy int32 */
        if (r[6] == 0 && r[7] == 0) pvx = pvy = 0.0;
        int own = r[2] == pidx;
        double rx = mx - r[3], ry = my - r[4], vv = pvx * pvx + pvy * pvy;
        double tca = vv > 1e-6 ? (rx * pvx + ry * pvy) / vv : 0.0; if (tca < 0) tca = 0.0;
        double cx = r[3] + pvx * tca - mx, cy = r[4] + pvy * tca - my;
        double v[12] = {1.0, dx, dy, pvx / 200.0, pvy / 200.0, (double)own, (double)(!own && r[2] >= 0), r[9] / 4000.0,
                        r[10] / 1000.0, (double)(r[12] != 0), (tca < 40 ? tca : 40) / 40.0, hypot(cx, cy) / R_SCALE};
        rows[n].key = dx * dx + dy * dy; rows[n].seq = n; for (int j = 0; j < 12; j++) rows[n].v[j] = (float)v[j]; n++;
    }
    qsort(rows, n, sizeof *rows, cmprow);
    for (int i = 0; i < K_PROJ && i < n; i++) memcpy(o + i * F_PROJ, rows[i].v, F_PROJ * sizeof(float));
    o += K_PROJ * F_PROJ;
    /* boxes */
    n = 0;
    for (int k = 0; k < s->nch && n < 4096; k++) {
        const int32_t *r = s->ch + k * C2;
        if (!is_box(r) || r[6] <= 0) continue;
        double dx = (r[3] - mx) / R_SCALE, dy = (r[4] - my) / R_SCALE;
        double v[5] = {1.0, dx, dy, (double)r[6] / (r[7] > 1 ? r[7] : 1), r[6] / 10000.0};
        rows[n].key = dx * dx + dy * dy; rows[n].seq = n; for (int j = 0; j < 5; j++) rows[n].v[j] = (float)v[j]; n++;
    }
    qsort(rows, n, sizeof *rows, cmprow);
    for (int i = 0; i < K_BOX && i < n; i++) memcpy(o + i * F_BOX, rows[i].v, F_BOX * sizeof(float));
    o += K_BOX * F_BOX;
    /* items */
    n = 0;
    for (int k = 0; k < s->nit && n < 4096; k++) {
        const int32_t *r = s->it + k * I2;
        double dx = (r[3] - mx) / R_SCALE, dy = (r[4] - my) / R_SCALE;
        double v[6] = {1.0, dx, dy, (double)(r[1] == CUBE_ITEM), (double)(r[1] == BOMB_ITEM || r[1] == BOMB_BOX_ITEM), (r[5] > 0 ? r[5] : 0) / 1500.0};
        rows[n].key = dx * dx + dy * dy; rows[n].seq = n; for (int j = 0; j < 6; j++) rows[n].v[j] = (float)v[j]; n++;
    }
    qsort(rows, n, sizeof *rows, cmprow);
    for (int i = 0; i < K_ITEM && i < n; i++) memcpy(o + i * F_ITEM, rows[i].v, F_ITEM * sizeof(float));
    /* egocentric grid */
    int half = GRID / 2, st = gas_step(tick);
    for (int gy = 0; gy < GRID; gy++) for (int gx = 0; gx < GRID; gx++) {
        int x = tx - half + gx, y = ty - half + gy, inside = x >= 0 && x < W && y >= 0 && y < H;
        int cx = x < 0 ? 0 : x > W - 1 ? W - 1 : x, cy = y < 0 ? 0 : y > H - 1 ? H - 1 : y, i = cy * W + cx;
        int a = cx < cy ? cx : cy, b = W - 1 - cx < H - 1 - cy ? W - 1 - cx : H - 1 - cy, gas = (a < b ? a : b) < st;
        grid[0 * 625 + gy * 25 + gx] = m->wall[i] && inside;
        grid[1 * 625 + gy * 25 + gx] = m->water[i] && inside;
        grid[2 * 625 + gy * 25 + gx] = m->bush[i] && inside;
        grid[3 * 625 + gy * 25 + gx] = gas || !inside;
        grid[4 * 625 + gy * 25 + gx] = !inside;
    }
    for (int k = 0; k < s->nch; k++) {
        const int32_t *r = s->ch + k * C2;
        if (!is_brawler(s, r) || r[2] == pidx || r[6] <= 0) continue;
        int gx = floordiv(r[3], TILE) - tx + half, gy = floordiv(r[4], TILE) - ty + half;
        if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) grid[5 * 625 + gy * 25 + gx] += 1.0f;
    }
    return 1;
}
