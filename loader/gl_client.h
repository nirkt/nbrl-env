/* guest side of the GL bridge (see work/glhost/nbgl_proto.h) */
#pragma once
#include "nbhost.h"
#include <string.h>
#include <GLES3/gl3.h>
#include "../glhost/nbgl_proto.h"

typedef struct { const char *name; void *fn; } NbProc;
extern const NbProc nb_procs[];
typedef struct { uint8_t *base; uint64_t *a; size_t blob, len; int nout; struct { void *dst; size_t n; size_t off; } out[8]; size_t roff; } NbCmd;
#define NB_A8(n) (((n) + 7) & ~(size_t)7)
void nb_begin(NbCmd *c, uint32_t op, int nargs, size_t blobs);
void nb_in(NbCmd *c, int i, const void *p, size_t n);
void nb_out(NbCmd *c, int i, void *p, size_t n);
void nb_end(NbCmd *c);
uint64_t nb_sync(NbCmd *c);
size_t nb_pix(int w, int h, int d, GLenum format, GLenum type, int pack);
size_t nb_idxsz(GLenum type);
int nb_get_count(GLenum pname);
int nb_block_count(GLuint prog, GLuint idx, GLenum pname);
void nb_check_ofs(const char *fn, const void *p);
static inline uint64_t nb_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
void nb_dbg_use(unsigned program);
void nb_dbg_mat(int location, int count, unsigned char transpose, const float *value);
void nb_dbg_draw(int count);
void nb_dbg_bind(unsigned target, unsigned buffer);
void nb_dbg_bsd(unsigned target, long offset, long size, const void *data);
