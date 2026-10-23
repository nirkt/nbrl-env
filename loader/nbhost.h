/* nbhost: run Null's Brawl's libg.so (Android/bionic, AArch64) as an ordinary Linux process.
 * The host binary is a static AArch64 glibc program; on this x86 PC it runs under qemu-aarch64 (user mode).
 * It maps the Android .so files itself, binds their imports to bionic-ABI shims written over glibc,
 * fakes JNI, and calls into the game. No Android, no ADB, no phone. */
#pragma once
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <elf.h>

typedef struct Module {
    char name[64];
    uint8_t *base;              /* load bias (link address 0 -> base) */
    size_t span;
    const Elf64_Phdr *phdr; int phnum;
    const Elf64_Sym *symtab; const char *strtab; uint32_t nsyms;
    uint64_t *init_array; size_t n_init;
    void (*init_fn)(void);
} Module;

#define MAX_MODULES 8
extern Module g_mods[MAX_MODULES];
extern int g_nmods;

/* logging */
extern int g_verbose;
extern FILE *g_log;
#define LOG(...) do { fprintf(g_log, __VA_ARGS__); fflush(g_log); } while (0)
#define VLOG(...) do { if (g_verbose) { fprintf(g_log, __VA_ARGS__); fflush(g_log); } } while (0)

/* modules */
Module *load_module(const char *path, const char *name);
void relocate_module(Module *m);
void run_initializers(Module *m);
void *module_sym(Module *m, const char *name);
Module *module_by_name(const char *name);
Module *module_for_addr(const void *p);
void *find_symbol(const char *name);   /* shims, then module exports */

/* thunks: tiny code stubs that enter thunk_dispatch with a data pointer */
typedef uint64_t (*thunk_fn)(void *data, uint64_t *x, double *d, void *lr);
void *make_thunk(thunk_fn fn, void *data);

/* shim table */
typedef struct { const char *name; void *addr; } Shim;
void *shim_lookup(const char *name);
void shims_init(void);
extern const char *g_asset_root;

/* fake JNI */
void *jni_vm(void);
void *jni_env(void);
void *jni_new_string(const char *s);
void *jni_new_object(const char *cls);
const char *jni_obj_desc(void *o);
void jni_dump_summary(void);
typedef struct { const char *cls; const char *name; const char *sig; void *fn; } NativeReg;
NativeReg *jni_find_native(const char *cls, const char *name);
extern int g_jni_nnatives;
extern NativeReg g_jni_natives[];

/* address printing */
const char *addr_desc(const void *p);
