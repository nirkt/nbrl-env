/* ELF loader for Android AArch64 shared objects (RELA, BIND_NOW; no TLS relocations needed here). */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

Module g_mods[MAX_MODULES];
int g_nmods;

static uint8_t *read_file(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *b = malloc(st.st_size);
    size_t got = 0;
    while (got < (size_t)st.st_size) { ssize_t r = read(fd, b + got, st.st_size - got); if (r <= 0) break; got += r; }
    close(fd); *len = got; return b;
}

static uint32_t gnu_hash_nsyms(const uint32_t *gh) {
    uint32_t nbuckets = gh[0], symoffset = gh[1], bloom_size = gh[2];
    const uint64_t *bloom = (const uint64_t *)(gh + 4);
    const uint32_t *buckets = (const uint32_t *)(bloom + bloom_size);
    const uint32_t *chain = buckets + nbuckets;
    uint32_t last = 0;
    for (uint32_t i = 0; i < nbuckets; i++) if (buckets[i] > last) last = buckets[i];
    if (last < symoffset) return symoffset;
    while (!(chain[last - symoffset] & 1)) last++;
    return last + 1;
}

Module *load_module(const char *path, const char *name) {
    size_t len; uint8_t *f = read_file(path, &len);
    Elf64_Ehdr *eh = (Elf64_Ehdr *)f;
    if (memcmp(eh->e_ident, ELFMAG, 4) || eh->e_machine != EM_AARCH64) { LOG("%s: not an AArch64 ELF\n", path); exit(1); }
    Elf64_Phdr *ph = (Elf64_Phdr *)(f + eh->e_phoff);
    uint64_t hi = 0;
    for (int i = 0; i < eh->e_phnum; i++)
        if (ph[i].p_type == PT_LOAD && ph[i].p_vaddr + ph[i].p_memsz > hi) hi = ph[i].p_vaddr + ph[i].p_memsz;
    size_t span = (hi + 0xffff) & ~0xffffUL;
    uint8_t *base = mmap(NULL, span, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); exit(1); }
    for (int i = 0; i < eh->e_phnum; i++)
        if (ph[i].p_type == PT_LOAD) memcpy(base + ph[i].p_vaddr, f + ph[i].p_offset, ph[i].p_filesz);
    Module *m = &g_mods[g_nmods++];
    memset(m, 0, sizeof *m);
    snprintf(m->name, sizeof m->name, "%s", name);
    m->base = base; m->span = span;
    m->phnum = eh->e_phnum;
    for (int i = 0; i < eh->e_phnum; i++) if (ph[i].p_type == PT_PHDR) m->phdr = (Elf64_Phdr *)(base + ph[i].p_vaddr);
    if (!m->phdr) m->phdr = (Elf64_Phdr *)(base + eh->e_phoff);   /* first PT_LOAD covers the headers */
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_DYNAMIC) continue;
        for (Elf64_Dyn *d = (Elf64_Dyn *)(base + ph[i].p_vaddr); d->d_tag != DT_NULL; d++) {
            switch (d->d_tag) {
            case DT_SYMTAB: m->symtab = (Elf64_Sym *)(base + d->d_un.d_ptr); break;
            case DT_STRTAB: m->strtab = (const char *)(base + d->d_un.d_ptr); break;
            case DT_HASH: m->nsyms = ((uint32_t *)(base + d->d_un.d_ptr))[1]; break;
            case DT_INIT_ARRAY: m->init_array = (uint64_t *)(base + d->d_un.d_ptr); break;
            case DT_INIT_ARRAYSZ: m->n_init = d->d_un.d_val / 8; break;
            case DT_INIT: m->init_fn = (void (*)(void))(base + d->d_un.d_ptr); break;
            }
        }
        for (Elf64_Dyn *d = (Elf64_Dyn *)(base + ph[i].p_vaddr); d->d_tag != DT_NULL; d++)
            if (d->d_tag == DT_GNU_HASH && !m->nsyms) m->nsyms = gnu_hash_nsyms((uint32_t *)(base + d->d_un.d_ptr));
    }
    free(f);
    LOG("[load] %-18s base=%p span=%#zx nsyms=%u init_array=%zu\n", name, base, span, m->nsyms, m->n_init);
    return m;
}

void *module_sym(Module *m, const char *name) {
    for (uint32_t i = 1; i < m->nsyms; i++) {
        const Elf64_Sym *s = &m->symtab[i];
        if (s->st_shndx == SHN_UNDEF || !s->st_name) continue;
        int bind = ELF64_ST_BIND(s->st_info);
        if (bind != STB_GLOBAL && bind != STB_WEAK) continue;
        if (!strcmp(m->strtab + s->st_name, name)) return m->base + s->st_value;
    }
    return NULL;
}
Module *module_by_name(const char *name) {
    for (int i = 0; i < g_nmods; i++) if (!strcmp(g_mods[i].name, name)) return &g_mods[i];
    return NULL;
}
Module *module_for_addr(const void *p) {
    for (int i = 0; i < g_nmods; i++)
        if ((const uint8_t *)p >= g_mods[i].base && (const uint8_t *)p < g_mods[i].base + g_mods[i].span) return &g_mods[i];
    return NULL;
}
const char *addr_desc(const void *p) {
    static __thread char buf[96];
    Module *m = module_for_addr(p);
    if (m) snprintf(buf, sizeof buf, "%s+%#lx", m->name, (unsigned long)((const uint8_t *)p - m->base));
    else snprintf(buf, sizeof buf, "%p", p);
    return buf;
}

/* missing imports: a thunk that logs the name and returns 0 */
static int g_missing_abort;
static uint64_t missing_dispatch(void *data, uint64_t *x, double *d, void *lr) {
    (void)d;
    LOG("[missing] %s(%#lx, %#lx, %#lx) called from %s\n", (const char *)data, x[0], x[1], x[2], addr_desc(lr));
    if (g_missing_abort) abort();
    return 0;
}
static uint8_t g_missing_data[1 << 16];

void *find_symbol(const char *name) {
    void *p = shim_lookup(name);
    if (p) return p;
    for (int i = 0; i < g_nmods; i++) if ((p = module_sym(&g_mods[i], name))) return p;
    return NULL;
}

int g_nmissing;
void relocate_module(Module *m) {
    int counts[4] = {0};
    for (int i = 0; i < m->phnum; i++) {
        if (m->phdr[i].p_type != PT_DYNAMIC) continue;
        uint64_t rela = 0, relasz = 0, jmprel = 0, pltsz = 0;
        for (Elf64_Dyn *d = (Elf64_Dyn *)(m->base + m->phdr[i].p_vaddr); d->d_tag != DT_NULL; d++) {
            if (d->d_tag == DT_RELA) rela = d->d_un.d_ptr;
            if (d->d_tag == DT_RELASZ) relasz = d->d_un.d_val;
            if (d->d_tag == DT_JMPREL) jmprel = d->d_un.d_ptr;
            if (d->d_tag == DT_PLTRELSZ) pltsz = d->d_un.d_val;
        }
        for (int pass = 0; pass < 2; pass++) {
            Elf64_Rela *r = (Elf64_Rela *)(m->base + (pass ? jmprel : rela));
            size_t n = (pass ? pltsz : relasz) / sizeof(Elf64_Rela);
            for (size_t k = 0; k < n; k++) {
                uint64_t *where = (uint64_t *)(m->base + r[k].r_offset);
                uint32_t type = ELF64_R_TYPE(r[k].r_info), si = ELF64_R_SYM(r[k].r_info);
                if (type == R_AARCH64_RELATIVE) { *where = (uint64_t)m->base + r[k].r_addend; counts[0]++; continue; }
                const Elf64_Sym *s = &m->symtab[si];
                const char *nm = m->strtab + s->st_name;
                void *S = NULL;
                /* our shims interpose libc-level names even when a module defines them */
                S = shim_lookup(nm);
                if (!S && s->st_shndx != SHN_UNDEF) S = m->base + s->st_value;
                if (!S) for (int j = 0; j < g_nmods && !S; j++) S = module_sym(&g_mods[j], nm);
                if (!S) {
                    g_nmissing++;
                    if (ELF64_ST_TYPE(s->st_info) == STT_OBJECT) S = g_missing_data;
                    else S = make_thunk(missing_dispatch, (void *)nm);
                    VLOG("[unresolved] %s: %s (type %d)\n", m->name, nm, ELF64_ST_TYPE(s->st_info));
                }
                switch (type) {
                case R_AARCH64_ABS64: *where = (uint64_t)S + r[k].r_addend; counts[1]++; break;
                case R_AARCH64_GLOB_DAT: case R_AARCH64_JUMP_SLOT: *where = (uint64_t)S + r[k].r_addend; counts[2]++; break;
                default: LOG("[reloc] %s: unsupported type %u at %#lx\n", m->name, type, (unsigned long)r[k].r_offset); counts[3]++;
                }
            }
        }
    }
    LOG("[reloc] %-18s relative=%d abs64=%d glob/jump=%d unsupported=%d (missing so far %d)\n", m->name,
        counts[0], counts[1], counts[2], counts[3], g_nmissing);
    __builtin___clear_cache((char *)m->base, (char *)m->base + m->span);
}

void run_initializers(Module *m) {
    if (m->init_fn) { VLOG("[init] %s DT_INIT %p\n", m->name, m->init_fn); m->init_fn(); }
    for (size_t i = 0; i < m->n_init; i++) {
        uint64_t f = m->init_array[i];
        if (f == 0 || f == (uint64_t)-1) continue;
        VLOG("[init] %s init_array[%zu] = %s\n", m->name, i, addr_desc((void *)f));
        ((void (*)(int, char **, char **))f)(0, NULL, NULL);
    }
    LOG("[init] %-18s %zu initializers done\n", m->name, m->n_init);
}

/* thunk pages */
extern void thunk_entry(void);
static uint8_t *thunk_page; static size_t thunk_used, thunk_cap;
void *make_thunk(thunk_fn fn, void *data) {
    struct { void *data; thunk_fn fn; } *box = malloc(sizeof *box);
    box->data = data; box->fn = fn;
    if (!thunk_page || thunk_used + 32 > thunk_cap) {
        thunk_cap = 1 << 20;
        thunk_page = mmap(NULL, thunk_cap, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        thunk_used = 0;
    }
    uint32_t *c = (uint32_t *)(thunk_page + thunk_used); thunk_used += 32;
    c[0] = 0x58000090;   /* ldr x16, #16  (box) */
    c[1] = 0x580000b1;   /* ldr x17, #20  (thunk_entry) */
    c[2] = 0xd61f0220;   /* br  x17 */
    c[3] = 0xd503201f;   /* nop */
    *(void **)(c + 4) = box;
    *(void **)(c + 6) = (void *)thunk_entry;
    __builtin___clear_cache((char *)c, (char *)(c + 8));
    return c;
}
uint64_t thunk_dispatch(void *box, uint64_t *x, double *d, void *lr) {
    struct { void *data; thunk_fn fn; } *b = box;
    return b->fn(b->data, x, d, lr);
}
