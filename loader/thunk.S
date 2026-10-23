// Generic thunk entry: runtime-generated stubs load x16 = data, x17 = this entry, then `br x17`.
// We save x0-x7 / d0-d7, call  uint64_t thunk_dispatch(void *data, uint64_t *xregs, double *dregs, void *lr)
// and return its result in x0 (and d0 = bit copy, for float-returning slots).
    .text
    .globl thunk_entry
    .type thunk_entry, %function
thunk_entry:
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x0, x1, [sp, #16]
    stp x2, x3, [sp, #32]
    stp x4, x5, [sp, #48]
    stp x6, x7, [sp, #64]
    stp d0, d1, [sp, #80]
    stp d2, d3, [sp, #96]
    stp d4, d5, [sp, #112]
    stp d6, d7, [sp, #128]
    mov x0, x16
    add x1, sp, #16
    add x2, sp, #80
    mov x3, x30
    bl thunk_dispatch
    fmov d0, x0
    ldp x29, x30, [sp], #160
    ret
    .size thunk_entry, .-thunk_entry
