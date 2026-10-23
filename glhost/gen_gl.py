#!/usr/bin/env python3
"""Generate the GLES 3.0 forwarding layer from /usr/include/GLES3/gl3.h:
  ../loader/gl_client_gen.c  guest side (AArch64, inside nbhost): one stub per GL function, serialising the call
  nbgl_server_gen.c          host side (x86, inside nbgl): function pointers + a dispatcher replaying the call
Pointer parameters need a size rule (RULES); functions listed in HAND are hand-written in gl_client.c / nbgl.c."""
import re, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open('/usr/include/GLES3/gl3.h').read()
protos = []
for m in re.finditer(r'GL_APICALL (.+?)\s*GL_APIENTRY (gl\w+) \((.*?)\);', src):
    ret, name, params = m.group(1).strip(), m.group(2), m.group(3).strip()
    ps = []
    if params != 'void':
        for p in params.split(','):
            p = p.strip(); pm = re.match(r'(.*?)(\w+)$', p)
            ps.append((pm.group(1).strip(), pm.group(2)))
    protos.append((ret, name, ps))

N = lambda n: {'1': 1, '2': 2, '3': 3, '4': 4}[n]
RULES = {}   # (func, param) -> ('in'|'out'|'str'|'ofs'|'idx', size expression)
def r(f, p, kind, size=None): RULES[(f, p)] = (kind, size)
r('glBindAttribLocation', 'name', 'str')
for f, p in [('glGetAttribLocation', 'name'), ('glGetUniformLocation', 'name'), ('glGetUniformBlockIndex', 'uniformBlockName'),
             ('glGetFragDataLocation', 'name')]: r(f, p, 'str')
r('glBufferData', 'data', 'in', 'size'); r('glBufferSubData', 'data', 'in', 'size')
for f in ['glCompressedTexImage2D', 'glCompressedTexSubImage2D', 'glCompressedTexImage3D', 'glCompressedTexSubImage3D']: r(f, 'data', 'in', 'imageSize')
for f, p in [('glDeleteBuffers', 'buffers'), ('glDeleteFramebuffers', 'framebuffers'), ('glDeleteRenderbuffers', 'renderbuffers'),
             ('glDeleteTextures', 'textures'), ('glDeleteVertexArrays', 'arrays'), ('glDeleteQueries', 'ids'),
             ('glDeleteTransformFeedbacks', 'ids'), ('glDrawBuffers', 'bufs')]: r(f, p, 'in', 'n*4')
r('glDeleteSamplers', 'samplers', 'in', 'count*4')
r('glInvalidateFramebuffer', 'attachments', 'in', 'numAttachments*4'); r('glInvalidateSubFramebuffer', 'attachments', 'in', 'numAttachments*4')
r('glClearBufferfv', 'value', 'in', '(buffer==0x1800?16:4)'); r('glClearBufferiv', 'value', 'in', '(buffer==0x1800?16:4)')
r('glClearBufferuiv', 'value', 'in', '16')
r('glShaderBinary', 'shaders', 'in', 'count*4'); r('glShaderBinary', 'binary', 'in', 'length'); r('glProgramBinary', 'binary', 'in', 'length')
r('glTexImage2D', 'pixels', 'in', 'nb_pix(width,height,1,format,type,0)'); r('glTexSubImage2D', 'pixels', 'in', 'nb_pix(width,height,1,format,type,0)')
r('glTexImage3D', 'pixels', 'in', 'nb_pix(width,height,depth,format,type,0)'); r('glTexSubImage3D', 'pixels', 'in', 'nb_pix(width,height,depth,format,type,0)')
for f in ['glTexParameterfv', 'glTexParameteriv']: r(f, 'params', 'in', '16')
for f in ['glSamplerParameteriv', 'glSamplerParameterfv']: r(f, 'param', 'in', '16')
for n in '1234':
    for t in ['f', 'i', 'ui']: r(f'glUniform{n}{t}v', 'value', 'in', f'count*{N(n)}*4')
    r(f'glVertexAttrib{n}fv', 'v', 'in', f'{N(n)*4}')
for n in '234': r(f'glUniformMatrix{n}fv', 'value', 'in', f'count*{N(n)*N(n)}*4')
for a, b in ['23', '32', '24', '42', '34', '43']: r(f'glUniformMatrix{a}x{b}fv', 'value', 'in', f'count*{int(a)*int(b)}*4')
r('glVertexAttribI4iv', 'v', 'in', '16'); r('glVertexAttribI4uiv', 'v', 'in', '16')
r('glGetActiveUniformsiv', 'uniformIndices', 'in', 'uniformCount*4'); r('glGetActiveUniformsiv', 'params', 'out', 'uniformCount*4')
r('glVertexAttribPointer', 'pointer', 'ofs'); r('glVertexAttribIPointer', 'pointer', 'ofs')
for f in ['glDrawElements', 'glDrawRangeElements', 'glDrawElementsInstanced']: r(f, 'indices', 'idx', 'count*nb_idxsz(type)')
for f, p in [('glGenBuffers', 'buffers'), ('glGenFramebuffers', 'framebuffers'), ('glGenRenderbuffers', 'renderbuffers'),
             ('glGenTextures', 'textures'), ('glGenVertexArrays', 'arrays'), ('glGenQueries', 'ids'),
             ('glGenTransformFeedbacks', 'ids')]: r(f, p, 'out', 'n*4')
r('glGenSamplers', 'samplers', 'out', 'count*4')
for f in ['glGetActiveAttrib', 'glGetActiveUniform', 'glGetTransformFeedbackVarying']:
    r(f, 'length', 'out', '4'); r(f, 'size', 'out', '4'); r(f, 'type', 'out', '4'); r(f, 'name', 'out', 'bufSize')
r('glGetAttachedShaders', 'count', 'out', '4'); r('glGetAttachedShaders', 'shaders', 'out', 'maxCount*4')
r('glGetBooleanv', 'data', 'out', 'nb_get_count(pname)'); r('glGetFloatv', 'data', 'out', 'nb_get_count(pname)*4')
r('glGetIntegerv', 'data', 'out', 'nb_get_count(pname)*4'); r('glGetInteger64v', 'data', 'out', 'nb_get_count(pname)*8')
r('glGetIntegeri_v', 'data', 'out', '4'); r('glGetInteger64i_v', 'data', 'out', '8')
for f in ['glGetBufferParameteriv', 'glGetFramebufferAttachmentParameteriv', 'glGetRenderbufferParameteriv', 'glGetProgramiv',
          'glGetShaderiv', 'glGetTexParameterfv', 'glGetTexParameteriv', 'glGetQueryiv', 'glGetQueryObjectuiv', 'glGetUniformfv',
          'glGetUniformiv', 'glGetUniformuiv']: r(f, 'params', 'out', '4')
r('glGetBufferParameteri64v', 'params', 'out', '8')
for f in ['glGetSamplerParameteriv', 'glGetSamplerParameterfv']: r(f, 'params', 'out', '4')
for f in ['glGetVertexAttribfv', 'glGetVertexAttribiv', 'glGetVertexAttribIiv', 'glGetVertexAttribIuiv']: r(f, 'params', 'out', '(pname==0x8626?16:4)')
r('glGetVertexAttribPointerv', 'pointer', 'out', '8')
for f, p in [('glGetProgramInfoLog', 'infoLog'), ('glGetShaderInfoLog', 'infoLog'), ('glGetShaderSource', 'source'),
             ('glGetActiveUniformBlockName', 'uniformBlockName')]: r(f, 'length', 'out', '4'); r(f, p, 'out', 'bufSize')
r('glGetShaderPrecisionFormat', 'range', 'out', '8'); r('glGetShaderPrecisionFormat', 'precision', 'out', '4')
r('glGetSynciv', 'length', 'out', '4'); r('glGetSynciv', 'values', 'out', 'count*4')
r('glGetInternalformativ', 'params', 'out', 'count*4')
r('glGetProgramBinary', 'length', 'out', '4'); r('glGetProgramBinary', 'binaryFormat', 'out', '4'); r('glGetProgramBinary', 'binary', 'out', 'bufSize')
r('glGetActiveUniformBlockiv', 'params', 'out', 'nb_block_count(program,uniformBlockIndex,pname)*4')
r('glReadPixels', 'pixels', 'out', 'nb_pix(width,height,1,format,type,1)')

HAND = {'glGetString', 'glGetStringi', 'glMapBufferRange', 'glUnmapBuffer', 'glFlushMappedBufferRange', 'glGetBufferPointerv',
        'glShaderSource', 'glTransformFeedbackVaryings', 'glGetUniformIndices', 'glGetError', 'glPixelStorei'}
FLOATS = {'GLfloat', 'GLclampf'}

client, server, names = [], [], []
server_ptrs = []
for op, (ret, name, ps) in enumerate(protos):
    opn = 16 + op
    sig = ', '.join(f'{t} {n}' if not t.endswith('*') else f'{t}{n}' for t, n in ps) or 'void'
    server_ptrs.append(f'static {ret} (GL_APIENTRY *f_{name})({sig if sig != "void" else "void"});')
    names.append((name, opn))
    if name in HAND: continue
    nargs = len(ps) + sum(1 for t, n in ps if RULES.get((name, n), ('',))[0] == 'idx')
    sync = ret != 'void' or any(RULES.get((name, n), ('',))[0] == 'out' for t, n in ps)
    body, sizes, sargs, insz = [], [], [], []
    extra = len(ps)
    for i, (t, n) in enumerate(ps):
        kind = 'val'
        if '*' in t:
            if (name, n) not in RULES: raise SystemExit(f'no rule for {name}.{n} ({t})')
            kind, size = RULES[(name, n)]
        if kind == 'val':
            if t in FLOATS: body.append(f'c.a[{i}] = nb_f2u({n});'); sargs.append(f'nb_u2f(A[{i}])')
            else: body.append(f'c.a[{i}] = (uint64_t)(uintptr_t){n};'); sargs.append(f'({t})(uintptr_t)A[{i}]')
        elif kind == 'in':
            sizes.append(f'size_t s{i} = {n} ? (size_t)({size}) : 0;'); body.append(f'nb_in(&c, {i}, {n}, s{i});'); insz.append(i)
            sargs.append(f'({t})IP({i})')
        elif kind == 'str':
            sizes.append(f'size_t s{i} = {n} ? strlen({n}) + 1 : 0;'); body.append(f'nb_in(&c, {i}, {n}, s{i});'); insz.append(i)
            sargs.append(f'({t})IP({i})')
        elif kind == 'out':
            sizes.append(f'size_t o{i} = {n} ? (size_t)({size}) : 0;')   # before nb_begin: may itself issue GL queries
            body.append(f'nb_out(&c, {i}, {n}, o{i});'); sargs.append(f'({t})OP({i})')
        elif kind == 'ofs':
            body.append(f'c.a[{i}] = (uint64_t)(uintptr_t){n}; nb_check_ofs("{name}", {n});'); sargs.append(f'({t})(uintptr_t)A[{i}]')
        elif kind == 'idx':
            sizes.append(f'int cl{i} = (uintptr_t){n} > 0xffffffffULL; size_t s{i} = cl{i} ? (size_t)({size}) : 0;'); insz.append(i)
            body.append(f'if (cl{i}) nb_in(&c, {i}, {n}, s{i}); else c.a[{i}] = (uint64_t)(uintptr_t){n}; c.a[{extra}] = cl{i};')
            sargs.append(f'(A[{extra}] ? ({t})IP({i}) : ({t})(uintptr_t)A[{i}])')
    blob = ' + '.join(f'NB_A8(s{i})' for i in insz) or '0'
    cl = [f'GL_APICALL {ret} GL_APIENTRY {name} ({sig}) {{']
    cl += ['    ' + s for s in sizes]
    cl.append(f'    NbCmd c; nb_begin(&c, {opn}{" | NBGL_SYNC" if sync else ""}, {nargs}, {blob});')
    cl += ['    ' + b for b in body]
    if sync:
        if ret == 'void': cl.append('    nb_sync(&c);')
        else: cl.append(f'    return ({ret})(uintptr_t)nb_sync(&c);')
    else: cl.append('    nb_end(&c);')
    cl.append('}')
    client.append('\n'.join(cl))
    call = f'f_{name}({", ".join(sargs)})'
    server.append(f'    case {opn}: ' + (f'R = (uint64_t)(uintptr_t){call};' if ret != 'void' else f'{call};') + ' break;')

with open(os.path.join(HERE, '..', 'loader', 'gl_client_gen.c'), 'w') as f:
    f.write('/* generated by work/glhost/gen_gl.py from GLES3/gl3.h - do not edit */\n#include "gl_client.h"\n\n')
    f.write('\n'.join(client))
    f.write('\n\nconst NbProc nb_procs[] = {\n' + ''.join(f'    {{"{n}", (void *){n}}},\n' for n, o in names) + '    {0, 0}\n};\n')
with open(os.path.join(HERE, 'nbgl_server_gen.c'), 'w') as f:
    f.write('/* generated by gen_gl.py from GLES3/gl3.h - do not edit */\n')
    f.write('\n'.join(server_ptrs) + '\n\nstatic void nb_load_procs(void) {\n')
    f.write(''.join(f'    f_{n} = (void *)eglGetProcAddress("{n}");\n' for n, o in names) + '}\n\n')
    f.write('static const char *nb_op_name(unsigned op) {\n    switch (op) {\n' + ''.join(f'    case {o}: return "{n}";\n' for n, o in names) + '    }\n    return "?";\n}\n\n')
    f.write('/* returns the value of non-void calls; A = args, IP/OP = in/out pointers */\n')
    f.write('static uint64_t nb_dispatch(unsigned op, const uint64_t *A, uint8_t *cmd, uint8_t *reply) {\n    uint64_t R = 0;\n    switch (op) {\n')
    f.write('\n'.join(server) + '\n    default: nb_special(op, A, cmd, reply, &R);\n    }\n    return R;\n}\n')
print(len(protos), 'functions,', len(client), 'generated stubs,', len(HAND), 'hand-written')
