/* Spike 3a: headless GLES 3.2 context on this PC's GPU (EGL surfaceless, Mesa), print what the game would see. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    EGLint ma, mi; if (!eglInitialize(d, &ma, &mi)) { printf("eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ca[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE}, n; EGLConfig c;
    eglChooseConfig(d, ca, &c, 1, &n);
    EGLint xa[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE};
    EGLContext x = eglCreateContext(d, n ? c : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, xa);
    if (!eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, x)) { printf("makecurrent failed\n"); return 1; }
    printf("EGL %d.%d vendor %s\nGL_VENDOR %s\nGL_RENDERER %s\nGL_VERSION %s\nGLSL %s\n", ma, mi, eglQueryString(d, EGL_VENDOR),
           glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
    GLint ne; glGetIntegerv(GL_NUM_EXTENSIONS, &ne); printf("extensions: %d\n", ne);
    const char *want[] = {"GL_KHR_texture_compression_astc_ldr", "GL_OES_texture_compression_astc", "GL_EXT_texture_compression_s3tc",
                          "GL_OES_compressed_ETC1_RGB8_texture", "GL_EXT_texture_compression_bptc", "GL_OES_get_program_binary",
                          "GL_EXT_texture_compression_astc_decode_mode", NULL};
    for (int k = 0; want[k]; k++) { int has = 0; for (int i = 0; i < ne; i++) if (!strcmp((const char *)glGetStringi(GL_EXTENSIONS, i), want[k])) has = 1; printf("  %-45s %s\n", want[k], has ? "yes" : "no"); }
    GLint nf; glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &nf); GLint f[512]; glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, f);
    int astc = 0, etc2 = 0; for (int i = 0; i < nf; i++) { if (f[i] >= 0x93B0 && f[i] <= 0x93DD) astc++; if (f[i] >= 0x9270 && f[i] <= 0x9279) etc2++; }
    printf("compressed formats: %d (ASTC %d, ETC2/EAC %d)\n", nf, astc, etc2);
    /* upload one ASTC 8x8 block-compressed 512x512 texture and check it is accepted */
    static unsigned char blk[512 / 8 * 512 / 8 * 16]; GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, 0x93B7 /* COMPRESSED_RGBA_ASTC_8x8_KHR */, 512, 512, 0, sizeof blk, blk);
    printf("ASTC 8x8 upload glGetError = %#x\n", glGetError());
    return 0;
}
