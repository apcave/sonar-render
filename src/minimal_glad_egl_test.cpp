// minimal_glad_egl_test.cpp
#include <iostream>
#include <glad/egl.h>
#include <glad/gl.h>

int main()
{
    // 1. Get EGL display
    EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY)
    {
        std::cerr << "Failed to get EGL display" << std::endl;
        return 1;
    }

    // 2. Initialize EGL
    if (!eglInitialize(eglDisplay, nullptr, nullptr))
    {
        std::cerr << "Failed to initialize EGL" << std::endl;
        return 1;
    }

    // 3. Choose EGL config
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs;
    if (!eglChooseConfig(eglDisplay, configAttribs, &eglConfig, 1, &numConfigs) || numConfigs == 0)
    {
        std::cerr << "Failed to choose EGL config" << std::endl;
        return 1;
    }

    // 4. Create a PBuffer surface
    EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        16,
        EGL_HEIGHT,
        16,
        EGL_NONE,
    };
    EGLSurface eglSurface = eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs);
    if (eglSurface == EGL_NO_SURFACE)
    {
        std::cerr << "Failed to create EGL PBuffer surface" << std::endl;
        return 1;
    }

    // 5. Bind OpenGL API
    if (!eglBindAPI(EGL_OPENGL_API))
    {
        std::cerr << "Failed to bind OpenGL API" << std::endl;
        return 1;
    }

    // 6. Create OpenGL context
    EGLint ctxAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
    EGLContext eglContext = eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, ctxAttribs);
    if (eglContext == EGL_NO_CONTEXT)
    {
        std::cerr << "Failed to create EGL context" << std::endl;
        return 1;
    }

    // 7. Make context current
    if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
    {
        std::cerr << "Failed to make EGL context current" << std::endl;
        return 1;
    }

    // 8. Load OpenGL functions with GLAD
    if (!gladLoadGL((GLADloadfunc)eglGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return 1;
    }

    // 9. Print OpenGL version
    const GLubyte *version = glGetString(GL_VERSION);
    if (version)
        std::cout << "OpenGL version: " << version << std::endl;
    else
        std::cerr << "Failed to get OpenGL version string!" << std::endl;

    // Cleanup
    eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(eglDisplay, eglContext);
    eglDestroySurface(eglDisplay, eglSurface);
    eglTerminate(eglDisplay);

    return 0;
}