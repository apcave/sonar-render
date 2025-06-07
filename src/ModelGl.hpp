#ifndef _MODEL_GL
#define _MODEL_GL
#include "Object.hpp"
#include "PressurePoint.hpp"

#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * @brief OpenGL class is used to manage the C style OpenGL API for rendering.
 *
 * It provides a wrapper around the OpenGL functions to make it easier to use.
 * It is intended to be used in conjunction with the ModelTes class for rendering the results of the calculations.
 *
 * The purpose of the program is to render the effect of an acoustic wave on a surface.
 * Initially the geometry is static and calculations are done on phase and frequency.
 * So the geometry is static and the textures are updated.
 */
class ModelGl
{
public:
private:
    // The openGL window.

    // This is the shader that processes the texture at a fragment level.
    GLuint textureShaderProgram = 0;

public:
    ModelGl();
    ~ModelGl();

    void FreeGl();

protected:
    void InitOpenGL();
    int MakeObjectsOnGl();
    int MakeTextureOnGl(double *dev_frag_stats);
    void ProcessFrame();

private:
    void MakeTextureShader();
    void PrintTextures();

private:
    int window_width = 800 * 8;
    int window_height = 600 * 8;

private:
    EGLDisplay eglDisplay;
    EGLContext eglContext;
    EGLSurface eglSurface;

    glm::mat4 view;
    glm::mat4 projection;

protected:
    std::vector<Object *> targetObjects;
    std::vector<Object *> fieldObjects;
    std::vector<Object *> sourceObjects;

    std::vector<PressurePoint *> sourcePoints;
    std::vector<PressurePoint *> fieldPoints;
};
#endif