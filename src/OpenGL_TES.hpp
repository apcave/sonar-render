#ifndef _OPENGL_TES
#define _OPENGL_TES

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

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

class OpenGL_TES
{
private:
    GLFWwindow *window;
    GLuint vbo, texture;
    cudaGraphicsResource *cuda_vbo_resource, *cuda_texture_resource;

public:
    OpenGL_TES();
    ~OpenGL_TES();

    // Captures the thread but enables zooming etc.
    void RenderGL();

    void Cleanup();

private:
    void InitOpenGL();
    void CreateBuffers();
    void UpdateBuffers();
    void RenderObject();

private:
    int window_width = 800;
    int window_height = 600;
};
#endif