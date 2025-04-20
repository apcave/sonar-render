#ifndef _OPENGL_TES
#define _OPENGL_TES
#include "Facet.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include <vector>

class FacetGL
{

public:
    GLuint textureID;
    int textureVert[3];
    cudaGraphicsResource *cudaResource;
    cudaSurfaceObject_t surface;
    cudaArray_t array;
    int numXpnts;
    int numYpnts;
};

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
public:
    GLuint textureShaderProgram = 0;
    GLuint uniformShaderProgram = 0;

private:
    GLFWwindow *window;
    GLuint vbo, vao, texture;
    cudaGraphicsResource *cuda_vbo_resource, *cuda_texture_resource;

public:
    OpenGL_TES();
    ~OpenGL_TES();

    // Captures the thread but enables zooming etc.
    void RenderGL();

    void Cleanup();

protected:
    void InitOpenGL();
    int MakeObjectOnGL(std::vector<Facet *> facets);
    void CreateTexture(int numXpnts, int numYpnts, GLuint *texture);
    void ProcessFrame();

private:
    void CreateBuffers();
    void UpdateBuffers();
    void RenderObject();

    void MakeTextureShader();
    void PrintTextures();

private:
    int window_width = 800;
    int window_height = 600;

protected:
    std::vector<FacetGL *> gl_object_facets;
    bool usingOpenGL = true;
};
#endif