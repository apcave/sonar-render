#ifndef _OPENGL_TES
#define _OPENGL_TES
#include "Facet.hpp"
#include "FacetGl.hpp"
#include "ObjectGl.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include <vector>

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
class OpenGlTes
{
public:
    GLuint textureShaderProgram = 0;
    GLuint uniformShaderProgram = 0;

private:
    GLFWwindow *window;
    GLuint vbo, vao, texture;
    cudaGraphicsResource *cuda_vbo_resource, *cuda_texture_resource;

public:
    OpenGlTes();
    ~OpenGlTes();

    void Cleanup();

protected:
    void InitOpenGL();
    int MakeObjectOnGl(std::vector<Facet *> facets);
    int MakeTextureOnGl(
        std::vector<std::vector<double *>> &dev_object_facet_Pr,
        std::vector<std::vector<double *>> &dev_object_facet_Pi,
        double *dev_frag_stats);
    void ProcessFrame();

private:
    void MakeTextureShader();
    void PrintTextures();

private:
    int window_width = 800;
    int window_height = 600;

protected:
    std::vector<ObjectGl *> objects;
    bool usingOpenGL = true;
};
#endif