#ifndef _OBJECT_GL
#define _OBJECT_GL

#include "Facet.hpp"

#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class ObjectGl
{
public:
    std::vector<Facet *> facets;

public:
    ObjectGl();
    ~ObjectGl();

    void RenderObject(GLint textureUniformLoc);

    void AllocateGl();

private:
    void PrintVBO();
    void MakeVBO();

private:
    GLuint vbo;
    GLuint vao;
};
#endif