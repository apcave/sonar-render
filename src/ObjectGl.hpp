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
    ObjectGl(std::vector<Facet *> facets);
    ~ObjectGl();

    void RenderObject(GLint textureUniformLoc);

    void AllocateGl();

private:
    GLuint vbo;
    GLuint vao;
};
#endif