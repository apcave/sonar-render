#ifndef _OBJECT_GL
#define _OBJECT_GL

#include "Globals.h"
#include "Facet.hpp"

#include <vector>
#include <epoxy/egl.h>
#include <epoxy/gl.h>

class ObjectGl
{
public:
    std::vector<Facet *> facets;

public:
    ObjectGl();
    ~ObjectGl();

    void RenderObject(GLint textureUniformLoc);

    void AllocateGl();
    void FreeGl();

private:
    void PrintVBO();
    void MakeVBO();

private:
    GLuint vbo = 0;
    GLuint vao = 0;

public:
    ObjectType objectType = OBJECT_TYPE_TARGET;
};
#endif