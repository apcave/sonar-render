#ifndef _OBJECT_GL
#define _OBJECT_GL

#include "Facet.hpp"

#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class ObjectGl
{
public:
    ObjectGl(std::vector<Facet *> facets);

    void RenderObject(GLint textureUniformLoc);
    void MakeTextureOnGl(
        std::vector<double *> &dev_facet_Pr,
        std::vector<double *> &dev_facet_Pi,
        double *dev_frag_stats);

private:
    std::vector<FacetGl *> facets;
    GLuint vbo;
    GLuint vao;
};
#endif