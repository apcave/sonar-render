#ifndef _OPEN_GL_FACET
#define _OPEN_GL_FACET

#include "Facet.hpp"

#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

/**
 * @brief FacetGl class is used to manage the OpenGL texture and CUDA graphics resource for a facet.
 *
 * The surface pressure values are stored as intermediate products in separate CUDA matrices.
 * The OpenGL texture is used to display the surface pressure values.
 * The pressure values are scaled to the range [0, 1] for display.
 */
class FacetGl
{

public:
    FacetGl();
    void WriteSurface(double *dev_Pr, double *dev_Pi, float *dev_frag_stats);
    void PrintOpenGlTexture();
    void Delete();

private:
    void CreateOpenGl();
    void MapToCuda();

private:
    bool readyToRender;
    GLuint textureID;
    int textureVert[3];
    cudaGraphicsResource *cudaResource;
    cudaResourceDesc resDesc;
    cudaSurfaceObject_t surface;
    cudaTextureDesc texDesc;
    cudaArray_t array;

protected:
    int numXpnts;
    int numYpnts;
    int numXpntsNegative
};
#endif