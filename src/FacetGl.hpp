#ifndef _OPEN_GL_FACET
#define _OPEN_GL_FACET

#include "Globals.h"

#include <GL/glew.h>
#include <EGL/egl.h>
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
    GLuint textureID = 0;

public:
    FacetGl();
    ~FacetGl();
    void AllocateGl();

    void PrintOpenGlTexture();

protected:
    void MapToCuda();

private:
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    cudaArray_t array;

protected:
    int numXpnts;
    int numYpnts;
    int numXpntsNegative;

    // Used by CUDA level to write to the OpenGL texture.
    cudaSurfaceObject_t surface;
    cudaGraphicsResource *cudaResource = nullptr;
    bool readyToRender;

    ObjectType objectType = OBJECT_TYPE_TARGET;
};
#endif