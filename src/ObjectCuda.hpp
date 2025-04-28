#ifndef _OBJECT_CUDA
#define _OBJECT_CUDA

#include "ObjectGl.hpp"

class ObjectCuda : public ObjectGl
{
public:
    ObjectCuda();
    ~ObjectCuda();
    void MakeCudaObjects();
    void GetSurfaceScalers(float *dev_frag_stats);
    void WriteSurfaceToGlTexture(float *dev_frag_stats);
    void PrintSurfacePressure();
};

#endif