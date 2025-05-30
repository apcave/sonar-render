#ifndef _OBJECT_CUDA
#define _OBJECT_CUDA

#include "ObjectGl.hpp"
#include "ModelShared.h"

class ObjectCuda : public ObjectGl
{
public:
    ObjectCuda();
    ~ObjectCuda();
    void MakeCudaObjects();
    void GetSurfaceScalers(float *dev_frag_stats);
    void WriteSurfaceToGlTexture(float *dev_frag_stats);
    void PrintSurfacePressure();
    dev_object MakeOptixStructArray();
    void AccumulatePressure();

private:
    dev_object h_obj = {};
};

#endif