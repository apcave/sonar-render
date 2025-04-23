#ifndef _OBJECT
#define _OBJECT

#include "ObjectCuda.hpp"

class Object : public ObjectCuda
{
public:
    Object();
    ~Object();
    void AddFacet(float3 v1, float3 v2, float3 v3);
    void MakeFragmentData(float frag_length);
};

#endif