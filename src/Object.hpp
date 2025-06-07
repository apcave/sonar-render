#ifndef _OBJECT
#define _OBJECT

#include "ObjectCuda.hpp"
#include "Object.hpp"

class Object : public ObjectCuda
{

public:
    Object(ObjectType type, float t_resolution);

    ~Object();
    void AddFacet(float3 v1, float3 v2, float3 v3);
    void MakeFragmentData();
    std::vector<float3> &GetCentroids();

private:
    std::vector<float3> centroids;

};

#endif