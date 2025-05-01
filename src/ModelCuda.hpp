#ifndef _MODEL_CUDA
#define _MODEL_CUDA
#include "Collision.hpp"
#include "Facet.hpp"
#include "ModelGl.hpp"
#include "dcomplex.h"
#include "PressurePoint.hpp"

using namespace std;

/**
 * @brief CudaModelTes is a class that manages the GPU resources and
 *        calculations for the pressure field in a 3D model.
 *
 * Memory on the device is allocated and stored in points dev_.
 * Data to be used on the host processor is marked with host_.
 *
 * The function definitions are spread across multiple CUDA files.
 * The CUDA kernels map some of the class functions as C style functions.
 * Each CUDA kernel is defined in a separate file with the C++ class function
 * that calls it.
 */

class ModelCuda : public ModelGl
{
protected:
    // Global parameters.
    // Complex wave number.
    dcomplex k_wave;
    float frag_length;
    dcomplex *dev_k_wave = 0;
    // Pixel length and width.
    float *dev_frag_delta = 0;

    // Used for scaling the textures.
    float *dev_frag_stats = 0;
    float host_frag_stats[3];

    int host_num_source_points;
    float3 *dev_source_points_position = 0;   // constant
    dcomplex *dev_source_points_pressure = 0; // constant

    int host_num_field_points;
    float3 *dev_field_points_position = 0;   // constant
    dcomplex *dev_field_points_pressure = 0; // constant

public:
    ModelCuda();
    ~ModelCuda();

protected:
    int SetGlobalParameters(dcomplex k_wave, float pixel_delta);
    int MakeObjectOnGPU(vector<Facet *> facets);
    int MakeSourcePointsOnGPU(vector<PressurePoint *> source_points);
    int MakeFieldPointsOnGPU(vector<PressurePoint *> field_points);
    int StartCuda();
    int StopCuda();

    int DoCalculations();

    int ProjectSourcePointsToFacet(std::vector<Object *> &target);
    int ProjectFromFacetsToFacets(std::vector<Object *> &scrObjects, std::vector<Object *> &dstObjects, bool reciprocity);
    int ProjectFromFacetsToFieldPoints();

    int GetFieldPointValGPU(dcomplex *field_points_pressure);

    void RenderCudaObjects();

    void MakeTextureOnGl();

    void WriteCudaToGlTexture();

    int GetSurfaceScalers();

    void ProjectSrcPointsToObjects();

protected:
    Collision optiXCol;
    bool cudaStarted = false;
};

#endif