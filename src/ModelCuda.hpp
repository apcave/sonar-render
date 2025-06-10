#ifndef _MODEL_CUDA
#define _MODEL_CUDA
#include "OptiX.hpp"
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

    dev_scratch_pad *dev_scratch = 0; // Used for OptiX calculations.

    int host_num_source_points;
    float3 *dev_source_points_position = 0;   // constant
    dcomplex *dev_source_points_pressure = 0; // constant

    int host_num_field_points;
    float3 *dev_field_points_position = 0;   // constant
    dcomplex *dev_field_points_pressure = 0; // constant

    float max_dB = -20.0f; // Maximum dB value for rendering
    float min_dB = -90.0f; // Minimum dB value for rendering
    bool render_phase_target = true;
    bool render_phase_field = true;

public:
    ModelCuda();
    ~ModelCuda();

    void SoundVisualisationInit(float maxDbValue, float minDbValue, bool renderPhaseTarget, bool renderPhaseField);
    void ProjectSrcPointsToObjects(bool projectFieldObjects);
    void ProjectTargetToFieldObjects();
    void ProjectTargetToFieldPoints();
    void ProjectTargetToTargetObjects(int number_reflections);

protected:
    int SetGlobalParameters(dcomplex k_wave, float pixel_delta);
    int MakeObjectOnGPU(vector<Facet *> facets);
    int MakeSourcePointsOnGPU(vector<PressurePoint *> source_points);
    int MakeFieldPointsOnGPU(vector<PressurePoint *> field_points);
    int StartCuda();
    int StopCuda();

    int DoCalculations();

    int GetFieldPointValGPU(dcomplex *field_points_pressure);

    void RenderCudaObjects();

    void MakeTextureOnGl();

    void WriteCudaToGlTexture();

    int GetSurfaceScalers();

protected:
    OptiX optiX;
    bool cudaStarted = false;
};

#endif