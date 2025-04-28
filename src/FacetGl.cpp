#include "FacetGl.hpp"
#include "OptiX/Exception.h"
#include <iostream>
#include <vector>

FacetGl::FacetGl()
{
    readyToRender = false;
}

void FacetGl::AllocateGl()
{
    readyToRender = false;
    if (!glfwGetCurrentContext())
    {
        std::cout << "Error Intializing OpenGL context before creating a texture." << std::endl;
        return;
    }

    // Create the OpenGl texture buffer and textureID.
    // Note buffer is always a float4 buffer even through it only uses a single channel.

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Allocate memory for a single-channel float texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, numXpnts, numYpnts, 0, GL_RED, GL_FLOAT, NULL);

    // Set texture filtering parameters
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Unbind the texture
    glBindTexture(GL_TEXTURE_2D, 0);
}

FacetGl::~FacetGl()
{
    // Unregister the OpenGL texture from CUDA
    if (cudaResource)
    {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }

    // Delete the OpenGL texture
    if (textureID)
    {
        glDeleteTextures(1, &textureID);
        textureID = 0;
    }
}

void FacetGl::MapToCuda()
{
    readyToRender = false;

    if (!glIsTexture(textureID))
    {
        std::cout << "Texture ID " << textureID << " is not valid." << std::endl;
        return;
    }

    if (cudaResource)
    {
        std::cout << "Error cudaResource not null." << std::endl;
        return;
    }

    // Map the OpenGL texture to CUDA
    cudaError_t err;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    err = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (err != cudaSuccess)
    {
        printf("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);
    if (err != cudaSuccess)
    {
        printf("cudaGraphicsSubResourceGetMappedArray failed: %s\n", cudaGetErrorString(err));
        return;
    }

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    err = cudaCreateSurfaceObject(&surface, &resDesc);
    if (err != cudaSuccess)
    {
        printf("cudaCreateSurfaceObject failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

void FacetGl::PrintOpenGlTexture()
{
    if (!glIsTexture(textureID))
    {
        std::cout << "Texture ID " << textureID << " is not valid." << std::endl;
    }
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Allocate a buffer to store the texture data
    std::vector<float> textureData(numXpnts * numYpnts * 4); // Assuming RGBA format

    // Copy the texture data to the buffer
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, textureData.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    std::cout << "Printing OpenGL texture data:" << std::endl;
    for (int j = numYpnts - 1; j >= 0; j--)
    {
        for (int i = 0; i < numXpnts; i++)
        {
            printf("%.2f ", textureData[4 * (j * numXpnts + i)]);
        }
        printf("\n");
    }
    printf("\n");
}