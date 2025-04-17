cmake_minimum_required(VERSION 3.18) # Minimum version for good CUDA support

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Set the CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES native) # Example for sm_75 (Turing GPUs)
project(sonar_render LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)


# Set the output library name
set(TARGET_NAME cuda_project)

# Add source files
set(CU_SRC
    CudaUtils.cu
    CudaModelTes.cu
    ProjectSourcePointsToFacet.cu
)

set(CPP_SRC
    python_interface.cpp
    Facet.cpp
)

# Add the shared library target
add_library(${TARGET_NAME} SHARED ${CU_SRC} ${CPP_SRC})

# Include directories
target_include_directories(${TARGET_NAME} PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set compiler options
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)

# Enable CUDA separable compilation
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)