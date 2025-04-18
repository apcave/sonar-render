cmake_minimum_required(VERSION 3.18) # Minimum version for good CUDA support

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Set the CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES native) # Example for sm_75 (Turing GPUs)
project(sonar_render LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

# Set the output library name
set(TARGET_NAME cuda_project)

# Add source files for the shared library
set(CU_SRC
    CudaUtils.cu
    CudaModelTes.cu
    ProjectSourcePointsToFacet.cu
    ProjectFromFacetsToFieldPoints.cu
    ProjectFromFacetsToFacets.cu
)

set(CPP_SRC
    ModelTes.cpp
    python_interface.cpp
    Facet.cpp
    OpenGL_TES.cpp
)

# Add the shared library target
add_library(${TARGET_NAME} SHARED ${CU_SRC} ${CPP_SRC})

# Include directories
target_include_directories(${TARGET_NAME} PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link OpenGL, GLEW, and GLFW libraries for the shared library
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    GLU
    cuda
    cudart
)

# Set compiler options for the shared library
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)

# Enable CUDA separable compilation for the shared library
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
