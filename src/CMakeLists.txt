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
    ProjectSourcePointsToFacet.cu
    ProjectFromFacetsToFieldPoints.cu
    ProjectFromFacetsToFacets.cu
    FacetCuda.cu
    CudaUtils.cu  
)

set(CPP_SRC
    Model.cpp
    ModelCuda.cpp
    ModelGl.cpp
    python_interface.cpp
    Facet.cpp
    FacetCuda.cpp
    FacetGl.cpp
    Object.cpp
    ObjectCuda.cpp
    ObjectGl.cpp
    Globals.cpp
    OptiX.cpp
)

# Apply -optix-ir only to specific files
# set_source_files_properties(Collision.cu PROPERTIES
#    COMPILE_OPTIONS "-optix-ir"
#)

# Add the shared library target
add_library(${TARGET_NAME} SHARED ${CU_SRC} ${CPP_SRC} ${OPX_SRC})

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

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Set compiler options for the shared library
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:>
    $<$<COMPILE_LANGUAGE:CXX>:>
)

# Enable CUDA separable compilation for the shared library
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

include_directories(/home/apcave/alex-portfolio/include)
link_directories(/home/apcave/alex-portfolio/lib)
#target_link_libraries(${TARGET_NAME} PRIVATE util_7_sdk)
# Link against the OptiX library
# link_directories(/home/apcave/alex-portfolio/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64/lib64)
# target_link_libraries(${TARGET_NAME} PRIVATE optix)


# add_custom_command(
#     OUTPUT ${CMAKE_BINARY_DIR}/Collision.optixir
#     COMMAND ${CMAKE_CUDA_COMPILER}
#     ARGS -I/home/apcave/alex-portfolio/include -optix-ir ${CMAKE_CURRENT_SOURCE_DIR}/Collision.cu -o ${CMAKE_BINARY_DIR}/Collision.optixir
#     DEPENDS Collision.cu
#     COMMENT "Compiling Collision.cu to OptiX IR"
# )

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/OptiX.ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
    ARGS -I/home/apcave/alex-portfolio/include -ptx ${CMAKE_CURRENT_SOURCE_DIR}/OptiX.cu -o ${CMAKE_BINARY_DIR}/OptiX.ptx
    DEPENDS OptiX.cu
    COMMENT "Compiling Collision.cu to OptiX IR"
)

# Add a custom target for CollisionIR
add_custom_target(OptiX_render DEPENDS ${CMAKE_BINARY_DIR}/OptiX.ptx)


# Ensure CollisionIR is built with cuda_project
add_dependencies(${TARGET_NAME} OptiX_render)