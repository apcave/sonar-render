cmake_minimum_required(VERSION 3.18)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Set the CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES native) # Example for sm_75 (Turing GPUs)
project(sonar_render LANGUAGES CXX CUDA)

# Set the output library name
set(TARGET_NAME cuda_project)

# Add source files for the shared library
set(CU_SRC
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
target_include_directories(${TARGET_NAME} PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/OptiX/include)

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
include_directories(${CMAKE_SOURCE_DIR}/OptiX/include)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/OptiX.ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
    ARGS -I${CMAKE_SOURCE_DIR}/OptiX/include -ptx ${CMAKE_CURRENT_SOURCE_DIR}/OptiX.cu -o ${CMAKE_BINARY_DIR}/OptiX.ptx
    DEPENDS OptiX.cu
    COMMENT "Compiling OptiX.cu to OptiX IR"
)

# Add a custom target for CollisionIR
add_custom_target(OptiX_render DEPENDS ${CMAKE_BINARY_DIR}/OptiX.ptx)


# Ensure CollisionIR is built with cuda_project
add_dependencies(${TARGET_NAME} OptiX_render)