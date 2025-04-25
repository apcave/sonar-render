
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Set paths to CUDA and OptiX
set(CMAKE_CUDA_ARCHITECTURES native)
project(OptixCollisionTest LANGUAGES C CXX CUDA)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/home/apcave/alex-portfolio/include)

# Source files
set(SOURCES main.cpp)
set(PTX_SOURCES device_programs.cu)

# Compile the PTX file
set(PTX_OUTPUT ${CMAKE_BINARY_DIR}/device_programs.ptx)
add_custom_target(
    device_programs_ptx
    COMMAND ${CMAKE_CUDA_COMPILER} -I/home/apcave/alex-portfolio/include -ptx ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu -o ${PTX_OUTPUT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMENT "Compiling PTX file: device_programs.cu"
)

# Add the executable
add_executable(main ${SOURCES})

# Make the executable depend on the PTX target
add_dependencies(main device_programs_ptx)

# Link libraries
target_link_libraries(main PRIVATE ${CUDA_LIBRARIES} dl)

