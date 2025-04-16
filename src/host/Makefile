export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXXFLAGS := -O2 -g -Iinclude --compiler-options '-fPIC' -rdc=true
PYTHON := -g -shared --compiler-options '-fPIC'

# Separate CUDA and C++ source files
CU_SRC := CudaUtils.cu CudaModelTes.cu ProjectSourcePointsToFacet.cu
CPP_SRC := python_interface.cpp Facet.cpp

# Generate object file lists
CU_OBJ := $(CU_SRC:.cu=.o)
CPP_OBJ := $(CPP_SRC:.cpp=.o)

# Combine all object files
OBJ := $(CU_OBJ) $(CPP_OBJ)

TARGET := cuda_project.so

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(PYTHON) -o $@ $^

# Rule for CUDA source files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Rule for C++ source files
%.o: %.cpp
	g++ -O2 -g -Iinclude -I$(CUDA_PATH)/include -fPIC -c $< -o $@

clean:
	rm -f $(CU_OBJ) $(CPP_OBJ) $(TARGET)

.PHONY: all clean