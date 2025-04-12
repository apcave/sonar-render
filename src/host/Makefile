CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXXFLAGS := -O2 -g -Iinclude
PYTHON := -shared --compiler-options '-fPIC'
SRC := python_interface.cpp Facet.cpp
OBJ := $(SRC:.cu=.o)
TARGET := cuda_project.so

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(PYTHON) -o $@ $^

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean