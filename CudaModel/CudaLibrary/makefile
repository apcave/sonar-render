# Makefile for CUDA Project

CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXXFLAGS := -O2 -g -Iinclude
SRC := CudaFunctions.cu ProjectFacetToPoint.cu
OBJ := $(SRC:.cu=.o)
TARGET := cuda_project

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) -o $@ $^

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean