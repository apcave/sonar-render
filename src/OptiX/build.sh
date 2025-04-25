#!/bin/bash

/usr/local/cuda/bin/nvcc -ptx device_programs.cu -o device_programs.ptx
g++ main.c -o main -I/path/to/optix/include -L/path/to/optix/lib64 -loptix -lcudart -ldl